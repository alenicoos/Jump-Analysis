from __future__ import annotations

"""Validated drop-jump data collection.

This script uses the same setup as the main workflow, but it does not compare
the trial with the mocap dataset and does not run anomaly detection. It is used
to collect clean data: if the jump protocol fails, no file is saved.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jump_analysis.features.front_2d_features import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    distance,
)
from jump_analysis.feedback import AudioFeedback
from jump_analysis.sensors import Bwt901clReader, load_imu_orientation_csv
from jump_analysis.video import (
    analyze_yolo_pose_frames,
    capture_yolo_pose_frames_with_open_capture,
    run_floor_box_setup_with_open_capture,
)
from jump_analysis.video.yolo_video import YoloPoseFrame, estimate_person_height_px


def ask_height_cm() -> float:
    """Ask for height until the user enters a plausible value."""

    while True:
        raw = input("Enter your height in cm: ").strip().replace(",", ".")
        try:
            height_cm = float(raw)
        except ValueError:
            print("Invalid value. Enter a number, for example 183.")
            continue
        if 80.0 <= height_cm <= 250.0:
            return height_cm
        print("Height out of range. Enter height in centimeters, for example 183.")


def frame_rows(
    frames: list[YoloPoseFrame],
    calibration,
    left_imu_csv: str | None,
    right_imu_csv: str | None,
    left_imu_series=None,
    right_imu_series=None,
) -> list[dict[str, float | int]]:
    """Convert the full jump trajectory to frame-by-frame CSV rows."""

    rows: list[dict[str, float | int]] = []
    timestamps = np.array([frame.timestamp_s for frame in frames], dtype=float)
    relative_timestamps = timestamps - timestamps[0] if len(timestamps) else timestamps
    duration = float(max(relative_timestamps[-1], 1e-6)) if len(relative_timestamps) else 1e-6

    left_sensor = left_imu_series or load_imu_orientation_csv(left_imu_csv)
    right_sensor = right_imu_series or load_imu_orientation_csv(right_imu_csv)
    left_sensor_data = _interpolate_sensor(left_sensor, relative_timestamps)
    right_sensor_data = _interpolate_sensor(right_sensor, relative_timestamps)

    for valid_frame_index, frame in enumerate(frames):
        keypoints_px = frame.raw_keypoints_xy if frame.raw_keypoints_xy is not None else np.full_like(frame.keypoints_xy, np.nan)
        body_height_px = estimate_person_height_px(frame.box_xyxy, keypoints_px)
        shoulder_width_px = distance(keypoints_px[LEFT_SHOULDER], keypoints_px[RIGHT_SHOULDER])
        hip_width_px = distance(keypoints_px[LEFT_HIP], keypoints_px[RIGHT_HIP])
        knee_width_px = distance(keypoints_px[LEFT_KNEE], keypoints_px[RIGHT_KNEE])
        ankle_width_px = distance(keypoints_px[LEFT_ANKLE], keypoints_px[RIGHT_ANKLE])

        row: dict[str, float | int] = {
            "valid_frame_index": valid_frame_index,
            "raw_frame_index": frame.frame_index,
            "timestamp_s": frame.timestamp_s,
            "time_from_start_s": float(relative_timestamps[valid_frame_index]),
            "normalized_time": float(relative_timestamps[valid_frame_index] / duration),
            "body_height_px": body_height_px,
            "shoulder_width_px": shoulder_width_px,
            "shoulder_width_m": shoulder_width_px * calibration.meters_per_pixel,
            "hip_width_px": hip_width_px,
            "hip_width_m": hip_width_px * calibration.meters_per_pixel,
            "knee_width_px": knee_width_px,
            "knee_width_m": knee_width_px * calibration.meters_per_pixel,
            "ankle_width_px": ankle_width_px,
            "ankle_width_m": ankle_width_px * calibration.meters_per_pixel,
            # IMU ground truth (sensor physically measures yaw, so it is kept)
            "left_sensor_pitch_deg": float(left_sensor_data.loc[valid_frame_index, "pitch_deg"]),
            "left_sensor_roll_deg": float(left_sensor_data.loc[valid_frame_index, "roll_deg"]),
            "left_sensor_yaw_deg": float(left_sensor_data.loc[valid_frame_index, "yaw_deg"]),
            "right_sensor_pitch_deg": float(right_sensor_data.loc[valid_frame_index, "pitch_deg"]),
            "right_sensor_roll_deg": float(right_sensor_data.loc[valid_frame_index, "roll_deg"]),
            "right_sensor_yaw_deg": float(right_sensor_data.loc[valid_frame_index, "yaw_deg"]),
        }
        for keypoint_index, (xy, confidence) in enumerate(zip(frame.keypoints_xy, frame.keypoints_conf)):
            raw_xy = keypoints_px[keypoint_index]
            row[f"kp_{keypoint_index:02d}_x_px"] = float(raw_xy[0])
            row[f"kp_{keypoint_index:02d}_y_px"] = float(raw_xy[1])
            row[f"kp_{keypoint_index:02d}_x_m"] = float(xy[0])
            row[f"kp_{keypoint_index:02d}_y_m"] = float(xy[1])
            row[f"kp_{keypoint_index:02d}_conf"] = float(confidence)
        if frame.box_xyxy is not None:
            row["box_x1"] = float(frame.box_xyxy[0])
            row["box_y1"] = float(frame.box_xyxy[1])
            row["box_x2"] = float(frame.box_xyxy[2])
            row["box_y2"] = float(frame.box_xyxy[3])
        rows.append(row)
    return rows


def _interpolate_sensor(sensor, relative_timestamps: np.ndarray) -> pd.DataFrame:
    """Align an IMU series to video timestamps, or create empty columns."""

    if sensor is None:
        return pd.DataFrame(
            {
                "pitch_deg": np.full(len(relative_timestamps), np.nan),
                "roll_deg": np.full(len(relative_timestamps), np.nan),
                "yaw_deg": np.full(len(relative_timestamps), np.nan),
            }
        )
    return sensor.interpolate(relative_timestamps)


def calibration_metadata(calibration) -> dict[str, Any]:
    """Extract JSON-serializable setup fields."""

    return {
        "floor_body_height_px": calibration.floor_body_height_px,
        "meters_per_pixel": calibration.meters_per_pixel,
        "measured_shoulder_width_m": calibration.measured_shoulder_width_m,
        "box_height_px": calibration.box_height_px,
        "scale_change_ratio": calibration.scale_change_ratio,
        "camera_roll_degrees": calibration.camera_roll_degrees,
        "pitch_proxy_ratio": calibration.pitch_proxy_ratio,
        "estimated_box_height_cm": calibration.estimated_box_height_cm,
        "floor_pose_keypoints_xy_px": calibration.floor_pose.keypoints_xy.tolist(),
        "box_pose_keypoints_xy_px": calibration.box_pose.keypoints_xy.tolist(),
    }


def write_valid_trial(
    output_root: Path,
    trial_id: str,
    frames: list[YoloPoseFrame],
    metadata: dict[str, float | int],
    calibration,
    args: argparse.Namespace,
    height_cm: float,
    left_imu_series=None,
    right_imu_series=None,
    left_reader: Bwt901clReader | None = None,
    right_reader: Bwt901clReader | None = None,
) -> Path:
    """Save a valid trial in a dedicated folder."""

    trial_dir = output_root / trial_id
    trial_dir.mkdir(parents=True, exist_ok=False)

    timeseries_path = trial_dir / "movement_timeseries.csv"
    metadata_path = trial_dir / "trial_metadata.json"

    first_video_timestamp = frames[0].timestamp_s if frames else None
    pd.DataFrame(
        frame_rows(
            frames,
            calibration,
            args.left_imu_csv,
            args.right_imu_csv,
            left_imu_series=left_imu_series,
            right_imu_series=right_imu_series,
        )
    ).to_csv(timeseries_path, index=False)
    if left_reader is not None:
        left_reader.save_csv(trial_dir / "left_bwt901cl_raw.csv", start_timestamp_s=first_video_timestamp)
    if right_reader is not None:
        right_reader.save_csv(trial_dir / "right_bwt901cl_raw.csv", start_timestamp_s=first_video_timestamp)

    left_sample_count = _jump_sample_count(left_reader, first_video_timestamp)
    right_sample_count = _jump_sample_count(right_reader, first_video_timestamp)
    if args.left_imu_port and args.right_imu_port:
        sensor_status = "live_serial_loaded"
    elif args.left_imu_csv and args.right_imu_csv:
        sensor_status = "csv_loaded"
    else:
        sensor_status = "missing_or_partial"

    trial_metadata = {
        "trial_id": trial_id,
        "participant_id": args.participant_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "height_cm": height_cm,
        "source": args.source,
        "model": args.model,
        "seconds": args.seconds,
        "prepare_seconds": args.prepare_seconds,
        "min_drop_ratio": args.min_drop_ratio,
        "max_wait_seconds": args.max_wait_seconds,
        "protocol": metadata,
        "setup_calibration": calibration_metadata(calibration),
        "files": {
            "movement_timeseries": timeseries_path.name,
            "trial_metadata": metadata_path.name,
        },
        "sensor_ground_truth": {
            "status": sensor_status,
            "planned_sensors": ["left_knee_bwt901cl", "right_knee_bwt901cl"],
            "left_imu_csv": args.left_imu_csv,
            "right_imu_csv": args.right_imu_csv,
            "left_imu_port": args.left_imu_port,
            "right_imu_port": args.right_imu_port,
            "live_reader_used": bool(args.left_imu_port and args.right_imu_port),
            "left_live_sample_count": left_sample_count,
            "right_live_sample_count": right_sample_count,
        },
    }
    metadata_path.write_text(json.dumps(trial_metadata, indent=2), encoding="utf-8")
    return trial_dir


def _jump_sample_count(reader: Bwt901clReader | None, start_timestamp_s: float | None) -> int | None:
    """Count live sensor samples saved from jump start onward."""

    if reader is None:
        return None
    frame = reader.to_frame(start_timestamp_s=start_timestamp_s)
    if start_timestamp_s is not None and "timestamp_s" in frame.columns:
        frame = frame[frame["timestamp_s"] >= 0.0]
    return len(frame)


def main() -> None:
    """Acquire valid trials and save pose/sensor time series without running a model."""

    parser = argparse.ArgumentParser(
        description="Collect a validated drop-jump trial without running reference comparison or anomaly detection."
    )
    parser.add_argument("--source", default="0", help="Webcam index or video path.")
    parser.add_argument("--model", default="models/yolo26n-pose.pt", help="YOLO pose model.")
    parser.add_argument("--seconds", type=float, default=8.0, help="Capture duration after drop detection.")
    parser.add_argument("--prepare-seconds", type=float, default=2.0, help="Seconds to stand still on the box before arming.")
    parser.add_argument("--max-wait-seconds", type=float, default=30.0, help="Maximum time to wait for the drop to start.")
    parser.add_argument("--min-drop-ratio", type=float, default=0.06, help="Drop trigger threshold as body-height fraction.")
    parser.add_argument("--height-cm", type=float, help="User height in centimeters for setup.")
    parser.add_argument("--participant-id", default="participant", help="Participant identifier used in the trial folder name.")
    parser.add_argument("--output-dir", default="collected_trials", help="Folder where valid trials are saved.")
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of jump trials to collect in one sensor session. Use 0 for interactive unlimited mode.",
    )
    parser.add_argument("--left-imu-csv", help="Optional left knee BWT901CL orientation CSV.")
    parser.add_argument("--right-imu-csv", help="Optional right knee BWT901CL orientation CSV.")
    parser.add_argument("--left-imu-port", help="Left knee BWT901CL serial/Bluetooth port, e.g. /dev/cu.xxx.")
    parser.add_argument("--right-imu-port", help="Right knee BWT901CL serial/Bluetooth port, e.g. /dev/cu.xxx.")
    parser.add_argument("--imu-baud", type=int, default=115200, help="BWT901CL baud rate. Bluetooth default is 115200.")
    parser.add_argument("--configure-imu-angle-output", action="store_true", help="Send WIT commands to enable BWT901CL angle packets.")
    parser.add_argument(
        "--require-imu",
        action="store_true",
        help="Reject the trial if live IMU ports are configured but no IMU samples are captured.",
    )
    parser.add_argument(
        "--imu-stagger-start-seconds",
        type=float,
        default=1.0,
        help="Pause between opening left and right Bluetooth IMU ports.",
    )
    parser.add_argument("--manual-box-setup", action="store_true", help="Do not auto-detect box entry during setup.")
    parser.add_argument("--audio", action="store_true", help="Enable spoken setup feedback. Kept for compatibility; audio is now enabled by default.")
    parser.add_argument("--no-audio", action="store_true", help="Disable spoken setup feedback.")
    parser.add_argument("--no-show", action="store_true", help="Do not show capture window.")
    args = parser.parse_args()
    setup_audio = args.audio or not args.no_audio
    feedback = AudioFeedback(enabled=setup_audio)

    height_cm = args.height_cm if args.height_cm is not None else ask_height_cm()
    source = int(args.source) if str(args.source).isdigit() else args.source

    print("Starting valid trial collection.")
    print("A trial is saved only if setup and drop-jump protocol validation pass.")
    left_reader = None
    right_reader = None
    left_live_series = None
    right_live_series = None

    if args.left_imu_port and args.right_imu_port:
        print("Live BWT901CL sensors: reading from serial/Bluetooth ports.")
        left_reader = Bwt901clReader(
            args.left_imu_port,
            baud_rate=args.imu_baud,
            name="left-knee",
            configure_angle_output=args.configure_imu_angle_output,
            backend="witmotion",
        )
        right_reader = Bwt901clReader(
            args.right_imu_port,
            baud_rate=args.imu_baud,
            name="right-knee",
            configure_angle_output=args.configure_imu_angle_output,
            backend="witmotion",
        )
    elif args.left_imu_csv and args.right_imu_csv:
        print("IMU CSV files provided: they will be interpolated onto video timestamps.")
    else:
        print("Incomplete IMU inputs: ground-truth columns will be saved empty.")

    model = YOLO(args.model)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    try:
        try:
            if left_reader is not None and right_reader is not None:
                if args.configure_imu_angle_output:
                    print("Warning: configure sensors one at a time if macOS loses the Bluetooth stream.")
                left_reader.start()
                if args.imu_stagger_start_seconds > 0:
                    time.sleep(args.imu_stagger_start_seconds)
                right_reader.start()

            trial_number = 0
            while args.trials == 0 or trial_number < args.trials:
                raw = input("Press ENTER to start setup for the next jump, or Q to quit: ").strip().lower()
                if raw == "q":
                    break

                try:
                    calibration = run_floor_box_setup_with_open_capture(
                        cap,
                        model,
                        height_cm=height_cm,
                        show=not args.no_show,
                        audio=setup_audio,
                        feedback=feedback,
                        auto_detect_box=not args.manual_box_setup,
                    )
                except RuntimeError as exc:
                    print(f"Setup failed: {exc}")
                    print("Trial NOT saved. Press ENTER to try setup again.")
                    continue

                if left_reader is not None:
                    left_reader.clear()
                if right_reader is not None:
                    right_reader.clear()

                left_live_series = None
                right_live_series = None
                try:
                    frames = capture_yolo_pose_frames_with_open_capture(
                        cap,
                        model,
                        args.seconds,
                        shoulder_width_m=calibration.measured_shoulder_width_m,
                        box_height_px=calibration.box_height_px,
                        show=not args.no_show,
                        prepare_seconds=args.prepare_seconds,
                        min_drop_ratio=args.min_drop_ratio,
                        max_wait_seconds=args.max_wait_seconds,
                        feedback=feedback,
                    )
                except RuntimeError as exc:
                    print(f"Capture failed: {exc}")
                    print("Trial NOT saved. Press ENTER to try again.")
                    continue

                metadata = analyze_yolo_pose_frames(frames)
                first_video_timestamp = frames[0].timestamp_s if frames else None
                if first_video_timestamp is not None:
                    if left_reader is not None:
                        left_live_series = left_reader.to_series(start_timestamp_s=first_video_timestamp)
                    if right_reader is not None:
                        right_live_series = right_reader.to_series(start_timestamp_s=first_video_timestamp)
                print("Protocol check:")
                for key, value in metadata.items():
                    if key.endswith("_passed"):
                        print(f"  {key}: {'OK' if value else 'FAIL'}")

                trial_number += 1
                if not metadata["protocol_passed"]:
                    print("Invalid trial: data NOT saved.")
                    print("Try again: start on the box, land with both feet, and immediately perform the second jump.")
                    continue

                if left_reader is not None and right_reader is not None:
                    left_samples = _jump_sample_count(left_reader, first_video_timestamp)
                    right_samples = _jump_sample_count(right_reader, first_video_timestamp)
                    if not left_samples or not right_samples:
                        print("Warning: live IMU samples are missing for this jump.")
                        print(f"Left IMU samples after jump start: {left_samples or 0}")
                        print(f"Right IMU samples after jump start: {right_samples or 0}")
                        if args.require_imu:
                            print("Invalid sensor capture: data NOT saved because --require-imu is enabled.")
                            print("Check that both sensors are streaming before collecting the next jump.")
                            continue

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_participant = "".join(char if char.isalnum() or char in "-_" else "_" for char in args.participant_id)
                trial_id = f"{safe_participant}_{timestamp}"
                trial_dir = write_valid_trial(
                    Path(args.output_dir),
                    trial_id,
                    frames,
                    metadata,
                    calibration,
                    args,
                    height_cm,
                    left_imu_series=left_live_series,
                    right_imu_series=right_live_series,
                    left_reader=left_reader,
                    right_reader=right_reader,
                )
                print(f"Valid trial saved in: {trial_dir}")
        finally:
            if left_reader is not None:
                left_reader.stop(close_serial=False)
            if right_reader is not None:
                right_reader.stop(close_serial=False)
    finally:
        cap.release()
        if not args.no_show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
