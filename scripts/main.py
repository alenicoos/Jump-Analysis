from __future__ import annotations

"""Full webcam workflow.

This is the main project command. It runs the full flow: user height, webcam
setup, drop-jump capture, protocol validation, and temporal anomaly detection.
"""

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jump_analysis.feedback import AudioFeedback
from jump_analysis.video import (
    analyze_yolo_pose_frames,
    capture_yolo_pose_frames_with_open_capture,
    find_still_end_frame,
    predict_knee_pitch,
    run_floor_box_setup_with_open_capture,
)
from jump_analysis.features.front_2d_features import AE_FEATURE_DIM, extract_ae_features


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


def main() -> None:
    """CLI entry point for full capture and analysis."""

    parser = argparse.ArgumentParser(
        description="Capture a front-view YOLO drop jump, validate protocol, and run temporal models."
    )
    parser.add_argument("--source", default="0", help="Webcam index or video path.")
    parser.add_argument("--model", default="models/yolo26n-pose.pt", help="YOLO pose model.")
    parser.add_argument("--seconds", type=float, default=8.0, help="Capture duration.")
    parser.add_argument("--prepare-seconds", type=float, default=2.0, help="Seconds to stand still on the box before arming.")
    parser.add_argument("--max-wait-seconds", type=float, default=30.0, help="Maximum time to wait for the drop to start.")
    parser.add_argument("--min-drop-ratio", type=float, default=0.06, help="Drop trigger threshold as body-height fraction.")
    parser.add_argument("--height-cm", type=float, help="User height in centimeters for setup box-height estimation.")
    parser.add_argument("--manual-box-setup", action="store_true", help="Do not auto-detect box entry during setup.")
    parser.add_argument("--audio", action="store_true", help="Enable spoken setup feedback. Kept for compatibility; audio is now enabled by default.")
    parser.add_argument("--no-audio", action="store_true", help="Disable spoken setup feedback.")
    parser.add_argument("--pitch-model",      default="models/pitch_transformer.pt", help="PitchTransformer weights (.pt). Set to '' to skip.")
    parser.add_argument("--autoencoder-model",default="models/jump_autoencoder_lstm.pt",  help="JumpAutoencoder weights (.pt). Set to '' to skip.")
    parser.add_argument("--no-show", action="store_true", help="Do not show capture window.")
    args = parser.parse_args()
    setup_audio = args.audio or not args.no_audio
    feedback = AudioFeedback(enabled=setup_audio)

    # Height is used to convert pixels to meters during setup.
    height_cm = args.height_cm if args.height_cm is not None else ask_height_cm()
    shoulder_width_m = None

    source = int(args.source) if str(args.source).isdigit() else args.source
    print("Starting model and webcam...")

    # Load YOLO once and keep the webcam open for both setup and jump capture.
    model = YOLO(args.model)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    try:
        # During setup we measure body height in pixels on the floor. With the
        # declared real height, we estimate meters/pixel and measured shoulder width.
        calibration = run_floor_box_setup_with_open_capture(
            cap,
            model,
            height_cm=height_cm,
            show=not args.no_show,
            audio=setup_audio,
            feedback=feedback,
            auto_detect_box=not args.manual_box_setup,
        )
        shoulder_width_m = calibration.measured_shoulder_width_m

        # This function waits on the box and starts recording only when it
        # detects the drop-jump descent.
        frames = capture_yolo_pose_frames_with_open_capture(
            cap,
            model,
            args.seconds,
            shoulder_width_m=shoulder_width_m,
            box_height_px=calibration.box_height_px,
            show=not args.no_show,
            prepare_seconds=args.prepare_seconds,
            min_drop_ratio=args.min_drop_ratio,
            max_wait_seconds=args.max_wait_seconds,
            feedback=feedback,
        )
    finally:
        cap.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    # Extract protocol metadata from valid frames.
    metadata = analyze_yolo_pose_frames(frames)
    model_frames = frames

    fps = 30.0
    if len(model_frames) >= 2:
        dur = model_frames[-1].timestamp_s - model_frames[0].timestamp_s
        if dur > 0:
            fps = (len(model_frames) - 1) / dur
    from jump_analysis.video.yolo_video import drop_trigger_index
    trigger_idx = drop_trigger_index(model_frames)
    ae_start = max(0, trigger_idx - 2)
    ic_frame = int(metadata.get("ic_valid_frame", ae_start))
    still_scan_start = max(ae_start, ic_frame + max(1, round(1.0 * fps)))
    still_end = find_still_end_frame(model_frames, still_scan_start, still_seconds=1.5, fps=fps)
    if still_end is not None:
        ae_end = max(ae_start, still_end - max(1, round(1.0 * fps)))
    else:
        ae_end = len(model_frames) - 1
    ae_frames = model_frames[ae_start : ae_end + 1]

    # ── PitchTransformer: frame-by-frame knee pitch ──────────────────────────
    pitch_series = None
    pitch_model  = None
    if args.pitch_model:
        pitch_model_path = Path(args.pitch_model)
        if pitch_model_path.exists():
            try:
                from jump_analysis.models.pitch_transformer import PitchTransformer
                pitch_model  = PitchTransformer.load(str(pitch_model_path))
                pitch_series = predict_knee_pitch(model_frames, pitch_model)   # (T, 2)
            except Exception as exc:
                print(f"[warn] PitchTransformer skipped: {exc}")
        else:
            print(f"[warn] Pitch model not found: {pitch_model_path}. Run train_pitch_transformer.py first.")

    # ── JumpAutoencoder: anomaly detection temporale ──────────────────────────
    ae_score    = None
    ae_is_anomaly = None
    if args.autoencoder_model:
        ae_path = Path(args.autoencoder_model)
        if ae_path.exists():
            try:
                from jump_analysis.models.jump_autoencoder import JumpAutoencoder
                from jump_analysis.video import frames_to_transformer_input

                ae_model = JumpAutoencoder.load(str(ae_path))

                if ae_model.input_dim == AE_FEATURE_DIM:
                    kp_seq = frames_to_transformer_input(ae_frames, input_dim=24)
                    ae_seq = extract_ae_features(kp_seq)
                else:
                    # Legacy compatibility for old keypoint+pitch autoencoders.
                    if pitch_model is None:
                        raise RuntimeError("legacy autoencoder requires PitchTransformer output")
                    ae_keypoint_dim = ae_model.input_dim - 2
                    kp_seq = frames_to_transformer_input(ae_frames, input_dim=ae_keypoint_dim)
                    import numpy as np
                    pit_seq = pitch_series if pitch_series is not None else pitch_model.predict_numpy(kp_seq)
                    ae_seq = np.concatenate([kp_seq, pit_seq], axis=1)

                ae_is_anomaly, ae_score = ae_model.is_anomaly(ae_seq)
                print(f"Autoencoder score: {ae_score:.5f}  "
                      f"(threshold={ae_model.anomaly_threshold:.5f})  "
                      f"→ {'ANOMALY' if ae_is_anomaly else 'NORMAL'}")
            except Exception as exc:
                print(f"[warn] JumpAutoencoder skipped: {exc}")
        else:
            print(f"[warn] Autoencoder model not found: {ae_path}. Run train_jump_autoencoder.py first.")

    print("Protocol check:")
    for key, value in metadata.items():
        if key.endswith("_passed"):
            print(f"  {key}: {'OK' if value else 'FAIL'}")
    if not metadata["protocol_passed"]:
        raise RuntimeError(
            "The recorded movement does not look like a valid LESS drop jump. "
            "Try again: start on the box, drop down, land, and immediately perform the second jump."
        )

    # Print pitch at key biomechanical frames.
    if pitch_series is not None:
        ic_idx = metadata["ic_valid_frame"]
        kf_idx = metadata["kfmax_valid_frame"]
        ic_l, ic_r = pitch_series[ic_idx]
        kf_l, kf_r = pitch_series[kf_idx]
        print()
        print("Knee pitch (PitchTransformer):")
        print(f"  Initial contact  — left: {ic_l:+.1f}°  right: {ic_r:+.1f}°")
        print(f"  Max knee flexion — left: {kf_l:+.1f}°  right: {kf_r:+.1f}°")
        print(f"  Δ pitch IC→KFmax — left: {kf_l - ic_l:+.1f}°  right: {kf_r - ic_r:+.1f}°")

    # Print autoencoder temporal anomaly result.
    if ae_score is not None:
        print()
        print("Temporal anomaly detection (JumpAutoencoder):")
        print(f"  Score:     {ae_score:.5f}")
        print(f"  Threshold: {ae_model.anomaly_threshold:.5f}")
        print(f"  Result:    {'⚠ ANOMALY' if ae_is_anomaly else '✓ NORMAL'}")


if __name__ == "__main__":
    main()
