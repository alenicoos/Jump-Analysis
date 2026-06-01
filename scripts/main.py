from __future__ import annotations

"""Full webcam workflow.

This is the main project command. It runs the full flow: user height, webcam
setup, drop-jump capture, feature extraction, dataset comparison, and anomaly
detection.
"""

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jump_analysis import RobustAnomalyModel
from jump_analysis.feedback import AudioFeedback
from jump_analysis.video import (
    capture_yolo_pose_frames_with_open_capture,
    compare_to_reference,
    extract_front_features_from_yolo_frames,
    run_floor_box_setup_with_open_capture,
)


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
        description="Capture a front-view YOLO drop jump and compare its 37 features to the mocap dataset."
    )
    parser.add_argument("--source", default="0", help="Webcam index or video path.")
    parser.add_argument("--model", default="yolo26n-pose.pt", help="YOLO pose model.")
    parser.add_argument("--seconds", type=float, default=8.0, help="Capture duration.")
    parser.add_argument("--prepare-seconds", type=float, default=2.0, help="Seconds to stand still on the box before arming.")
    parser.add_argument("--max-wait-seconds", type=float, default=30.0, help="Maximum time to wait for the drop to start.")
    parser.add_argument("--min-drop-ratio", type=float, default=0.06, help="Drop trigger threshold as body-height fraction.")
    parser.add_argument("--height-cm", type=float, help="User height in centimeters for setup box-height estimation.")
    parser.add_argument("--manual-box-setup", action="store_true", help="Do not auto-detect box entry during setup.")
    parser.add_argument("--audio", action="store_true", help="Enable spoken setup feedback. Kept for compatibility; audio is now enabled by default.")
    parser.add_argument("--no-audio", action="store_true", help="Disable spoken setup feedback.")
    parser.add_argument("--reference", default="mocap_front_37_features.csv", help="Mocap feature CSV.")
    parser.add_argument("--features-output", default="yolo_front_features.csv", help="Output YOLO feature CSV.")
    parser.add_argument("--comparison-output", default="yolo_vs_mocap_comparison.csv", help="Output comparison CSV.")
    parser.add_argument("--analysis-output", default="analysis_result.csv", help="Output anomaly analysis CSV.")
    parser.add_argument("--z-threshold", type=float, default=4.0, help="Robust anomaly score threshold.")
    parser.add_argument("--max-outlier-features", type=int, default=8, help="Maximum abnormal features before anomaly.")
    parser.add_argument("--include-crop-length", action="store_true", help="Include crop_length_frames in anomaly detection.")
    parser.add_argument("--no-show", action="store_true", help="Do not show capture window.")
    parser.add_argument("--allow-invalid-protocol", action="store_true", help="Continue even if the captured drop jump protocol fails.")
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

    # Extract features and protocol metadata from valid frames.
    features, metadata = extract_front_features_from_yolo_frames(frames)
    print("Protocol check:")
    for key, value in metadata.items():
        if key.endswith("_passed"):
            print(f"  {key}: {'OK' if value else 'FAIL'}")
    if not metadata["protocol_passed"] and not args.allow_invalid_protocol:
        raise RuntimeError(
            "The recorded movement does not look like a valid LESS drop jump. "
            "Try again: start on the box, drop down, land, and immediately perform the second jump. "
            "Use --allow-invalid-protocol only for debugging."
        )

    # Raw trial CSV: metadata + 37 features.
    pd.DataFrame([{**metadata, **features}]).to_csv(args.features_output, index=False)

    # Feature-by-feature descriptive comparison: classic z-score and percentile.
    comparison = compare_to_reference(features, args.reference)
    comparison.to_csv(args.comparison_output, index=False)

    # Robust model: uses the reference as normality and predicts normal/anomaly.
    reference = pd.read_csv(args.reference)
    anomaly_model = RobustAnomalyModel.fit_reference(
        reference,
        excluded_features=[] if args.include_crop_length else ["crop_length_frames"],
        z_threshold=args.z_threshold,
        max_outlier_features=args.max_outlier_features,
    )
    analysis = anomaly_model.predict(pd.DataFrame([features]))
    analysis.to_csv(args.analysis_output, index=False)

    print(f"YOLO features saved to {args.features_output}")
    print(f"Comparison saved to {args.comparison_output}")
    print(f"Anomaly analysis saved to {args.analysis_output}")
    print()
    print("Anomaly analysis:")
    print(analysis.to_string(index=False))
    print()
    print("Top feature differences by absolute z-score:")
    print(
        comparison.reindex(comparison["z_score"].abs().sort_values(ascending=False).index)
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
