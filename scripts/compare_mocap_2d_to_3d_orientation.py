from __future__ import annotations

"""Compare front-view 2D knee orientation proxies with 3D mocap orientation."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jump_analysis.data.mocap_dataset import KinematicDataConverter
from jump_analysis.data.mocap_knee_orientation import MocapKneeOrientationExporter
from jump_analysis.features.knee_orientation import estimate_knee_orientation_from_pose


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Project mocap drop jumps to 2D, estimate video pitch/roll/yaw proxies, and compare with 3D mocap."
    )
    parser.add_argument("--root", default="/Users/ale/Kinematic_Data", help="Path to Kinematic_Data.")
    parser.add_argument("--output", default="mocap_2d_vs_3d_orientation_comparison.csv", help="Frame-level output CSV.")
    parser.add_argument("--summary-output", default="mocap_2d_vs_3d_orientation_summary.csv", help="Metric summary CSV.")
    parser.add_argument("--max-subjects", type=int, help="Optional limit for quick tests.")
    args = parser.parse_args()

    frame = build_comparison_frame(Path(args.root), max_subjects=args.max_subjects)
    frame.to_csv(args.output, index=False)
    summary = summarize_comparison(frame)
    summary.to_csv(args.summary_output, index=False)

    print(f"Saved {len(frame)} frame rows to {args.output}")
    print(f"Saved summary to {args.summary_output}")
    print()
    print(summary.to_string(index=False))


def build_comparison_frame(root: Path, max_subjects: int | None = None) -> pd.DataFrame:
    """Build a frame-level 2D-vs-3D comparison table."""

    converter = KinematicDataConverter.from_path(root)
    exporter = MocapKneeOrientationExporter.from_path(root)
    rows: list[pd.DataFrame] = []
    errors = []
    subject_dirs = converter.subject_dirs()
    if max_subjects is not None:
        subject_dirs = subject_dirs[:max_subjects]

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        try:
            rows.append(compare_subject(subject_id, converter, exporter))
        except Exception as exc:
            errors.append({"subject_id": subject_id, "error": str(exc)})

    if errors:
        error_path = root / "orientation_2d_3d_comparison_errors.csv"
        pd.DataFrame(errors).to_csv(error_path, index=False)
        print(f"Skipped {len(errors)} subjects. Details saved to {error_path}")
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def compare_subject(
    subject_id: str,
    converter: KinematicDataConverter,
    exporter: MocapKneeOrientationExporter,
) -> pd.DataFrame:
    """Compare one subject's projected 2D proxy against mocap 3D orientation."""

    subject_dir = converter.root / subject_id
    jc = converter._load_mat(subject_dir / f"JC_{subject_id}.mat")
    ground_truth = exporter.export_subject(subject_id)
    task_index = int(ground_truth["task_index"].iloc[0])

    proxy_rows = []
    for frame_index in ground_truth["frame_index"].to_numpy(dtype=int):
        keypoints = converter._front_keypoints(jc, task_index, int(frame_index))
        left_proxy = estimate_knee_orientation_from_pose(keypoints, "left")
        right_proxy = estimate_knee_orientation_from_pose(keypoints, "right")
        proxy_rows.append(
            {
                "left_knee_2d_pitch_deg": left_proxy.pitch_deg,
                "left_knee_2d_roll_deg": left_proxy.roll_deg,
                "left_knee_2d_yaw_deg": left_proxy.yaw_deg,
                "right_knee_2d_pitch_deg": right_proxy.pitch_deg,
                "right_knee_2d_roll_deg": right_proxy.roll_deg,
                "right_knee_2d_yaw_deg": right_proxy.yaw_deg,
            }
        )

    frame = pd.concat([ground_truth.reset_index(drop=True), pd.DataFrame(proxy_rows)], axis=1)
    for side in ("left", "right"):
        for axis in ("pitch", "roll", "yaw"):
            proxy = frame[f"{side}_knee_2d_{axis}_deg"]
            mocap = frame[f"{side}_knee_mocap_{axis}_deg"]
            frame[f"{side}_{axis}_raw_error_deg"] = angle_difference_deg(proxy, mocap)
    return frame


def summarize_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Return raw and bias-aligned errors for every side/axis pair."""

    rows = []
    for side in ("left", "right"):
        for axis in ("pitch", "roll", "yaw"):
            proxy_col = f"{side}_knee_2d_{axis}_deg"
            mocap_col = f"{side}_knee_mocap_{axis}_deg"
            raw_error = angle_difference_deg(frame[proxy_col], frame[mocap_col])
            per_subject_bias = raw_error.groupby(frame["subject_id"]).transform("mean")
            aligned_error = raw_error - per_subject_bias
            rows.append(
                {
                    "side": side,
                    "axis": axis,
                    "raw_mae_deg": float(np.nanmean(np.abs(raw_error))),
                    "raw_rmse_deg": float(np.sqrt(np.nanmean(raw_error**2))),
                    "bias_aligned_mae_deg": float(np.nanmean(np.abs(aligned_error))),
                    "bias_aligned_rmse_deg": float(np.sqrt(np.nanmean(aligned_error**2))),
                    "correlation": safe_corr(frame[proxy_col], frame[mocap_col]),
                    "proxy_mean_deg": float(np.nanmean(frame[proxy_col])),
                    "mocap_mean_deg": float(np.nanmean(frame[mocap_col])),
                }
            )
    return pd.DataFrame(rows)


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    """Correlation that returns NaN for constant/empty arrays."""

    valid = np.isfinite(a.to_numpy(dtype=float)) & np.isfinite(b.to_numpy(dtype=float))
    if valid.sum() < 2:
        return float("nan")
    aa = a.to_numpy(dtype=float)[valid]
    bb = b.to_numpy(dtype=float)[valid]
    if np.nanstd(aa) < 1e-9 or np.nanstd(bb) < 1e-9:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def angle_difference_deg(a: pd.Series, b: pd.Series) -> pd.Series:
    """Smallest signed difference between two angle series."""

    return ((a - b + 180.0) % 360.0) - 180.0


if __name__ == "__main__":
    main()
