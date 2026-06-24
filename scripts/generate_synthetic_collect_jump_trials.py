from __future__ import annotations

"""Generate synthetic full trial folders that mimic collect_jump_data.py output.

These folders are synthetic approximations of valid drop-jump captures. They
match the saved file structure and column schema of the data collection script,
but they do not come from a real camera/setup session.
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TIMESERIES_NUMERIC_EXCLUDE = {"valid_frame_index", "raw_frame_index"}
FEATURE_INT_COLUMNS = {
    "valid_pose_frames",
    "ic_valid_frame",
    "kfmax_valid_frame",
    "ic_raw_frame",
    "kfmax_raw_frame",
    "protocol_passed",
    "drop_started_from_height_passed",
    "second_jump_passed",
    "stable_after_second_landing_passed",
    "crop_length_frames",
}
PASSTHROUGH_FILES = {
    "movement_timeseries",
    "front_2d_features",
    "trial_metadata",
}


@dataclass
class SeedTrial:
    name: str
    timeseries: pd.DataFrame
    features: pd.DataFrame
    metadata: dict[str, object]
    left_imu: pd.DataFrame
    right_imu: pd.DataFrame


def load_seed_trials() -> list[SeedTrial]:
    names = [
        "participant_20260601_144304",
        "participant_20260601_144424",
    ]
    seeds: list[SeedTrial] = []
    for name in names:
        trial_dir = ROOT / "collected_trials" / name
        seeds.append(
            SeedTrial(
                name=name,
                timeseries=pd.read_csv(trial_dir / "movement_timeseries.csv"),
                features=pd.read_csv(trial_dir / "front_2d_features.csv"),
                metadata=json.loads((trial_dir / "trial_metadata.json").read_text(encoding="utf-8")),
                left_imu=pd.read_csv(trial_dir / "left_bwt901cl_raw.csv"),
                right_imu=pd.read_csv(trial_dir / "right_bwt901cl_raw.csv"),
            )
        )
    return seeds


def smooth_noise(rng: np.random.Generator, size: int, scale: float) -> np.ndarray:
    noise = rng.normal(0.0, scale, size=size)
    kernel = np.array([0.2, 0.6, 0.2], dtype=float)
    return np.convolve(noise, kernel, mode="same")


def resample_frame(frame: pd.DataFrame, timestamps: np.ndarray) -> pd.DataFrame:
    source_t = frame["time_from_start_s"].to_numpy(dtype=float)
    out = pd.DataFrame({"time_from_start_s": timestamps})
    for column in frame.columns:
        if column in {"time_from_start_s", "valid_frame_index", "raw_frame_index", "normalized_time"}:
            continue
        out[column] = np.interp(timestamps, source_t, frame[column].to_numpy(dtype=float))
    return out


def resample_imu(frame: pd.DataFrame, timestamps: np.ndarray) -> pd.DataFrame:
    source_t = frame["timestamp_s"].to_numpy(dtype=float)
    out = pd.DataFrame({"timestamp_s": timestamps})
    for column in frame.columns:
        if column == "timestamp_s":
            continue
        out[column] = np.interp(timestamps, source_t, frame[column].to_numpy(dtype=float))
    return out


def athletic_profile_adjustments(length: int, duration_s: float, rng: np.random.Generator) -> np.ndarray:
    ts = np.linspace(0.0, duration_s, length)
    center = rng.uniform(0.95, 1.45)
    width = rng.uniform(0.24, 0.45)
    rel = (ts - center) / max(width, 1e-6)
    envelope = np.exp(-0.5 * rel * rel)
    return envelope * np.clip(1.0 - ts / max(duration_s, 1e-6), 0.2, 1.0)


def synthesize_timeseries(seed: SeedTrial, rng: np.random.Generator) -> pd.DataFrame:
    source = seed.timeseries.copy()
    new_length = int(rng.integers(max(60, len(source) - 12), min(132, len(source) + 14)))
    duration_s = float(rng.uniform(4.2, 6.0))
    time_from_start = np.linspace(0.0, duration_s, new_length)
    out = resample_frame(source, time_from_start)

    # Athletic build plus strong rebound: slightly larger joint excursions and
    # cleaner frame confidence while preserving the original schema.
    xy_scale = rng.uniform(1.02, 1.08)
    horizontal_shift_px = rng.uniform(-20.0, 20.0)
    envelope = athletic_profile_adjustments(new_length, duration_s, rng)

    for col in out.columns:
        if col in TIMESERIES_NUMERIC_EXCLUDE or col in {"time_from_start_s", "timestamp_s", "normalized_time"}:
            continue
        if col.endswith("_conf"):
            out[col] = np.clip(out[col] + rng.uniform(0.0, 0.01), 0.75, 1.0)
            continue
        if col.endswith("_x_px") or col in {"box_x1", "box_x2"}:
            out[col] = out[col] * xy_scale + horizontal_shift_px + smooth_noise(rng, new_length, 1.4)
            continue
        if col.endswith("_y_px") or col in {"box_y1", "box_y2"}:
            out[col] = out[col] * xy_scale + smooth_noise(rng, new_length, 1.8)
            continue
        if col.endswith("_x_m"):
            out[col] = out[col] * rng.uniform(1.00, 1.05) + smooth_noise(rng, new_length, 0.004)
            continue
        if col.endswith("_y_m"):
            out[col] = out[col] * rng.uniform(1.00, 1.05) + smooth_noise(rng, new_length, 0.004)
            continue
        if col in {"left_video_pitch_deg", "right_video_pitch_deg"}:
            side_boost = rng.uniform(2.0, 5.0)
            sign = 1.0 if "left" in col else -1.0
            out[col] = out[col] * rng.uniform(1.03, 1.14) + sign * envelope * side_boost + smooth_noise(rng, new_length, 0.35)
            continue
        if col in {"left_video_roll_deg", "right_video_roll_deg", "left_video_yaw_deg", "right_video_yaw_deg"}:
            out[col] = out[col] * rng.uniform(0.98, 1.08) + smooth_noise(rng, new_length, 0.30)
            continue
        if col in {"left_sensor_pitch_deg", "right_sensor_pitch_deg"}:
            side_boost = rng.uniform(3.0, 7.0)
            sign = 1.0 if "left" in col else -1.0
            out[col] = out[col] * rng.uniform(1.05, 1.18) + sign * envelope * side_boost + smooth_noise(rng, new_length, 0.4)
            continue
        if col.startswith("left_sensor_") or col.startswith("right_sensor_"):
            out[col] = out[col] * rng.uniform(0.98, 1.08) + smooth_noise(rng, new_length, 0.35)
            continue
        if col.startswith("body_height") or col.endswith("_width_px") or col.endswith("_width_m"):
            out[col] = out[col] * rng.uniform(1.00, 1.04) + smooth_noise(rng, new_length, 0.6)
            continue
        out[col] = out[col] + smooth_noise(rng, new_length, 0.05)

    start_timestamp = float(rng.uniform(15000.0, 28000.0))
    out.insert(0, "valid_frame_index", np.arange(new_length, dtype=int))
    raw_start = int(rng.integers(45, 80))
    raw_step = max(1, int(round(rng.uniform(0.95, 1.1))))
    out.insert(1, "raw_frame_index", raw_start + np.arange(new_length, dtype=int) * raw_step)
    out["timestamp_s"] = start_timestamp + out["time_from_start_s"]
    out["normalized_time"] = out["time_from_start_s"] / max(duration_s, 1e-6)
    return out[source.columns]


def synthesize_feature_row(seed: SeedTrial, ts: pd.DataFrame, rng: np.random.Generator) -> dict[str, float | int]:
    source = seed.features.iloc[0].to_dict()
    row: dict[str, float | int] = {}
    length = len(ts)
    ic_valid = int(rng.integers(2, 6))
    kfmax_valid = int(min(length - 4, ic_valid + rng.integers(3, 8)))
    ic_raw = int(ts.iloc[ic_valid]["raw_frame_index"])
    kfmax_raw = int(ts.iloc[kfmax_valid]["raw_frame_index"])
    for key, value in source.items():
        if key == "valid_pose_frames":
            row[key] = length
        elif key == "ic_valid_frame":
            row[key] = ic_valid
        elif key == "kfmax_valid_frame":
            row[key] = kfmax_valid
        elif key == "ic_raw_frame":
            row[key] = ic_raw
        elif key == "kfmax_raw_frame":
            row[key] = kfmax_raw
        elif key in {"protocol_passed", "drop_started_from_height_passed", "second_jump_passed", "stable_after_second_landing_passed"}:
            row[key] = 1
        elif key == "drop_started_from_height_value":
            base = float(source["drop_started_from_height_threshold"])
            row[key] = base * rng.uniform(1.10, 1.40)
        elif key == "second_jump_value":
            base = float(source["second_jump_threshold"])
            row[key] = base * rng.uniform(3.8, 9.0)
        elif key == "stable_after_second_landing_value":
            base = float(source["stable_after_second_landing_threshold"])
            row[key] = base * rng.uniform(0.70, 0.96)
        elif key == "crop_length_frames":
            row[key] = int(rng.integers(4, 8))
        else:
            perturb = rng.uniform(0.96, 1.04)
            if "degrees" in key:
                row[key] = float(value) + rng.uniform(-2.0, 2.0)
            else:
                row[key] = float(value) * perturb + rng.uniform(-0.01, 0.01)

    for key in FEATURE_INT_COLUMNS:
        row[key] = int(row[key])
    return row


def synthesize_setup_calibration(seed: SeedTrial, rng: np.random.Generator) -> dict[str, object]:
    calibration = json.loads(json.dumps(seed.metadata["setup_calibration"]))
    for key in [
        "floor_body_height_px",
        "meters_per_pixel",
        "measured_shoulder_width_m",
        "box_height_px",
        "scale_change_ratio",
        "camera_roll_degrees",
        "pitch_proxy_ratio",
        "estimated_box_height_cm",
    ]:
        calibration[key] = float(calibration[key]) * rng.uniform(0.97, 1.03)
    for pose_key in ["floor_pose_keypoints_xy_px", "box_pose_keypoints_xy_px"]:
        pose = np.array(calibration[pose_key], dtype=float)
        pose[:, 0] += rng.uniform(-15.0, 15.0)
        pose[:, 1] += rng.uniform(-15.0, 15.0)
        calibration[pose_key] = pose.tolist()
    return calibration


def synthesize_imu(seed_frame: pd.DataFrame, duration_s: float, rng: np.random.Generator, side: str) -> pd.DataFrame:
    sample_count = int(rng.integers(44, 82))
    timestamps = np.linspace(0.0, duration_s, sample_count)
    out = resample_imu(seed_frame, timestamps)
    envelope = athletic_profile_adjustments(sample_count, duration_s, rng)
    sign = 1.0 if side == "left" else -1.0

    for col in ["pitch_deg", "roll_deg", "yaw_deg"]:
        scale = rng.uniform(1.02, 1.16) if col == "pitch_deg" else rng.uniform(0.98, 1.08)
        out[col] = out[col] * scale + sign * envelope * (rng.uniform(3.0, 7.0) if col == "pitch_deg" else rng.uniform(0.8, 2.0))
        out[col] = out[col] + smooth_noise(rng, sample_count, 0.35)

    for col in ["accel_x", "accel_y", "accel_z"]:
        out[col] = out[col] * rng.uniform(1.05, 1.28) + smooth_noise(rng, sample_count, 0.16)
    for col in ["gyro_x", "gyro_y", "gyro_z"]:
        out[col] = out[col] * rng.uniform(1.08, 1.36) + smooth_noise(rng, sample_count, 4.8)
    return out


def synthesize_trial_metadata(
    jump_name: str,
    seed: SeedTrial,
    feature_row: dict[str, float | int],
    setup_calibration: dict[str, object],
    left_imu: pd.DataFrame,
    right_imu: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, object]:
    created_at = datetime(2026, 6, 4, 12, 0, 0) + timedelta(minutes=int(rng.integers(0, 300)))
    protocol = {key: feature_row[key] for key in seed.features.columns[:15]}
    metadata = {
        "trial_id": jump_name,
        "participant_id": "synthetic_athletic_180cm_75kg",
        "created_at": created_at.isoformat(timespec="seconds"),
        "height_cm": 180.0,
        "source": "synthetic_collect_jump_data_recreation",
        "model": "yolo26n-pose.pt",
        "seconds": 8.0,
        "prepare_seconds": 2.0,
        "min_drop_ratio": 0.06,
        "max_wait_seconds": 30.0,
        "protocol": protocol,
        "setup_calibration": setup_calibration,
        "files": {
            "movement_timeseries": "movement_timeseries.csv",
            "front_2d_features": "front_2d_features.csv",
            "trial_metadata": "trial_metadata.json",
        },
        "sensor_ground_truth": {
            "status": "synthetic_generated",
            "planned_sensors": ["left_knee_bwt901cl", "right_knee_bwt901cl"],
            "left_imu_csv": None,
            "right_imu_csv": None,
            "left_imu_port": None,
            "right_imu_port": None,
            "live_reader_used": False,
            "left_live_sample_count": int(len(left_imu)),
            "right_live_sample_count": int(len(right_imu)),
        },
        "synthetic_generation": {
            "based_on_trial": seed.name,
            "subject_profile": {
                "height_cm": 180.0,
                "weight_kg": 75.0,
                "athletic_level": "quite athletic",
                "notes": "near-dunk vertical ability; synthesized for fast rebound drop jumps",
            },
        },
    }
    return metadata


def write_trial(output_root: Path, jump_name: str, ts: pd.DataFrame, features: dict[str, float | int], metadata: dict[str, object], left_imu: pd.DataFrame, right_imu: pd.DataFrame) -> None:
    trial_dir = output_root / jump_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    ts.to_csv(trial_dir / "movement_timeseries.csv", index=False)
    pd.DataFrame([features]).to_csv(trial_dir / "front_2d_features.csv", index=False)
    left_imu.to_csv(trial_dir / "left_bwt901cl_raw.csv", index=False)
    right_imu.to_csv(trial_dir / "right_bwt901cl_raw.csv", index=False)
    (trial_dir / "trial_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_readme(output_root: Path, count: int, seed: int) -> None:
    text = (
        "# Synthetic collect_jump_data-style trials\n\n"
        f"This folder contains {count} synthetic drop-jump trial folders named `jump_1` ... `jump_{count}`.\n\n"
        "Each folder mirrors the saved output structure of `scripts/collect_jump_data.py`:\n"
        "- `movement_timeseries.csv`\n"
        "- `front_2d_features.csv`\n"
        "- `trial_metadata.json`\n"
        "- `left_bwt901cl_raw.csv`\n"
        "- `right_bwt901cl_raw.csv`\n\n"
        "These are synthetic recreations designed to match the script's output structure and a valid drop-jump protocol profile.\n"
        f"Generation seed: {seed}\n"
    )
    (output_root / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic collect_jump_data-style trial folders.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=18075)
    args = parser.parse_args()

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    seeds = load_seed_trials()

    for index in range(1, args.count + 1):
        seed = seeds[int(rng.integers(0, len(seeds)))]
        jump_name = f"jump_{index}"
        ts = synthesize_timeseries(seed, rng)
        feature_row = synthesize_feature_row(seed, ts, rng)
        setup_calibration = synthesize_setup_calibration(seed, rng)
        duration_s = float(ts["time_from_start_s"].iloc[-1])
        left_imu = synthesize_imu(seed.left_imu, duration_s, rng, "left")
        right_imu = synthesize_imu(seed.right_imu, duration_s, rng, "right")
        metadata = synthesize_trial_metadata(jump_name, seed, feature_row, setup_calibration, left_imu, right_imu, rng)
        write_trial(output_root, jump_name, ts, feature_row, metadata, left_imu, right_imu)

    write_readme(output_root, args.count, args.seed)


if __name__ == "__main__":
    main()
