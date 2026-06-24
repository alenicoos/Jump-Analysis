from __future__ import annotations

"""Generate synthetic IMU-only drop-jump recordings.

The output mirrors the BWT901CL raw CSV format already used by the project:
each jump folder contains a left and right CSV with timestamp/orientation plus
accel/gyro columns. A metadata file documents that the recordings are
synthetic and tuned for an athletic subject profile.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class AthleteProfile:
    height_cm: float
    weight_kg: float
    athletic_level: str
    notes: str


def _load_seed_pairs() -> list[tuple[pd.DataFrame, pd.DataFrame, str]]:
    trial_names = [
        "participant_20260601_144304",
        "participant_20260601_144424",
    ]
    seeds: list[tuple[pd.DataFrame, pd.DataFrame, str]] = []
    for trial_name in trial_names:
        trial_dir = ROOT / "collected_trials" / trial_name
        left = pd.read_csv(trial_dir / "left_bwt901cl_raw.csv")
        right = pd.read_csv(trial_dir / "right_bwt901cl_raw.csv")
        seeds.append((left, right, trial_name))
    return seeds


def _resample_seed(frame: pd.DataFrame, timestamps: np.ndarray) -> pd.DataFrame:
    source_t = frame["timestamp_s"].to_numpy(dtype=float)
    out = pd.DataFrame({"timestamp_s": timestamps})
    for column in frame.columns:
        if column == "timestamp_s":
            continue
        out[column] = np.interp(timestamps, source_t, frame[column].to_numpy(dtype=float))
    return out


def _smooth_noise(rng: np.random.Generator, size: int, scale: float) -> np.ndarray:
    noise = rng.normal(0.0, scale, size=size)
    kernel = np.array([0.2, 0.6, 0.2], dtype=float)
    return np.convolve(noise, kernel, mode="same")


def _athletic_envelope(timestamps: np.ndarray, duration_s: float, center_s: float, width_s: float) -> np.ndarray:
    relative = (timestamps - center_s) / max(width_s, 1e-6)
    return np.exp(-0.5 * relative * relative) * np.clip(1.0 - timestamps / max(duration_s, 1e-6), 0.15, 1.0)


def _augment_side(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    *,
    duration_s: float,
    pitch_gain: float,
    roll_gain: float,
    yaw_gain: float,
    side_sign: float,
) -> pd.DataFrame:
    out = frame.copy()
    ts = out["timestamp_s"].to_numpy(dtype=float)

    center_s = rng.uniform(0.95, 1.45)
    width_s = rng.uniform(0.28, 0.50)
    envelope = _athletic_envelope(ts, duration_s, center_s, width_s)

    pitch = out["pitch_deg"].to_numpy(dtype=float)
    baseline_pitch = float(np.median(pitch[: max(3, len(pitch) // 8)]))
    pitch_delta = pitch - baseline_pitch
    pitch_delta = pitch_delta * pitch_gain
    pitch_delta += side_sign * envelope * rng.uniform(6.0, 11.0)
    pitch_delta += _smooth_noise(rng, len(out), rng.uniform(0.18, 0.55))
    out["pitch_deg"] = baseline_pitch + pitch_delta

    roll = out["roll_deg"].to_numpy(dtype=float)
    baseline_roll = float(np.median(roll[: max(3, len(roll) // 8)]))
    out["roll_deg"] = (
        baseline_roll
        + (roll - baseline_roll) * roll_gain
        + side_sign * envelope * rng.uniform(1.5, 4.0)
        + _smooth_noise(rng, len(out), rng.uniform(0.12, 0.40))
    )

    yaw = out["yaw_deg"].to_numpy(dtype=float)
    baseline_yaw = float(np.median(yaw[: max(3, len(yaw) // 8)]))
    out["yaw_deg"] = (
        baseline_yaw
        + (yaw - baseline_yaw) * yaw_gain
        + side_sign * envelope * rng.uniform(0.8, 3.0)
        + _smooth_noise(rng, len(out), rng.uniform(0.10, 0.35))
    )

    accel_scale = rng.uniform(1.05, 1.30)
    gyro_scale = rng.uniform(1.10, 1.38)
    for column in ["accel_x", "accel_y", "accel_z"]:
        base = out[column].to_numpy(dtype=float)
        out[column] = base * accel_scale + envelope * rng.uniform(-0.8, 0.8) + _smooth_noise(rng, len(out), 0.12)
    for column in ["gyro_x", "gyro_y", "gyro_z"]:
        base = out[column].to_numpy(dtype=float)
        out[column] = base * gyro_scale + envelope * rng.uniform(-35.0, 35.0) + _smooth_noise(rng, len(out), 4.0)

    return out


def _generate_jump_pair(
    seeds: list[tuple[pd.DataFrame, pd.DataFrame, str]],
    rng: np.random.Generator,
    athlete: AthleteProfile,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    seed_left, seed_right, seed_name = seeds[int(rng.integers(0, len(seeds)))]

    duration_s = float(rng.uniform(3.9, 5.4))
    sample_count = int(rng.integers(42, 58))
    timestamps = np.linspace(0.0, duration_s, sample_count)

    left = _resample_seed(seed_left, timestamps)
    right = _resample_seed(seed_right, timestamps)

    # Athletic subject: stronger but still plausible rebound, with mild asymmetry.
    rebound_gain = rng.uniform(1.12, 1.34)
    asymmetry = rng.uniform(-0.08, 0.08)
    left = _augment_side(
        left,
        rng,
        duration_s=duration_s,
        pitch_gain=rebound_gain * (1.0 + asymmetry),
        roll_gain=rng.uniform(0.96, 1.10),
        yaw_gain=rng.uniform(0.94, 1.08),
        side_sign=1.0,
    )
    right = _augment_side(
        right,
        rng,
        duration_s=duration_s,
        pitch_gain=rebound_gain * (1.0 - asymmetry),
        roll_gain=rng.uniform(0.96, 1.10),
        yaw_gain=rng.uniform(0.94, 1.08),
        side_sign=-1.0,
    )

    metadata = {
        "seed_trial": seed_name,
        "duration_s": duration_s,
        "sample_count": sample_count,
        "athlete_profile": asdict(athlete),
        "synthetic_profile": {
            "rebound_gain": rebound_gain,
            "left_right_asymmetry_factor": asymmetry,
            "contact_style": "athletic_drop_jump_fast_rebound",
        },
    }
    return left, right, metadata


def _write_readme(output_root: Path, athlete: AthleteProfile, count: int, seed: int) -> None:
    text = (
        "# Synthetic Drop Jump IMU Recordings\n\n"
        f"This repository contains {count} synthetic IMU-only drop jump recordings.\n\n"
        "Format:\n"
        "- `jump_1` ... `jump_20`\n"
        "- each folder contains `left_bwt901cl_raw.csv`, `right_bwt901cl_raw.csv`, and `recording_metadata.json`\n\n"
        "Subject profile used for synthesis:\n"
        f"- height: {athlete.height_cm:.0f} cm\n"
        f"- weight: {athlete.weight_kg:.0f} kg\n"
        f"- athletic level: {athlete.athletic_level}\n"
        f"- notes: {athlete.notes}\n\n"
        f"Generation seed: {seed}\n"
    )
    (output_root / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic IMU-only drop jump recordings.")
    parser.add_argument("--output-dir", required=True, help="Directory where jump_1 ... jump_N will be created.")
    parser.add_argument("--count", type=int, default=20, help="Number of synthetic jump recordings to create.")
    parser.add_argument("--seed", type=int, default=18075, help="Random seed for reproducible generation.")
    parser.add_argument("--height-cm", type=float, default=180.0)
    parser.add_argument("--weight-kg", type=float, default=75.0)
    args = parser.parse_args()

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    athlete = AthleteProfile(
        height_cm=args.height_cm,
        weight_kg=args.weight_kg,
        athletic_level="quite athletic",
        notes="near-dunk jump ability; synthetic drop jumps emphasize fast, strong rebound",
    )
    rng = np.random.default_rng(args.seed)
    seeds = _load_seed_pairs()

    for index in range(1, args.count + 1):
        jump_dir = output_root / f"jump_{index}"
        jump_dir.mkdir(parents=True, exist_ok=True)
        left, right, metadata = _generate_jump_pair(seeds, rng, athlete)
        left.to_csv(jump_dir / "left_bwt901cl_raw.csv", index=False)
        right.to_csv(jump_dir / "right_bwt901cl_raw.csv", index=False)
        (jump_dir / "recording_metadata.json").write_text(
            json.dumps(
                {
                    "jump_name": f"jump_{index}",
                    "data_type": "synthetic_imu_only_drop_jump",
                    **metadata,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    _write_readme(output_root, athlete, args.count, args.seed)


if __name__ == "__main__":
    main()
