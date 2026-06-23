from __future__ import annotations

"""Validazione del PitchTransformer sui trial reali con IMU.

Carica models/pitch_transformer.pt e valuta sui 20 trial personali.
Produce scatter plot e timeseries plot per confrontare video e sensori.

Usage
-----
    python scripts/validate_pitch_transformer.py
    python scripts/validate_pitch_transformer.py --trials-dir collected_trials --model models/pitch_transformer.pt
"""

import argparse
import os
import sys
from pathlib import Path

MPLCONFIGDIR = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jump_analysis.models.pitch_transformer import PitchTransformer
from jump_analysis.features.front_2d_features import select_temporal_features


KP_X_COLS  = [f"kp_{i:02d}_x_px" for i in range(17)]
KP_Y_COLS  = [f"kp_{i:02d}_y_px" for i in range(17)]
PITCH_COLS = ["left_sensor_pitch_deg", "right_sensor_pitch_deg"]
LABELS     = {"left": "Left Pitch", "right": "Right Pitch"}


def angle_diff(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return ((pred - true + 180.0) % 360.0) - 180.0


def evaluate(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    diff  = angle_diff(pred, true)
    valid = np.isfinite(pred) & np.isfinite(true)
    mae   = float(np.nanmean(np.abs(diff)))
    rmse  = float(np.sqrt(np.nanmean(diff ** 2)))
    corr  = float(np.corrcoef(pred[valid], true[valid])[0, 1]) if valid.sum() > 2 else float("nan")
    bias  = float(np.nanmean(diff))
    return {"mae_deg": mae, "rmse_deg": rmse, "correlation": corr, "bias_deg": bias}


def load_trial(trial_dir: Path, model: PitchTransformer, device: str) -> dict | None:
    ts_path = trial_dir / "movement_timeseries.csv"
    if not ts_path.exists():
        return None

    ts = pd.read_csv(ts_path)

    missing = [c for c in KP_X_COLS + KP_Y_COLS + PITCH_COLS + ["body_height_px"]
               if c not in ts.columns]
    if missing:
        print(f"  [skip] {trial_dir.name}: colonne mancanti {missing[:3]}")
        return None

    if ts[PITCH_COLS].isna().all().any():
        print(f"  [skip] {trial_dir.name}: IMU columns missing")
        return None

    height = ts["body_height_px"].median()
    if height < 10:
        print(f"  [skip] {trial_dir.name}: body_height_px troppo piccola")
        return None

    # Keypoint normalizzati. New models use body-only channels.
    kp_x = ts[KP_X_COLS].to_numpy(dtype=np.float32) / height
    kp_y = ts[KP_Y_COLS].to_numpy(dtype=np.float32) / height
    seq  = np.concatenate([kp_x, kp_y], axis=1)
    if model.input_proj.in_features == 24:
        seq = select_temporal_features(seq, include_head=False)

    # Predizione Transformer → (T, 2) gradi (delta pitch)
    pred = model.predict_numpy(seq, device=device)  # (T, 2)

    # IMU: delta dal primo frame
    imu = ts[PITCH_COLS].to_numpy(dtype=np.float32)
    imu_delta = imu - imu[0]

    time = ts["time_from_start_s"].to_numpy() if "time_from_start_s" in ts.columns else np.arange(len(ts))

    return {
        "trial": trial_dir.name,
        "time":  time,
        "pred":  {"left": pred[:, 0], "right": pred[:, 1]},
        "imu":   {"left": imu_delta[:, 0], "right": imu_delta[:, 1]},
    }


def plot_timeseries(trials: list[dict], output_path: str) -> None:
    n = len(trials)
    fig, axes = plt.subplots(n, 2, figsize=(14, 2.5 * n), sharex=False)
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, trial in enumerate(trials):
        for col, side in enumerate(["left", "right"]):
            ax  = axes[row, col]
            t   = trial["time"]
            pr  = trial["pred"][side]
            imu = trial["imu"][side]

            ax.plot(t, pr,  label="Transformer", color="steelblue",  linewidth=1.2)
            ax.plot(t, imu, label="IMU",          color="darkorange", linewidth=1.2, alpha=0.85)
            ax.set_ylabel("Δ pitch (°)")
            ax.set_title(f"{trial['trial'][-15:]}  —  {LABELS[side]}", fontsize=7)
            ax.legend(fontsize=6, loc="upper right")
            if row == n - 1:
                ax.set_xlabel("Tempo (s)")

    fig.suptitle("Validazione PitchTransformer: camera vs IMU (Δ rispetto al baseline)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Timeseries plot saved to {output_path}")


def plot_scatter(trials: list[dict], output_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for col, side in enumerate(["left", "right"]):
        all_pred = np.concatenate([t["pred"][side] for t in trials])
        all_imu  = np.concatenate([t["imu"][side]  for t in trials])
        valid    = np.isfinite(all_pred) & np.isfinite(all_imu)
        p, i = all_pred[valid], all_imu[valid]

        metrics = evaluate(p, i)
        ax = axes[col]
        ax.scatter(i, p, alpha=0.2, s=5, color="steelblue", rasterized=True)
        lim = max(np.abs(i).max(), np.abs(p).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1, label="ideale")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("IMU (°)")
        ax.set_ylabel("Transformer predetto (°)")
        ax.set_title(
            f"{LABELS[side]}\n"
            f"MAE={metrics['mae_deg']:.2f}°  RMSE={metrics['rmse_deg']:.2f}°  "
            f"r={metrics['correlation']:.3f}  bias={metrics['bias_deg']:+.2f}°"
        )
        ax.legend(fontsize=8)

    fig.suptitle("PitchTransformer — Camera vs IMU su 20 trial reali", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Scatter plot saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials-dir",         default="collected_trials")
    parser.add_argument("--model",               default="models/pitch_transformer.pt")
    parser.add_argument("--timeseries-output",   default="outputs/plots/transformer_validation_timeseries.png")
    parser.add_argument("--scatter-output",      default="outputs/plots/transformer_validation_scatter.png")
    parser.add_argument("--csv-output",          default="outputs/transformer_validation_results.csv")
    args = parser.parse_args()
    Path(args.timeseries_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.scatter_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.csv_output).parent.mkdir(parents=True, exist_ok=True)

    device = PitchTransformer.best_device()
    print(f"Device: {device}")

    model = PitchTransformer.load(args.model, device=device)
    model.to(device)

    trials_dir = Path(args.trials_dir)
    trial_dirs = sorted([d for d in trials_dir.iterdir() if d.is_dir()])
    print(f"Trial trovati: {len(trial_dirs)}")

    trials, rows = [], []
    for td in trial_dirs:
        result = load_trial(td, model, device)
        if result is None:
            continue
        trials.append(result)

        for side in ["left", "right"]:
            metrics = evaluate(result["pred"][side], result["imu"][side])
            rows.append({"trial": result["trial"], "side": side, **metrics})
            print(f"  {result['trial'][-15:]}  {side:5s}  MAE={metrics['mae_deg']:.2f}°  "
                  f"r={metrics['correlation']:.3f}  bias={metrics['bias_deg']:+.2f}°")

    if not trials:
        print("Nessun trial valido.")
        return

    results_df = pd.DataFrame(rows)
    results_df.to_csv(args.csv_output, index=False)

    print("\n=== Summary per lato ===")
    for side, grp in results_df.groupby("side"):
        print(f"  {side}  MAE={grp['mae_deg'].mean():.2f}±{grp['mae_deg'].std():.2f}°  "
              f"r={grp['correlation'].mean():.3f}  bias={grp['bias_deg'].mean():+.2f}°")

    plot_timeseries(trials, args.timeseries_output)
    plot_scatter(trials, args.scatter_output)
    print(f"\nRisultati salvati in {args.csv_output}")


if __name__ == "__main__":
    main()
