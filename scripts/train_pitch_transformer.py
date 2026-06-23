from __future__ import annotations

"""Training del PitchTransformer su dati sintetici, validazione sui trial reali.

Flusso:
  1. Carica data/generated/synthetic_trials.npz (generato da generate_synthetic_trials.py).
  2. Split: trial reali → validation, trial sintetici → training.
  3. Addestra PitchTransformer con MSE loss e maschera di padding.
  4. Valuta su trial reali: MAE, RMSE, correlazione, bias.
  5. Salva modello e plot.

Usage
-----
    python scripts/train_pitch_transformer.py
    python scripts/train_pitch_transformer.py --data data/generated/synthetic_trials.npz --epochs 200
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jump_analysis.models.pitch_transformer import PitchTransformer
from jump_analysis.features.front_2d_features import select_temporal_features


# ── dataset ───────────────────────────────────────────────────────────────────

class PitchDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray, lengths: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets   = torch.tensor(targets,   dtype=torch.float32)
        self.lengths   = lengths

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]   # (T, D)
        tgt = self.targets[idx]     # (T, 2)
        L   = int(self.lengths[idx])
        # Padding mask: True = frame da ignorare
        T = seq.shape[0]
        mask = torch.zeros(T, dtype=torch.bool)
        mask[L:] = True
        return seq, tgt, mask, L


# ── metriche ──────────────────────────────────────────────────────────────────

def evaluate_numpy(pred: np.ndarray, true: np.ndarray, length: int) -> dict:
    p, t = pred[:length], true[:length]
    diff = ((p - t + 180) % 360) - 180
    mae  = float(np.nanmean(np.abs(diff)))
    rmse = float(np.sqrt(np.nanmean(diff ** 2)))
    bias = float(np.nanmean(diff))
    valid = np.isfinite(p.ravel()) & np.isfinite(t.ravel())
    corr = float(np.corrcoef(p.ravel()[valid], t.ravel()[valid])[0, 1]) if valid.sum() > 2 else float("nan")
    return {"mae": mae, "rmse": rmse, "bias": bias, "corr": corr}


# ── training ──────────────────────────────────────────────────────────────────

def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE ignorando i frame di padding."""
    valid = ~mask.unsqueeze(-1).expand_as(pred)
    diff  = (pred - target)[valid]
    return (diff ** 2).mean()


def train(args: argparse.Namespace) -> None:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # ── dati ─────────────────────────────────────────────────────────────────
    data     = np.load(args.data)
    sequences = data["sequences"]  # (N, T, 34) or (N, T, 24)
    if sequences.shape[-1] == 34:
        sequences = select_temporal_features(sequences, include_head=False)
    targets   = data["targets"]    # (N, T, 2)
    lengths   = data["lengths"]    # (N,)
    is_real   = data["is_real"]    # (N,)

    train_idx = np.where(~is_real)[0]
    val_idx   = np.where(is_real)[0]
    print(f"Training (sintetici): {len(train_idx)}  |  Validation (reali): {len(val_idx)}")

    train_ds = PitchDataset(sequences[train_idx], targets[train_idx], lengths[train_idx])
    val_seqs  = sequences[val_idx]
    val_tgts  = targets[val_idx]
    val_lens  = lengths[val_idx]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # ── modello ───────────────────────────────────────────────────────────────
    model = PitchTransformer(
        input_dim=sequences.shape[-1],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parametri: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # ── loop ──────────────────────────────────────────────────────────────────
    train_losses, val_maes = [], []
    best_val_mae = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for seq, tgt, mask, _ in train_loader:
            seq, tgt, mask = seq.to(device), tgt.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(seq, src_key_padding_mask=mask, causal=True)
            loss = masked_mse(pred, tgt, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation ogni 10 epoch
        if epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            all_metrics = []
            with torch.no_grad():
                for i in range(len(val_seqs)):
                    seq = torch.tensor(val_seqs[i], dtype=torch.float32, device=device).unsqueeze(0)
                    pred = model(seq, causal=True).squeeze(0).cpu().numpy()
                    m = evaluate_numpy(pred, val_tgts[i], val_lens[i])
                    all_metrics.append(m)

            val_mae  = float(np.mean([m["mae"]  for m in all_metrics]))
            val_corr = float(np.mean([m["corr"] for m in all_metrics]))
            val_maes.append((epoch, val_mae))
            print(f"Epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.4f}  val_MAE={val_mae:.2f}°  val_r={val_corr:.3f}")

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                model.save(args.output)
                print(f"  → Nuovo best, modello salvato ({args.output})")

    # ── valutazione finale ────────────────────────────────────────────────────
    print(f"\nCarico best model ({args.output}) per valutazione finale...")
    best_model = PitchTransformer.load(args.output, device=device)
    best_model.to(device)
    best_model.eval()

    all_metrics = {"left": [], "right": []}
    with torch.no_grad():
        for i in range(len(val_seqs)):
            seq  = torch.tensor(val_seqs[i], dtype=torch.float32, device=device).unsqueeze(0)
            pred = best_model(seq, causal=True).squeeze(0).cpu().numpy()
            L    = val_lens[i]
            for j, side in enumerate(["left", "right"]):
                m = evaluate_numpy(pred[:, j:j+1], val_tgts[i, :, j:j+1], L)
                all_metrics[side].append(m)

    print("\n=== Risultati finali su trial reali ===")
    for side in ["left", "right"]:
        mae  = np.mean([m["mae"]  for m in all_metrics[side]])
        rmse = np.mean([m["rmse"] for m in all_metrics[side]])
        corr = np.mean([m["corr"] for m in all_metrics[side]])
        bias = np.mean([m["bias"] for m in all_metrics[side]])
        print(f"  {side:5s}  MAE={mae:.2f}°  RMSE={rmse:.2f}°  r={corr:.3f}  bias={bias:+.2f}°")

    # ── plot training loss ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses, color="steelblue", linewidth=1)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss (training sintetici)")
    axes[0].set_title("Training loss")

    val_epochs, val_mae_vals = zip(*val_maes)
    axes[1].plot(val_epochs, val_mae_vals, color="darkorange", marker="o", markersize=3)
    axes[1].axhline(best_val_mae, color="red", linestyle="--", linewidth=1,
                    label=f"best={best_val_mae:.2f}°")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (°) su trial reali")
    axes[1].set_title("Validation MAE (pitch aggregato)")
    axes[1].legend()

    plt.suptitle("PitchTransformer — Training", fontsize=12)
    plt.tight_layout()
    plt.savefig(args.plot_output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot salvato in {args.plot_output}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train PitchTransformer su dati sintetici.")
    parser.add_argument("--data",            default="data/generated/synthetic_trials.npz")
    parser.add_argument("--output",          default="models/pitch_transformer.pt")
    parser.add_argument("--plot-output",     default="outputs/plots/pitch_transformer_training.png")
    parser.add_argument("--epochs",          type=int,   default=300)
    parser.add_argument("--batch-size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--d-model",         type=int,   default=64)
    parser.add_argument("--nhead",           type=int,   default=4)
    parser.add_argument("--num-layers",      type=int,   default=3)
    parser.add_argument("--dim-feedforward", type=int,   default=128)
    parser.add_argument("--dropout",         type=float, default=0.1)
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.plot_output).parent.mkdir(parents=True, exist_ok=True)

    train(args)


if __name__ == "__main__":
    main()
