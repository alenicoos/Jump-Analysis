from __future__ import annotations

"""Training del JumpAutoencoder (LSTM).

Flusso
------
1. Carica data/generated/mocap_sequences.npz, estratto dal dataset
   "Three-Dimensional Motion Capture Data of a Movement Screen from 183
   Athletes" di Zhao et al. (2023).
2. Augmenta: genera --n-synthetic sequenze sintetiche dai 183 reali
   (noise, time warp, scale, flip, shift). Di queste, --n-val-synthetic
   vanno in validation, le restanti in training.
   Training totale = 183 reali + (n_synthetic - n_val_synthetic) sintetici.
   Validation = n_val_synthetic sintetici.
3. Addestra JumpAutoencoder con MSE loss (masked padding).
4. Calibra soglia al percentile --threshold-percentile degli errori sui 183 reali.
5. Valuta su data/generated/anomalous_trials.npz.
6. Salva modello e plot di training.

Usage
-----
    python scripts/train_jump_autoencoder.py
    python scripts/train_jump_autoencoder.py --epochs 300 --n-synthetic 1000 --n-val-synthetic 200
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

from jump_analysis.models.jump_autoencoder import JumpAutoencoder
from jump_analysis.features.front_2d_features import (
    select_temporal_features,
    extract_ae_features,
    AE_FEATURE_DIM,
)


# ── Dataset ───────────────────────────────────────────────────────────────────

class JumpDataset(Dataset):
    def __init__(self, sequences: np.ndarray, lengths: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.lengths   = lengths

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]          # (seq_len, D)
        L   = int(self.lengths[idx])
        T   = seq.shape[0]
        mask = torch.zeros(T, dtype=torch.bool)
        mask[L:] = True                    # True = padding
        return seq, mask, L






# ── Augmentazione sequenze normali ────────────────────────────────────────────

# Flip pairs for body-only layout (12 keypoints, indices 0-11):
# [L_shoulder, R_shoulder, L_elbow, R_elbow, L_wrist, R_wrist,
#  L_hip, R_hip, L_knee, R_knee, L_ankle, R_ankle]
FLIP_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]


def augment_normal(seq_full: np.ndarray, L: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Augmenta una sequenza di keypoint grezzi (T, 24) e restituisce (T, 24) augmentato.

    Il noise viene applicato sui keypoint con sigma moderato per evitare
    esplosioni negli angoli derivati. L'augmentazione YOLO-realistica viene
    applicata in seguito, nello spazio delle feature (vedi augment_features).
    """
    seq_len     = seq_full.shape[0]
    kp_dim      = seq_full.shape[1]   # 24
    n_keypoints = kp_dim // 2
    kp = seq_full[:L].copy()

    # 1. Time warp
    warp    = rng.uniform(0.80, 1.20)
    new_L   = max(10, int(L * warp))
    old_idx = np.linspace(0, L - 1, new_L)
    kp = np.array([np.interp(old_idx, np.arange(L), kp[:, d]) for d in range(kp_dim)]).T.astype(np.float32)
    L  = new_L

    # 2. Scale
    kp = kp * rng.uniform(0.90, 1.10)

    # 3. Shift orizzontale x
    kp[:, :n_keypoints] += rng.uniform(-0.05, 0.05)

    # 4. Flip laterale (50%)
    if rng.random() < 0.5:
        x_center = kp[:, :n_keypoints].mean()
        kp[:, :n_keypoints] = 2 * x_center - kp[:, :n_keypoints]
        for l, r in FLIP_PAIRS:
            kp[:, [l, r]] = kp[:, [r, l]]
            kp[:, [n_keypoints + l, n_keypoints + r]] = kp[:, [n_keypoints + r, n_keypoints + l]]

    # 5. Noise moderato sui keypoint (sigma << distanze tipiche tra keypoint)
    kp = kp + rng.normal(0, rng.uniform(0.003, 0.010), kp.shape).astype(np.float32)

    L_out = min(L, seq_len)
    out   = np.zeros((seq_len, kp_dim), dtype=np.float32)
    out[:L_out] = kp[:L_out]
    return out, L_out


def augment_features(feat: np.ndarray, L: int, rng: np.random.Generator) -> np.ndarray:
    """Applica noise YOLO-realistico direttamente sulle feature (T, 16).

    Noise separato per angoli (gradi) e ratios (adimensionali), calibrato
    sulla variabilità reale dei due tipi di feature.
    """
    f = feat.copy()
    # Angoli: feature 0-3 (knee/hip flex) e 12-14 (asimmetrie angolari) — in gradi
    angle_idx  = [0, 1, 2, 3, 12, 13]
    # Tilt: feature 15 (shoulder_tilt) — in unità sw, range ~0-0.3
    tilt_idx   = [15]
    # Ratio/x features: tutto il resto
    ratio_idx  = [i for i in range(feat.shape[1]) if i not in angle_idx + tilt_idx]

    sigma_angle = rng.uniform(1.0, 4.0)   # gradi — calibrato su rumore YOLO
    sigma_ratio = rng.uniform(0.02, 0.06) # ~2-6% di variazione su ratios
    sigma_tilt  = rng.uniform(0.01, 0.04)

    f[:L, angle_idx] += rng.normal(0, sigma_angle, (L, len(angle_idx))).astype(np.float32)
    f[:L, ratio_idx] += rng.normal(0, sigma_ratio, (L, len(ratio_idx))).astype(np.float32)
    f[:L, tilt_idx]  += rng.normal(0, sigma_tilt,  (L, len(tilt_idx))).astype(np.float32)
    return f


def generate_augmented_normals(
    kp_seqs: np.ndarray,      # (N, seq_len, 24) raw keypoints
    lengths: np.ndarray,      # (N,)
    n_synthetic: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera n_synthetic sequenze augmentate dai normali.

    Augmentazione in keypoint space, poi conversione in feature invarianti.
    """
    N = len(kp_seqs)
    seq_len = kp_seqs.shape[1]
    out_seqs = np.zeros((n_synthetic, seq_len, AE_FEATURE_DIM), dtype=np.float32)
    out_lens = np.zeros(n_synthetic, dtype=np.int32)

    for i in range(n_synthetic):
        src = int(rng.integers(0, N))
        aug_kp, L = augment_normal(kp_seqs[src], int(lengths[src]), rng)
        feat = extract_ae_features(aug_kp)             # keypoint aug → features
        feat = augment_features(feat, L, rng)          # feature-space noise
        out_seqs[i] = feat
        out_lens[i] = L

    return out_seqs, out_lens


# ── Loss con maschera padding ──────────────────────────────────────────────────

def masked_mse(
    recon: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid = ~mask.unsqueeze(-1).expand_as(recon)
    diff  = (recon - target)[valid]
    return (diff ** 2).mean()


# ── Training ──────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = JumpAutoencoder.best_device()
    print(f"Device: {device}")

    # ── Carica dati normali ───────────────────────────────────────────────────
    data = np.load(args.normal_data)
    kp_seqs    = data["sequences"]    # (N, seq_len, 34) or (N, seq_len, 24)
    lengths    = data["lengths"]      # (N,)
    N = len(kp_seqs)
    print(f"Sequenze normali: {N}")

    if kp_seqs.shape[-1] == 34:
        kp_seqs = select_temporal_features(kp_seqs, include_head=False)
    # kp_seqs: (N, seq_len, 24) — raw keypoints, used for augmentation

    # Convert to domain-invariant features (N, seq_len, 14)
    full_seqs = np.stack([
        extract_ae_features(kp_seqs[i]) for i in range(len(kp_seqs))
    ], axis=0).astype(np.float32)

    # ── Augmentazione: genera sintetici per training e validation ─────────────
    rng = np.random.default_rng(42)
    n_trn_aug = args.n_synthetic - args.n_val_synthetic
    print(f"Generazione {args.n_synthetic} sequenze augmentate "
          f"({n_trn_aug} training + {args.n_val_synthetic} validation)...")

    aug_trn_seqs, aug_trn_lens = generate_augmented_normals(kp_seqs, lengths, n_trn_aug, rng)
    aug_val_seqs, aug_val_lens = generate_augmented_normals(kp_seqs, lengths, args.n_val_synthetic, rng)

    # Training = 183 reali (features) + sintetici di training
    trn_seqs = np.concatenate([full_seqs, aug_trn_seqs], axis=0)
    trn_lens = np.concatenate([lengths,   aug_trn_lens])

    # Validation = solo sintetici (mai visti durante training)
    val_seqs = aug_val_seqs
    val_lens = aug_val_lens

    print(f"Training totale: {len(trn_seqs)} ({N} reali + {n_trn_aug} sintetici)")
    print(f"Validation:      {len(val_seqs)} sintetici")

    # ── Autoencoder ───────────────────────────────────────────────────────────
    D = full_seqs.shape[-1]
    model = JumpAutoencoder(
        input_dim=D,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    model.fit_normalization(full_seqs, lengths)
    model.input_mean = model.input_mean.to(device)
    model.input_std  = model.input_std.to(device)

    print(f"Parametri autoencoder: {sum(p.numel() for p in model.parameters()):,}")

    train_ds = JumpDataset(trn_seqs, trn_lens)
    val_ds   = JumpDataset(val_seqs, val_lens)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5,
    )

    train_losses, val_losses = [], []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for seq, mask, _ in train_dl:
            seq, mask = seq.to(device), mask.to(device)
            seq_norm  = model.normalize(seq)
            recon     = model(seq_norm, mask)
            loss      = masked_mse(recon, seq_norm, mask)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(train_dl)
        train_losses.append(avg)

        if epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            val_errors = []
            with torch.no_grad():
                for seq, mask, _ in DataLoader(val_ds, batch_size=32):
                    seq, mask = seq.to(device), mask.to(device)
                    err = model.reconstruction_error(seq, mask)
                    val_errors.extend(err.cpu().numpy().tolist())
            val_loss = float(np.mean(val_errors))
            val_losses.append((epoch, val_loss))
            scheduler.step(val_loss)
            print(f"Epoch {epoch:4d}/{args.epochs}  train_loss={avg:.5f}  val_recon_err={val_loss:.5f}  lr={optimizer.param_groups[0]['lr']:.2e}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "config": {"input_dim": full_seqs.shape[-1], "hidden_dim": args.hidden_dim,
                               "latent_dim": args.latent_dim, "num_layers": args.num_layers,
                               "dropout": 0.0},
                    "state_dict": model.state_dict(),
                    "anomaly_threshold": None,
                }, args.output + ".tmp")

    # ── Carica best, calibra soglia ───────────────────────────────────────────
    ckpt = torch.load(args.output + ".tmp", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Calcola errori di ricostruzione su TUTTI i normali
    all_errors = []
    with torch.no_grad():
        for seq, mask, _ in DataLoader(
            JumpDataset(full_seqs, lengths), batch_size=32
        ):
            seq, mask = seq.to(device), mask.to(device)
            err = model.reconstruction_error(seq, mask)
            all_errors.extend(err.cpu().numpy().tolist())

    threshold = float(np.percentile(all_errors, args.threshold_percentile))
    model.anomaly_threshold = threshold
    print(f"\nSoglia anomalia ({args.threshold_percentile}° percentile normali): {threshold:.5f}")
    model.save(args.output)
    print(f"Autoencoder salvato: {args.output}")


    # ── Valutazione su anomalie sintetiche ────────────────────────────────────
    anom_path = Path(args.anomalous_data)
    if not anom_path.exists():
        print(f"\n[info] {anom_path} non trovato — salto valutazione anomalie.")
        print("Esegui prima: python scripts/generate_anomalous_trials.py")
    else:
        print("\n=== Valutazione su anomalie sintetiche ===")
        anom = np.load(anom_path)
        anom_kp    = anom["sequences"]   # (M, seq_len, 34) or (M, seq_len, 24)
        if anom_kp.shape[-1] == 34:
            anom_kp = select_temporal_features(anom_kp, include_head=False)
        anom_lens  = anom["lengths"]
        anom_types = anom["anomaly_type"]
        # Convert anomalous keypoints to domain-invariant features
        anom_full  = np.stack([
            extract_ae_features(anom_kp[i]) for i in range(len(anom_kp))
        ], axis=0).astype(np.float32)

        # Autoencoder scores
        ae_scores_anom = []
        with torch.no_grad():
            for i in range(len(anom_full)):
                L   = int(anom_lens[i])
                seq = torch.tensor(anom_full[i], dtype=torch.float32, device=device).unsqueeze(0)
                T   = seq.shape[1]
                mask = torch.zeros(1, T, dtype=torch.bool, device=device)
                mask[0, L:] = True
                err = model.reconstruction_error(seq, mask)
                ae_scores_anom.append(float(err.item()))

        print(f"\nNormali — AE errore medio: {np.mean(all_errors):.5f}  std: {np.std(all_errors):.5f}")
        print(f"Soglia: {threshold:.5f}")
        print()
        print(f"{'Tipo anomalia':<25}  {'AE score':>10}  {'AE detect%':>10}")
        print("-" * 48)
        for atype in np.unique(anom_types):
            mask_t = anom_types == atype
            ae_s   = np.array(ae_scores_anom)[mask_t]
            ae_det = float((ae_s > threshold).mean() * 100)
            print(f"  {atype:<23}  {ae_s.mean():>10.5f}  {ae_det:>9.1f}%")

    # Training loss plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses, color="steelblue", linewidth=1)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training loss")

    val_ep, val_v = zip(*val_losses)
    axes[1].plot(val_ep, val_v, color="darkorange", marker="o", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Reconstruction error (MSE)")
    axes[1].set_title("Validation reconstruction error")

    plt.suptitle("JumpAutoencoder — Training", fontsize=11)
    plt.tight_layout()
    plt.savefig(args.training_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training plot salvato: {args.training_plot}")

    Path(args.output + ".tmp").unlink(missing_ok=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Addestra JumpAutoencoder su dati mocap normali.")
    parser.add_argument("--normal-data",      default="data/generated/mocap_sequences.npz")
    parser.add_argument("--anomalous-data",   default="data/generated/anomalous_trials.npz")
    parser.add_argument("--output",           default="models/jump_autoencoder_lstm.pt")
    parser.add_argument("--training-plot",    default="outputs/plots/autoencoder_training.png")
    parser.add_argument("--epochs",             type=int,   default=300)
    parser.add_argument("--batch-size",         type=int,   default=32)
    parser.add_argument("--lr",                 type=float, default=1e-3)
    parser.add_argument("--hidden-dim",         type=int,   default=64)
    parser.add_argument("--latent-dim",         type=int,   default=32)
    parser.add_argument("--num-layers",         type=int,   default=2)
    parser.add_argument("--dropout",            type=float, default=0.2)
    parser.add_argument("--n-synthetic",        type=int,   default=1000,
                        help="Totale sequenze sintetiche augmentate (training + validation).")
    parser.add_argument("--n-val-synthetic",    type=int,   default=200,
                        help="Quante sintetiche usare per validation (le restanti vanno in training).")
    parser.add_argument("--seed",                 type=int,   default=42)
    parser.add_argument("--threshold-percentile", type=float, default=99.0,
                        help="Percentile degli errori normali per la soglia anomalia.")
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.training_plot).parent.mkdir(parents=True, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
