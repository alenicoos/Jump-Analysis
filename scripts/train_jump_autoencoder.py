from __future__ import annotations

"""Training del JumpAutoencoder (LSTM).

Flusso
------
1. Carica data/generated/mocap_sequences.npz (183 atleti normali).
2. Genera pitch predetto con PitchTransformer → concatena (T,36).
3. Augmenta: genera --n-synthetic sequenze sintetiche dai 183 reali
   (noise, time warp, scale, flip, shift). Di queste, --n-val-synthetic
   vanno in validation, le restanti in training.
   Training totale = 183 reali + (n_synthetic - n_val_synthetic) sintetici.
   Validation = n_val_synthetic sintetici.
4. Addestra JumpAutoencoder con MSE loss (masked padding).
5. Calibra soglia al percentile --threshold-percentile degli errori sui 183 reali.
6. Valuta su data/generated/anomalous_trials.npz.
7. Salva modello e plot di training.

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
from jump_analysis.models.pitch_transformer import PitchTransformer


# ── Dataset ───────────────────────────────────────────────────────────────────

class JumpDataset(Dataset):
    def __init__(self, sequences: np.ndarray, lengths: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.lengths   = lengths

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]          # (seq_len, 36)
        L   = int(self.lengths[idx])
        T   = seq.shape[0]
        mask = torch.zeros(T, dtype=torch.bool)
        mask[L:] = True                    # True = padding
        return seq, mask, L


# ── Generazione pitch dal Transformer ─────────────────────────────────────────

def generate_pitch_sequences(
    keypoints: np.ndarray,    # (N, seq_len, 34)
    lengths: np.ndarray,      # (N,)
    pitch_model: PitchTransformer,
    device: str,
) -> np.ndarray:
    """Esegue PitchTransformer su tutte le sequenze mocap → (N, seq_len, 2)."""
    N, seq_len, _ = keypoints.shape
    pitch_out = np.zeros((N, seq_len, 2), dtype=np.float32)

    pitch_model.eval()
    with torch.no_grad():
        for i in range(N):
            L   = int(lengths[i])
            seq = keypoints[i, :L]   # (L, 34)
            p   = pitch_model.predict_numpy(seq, device=device)   # (L, 2)
            pitch_out[i, :L] = p

    return pitch_out




# ── Augmentazione sequenze normali ────────────────────────────────────────────

FLIP_PAIRS = [
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16),
]


def augment_normal(seq36: np.ndarray, L: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Applica augmentazioni casuali a una sequenza (seq_len, 36) normale.

    Augmenta i 34 keypoint (colonne 0-33) e il pitch (colonne 34-35).
    Restituisce (seq_aug, L_aug) con la stessa seq_len originale (padded).
    """
    seq_len = seq36.shape[0]
    kp  = seq36[:L, :34].copy()   # (L, 34)
    pit = seq36[:L, 34:].copy()   # (L, 2)

    # 1. Time warp
    warp   = rng.uniform(0.80, 1.20)
    new_L  = max(10, int(L * warp))
    old_idx = np.linspace(0, L - 1, new_L)
    kp  = np.array([np.interp(old_idx, np.arange(L), kp[:, d])  for d in range(34)]).T.astype(np.float32)
    pit = np.array([np.interp(old_idx, np.arange(L), pit[:, d]) for d in range(2)]).T.astype(np.float32)
    L   = new_L

    # 2. Scale keypoint
    scale = rng.uniform(0.90, 1.10)
    kp = kp * scale

    # 3. Shift orizzontale keypoint x
    shift = rng.uniform(-0.05, 0.05)
    kp[:, :17] += shift

    # 4. Flip laterale (50%)
    if rng.random() < 0.5:
        x_center = kp[:, :17].mean()
        kp[:, :17] = 2 * x_center - kp[:, :17]
        for l, r in FLIP_PAIRS:
            kp[:, [l, r]]       = kp[:, [r, l]]
            kp[:, [17+l, 17+r]] = kp[:, [17+r, 17+l]]
        pit = pit[:, [1, 0]]   # swap left/right pitch

    # 5. Gaussian noise keypoint
    sigma = rng.uniform(0.003, 0.010)
    kp = kp + rng.normal(0, sigma, kp.shape).astype(np.float32)

    # 6. Piccolo noise sul pitch
    pit = pit + rng.normal(0, 0.5, pit.shape).astype(np.float32)

    # Ripadda a seq_len
    L_out = min(L, seq_len)
    out   = np.zeros((seq_len, 36), dtype=np.float32)
    out[:L_out, :34] = kp[:L_out]
    out[:L_out, 34:] = pit[:L_out]
    return out, L_out


def generate_augmented_normals(
    full_seqs: np.ndarray,    # (N, seq_len, 36)
    lengths: np.ndarray,      # (N,)
    n_synthetic: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera n_synthetic sequenze augmentate dai normali."""
    N = len(full_seqs)
    seq_len = full_seqs.shape[1]
    out_seqs = np.zeros((n_synthetic, seq_len, 36), dtype=np.float32)
    out_lens = np.zeros(n_synthetic, dtype=np.int32)

    for i in range(n_synthetic):
        src = int(rng.integers(0, N))
        aug, L = augment_normal(full_seqs[src], int(lengths[src]), rng)
        out_seqs[i] = aug
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
    device = JumpAutoencoder.best_device()
    print(f"Device: {device}")

    # ── Carica dati normali ───────────────────────────────────────────────────
    data = np.load(args.normal_data)
    kp_seqs    = data["sequences"]    # (N, seq_len, 34)
    lengths    = data["lengths"]      # (N,)
    ic_frames  = data["ic_frames"]
    kfmax_frames = data["kfmax_frames"]
    N = len(kp_seqs)
    print(f"Sequenze normali: {N}")

    # ── Genera pitch con PitchTransformer ─────────────────────────────────────
    print("Generazione pitch con PitchTransformer...")
    pitch_model = PitchTransformer.load(args.pitch_model, device=device)
    pitch_model.to(device)
    pitch_seqs = generate_pitch_sequences(kp_seqs, lengths, pitch_model, device)
    # (N, seq_len, 2)

    # Concatena: (N, seq_len, 36)
    full_seqs = np.concatenate([kp_seqs, pitch_seqs], axis=2).astype(np.float32)

    # ── Augmentazione: genera sintetici per training e validation ─────────────
    rng = np.random.default_rng(42)
    n_trn_aug = args.n_synthetic - args.n_val_synthetic
    print(f"Generazione {args.n_synthetic} sequenze augmentate "
          f"({n_trn_aug} training + {args.n_val_synthetic} validation)...")

    aug_trn_seqs, aug_trn_lens = generate_augmented_normals(full_seqs, lengths, n_trn_aug, rng)
    aug_val_seqs, aug_val_lens = generate_augmented_normals(full_seqs, lengths, args.n_val_synthetic, rng)

    # Training = 183 reali + sintetici di training
    trn_seqs = np.concatenate([full_seqs, aug_trn_seqs], axis=0)
    trn_lens = np.concatenate([lengths,   aug_trn_lens])

    # Validation = solo sintetici (mai visti durante training)
    val_seqs = aug_val_seqs
    val_lens = aug_val_lens

    print(f"Training totale: {len(trn_seqs)} ({N} reali + {n_trn_aug} sintetici)")
    print(f"Validation:      {len(val_seqs)} sintetici")

    # ── Autoencoder ───────────────────────────────────────────────────────────
    model = JumpAutoencoder(
        input_dim=36,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    model.fit_normalization(full_seqs, lengths)   # normalizzazione sui 183 reali
    model.input_mean = model.input_mean.to(device)
    model.input_std  = model.input_std.to(device)

    print(f"Parametri autoencoder: {sum(p.numel() for p in model.parameters()):,}")

    train_ds = JumpDataset(trn_seqs, trn_lens)
    val_ds   = JumpDataset(val_seqs, val_lens)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

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
        scheduler.step()

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
            print(f"Epoch {epoch:4d}/{args.epochs}  train_loss={avg:.5f}  val_recon_err={val_loss:.5f}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "config": {"input_dim": 36, "hidden_dim": args.hidden_dim,
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
        anom_kp    = anom["sequences"]   # (M, seq_len, 34)
        anom_lens  = anom["lengths"]
        anom_types = anom["anomaly_type"]
        anom_ic    = anom["ic_frames"]
        anom_kf    = anom["kfmax_frames"]

        # Genera pitch per le anomalie
        print("Generazione pitch per trial anomali...")
        anom_pitch = generate_pitch_sequences(anom_kp, anom_lens, pitch_model, device)
        anom_full  = np.concatenate([anom_kp, anom_pitch], axis=2).astype(np.float32)

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
    parser.add_argument("--pitch-model",      default="models/pitch_transformer.pt")
    parser.add_argument("--output",           default="models/jump_autoencoder.pt")
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
    parser.add_argument("--threshold-percentile", type=float, default=99.0,
                        help="Percentile degli errori normali per la soglia anomalia.")
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.training_plot).parent.mkdir(parents=True, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
