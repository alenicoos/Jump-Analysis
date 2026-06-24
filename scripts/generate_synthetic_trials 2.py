from __future__ import annotations

"""Genera dati sintetici per il training del Transformer pitch.

Prende i 20 trial reali da collected_trials/ e produce 200 trial sintetici
applicando augmentazioni biomeccanicamente plausibili.

Input per ogni trial:
  - Keypoint 2D (17 × 2) normalizzati per body_height_px → sequenza (T, 34)
  - Pitch IMU normalizzato (baseline = primo frame) → sequenza (T, 2)

Augmentazioni applicate in combinazione casuale:
  - Gaussian noise sui keypoint (σ = 0.005–0.015 in unità normalizzate)
  - Time warping: ricampionamento lineare a velocità ±20%
  - Flip laterale: specchio sinistra/destra + swap left/right pitch
  - Scale: fattore 0.85–1.15 su tutti i keypoint
  - Shift orizzontale: traslazione ±0.1 della larghezza normalizzata

Output: data/generated/synthetic_trials.npz con:
  - sequences: (N, T, 34)  float32
  - targets:   (N, T, 2)   float32  [left_pitch, right_pitch]
  - lengths:   (N,)         int32   (frame validi prima del padding)
  - is_real:   (N,)         bool    (True = trial reale)

Usage
-----
    python scripts/generate_synthetic_trials.py
    python scripts/generate_synthetic_trials.py --n-synthetic 200 --seq-len 128
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ── costanti ──────────────────────────────────────────────────────────────────
KP_X_COLS = [f"kp_{i:02d}_x_px" for i in range(17)]
KP_Y_COLS = [f"kp_{i:02d}_y_px" for i in range(17)]
PITCH_COLS = ["left_sensor_pitch_deg", "right_sensor_pitch_deg"]

# Indici COCO per flip laterale: mappa ogni keypoint al suo speculare
FLIP_PAIRS = [
    (1, 2), (3, 4),           # occhi, orecchie
    (5, 6), (7, 8), (9, 10),  # spalle, gomiti, polsi
    (11, 12), (13, 14), (15, 16),  # anche, ginocchia, caviglie
]


# ── caricamento trial reali ───────────────────────────────────────────────────

def load_real_trial(trial_dir: Path) -> dict | None:
    """Carica un trial e restituisce keypoint normalizzati + pitch delta."""
    ts_path = trial_dir / "movement_timeseries.csv"
    if not ts_path.exists():
        return None

    ts = pd.read_csv(ts_path)

    # Verifica colonne necessarie
    missing = [c for c in KP_X_COLS + KP_Y_COLS + PITCH_COLS + ["body_height_px"]
               if c not in ts.columns]
    if missing:
        print(f"  [skip] {trial_dir.name}: colonne mancanti {missing[:3]}")
        return None

    if ts[PITCH_COLS].isna().all().any():
        print(f"  [skip] {trial_dir.name}: pitch IMU mancante")
        return None

    T = len(ts)
    height = ts["body_height_px"].median()
    if height < 10:
        print(f"  [skip] {trial_dir.name}: body_height_px troppo piccola")
        return None

    # Keypoint normalizzati per altezza corpo: shape (T, 34)
    kp_x = ts[KP_X_COLS].to_numpy(dtype=np.float32) / height
    kp_y = ts[KP_Y_COLS].to_numpy(dtype=np.float32) / height
    seq = np.concatenate([kp_x, kp_y], axis=1)  # (T, 34)

    # Pitch IMU: sottrai primo frame (baseline)
    pitch = ts[PITCH_COLS].to_numpy(dtype=np.float32)
    pitch_delta = pitch - pitch[0]  # (T, 2)

    return {"seq": seq, "target": pitch_delta, "name": trial_dir.name}


# ── augmentazioni ─────────────────────────────────────────────────────────────

def augment(trial: dict, rng: np.random.Generator, seq_len: int) -> dict:
    """Applica augmentazioni casuali a un trial e restituisce un nuovo trial."""
    seq    = trial["seq"].copy()    # (T, 34)
    target = trial["target"].copy() # (T, 2)
    T = len(seq)

    # 1. Time warp: ricampiona a velocità diversa, poi porta a T frame
    warp = rng.uniform(0.80, 1.20)
    new_T = max(20, int(T * warp))
    old_idx = np.linspace(0, T - 1, new_T)
    seq    = np.array([np.interp(old_idx, np.arange(T), seq[:, d])    for d in range(34)]).T
    target = np.array([np.interp(old_idx, np.arange(T), target[:, d]) for d in range(2)]).T

    # 2. Scale: tutti i keypoint scalati di un fattore uniforme
    scale = rng.uniform(0.85, 1.15)
    seq = seq * scale

    # 3. Shift orizzontale: sposta x di tutte le colonne
    shift = rng.uniform(-0.10, 0.10)
    seq[:, :17] += shift   # prime 17 col = x

    # 4. Flip laterale (50% probabilità): specchio sx/dx
    if rng.random() < 0.5:
        # Inverti x: rifletti rispetto al centro della distribuzione x
        x_center = seq[:, :17].mean()
        seq[:, :17] = 2 * x_center - seq[:, :17]
        # Swappa keypoint simmetrici
        for l, r in FLIP_PAIRS:
            seq[:, [l, r]]      = seq[:, [r, l]]       # x
            seq[:, [17+l, 17+r]] = seq[:, [17+r, 17+l]] # y
        # Swappa left/right pitch
        target = target[:, [1, 0]]

    # 5. Gaussian noise
    sigma = rng.uniform(0.005, 0.015)
    seq = seq + rng.normal(0, sigma, size=seq.shape).astype(np.float32)

    return {"seq": seq.astype(np.float32), "target": target.astype(np.float32)}


# ── padding a lunghezza fissa ─────────────────────────────────────────────────

def pad_or_truncate(seq: np.ndarray, target: np.ndarray, seq_len: int):
    """Porta sequenze a lunghezza fissa con padding a zero."""
    T = len(seq)
    length = min(T, seq_len)

    seq_out    = np.zeros((seq_len, seq.shape[1]),    dtype=np.float32)
    target_out = np.zeros((seq_len, target.shape[1]), dtype=np.float32)

    seq_out[:length]    = seq[:length]
    target_out[:length] = target[:length]

    return seq_out, target_out, length


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Genera trial sintetici per il Transformer pitch.")
    parser.add_argument("--trials-dir",   default="collected_trials")
    parser.add_argument("--output",       default="data/generated/synthetic_trials.npz")
    parser.add_argument("--mocap-sequences", default="data/generated/mocap_sequences.npz",
                        help="Percorso a mocap_sequences.npz (generato da generate_mocap_sequences.py). "
                             "Se fornito, i 183 atleti vengono aggiunti al training set.")
    parser.add_argument("--n-synthetic",  type=int, default=200,
                        help="Numero di trial sintetici dai 20 trial reali.")
    parser.add_argument("--seq-len",      type=int, default=128,
                        help="Lunghezza fissa delle sequenze dopo padding.")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    trials_dir = Path(args.trials_dir)

    # ── Trial reali (20 salti personali) ─────────────────────────────────────
    real_trials = []
    for td in sorted([d for d in trials_dir.iterdir() if d.is_dir()]):
        t = load_real_trial(td)
        if t is not None:
            real_trials.append(t)
            print(f"  Caricato: {td.name}  ({len(t['seq'])} frame)")

    n_real = len(real_trials)
    print(f"\nTrial reali caricati: {n_real}")
    if n_real == 0:
        print("Nessun trial valido.")
        return

    # ── Sintetici da trial reali ──────────────────────────────────────────────
    aug_per_trial = args.n_synthetic // n_real + 1
    synthetic = []
    for trial in real_trials:
        for _ in range(aug_per_trial):
            if len(synthetic) >= args.n_synthetic:
                break
            synthetic.append(augment(trial, rng, args.seq_len))
        if len(synthetic) >= args.n_synthetic:
            break
    synthetic = synthetic[:args.n_synthetic]
    print(f"Trial sintetici (da real) generati: {len(synthetic)}")

    # ── Sequenze mocap 183 atleti (opzionale) ─────────────────────────────────
    mocap_seqs, mocap_tgts, mocap_lens = None, None, None
    if args.mocap_sequences:
        mpath = Path(args.mocap_sequences)
        if mpath.exists():
            data = np.load(mpath)
            mocap_seqs = data["sequences"]   # (M, seq_len, 34)
            mocap_tgts = data["targets"]     # (M, seq_len, 2)
            mocap_lens = data["lengths"]     # (M,)
            print(f"Sequenze mocap caricate: {len(mocap_seqs)} atleti da {mpath}")
        else:
            print(f"[warn] --mocap-sequences non trovato: {mpath}. Ignorato.")

    # ── Assembla dataset ──────────────────────────────────────────────────────
    # Parte 1: trial reali (is_real=True, usati solo per validation)
    # Parte 2: sintetici da real (is_real=False, training)
    # Parte 3: mocap atleti (is_real=False, training)
    all_trials = [(t, True) for t in real_trials] + [(t, False) for t in synthetic]
    N_base = len(all_trials)
    N_mocap = len(mocap_seqs) if mocap_seqs is not None else 0
    N = N_base + N_mocap

    sequences = np.zeros((N, args.seq_len, 34),  dtype=np.float32)
    targets   = np.zeros((N, args.seq_len, 2),   dtype=np.float32)
    lengths   = np.zeros(N,                       dtype=np.int32)
    is_real   = np.zeros(N,                       dtype=bool)

    for i, (t, real) in enumerate(all_trials):
        s, tgt, L = pad_or_truncate(t["seq"], t["target"], args.seq_len)
        sequences[i] = s
        targets[i]   = tgt
        lengths[i]   = L
        is_real[i]   = real

    if mocap_seqs is not None:
        # Riadatta seq_len se necessario
        ml = mocap_seqs.shape[1]
        sl = args.seq_len
        for j, idx in enumerate(range(N_base, N)):
            src_s = mocap_seqs[j]  # (ml, 34)
            src_t = mocap_tgts[j]  # (ml, 2)
            L     = int(mocap_lens[j])
            # Tronca o zero-pad a sl
            s_out = np.zeros((sl, 34),  dtype=np.float32)
            t_out = np.zeros((sl, 2),   dtype=np.float32)
            clip  = min(L, sl, ml)
            s_out[:clip] = src_s[:clip]
            t_out[:clip] = src_t[:clip]
            sequences[idx] = s_out
            targets[idx]   = t_out
            lengths[idx]   = clip
            is_real[idx]   = False

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        sequences=sequences,
        targets=targets,
        lengths=lengths,
        is_real=is_real,
    )
    print(f"\nSalvato: {output_path}")
    print(f"  Shape sequences: {sequences.shape}")
    print(f"  Trial reali (val): {is_real.sum()}")
    print(f"  Training (sintetici da real): {len(synthetic)}")
    print(f"  Training (mocap atleti):      {N_mocap}")
    print(f"  Training totale:              {(~is_real).sum()}")
    print(f"  Pitch delta range (reali): "
          f"{targets[is_real].min():.1f}° → {targets[is_real].max():.1f}°")


if __name__ == "__main__":
    main()
