from __future__ import annotations

"""Genera trial anomali sintetici partendo dalle sequenze mocap normali.

Ogni tipo di anomalia è biomeccanicamente plausibile — cose che accadono
davvero in atleti con deficit di controllo motorio — e abbastanza marcata
da essere rilevabile, ma non irrealizzabile.

Tipi di anomalia implementati
------------------------------
knee_valgus
    Le ginocchia collassano verso l'interno durante la fase di flessione.
    Pattern tipico a rischio ACL. Le x dei keypoint ginocchio si spostano
    verso il centro nella finestra ic→kfmax.

asymmetric_flexion
    Un lato si flette molto più dell'altro (ratio 1:2 circa).
    Riflette asimmetria di forza o dolore unilaterale.
    Il keypoint ginocchio di un lato scende di più.

shallow_landing
    La flessione è troppo superficiale — il soggetto atterra rigido.
    I keypoint anca/ginocchio vengono compressi verticalmente.
    Alto rischio di infortuni da impatto.

trunk_lateral_lean
    Il tronco si inclina lateralmente durante l'atterraggio.
    Le anche e le spalle si spostano su un lato.

wide_stance / narrow_stance
    I piedi sono troppo distanti o troppo vicini durante il salto.
    Altera tutto il pattern di ginocchio e caviglia.

Output: data/generated/anomalous_trials.npz con:
  - sequences:    (N, seq_len, 24)  float32
  - lengths:      (N,)              int32
  - anomaly_type: (N,)              str array
  - source_id:    (N,)              str array  (subject_id di origine)
  - ic_frames:    (N,)              int32
  - kfmax_frames: (N,)              int32

Usage
-----
    python scripts/generate_anomalous_trials.py
    python scripts/generate_anomalous_trials.py --input data/generated/mocap_sequences.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jump_analysis.features.front_2d_features import BODY_MODEL_KEYPOINTS, select_temporal_features

# Indici COCO keypoint.
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP           = 11, 12
L_KNEE, R_KNEE         = 13, 14
L_ANKLE, R_ANKLE       = 15, 16


def keypoint_column(seq: np.ndarray, kp: int) -> int | None:
    """Return the x-column for a COCO keypoint in full or body-only sequences."""
    n_keypoints = seq.shape[1] // 2
    if n_keypoints == 17:
        return kp
    if n_keypoints == len(BODY_MODEL_KEYPOINTS) and kp in BODY_MODEL_KEYPOINTS:
        return BODY_MODEL_KEYPOINTS.index(kp)
    return None


def get_x(seq: np.ndarray, kp: int) -> np.ndarray:
    col = keypoint_column(seq, kp)
    return seq[:, col] if col is not None else np.zeros(len(seq), dtype=seq.dtype)

def get_y(seq: np.ndarray, kp: int) -> np.ndarray:
    col = keypoint_column(seq, kp)
    return seq[:, seq.shape[1] // 2 + col] if col is not None else np.zeros(len(seq), dtype=seq.dtype)

def set_x(seq: np.ndarray, kp: int, val: np.ndarray) -> None:
    col = keypoint_column(seq, kp)
    if col is not None:
        seq[:, col] = val

def set_y(seq: np.ndarray, kp: int, val: np.ndarray) -> None:
    col = keypoint_column(seq, kp)
    if col is not None:
        seq[:, seq.shape[1] // 2 + col] = val


def smooth_ramp(T: int, start: int, end: int, ramp_in: int = 5, ramp_out: int = 5) -> np.ndarray:
    """Finestra 0→1→0 con rampe smooth tra start e end."""
    w = np.zeros(T, dtype=np.float32)
    for t in range(T):
        if t < start:
            w[t] = 0.0
        elif t < start + ramp_in:
            w[t] = (t - start) / max(ramp_in, 1)
        elif t <= end - ramp_out:
            w[t] = 1.0
        elif t <= end:
            w[t] = (end - t) / max(ramp_out, 1)
    return w


def ramp_from(T: int, start: int, ramp_in: int = 5) -> np.ndarray:
    """Finestra 0→1 che sale a start e rimane a 1 fino alla fine."""
    w = np.zeros(T, dtype=np.float32)
    for t in range(T):
        if t >= start:
            w[t] = min(1.0, (t - start) / max(ramp_in, 1))
    return w


# ── Anomalie ──────────────────────────────────────────────────────────────────

def apply_knee_valgus(
    seq: np.ndarray, ic: int, kfmax: int, takeoff: int,
    rng: np.random.Generator, severity: float | None = None,
) -> np.ndarray:
    """Ginocchia che collassano verso l'interno (valgismo dinamico)."""
    seq = seq.copy()
    T = len(seq)
    if severity is None:
        severity = rng.uniform(0.02, 0.05)
    w_flex = smooth_ramp(T, ic, kfmax, ramp_in=5, ramp_out=3)
    w_ext  = smooth_ramp(T, kfmax, takeoff, ramp_in=3, ramp_out=8) * 0.4
    w = np.clip(w_flex + w_ext, 0, 1)
    shift = severity * w
    set_x(seq, L_KNEE, get_x(seq, L_KNEE) + shift)
    set_x(seq, R_KNEE, get_x(seq, R_KNEE) - shift)
    set_x(seq, L_HIP,  get_x(seq, L_HIP)  + shift * 0.2)
    set_x(seq, R_HIP,  get_x(seq, R_HIP)  - shift * 0.2)
    return seq


def apply_asymmetric_flexion(
    seq: np.ndarray, ic: int, kfmax: int, takeoff: int,
    rng: np.random.Generator, severity: float | None = None,
) -> np.ndarray:
    """Flessione asimmetrica: un lato si flette visibilmente più dell'altro."""
    seq = seq.copy()
    T = len(seq)
    if severity is None:
        heavy = rng.choice(["left", "right"])
        # Shift più ampio: 0.08-0.15 unità normalizzate per produrre
        # una variazione angolare chiaramente rilevabile (~5-15°)
        drop  = rng.uniform(0.08, 0.15)
    else:
        heavy = "left"
        drop  = severity
    w = smooth_ramp(T, ic, kfmax)
    # Sposta sia il ginocchio che l'anca del lato pesante per amplificare
    # la variazione dell'angolo di flessione
    kp_knee = L_KNEE if heavy == "left" else R_KNEE
    kp_hip  = L_HIP  if heavy == "left" else R_HIP
    set_y(seq, kp_knee, get_y(seq, kp_knee) + drop * w)
    set_y(seq, kp_hip,  get_y(seq, kp_hip)  + drop * 0.4 * w)
    return seq


def apply_shallow_landing(
    seq: np.ndarray, ic: int, kfmax: int, takeoff: int,
    rng: np.random.Generator, severity: float | None = None,
) -> np.ndarray:
    """Landing rigido: flessione ridotta durante la fase eccentrica."""
    seq = seq.copy()
    T = len(seq)
    if severity is None:
        scale = rng.uniform(0.25, 0.55)   # più aggressivo: riduce la flessione di 45-75%
    else:
        scale = max(0.1, 1.0 - severity)
    w = smooth_ramp(T, ic, kfmax)
    for kp in [L_HIP, R_HIP, L_KNEE, R_KNEE]:
        y_orig = get_y(seq, kp).copy()
        y_ic   = y_orig[ic]
        delta  = y_orig - y_ic
        set_y(seq, kp, y_ic + delta * (1 - (1 - scale) * w))
    return seq


def apply_trunk_lateral_lean(
    seq: np.ndarray, ic: int, kfmax: int, takeoff: int,
    rng: np.random.Generator, severity: float | None = None,
) -> np.ndarray:
    """Trunk laterale: il tronco si inclina su un lato durante l'atterraggio."""
    seq = seq.copy()
    T = len(seq)
    if severity is None:
        direction = rng.choice([-1, 1])
        # Aumentato da 0.02-0.05 a 0.06-0.12 per produrre variazione rilevabile
        amount = rng.uniform(0.06, 0.12) * direction
    else:
        amount = severity
    w = smooth_ramp(T, ic, takeoff, ramp_in=5, ramp_out=8)
    shift = amount * w
    # Le spalle si spostano più delle anche (inclinazione del tronco)
    for kp in [L_SHOULDER, R_SHOULDER, NOSE]:
        set_x(seq, kp, get_x(seq, kp) + shift)
    for kp in [L_HIP, R_HIP]:
        set_x(seq, kp, get_x(seq, kp) + shift * 0.3)
    return seq


def apply_wide_stance(
    seq: np.ndarray, ic: int, kfmax: int, takeoff: int,
    rng: np.random.Generator, severity: float | None = None,
) -> np.ndarray:
    """Piedi troppo larghi all'atterraggio."""
    seq = seq.copy()
    T = len(seq)
    if severity is None:
        extra = rng.uniform(0.03, 0.07)
    else:
        extra = severity
    w = ramp_from(T, ic, ramp_in=3)
    for kp in [L_ANKLE, L_KNEE]:
        set_x(seq, kp, get_x(seq, kp) - extra * w)
    for kp in [R_ANKLE, R_KNEE]:
        set_x(seq, kp, get_x(seq, kp) + extra * w)
    return seq


def apply_narrow_stance(
    seq: np.ndarray, ic: int, kfmax: int, takeoff: int,
    rng: np.random.Generator, severity: float | None = None,
) -> np.ndarray:
    """Piedi troppo vicini all'atterraggio."""
    if severity is None:
        severity = rng.uniform(0.03, 0.06)
    return apply_wide_stance(seq, ic, kfmax, takeoff, rng, severity=-severity)


ANOMALY_FUNCTIONS = {
    "knee_valgus":        apply_knee_valgus,
    "asymmetric_flexion": apply_asymmetric_flexion,
    "shallow_landing":    apply_shallow_landing,
    "trunk_lateral_lean": apply_trunk_lateral_lean,
    "wide_stance":        apply_wide_stance,
    "narrow_stance":      apply_narrow_stance,
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Genera trial anomali sintetici dal dataset mocap.")
    parser.add_argument("--input",  default="data/generated/mocap_sequences.npz")
    parser.add_argument("--output", default="data/generated/anomalous_trials.npz")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    rng  = np.random.default_rng(args.seed)
    data = np.load(args.input)

    seqs         = data["sequences"]       # (N, seq_len, 24) or legacy (N, seq_len, 34)
    if seqs.shape[-1] == 34:
        seqs = select_temporal_features(seqs, include_head=False)
    lengths      = data["lengths"]         # (N,)
    subject_ids  = data["subject_ids"]     # (N,)
    ic_frames    = data["ic_frames"]       # (N,)
    kfmax_frames = data["kfmax_frames"]    # (N,)
    takeoff_frames = data["takeoff_frames"] # (N,)

    N_src   = len(seqs)
    n_types = len(ANOMALY_FUNCTIONS)
    N_out   = N_src * n_types
    seq_len = seqs.shape[1]

    out_sequences  = np.zeros((N_out, seq_len, seqs.shape[-1]), dtype=np.float32)
    out_lengths    = np.zeros(N_out, dtype=np.int32)
    out_types      = []
    out_source_ids = []
    out_ic         = np.zeros(N_out, dtype=np.int32)
    out_kfmax      = np.zeros(N_out, dtype=np.int32)

    idx = 0
    for i in range(N_src):
        seq     = seqs[i]
        L       = int(lengths[i])
        ic      = int(ic_frames[i])
        kf      = int(kfmax_frames[i])
        takeoff = min(int(takeoff_frames[i]), L - 1)
        sid     = str(subject_ids[i])
        seq_v   = seq[:L].copy()

        for atype, afunc in ANOMALY_FUNCTIONS.items():
            s_anom = afunc(seq_v, ic, kf, takeoff, rng)
            out_sequences[idx, :L] = s_anom[:L]
            out_lengths[idx]       = L
            out_types.append(atype)
            out_source_ids.append(sid)
            out_ic[idx]    = ic
            out_kfmax[idx] = kf
            idx += 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        sequences=out_sequences,
        lengths=out_lengths,
        anomaly_type=np.array(out_types),
        source_id=np.array(out_source_ids),
        ic_frames=out_ic,
        kfmax_frames=out_kfmax,
    )
    print(f"Salvato: {output_path}")
    print(f"  Sorgenti normali:     {N_src}")
    print(f"  Tipi di anomalia:     {n_types}  ({', '.join(ANOMALY_FUNCTIONS.keys())})")
    print(f"  Trial anomali totali: {N_out}")
    for atype in ANOMALY_FUNCTIONS:
        print(f"    {atype}: {sum(1 for t in out_types if t == atype)}")


if __name__ == "__main__":
    main()
