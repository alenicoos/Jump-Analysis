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
    Il pitch di un lato viene amplificato e l'altro ridotto.

shallow_landing
    La flessione è troppo superficiale — il soggetto atterra rigido.
    Sia i keypoint che il pitch vengono compressi verticalmente.
    Alto rischio di infortuni da impatto.

trunk_lateral_lean
    Il tronco si inclina lateralmente durante l'atterraggio.
    Le anche e le spalle si spostano su un lato.

wide_stance / narrow_stance
    I piedi sono troppo distanti o troppo vicini durante il salto.
    Altera tutto il pattern di ginocchio e caviglia.

asymmetric_pitch_only
    Solo il pitch è asimmetrico (un lato molto più flesso dell'altro)
    senza che i keypoint cambino visibilmente — testa il contributo
    del pitch al modello.

Output: data/generated/anomalous_trials.npz con:
  - sequences:    (N, seq_len, 34)  float32
  - pitch_gt:     (N, seq_len, 2)   float32  (pitch ground truth mocap)
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
from pathlib import Path

import numpy as np

# Indici COCO keypoint nelle colonne (prime 17 = x, seconde 17 = y)
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW       = 7, 8
L_WRIST, R_WRIST       = 9, 10
L_HIP, R_HIP           = 11, 12
L_KNEE, R_KNEE         = 13, 14
L_ANKLE, R_ANKLE       = 15, 16


def get_x(seq: np.ndarray, kp: int) -> np.ndarray:
    """Colonna x di un keypoint in tutta la sequenza (T,)."""
    return seq[:, kp]

def get_y(seq: np.ndarray, kp: int) -> np.ndarray:
    return seq[:, 17 + kp]

def set_x(seq: np.ndarray, kp: int, val: np.ndarray) -> None:
    seq[:, kp] = val

def set_y(seq: np.ndarray, kp: int, val: np.ndarray) -> None:
    seq[:, 17 + kp] = val


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
        else:
            w[t] = 0.0
    return w


def ramp_from(T: int, start: int, ramp_in: int = 5) -> np.ndarray:
    """Finestra 0→1 che sale a start e rimane a 1 fino alla fine."""
    w = np.zeros(T, dtype=np.float32)
    for t in range(T):
        if t < start:
            w[t] = 0.0
        elif t < start + ramp_in:
            w[t] = (t - start) / max(ramp_in, 1)
        else:
            w[t] = 1.0
    return w


# ── Anomalie ──────────────────────────────────────────────────────────────────

def apply_knee_valgus(
    seq: np.ndarray,
    pitch: np.ndarray,
    ic: int,
    kfmax: int,
    takeoff: int,
    rng: np.random.Generator,
    severity: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Ginocchia che collassano verso l'interno (valgismo dinamico).

    Timing: ic → kfmax (picco), poi si riduce lentamente fino a takeoff.
    Non presente durante la discesa pre-ic.
    """
    seq, pitch = seq.copy(), pitch.copy()
    T = len(seq)
    if severity is None:
        severity = rng.uniform(0.02, 0.05)

    # Picco a kfmax, riduzione lenta verso takeoff
    w_flex = smooth_ramp(T, ic, kfmax, ramp_in=5, ramp_out=3)
    w_ext  = smooth_ramp(T, kfmax, takeoff, ramp_in=3, ramp_out=8) * 0.4  # 40% persiste in estensione
    w = np.clip(w_flex + w_ext, 0, 1)

    shift = severity * w
    set_x(seq, L_KNEE, get_x(seq, L_KNEE) + shift)
    set_x(seq, R_KNEE, get_x(seq, R_KNEE) - shift)
    set_x(seq, L_HIP,  get_x(seq, L_HIP)  + shift * 0.2)
    set_x(seq, R_HIP,  get_x(seq, R_HIP)  - shift * 0.2)

    return seq, pitch


def apply_asymmetric_flexion(
    seq: np.ndarray,
    pitch: np.ndarray,
    ic: int,
    kfmax: int,
    takeoff: int,
    rng: np.random.Generator,
    severity: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Flessione asimmetrica: un lato si flette più dell'altro.

    Timing: ic → kfmax. Caratteristica della fase eccentrica.
    """
    seq, pitch = seq.copy(), pitch.copy()
    T = len(seq)
    if severity is None:
        heavy = rng.choice(["left", "right"])
        amp_heavy = rng.uniform(1.2, 1.4)
        amp_light = rng.uniform(0.6, 0.8)
    else:
        heavy = "left"
        amp_heavy, amp_light = 1.0 + severity, 1.0 - severity * 0.5

    w = smooth_ramp(T, ic, kfmax)

    if heavy == "left":
        pitch[:, 0] *= (1 + (amp_heavy - 1) * w)
        pitch[:, 1] *= (1 - (1 - amp_light) * w)
        set_y(seq, L_KNEE, get_y(seq, L_KNEE) + 0.015 * w)
    else:
        pitch[:, 1] *= (1 + (amp_heavy - 1) * w)
        pitch[:, 0] *= (1 - (1 - amp_light) * w)
        set_y(seq, R_KNEE, get_y(seq, R_KNEE) + 0.015 * w)

    return seq, pitch


def apply_shallow_landing(
    seq: np.ndarray,
    pitch: np.ndarray,
    ic: int,
    kfmax: int,
    takeoff: int,
    rng: np.random.Generator,
    severity: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Landing rigido: flessione ridotta durante la fase eccentrica.

    Timing: ic → kfmax. La discesa pre-ic non cambia.
    """
    seq, pitch = seq.copy(), pitch.copy()
    T = len(seq)
    if severity is None:
        scale = rng.uniform(0.45, 0.70)
    else:
        scale = max(0.1, 1.0 - severity)

    w = smooth_ramp(T, ic, kfmax)

    pitch[:, 0] *= (1 - (1 - scale) * w)
    pitch[:, 1] *= (1 - (1 - scale) * w)

    for kp in [L_HIP, R_HIP, L_KNEE, R_KNEE]:
        y_orig = get_y(seq, kp).copy()
        y_ic   = y_orig[ic]
        delta  = y_orig - y_ic
        set_y(seq, kp, y_ic + delta * (1 - (1 - scale) * w))

    return seq, pitch


def apply_trunk_lateral_lean(
    seq: np.ndarray,
    pitch: np.ndarray,
    ic: int,
    kfmax: int,
    takeoff: int,
    rng: np.random.Generator,
    severity: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Trunk laterale: il tronco si inclina su un lato durante l'atterraggio.

    Timing: ic → takeoff (il tronco rimane inclinato per tutta la fase di contatto).
    Non presente durante la discesa pre-ic.
    """
    seq, pitch = seq.copy(), pitch.copy()
    T = len(seq)
    if severity is None:
        direction = rng.choice([-1, 1])
        amount = rng.uniform(0.02, 0.05) * direction
    else:
        amount = severity

    # Sale a ic, rimane durante tutto il contatto, scende verso takeoff
    w = smooth_ramp(T, ic, takeoff, ramp_in=5, ramp_out=8)
    shift = amount * w

    for kp in [L_SHOULDER, R_SHOULDER, NOSE]:
        set_x(seq, kp, get_x(seq, kp) + shift)
    for kp in [L_HIP, R_HIP]:
        set_x(seq, kp, get_x(seq, kp) + shift * 0.5)

    return seq, pitch


def apply_wide_stance(
    seq: np.ndarray,
    pitch: np.ndarray,
    ic: int,
    kfmax: int,
    takeoff: int,
    rng: np.random.Generator,
    severity: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Piedi troppo larghi all'atterraggio.

    Timing: sale a ic e rimane fino alla fine. Durante la discesa pre-ic
    i piedi sono in aria, quindi non è applicabile.
    """
    seq, pitch = seq.copy(), pitch.copy()
    T = len(seq)
    if severity is None:
        extra = rng.uniform(0.03, 0.07)
    else:
        extra = severity

    w = ramp_from(T, ic, ramp_in=3)   # entra a ic, rimane costante
    for kp in [L_ANKLE, L_KNEE]:
        set_x(seq, kp, get_x(seq, kp) - extra * w)
    for kp in [R_ANKLE, R_KNEE]:
        set_x(seq, kp, get_x(seq, kp) + extra * w)

    return seq, pitch


def apply_narrow_stance(
    seq: np.ndarray,
    pitch: np.ndarray,
    ic: int,
    kfmax: int,
    takeoff: int,
    rng: np.random.Generator,
    severity: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Piedi troppo vicini all'atterraggio."""
    seq, pitch = seq.copy(), pitch.copy()
    if severity is None:
        severity = rng.uniform(0.03, 0.06)
    return apply_wide_stance(seq, pitch, ic, kfmax, takeoff, rng, severity=-severity)


def apply_asymmetric_pitch_only(
    seq: np.ndarray,
    pitch: np.ndarray,
    ic: int,
    kfmax: int,
    takeoff: int,
    rng: np.random.Generator,
    severity: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solo il pitch è asimmetrico, i keypoint restano normali.

    Timing: ic → kfmax. Testa il contributo del pitch al modello.
    """
    seq, pitch = seq.copy(), pitch.copy()
    T = len(seq)
    w = smooth_ramp(T, ic, kfmax)

    amp = rng.uniform(1.2, 1.5) if severity is None else 1.0 + severity
    side = rng.choice([0, 1])
    pitch[:, side]     *= (1 + (amp - 1) * w)
    pitch[:, 1 - side] *= (1 - 0.3 * w)

    return seq, pitch


ANOMALY_FUNCTIONS = {
    "knee_valgus":           apply_knee_valgus,
    "asymmetric_flexion":    apply_asymmetric_flexion,
    "shallow_landing":       apply_shallow_landing,
    "trunk_lateral_lean":    apply_trunk_lateral_lean,
    "wide_stance":           apply_wide_stance,
    "narrow_stance":         apply_narrow_stance,
    "asymmetric_pitch_only": apply_asymmetric_pitch_only,
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Genera trial anomali sintetici dal dataset mocap.")
    parser.add_argument("--input",  default="data/generated/mocap_sequences.npz",
                        help="File mocap_sequences.npz generato da generate_mocap_sequences.py.")
    parser.add_argument("--output", default="data/generated/anomalous_trials.npz")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    rng  = np.random.default_rng(args.seed)
    data = np.load(args.input)

    seqs         = data["sequences"]      # (N, seq_len, 34)
    pitch_gt     = data["targets"]        # (N, seq_len, 2)
    lengths      = data["lengths"]        # (N,)
    subject_ids  = data["subject_ids"]    # (N,)
    ic_frames    = data["ic_frames"]      # (N,)
    kfmax_frames = data["kfmax_frames"]   # (N,)
    takeoff_frames = data["takeoff_frames"]  # (N,)

    N_src   = len(seqs)
    n_types = len(ANOMALY_FUNCTIONS)
    N_out   = N_src * n_types

    seq_len = seqs.shape[1]

    out_sequences  = np.zeros((N_out, seq_len, 34), dtype=np.float32)
    out_pitch      = np.zeros((N_out, seq_len, 2),  dtype=np.float32)
    out_lengths    = np.zeros(N_out, dtype=np.int32)
    out_types      = []
    out_source_ids = []
    out_ic         = np.zeros(N_out, dtype=np.int32)
    out_kfmax      = np.zeros(N_out, dtype=np.int32)

    idx = 0
    for i in range(N_src):
        seq     = seqs[i]
        p       = pitch_gt[i]
        L       = int(lengths[i])
        ic      = int(ic_frames[i])
        kf      = int(kfmax_frames[i])
        takeoff = min(int(takeoff_frames[i]), L - 1)
        sid     = str(subject_ids[i])

        seq_v = seq[:L].copy()
        p_v   = p[:L].copy()

        for atype, afunc in ANOMALY_FUNCTIONS.items():
            s_anom, p_anom = afunc(seq_v, p_v, ic, kf, takeoff, rng)

            # Ripadda a seq_len
            out_sequences[idx, :L] = s_anom[:L]
            out_pitch[idx, :L]     = p_anom[:L]
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
        pitch_gt=out_pitch,
        lengths=out_lengths,
        anomaly_type=np.array(out_types),
        source_id=np.array(out_source_ids),
        ic_frames=out_ic,
        kfmax_frames=out_kfmax,
    )

    print(f"Salvato: {output_path}")
    print(f"  Sorgenti normali:    {N_src}")
    print(f"  Tipi di anomalia:    {n_types}  ({', '.join(ANOMALY_FUNCTIONS.keys())})")
    print(f"  Trial anomali totali:{N_out}")
    for atype in ANOMALY_FUNCTIONS:
        n = sum(1 for t in out_types if t == atype)
        print(f"    {atype}: {n}")


if __name__ == "__main__":
    main()
