from __future__ import annotations

"""Estrae sequenze temporali 2D + knee pitch dai file .mat dei 183 atleti mocap.

Per ogni atleta legge la traiettoria completa dal frame ic al frame kfmax,
calcola i 34 keypoint 2D (proiezione frontale, normalizzati per altezza corpo)
e il pitch sagittale dello stinco in gradi come target.

Il pitch è calcolato come angolo del segmento tibia nel piano sagittale (X-Z)
rispetto alla verticale, espresso come delta dal frame ic (baseline).

Output: data/generated/mocap_sequences.npz con:
  - sequences: (N, seq_len, 34)  float32   — keypoint 2D normalizzati
  - targets:   (N, seq_len, 2)   float32   — [left_pitch_delta, right_pitch_delta] in gradi
  - lengths:   (N,)              int32     — frame validi prima del padding
  - subject_ids: (N,)            str array — ID soggetto

Usage
-----
    python scripts/generate_mocap_sequences.py
    python scripts/generate_mocap_sequences.py --root /Users/ale/Kinematic_Data --seq-len 128
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ── Indici COCO keypoint ──────────────────────────────────────────────────────
NOSE           = 0
LEFT_EYE       = 1
RIGHT_EYE      = 2
LEFT_EAR       = 3
RIGHT_EAR      = 4
LEFT_SHOULDER  = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW     = 7
RIGHT_ELBOW    = 8
LEFT_WRIST     = 9
RIGHT_WRIST    = 10
LEFT_HIP       = 11
RIGHT_HIP      = 12
LEFT_KNEE      = 13
RIGHT_KNEE     = 14
LEFT_ANKLE     = 15
RIGHT_ANKLE    = 16

DROP_JUMP_NAME = "drop jump"


# ── Utilità mat ───────────────────────────────────────────────────────────────

def load_mat(path: Path) -> dict:
    return loadmat(path, simplify_cells=True)


def xyz(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, :3]


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def find_drop_jump_index(jc: dict) -> int:
    for i, name in enumerate(jc["FILE_NAME"]):
        if DROP_JUMP_NAME in str(name).lower():
            return i
    raise ValueError("Drop jump task not found")


def matlab_to_idx(frame) -> int:
    return max(0, int(round(float(frame))) - 1)


# ── Proiezione 2D e keypoint ──────────────────────────────────────────────────

def project_front(p: np.ndarray) -> np.ndarray:
    """Proiezione frontale: Y=mediale-laterale (→ x screen), Z=verticale (→ y screen)."""
    return np.array([p[1], p[2]], dtype=float)


def get_marker(jc: dict, field: str, task_idx: int, frame: int) -> np.ndarray:
    return xyz(jc[field][task_idx])[frame]


def build_keypoints_frame(jc: dict, task_idx: int, frame: int) -> np.ndarray:
    """Costruisce array (17, 2) di keypoint COCO in proiezione frontale."""
    kp = np.zeros((17, 2), dtype=float)

    left_hip      = project_front(get_marker(jc, "LTHP", task_idx, frame))
    right_hip     = project_front(get_marker(jc, "RTHP", task_idx, frame))
    left_knee_3d  = midpoint(get_marker(jc, "LTHD", task_idx, frame),
                              get_marker(jc, "LSHP", task_idx, frame))
    right_knee_3d = midpoint(get_marker(jc, "RTHD", task_idx, frame),
                              get_marker(jc, "RSHP", task_idx, frame))
    left_ankle    = project_front(get_marker(jc, "LSHD", task_idx, frame))
    right_ankle   = project_front(get_marker(jc, "RSHD", task_idx, frame))
    left_shoulder = project_front(get_marker(jc, "LFAP", task_idx, frame))
    right_shoulder= project_front(get_marker(jc, "RFAP", task_idx, frame))
    nose          = project_front(get_marker(jc, "FHEAD", task_idx, frame))

    kp[NOSE]           = nose
    kp[LEFT_SHOULDER]  = left_shoulder
    kp[RIGHT_SHOULDER] = right_shoulder
    kp[LEFT_HIP]       = left_hip
    kp[RIGHT_HIP]      = right_hip
    kp[LEFT_KNEE]      = project_front(left_knee_3d)
    kp[RIGHT_KNEE]     = project_front(right_knee_3d)
    kp[LEFT_ANKLE]     = left_ankle
    kp[RIGHT_ANKLE]    = right_ankle

    # Keypoint mancanti (occhi, orecchie, gomiti, polsi) → interpolati o zero.
    # Occhi/orecchie: stima intorno al naso.
    kp[LEFT_EYE]   = nose + np.array([-0.05, 0.02])
    kp[RIGHT_EYE]  = nose + np.array([ 0.05, 0.02])
    kp[LEFT_EAR]   = nose + np.array([-0.10, 0.00])
    kp[RIGHT_EAR]  = nose + np.array([ 0.10, 0.00])
    # Gomiti: midpoint spalla-polso (approssimato come spalla + offset)
    kp[LEFT_ELBOW]  = left_shoulder  + (left_hip  - left_shoulder)  * 0.3
    kp[RIGHT_ELBOW] = right_shoulder + (right_hip - right_shoulder) * 0.3
    kp[LEFT_WRIST]  = left_hip  + (left_shoulder  - left_hip)  * 0.05
    kp[RIGHT_WRIST] = right_hip + (right_shoulder - right_hip) * 0.05

    return kp


def body_height_px(kp: np.ndarray) -> float:
    """Altezza corpo approssimata: distanza spalla-caviglia media."""
    left  = np.linalg.norm(kp[LEFT_SHOULDER]  - kp[LEFT_ANKLE])
    right = np.linalg.norm(kp[RIGHT_SHOULDER] - kp[RIGHT_ANKLE])
    h = (left + right) / 2.0
    return h if h > 1e-3 else 1.0


# ── Pitch sagittale dello stinco ──────────────────────────────────────────────

def knee_pitch_deg(jc: dict, task_idx: int, frame: int, side: str) -> float:
    """Pitch della tibia nel piano sagittale (X = ant/post, Z = verticale).

    Angolo del segmento caviglia→ginocchio rispetto alla verticale (Z),
    positivo = inclinazione anteriore.
    """
    if side == "left":
        ankle_3d = get_marker(jc, "LSHD", task_idx, frame)
        knee_3d  = midpoint(get_marker(jc, "LTHD", task_idx, frame),
                            get_marker(jc, "LSHP", task_idx, frame))
    else:
        ankle_3d = get_marker(jc, "RSHD", task_idx, frame)
        knee_3d  = midpoint(get_marker(jc, "RTHD", task_idx, frame),
                            get_marker(jc, "RSHP", task_idx, frame))

    dx = knee_3d[0] - ankle_3d[0]   # anterior/posterior
    dz = knee_3d[2] - ankle_3d[2]   # vertical
    return float(np.degrees(np.arctan2(dx, dz)))


# ── Ricampionamento temporale ─────────────────────────────────────────────────

def resample_sequence(seq: np.ndarray, target_frames: int) -> np.ndarray:
    """Ricampiona (T_src, D) a (target_frames, D) con interpolazione lineare."""
    T_src = len(seq)
    if T_src == target_frames:
        return seq
    old_idx = np.linspace(0, T_src - 1, target_frames)
    return np.array([
        np.interp(old_idx, np.arange(T_src), seq[:, d])
        for d in range(seq.shape[1])
    ]).T.astype(seq.dtype)


# ── Estrazione trial ──────────────────────────────────────────────────────────

def extract_trial(
    subject_dir: Path,
    seq_len: int,
    mocap_fps: float = 100.0,
    target_fps: float = 30.0,
    descent_seconds: float = 0.5,
) -> dict | None:
    """Estrae la sequenza completa: discesa → primo atterraggio → kfmax → takeoff.

    La finestra inizia `descent_seconds` prima di ic (la discesa dalla pedana)
    e finisce al frame di takeoff. Viene ricampionata a `target_fps`.
    """
    sid = subject_dir.name
    try:
        jc     = load_mat(subject_dir / f"JC_{sid}.mat")
        ja     = load_mat(subject_dir / f"JA_{sid}.mat")
        events = load_mat(subject_dir / f"startstop_{sid}_DJ_final.mat")
    except FileNotFoundError as e:
        print(f"  [skip] {sid}: file mancante — {e}")
        return None

    try:
        task_idx = find_drop_jump_index(jc)
    except ValueError as e:
        print(f"  [skip] {sid}: {e}")
        return None

    total_frames = int(xyz(jc["LTHP"][task_idx]).shape[0])
    ic      = int(np.clip(matlab_to_idx(events["jumpdown"]), 0, total_frames - 1))
    takeoff = int(np.clip(matlab_to_idx(events["takeoff"]),  ic + 1, total_frames - 1))

    # kfmax tra ic e takeoff
    left_kn   = xyz(ja["L_KN_P"][task_idx])[:, 0]
    right_kn  = xyz(ja["R_KN_P"][task_idx])[:, 0]
    knee_flex = np.nanmean(np.abs(np.vstack([left_kn, right_kn])), axis=0)
    kfmax     = ic + int(np.nanargmax(knee_flex[ic:takeoff]))

    # Finestra: discesa (descent_seconds prima di ic) → takeoff
    pre_ic_frames = int(round(descent_seconds * mocap_fps))
    start = max(0, ic - pre_ic_frames)
    end   = takeoff + 1   # incluso
    if end > total_frames:
        end = total_frames
    T_raw = end - start
    if T_raw < 10:
        print(f"  [skip] {sid}: finestra troppo corta ({T_raw} frame)")
        return None

    # Costruisci sequenza grezza a mocap_fps
    kp_seq    = np.zeros((T_raw, 17, 2), dtype=np.float64)
    pitch_seq = np.zeros((T_raw, 2),     dtype=np.float64)

    for i, frame in enumerate(range(start, end)):
        try:
            kp = build_keypoints_frame(jc, task_idx, frame)
        except Exception:
            kp = np.zeros((17, 2))
        kp_seq[i]       = kp
        pitch_seq[i, 0] = knee_pitch_deg(jc, task_idx, frame, "left")
        pitch_seq[i, 1] = knee_pitch_deg(jc, task_idx, frame, "right")

    # Normalizza keypoint per altezza corpo (mediana sulla sequenza grezza)
    heights = np.array([body_height_px(kp_seq[i]) for i in range(T_raw)])
    height  = float(np.median(heights))
    if height < 1e-3:
        print(f"  [skip] {sid}: altezza corpo nulla")
        return None
    kp_norm = kp_seq / height   # (T_raw, 17, 2)

    # Flatten → (T_raw, 34): prima tutte le x, poi tutte le y
    seq_flat   = np.concatenate([kp_norm[:, :, 0], kp_norm[:, :, 1]], axis=1).astype(np.float32)
    pitch_flat = pitch_seq.astype(np.float32)

    # Pitch delta dal frame ic (ic_local = posizione di ic dentro la finestra)
    ic_local       = ic - start
    pitch_baseline = pitch_flat[ic_local]
    pitch_delta    = pitch_flat - pitch_baseline   # (T_raw, 2)

    # Ricampiona a target_fps: durata_reale * target_fps frame
    duration_s     = T_raw / mocap_fps
    T_resampled    = max(4, int(round(duration_s * target_fps)))
    seq_resampled   = resample_sequence(seq_flat,   T_resampled)   # (T_res, 34)
    pitch_resampled = resample_sequence(pitch_delta, T_resampled)  # (T_res, 2)

    # Indici chiave ricampionati (per riferimento)
    ic_res     = int(round(ic_local / T_raw * T_resampled))
    kfmax_res  = int(round((kfmax - start) / T_raw * T_resampled))
    takeoff_res = T_resampled - 1

    return {
        "seq":        seq_resampled,
        "target":     pitch_resampled,
        "subject_id": sid,
        "T":          T_resampled,
        "ic":         ic_res,
        "kfmax":      kfmax_res,
        "takeoff":    takeoff_res,
    }


# ── Padding ───────────────────────────────────────────────────────────────────

def pad(seq: np.ndarray, target: np.ndarray, seq_len: int):
    T = len(seq)
    length = min(T, seq_len)
    s = np.zeros((seq_len, seq.shape[1]),    dtype=np.float32)
    t = np.zeros((seq_len, target.shape[1]), dtype=np.float32)
    s[:length] = seq[:length]
    t[:length] = target[:length]
    return s, t, length


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Estrae sequenze temporali dai .mat mocap per il Transformer pitch.")
    parser.add_argument("--root",           default="/Users/ale/Kinematic_Data")
    parser.add_argument("--output",         default="data/generated/mocap_sequences.npz")
    parser.add_argument("--seq-len",        type=int,   default=128)
    parser.add_argument("--mocap-fps",      type=float, default=100.0,
                        help="Frame rate del mocap (default 100 Hz).")
    parser.add_argument("--target-fps",     type=float, default=30.0,
                        help="Frame rate target dopo ricampionamento (default 30 fps).")
    parser.add_argument("--descent-seconds", type=float, default=0.5,
                        help="Secondi prima di ic da includere (discesa dalla pedana).")
    args = parser.parse_args()

    root = Path(args.root)
    subject_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    print(f"Soggetti trovati: {len(subject_dirs)}")
    print(f"Finestra: {args.descent_seconds}s prima di ic → takeoff")
    print(f"Ricampionamento: {args.mocap_fps} Hz → {args.target_fps} fps")

    results, subject_ids = [], []
    for sd in subject_dirs:
        trial = extract_trial(
            sd, args.seq_len,
            mocap_fps=args.mocap_fps,
            target_fps=args.target_fps,
            descent_seconds=args.descent_seconds,
        )
        if trial is None:
            continue
        results.append(trial)
        subject_ids.append(trial["subject_id"])
        print(f"  {trial['subject_id']}  T={trial['T']}frames  "
              f"ic={trial['ic']}  kfmax={trial['kfmax']}  takeoff={trial['takeoff']}  "
              f"pitch=[{trial['target'].min():.1f}°,{trial['target'].max():.1f}°]")

    if not results:
        print("Nessun trial valido estratto.")
        return

    N = len(results)
    sequences   = np.zeros((N, args.seq_len, 34), dtype=np.float32)
    targets     = np.zeros((N, args.seq_len, 2),  dtype=np.float32)
    lengths     = np.zeros(N, dtype=np.int32)
    ic_frames   = np.zeros(N, dtype=np.int32)
    kfmax_frames= np.zeros(N, dtype=np.int32)
    takeoff_frames = np.zeros(N, dtype=np.int32)

    for i, r in enumerate(results):
        s, t, L = pad(r["seq"], r["target"], args.seq_len)
        sequences[i]      = s
        targets[i]        = t
        lengths[i]        = L
        ic_frames[i]      = r["ic"]
        kfmax_frames[i]   = r["kfmax"]
        takeoff_frames[i] = r["takeoff"]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        sequences=sequences,
        targets=targets,
        lengths=lengths,
        subject_ids=np.array(subject_ids),
        ic_frames=ic_frames,
        kfmax_frames=kfmax_frames,
        takeoff_frames=takeoff_frames,
    )
    print(f"\nSalvato: {output_path}")
    print(f"  N atleti:          {N}")
    print(f"  Shape sequences:   {sequences.shape}")
    print(f"  Pitch delta range: {targets.min():.1f}° → {targets.max():.1f}°")
    print(f"  Frames medi (post-resample): {lengths.mean():.0f}")
    print(f"  ic medio:      {ic_frames.mean():.0f} frame")
    print(f"  kfmax medio:   {kfmax_frames.mean():.0f} frame")
    print(f"  takeoff medio: {takeoff_frames.mean():.0f} frame")


if __name__ == "__main__":
    main()
