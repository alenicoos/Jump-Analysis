from __future__ import annotations

"""Front-view pose geometry helpers.

This module does not open the webcam and does not call YOLO directly. It keeps
the COCO keypoint constants and small geometry functions shared by setup,
protocol validation, and video capture.
"""

import math

import numpy as np


# COCO indices used by YOLO pose models.
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16
HEAD_KEYPOINTS = (0, 1, 2, 3, 4)
BODY_MODEL_KEYPOINTS = tuple(index for index in range(17) if index not in HEAD_KEYPOINTS)


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Midpoint between two 2D coordinates."""

    return (a + b) / 2.0


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """2D Euclidean distance between two keypoints."""

    return float(np.linalg.norm(a - b))


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """ABC angle in degrees.

    `b` is the angle vertex. For example, hip-knee-ankle measures knee angle in
    the frontal plane.
    """

    v1 = a - b
    v2 = c - b
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return float("nan")
    cosine = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def body_keypoint(k: np.ndarray) -> np.ndarray:
    """Approximate trunk center between shoulder center and hip center."""

    shoulder_center = _midpoint(k[LEFT_SHOULDER], k[RIGHT_SHOULDER])
    hip_center = _midpoint(k[LEFT_HIP], k[RIGHT_HIP])
    return _midpoint(shoulder_center, hip_center)


def _temporal_feature_indices(include_head: bool = False) -> list[int]:
    """Flattened x-then-y temporal feature indices for model inputs."""

    keypoints = range(17) if include_head else BODY_MODEL_KEYPOINTS
    return [*keypoints, *(17 + index for index in keypoints)]


def select_temporal_features(sequence: np.ndarray, include_head: bool = False) -> np.ndarray:
    """Select temporal keypoint channels used by the ML models."""

    return sequence[..., _temporal_feature_indices(include_head)]


# ── Domain-invariant AE features ─────────────────────────────────────────────
# Body-only layout (12 keypoints, indices 0-11 in the 24-dim vector):
#   0:L_shoulder  1:R_shoulder  2:L_elbow  3:R_elbow  4:L_wrist  5:R_wrist
#   6:L_hip       7:R_hip       8:L_knee   9:R_knee  10:L_ankle 11:R_ankle
# x columns: 0-11,  y columns: 12-23

_LS, _RS = 0, 1
_LH, _RH = 6, 7
_LK, _RK = 8, 9
_LA, _RA = 10, 11

AE_FEATURE_NAMES: list[str] = [
    "left_knee_flexion",
    "right_knee_flexion",
    "left_hip_flexion",
    "right_hip_flexion",
    "trunk_lateral_lean",
    "knee_width_ratio",
    "left_knee_valgus",
    "right_knee_valgus",
    "knee_center_vs_ankle_center",
    "body_lean_over_ankles",
    "left_leg_length_ratio",
    "right_leg_length_ratio",
    "knee_flexion_asymmetry",
    "hip_flexion_asymmetry",
    "leg_ratio_asymmetry",
    "shoulder_tilt",
]
AE_FEATURE_DIM: int = len(AE_FEATURE_NAMES)


def _angle_abc_seq(
    ax: np.ndarray, ay: np.ndarray,
    bx: np.ndarray, by: np.ndarray,
    cx: np.ndarray, cy: np.ndarray,
) -> np.ndarray:
    """Vectorised angle at B for sequences of points A, B, C. Returns degrees (T,)."""
    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by
    dot  = v1x * v2x + v1y * v2y
    n1   = np.sqrt(v1x ** 2 + v1y ** 2)
    n2   = np.sqrt(v2x ** 2 + v2y ** 2)
    cos_ = np.clip(dot / np.maximum(n1 * n2, 1e-8), -1.0, 1.0)
    return np.degrees(np.arccos(cos_))


def _dist_seq(
    ax: np.ndarray, ay: np.ndarray,
    bx: np.ndarray, by: np.ndarray,
) -> np.ndarray:
    """Euclidean distance between two sequences of points. Returns (T,)."""
    return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def extract_ae_features(kp_seq: np.ndarray) -> np.ndarray:
    """Convert body-only keypoint sequences to domain-invariant AE features.

    Parameters
    ----------
    kp_seq : (T, 24) float array — body-only keypoints, x-then-y layout,
             normalised by body height (as produced by frames_to_transformer_input
             or generate_mocap_sequences).

    Returns
    -------
    features : (T, 16) float32 array

    All features are invariant to camera distance, absolute frame position,
    and body size.  Angles and horizontal ratios are invariant to y-axis
    direction. Explicit asymmetry features (bilateral differences) are also
    invariant. Y-magnitude ratios use |Δy|/sw which is invariant to y sign.
    """
    T = len(kp_seq)
    x = kp_seq[:, :12].copy()   # (T, 12)  horizontal
    y = kp_seq[:, 12:]          # (T, 12)  vertical

    # ── Normalize x direction to mocap convention: person's right at positive x.
    # Use only frames where the shoulders are not zero (i.e. not padding frames).
    valid_mask = (np.abs(x[:, _RS]) + np.abs(x[:, _LS])) > 1e-6
    if valid_mask.any():
        sw_valid = (x[valid_mask, _RS] - x[valid_mask, _LS])
        if float(np.median(sw_valid)) < 0:
            x = -x
    elif float(np.median(x[:, _RS] - x[:, _LS])) < 0:
        x = -x

    # Shoulder width (reference scale for horizontal ratios) — always positive
    sw = np.maximum(x[:, _RS] - x[:, _LS], 1e-6)

    # Body center x (used for balance feature)
    body_cx = (x[:, _LS] + x[:, _RS] + x[:, _LH] + x[:, _RH]) / 4.0

    # Mid-points x
    shoulder_mid_x = (x[:, _LS] + x[:, _RS]) / 2.0
    hip_mid_x      = (x[:, _LH] + x[:, _RH]) / 2.0
    knee_mid_x     = (x[:, _LK] + x[:, _RK]) / 2.0
    ankle_mid_x    = (x[:, _LA] + x[:, _RA]) / 2.0

    # ── Angles (invariant to scale and y-axis direction) ─────────────────────
    left_knee_flex  = 180.0 - _angle_abc_seq(
        x[:,_LH], y[:,_LH], x[:,_LK], y[:,_LK], x[:,_LA], y[:,_LA])
    right_knee_flex = 180.0 - _angle_abc_seq(
        x[:,_RH], y[:,_RH], x[:,_RK], y[:,_RK], x[:,_RA], y[:,_RA])
    left_hip_flex   = 180.0 - _angle_abc_seq(
        x[:,_LS], y[:,_LS], x[:,_LH], y[:,_LH], x[:,_LK], y[:,_LK])
    right_hip_flex  = 180.0 - _angle_abc_seq(
        x[:,_RS], y[:,_RS], x[:,_RH], y[:,_RH], x[:,_RK], y[:,_RK])

    # ── Horizontal ratios (invariant to scale, position, and y direction) ────
    trunk_lean      = (hip_mid_x - shoulder_mid_x) / sw
    knee_wr         = (x[:, _RK] - x[:, _LK]) / sw
    l_valgus        = (x[:, _LK] - x[:, _LA]) / sw   # negative = valgus
    r_valgus        = (x[:, _RK] - x[:, _RA]) / sw   # positive = valgus
    knee_vs_ankle   = (knee_mid_x - ankle_mid_x) / sw
    balance         = (body_cx   - ankle_mid_x) / sw

    # ── Leg-length ratios (invariant to scale and y direction) ───────────────
    l_thigh  = _dist_seq(x[:,_LH], y[:,_LH], x[:,_LK], y[:,_LK])
    l_shank  = _dist_seq(x[:,_LK], y[:,_LK], x[:,_LA], y[:,_LA])
    r_thigh  = _dist_seq(x[:,_RH], y[:,_RH], x[:,_RK], y[:,_RK])
    r_shank  = _dist_seq(x[:,_RK], y[:,_RK], x[:,_RA], y[:,_RA])
    l_ratio  = l_thigh / np.maximum(l_shank, 1e-6)
    r_ratio  = r_thigh / np.maximum(r_shank, 1e-6)

    # ── Explicit bilateral asymmetries ────────────────────────────────────────
    # Signed differences — invariant to y-axis direction because they are
    # computed from quantities that are themselves already invariant.
    knee_flex_asym = left_knee_flex  - right_knee_flex
    hip_flex_asym  = left_hip_flex   - right_hip_flex
    leg_ratio_asym = l_ratio         - r_ratio

    # ── Y-magnitude ratios — invariant to y-axis sign ─────────────────────────
    shoulder_tilt  = np.abs(y[:, _RS] - y[:, _LS]) / sw

    features = np.stack([
        left_knee_flex, right_knee_flex,
        left_hip_flex,  right_hip_flex,
        trunk_lean,
        knee_wr,
        l_valgus, r_valgus,
        knee_vs_ankle, balance,
        l_ratio, r_ratio,
        knee_flex_asym, hip_flex_asym, leg_ratio_asym,
        shoulder_tilt,
    ], axis=1)

    return features.astype(np.float32)
