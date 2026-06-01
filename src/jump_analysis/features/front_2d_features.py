from __future__ import annotations

"""Feature extraction from two front-view YOLO keyframes.

This module does not open the webcam and does not call YOLO directly. It only
receives selected/normalized 2D keypoint coordinates and computes the
front-view biomechanical features used by the dataset and model.
"""

import math
from dataclasses import dataclass

import numpy as np

from jump_analysis.data import BASE_FRONT_2D_FEATURES, FRONT_2D_FEATURE_COLUMNS


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
EPSILON = 1e-6


@dataclass
class FrontKeyframes:
    """The two jump moments used to extract the 37 features."""

    initial_contact: np.ndarray
    max_knee_flexion: np.ndarray
    crop_length_frames: int


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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


def horizontal_tilt_degrees(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute line tilt relative to horizontal.

    A nearly horizontal line may have a raw angle near 0 or 180 degrees,
    depending on point order. Always return the small value: 0 degrees means
    horizontal, 10 degrees means tilted by 10 degrees.
    """

    raw = abs(math.degrees(math.atan2(float(b[1] - a[1]), float(b[0] - a[0]))))
    return min(raw, abs(180.0 - raw))


def body_keypoint(k: np.ndarray) -> np.ndarray:
    """Approximate trunk center between shoulder center and hip center."""

    shoulder_center = midpoint(k[LEFT_SHOULDER], k[RIGHT_SHOULDER])
    hip_center = midpoint(k[LEFT_HIP], k[RIGHT_HIP])
    return midpoint(shoulder_center, hip_center)


def extract_front_2d_features_for_keyframe(k: np.ndarray) -> dict[str, float]:
    """Compute the 18 front-view features for one keyframe.

    Raw distances remain in normalized scale. Ratios are usually the most
    comparable features because they reduce the effect of camera distance and
    body size.
    """

    knee_distance = distance(k[LEFT_KNEE], k[RIGHT_KNEE])
    ankle_distance = distance(k[LEFT_ANKLE], k[RIGHT_ANKLE])
    hip_distance = distance(k[LEFT_HIP], k[RIGHT_HIP])
    shoulder_distance = distance(k[LEFT_SHOULDER], k[RIGHT_SHOULDER])
    reference_width = shoulder_distance if shoulder_distance > EPSILON else hip_distance
    knee_center = midpoint(k[LEFT_KNEE], k[RIGHT_KNEE])
    ankle_center = midpoint(k[LEFT_ANKLE], k[RIGHT_ANKLE])
    hip_center = midpoint(k[LEFT_HIP], k[RIGHT_HIP])
    shoulder_center = midpoint(k[LEFT_SHOULDER], k[RIGHT_SHOULDER])

    # Trunk lateral lean: if the hip-shoulder line is vertical, lateral lean is near 0.
    trunk_lateral_lean = abs(90.0 - abs(math.degrees(math.atan2(
        float(shoulder_center[1] - hip_center[1]),
        float(shoulder_center[0] - hip_center[0]),
    ))))
    shoulder_tilt = horizontal_tilt_degrees(k[LEFT_SHOULDER], k[RIGHT_SHOULDER])
    hip_tilt = horizontal_tilt_degrees(k[LEFT_HIP], k[RIGHT_HIP])

    return {
        # Main distances between left and right sides of the body.
        "knee_distance": knee_distance,
        "ankle_distance": ankle_distance,
        "hip_distance": hip_distance,
        "shoulder_distance": shoulder_distance,
        # Normalized ratios: useful for comparing different people/videos.
        "knee_shoulder_width_ratio": knee_distance / max(reference_width, EPSILON),
        "ankle_shoulder_width_ratio": ankle_distance / max(reference_width, EPSILON),
        "knee_ankle_width_ratio": knee_distance / max(ankle_distance, EPSILON),
        # Medial offsets: how much knees move inward/outward relative to ankles
        # in the frontal plane.
        "left_knee_medial_offset_ratio": (k[LEFT_KNEE][0] - k[LEFT_ANKLE][0]) / max(reference_width, EPSILON),
        "right_knee_medial_offset_ratio": (k[RIGHT_ANKLE][0] - k[RIGHT_KNEE][0]) / max(reference_width, EPSILON),
        "knee_center_ankle_center_offset_ratio": (knee_center[0] - ankle_center[0]) / max(reference_width, EPSILON),
        "left_hip_knee_ankle_frontal_angle": angle(k[LEFT_HIP], k[LEFT_KNEE], k[LEFT_ANKLE]),
        "right_hip_knee_ankle_frontal_angle": angle(k[RIGHT_HIP], k[RIGHT_KNEE], k[RIGHT_ANKLE]),
        "shoulder_tilt_degrees": shoulder_tilt,
        "hip_tilt_degrees": hip_tilt,
        "trunk_lateral_lean_degrees": trunk_lateral_lean,
        "left_right_ankle_y_difference_ratio": abs(k[LEFT_ANKLE][1] - k[RIGHT_ANKLE][1]) / max(reference_width, EPSILON),
        "left_right_knee_y_difference_ratio": abs(k[LEFT_KNEE][1] - k[RIGHT_KNEE][1]) / max(reference_width, EPSILON),
        "body_center_x_over_ankle_center_offset_ratio": (hip_center[0] - ankle_center[0]) / max(reference_width, EPSILON),
    }


def build_front_2d_feature_row(keyframes: FrontKeyframes) -> dict[str, float]:
    """Build one row in the 37-column feature format.

    The first 18 columns are prefixed `ic_`, the next 18 `kfmax_`, then
    `crop_length_frames` is added.
    """

    row = {}
    for prefix, keypoints in (
        ("ic", keyframes.initial_contact),
        ("kfmax", keyframes.max_knee_flexion),
    ):
        values = extract_front_2d_features_for_keyframe(keypoints)
        for name in BASE_FRONT_2D_FEATURES:
            row[f"{prefix}_{name}"] = values[name]
    row["crop_length_frames"] = keyframes.crop_length_frames
    return {column: row[column] for column in FRONT_2D_FEATURE_COLUMNS}
