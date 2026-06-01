from __future__ import annotations

"""Knee orientation time-series features.

A front-view webcam cannot reconstruct true 3D knee pitch/roll/yaw. This module
therefore computes time-consistent video proxies. True ground truth remains the
IMU sensors mounted near the knees.
"""

from dataclasses import dataclass

import numpy as np

from jump_analysis.features.front_2d_features import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    angle,
    distance,
)


@dataclass
class KneeOrientation:
    """Knee orientation/proxy in degrees."""

    pitch_deg: float
    roll_deg: float
    yaw_deg: float


def signed_vertical_angle_deg(start: np.ndarray, end: np.ndarray) -> float:
    """Signed segment angle relative to image vertical."""

    vector = end - start
    return float(np.degrees(np.arctan2(vector[0], vector[1])))


def estimate_knee_orientation_from_pose(keypoints_xy: np.ndarray, side: str) -> KneeOrientation:
    """Estimate pitch/roll/yaw proxies from front-view 2D keypoints.

    `pitch_deg` uses hip-knee-ankle flexion as a proxy.
    `roll_deg` uses average thigh/shank tilt in the frontal plane.
    `yaw_deg` uses medial-lateral knee-to-ankle displacement as a proxy.

    These are not true 3D angles. They are video-derived signals to use as model
    inputs and compare/correct against sensors.
    """

    if side == "left":
        hip_index, knee_index, ankle_index = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
        medial_sign = 1.0
    elif side == "right":
        hip_index, knee_index, ankle_index = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
        medial_sign = -1.0
    else:
        raise ValueError("side must be 'left' or 'right'")

    hip = keypoints_xy[hip_index]
    knee = keypoints_xy[knee_index]
    ankle = keypoints_xy[ankle_index]

    knee_angle = angle(hip, knee, ankle)
    pitch = 180.0 - knee_angle
    thigh_roll = signed_vertical_angle_deg(hip, knee)
    shank_roll = signed_vertical_angle_deg(knee, ankle)
    roll = float(np.nanmean([thigh_roll, shank_roll]))

    segment_length = max(distance(hip, knee), distance(knee, ankle), 1e-6)
    yaw = float(np.degrees(np.arctan2(medial_sign * (knee[0] - ankle[0]), segment_length)))
    return KneeOrientation(pitch_deg=float(pitch), roll_deg=roll, yaw_deg=yaw)
