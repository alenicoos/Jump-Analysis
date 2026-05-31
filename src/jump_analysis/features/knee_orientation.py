from __future__ import annotations

"""Knee orientation time-series features.

La webcam frontale non puo' ricostruire il vero pitch/roll/yaw 3D del ginocchio.
Qui calcoliamo quindi proxy video coerenti nel tempo. Il ground truth vero resta
quello dei sensori IMU montati sulle ginocchia.
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
    """Orientamento/proxy di un ginocchio in gradi."""

    pitch_deg: float
    roll_deg: float
    yaw_deg: float


def signed_vertical_angle_deg(start: np.ndarray, end: np.ndarray) -> float:
    """Angolo signed del segmento rispetto alla verticale immagine."""

    vector = end - start
    return float(np.degrees(np.arctan2(vector[0], vector[1])))


def estimate_knee_orientation_from_pose(keypoints_xy: np.ndarray, side: str) -> KneeOrientation:
    """Stima proxy pitch/roll/yaw da keypoint 2D frontali.

    `pitch_deg` usa la flessione anca-ginocchio-caviglia come proxy.
    `roll_deg` usa l'inclinazione media di femore e tibia nel piano frontale.
    `yaw_deg` usa lo spostamento medio-laterale ginocchio-caviglia come proxy.

    Questi non sono angoli 3D veri. Sono segnali video-derived da usare come
    input del modello e confrontare/correggere con i sensori.
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
