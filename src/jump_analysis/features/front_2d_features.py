from __future__ import annotations

"""Feature extraction from two front-view YOLO keyframes.

Questo modulo non apre la webcam e non usa direttamente YOLO. Riceve solo
coordinate 2D dei keypoint, gia' selezionate/normalizzate, e calcola le feature
biomeccaniche frontali usate dal dataset e dal modello.
"""

import math
from dataclasses import dataclass

import numpy as np

from jump_analysis.data import BASE_FRONT_2D_FEATURES, FRONT_2D_FEATURE_COLUMNS


# Indici COCO usati dai modelli YOLO pose.
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
    """I due momenti del salto da cui estraiamo le 37 feature."""

    initial_contact: np.ndarray
    max_knee_flexion: np.ndarray
    crop_length_frames: int


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Punto medio tra due coordinate 2D."""

    return (a + b) / 2.0


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distanza euclidea 2D tra due keypoint."""

    return float(np.linalg.norm(a - b))


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angolo ABC in gradi.

    `b` e' il vertice dell'angolo. Per esempio anca-ginocchio-caviglia misura
    l'angolo del ginocchio nel piano frontale.
    """

    v1 = a - b
    v2 = c - b
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return float("nan")
    cosine = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def horizontal_tilt_degrees(a: np.ndarray, b: np.ndarray) -> float:
    """Inclinazione assoluta di una linea rispetto all'orizzontale.

    Una linea quasi orizzontale puo' avere angolo raw vicino a 0 o a 180 gradi,
    a seconda dell'ordine dei punti. Qui riportiamo sempre il valore piccolo:
    0 gradi significa orizzontale, 10 gradi significa inclinata di 10 gradi.
    """

    raw = abs(math.degrees(math.atan2(float(b[1] - a[1]), float(b[0] - a[0]))))
    return min(raw, abs(180.0 - raw))


def body_keypoint(k: np.ndarray) -> np.ndarray:
    """Centro approssimato del tronco, tra centro spalle e centro anche."""

    shoulder_center = midpoint(k[LEFT_SHOULDER], k[RIGHT_SHOULDER])
    hip_center = midpoint(k[LEFT_HIP], k[RIGHT_HIP])
    return midpoint(shoulder_center, hip_center)


def extract_front_2d_features_for_keyframe(k: np.ndarray) -> dict[str, float]:
    """Calcola le 18 feature frontali per un singolo keyframe.

    Le distanze pure restano nella scala normalizzata. Le feature piu'
    confrontabili sono soprattutto i rapporti, perche' riducono l'effetto di
    distanza dalla camera e corporatura.
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

    # Lean laterale del tronco: se la linea anche-spalle e' verticale,
    # l'inclinazione laterale e' vicina a 0.
    trunk_lateral_lean = abs(90.0 - abs(math.degrees(math.atan2(
        float(shoulder_center[1] - hip_center[1]),
        float(shoulder_center[0] - hip_center[0]),
    ))))
    shoulder_tilt = horizontal_tilt_degrees(k[LEFT_SHOULDER], k[RIGHT_SHOULDER])
    hip_tilt = horizontal_tilt_degrees(k[LEFT_HIP], k[RIGHT_HIP])

    return {
        # Distanze principali tra lato sinistro e destro del corpo.
        "knee_distance": knee_distance,
        "ankle_distance": ankle_distance,
        "hip_distance": hip_distance,
        "shoulder_distance": shoulder_distance,
        # Rapporti normalizzati: utili per confrontare persone/video diversi.
        "knee_shoulder_width_ratio": knee_distance / max(reference_width, EPSILON),
        "ankle_shoulder_width_ratio": ankle_distance / max(reference_width, EPSILON),
        "knee_ankle_width_ratio": knee_distance / max(ankle_distance, EPSILON),
        # Offset mediali: indicano quanto le ginocchia entrano/escono rispetto
        # alle caviglie nel piano frontale.
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
    """Costruisce una singola riga nel formato a 37 colonne.

    Le prime 18 colonne sono prefissate `ic_`, le successive 18 `kfmax_`,
    poi aggiungiamo `crop_length_frames`.
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
