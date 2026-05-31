from __future__ import annotations

"""Models for knee IMU ground-truth prediction.

Entrambi i modelli sono baseline lineari ridge frame-by-frame. La separazione
importante e' quali input possono usare:

- `VideoOrientationCorrectionModel`: usa i proxy pitch/roll/yaw calcolati dal
  video piu' dati di scala/contesto, ma non usa le traiettorie dei keypoint.
- `PoseTrajectoryKneeOrientationModel`: usa le traiettorie dei keypoint e dati
  di scala/contesto, ma non usa i proxy pitch/roll/yaw calcolati dal video.

Piu' avanti potremo sostituire queste baseline con modelli sequenziali, ma la
divisione sperimentale rimane la stessa.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SENSOR_TARGET_COLUMNS = [
    "left_sensor_pitch_deg",
    "left_sensor_roll_deg",
    "left_sensor_yaw_deg",
    "right_sensor_pitch_deg",
    "right_sensor_roll_deg",
    "right_sensor_yaw_deg",
]

VIDEO_ORIENTATION_INPUT_COLUMNS = [
    "time_from_start_s",
    "normalized_time",
    "body_height_px",
    "shoulder_width_px",
    "shoulder_width_m",
    "hip_width_px",
    "knee_width_px",
    "ankle_width_px",
    "left_video_pitch_deg",
    "left_video_roll_deg",
    "left_video_yaw_deg",
    "right_video_pitch_deg",
    "right_video_roll_deg",
    "right_video_yaw_deg",
]

POSE_TRAJECTORY_EXTRA_COLUMNS = [
    "time_from_start_s",
    "normalized_time",
    "body_height_px",
    "shoulder_width_px",
    "shoulder_width_m",
    "hip_width_px",
    "knee_width_px",
    "ankle_width_px",
]


@dataclass
class RidgeSequenceRegressor:
    """Baseline ridge multi-output per predire una serie frame-by-frame."""

    input_columns: list[str]
    target_columns: list[str]
    alpha: float = 1.0
    weights: np.ndarray | None = None
    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None

    def fit(self, frame_data: pd.DataFrame) -> "RidgeSequenceRegressor":
        """Allena il modello sui frame che hanno ground truth sensore."""

        data = frame_data[self.input_columns + self.target_columns].apply(pd.to_numeric, errors="coerce").dropna()
        if data.empty:
            raise ValueError("No complete rows available for training.")

        x = data[self.input_columns].to_numpy(dtype=float)
        y = data[self.target_columns].to_numpy(dtype=float)
        self.feature_mean = x.mean(axis=0)
        self.feature_std = x.std(axis=0)
        self.feature_std[self.feature_std == 0.0] = 1.0

        x_norm = (x - self.feature_mean) / self.feature_std
        x_design = np.column_stack([np.ones(len(x_norm)), x_norm])
        regularizer = self.alpha * np.eye(x_design.shape[1])
        regularizer[0, 0] = 0.0
        self.weights = np.linalg.solve(x_design.T @ x_design + regularizer, x_design.T @ y)
        return self

    def predict(self, frame_data: pd.DataFrame) -> pd.DataFrame:
        """Predice pitch/roll/yaw sensore per ogni frame."""

        if self.weights is None or self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Model is not fitted.")

        x = frame_data[self.input_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        x = np.nan_to_num(x, nan=0.0)
        x_norm = (x - self.feature_mean) / self.feature_std
        x_design = np.column_stack([np.ones(len(x_norm)), x_norm])
        prediction = x_design @ self.weights
        return pd.DataFrame(prediction, columns=[f"pred_{column}" for column in self.target_columns])

    def save(self, path: str | Path) -> None:
        """Salva i pesi in formato `.npz`."""

        if self.weights is None or self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Model is not fitted.")
        np.savez(
            path,
            input_columns=np.array(self.input_columns),
            target_columns=np.array(self.target_columns),
            alpha=np.array([self.alpha], dtype=float),
            weights=self.weights,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
        )

    @classmethod
    def load(cls, path: str | Path) -> "RidgeSequenceRegressor":
        """Carica un modello salvato con `save`."""

        data = np.load(path, allow_pickle=True)
        model = cls(
            input_columns=data["input_columns"].tolist(),
            target_columns=data["target_columns"].tolist(),
            alpha=float(data["alpha"][0]),
        )
        model.weights = data["weights"]
        model.feature_mean = data["feature_mean"]
        model.feature_std = data["feature_std"]
        return model


class VideoOrientationCorrectionModel(RidgeSequenceRegressor):
    """Predice ground truth IMU partendo dai proxy video pitch/roll/yaw."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__(
            input_columns=VIDEO_ORIENTATION_INPUT_COLUMNS,
            target_columns=SENSOR_TARGET_COLUMNS,
            alpha=alpha,
        )


class PoseTrajectoryKneeOrientationModel(RidgeSequenceRegressor):
    """Predice ground truth IMU usando traiettorie keypoint e dati di scala."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__(
            input_columns=pose_trajectory_input_columns(),
            target_columns=SENSOR_TARGET_COLUMNS,
            alpha=alpha,
        )


def pose_trajectory_input_columns() -> list[str]:
    """Colonne input del modello basato su traiettorie dei keypoint."""

    columns = list(POSE_TRAJECTORY_EXTRA_COLUMNS)
    for keypoint_index in range(17):
        columns.extend(
            [
                f"kp_{keypoint_index:02d}_x_m",
                f"kp_{keypoint_index:02d}_y_m",
                f"kp_{keypoint_index:02d}_conf",
            ]
        )
    return columns
