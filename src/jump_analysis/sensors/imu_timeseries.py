from __future__ import annotations

"""IMU orientation time-series loading.

Per ora supportiamo CSV gia' esportati/registrati con colonne temporali e
orientamento. Il lettore live del BWT901CL verra' aggiunto dopo aver fissato la
modalita' di connessione usata in laboratorio.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ImuOrientationSeries:
    """Serie pitch/roll/yaw interpolabile sui timestamp video."""

    timestamp_s: np.ndarray
    pitch_deg: np.ndarray
    roll_deg: np.ndarray
    yaw_deg: np.ndarray

    def interpolate(self, target_timestamp_s: np.ndarray) -> pd.DataFrame:
        """Interpola pitch/roll/yaw sui timestamp della webcam."""

        if len(self.timestamp_s) == 0:
            return pd.DataFrame(
                {
                    "pitch_deg": np.full(len(target_timestamp_s), np.nan),
                    "roll_deg": np.full(len(target_timestamp_s), np.nan),
                    "yaw_deg": np.full(len(target_timestamp_s), np.nan),
                }
            )
        return pd.DataFrame(
            {
                "pitch_deg": np.interp(target_timestamp_s, self.timestamp_s, self.pitch_deg),
                "roll_deg": np.interp(target_timestamp_s, self.timestamp_s, self.roll_deg),
                "yaw_deg": np.interp(target_timestamp_s, self.timestamp_s, self.yaw_deg),
            }
        )


def load_imu_orientation_csv(path: str | Path | None) -> ImuOrientationSeries | None:
    """Carica un CSV IMU con timestamp e pitch/roll/yaw.

    Colonne accettate:
    - tempo: `timestamp_s`, `time_s`, `time`, oppure `timestamp`;
    - angoli: `pitch_deg`/`roll_deg`/`yaw_deg` oppure `pitch`/`roll`/`yaw`.
    """

    if path is None:
        return None

    data = pd.read_csv(path)
    time_column = _first_existing(data, ["timestamp_s", "time_s", "time", "timestamp"])
    pitch_column = _first_existing(data, ["pitch_deg", "pitch"])
    roll_column = _first_existing(data, ["roll_deg", "roll"])
    yaw_column = _first_existing(data, ["yaw_deg", "yaw"])

    numeric = data[[time_column, pitch_column, roll_column, yaw_column]].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna().sort_values(time_column)
    return ImuOrientationSeries(
        timestamp_s=numeric[time_column].to_numpy(dtype=float),
        pitch_deg=numeric[pitch_column].to_numpy(dtype=float),
        roll_deg=numeric[roll_column].to_numpy(dtype=float),
        yaw_deg=numeric[yaw_column].to_numpy(dtype=float),
    )


def _first_existing(data: pd.DataFrame, candidates: list[str]) -> str:
    """Restituisce il primo nome colonna disponibile."""

    for candidate in candidates:
        if candidate in data.columns:
            return candidate
    raise ValueError(f"Missing one of columns: {candidates}")
