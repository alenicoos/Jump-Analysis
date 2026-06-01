from __future__ import annotations

"""IMU orientation time-series loading.

For now this supports already exported/recorded CSV files with time and
orientation columns.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ImuOrientationSeries:
    """Pitch/roll/yaw series that can be interpolated onto video timestamps."""

    timestamp_s: np.ndarray
    pitch_deg: np.ndarray
    roll_deg: np.ndarray
    yaw_deg: np.ndarray

    def interpolate(self, target_timestamp_s: np.ndarray) -> pd.DataFrame:
        """Interpolate pitch/roll/yaw onto webcam timestamps."""

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
    """Load an IMU CSV with timestamp and pitch/roll/yaw.

    Accepted columns:
    - time: `timestamp_s`, `time_s`, `time`, or `timestamp`;
    - angles: `pitch_deg`/`roll_deg`/`yaw_deg` or `pitch`/`roll`/`yaw`.
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
    """Return the first available column name."""

    for candidate in candidates:
        if candidate in data.columns:
            return candidate
    raise ValueError(f"Missing one of columns: {candidates}")
