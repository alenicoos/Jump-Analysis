from __future__ import annotations

"""Model layer for LESS/anomaly analysis.

For now, the mocap dataset is used as the "normal" reference and new feature
rows are scored by how far they move away from that reference.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from jump_analysis.data import FRONT_2D_FEATURE_COLUMNS


@dataclass
class RobustAnomalyModel:
    """Robust anomaly detector fitted on normal reference features."""

    reference_median: pd.Series
    reference_mad: pd.Series
    reference_low: pd.Series
    reference_high: pd.Series
    feature_columns: list[str]
    z_threshold: float = 4.0
    max_outlier_features: int = 8
    top_k: int = 5
    name: str = "robust-reference-anomaly"

    @classmethod
    def fit_reference(
        cls,
        reference: pd.DataFrame,
        feature_columns: list[str] | None = None,
        excluded_features: list[str] | None = None,
        z_threshold: float = 4.0,
        max_outlier_features: int = 8,
        top_k: int = 5,
        central_percentile: float = 95.0,
    ) -> "RobustAnomalyModel":
        """Build the model from reference-dataset features.

        Robust statistics are used:
        - median instead of mean;
        - MAD instead of standard deviation;
        - central percentile band to count out-of-range features.

        `crop_length_frames` is excluded by default because it depends on frame
        rate and can vary substantially across mocap, webcam, and online videos.
        """

        excluded = set(excluded_features if excluded_features is not None else ["crop_length_frames"])
        columns = [column for column in (feature_columns or FRONT_2D_FEATURE_COLUMNS) if column not in excluded]
        missing = [column for column in columns if column not in reference.columns]
        if missing:
            raise ValueError(f"Missing reference feature columns: {missing}")

        numeric = reference[columns].apply(pd.to_numeric, errors="coerce")
        # MAD = median absolute deviation. It is much less sensitive to outliers
        # than classic standard deviation.
        median = numeric.median(axis=0)
        mad = (numeric - median).abs().median(axis=0)
        tail = (100.0 - central_percentile) / 2.0
        low = numeric.quantile(tail / 100.0, axis=0)
        high = numeric.quantile(1.0 - tail / 100.0, axis=0)
        return cls(
            reference_median=median,
            reference_mad=mad,
            reference_low=low,
            reference_high=high,
            feature_columns=columns,
            z_threshold=z_threshold,
            max_outlier_features=max_outlier_features,
            top_k=top_k,
        )

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Compute anomaly score and most abnormal feature for each row."""

        missing = [column for column in self.feature_columns if column not in features.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        numeric = features[self.feature_columns].apply(pd.to_numeric, errors="coerce")
        # 1.4826 makes MAD comparable to standard deviation for Gaussian data.
        # If a feature has zero MAD, use the percentile band as fallback.
        scale = (1.4826 * self.reference_mad).replace(0.0, np.nan)
        fallback_scale = (self.reference_high - self.reference_low).abs() / 4.0
        scale = scale.fillna(fallback_scale).replace(0.0, 1e-6)
        robust_z = (numeric - self.reference_median) / scale
        abs_z = robust_z.abs()

        # Two complementary ways to flag "unusual":
        # - outside the central dataset band;
        # - robust z-score above threshold.
        percentile_outliers = (numeric.lt(self.reference_low) | numeric.gt(self.reference_high)).sum(axis=1)
        z_outliers = abs_z.gt(self.z_threshold).sum(axis=1)
        outlier_count = pd.concat([percentile_outliers, z_outliers], axis=1).max(axis=1)

        top_k = max(1, min(self.top_k, len(self.feature_columns)))
        # Score is the average of the worst features, not the sum of all of
        # them. This lets a few strong signals matter without exploding only
        # because there are many columns.
        top_k_score = abs_z.apply(lambda row: float(row.nlargest(top_k).mean()), axis=1)
        max_z = abs_z.max(axis=1)
        worst_features = abs_z.idxmax(axis=1)

        result = pd.DataFrame(index=features.index)
        result["prediction"] = np.where(
            (top_k_score > self.z_threshold) | (outlier_count >= self.max_outlier_features),
            "anomaly",
            "normal",
        )
        result["anomaly_score"] = top_k_score
        result["score_threshold"] = self.z_threshold
        result["outlier_feature_count"] = outlier_count.astype(int)
        result["outlier_feature_limit"] = self.max_outlier_features
        result["analyzed_feature_count"] = len(self.feature_columns)
        result["max_abs_robust_z"] = max_z
        result["worst_feature"] = worst_features
        result["worst_feature_z"] = [
            float(robust_z.loc[index, feature]) for index, feature in worst_features.items()
        ]
        result["worst_feature_value"] = [
            float(numeric.loc[index, feature]) for index, feature in worst_features.items()
        ]
        result["worst_feature_reference_median"] = [
            float(self.reference_median[feature]) for feature in worst_features
        ]
        result["model"] = self.name
        return result
