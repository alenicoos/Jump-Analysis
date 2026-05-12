from __future__ import annotations

"""Model layer for LESS/anomaly analysis.

Per ora usiamo il dataset mocap come riferimento "normale" 
e segnaliamo quanto una nuova riga di feature si allontana da quel riferimento.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from jump_analysis.data import FRONT_2D_FEATURE_COLUMNS


@dataclass
class RobustAnomalyModel:
    """Anomaly detector robusto addestrato su feature di riferimento normali."""

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
        """Costruisce il modello dalle feature del dataset di riferimento.

        Usiamo statistiche robuste:
        - mediana al posto della media;
        - MAD al posto della deviazione standard;
        - banda percentile centrale per contare feature fuori range.

        `crop_length_frames` e' esclusa di default perche' dipende dal frame
        rate e puo' cambiare molto tra mocap, webcam e video online.
        """

        excluded = set(excluded_features if excluded_features is not None else ["crop_length_frames"])
        columns = [column for column in (feature_columns or FRONT_2D_FEATURE_COLUMNS) if column not in excluded]
        missing = [column for column in columns if column not in reference.columns]
        if missing:
            raise ValueError(f"Missing reference feature columns: {missing}")

        numeric = reference[columns].apply(pd.to_numeric, errors="coerce")
        # MAD = median absolute deviation. Molto meno sensibile agli outlier
        # rispetto alla deviazione standard classica.
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
        """Calcola anomaly score e feature piu' anomala per ogni riga."""

        missing = [column for column in self.feature_columns if column not in features.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        numeric = features[self.feature_columns].apply(pd.to_numeric, errors="coerce")
        # 1.4826 rende il MAD confrontabile con la deviazione standard se i dati
        # fossero gaussiani. Se una feature ha MAD zero, usiamo la banda
        # percentile come fallback.
        scale = (1.4826 * self.reference_mad).replace(0.0, np.nan)
        fallback_scale = (self.reference_high - self.reference_low).abs() / 4.0
        scale = scale.fillna(fallback_scale).replace(0.0, 1e-6)
        robust_z = (numeric - self.reference_median) / scale
        abs_z = robust_z.abs()

        # Due modi complementari per dire "strano":
        # - fuori dalla banda centrale del dataset;
        # - robust z-score oltre soglia.
        percentile_outliers = (numeric.lt(self.reference_low) | numeric.gt(self.reference_high)).sum(axis=1)
        z_outliers = abs_z.gt(self.z_threshold).sum(axis=1)
        outlier_count = pd.concat([percentile_outliers, z_outliers], axis=1).max(axis=1)

        top_k = max(1, min(self.top_k, len(self.feature_columns)))
        # Lo score e' la media delle peggiori feature, non la somma di tutte.
        # Cosi' pochi segnali forti pesano, ma il risultato non esplode solo
        # perche' abbiamo molte colonne.
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
