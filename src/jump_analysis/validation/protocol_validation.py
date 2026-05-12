from __future__ import annotations

"""Drop-jump protocol validation.

Questo modulo controlla se il movimento registrato ha la struttura minima di un
drop jump: partenza da un rialzo, atterraggio a due piedi e salto successivo.
Non assegna un LESS score e non valuta ancora la qualita' clinica completa.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from jump_analysis.features.front_2d_features import (
    LEFT_ANKLE,
    LEFT_SHOULDER,
    RIGHT_ANKLE,
    RIGHT_SHOULDER,
    body_keypoint,
    distance,
)


@dataclass
class ProtocolCheck:
    """Risultato numerico di un singolo check del protocollo."""

    name: str
    passed: bool
    value: float
    threshold: float


@dataclass
class DropJumpProtocolResult:
    """Risultato complessivo dei check sul drop jump."""

    passed: bool
    checks: list[ProtocolCheck]

    def as_metadata(self) -> dict[str, float | int]:
        """Converte i check in colonne salvabili dentro il CSV delle feature."""

        metadata: dict[str, float | int] = {"protocol_passed": int(self.passed)}
        for check in self.checks:
            metadata[f"{check.name}_passed"] = int(check.passed)
            metadata[f"{check.name}_value"] = check.value
            metadata[f"{check.name}_threshold"] = check.threshold
        return metadata


class DropJumpProtocolValidator:
    """Valida la sequenza base del drop jump dopo il setup."""

    def __init__(
        self,
        min_drop_height_ratio: float = 0.15,
        max_two_foot_contact_ratio: float = 0.10,
        min_second_jump_ratio: float = 0.12,
        second_jump_window_frames: int = 60,
    ) -> None:
        self.min_drop_height_ratio = min_drop_height_ratio
        self.max_two_foot_contact_ratio = max_two_foot_contact_ratio
        self.min_second_jump_ratio = min_second_jump_ratio
        self.second_jump_window_frames = second_jump_window_frames

    def validate(
        self,
        frames: list[Any],
        initial_contact_index: int,
        max_knee_flexion_index: int,
    ) -> DropJumpProtocolResult:
        """Esegue i tre controlli principali sul movimento registrato.

        Input importante:
        - `frames`: frame validi del salto, gia' normalizzati;
        - `initial_contact_index`: frame in cui stimiamo l'atterraggio;
        - `max_knee_flexion_index`: frame in cui stimiamo la massima flessione.

        Tutte le soglie sono relative alla larghezza spalle. Cosi' il controllo
        non dipende direttamente dalla scala della webcam.
        """

        # Serie temporale della quota media delle caviglie.
        # In OpenCV/Yolo la y cresce verso il basso:
        # - caviglie piu' alte nello schermo => y piu' piccola;
        # - caviglie piu' basse/atterraggio => y piu' grande.
        ankle_y = np.array([self.ankle_mean_y(frame) for frame in frames])

        # Serie temporale della quota del centro corpo. Serve come secondo
        # segnale per capire se dopo l'atterraggio il corpo si solleva davvero.
        body_y = np.array([body_keypoint(frame.keypoints_xy)[1] for frame in frames])

        # Usiamo la larghezza spalle mediana come "unità" del corpo.
        # Esempio: una soglia 0.15 significa 15% della larghezza spalle.
        
        # Serve per rendere le soglie “relative al corpo”, non ai pixel grezzi della webcam.
        # Se usassimo una soglia fissa tipo “il drop deve essere almeno 40 pixel”, sarebbe sbagliato
        
        reference_width = self.reference_width(frames)

        # CHECK 1 - drop_started_from_height
        #
        # Vogliamo verificare che il soggetto sia partito da un rialzo.
        # Prendiamo:
        # - `start_y`: la mediana delle caviglie prima dell'atterraggio;
        # - `landing_y`: la posizione delle caviglie all'initial contact.
        #
        # Se il soggetto e' sceso dal box, landing_y deve essere abbastanza piu'
        # grande di start_y, perche' le caviglie sono scese nell'immagine.
        landing_y = float(ankle_y[initial_contact_index])
        pre_window = ankle_y[:max(1, initial_contact_index)]
        start_y = float(np.median(pre_window)) if len(pre_window) else float(ankle_y[0])
        drop = landing_y - start_y
        min_drop = self.min_drop_height_ratio * reference_width

        # CHECK 2 - two_foot_contact
        #
        # Al frame di contatto, controlliamo se le due caviglie sono circa alla
        # stessa altezza verticale. Se una caviglia e' molto piu' alta/bassa
        # dell'altra, probabilmente l'atterraggio non e' a due piedi o YOLO ha
        # perso una gamba.
        contact_diff = abs(
            frames[initial_contact_index].keypoints_xy[LEFT_ANKLE][1]
            - frames[initial_contact_index].keypoints_xy[RIGHT_ANKLE][1]
        )
        max_contact_diff = self.max_two_foot_contact_ratio * reference_width

        # CHECK 3 - second_jump
        #
        # Dopo l'atterraggio e la massima flessione, il drop jump prevede un
        # secondo salto. Lo cerchiamo nei frame successivi:
        # - `second_lift`: quanto risalgono le caviglie rispetto al landing;
        # - `body_lift`: quanto risale il centro corpo rispetto al landing.
        #
        # Usiamo il massimo dei due per essere un po' robusti: a volte YOLO
        # vede meglio le caviglie, a volte il centro corpo.
        second_start = max(max_knee_flexion_index, initial_contact_index + 1)
        second_end = min(len(frames), second_start + self.second_jump_window_frames)
        if second_end > second_start:
            second_lift = landing_y - float(np.min(ankle_y[second_start:second_end]))
            body_lift = float(body_y[initial_contact_index] - np.min(body_y[second_start:second_end]))
        else:
            second_lift = 0.0
            body_lift = 0.0
        jump_lift = max(second_lift, body_lift)
        min_second_jump = self.min_second_jump_ratio * reference_width

        checks = [
            ProtocolCheck(
                name="drop_started_from_height",
                passed=bool(drop >= min_drop),
                value=drop,
                threshold=min_drop,
            ),
            ProtocolCheck(
                name="two_foot_contact",
                passed=bool(contact_diff <= max_contact_diff),
                value=contact_diff,
                threshold=max_contact_diff,
            ),
            ProtocolCheck(
                name="second_jump",
                passed=bool(jump_lift >= min_second_jump),
                value=jump_lift,
                threshold=min_second_jump,
            ),
        ]
        return DropJumpProtocolResult(
            passed=all(check.passed for check in checks),
            checks=checks,
        )


    @staticmethod
    def ankle_mean_y(frame: Any) -> float:
        """Media verticale tra caviglia sinistra e destra."""

        k = frame.keypoints_xy
        return float((k[LEFT_ANKLE][1] + k[RIGHT_ANKLE][1]) / 2.0)

    @staticmethod
    def reference_width(frames: list[Any]) -> float:
        """Scala di riferimento: mediana della larghezza spalle nei frame."""

        shoulder_widths = [
            distance(frame.keypoints_xy[LEFT_SHOULDER], frame.keypoints_xy[RIGHT_SHOULDER])
            for frame in frames
        ]
        median_width = float(np.median(shoulder_widths))
        return max(median_width, 1e-6)

    def detect_drop_start(
        self,
        current_ankle_y: float,
        baseline_ankle_y: float,
        body_scale_px: float,
        min_drop_ratio: float,
    ) -> bool:
        """Utility per decidere se una discesa supera la soglia di trigger."""

        return bool(current_ankle_y - baseline_ankle_y >= min_drop_ratio * max(body_scale_px, 1e-6))
