from __future__ import annotations

"""Setup validation before acquiring a drop jump.

Qui controlliamo solo se la calibrazione iniziale e' abbastanza affidabile:
- posa a terra stabile;
- posa sul box stabile;
- utente non troppo piu' vicino/lontano dalla camera;
- camera non troppo ruotata;
- caviglie chiaramente piu' alte sul box che a terra.
"""

import math
from dataclasses import dataclass
from collections import deque

import numpy as np
from jump_analysis.features.front_2d_features import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    distance,
)


@dataclass
class SetupCheck:
    """Risultato di un singolo controllo del setup."""

    name: str
    passed: bool
    severity: str
    value: float
    threshold: float
    message: str


@dataclass
class CalibrationPose:
    """Wrapper leggero intorno ai keypoint di una posa di calibrazione."""

    keypoints_xy: np.ndarray

    @property
    def ankle_y(self) -> float:
        """Coordinata verticale media delle caviglie."""

        return float((self.keypoints_xy[LEFT_ANKLE][1] + self.keypoints_xy[RIGHT_ANKLE][1]) / 2.0)

    @property
    def shoulder_width(self) -> float:
        return distance(self.keypoints_xy[LEFT_SHOULDER], self.keypoints_xy[RIGHT_SHOULDER])

    @property
    def hip_width(self) -> float:
        return distance(self.keypoints_xy[LEFT_HIP], self.keypoints_xy[RIGHT_HIP])

    @property
    def knee_width(self) -> float:
        return distance(self.keypoints_xy[LEFT_KNEE], self.keypoints_xy[RIGHT_KNEE])

    @property
    def ankle_width(self) -> float:
        return distance(self.keypoints_xy[LEFT_ANKLE], self.keypoints_xy[RIGHT_ANKLE])

    @property
    def body_scale(self) -> float:
        """Scala del corpo usata per soglie relative in pixel.
        
        Se non riusciamo a misurare la larghezza delle spalle,
        usiamo quella delle anche come fallback.
        """

        if self.shoulder_width > 1e-6:
            return self.shoulder_width
        return self.hip_width


@dataclass
class SetupCalibration:
    """Dati finali prodotti dal setup terra/box."""

    floor_pose: CalibrationPose
    box_pose: CalibrationPose
    floor_body_height_px: float
    meters_per_pixel: float
    measured_shoulder_width_m: float
    box_height_px: float
    scale_change_ratio: float
    camera_roll_degrees: float
    pitch_proxy_ratio: float
    estimated_box_height_cm: float | None = None


@dataclass
class SetupValidationResult:
    """Risultato complessivo del setup."""

    passed: bool
    messages: list[str]
    checks: list[SetupCheck]
    calibration: SetupCalibration | None = None


class SetupValidator:
    """Valida camera e setup prima dell'acquisizione del drop jump."""

    def __init__(
        self,
        max_scale_change_ratio: float = 0.10,
        max_camera_roll_degrees: float = 10.0,
        max_pitch_proxy_ratio: float = 0.45,
        min_box_height_ratio: float = 0.05,
    ) -> None:
        self.max_scale_change_ratio = max_scale_change_ratio
        self.max_camera_roll_degrees = max_camera_roll_degrees
        self.max_pitch_proxy_ratio = max_pitch_proxy_ratio
        self.min_box_height_ratio = min_box_height_ratio

    def validate_floor_and_box(
        self,
        floor_keypoints_xy: np.ndarray,
        box_keypoints_xy: np.ndarray,
        height_cm: float,
        floor_body_height_px: float,
    ) -> SetupValidationResult:
        """Confronta posa a terra e posa sul box.

        Il controllo centrale e' la differenza verticale delle caviglie.
        Usiamo anche la variazione di scala del corpo per intercettare un caso
        comune: l'utente si avvicina alla camera e sembra "piu' alto" anche se
        non e' davvero salito sul box.
        """

        floor_pose = CalibrationPose(floor_keypoints_xy)
        box_pose = CalibrationPose(box_keypoints_xy)
        checks: list[SetupCheck] = []

        # Roll della camera: se spalle/anche/ginocchia/caviglie risultano molto
        # inclinate, probabilmente il telefono non e' dritto.
        floor_roll = self._max_horizontal_tilt(floor_keypoints_xy)
        checks.append(
            SetupCheck(
                name="camera_roll",
                passed=floor_roll <= self.max_camera_roll_degrees,
                severity="warning",
                value=floor_roll,
                threshold=self.max_camera_roll_degrees,
                message="Camera may be rotated; horizontal body landmarks are tilted.",
            )
        )

        # Proxy di prospettiva: se larghezza spalle, anche, ginocchia e caviglie
        # cambiano molto tra loro, la camera potrebbe essere troppo alta/bassa
        # o troppo vicina.
        pitch_proxy = self._pitch_proxy_ratio(floor_pose)
        checks.append(
            SetupCheck(
                name="camera_pitch_or_perspective",
                passed=pitch_proxy <= self.max_pitch_proxy_ratio,
                severity="warning",
                value=pitch_proxy,
                threshold=self.max_pitch_proxy_ratio,
                message="Camera may be too high/low or subject may be too close; body widths change strongly with height.",
            )
        )

        # Se la persona si avvicina o si allontana tra posa a terra e posa sul
        # box, la stima dell'altezza del box diventa poco affidabile.
        scale_change = abs(box_pose.body_scale - floor_pose.body_scale) / max(floor_pose.body_scale, 1e-6)
        checks.append(
            SetupCheck(
                name="floor_box_scale_stability",
                passed=scale_change <= self.max_scale_change_ratio,
                severity="error",
                value=scale_change,
                threshold=self.max_scale_change_ratio,
                message="User moved toward/away from camera between floor and box setup.",
            )
        )

        # In OpenCV y aumenta verso il basso: sul box le caviglie devono avere
        # y minore, quindi floor_y - box_y deve essere positivo.
        box_height_px = floor_pose.ankle_y - box_pose.ankle_y
        min_box_height_px = self.min_box_height_ratio * max(floor_pose.body_scale, 1e-6)
        checks.append(
            SetupCheck(
                name="box_height_detected",
                passed=box_height_px > min_box_height_px,
                severity="error",
                value=box_height_px,
                threshold=min_box_height_px,
                message="Ankles are not clearly higher on the box than on the floor.",
            )
        )

        # Conversione pixel->metri: durante il setup a terra misuriamo
        # l'altezza della persona in pixel con `estimate_person_height_px` e
        # sappiamo l'altezza reale inserita dall'utente. Questa scala viene poi
        # usata anche per la larghezza spalle e per l'altezza del box.
        meters_per_pixel = (height_cm / 100.0) / max(floor_body_height_px, 1e-6)
        measured_shoulder_width_m = floor_pose.shoulder_width * meters_per_pixel
        estimated_cm = box_height_px * meters_per_pixel * 100.0

        calibration = SetupCalibration(
            floor_pose=floor_pose,
            box_pose=box_pose,
            floor_body_height_px=floor_body_height_px,
            meters_per_pixel=meters_per_pixel,
            measured_shoulder_width_m=measured_shoulder_width_m,
            box_height_px=box_height_px,
            scale_change_ratio=scale_change,
            camera_roll_degrees=floor_roll,
            pitch_proxy_ratio=pitch_proxy,
            estimated_box_height_cm=estimated_cm,
        )
        messages = [
            f"{check.severity.upper()} {check.name}: {check.message} "
            f"(value={check.value:.3f}, threshold={check.threshold:.3f})"
            for check in checks
            if not check.passed
        ]
        passed = all(check.passed for check in checks if check.severity == "error")
        return SetupValidationResult(passed=passed, messages=messages, checks=checks, calibration=calibration)

    def _max_horizontal_tilt(self, keypoints_xy: np.ndarray) -> float:
        """Massima inclinazione tra segmenti che dovrebbero essere orizzontali."""

        tilts = [
            self._line_tilt_degrees(keypoints_xy[LEFT_SHOULDER], keypoints_xy[RIGHT_SHOULDER]),
            self._line_tilt_degrees(keypoints_xy[LEFT_HIP], keypoints_xy[RIGHT_HIP]),
            self._line_tilt_degrees(keypoints_xy[LEFT_KNEE], keypoints_xy[RIGHT_KNEE]),
            self._line_tilt_degrees(keypoints_xy[LEFT_ANKLE], keypoints_xy[RIGHT_ANKLE]),
        ]
        return max(abs(value) for value in tilts if not math.isnan(value))

    def _line_tilt_degrees(self, left: np.ndarray, right: np.ndarray) -> float:
        """Inclinazione in gradi della linea sinistra-destra."""

        delta = right - left
        if np.linalg.norm(delta) == 0:
            return float("nan")
        return math.degrees(math.atan2(float(delta[1]), float(delta[0])))

    def _pitch_proxy_ratio(self, pose: CalibrationPose) -> float:
        """Proxy non rigoroso per prospettiva alto/basso.

        Non corregge la distorsione; segnala solo quando le larghezze del corpo
        cambiano troppo tra segmenti orizzontali.
        """

        widths = np.array([pose.shoulder_width, pose.hip_width, pose.knee_width, pose.ankle_width], dtype=float)
        widths = widths[np.isfinite(widths) & (widths > 1e-6)]
        if len(widths) < 2:
            return 0.0
        return float((widths.max() - widths.min()) / max(np.median(widths), 1e-6))


class StablePoseBuffer:
    """Accumula frame e restituisce una posa solo quando l'utente e' fermo."""

    def __init__(self, maxlen: int = 30, min_frames: int = 12, max_motion_ratio: float = 0.025) -> None:
        self.maxlen = maxlen
        self.min_frames = min_frames
        self.max_motion_ratio = max_motion_ratio
        self._items: deque[np.ndarray] = deque(maxlen=maxlen)

    def clear(self) -> None:
        """Svuota il buffer quando la posa diventa incompleta o cambia troppo."""

        self._items.clear()

    def add(self, keypoints_xy: np.ndarray) -> None:
        """Aggiunge una copia dei keypoint per evitare modifiche accidentali."""

        self._items.append(np.asarray(keypoints_xy, dtype=float).copy())

    def stable_pose(self) -> np.ndarray | None:
        """Restituisce la mediana dei frame se il movimento e' sotto soglia."""

        if len(self._items) < self.min_frames:
            return None

        stack = np.stack(list(self._items), axis=0)
        median_pose = np.median(stack, axis=0)
        tracked = [
            LEFT_SHOULDER,
            RIGHT_SHOULDER,
            LEFT_HIP,
            RIGHT_HIP,
            LEFT_KNEE,
            RIGHT_KNEE,
            LEFT_ANKLE,
            RIGHT_ANKLE,
        ]
        motion = np.linalg.norm(stack[:, tracked, :] - median_pose[tracked], axis=2)
        motion_px = float(np.nanpercentile(motion, 90))
        body_scale = CalibrationPose(median_pose).body_scale
        if motion_px <= self.max_motion_ratio * max(body_scale, 1e-6):
            return median_pose
        return None
