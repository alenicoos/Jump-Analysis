from __future__ import annotations

"""Setup validation before acquiring a drop jump.

This module checks whether the initial calibration is reliable enough:
- stable floor pose;
- stable box pose;
- user did not move too much toward/away from the camera;
- camera is not too rotated;
- ankles are clearly higher on the box than on the floor.
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
    """Result of one setup check."""

    name: str
    passed: bool
    severity: str
    value: float
    threshold: float
    message: str


@dataclass
class CalibrationPose:
    """Light wrapper around calibration-pose keypoints."""

    keypoints_xy: np.ndarray

    @property
    def ankle_y(self) -> float:
        """Average vertical coordinate of the ankles."""

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
        """Body scale used for relative pixel thresholds.
        
        If shoulder width cannot be measured, hip width is used as fallback.
        """

        if self.shoulder_width > 1e-6:
            return self.shoulder_width
        return self.hip_width


@dataclass
class SetupCalibration:
    """Final data produced by floor/box setup."""

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
    """Overall setup validation result."""

    passed: bool
    messages: list[str]
    checks: list[SetupCheck]
    calibration: SetupCalibration | None = None


class SetupValidator:
    """Validate camera and setup before drop-jump acquisition."""

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
        """Compare floor pose and box pose.

        The central check is vertical ankle difference. Body-scale change is
        also used to catch a common failure mode: the user moves closer to the
        camera and appears "higher" without actually stepping onto the box.
        """

        floor_pose = CalibrationPose(floor_keypoints_xy)
        box_pose = CalibrationPose(box_keypoints_xy)
        checks: list[SetupCheck] = []

        # Camera roll: if shoulders/hips/knees/ankles are strongly tilted, the
        # phone/camera is probably not straight.
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

        # Perspective proxy: if shoulder, hip, knee, and ankle widths differ
        # strongly, the camera may be too high/low or too close.
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

        # If the person moves toward/away from the camera between floor and box
        # pose, box-height estimation becomes unreliable.
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

        # In OpenCV, y increases downward. On the box, ankles should have a
        # lower y value, so floor_y - box_y should be positive.
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

        # Pixel-to-meter conversion: during floor setup we measure body height
        # in pixels and know the user's declared real height. The same scale is
        # then used for shoulder width and box height.
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
        """Maximum tilt among segments that should be horizontal."""

        tilts = [
            self._line_tilt_degrees(keypoints_xy[LEFT_SHOULDER], keypoints_xy[RIGHT_SHOULDER]),
            self._line_tilt_degrees(keypoints_xy[LEFT_HIP], keypoints_xy[RIGHT_HIP]),
            self._line_tilt_degrees(keypoints_xy[LEFT_KNEE], keypoints_xy[RIGHT_KNEE]),
            self._line_tilt_degrees(keypoints_xy[LEFT_ANKLE], keypoints_xy[RIGHT_ANKLE]),
        ]
        return max(abs(value) for value in tilts if not math.isnan(value))

    def _line_tilt_degrees(self, left: np.ndarray, right: np.ndarray) -> float:
        """Tilt in degrees of a left-right line."""

        delta = right - left
        if np.linalg.norm(delta) == 0:
            return float("nan")
        return math.degrees(math.atan2(float(delta[1]), float(delta[0])))

    def _pitch_proxy_ratio(self, pose: CalibrationPose) -> float:
        """Loose proxy for high/low camera perspective.

        This does not correct distortion. It only flags when body widths change
        too much across horizontal segments.
        """

        widths = np.array([pose.shoulder_width, pose.hip_width, pose.knee_width, pose.ankle_width], dtype=float)
        widths = widths[np.isfinite(widths) & (widths > 1e-6)]
        if len(widths) < 2:
            return 0.0
        return float((widths.max() - widths.min()) / max(np.median(widths), 1e-6))


class StablePoseBuffer:
    """Accumulate frames and return a pose only when the user is still."""

    def __init__(self, maxlen: int = 30, min_frames: int = 12, max_motion_ratio: float = 0.025) -> None:
        self.maxlen = maxlen
        self.min_frames = min_frames
        self.max_motion_ratio = max_motion_ratio
        self._items: deque[np.ndarray] = deque(maxlen=maxlen)

    def clear(self) -> None:
        """Clear the buffer when the pose becomes incomplete or changes too much."""

        self._items.clear()

    def add(self, keypoints_xy: np.ndarray) -> None:
        """Add a copy of keypoints to avoid accidental mutation."""

        self._items.append(np.asarray(keypoints_xy, dtype=float).copy())

    def size(self) -> int:
        """Return the number of buffered frames."""

        return len(self._items)

    def stability_snapshot(self) -> dict[str, float | int | None]:
        """Return current stability metrics for logging/debugging."""

        snapshot: dict[str, float | int | None] = {
            "buffered_frames": len(self._items),
            "min_frames": self.min_frames,
            "motion_px": None,
            "motion_threshold_px": None,
            "body_scale_px": None,
        }
        if len(self._items) < self.min_frames:
            return snapshot

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
        snapshot["motion_px"] = motion_px
        snapshot["body_scale_px"] = body_scale
        snapshot["motion_threshold_px"] = self.max_motion_ratio * max(body_scale, 1e-6)
        return snapshot

    def stable_pose(self) -> np.ndarray | None:
        """Return the frame median if movement is below threshold."""

        snapshot = self.stability_snapshot()
        if snapshot["buffered_frames"] < self.min_frames:
            return None

        stack = np.stack(list(self._items), axis=0)
        median_pose = np.median(stack, axis=0)
        motion_px = float(snapshot["motion_px"] or 0.0)
        motion_threshold_px = float(snapshot["motion_threshold_px"] or 0.0)
        if motion_px <= motion_threshold_px:
            return median_pose
        return None
