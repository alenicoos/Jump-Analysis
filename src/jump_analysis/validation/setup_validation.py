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
    NOSE,
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

    # Expected vertical segment proportions relative to full standing height.
    # Source: De Leva (1996) body segment parameters.
    # These are shoulder-to-ankle segments only (head above shoulders ~13% excluded).
    SEGMENT_PROPORTIONS: dict[str, float] = {
        "shoulder_to_hip": 0.290,
        "hip_to_knee":     0.245,
        "knee_to_ankle":   0.246,
    }

    def __init__(
        self,
        max_scale_change_ratio: float = 0.10,
        max_camera_roll_degrees: float = 10.0,
        max_pitch_proxy_ratio: float = 0.45,
        min_box_height_ratio: float = 0.05,
        max_segment_ratio_deviation: float = 0.20,
        max_horizontal_offset_ratio: float = 0.15,
        max_body_yaw_ratio: float = 0.20,
    ) -> None:
        self.max_scale_change_ratio = max_scale_change_ratio
        self.max_camera_roll_degrees = max_camera_roll_degrees
        self.max_pitch_proxy_ratio = max_pitch_proxy_ratio
        self.min_box_height_ratio = min_box_height_ratio
        self.max_segment_ratio_deviation = max_segment_ratio_deviation
        self.max_horizontal_offset_ratio = max_horizontal_offset_ratio
        self.max_body_yaw_ratio = max_body_yaw_ratio

    def validate_floor_and_box(
        self,
        floor_keypoints_xy: np.ndarray,
        box_keypoints_xy: np.ndarray,
        height_cm: float,
        floor_body_height_px: float,
        frame_width: int = 0,
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

        # Camera height check: compare observed vertical segment lengths to
        # expected anthropometric proportions. A camera placed too high compresses
        # the lower segments (knee→ankle appears short); too low compresses the
        # upper segments (shoulder→hip appears short). Height in cm is used to
        # anchor the expected proportions to absolute pixel scale.
        segment_deviation, worst_segment, direction_hint = self._camera_height_deviation(
            floor_keypoints_xy, floor_body_height_px
        )
        checks.append(
            SetupCheck(
                name="camera_height_perspective",
                passed=segment_deviation <= self.max_segment_ratio_deviation,
                severity="warning",
                value=segment_deviation,
                threshold=self.max_segment_ratio_deviation,
                message=(
                    f"Camera height may be off: '{worst_segment.replace('_', ' ')}' segment "
                    f"is {segment_deviation * 100:.0f}% from expected proportion. "
                    f"{direction_hint} Aim for hip height."
                ),
            )
        )

        # Horizontal centering: the body midpoint should be near the frame center.
        # Off-center subjects introduce asymmetric perspective on left/right features.
        if frame_width > 0:
            offset_ratio = self._horizontal_offset_ratio(floor_keypoints_xy, frame_width)
            direction = "Move left." if offset_ratio > 0 else "Move right."
            checks.append(
                SetupCheck(
                    name="subject_horizontal_centering",
                    passed=abs(offset_ratio) <= self.max_horizontal_offset_ratio,
                    severity="warning",
                    value=abs(offset_ratio),
                    threshold=self.max_horizontal_offset_ratio,
                    message=(
                        f"Subject is off-center by {abs(offset_ratio) * 100:.0f}% of frame width. "
                        f"{direction}"
                    ),
                )
            )

        # Frontal orientation (yaw): the subject should face the camera directly.
        # Two complementary cues are combined:
        # 1. Nose offset from the shoulder midpoint — if rotated, the nose shifts
        #    toward the side the subject is facing.
        # 2. Left/right shoulder asymmetry around the body centre — one half
        #    appears compressed when the subject is turned.
        yaw_ratio, yaw_direction = self._body_yaw_ratio(floor_keypoints_xy)
        if yaw_ratio is not None:
            checks.append(
                SetupCheck(
                    name="subject_frontal_orientation",
                    passed=yaw_ratio <= self.max_body_yaw_ratio,
                    severity="warning",
                    value=yaw_ratio,
                    threshold=self.max_body_yaw_ratio,
                    message=(
                        f"Subject may be rotated {yaw_direction}. "
                        f"Turn to face the camera directly (yaw score {yaw_ratio:.2f})."
                    ),
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

    def _camera_height_deviation(
        self,
        keypoints_xy: np.ndarray,
        floor_body_height_px: float,
    ) -> tuple[float, str, str]:
        """Compare observed vertical segment ratios to anthropometric expectations.

        Returns (max_deviation, worst_segment_name, direction_hint).

        A camera that is too high makes the lower body appear vertically
        compressed (knee→ankle ratio smaller than expected). A camera that is
        too low compresses the upper body (shoulder→hip ratio smaller).
        """

        shoulder_y = float((keypoints_xy[LEFT_SHOULDER][1] + keypoints_xy[RIGHT_SHOULDER][1]) / 2.0)
        hip_y = float((keypoints_xy[LEFT_HIP][1] + keypoints_xy[RIGHT_HIP][1]) / 2.0)
        knee_y = float((keypoints_xy[LEFT_KNEE][1] + keypoints_xy[RIGHT_KNEE][1]) / 2.0)
        ankle_y = float((keypoints_xy[LEFT_ANKLE][1] + keypoints_xy[RIGHT_ANKLE][1]) / 2.0)

        # y increases downward in OpenCV, so these should all be positive.
        observed_px = {
            "shoulder_to_hip": hip_y - shoulder_y,
            "hip_to_knee":     knee_y - hip_y,
            "knee_to_ankle":   ankle_y - knee_y,
        }

        if any(v <= 0 for v in observed_px.values()) or floor_body_height_px < 1e-6:
            return 0.0, "none", ""

        # Normalise by full body height in pixels so the comparison is
        # scale-independent (works at any camera distance).
        observed_ratio = {k: v / floor_body_height_px for k, v in observed_px.items()}
        deviations = {
            k: (observed_ratio[k] - self.SEGMENT_PROPORTIONS[k]) / self.SEGMENT_PROPORTIONS[k]
            for k in self.SEGMENT_PROPORTIONS
        }

        worst = max(deviations, key=lambda k: abs(deviations[k]))
        max_dev = abs(deviations[worst])

        # Direction hint: negative deviation on a lower segment → camera too high.
        # Negative deviation on an upper segment → camera too low.
        d = deviations[worst]
        if worst == "knee_to_ankle" and d < 0:
            hint = "Move the camera lower."
        elif worst == "shoulder_to_hip" and d < 0:
            hint = "Move the camera higher."
        else:
            hint = "Adjust camera height."

        return max_dev, worst, hint

    def _body_yaw_ratio(self, keypoints_xy: np.ndarray) -> tuple[float | None, str]:
        """Estimate frontal yaw from two complementary 2D cues.

        Cue 1 — nose offset: the nose should sit on the shoulder/hip midline.
        If the subject is rotated, the nose shifts toward the facing direction.
        Expressed as a fraction of shoulder width.

        Cue 2 — shoulder left/right asymmetry: when rotated, one half of the
        shoulder span (body_center→shoulder) appears shorter than the other.
        Asymmetry = |left_half - right_half| / (left_half + right_half).

        Returns the max of the two scores and a direction hint, or (None, "")
        if keypoints are insufficient.
        """

        shoulder_mid_x = float((keypoints_xy[LEFT_SHOULDER][0] + keypoints_xy[RIGHT_SHOULDER][0]) / 2.0)
        hip_mid_x = float((keypoints_xy[LEFT_HIP][0] + keypoints_xy[RIGHT_HIP][0]) / 2.0)
        body_mid_x = (shoulder_mid_x + hip_mid_x) / 2.0
        shoulder_width = float(abs(keypoints_xy[RIGHT_SHOULDER][0] - keypoints_xy[LEFT_SHOULDER][0]))

        if shoulder_width < 1e-6:
            return None, ""

        # Cue 1: nose offset from body midline.
        nose_x = float(keypoints_xy[NOSE][0])
        nose_offset = (nose_x - body_mid_x) / shoulder_width

        # Cue 2: left/right shoulder half-span asymmetry.
        left_half = body_mid_x - float(keypoints_xy[LEFT_SHOULDER][0])
        right_half = float(keypoints_xy[RIGHT_SHOULDER][0]) - body_mid_x
        span_sum = abs(left_half) + abs(right_half)
        shoulder_asymmetry = abs(left_half - right_half) / max(span_sum, 1e-6)

        yaw_score = max(abs(nose_offset), shoulder_asymmetry)

        # Direction: positive nose offset → nose right of midline → subject faces right.
        if abs(nose_offset) >= shoulder_asymmetry:
            direction = "to the right" if nose_offset > 0 else "to the left"
        else:
            direction = "to the right" if right_half < left_half else "to the left"

        return yaw_score, direction

    def _horizontal_offset_ratio(self, keypoints_xy: np.ndarray, frame_width: int) -> float:
        """Signed horizontal offset of body center from frame center.

        Positive = body is to the right of center, negative = to the left.
        Expressed as a fraction of frame width so it is resolution-independent.
        """

        body_keypoints = [
            LEFT_SHOULDER, RIGHT_SHOULDER,
            LEFT_HIP, RIGHT_HIP,
            LEFT_KNEE, RIGHT_KNEE,
            LEFT_ANKLE, RIGHT_ANKLE,
        ]
        xs = keypoints_xy[body_keypoints, 0]
        xs = xs[np.isfinite(xs)]
        if len(xs) == 0:
            return 0.0
        body_center_x = float(np.mean(xs))
        frame_center_x = frame_width / 2.0
        return (body_center_x - frame_center_x) / frame_width

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

    def stable_pose(self) -> np.ndarray | None:
        """Return the frame median if movement is below threshold."""

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
