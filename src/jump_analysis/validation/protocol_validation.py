from __future__ import annotations

"""Drop-jump protocol validation.

This module checks whether the recorded movement has the minimum structure of a
drop jump: start from a box, two-foot landing, and immediate follow-up jump.
It does not assign a LESS score and does not yet evaluate full clinical quality.
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
    """Numeric result of one protocol check."""

    name: str
    passed: bool
    value: float
    threshold: float


@dataclass
class DropJumpProtocolResult:
    """Overall result of drop-jump checks."""

    passed: bool
    checks: list[ProtocolCheck]

    def as_metadata(self) -> dict[str, float | int]:
        """Convert checks to columns that can be saved in the feature CSV."""

        metadata: dict[str, float | int] = {"protocol_passed": int(self.passed)}
        for check in self.checks:
            metadata[f"{check.name}_passed"] = int(check.passed)
            metadata[f"{check.name}_value"] = check.value
            metadata[f"{check.name}_threshold"] = check.threshold
        return metadata


class DropJumpProtocolValidator:
    """Validate the basic drop-jump sequence after setup."""

    def __init__(
        self,
        min_drop_height_ratio: float = 0.15,
        min_second_jump_ratio: float = 0.12,
        second_jump_window_frames: int = 60,
        max_post_landing_fall_ratio: float = 0.25,
    ) -> None:
        self.min_drop_height_ratio = min_drop_height_ratio
        self.min_second_jump_ratio = min_second_jump_ratio
        self.second_jump_window_frames = second_jump_window_frames
        self.max_post_landing_fall_ratio = max_post_landing_fall_ratio

    def validate(
        self,
        frames: list[Any],
        initial_contact_index: int,
        max_knee_flexion_index: int,
    ) -> DropJumpProtocolResult:
        """Run the main checks on the recorded movement.

        Important inputs:
        - `frames`: valid jump frames, already normalized;
        - `initial_contact_index`: estimated landing frame;
        - `max_knee_flexion_index`: estimated maximum-flexion frame.

        All thresholds are relative to shoulder width, so the check does not
        depend directly on webcam pixel scale.
        """

        # Time series of average ankle height. In OpenCV/YOLO, y grows downward:
        # - ankles higher on screen => smaller y;
        # - ankles lower/landing => larger y.
        ankle_y = np.array([self.ankle_mean_y(frame) for frame in frames])

        # Body-center height time series. This is a second cue for checking
        # whether the body really rises after landing.
        body_y = np.array([body_keypoint(frame.keypoints_xy)[1] for frame in frames])

        # Median shoulder width is the body-relative unit. For example, a 0.15
        # threshold means 15% of shoulder width. This keeps thresholds
        # body-relative rather than raw webcam pixels. A
        # fixed threshold such as "the drop must be at least 40 pixels" would be wrong.
        reference_width = self.reference_width(frames)

        # CHECK 1 - drop_started_from_height
        #
        # Check whether the subject started from a box.
        #
        # First use the trigger measured during capture: recording starts only
        # when ankles move downward from the stable box baseline. This is more
        # reliable than reconstructing the whole box-to-floor drop from the
        # saved frames, because saved frames intentionally start at jump start.
        #
        # As fallback, estimate the drop from the saved sequence:
        # - `start_y`: median ankle position before landing;
        # - `landing_y`: ankle position at initial contact.
        landing_y = float(ankle_y[initial_contact_index])
        pre_window = ankle_y[:max(1, initial_contact_index)]
        start_y = float(np.median(pre_window)) if len(pre_window) else float(ankle_y[0])
        frame_drop = landing_y - start_y
        trigger_drop = self.drop_trigger_value(frames)
        trigger_threshold = self.required_drop_value(frames)
        if trigger_drop is not None and trigger_threshold is not None:
            drop = trigger_drop
            min_drop = trigger_threshold
        else:
            drop = frame_drop
            min_drop = self.min_drop_height_ratio * reference_width

        # CHECK 2 - second_jump
        #
        # After landing and maximum flexion, the drop jump requires a second
        # jump. Search in following frames:
        # - `second_lift`: how much ankles rise relative to landing;
        # - `body_lift`: how much body center rises relative to landing.
        #
        # Use the maximum of the two for robustness: sometimes YOLO sees ankles
        # better, sometimes the body center.
        second_start = max(max_knee_flexion_index, initial_contact_index + 1)
        second_end = min(len(frames), second_start + self.second_jump_window_frames)
        if second_end > second_start:
            second_window_ankle = ankle_y[second_start:second_end]
            second_window_body = body_y[second_start:second_end]
            second_lift = landing_y - float(np.min(second_window_ankle))
            body_lift = float(body_y[initial_contact_index] - np.min(second_window_body))
            second_takeoff_index = second_start + int(np.argmin(second_window_ankle))
        else:
            second_lift = 0.0
            body_lift = 0.0
            second_takeoff_index = second_start
        jump_lift = max(second_lift, body_lift)
        min_second_jump = self.min_second_jump_ratio * reference_width

        # CHECK 3 - stable_after_second_landing
        #
        # After the second jump, find the next landing as the first frame after
        # takeoff where ankles return close to the first landing level. Then
        # verify that the body/ankles do not keep dropping substantially, which
        # would suggest the subject lost balance or fell.
        second_landing_index = self.find_second_landing_index(
            ankle_y=ankle_y,
            landing_y=landing_y,
            second_takeoff_index=second_takeoff_index,
            reference_width=reference_width,
        )
        max_fall = self.max_post_landing_fall_ratio * reference_width
        if second_landing_index is not None:
            post_ankle = ankle_y[second_landing_index:]
            post_body = body_y[second_landing_index:]
            ankle_fall = float(np.max(post_ankle) - ankle_y[second_landing_index]) if len(post_ankle) else 0.0
            body_fall = float(np.max(post_body) - body_y[second_landing_index]) if len(post_body) else 0.0
            post_landing_fall = max(ankle_fall, body_fall)
        else:
            post_landing_fall = float("inf")

        checks = [
            ProtocolCheck(
                name="drop_started_from_height",
                passed=bool(drop >= min_drop),
                value=drop,
                threshold=min_drop,
            ),
            ProtocolCheck(
                name="second_jump",
                passed=bool(jump_lift >= min_second_jump),
                value=jump_lift,
                threshold=min_second_jump,
            ),
            ProtocolCheck(
                name="stable_after_second_landing",
                passed=bool(post_landing_fall <= max_fall),
                value=post_landing_fall,
                threshold=max_fall,
            ),
        ]
        return DropJumpProtocolResult(
            passed=all(check.passed for check in checks),
            checks=checks,
        )


    @staticmethod
    def drop_trigger_value(frames: list[Any]) -> float | None:
        """Return the capture-time drop trigger value when available."""

        for frame in frames:
            value = getattr(frame, "drop_trigger_px", None)
            if value is not None:
                return float(value)
        return None

    @staticmethod
    def required_drop_value(frames: list[Any]) -> float | None:
        """Return the capture-time required drop threshold when available."""

        for frame in frames:
            value = getattr(frame, "required_drop_px", None)
            if value is not None:
                return float(value)
        return None

    @staticmethod
    def ankle_mean_y(frame: Any) -> float:
        """Average vertical coordinate between left and right ankles."""

        k = frame.keypoints_xy
        return float((k[LEFT_ANKLE][1] + k[RIGHT_ANKLE][1]) / 2.0)

    @staticmethod
    def reference_width(frames: list[Any]) -> float:
        """Reference scale: median shoulder width across frames."""

        shoulder_widths = [
            distance(frame.keypoints_xy[LEFT_SHOULDER], frame.keypoints_xy[RIGHT_SHOULDER])
            for frame in frames
        ]
        median_width = float(np.median(shoulder_widths))
        return max(median_width, 1e-6)

    def find_second_landing_index(
        self,
        ankle_y: np.ndarray,
        landing_y: float,
        second_takeoff_index: int,
        reference_width: float,
    ) -> int | None:
        """Find the second landing after the rebound jump."""

        if second_takeoff_index >= len(ankle_y) - 2:
            return None
        landing_tolerance = 0.10 * reference_width
        search = ankle_y[second_takeoff_index + 1 :]
        candidates = np.where(search >= landing_y - landing_tolerance)[0]
        if len(candidates) == 0:
            return None
        return second_takeoff_index + 1 + int(candidates[0])

    def detect_drop_start(
        self,
        current_ankle_y: float,
        baseline_ankle_y: float,
        body_scale_px: float,
        min_drop_ratio: float,
    ) -> bool:
        """Return whether a descent exceeds the trigger threshold."""

        return bool(current_ankle_y - baseline_ankle_y >= min_drop_ratio * max(body_scale_px, 1e-6))
