from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from jump_analysis.validation.protocol_validation import DropJumpProtocolValidator
from jump_analysis.video.yolo_video import YoloPoseFrame


def _frame(
    left_ankle_y: float,
    right_ankle_y: float,
    body_y: float,
    *,
    takeoff: bool = False,
    landing: bool = False,
) -> YoloPoseFrame:
    kpts = np.zeros((17, 2), dtype=float)
    kpts[5] = [0.0, body_y - 20.0]
    kpts[6] = [100.0, body_y - 20.0]
    kpts[15] = [40.0, left_ankle_y]
    kpts[16] = [60.0, right_ankle_y]
    return YoloPoseFrame(
        frame_index=0,
        keypoints_xy=kpts,
        keypoints_conf=np.ones(17, dtype=float),
        box_xyxy=None,
        live_second_takeoff_hint=takeoff,
        live_second_landing_hint=landing,
    )


class ProtocolValidationTests(unittest.TestCase):
    def test_live_second_landing_hint_prevents_infinite_post_landing_fall(self) -> None:
        frames = [
            _frame(500.0, 500.0, 300.0),
            _frame(560.0, 558.0, 340.0),
            _frame(520.0, 522.0, 310.0, takeoff=True),
            _frame(565.0, 566.0, 345.0, landing=True),
            _frame(566.0, 565.0, 346.0),
        ]
        validator = DropJumpProtocolValidator()

        result = validator.validate(frames, initial_contact_index=1, max_knee_flexion_index=1)
        checks = {check.name: check for check in result.checks}

        self.assertTrue(checks["second_jump"].passed)
        self.assertTrue(np.isfinite(checks["stable_after_second_landing"].value))


if __name__ == "__main__":
    unittest.main()
