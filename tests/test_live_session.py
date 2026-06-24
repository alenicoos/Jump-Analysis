from __future__ import annotations

import time
import unittest

import numpy as np

from jump_analysis.live_session import LIVE_MIN_RECORDING_FRAMES, LiveGuidanceProcessor
from jump_analysis.video.yolo_video import YoloPoseFrame


def _placeholder_frame() -> YoloPoseFrame:
    return YoloPoseFrame(
        frame_index=0,
        keypoints_xy=np.zeros((17, 2), dtype=float),
        keypoints_conf=np.ones(17, dtype=float),
        box_xyxy=None,
    )


class LiveSessionTests(unittest.TestCase):
    def test_recording_completion_uses_partial_pose_when_ankles_remain_visible(self) -> None:
        processor = LiveGuidanceProcessor(
            analysis_service=None,
            model=None,
            height_cm=180.0,
        )
        processor.phase = "recording"
        processor.recording_started_monotonic = time.monotonic() - 2.0
        processor.recording_frame_count = LIVE_MIN_RECORDING_FRAMES
        processor.frames = [_placeholder_frame() for _ in range(12)]
        processor.first_landing_y = 100.0
        processor.second_takeoff_seen = True
        processor.second_landing_seen = True
        processor.second_landing_y = 100.0
        processor.post_landing_feet_y.extend([99.0, 100.0, 100.0, 101.0])

        conf = np.zeros(17, dtype=float)
        conf[15] = 0.95
        conf[16] = 0.95

        self.assertTrue(processor._can_track_recording_frame(conf, np.array([0.0, 0.0, 10.0, 120.0])))

        processor._update_jump_completion(feet_y=100.0, body_height=120.0)
        events = processor._recording_guidance_events(
            "Recording. Keep both ankles visible while you perform the rebound jump.",
            level="warning",
        )

        self.assertEqual(processor.phase, "analyzing")
        self.assertEqual(events[0].type, "guidance")
        self.assertEqual(events[1].type, "stop_streaming")

    def test_rebound_tracking_starts_soon_after_drop(self) -> None:
        processor = LiveGuidanceProcessor(
            analysis_service=None,
            model=None,
            height_cm=180.0,
        )
        processor.phase = "recording"
        processor.recording_started_monotonic = time.monotonic() - 0.1
        processor.recording_frame_count = LIVE_MIN_RECORDING_FRAMES

        processor._update_jump_completion(feet_y=100.0, body_height=120.0)
        self.assertEqual(processor.first_landing_y, 100.0)

    def test_post_landing_stability_uses_recent_window(self) -> None:
        processor = LiveGuidanceProcessor(
            analysis_service=None,
            model=None,
            height_cm=180.0,
        )
        processor.phase = "recording"
        processor.recording_started_monotonic = time.monotonic() - 1.0
        processor.recording_frame_count = LIVE_MIN_RECORDING_FRAMES
        processor.frames = [_placeholder_frame() for _ in range(12)]
        processor.first_landing_y = 100.0
        processor.second_takeoff_seen = True
        processor.second_landing_seen = True
        processor.second_landing_y = 95.0

        for feet_y in (102.0, 104.0, 103.0, 104.0, 103.0):
            processor.recording_frame_count += 1
            processor._update_jump_completion(feet_y=feet_y, body_height=120.0)

        self.assertEqual(processor.phase, "completed")


if __name__ == "__main__":
    unittest.main()
