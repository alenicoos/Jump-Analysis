from __future__ import annotations

import base64
import json
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from jump_analysis.features.front_2d_features import LEFT_ANKLE, RIGHT_ANKLE
from jump_analysis.service import VideoAnalysisService
from jump_analysis.validation import StablePoseBuffer, SetupValidator
from jump_analysis.video.yolo_video import (
    YoloPoseFrame,
    estimate_person_height_px,
    full_body_box_visible,
    normalize_keypoints_to_mocap_scale,
    required_visible,
    select_pose,
)


@dataclass
class LiveGuidanceEvent:
    type: str
    phase: str
    text: str
    level: str = "info"
    speak: bool = True
    payload: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        body = {
            "type": self.type,
            "phase": self.phase,
            "text": self.text,
            "level": self.level,
            "speak": self.speak,
        }
        if self.payload is not None:
            body["payload"] = self.payload
        return body


class LiveGuidanceProcessor:
    def __init__(
        self,
        analysis_service: VideoAnalysisService,
        model: YOLO,
        height_cm: float,
    ) -> None:
        self.analysis_service = analysis_service
        self.model = model
        self.height_cm = height_cm

        self.phase = "floor_setup"
        self.locked_track_id: int | None = None
        self.floor_buffer = StablePoseBuffer(min_frames=12)
        self.box_buffer = StablePoseBuffer(min_frames=12)
        self.floor_body_height_buffer: deque[float] = deque(maxlen=12)
        self.prep_feet_y: deque[float] = deque(maxlen=45)
        self.prep_body_height: deque[float] = deque(maxlen=45)
        self.frames: list[YoloPoseFrame] = []
        self.validator = SetupValidator()

        self.floor_pose: np.ndarray | None = None
        self.floor_body_height_px: float | None = None
        self.setup_calibration = None
        self.recording_started_monotonic: float | None = None
        self.session_started_at = datetime.now(UTC)
        self.last_spoken_by_message_key: dict[tuple[str, str], float] = {}
        self.last_analysis_payload: dict[str, Any] | None = None
        self.analysis_started = False
        self.analysis_completed = False

        self.first_landing_y: float | None = None
        self.second_landing_y: float | None = None
        self.second_takeoff_seen = False
        self.second_landing_seen = False
        self.stable_after_landing_count = 0
        self.frame_index = 0

    def process_text_message(self, message_text: str) -> list[dict[str, Any]]:
        try:
            payload = json.loads(message_text)
        except json.JSONDecodeError:
            return [self._event("error", "invalid_message", "Received malformed live session payload.", level="error").as_dict()]

        message_type = payload.get("type")
        if message_type == "stop":
            self.phase = "stopped"
            return [self._event("status", self.phase, "Live guidance stopped.", speak=False).as_dict()]
        if message_type != "frame":
            return []

        image_base64 = payload.get("image_base64")
        if not isinstance(image_base64, str) or not image_base64:
            return [self._event("error", self.phase, "Missing frame image data.", level="error").as_dict()]

        try:
            jpeg_bytes = base64.b64decode(image_base64)
        except ValueError:
            return [self._event("error", self.phase, "Could not decode streamed frame.", level="error").as_dict()]

        return [event.as_dict() for event in self.process_jpeg_frame(jpeg_bytes)]

    def process_jpeg_frame(self, jpeg_bytes: bytes) -> list[LiveGuidanceEvent]:
        decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            return [self._event("error", self.phase, "Could not decode streamed frame.", level="error")]

        now = time.monotonic()
        result = self.model.track(decoded, stream=False, persist=True, verbose=False)[0]
        selection = select_pose(result, self.locked_track_id)
        self.frame_index += 1

        if self.phase == "completed":
            events = [self._event("status", self.phase, "Live analysis already completed.", speak=False)]
            if self.last_analysis_payload is not None:
                events.append(
                    LiveGuidanceEvent(
                        "analysis_result",
                        self.phase,
                        "Live analysis result available.",
                        speak=False,
                        payload=self.last_analysis_payload,
                    )
                )
            return events
        if self.phase == "analyzing":
            return [self._event("status", self.phase, "I captured the jump. Hold still while I analyze it.", speak=False)]

        if selection is None:
            self._clear_setup_buffers_if_needed()
            return [self._event("guidance", self.phase, "I cannot see a person. Step into the frame and show your full body.", level="warning")]

        _, detected_track_id, kpts_xy, kpts_conf, box = selection
        if self.locked_track_id is None and detected_track_id is not None:
            self.locked_track_id = detected_track_id

        if not required_visible(kpts_conf):
            self._clear_setup_buffers_if_needed()
            return [self._event("guidance", self.phase, "Keep shoulders, hips, knees, and ankles visible before you continue.", level="warning")]

        if not full_body_box_visible(box, decoded.shape):
            self._clear_setup_buffers_if_needed()
            return [self._event("guidance", self.phase, "Move back until your whole body fits inside the frame.", level="warning")]

        feet_y = float((kpts_xy[LEFT_ANKLE][1] + kpts_xy[RIGHT_ANKLE][1]) / 2.0)
        body_height = estimate_person_height_px(box, kpts_xy)

        if self.floor_pose is None:
            self.floor_buffer.add(kpts_xy)
            self.floor_body_height_buffer.append(body_height)
            stable = self.floor_buffer.stable_pose()
            if stable is None:
                return [self._event("guidance", self.phase, "Stand on the floor facing the camera and stay still.", speak=False)]

            self.floor_pose = stable
            self.floor_body_height_px = float(np.median(self.floor_body_height_buffer))
            self.phase = "box_setup"
            return [self._event("guidance", self.phase, "Good. Now step onto the box and stay still.", level="success")]

        if self.setup_calibration is None:
            floor_ankle_y = float((self.floor_pose[LEFT_ANKLE][1] + self.floor_pose[RIGHT_ANKLE][1]) / 2.0)
            floor_scale = max(float(np.linalg.norm(self.floor_pose[5] - self.floor_pose[6])), 1.0)
            min_box_height_px = 0.05 * floor_scale
            box_height_px = floor_ankle_y - feet_y

            if box_height_px <= min_box_height_px:
                self.box_buffer.clear()
                return [self._event("guidance", self.phase, "Step onto the box and keep both feet clearly higher than the floor pose.", speak=False)]

            self.box_buffer.add(kpts_xy)
            stable_box = self.box_buffer.stable_pose()
            if stable_box is None:
                return [self._event("guidance", self.phase, "Stay still on the box for setup.", speak=False)]

            validation = self.validator.validate_floor_and_box(
                self.floor_pose,
                stable_box,
                height_cm=self.height_cm,
                floor_body_height_px=self.floor_body_height_px or body_height,
            )
            if not validation.passed or validation.calibration is None:
                self.box_buffer.clear()
                message = validation.messages[0] if validation.messages else "Setup validation failed. Reposition yourself and try again."
                return [self._event("guidance", self.phase, message, level="error")]

            self.setup_calibration = validation.calibration
            self.phase = "armed"
            self.prep_feet_y.clear()
            self.prep_body_height.clear()
            return [self._event("guidance", self.phase, "Setup complete. Stay still on the box. Recording starts when you drop.", level="success")]

        normalized = normalize_keypoints_to_mocap_scale(kpts_xy, self.setup_calibration.measured_shoulder_width_m)
        if self.recording_started_monotonic is None:
            self.prep_feet_y.append(feet_y)
            self.prep_body_height.append(body_height)
            if len(self.prep_feet_y) < 8:
                return [self._event("guidance", self.phase, "Hold still on the box. I am arming the recording.", speak=False)]

            baseline = float(np.median(self.prep_feet_y))
            required_drop_px = 0.06 * float(np.median(self.prep_body_height))
            current_drop_px = feet_y - baseline
            if len(self.prep_feet_y) >= 30 and current_drop_px > required_drop_px:
                self.recording_started_monotonic = now
                self.phase = "recording"
                self.frames = [
                    YoloPoseFrame(
                        self.frame_index,
                        normalized,
                        kpts_conf,
                        box,
                        timestamp_s=now,
                        raw_keypoints_xy=kpts_xy.copy(),
                        drop_trigger_px=current_drop_px,
                        required_drop_px=required_drop_px,
                    )
                ]
                return [self._event("guidance", self.phase, "Drop detected. Land and immediately jump again.", level="success")]
            return [self._event("guidance", self.phase, "Ready when you drop.", speak=False)]

        self.frames.append(
            YoloPoseFrame(
                self.frame_index,
                normalized,
                kpts_conf,
                box,
                timestamp_s=now,
                raw_keypoints_xy=kpts_xy.copy(),
            )
        )
        self._update_jump_completion(feet_y, body_height)
        if self.phase != "completed":
            return [self._event("guidance", self.phase, "Recording. Land and perform the rebound jump.", speak=False)]
        self.phase = "analyzing"
        return [
            self._event("guidance", self.phase, "Jump captured. Stop moving while I analyze it.", level="success"),
            LiveGuidanceEvent(
                type="stop_streaming",
                phase=self.phase,
                text="Jump captured. Stop streaming while the server analyzes the jump.",
                level="success",
                speak=False,
            ),
        ]

    def consume_pending_analysis(self) -> list[dict[str, Any]]:
        if self.phase != "analyzing" or self.analysis_started or self.analysis_completed:
            return []

        self.analysis_started = True
        fps = self._estimated_fps()
        result = self.analysis_service.analyze_pose_frames(
            self.frames,
            fps=fps,
            estimated_shoulder_width_m=self.setup_calibration.measured_shoulder_width_m,
            recording_started_at=self.session_started_at,
        )
        self.last_analysis_payload = result.as_json()
        self.analysis_completed = True
        self.phase = "completed"
        return [
            self._event("guidance", self.phase, result.summary, level="success").as_dict(),
            LiveGuidanceEvent(
                type="analysis_result",
                phase=self.phase,
                text=result.summary,
                level="success",
                speak=False,
                payload=self.last_analysis_payload,
            ).as_dict(),
        ]

    def _update_jump_completion(self, feet_y: float, body_height: float) -> None:
        if self.recording_started_monotonic is None:
            return
        if time.monotonic() - self.recording_started_monotonic < 1.5:
            return

        body_scale = max(body_height, 1.0)
        if self.first_landing_y is None or (not self.second_takeoff_seen and feet_y > self.first_landing_y):
            self.first_landing_y = feet_y
        if self.first_landing_y is not None and not self.second_takeoff_seen:
            self.second_takeoff_seen = feet_y < self.first_landing_y - 0.08 * body_scale
        if self.first_landing_y is not None and self.second_takeoff_seen and not self.second_landing_seen:
            self.second_landing_seen = feet_y >= self.first_landing_y - 0.08 * body_scale
            if self.second_landing_seen:
                self.second_landing_y = feet_y
        if self.second_landing_seen and self.second_landing_y is not None:
            if feet_y > self.second_landing_y + 0.12 * body_scale:
                self.stable_after_landing_count = 0
            elif abs(feet_y - self.second_landing_y) <= 0.04 * body_scale:
                self.stable_after_landing_count += 1
            else:
                self.stable_after_landing_count = max(0, self.stable_after_landing_count - 1)
            if self.stable_after_landing_count >= 5:
                self.phase = "completed"

    def _estimated_fps(self) -> float:
        if len(self.frames) < 2:
            return 15.0
        deltas = np.diff([frame.timestamp_s for frame in self.frames])
        mean_delta = float(np.mean(deltas[deltas > 0])) if np.any(deltas > 0) else 0.0
        if mean_delta <= 0:
            return 15.0
        return 1.0 / mean_delta

    def _clear_setup_buffers_if_needed(self) -> None:
        if self.setup_calibration is not None:
            return
        if self.floor_pose is None:
            self.floor_buffer.clear()
            self.floor_body_height_buffer.clear()
        else:
            self.box_buffer.clear()

    def _event(
        self,
        event_type: str,
        phase: str,
        text: str,
        *,
        level: str = "info",
        speak: bool = True,
    ) -> LiveGuidanceEvent:
        key = (phase, text)
        if speak:
            now = time.monotonic()
            min_repeat_seconds = 4.0 if level in {"warning", "error"} else 7.0
            last_spoken_at = self.last_spoken_by_message_key.get(key, 0.0)
            if now - last_spoken_at < min_repeat_seconds:
                speak = False
            else:
                self.last_spoken_by_message_key[key] = now
        return LiveGuidanceEvent(event_type, phase, text, level=level, speak=speak)
