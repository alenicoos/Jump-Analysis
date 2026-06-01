"""YOLO video pipeline.

This file contains the full video side of the project:
- reads frames from a webcam or video;
- uses YOLO pose to estimate body keypoints;
- checks that required keypoints are visible;
- runs floor/box setup without closing the webcam;
- records the drop jump when the descent from the box is detected;
- finds the main biomechanical keyframes;
- converts YOLO keypoints into the 37 front-view features comparable with the dataset.

Clinical/model logic stays outside this file: this module prepares data.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from jump_analysis.data import FRONT_2D_FEATURE_COLUMNS
from jump_analysis.feedback import AudioFeedback
from jump_analysis.features.front_2d_features import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    FrontKeyframes,
    angle,
    body_keypoint,
    build_front_2d_feature_row,
)
from jump_analysis.validation import (
    DropJumpProtocolValidator,
    StablePoseBuffer,
    SetupCalibration,
    SetupValidator,
)


# Minimum YOLO confidence needed to treat a keypoint as visible.
CONFIDENCE_THRESHOLD = 0.30

# The 37 front-view features do not require the head.
# Shoulders, hips, knees, and ankles are required; without them some features
# would be invented or too noisy.
REQUIRED_POSE_KEYPOINTS = (
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
)

# Connections used only to draw the skeleton in the OpenCV window.
SKELETON = [
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


@dataclass
class YoloPoseFrame:
    """A valid normalized frame ready for features/keyframes."""

    frame_index: int
    keypoints_xy: np.ndarray
    keypoints_conf: np.ndarray
    box_xyxy: np.ndarray | None
    timestamp_s: float = 0.0
    raw_keypoints_xy: np.ndarray | None = None
    drop_trigger_px: float | None = None
    required_drop_px: float | None = None


@dataclass
class StablePoseCapture:
    """Stable setup pose plus body height in pixels."""

    keypoints_xy: np.ndarray
    body_height_px: float


def visible(conf: np.ndarray, index: int) -> bool:
    """Return whether a YOLO keypoint passes the minimum confidence threshold."""

    return conf[index] >= CONFIDENCE_THRESHOLD


def required_visible(conf: np.ndarray) -> bool:
    """Return whether all required keypoints are visible in the frame."""

    return all(visible(conf, index) for index in REQUIRED_POSE_KEYPOINTS)


def missing_required(conf: np.ndarray) -> list[int]:
    """Return COCO indices for missing required keypoints."""

    return [index for index in REQUIRED_POSE_KEYPOINTS if not visible(conf, index)]


def select_pose(result, locked_track_id):
    """Choose which person to follow in a YOLO frame.

    If YOLO tracking already assigned a track id, keep using the same person.
    This avoids switching to another person if someone enters the frame.

    If no track id is locked yet, choose the person with the largest bounding
    box, usually the closest and most central person.
    """

    if result.keypoints is None or result.keypoints.xy is None or len(result.keypoints.xy) == 0:
        return None

    # YOLO returns tensors; convert to numpy for simple calculations.
    kpts_xy = result.keypoints.xy.cpu().numpy()
    kpts_conf = result.keypoints.conf.cpu().numpy()
    boxes_xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else None
    track_ids = None
    if result.boxes is not None and result.boxes.id is not None:
        track_ids = result.boxes.id.cpu().numpy().astype(int)

    if locked_track_id is not None and track_ids is not None:
        matches = np.where(track_ids == locked_track_id)[0]
        if len(matches) > 0:
            idx = int(matches[0])
            box = boxes_xyxy[idx] if boxes_xyxy is not None else None
            return idx, locked_track_id, kpts_xy[idx], kpts_conf[idx], box

    idx = 0
    if boxes_xyxy is not None and len(boxes_xyxy) > 0:
        areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
        idx = int(np.argmax(areas))

    track_id = None
    if track_ids is not None and len(track_ids) > idx:
        track_id = int(track_ids[idx])
    box = boxes_xyxy[idx] if boxes_xyxy is not None else None
    return idx, track_id, kpts_xy[idx], kpts_conf[idx], box


def draw_pose(frame: np.ndarray, kpts_xy: np.ndarray, kpts_conf: np.ndarray) -> None:
    """Draw skeleton points and segments on the displayed frame."""

    for i, j in SKELETON:
        if visible(kpts_conf, i) and visible(kpts_conf, j):
            cv2.line(
                frame,
                tuple(kpts_xy[i].astype(int).tolist()),
                tuple(kpts_xy[j].astype(int).tolist()),
                (255, 0, 0),
                2,
            )
    for index, (x, y) in enumerate(kpts_xy):
        if visible(kpts_conf, index):
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)


def put_lines(frame: np.ndarray, lines: list[str]) -> None:
    """Draw readable messages over the webcam frame.

    OpenCV draws text directly on the image; we add a dark rectangle behind it
    so the text stays readable.
    """

    x, y = 16, 20
    line_height = 28
    width = max(520, max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0][0] for line in lines) + 32)
    height = 18 + line_height * len(lines)
    cv2.rectangle(frame, (x - 8, y - 8), (x + width, y + height), (36, 36, 36), -1)
    for idx, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + line_height * (idx + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def shoulder_width_px(kpts_xy: np.ndarray) -> float:
    """Shoulder width in pixels between left and right keypoints."""

    return float(np.linalg.norm(kpts_xy[LEFT_SHOULDER] - kpts_xy[RIGHT_SHOULDER]))


def normalize_keypoints_to_mocap_scale(
    kpts_xy: np.ndarray,
    shoulder_width_m: float,
) -> np.ndarray:
    """Convert YOLO coordinates to a mocap-like scale.

    YOLO works in pixels: if the user moves closer to the camera, distances
    increase even though the body is the same. To compare jumps with the
    dataset, keypoints are rescaled using shoulder width measured during setup.

    The result is not a 3D reconstruction; it is a 2D normalization that makes
    features more comparable across videos.
    """

    scale = shoulder_width_m / max(shoulder_width_px(kpts_xy), 1.0)
    return kpts_xy * scale


def estimate_person_height_px(box_xyxy: np.ndarray | None, kpts_xy: np.ndarray) -> float:
    """Estimate person height in pixels.

    If YOLO provides the person bounding box, use it because it better captures
    the full body. If it is missing, use the vertical range of available
    keypoints.
    """

    if box_xyxy is not None:
        return float(max(box_xyxy[3] - box_xyxy[1], 1.0))
    return float(max(kpts_xy[:, 1].max() - kpts_xy[:, 1].min(), 1.0))


def full_body_box_visible(box_xyxy: np.ndarray | None, frame_shape: tuple[int, ...], tolerance_ratio: float = 0.04) -> bool:
    """Return whether the person box is not severely clipped by the image.

    YOLO boxes are not exact body masks and often touch the image border even
    when the person is usable. This check is intentionally lenient: it only
    fails when the box extends clearly outside the frame.
    """

    if box_xyxy is None:
        return True
    height, width = frame_shape[:2]
    tolerance_x = width * tolerance_ratio
    tolerance_y = height * tolerance_ratio
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    return x1 >= -tolerance_x and y1 >= -tolerance_y and x2 <= width + tolerance_x and y2 <= height + tolerance_y


def capture_stable_setup_pose(
    cap: cv2.VideoCapture,
    model: YOLO,
    prompt_lines: list[str],
    show: bool = True,
    timeout_seconds: float = 30.0,
    stable_frames: int = 12,
    feedback: AudioFeedback | None = None,
    grace_seconds: float = 5.0,
    require_full_body_visible: bool = False,
) -> StablePoseCapture:
    """Capture a stable pose during setup.

    Used both for the floor pose and manual box mode. The pose is not captured
    immediately: several frames are accumulated and the median is accepted only
    once the body has stayed stable enough.
    """

    # StablePoseBuffer avoids calibrating while the user is still entering the
    # frame, moving the phone, or finding position.
    buffer = StablePoseBuffer(min_frames=stable_frames)
    body_height_buffer: deque[float] = deque(maxlen=stable_frames)
    started = time.monotonic()
    locked_track_id = None
    last_debug_print = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Cannot read frame during setup.")
        if time.monotonic() - started > timeout_seconds:
            raise RuntimeError("Setup timeout: I did not see a stable pose in time.")

        result = model.track(frame, stream=False, persist=True, verbose=False)[0]
        selection = select_pose(result, locked_track_id)
        display = frame.copy()
        lines = list(prompt_lines)
        checking_started = time.monotonic() - started >= grace_seconds

        if selection is not None:
            _, detected_track_id, kpts_xy, kpts_conf, box = selection
            # Once the first good person is detected, keep following the same
            # track id so setup does not switch to another person.
            if locked_track_id is None and detected_track_id is not None:
                locked_track_id = detected_track_id
            draw_pose(display, kpts_xy, kpts_conf)
            if not checking_started:
                lines.append("Get into position...")
            elif require_full_body_visible and not full_body_box_visible(box, frame.shape):
                lines.append("Full body is not inside the frame")
                if feedback is not None:
                    feedback.error("Make sure your whole body is visible from head to feet, facing the camera.")
                buffer.clear()
                body_height_buffer.clear()
            elif required_visible(kpts_conf):
                buffer.add(kpts_xy)
                body_height_buffer.append(estimate_person_height_px(box, kpts_xy))
                stable = buffer.stable_pose()
                if stable is not None:
                    # Return a median/stable pose, not a single frame. This
                    # makes floor and box estimation less noisy.
                    lines.append("Stable pose captured")
                    if show:
                        put_lines(display, lines)
                        cv2.imshow("YOLO setup", display)
                        cv2.waitKey(250)
                    body_height_px = float(np.median(body_height_buffer))
                    return StablePoseCapture(stable, body_height_px)
                lines.append("Stay still...")
            else:
                missing = missing_required(kpts_conf)
                lines.append(f"Incomplete body: missing {missing}")
                if checking_started and feedback is not None:
                    feedback.error("Incomplete body. Shoulders, hips, knees, and ankles must be visible.")
                now = time.monotonic()
                if now - last_debug_print > 1.0:
                    print(f"Incomplete setup. Missing keypoints: {missing}", flush=True)
                    last_debug_print = now
                # If required points are missing, clear the buffer so old valid
                # frames are not mixed with a new/incomplete pose.
                buffer.clear()
                body_height_buffer.clear()
        else:
            lines.append("No person")
            if checking_started and feedback is not None:
                feedback.error("I cannot see a person. Step into the frame.")
            buffer.clear()
            body_height_buffer.clear()

        if show:
            put_lines(display, lines)
            cv2.imshow("YOLO setup", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise RuntimeError("Setup interrupted by user.")


def capture_stable_box_pose_after_floor(
    cap: cv2.VideoCapture,
    model: YOLO,
    floor_pose: np.ndarray,
    show: bool = True,
    timeout_seconds: float = 45.0,
    stable_frames: int = 12,
    min_box_height_ratio: float = 0.05,
    feedback: AudioFeedback | None = None,
    grace_seconds: float = 3.0,
) -> np.ndarray:
    """Automatically capture the box pose after the floor pose.

    The user does not need to press enter. The script detects box entry when
    the ankles are clearly higher than in the floor pose.
    """

    buffer = StablePoseBuffer(min_frames=stable_frames)
    started = time.monotonic()
    locked_track_id = None
    last_debug_print = 0.0
    floor_ankle_y = float((floor_pose[LEFT_ANKLE][1] + floor_pose[RIGHT_ANKLE][1]) / 2.0)
    floor_scale = max(float(np.linalg.norm(floor_pose[LEFT_SHOULDER] - floor_pose[RIGHT_SHOULDER])), 1.0)
    # The threshold is relative to shoulder width, so it depends less on webcam
    # resolution.
    min_box_height_px = min_box_height_ratio * floor_scale

    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Cannot read frame during box setup.")
        if time.monotonic() - started > timeout_seconds:
            raise RuntimeError("Setup timeout: I did not see the user step onto the box and stay still.")

        result = model.track(frame, stream=False, persist=True, verbose=False)[0]
        selection = select_pose(result, locked_track_id)
        display = frame.copy()
        checking_started = time.monotonic() - started >= grace_seconds
        lines = [
            "SETUP 2/2: step onto the box",
            "Stay still once you are on the box",
        ]

        if selection is not None:
            _, detected_track_id, kpts_xy, kpts_conf, _ = selection
            if locked_track_id is None and detected_track_id is not None:
                locked_track_id = detected_track_id
            draw_pose(display, kpts_xy, kpts_conf)
            if not checking_started:
                lines.append("Step onto the box...")
            elif required_visible(kpts_conf):
                ankle_y = float((kpts_xy[LEFT_ANKLE][1] + kpts_xy[RIGHT_ANKLE][1]) / 2.0)
                # In image coordinates, y grows downward. If the user steps onto
                # the box, ankles move upward in the image, so y gets smaller:
                # floor_ankle_y - ankle_y is positive.
                box_height_px = floor_ankle_y - ankle_y
                lines.append(f"Detected height: {box_height_px:.1f}px / {min_box_height_px:.1f}px")
                if box_height_px > min_box_height_px:
                    buffer.add(kpts_xy)
                    stable = buffer.stable_pose()
                    if stable is not None:
                        lines.append("Box pose captured")
                        if show:
                            put_lines(display, lines)
                            cv2.imshow("YOLO setup", display)
                            cv2.waitKey(250)
                        return stable
                    lines.append("Stay still on the box...")
                else:
                    buffer.clear()
                    lines.append("You are not clearly above the floor yet")
                    if feedback is not None:
                        feedback.error("You do not look clearly on the box yet. Step onto the box and stay still.")
            else:
                missing = missing_required(kpts_conf)
                lines.append(f"Incomplete body: missing {missing}")
                if feedback is not None:
                    feedback.error("Incomplete body on the box. Shoulders, hips, knees, and ankles must be visible.")
                now = time.monotonic()
                if now - last_debug_print > 1.0:
                    print(f"Incomplete box setup. Missing keypoints: {missing}", flush=True)
                    last_debug_print = now
                buffer.clear()
        else:
            lines.append("No person")
            if checking_started and feedback is not None:
                feedback.error("I cannot see a person. Step into the frame.")
            buffer.clear()

        if show:
            put_lines(display, lines)
            cv2.imshow("YOLO setup", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise RuntimeError("Setup interrupted by user.")


def run_floor_box_setup_with_open_capture(
    cap: cv2.VideoCapture,
    model: YOLO,
    height_cm: float,
    show: bool = True,
    audio: bool = False,
    feedback: AudioFeedback | None = None,
    auto_detect_box: bool = True,
) -> SetupCalibration:
    """Run floor/box setup using an already-open webcam.

    Used by `scripts/main.py`: first calibrates the user on the floor, then on
    the box, validates geometry/height, and keeps the webcam open so the drop
    jump can start immediately.
    """

    feedback = feedback if feedback is not None else AudioFeedback(enabled=audio)
    feedback.speak(
        "Stand next to the box so your whole body is visible from head to feet, facing the camera.",
        force=True,
    )
    print("\nSETUP 1/2: stand on the floor, stay still, and face the camera.")
    floor_sample = capture_stable_setup_pose(
        cap,
        model,
        ["SETUP 1/2: stand next to the box", "Full body visible from head to feet"],
        show=show,
        feedback=feedback,
        grace_seconds=5.0,
        require_full_body_visible=True,
    )
    floor_pose = floor_sample.keypoints_xy

    feedback.speak("Floor pose captured.", force=True)
    print("Floor pose captured.")
    print(f"Body height in pixels: {floor_sample.body_height_px:.1f}px")
    print("\nSETUP 2/2: step onto the box while keeping the same distance from the camera.")
    feedback.speak("Now step onto the box and stay still.", force=True)
    if auto_detect_box:
        box_pose = capture_stable_box_pose_after_floor(
            cap,
            model,
            floor_pose,
            show=show,
            min_box_height_ratio=SetupValidator().min_box_height_ratio,
            feedback=feedback,
        )
    else:
        box_sample = capture_stable_setup_pose(
            cap,
            model,
            ["SETUP 2/2: step onto the box", "Keep the same distance from the camera"],
            show=show,
            feedback=feedback,
        )
        box_pose = box_sample.keypoints_xy

    result = SetupValidator().validate_floor_and_box(
        floor_pose,
        box_pose,
        height_cm=height_cm,
        floor_body_height_px=floor_sample.body_height_px,
    )
    for message in result.messages:
        print(message, flush=True)
    for check in result.checks:
        if not check.passed and check.severity == "error":
            if check.name == "floor_box_scale_stability":
                feedback.error("You moved closer to or farther from the camera between floor and box. Return to the same distance.")
            elif check.name == "box_height_detected":
                feedback.error("I cannot clearly measure the box height. Step onto the box and stay still.")
    if result.calibration and result.calibration.estimated_box_height_cm is not None:
        feedback.speak(
            f"Estimated box height {result.calibration.estimated_box_height_cm:.1f} centimeters.",
            force=True,
        )
        print(f"Pixel scale: {result.calibration.meters_per_pixel:.6f} m/px")
        print(f"Measured shoulder width: {result.calibration.measured_shoulder_width_m * 100:.1f} cm")
        print(f"Estimated box height: {result.calibration.estimated_box_height_cm:.1f} cm")
    if not result.passed:
        raise RuntimeError("Invalid setup: fix the errors and try again.")
    return result.calibration


def capture_yolo_pose_frames_with_open_capture(
    cap: cv2.VideoCapture,
    model: YOLO,
    seconds: float,
    shoulder_width_m: float,
    show: bool = True,
    prepare_seconds: float = 2.0,
    min_drop_ratio: float = 0.06,
    max_wait_seconds: float = 30.0,
    min_recording_seconds: float = 1.5,
    stable_after_landing_frames: int = 5,
    feedback: AudioFeedback | None = None,
) -> list[YoloPoseFrame]:
    """Record valid drop-jump frames.

    Recording does not start immediately. First the user must stay still on the
    box; then an ankle-position baseline is built and recording starts once a
    sufficient descent is detected.

    In practice:
    1. user is still on the box;
    2. ankle baseline is built;
    3. ankles move downward in the image;
    4. the drop has started;
    5. normalized frames are saved for feature extraction.
    """

    frames: list[YoloPoseFrame] = []

    # Preparation buffer: average foot position and body height before descent.
    # This is used to detect when the drop starts.
    prep_feet_y = deque(maxlen=45)
    prep_body_height = deque(maxlen=45)

    # Keep a few frames before the trigger only for baseline robustness. These
    # frames are not returned because saved trials must start at the jump start.
    pre_roll_frames: deque[YoloPoseFrame] = deque(maxlen=60)

    locked_track_id = None
    opened_at = time.monotonic()
    recording_started_at = None
    frame_index = 0
    detected_frames = 0
    incomplete_frames = 0
    last_debug_print = 0.0
    first_landing_y = None
    second_landing_y = None
    second_takeoff_seen = False
    second_landing_seen = False
    stable_after_landing_count = 0
    jump_finished = False

    if feedback is not None:
        feedback.speak(
            "Now start the jump. Drop from the box, land with both feet, and immediately jump again.",
            force=True,
        )

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.monotonic()
        if recording_started_at is None and now - opened_at > max_wait_seconds:
            raise RuntimeError("Timeout: I did not see a valid start from height within the maximum wait time.")
        if recording_started_at is not None and jump_finished:
            break
        if recording_started_at is not None and seconds and now - recording_started_at > seconds:
            break

        # `persist=True` keeps tracking across consecutive frames.
        result = model.track(frame, stream=False, persist=True, verbose=False)[0]
        selection = select_pose(result, locked_track_id)
        display = frame.copy()
        if recording_started_at is None:
            lines = [
                "Preparation: step onto the box/chair and stay still",
                "Recording starts when you begin to drop",
            ]
        else:
            lines = ["Recording drop jump", "Land and immediately perform the second jump"]
        if selection is not None:
            _, detected_track_id, kpts_xy, kpts_conf, box = selection
            detected_frames += 1
            if locked_track_id is None and detected_track_id is not None:
                locked_track_id = detected_track_id
            draw_pose(display, kpts_xy, kpts_conf)
            if required_visible(kpts_conf):
                # Ankle y coordinate: in OpenCV y increases downward, so a
                # descent increases this value.
                feet_y = float((kpts_xy[LEFT_ANKLE][1] + kpts_xy[RIGHT_ANKLE][1]) / 2.0)
                body_height = estimate_person_height_px(box, kpts_xy)

                # From this point on, save normalized coordinates, not raw
                # pixels. Scale is based on shoulder width measured during setup.
                normalized = normalize_keypoints_to_mocap_scale(kpts_xy, shoulder_width_m)

                if recording_started_at is None:
                    if len(prep_feet_y) >= 8:
                        baseline = float(np.median(prep_feet_y))
                        required_drop_px = min_drop_ratio * float(np.median(prep_body_height))
                        current_drop_px = feet_y - baseline
                        lines.append(f"Ready when you drop: {current_drop_px:.1f}px / {required_drop_px:.1f}px")
                        # Drop trigger: enough preparation frames plus ankles
                        # descending beyond the relative height threshold.
                        if len(prep_feet_y) >= int(prepare_seconds * 15) and current_drop_px > required_drop_px:
                            recording_started_at = now
                            frames = []
                            print("Drop detected: recording started.", flush=True)
                            frames.append(
                                YoloPoseFrame(
                                    frame_index,
                                    normalized,
                                    kpts_conf,
                                    box,
                                    timestamp_s=now,
                                    raw_keypoints_xy=kpts_xy.copy(),
                                    drop_trigger_px=current_drop_px,
                                    required_drop_px=required_drop_px,
                                )
                            )
                        else:
                            # Drop has not started yet: update both baseline and pre-roll.
                            pre_roll_frames.append(
                                YoloPoseFrame(
                                    frame_index,
                                    normalized,
                                    kpts_conf,
                                    box,
                                    timestamp_s=now,
                                    raw_keypoints_xy=kpts_xy.copy(),
                                )
                            )
                            prep_feet_y.append(feet_y)
                            prep_body_height.append(body_height)
                    else:
                        # Initial phase: collect enough frames for a stable baseline.
                        pre_roll_frames.append(
                            YoloPoseFrame(
                                frame_index,
                                normalized,
                                kpts_conf,
                                box,
                                timestamp_s=now,
                                raw_keypoints_xy=kpts_xy.copy(),
                            )
                        )
                        prep_feet_y.append(feet_y)
                        prep_body_height.append(body_height)
                        lines.append("Stay still on the box...")
                else:
                    # Recording has already started: every valid frame is used
                    # for keyframes and features.
                    frames.append(
                        YoloPoseFrame(
                            frame_index,
                            normalized,
                            kpts_conf,
                            box,
                            timestamp_s=now,
                            raw_keypoints_xy=kpts_xy.copy(),
                        )
                    )
                    lines.append(f"Valid frames: {len(frames)}")
                    if now - recording_started_at >= min_recording_seconds:
                        body_scale = max(body_height, 1.0)
                        if first_landing_y is None or (not second_takeoff_seen and feet_y > first_landing_y):
                            first_landing_y = feet_y
                        if first_landing_y is not None and not second_takeoff_seen:
                            second_takeoff_seen = feet_y < first_landing_y - 0.08 * body_scale
                        if first_landing_y is not None and second_takeoff_seen and not second_landing_seen:
                            second_landing_seen = feet_y >= first_landing_y - 0.08 * body_scale
                            if second_landing_seen:
                                second_landing_y = feet_y
                        if second_landing_seen and second_landing_y is not None:
                            if feet_y > second_landing_y + 0.12 * body_scale:
                                stable_after_landing_count = 0
                            elif abs(feet_y - second_landing_y) <= 0.04 * body_scale:
                                stable_after_landing_count += 1
                            else:
                                stable_after_landing_count = max(0, stable_after_landing_count - 1)
                            jump_finished = stable_after_landing_count >= stable_after_landing_frames
                        if second_landing_seen:
                            lines.append(f"Stable after landing: {stable_after_landing_count}/{stable_after_landing_frames}")
            else:
                incomplete_frames += 1
                missing = missing_required(kpts_conf)
                lines.append(f"Incomplete body: missing {missing}")
                if recording_started_at is None and feedback is not None:
                    feedback.error("Incomplete body. Shoulders, hips, knees, and ankles must be visible before you start.")
                now = time.monotonic()
                if now - last_debug_print > 1.0:
                    print(f"Incomplete body. Missing keypoints: {missing}", flush=True)
                    last_debug_print = now
        else:
            lines.append("No person")
            if recording_started_at is None and feedback is not None:
                feedback.error("I cannot see a person. Step into the frame before you start.")

        if show:
            put_lines(display, lines)
            cv2.imshow("YOLO front capture", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_index += 1

    if show:
        cv2.destroyWindow("YOLO front capture")
    if len(frames) < 10:
        # With too few valid frames, keyframes and features cannot be estimated
        # reliably, so stop and ask the user to try again.
        raise RuntimeError(
            f"Too few valid pose frames: {len(frames)}. "
            f"Detected frames: {detected_frames}, incomplete frames: {incomplete_frames}. "
            "Try --model yolo26n-pose.pt or --model yolo11n-pose.pt, "
            "move farther away until shoulders, hips, knees, and ankles are always visible, "
            "and use good lighting."
        )
    return frames


def ankle_mean_y(frame: YoloPoseFrame) -> float:
    """Average vertical coordinate between left and right ankles.
    
    This is often more stable than using the body center, and in some videos it
    may be the only reliable cue for landing detection.
    """

    k = frame.keypoints_xy
    return float((k[LEFT_ANKLE][1] + k[RIGHT_ANKLE][1]) / 2.0)


def knee_flexion_proxy(frame: YoloPoseFrame) -> float:
    """Simple proxy for knee flexion.

    Use 180 - hip-knee-ankle angle: the more the knee bends, the more this
    value increases. It is used to find maximum knee flexion.
    """

    k = frame.keypoints_xy
    left = 180.0 - angle(k[LEFT_HIP], k[LEFT_KNEE], k[LEFT_ANKLE])
    right = 180.0 - angle(k[RIGHT_HIP], k[RIGHT_KNEE], k[RIGHT_ANKLE])
    return float(np.nanmean([left, right]))


def find_yolo_keyframes(frames: list[YoloPoseFrame]) -> tuple[int, int]:
    """Find initial contact and maximum knee flexion.

    `ic` is estimated by finding when ankles are lowest in the image, i.e. when
    the user lands.

    `kf` combines two cues:
    - maximum knee-flexion proxy;
    - maximum body-center lowering after contact.
    """

    ankle_y = np.array([ankle_mean_y(frame) for frame in frames])
    body_y = np.array([body_keypoint(frame.keypoints_xy)[1] for frame in frames])
    knee_flex = np.array([knee_flexion_proxy(frame) for frame in frames])

    landing_level = float(np.percentile(ankle_y, 90))
    # Choose the first frame near landing level, not necessarily the absolute
    # maximum, to avoid selecting too late.
    candidates = np.where(ankle_y >= landing_level - 0.04 * max(np.ptp(ankle_y), 1.0))[0]
    ic = int(candidates[0]) if len(candidates) else int(np.argmax(ankle_y))
    end = min(len(frames), ic + 60)
    if end <= ic + 2:
        end = len(frames)

    kf_by_knee = ic + int(np.nanargmax(knee_flex[ic:end]))
    kf_by_body = ic + int(np.argmax(body_y[ic:end]))
    kf = int(round((kf_by_knee + kf_by_body) / 2))
    return ic, kf


def extract_front_features_from_yolo_frames(frames: list[YoloPoseFrame]) -> tuple[dict[str, float], dict[str, int]]:
    """Extract front-view features and metadata from a YOLO recording."""

    ic, kf = find_yolo_keyframes(frames)

    # The validator checks that the recorded movement looks like a real drop
    # jump: start from height, two-foot contact, second jump.
    protocol = DropJumpProtocolValidator().validate(frames, ic, kf)

    # The 37 dataset features are computed only on two keyframes: initial
    # contact and maximum knee flexion.
    keyframes = FrontKeyframes(
        initial_contact=frames[ic].keypoints_xy,
        max_knee_flexion=frames[kf].keypoints_xy,
        crop_length_frames=frames[kf].frame_index - frames[ic].frame_index + 1,
    )
    metadata = {
        "valid_pose_frames": len(frames),
        "ic_valid_frame": ic,
        "kfmax_valid_frame": kf,
        "ic_raw_frame": frames[ic].frame_index,
        "kfmax_raw_frame": frames[kf].frame_index,
    }
    metadata.update(protocol.as_metadata())
    return build_front_2d_feature_row(keyframes), metadata


def compare_to_reference(features: dict[str, float], reference_csv: str | Path) -> pd.DataFrame:
    """Compare one YOLO feature row with the reference mocap CSV."""

    reference = pd.read_csv(reference_csv)
    rows = []
    for column in FRONT_2D_FEATURE_COLUMNS:
        # For every feature, compute simple statistics: mean, standard
        # deviation, z-score, and empirical percentile in the dataset.
        ref = pd.to_numeric(reference[column], errors="coerce").dropna()
        value = float(features[column])
        mean = float(ref.mean())
        std = float(ref.std(ddof=0))
        percentile = float((ref <= value).mean() * 100.0)
        z_score = (value - mean) / std if std > 0 else 0.0
        rows.append(
            {
                "feature": column,
                "value": value,
                "reference_mean": mean,
                "reference_std": std,
                "z_score": z_score,
                "percentile": percentile,
            }
        )
    return pd.DataFrame(rows)
