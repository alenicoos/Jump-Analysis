"""YOLO video pipeline.

This file contains the full video side of the project:
- reads frames from a webcam or video;
- uses YOLO pose to estimate body keypoints;
- checks that required keypoints are visible;
- runs floor/box setup without closing the webcam;
- records the drop jump when the descent from the box is detected;
- finds the main biomechanical keyframes;
- converts YOLO pose frames into temporal model inputs.

Clinical/model logic stays outside this file: this module prepares data.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from jump_analysis.feedback import AudioFeedback
from jump_analysis.features.front_2d_features import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    NOSE,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    angle,
    body_keypoint,
    select_temporal_features,
)
from jump_analysis.validation import (
    DropJumpProtocolValidator,
    StablePoseBuffer,
    SetupCalibration,
    SetupValidator,
)


# Minimum YOLO confidence needed to treat a keypoint as visible.
CONFIDENCE_THRESHOLD = 0.30

# The protocol and temporal models require shoulders, hips, knees, and ankles.
# Without them, body position, landing depth, and sequence inputs become too noisy.
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
    """A valid normalized frame ready for protocol checks and temporal models."""

    frame_index: int
    keypoints_xy: np.ndarray
    keypoints_conf: np.ndarray
    box_xyxy: np.ndarray | None
    timestamp_s: float = 0.0
    raw_keypoints_xy: np.ndarray | None = None
    drop_trigger_px: float | None = None
    required_drop_px: float | None = None
    is_drop_trigger_frame: bool = False


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


def render_display_frame(
    display: np.ndarray,
    lines: list[str],
    window_name: str,
    show: bool,
    preview_callback: FramePreviewCallback | None = None,
    wait_ms: int = 1,
) -> bool:
    """Render annotated feedback to OpenCV, Streamlit, or both.

    Returns True when the OpenCV window receives `q`.
    """

    put_lines(display, lines)
    if preview_callback is not None:
        preview_callback(display)
    if not show:
        return False
    cv2.imshow(window_name, display)
    return bool(cv2.waitKey(wait_ms) & 0xFF == ord("q"))


def shoulder_width_px(kpts_xy: np.ndarray) -> float:
    """Shoulder width in pixels between left and right keypoints."""

    return float(np.linalg.norm(kpts_xy[LEFT_SHOULDER] - kpts_xy[RIGHT_SHOULDER]))


def normalize_keypoints_to_mocap_scale(
    kpts_xy: np.ndarray,
    shoulder_width_m: float,
) -> np.ndarray:
    """Convert YOLO coordinates to a mocap-like scale.

    YOLO works in pixels: if the user moves closer to the camera, distances
    increase even though the body is the same. Keypoints are rescaled using
    shoulder width measured during setup.

    The result is not a 3D reconstruction; it is a 2D normalization that makes
    protocol checks more stable across videos.
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
    preview_callback: FramePreviewCallback | None = None,
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
                    feedback.error("Full body in frame.")
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
                    render_display_frame(
                        display,
                        lines,
                        "YOLO setup",
                        show,
                        preview_callback=preview_callback,
                        wait_ms=250,
                    )
                    body_height_px = float(np.median(body_height_buffer))
                    return StablePoseCapture(stable, body_height_px)
                lines.append("Stay still...")
            else:
                missing = missing_required(kpts_conf)
                lines.append(f"Incomplete body: missing {missing}")
                if checking_started and feedback is not None:
                    feedback.error("Show shoulders, hips, knees, ankles.")
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
                feedback.error("Step into frame.")
            buffer.clear()
            body_height_buffer.clear()

        if render_display_frame(
            display,
            lines,
            "YOLO setup",
            show,
            preview_callback=preview_callback,
        ):
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
    preview_callback: FramePreviewCallback | None = None,
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
                        render_display_frame(
                            display,
                            lines,
                            "YOLO setup",
                            show,
                            preview_callback=preview_callback,
                            wait_ms=250,
                        )
                        return stable
                    lines.append("Stay still on the box...")
                else:
                    buffer.clear()
                    lines.append("You are not clearly above the floor yet")
                    if feedback is not None:
                        feedback.error("Step onto the box.")
            else:
                missing = missing_required(kpts_conf)
                lines.append(f"Incomplete body: missing {missing}")
                if feedback is not None:
                    feedback.error("Show full body.")
                now = time.monotonic()
                if now - last_debug_print > 1.0:
                    print(f"Incomplete box setup. Missing keypoints: {missing}", flush=True)
                    last_debug_print = now
                buffer.clear()
        else:
            lines.append("No person")
            if checking_started and feedback is not None:
                feedback.error("Step into frame.")
            buffer.clear()

        if render_display_frame(
            display,
            lines,
            "YOLO setup",
            show,
            preview_callback=preview_callback,
        ):
            raise RuntimeError("Setup interrupted by user.")


def run_floor_box_setup_with_open_capture(
    cap: cv2.VideoCapture,
    model: YOLO,
    height_cm: float,
    show: bool = True,
    audio: bool = False,
    feedback: AudioFeedback | None = None,
    auto_detect_box: bool = True,
    preview_callback: FramePreviewCallback | None = None,
) -> SetupCalibration:
    """Run floor/box setup using an already-open webcam.

    Used by `scripts/main.py`: first calibrates the user on the floor, then on
    the box, validates geometry/height, and keeps the webcam open so the drop
    jump can start immediately.
    """

    feedback = feedback if feedback is not None else AudioFeedback(enabled=audio)
    feedback.speak(
        "Stand on the floor, near the box.",
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
        preview_callback=preview_callback,
    )
    floor_pose = floor_sample.keypoints_xy

    feedback.speak("Floor captured.", force=True)
    print("Floor pose captured.")
    print(f"Body height in pixels: {floor_sample.body_height_px:.1f}px")
    print("\nSETUP 2/2: step onto the box while keeping the same distance from the camera.")
    feedback.speak("Step onto the box.", force=True)
    if auto_detect_box:
        box_pose = capture_stable_box_pose_after_floor(
            cap,
            model,
            floor_pose,
            show=show,
            min_box_height_ratio=SetupValidator().min_box_height_ratio,
            feedback=feedback,
            preview_callback=preview_callback,
        )
    else:
        box_sample = capture_stable_setup_pose(
            cap,
            model,
            ["SETUP 2/2: step onto the box", "Keep the same distance from the camera"],
            show=show,
            feedback=feedback,
            preview_callback=preview_callback,
        )
        box_pose = box_sample.keypoints_xy

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    result = SetupValidator().validate_floor_and_box(
        floor_pose,
        box_pose,
        height_cm=height_cm,
        floor_body_height_px=floor_sample.body_height_px,
        frame_width=frame_width,
    )
    for message in result.messages:
        print(message, flush=True)
    for check in result.checks:
        if not check.passed:
            if check.severity == "error":
                if check.name == "floor_box_scale_stability":
                    feedback.error("Keep the same distance.")
                elif check.name == "box_height_detected":
                    feedback.error("Box height unclear.")
            elif check.severity == "warning":
                if check.name == "subject_horizontal_centering":
                    # Direction is embedded in check.message: extract "Move left/right."
                    direction = "Move left." if "Move left" in check.message else "Move right."
                    feedback.warn(f"You are off-center. {direction}")
                elif check.name == "camera_roll":
                    feedback.warn("Camera appears tilted. Adjust the camera angle.")
                elif check.name == "subject_frontal_orientation":
                    feedback.warn("Face the camera directly.")
    if result.calibration and result.calibration.estimated_box_height_cm is not None:
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
    box_height_px: float | None = None,
    nose_drop_box_ratio: float = 0.60,
    show: bool = True,
    prepare_seconds: float = 2.0,
    min_drop_ratio: float = 0.06,
    max_pre_drop_upward_ratio: float = 0.05,
    max_wait_seconds: float = 30.0,
    feedback: AudioFeedback | None = None,
    preview_callback: FramePreviewCallback | None = None,
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

    # Preparation buffer: head/foot/body position and body height before descent.
    # This is used to detect when the drop starts.
    prep_nose_y = deque(maxlen=45)
    prep_feet_y = deque(maxlen=45)
    prep_left_ankle_y = deque(maxlen=45)
    prep_right_ankle_y = deque(maxlen=45)
    prep_body_center_y = deque(maxlen=45)
    prep_body_height = deque(maxlen=45)

    locked_track_id = None
    opened_at = time.monotonic()
    recording_started_at = None
    frame_index = 0
    detected_frames = 0
    incomplete_frames = 0
    last_debug_print = 0.0
    pre_record_frames: deque[YoloPoseFrame] = deque(maxlen=15)

    if feedback is not None:
        feedback.speak(
            "Drop, land, jump.",
            force=True,
        )

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.monotonic()
        if recording_started_at is None and now - opened_at > max_wait_seconds:
            raise RuntimeError("Timeout: I did not see a valid start from height within the maximum wait time.")
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
                left_ankle_y = float(kpts_xy[LEFT_ANKLE][1])
                right_ankle_y = float(kpts_xy[RIGHT_ANKLE][1])
                nose_visible = visible(kpts_conf, NOSE)
                nose_y = float(kpts_xy[NOSE][1]) if nose_visible else None
                feet_y = float((left_ankle_y + right_ankle_y) / 2.0)
                body_center_y = float(body_keypoint(kpts_xy)[1])
                body_height = estimate_person_height_px(box, kpts_xy)

                # From this point on, save normalized coordinates, not raw
                # pixels. Scale is based on shoulder width measured during setup.
                normalized = normalize_keypoints_to_mocap_scale(kpts_xy, shoulder_width_m)
                pose_frame = YoloPoseFrame(
                    frame_index,
                    normalized,
                    kpts_conf,
                    box,
                    timestamp_s=now,
                    raw_keypoints_xy=kpts_xy.copy(),
                )

                if recording_started_at is None:
                    if len(prep_feet_y) >= 8:
                        nose_baseline = float(np.median(prep_nose_y)) if len(prep_nose_y) >= 4 else None
                        baseline = float(np.median(prep_feet_y))
                        left_baseline = float(np.median(prep_left_ankle_y))
                        right_baseline = float(np.median(prep_right_ankle_y))
                        body_baseline = float(np.median(prep_body_center_y))
                        median_body_height = float(np.median(prep_body_height))
                        required_drop_px = min_drop_ratio * median_body_height
                        max_upward_px = max_pre_drop_upward_ratio * median_body_height
                        current_drop_px = feet_y - baseline
                        nose_drop_px = (nose_y - nose_baseline) if (nose_y is not None and nose_baseline is not None) else None
                        required_nose_drop_px = (
                            nose_drop_box_ratio * box_height_px
                            if box_height_px is not None and box_height_px > 1
                            else None
                        )
                        left_drop_px = left_ankle_y - left_baseline
                        right_drop_px = right_ankle_y - right_baseline
                        bilateral_drop_px = min(left_drop_px, right_drop_px)
                        body_drop_px = body_center_y - body_baseline
                        current_upward_px = baseline - feet_y
                        if required_nose_drop_px is not None and nose_drop_px is not None:
                            lines.append(f"Nose drop: {nose_drop_px:.1f}px / {required_nose_drop_px:.1f}px")
                        else:
                            lines.append(f"Ready when you drop: {current_drop_px:.1f}px / {required_drop_px:.1f}px")
                        if len(prep_feet_y) >= int(prepare_seconds * 15) and current_upward_px > max_upward_px:
                            if feedback is not None:
                                feedback.error("Do not jump up. Drop first.")
                            raise RuntimeError(
                                "Invalid start: your ankles moved upward from the box. "
                                "Do not jump off the box. Drop down first, land, then jump."
                            )
                        # Drop trigger: prefer the nose/head descent because a
                        # single foot can move off the box before the real drop.
                        # Fall back to ankle/body cues if the nose is not visible
                        # or no box height was calibrated.
                        nose_ready = (
                            required_nose_drop_px is not None
                            and nose_drop_px is not None
                            and nose_drop_px > required_nose_drop_px
                        )
                        bilateral_ready = bilateral_drop_px > required_drop_px * 0.45
                        body_ready = body_drop_px > required_drop_px * 0.25
                        mean_ready = current_drop_px > required_drop_px
                        fallback_ready = mean_ready and bilateral_ready and body_ready
                        if len(prep_feet_y) >= int(prepare_seconds * 15) and (nose_ready or fallback_ready):
                            recording_started_at = now
                            trigger_value = float(nose_drop_px if nose_ready and nose_drop_px is not None else current_drop_px)
                            trigger_threshold = float(required_nose_drop_px if nose_ready and required_nose_drop_px is not None else required_drop_px)
                            start_frames = list(pre_record_frames)[-7:]
                            pose_frame.drop_trigger_px = trigger_value
                            pose_frame.required_drop_px = trigger_threshold
                            pose_frame.is_drop_trigger_frame = True
                            frames = start_frames + [pose_frame]
                            print("Drop detected: recording started.", flush=True)
                        else:
                            # Drop has not started yet. Keep updating the
                            # baseline only while both feet still look close to
                            # the box stance; if one leg is already stepping
                            # off, do not let that single-leg motion contaminate
                            # the stable baseline.
                            asymmetric_step = abs(left_drop_px - right_drop_px) > required_drop_px * 0.7
                            if not asymmetric_step:
                                if nose_y is not None:
                                    prep_nose_y.append(nose_y)
                                prep_feet_y.append(feet_y)
                                prep_left_ankle_y.append(left_ankle_y)
                                prep_right_ankle_y.append(right_ankle_y)
                                prep_body_center_y.append(body_center_y)
                                prep_body_height.append(body_height)
                                pre_record_frames.append(pose_frame)
                    else:
                        # Initial phase: collect enough frames for a stable baseline.
                        if nose_y is not None:
                            prep_nose_y.append(nose_y)
                        prep_feet_y.append(feet_y)
                        prep_left_ankle_y.append(left_ankle_y)
                        prep_right_ankle_y.append(right_ankle_y)
                        prep_body_center_y.append(body_center_y)
                        prep_body_height.append(body_height)
                        pre_record_frames.append(pose_frame)
                        lines.append("Stay still on the box...")
                else:
                    # Recording has already started: every valid frame is used
                    # for keyframes and temporal models.
                    frames.append(pose_frame)
                    lines.append(f"Valid frames: {len(frames)}")
            else:
                incomplete_frames += 1
                missing = missing_required(kpts_conf)
                lines.append(f"Incomplete body: missing {missing}")
                if recording_started_at is None and feedback is not None:
                    feedback.error("Show full body.")
                now = time.monotonic()
                if now - last_debug_print > 1.0:
                    print(f"Incomplete body. Missing keypoints: {missing}", flush=True)
                    last_debug_print = now
        else:
            lines.append("No person")
            if recording_started_at is None and feedback is not None:
                feedback.error("Step into frame.")

        if render_display_frame(
            display,
            lines,
            "YOLO front capture",
            show,
            preview_callback=preview_callback,
        ):
            break

        frame_index += 1

    if show:
        cv2.destroyWindow("YOLO front capture")
    if len(frames) < 10:
        # With too few valid frames, keyframes and protocol checks cannot be estimated
        # reliably, so stop and ask the user to try again.
        raise RuntimeError(
            f"Too few valid pose frames: {len(frames)}. "
            f"Detected frames: {detected_frames}, incomplete frames: {incomplete_frames}. "
            "Try --model models/yolo26n-pose.pt or --model yolo11n-pose.pt, "
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


def drop_trigger_index(frames: list[YoloPoseFrame]) -> int:
    """Return the frame where capture detected the real drop start."""

    for index, frame in enumerate(frames):
        if getattr(frame, "is_drop_trigger_frame", False):
            return index
        if getattr(frame, "drop_trigger_px", None) is not None:
            return index
    return 0


def knee_flexion_proxy(frame: YoloPoseFrame) -> float:
    """Simple proxy for knee flexion.

    Use 180 - hip-knee-ankle angle: the more the knee bends, the more this
    value increases. It is used to find maximum knee flexion.
    """

    k = frame.keypoints_xy
    left = 180.0 - angle(k[LEFT_HIP], k[LEFT_KNEE], k[LEFT_ANKLE])
    right = 180.0 - angle(k[RIGHT_HIP], k[RIGHT_KNEE], k[RIGHT_ANKLE])
    return float(np.nanmean([left, right]))


def smooth_series(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Return a short moving-average smoothing of a 1D signal."""

    if len(values) < 3 or window <= 1:
        return values.astype(float)
    window = min(window, len(values))
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(values.astype(float), (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def find_still_end_frame(
    frames: list[YoloPoseFrame],
    start: int,
    still_seconds: float = 2.5,
    fps: float = 30.0,
    still_threshold_ratio: float = 0.03,
    stable_fraction: float = 0.85,
) -> int | None:
    """Return the last frame of the first window of `still_seconds` consecutive
    stillness found after `start`, or None if no such window exists.

    Stillness is defined as smoothed frame-to-frame body-center displacement
    below `still_threshold_ratio * shoulder_width` for most frames in the
    window. A fractional criterion is more robust to YOLO keypoint jitter than
    requiring every single frame to be below threshold.
    """
    if not frames:
        return None

    still_frames = max(1, round(still_seconds * fps))
    if len(frames) - start < still_frames:
        return None

    body_y = np.array([body_keypoint(f.keypoints_xy)[1] for f in frames])
    smooth_window = max(3, round(0.20 * fps))
    body_y = smooth_series(body_y, window=smooth_window)
    ref_scale = max(
        float(np.median([shoulder_width_px(f.keypoints_xy) for f in frames])),
        1e-6,
    )
    threshold = still_threshold_ratio * ref_scale
    velocity = np.abs(np.diff(body_y, prepend=body_y[0]))
    stable_fraction = float(np.clip(stable_fraction, 0.0, 1.0))

    for i in range(max(0, start), len(frames) - still_frames + 1):
        window_velocity = velocity[i : i + still_frames]
        if float(np.mean(window_velocity < threshold)) >= stable_fraction:
            return i + still_frames - 1

    return None


def find_yolo_keyframes(frames: list[YoloPoseFrame]) -> tuple[int, int]:
    """Find initial contact (ic) and maximum knee flexion (kf).

    `ic` — first ankle-y peak followed by a clear rebound, i.e. when the user
    lands after dropping from the box.

    `kf` — combines two cues within a short window after ic:
    - maximum knee-flexion proxy (hip-knee-ankle angle);
    - maximum body-center lowering (body_y).
    The window is capped at ic+30 frames so a later second landing cannot be
    mistaken for the first deep crouch.
    """

    left_ankle_y = smooth_series(np.array([frame.keypoints_xy[LEFT_ANKLE][1] for frame in frames]), window=5)
    right_ankle_y = smooth_series(np.array([frame.keypoints_xy[RIGHT_ANKLE][1] for frame in frames]), window=5)
    ankle_y   = smooth_series(np.array([ankle_mean_y(frame) for frame in frames]), window=5)
    body_y    = smooth_series(np.array([body_keypoint(frame.keypoints_xy)[1] for frame in frames]), window=5)
    knee_flex = smooth_series(np.array([knee_flexion_proxy(frame) for frame in frames]), window=5)
    T = len(frames)

    motion_range = max(float(np.ptp(ankle_y)), 1e-6)
    min_rebound_lift = 0.07 * motion_range
    trigger_idx = drop_trigger_index(frames)
    baseline_stop = max(1, trigger_idx)
    baseline_left = float(np.median(left_ankle_y[:baseline_stop]))
    baseline_right = float(np.median(right_ankle_y[:baseline_stop]))
    baseline_body = float(np.median(body_y[:baseline_stop]))
    reference_width = max(float(np.median([shoulder_width_px(frame.keypoints_xy) for frame in frames])), 1e-6)
    min_landing_drop = max(0.10 * motion_range, 0.035 * reference_width)
    min_body_drop = max(0.04 * motion_range, 0.015 * reference_width)

    # First landing: first credible ankle-y peak after the real drop trigger.
    # Keep the window short so a late second landing or post-landing movement
    # cannot become the initial contact line.
    search_start = min(max(trigger_idx + 1, 1), max(1, T - 2))
    search_stop = max(search_start + 1, min(T - 2, trigger_idx + 42))
    ic_candidates: list[int] = []
    for i in range(search_start, search_stop):
        if ankle_y[i] < ankle_y[i - 1] or ankle_y[i] < ankle_y[i + 1]:
            continue
        left_drop = left_ankle_y[i] - baseline_left
        right_drop = right_ankle_y[i] - baseline_right
        body_drop = body_y[i] - baseline_body
        if min(left_drop, right_drop) < min_landing_drop * 0.6 or body_drop < min_body_drop:
            continue
        lookahead_end = min(T, i + 45)
        if lookahead_end <= i + 2:
            continue
        if ankle_y[i] - float(np.min(ankle_y[i + 1:lookahead_end])) >= min_rebound_lift:
            ic_candidates.append(i)
            break
    if ic_candidates:
        ic = ic_candidates[0]
    else:
        search_slice = ankle_y[search_start:search_stop]
        local_peak_candidates = []
        level_candidates = []
        landing_level = float(np.percentile(search_slice, 75)) if len(search_slice) else float(np.max(ankle_y))
        for local_idx, value in enumerate(search_slice):
            i = search_start + local_idx
            left_drop = left_ankle_y[i] - baseline_left
            right_drop = right_ankle_y[i] - baseline_right
            body_drop = body_y[i] - baseline_body
            if i > 0 and i < T - 1 and ankle_y[i] >= ankle_y[i - 1] and ankle_y[i] >= ankle_y[i + 1]:
                local_peak_candidates.append(i)
            if value >= landing_level and body_drop >= min_body_drop:
                # Avoid one-leg setup drift: either both ankles have moved
                # down at least a little, or the body center has clearly
                # dropped enough to indicate real descent.
                if min(left_drop, right_drop) > 0 or body_drop >= min_body_drop * 1.8:
                    level_candidates.append(i)

        if len(level_candidates):
            ic = int(level_candidates[0])
        elif len(local_peak_candidates):
            ic = int(local_peak_candidates[0])
        elif len(search_slice):
            # Last resort: choose the earliest high frame in the short landing
            # window, not the maximum, to avoid drifting toward late movement.
            fallback_level = float(np.percentile(search_slice, 65))
            high = np.where(search_slice >= fallback_level)[0]
            ic = search_start + int(high[0]) if len(high) else search_start
        else:
            ic = search_start

    # Maximum knee flexion: find the first local peak of knee_flex after ic.
    # Using the first local peak (rather than the global max) avoids confusing
    # the first crouch with a deeper second-landing crouch later in the clip.
    kf_search_end = min(T, ic + 35)
    kf_window_knee = knee_flex[ic:kf_search_end]
    kf_window_body = body_y[ic:kf_search_end]

    # First local maximum of knee_flex: the value must be higher than both
    # its neighbours. Fall back to global argmax if no local peak is found.
    kf_by_knee = ic  # default
    for j in range(1, len(kf_window_knee) - 1):
        if kf_window_knee[j] >= kf_window_knee[j - 1] and kf_window_knee[j] >= kf_window_knee[j + 1]:
            kf_by_knee = ic + j
            break
    else:
        kf_by_knee = ic + int(np.nanargmax(kf_window_knee))

    # First local maximum of body_y (body lowest = most crouched).
    kf_by_body = ic  # default
    for j in range(1, len(kf_window_body) - 1):
        if kf_window_body[j] >= kf_window_body[j - 1] and kf_window_body[j] >= kf_window_body[j + 1]:
            kf_by_body = ic + j
            break
    else:
        kf_by_body = ic + int(np.argmax(kf_window_body))

    kf = int(np.clip(int(round((kf_by_knee + kf_by_body) / 2)), ic, kf_search_end - 1))

    return ic, kf


def analyze_yolo_pose_frames(frames: list[YoloPoseFrame]) -> dict[str, int | float | bool]:
    """Extract keyframe and protocol metadata from a YOLO recording."""

    ic, kf = find_yolo_keyframes(frames)

    # The validator checks that the recorded movement looks like a real drop
    # jump: start from height, two-foot contact, second jump.
    protocol = DropJumpProtocolValidator().validate(frames, ic, kf)

    metadata = {
        "valid_pose_frames": len(frames),
        "ic_valid_frame": ic,
        "kfmax_valid_frame": kf,
        "ic_raw_frame": frames[ic].frame_index,
        "kfmax_raw_frame": frames[kf].frame_index,
    }
    metadata.update(protocol.as_metadata())
    return metadata


def slice_frames_for_ae_training_window(
    frames: list[YoloPoseFrame],
    metadata: dict[str, int | float | bool],
    pre_ic_seconds: float = 0.5,
) -> tuple[list[YoloPoseFrame], dict[str, int | float | bool]]:
    """Return the AE window used during training: pre-IC context to takeoff.

    Mocap training sequences are extracted from 0.5 seconds before initial
    contact through takeoff. This helper applies the same window to live YOLO
    captures and shifts keyframe metadata so phase labels remain aligned to the
    sliced sequence.
    """

    if not frames:
        return frames, dict(metadata)

    ic = int(metadata.get("ic_valid_frame", 0))
    takeoff = int(metadata.get("takeoff_valid_frame", len(frames) - 1))
    ic = int(np.clip(ic, 0, len(frames) - 1))
    takeoff = int(np.clip(takeoff, ic, len(frames) - 1))

    ic_time = frames[ic].timestamp_s
    if ic_time:
        target_start_time = ic_time - pre_ic_seconds
        start = max(
            0,
            max(
                (i for i, frame in enumerate(frames[:ic + 1]) if frame.timestamp_s <= target_start_time),
                default=0,
            ),
        )
    else:
        start = max(0, ic - 15)

    end = takeoff
    sliced = frames[start:end + 1]
    shifted = dict(metadata)
    for key in (
        "ic_valid_frame",
        "kfmax_valid_frame",
        "takeoff_valid_frame",
        "second_landing_valid_frame",
    ):
        if key in shifted:
            shifted[key] = int(np.clip(int(shifted[key]) - start, 0, len(sliced) - 1))
    shifted["valid_pose_frames"] = len(sliced)
    shifted["ae_window_start_valid_frame"] = start
    shifted["ae_window_end_valid_frame"] = end
    return sliced, shifted


def frames_to_transformer_input(frames: list[YoloPoseFrame], input_dim: int = 24) -> np.ndarray:
    """Convert a YoloPoseFrame list to temporal keypoint input.

    Uses raw pixel keypoints (frame.raw_keypoints_xy) normalized by the median
    body height across frames, matching the normalization used during training.

    Falls back to the mocap-scale keypoints (frame.keypoints_xy) if raw
    keypoints are not available, normalizing by the shoulder-to-ankle distance.
    """

    T = len(frames)
    kp_raw = np.zeros((T, 17, 2), dtype=np.float32)
    heights = np.zeros(T, dtype=np.float32)

    for i, frame in enumerate(frames):
        kp = (frame.raw_keypoints_xy if frame.raw_keypoints_xy is not None
              else frame.keypoints_xy).astype(np.float32)
        kp_raw[i] = kp
        # Normalize by shoulder-to-ankle distance (same as mocap pipeline) so
        # that the feature distributions are consistent across domains.
        left_h  = float(np.linalg.norm(kp[LEFT_SHOULDER]  - kp[LEFT_ANKLE]))
        right_h = float(np.linalg.norm(kp[RIGHT_SHOULDER] - kp[RIGHT_ANKLE]))
        heights[i] = max((left_h + right_h) / 2.0, 1e-3)

    height = float(np.median(heights[heights > 0])) if (heights > 0).any() else 1.0
    kp_norm = kp_raw / height   # (T, 17, 2)

    # Layout: first all x (columns 0-16), then all y (columns 17-33).
    seq = np.concatenate([kp_norm[:, :, 0], kp_norm[:, :, 1]], axis=1)  # (T, 34)
    if input_dim == 24:
        seq = select_temporal_features(seq, include_head=False)
    elif input_dim != 34:
        raise ValueError(f"Unsupported temporal input dimension: {input_dim}. Expected 24 or 34.")
    return seq.astype(np.float32)


def predict_knee_pitch(
    frames: list[YoloPoseFrame],
    pitch_model,
    device: str | None = None,
) -> np.ndarray:
    """Run PitchTransformer on a recording.

    Parameters
    ----------
    frames:
        The recorded YoloPoseFrame list from capture_yolo_pose_frames_with_open_capture.
    pitch_model:
        A loaded PitchTransformer instance (from jump_analysis.models.pitch_transformer).
    device:
        Torch device string. If None, uses pitch_model.best_device().

    Returns
    -------
    np.ndarray of shape (T, 2): predicted [left_pitch_delta, right_pitch_delta]
    in degrees for every frame. Values are relative to the first frame (baseline = 0).
    """

    input_dim = getattr(getattr(pitch_model, "input_proj", None), "in_features", 24)
    seq = frames_to_transformer_input(frames, input_dim=input_dim)
    return pitch_model.predict_numpy(seq, device=device)   # (T, 2)
