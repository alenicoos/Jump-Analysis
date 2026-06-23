from __future__ import annotations

# User-facing Streamlit GUI for the jump-analysis pipeline.

import base64
import html
import sys
import os
from pathlib import Path

os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ASSETS = ROOT / "assets"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PROTOCOL_ERROR_MESSAGE = "The movement did not pass the drop-jump protocol checks."
RETAKE_ERROR_PREFIXES = (
    PROTOCOL_ERROR_MESSAGE,
    "Too few valid pose frames:",
    "Invalid start: your ankles moved upward from the box.",
)
AE_RUNTIME_THRESHOLD_MULTIPLIER = 10.0
AE_MIN_CONSECUTIVE_ANOMALOUS_FRAMES = 2


class ProtocolValidationError(RuntimeError):
    """Raised when movement capture succeeded but protocol checks failed."""

    def __init__(self, metadata: dict[str, int | float]) -> None:
        super().__init__(PROTOCOL_ERROR_MESSAGE)
        self.metadata = metadata

from jump_analysis.feedback import AudioFeedback
from jump_analysis.video import (
    analyze_yolo_pose_frames,
    capture_yolo_pose_frames_with_open_capture,
    find_still_end_frame,
    frames_to_transformer_input,
    run_floor_box_setup_with_open_capture,
)
from jump_analysis.features.front_2d_features import AE_FEATURE_DIM, extract_ae_features, AE_FEATURE_NAMES


def longest_true_run(mask: np.ndarray) -> int:
    """Return the longest consecutive True run in a boolean vector."""

    longest = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def first_drop_trigger_index(frames) -> int:
    for index, frame in enumerate(frames):
        if getattr(frame, "is_drop_trigger_frame", False):
            return index
        if getattr(frame, "drop_trigger_px", None) is not None:
            return index
    return 0


def shift_metadata_window(metadata: dict, start: int, length: int) -> dict:
    shifted = dict(metadata)
    if length <= 0:
        return shifted
    for key in (
        "ic_valid_frame",
        "kfmax_valid_frame",
    ):
        if key in shifted:
            shifted[key] = int(np.clip(int(shifted[key]) - start, 0, length - 1))
    shifted["valid_pose_frames"] = length
    return shifted


st.set_page_config(
    page_title="JumpGuard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    :root {
      --bg-0: #0b1028;
      --bg-1: #171b46;
      --bg-2: #2a2257;
      --card: rgba(255, 255, 255, 0.105);
      --card-strong: rgba(255, 255, 255, 0.145);
      --stroke: rgba(255, 255, 255, 0.16);
      --ink: #f8faff;
      --muted: #c6c8e6;
      --accent: #7b61ff;
      --accent-2: #34d7ff;
      --accent-3: #f365ff;
      --danger: #ff5c8a;
      --control-width: 420px;
    }
    .stApp {
      background:
        radial-gradient(circle at 18% 12%, rgba(52, 215, 255, 0.26), transparent 30%),
        radial-gradient(circle at 82% 18%, rgba(243, 101, 255, 0.28), transparent 32%),
        linear-gradient(135deg, var(--bg-0) 0%, var(--bg-1) 48%, var(--bg-2) 100%);
      color: var(--ink);
    }
    [data-testid="stHeader"] {
      background: transparent;
    }
    #MainMenu, footer {
      visibility: hidden;
    }
    .main .block-container {
      padding-top: 4.4rem;
      max-width: 740px;
    }
    h1 {
      letter-spacing: 0;
      margin: 0 0 0.15rem;
      color: var(--ink);
      font-size: 3.2rem;
      line-height: 1;
    }
    .hero {
      display: flex;
      align-items: center;
      justify-content: flex-start;
      gap: 24px;
      margin: 0 0 42px;
      text-align: left;
      background:
        linear-gradient(110deg, rgba(91, 111, 255, 0.16), rgba(243, 101, 255, 0.13)),
        var(--card);
      border: 1px solid var(--stroke);
      border-radius: 34px;
      padding: 34px 38px;
      box-shadow: 0 24px 74px rgba(0, 0, 0, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.13);
      backdrop-filter: blur(18px);
    }
    .hero-logo {
      width: 96px;
      height: 96px;
      border-radius: 28px;
      flex: 0 0 auto;
      box-shadow: 0 18px 46px rgba(123, 97, 255, 0.34);
    }
    .hero-copy p {
      color: var(--muted);
      margin: 8px 0 0;
      font-size: 1.1rem;
      line-height: 1.45;
      max-width: 440px;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
      background: var(--card);
      border-color: var(--stroke);
      box-shadow: 0 18px 52px rgba(0, 0, 0, 0.22);
      max-width: 560px;
      margin: 0 auto 44px;
      backdrop-filter: blur(18px);
    }
    label, [data-testid="stWidgetLabel"] p {
      color: var(--ink) !important;
      font-weight: 650;
    }
    [data-testid="stNumberInput"],
    [data-testid="stTextInput"],
    [data-testid="stToggle"],
    [data-testid="stCheckbox"],
    div[data-testid="stButton"],
    .stButton,
    .hint {
      width: var(--control-width) !important;
      max-width: 100%;
      margin-left: auto !important;
      margin-right: auto !important;
      align-self: center !important;
    }
    .stElementContainer:has([data-testid="stTextInput"]),
    .stElementContainer:has([data-testid="stCheckbox"]),
    .stElementContainer:has(.stButton),
    .stElementContainer:has(.hint) {
      width: var(--control-width) !important;
      max-width: 100%;
      margin-left: auto !important;
      margin-right: auto !important;
      align-self: center !important;
    }
    [data-testid="stNumberInput"],
    [data-testid="stTextInput"] {
      margin-bottom: 22px !important;
    }
    [data-testid="stToggle"],
    [data-testid="stCheckbox"] {
      margin-bottom: 24px !important;
      text-align: center;
    }
    [data-testid="stToggle"] label,
    [data-testid="stCheckbox"] label {
      width: max-content !important;
      margin-left: auto !important;
      margin-right: auto !important;
    }
    div[data-baseweb="input"] {
      width: var(--control-width) !important;
      max-width: 100% !important;
      margin: 0 auto;
      border-color: rgba(52, 215, 255, 0.42) !important;
      background: rgba(80, 92, 176, 0.34) !important;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.10), 0 0 0 1px rgba(123, 97, 255, 0.18);
    }
    div[data-baseweb="input"] > div {
      background: rgba(80, 92, 176, 0.18) !important;
    }
    div[data-baseweb="input"]:focus-within {
      border-color: var(--accent-2) !important;
      box-shadow: 0 0 0 1px var(--accent-2), 0 0 22px rgba(52, 215, 255, 0.18) !important;
      outline-color: var(--accent) !important;
    }
    div[data-baseweb="input"] input {
      color: #ffffff !important;
      -webkit-text-fill-color: #ffffff !important;
      caret-color: #ffffff !important;
    }
    div[data-baseweb="input"] button {
      color: #ffffff !important;
    }
    [data-testid="stNumberInputStepDown"],
    [data-testid="stNumberInputStepUp"] {
      display: none !important;
    }
    [data-testid="stNumberInputField"] {
      width: 100% !important;
      border-radius: 8px !important;
    }
    [data-testid="stBaseButton-primary"] {
      background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
      border: 1px solid rgba(255, 255, 255, 0.14) !important;
      color: #ffffff !important;
    }
    [data-testid="stBaseButton-primary"]:hover {
      background: linear-gradient(90deg, #8d78ff, #5ee6ff) !important;
      border-color: rgba(255, 255, 255, 0.22) !important;
    }
    [data-testid="stCheckbox"] label, [data-testid="stCheckbox"] p {
      color: var(--ink) !important;
    }
    div[data-testid="stMetric"] {
      background: var(--card);
      border: 1px solid var(--stroke);
      border-radius: 8px;
      padding: 14px 16px;
      box-shadow: 0 8px 26px rgba(0, 0, 0, 0.18);
    }
    .report-panel {
      background: var(--card);
      border: 1px solid var(--stroke);
      border-radius: 22px;
      padding: 18px 20px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18);
      color: var(--ink);
      margin-top: 18px;
    }
    .stButton > button {
      border-radius: 14px;
      min-height: 46px;
      font-weight: 700;
      width: var(--control-width) !important;
      max-width: 100%;
      display: block;
      margin: 0 auto;
    }
    [data-testid="stNumberInput"] label,
    [data-testid="stTextInput"] label,
    [data-testid="stToggle"] label {
      justify-content: center;
    }
    [data-testid="stTextInput"] [data-baseweb="input"],
    [data-testid="stTextInput"] input,
    [data-testid="stBaseButton-primary"] {
      box-sizing: border-box !important;
    }
    [data-testid="stToggle"] [role="switch"],
    [data-testid="stToggle"] [aria-checked="true"] {
      background-color: var(--accent) !important;
      border-color: var(--accent-2) !important;
    }
    [data-testid="stToggle"] [aria-checked="false"] {
      background-color: rgba(255, 255, 255, 0.16) !important;
      border-color: rgba(255, 255, 255, 0.24) !important;
    }
    .hint {
      color: var(--muted);
      font-size: 0.96rem;
      margin-top: 18px;
      text-align: center;
    }
    .status-ok {
      color: var(--accent-2);
      font-weight: 700;
    }
    .status-warn {
      color: var(--accent-3);
      font-weight: 700;
    }
    .status-danger {
      color: var(--danger);
      font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def run_jump_pipeline(
    *,
    height_cm: float,
    source: str,
    model_name: str,
    seconds: float,
    prepare_seconds: float,
    max_wait_seconds: float,
    min_drop_ratio: float,
    audio_enabled: bool,
    show_windows: bool,
    preview_placeholder=None,
) -> dict:
    """Run the webcam analysis and return data for the GUI report."""

    video_source = int(source) if str(source).isdigit() else source
    feedback = AudioFeedback(enabled=audio_enabled)
    yolo_model = YOLO(model_name)
    cap = open_video_capture(video_source)
    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open the camera. On macOS, grant camera access to the app or terminal "
            "running Streamlit, then restart the Streamlit server and try again."
        )

    def show_frame_in_page(frame_bgr):
        if preview_placeholder is None:
            return
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        preview_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    try:
        calibration = run_floor_box_setup_with_open_capture(
            cap, yolo_model, height_cm=height_cm, show=show_windows,
            audio=audio_enabled, feedback=feedback, auto_detect_box=True,
            preview_callback=show_frame_in_page,
        )
        frames = capture_yolo_pose_frames_with_open_capture(
            cap, yolo_model, seconds,
            shoulder_width_m=calibration.measured_shoulder_width_m,
            box_height_px=calibration.box_height_px,
            show=show_windows, prepare_seconds=prepare_seconds,
            min_drop_ratio=min_drop_ratio, max_wait_seconds=max_wait_seconds,
            feedback=feedback, preview_callback=show_frame_in_page,
        )
    finally:
        cap.release()
        if show_windows:
            cv2.destroyAllWindows()

    metadata = analyze_yolo_pose_frames(frames)
    if not metadata["protocol_passed"]:
        raise ProtocolValidationError(metadata)
    model_frames = frames
    trigger_idx = first_drop_trigger_index(model_frames)

    # ── AE window: trigger-2 → still_end - 2.0s (fallback: end of recording) ──
    fps = 30.0
    if len(model_frames) >= 2:
        dur = model_frames[-1].timestamp_s - model_frames[0].timestamp_s
        if dur > 0:
            fps = (len(model_frames) - 1) / dur
    ae_start = max(0, trigger_idx - 2)
    ic_frame = int(metadata.get("ic_valid_frame", ae_start))
    still_scan_start = max(ae_start, ic_frame + max(1, round(1.0 * fps)))
    still_end = find_still_end_frame(model_frames, still_scan_start, still_seconds=1.5, fps=fps)
    if still_end is not None:
        ae_end = max(ae_start, still_end - max(1, round(1.0 * fps)))
    else:
        ae_end = len(model_frames) - 1
    ae_frames = model_frames[ae_start : ae_end + 1]
    ae_metadata = dict(metadata)
    ae_metadata["ae_window_start_valid_frame"] = ae_start
    ae_metadata["ae_window_end_valid_frame"] = ae_end
    for key in ("ic_valid_frame", "kfmax_valid_frame"):
        if key in ae_metadata:
            ae_metadata[key] = int(np.clip(int(ae_metadata[key]) - ae_start, 0, len(ae_frames) - 1))

    replay_start = max(0, trigger_idx - 7)
    replay_frames = model_frames[replay_start:]
    replay_metadata = shift_metadata_window(metadata, replay_start, len(replay_frames))
    replay_ae_start = max(0, ae_start - replay_start)
    replay_ae_skip = max(0, replay_start - ae_start)

    pitch_plot_start = ae_start
    pitch_plot_end = ae_end
    stickman_kp_seq = frames_to_transformer_input(replay_frames, input_dim=24)

    # ── PitchTransformer ──────────────────────────────────────────────────────
    pitch_series  = None
    ae_pitch_series = None
    pitch_model   = None
    pitch_error   = None
    pitch_model_path = ROOT / "models" / "pitch_transformer.pt"
    if not pitch_model_path.exists():
        pitch_error = f"pitch_transformer.pt not found in {pitch_model_path.parent}"
    else:
        try:
            from jump_analysis.models.pitch_transformer import PitchTransformer
            pitch_model  = PitchTransformer.load(str(pitch_model_path))
            pitch_input_dim = pitch_model.input_proj.in_features
            kp_seq       = frames_to_transformer_input(model_frames, input_dim=pitch_input_dim)
            pitch_series = pitch_model.predict_numpy(kp_seq)
            ae_start = int(ae_metadata.get("ae_window_start_valid_frame", 0))
            ae_end = int(ae_metadata.get("ae_window_end_valid_frame", len(model_frames) - 1))
            ae_pitch_series = pitch_series[ae_start:ae_end + 1]
        except Exception as e:
            pitch_error = f"PitchTransformer error: {e}"

    # ── JumpAutoencoder ───────────────────────────────────────────────────────
    ae_score       = None
    ae_is_anomaly  = None
    ae_threshold   = None
    ae_frame_errors= None
    ae_feature_errors = None
    ae_longest_anomalous_run = 0
    ae_error       = None
    kp_seq         = None
    ae_model_path  = ROOT / "models" / "jump_autoencoder_lstm.pt"
    if not ae_model_path.exists():
        ae_error = f"jump_autoencoder_lstm.pt not found in {ae_model_path.parent}"
    else:
        try:
            from jump_analysis.models.jump_autoencoder import JumpAutoencoder
            ae_model = JumpAutoencoder.load(str(ae_model_path))
            if ae_model.input_dim != AE_FEATURE_DIM:
                raise RuntimeError(
                    f"Autoencoder expects {ae_model.input_dim} features, but the active AE feature set has {AE_FEATURE_DIM}."
            )
            kp_seq            = frames_to_transformer_input(ae_frames, input_dim=24)
            ae_seq            = extract_ae_features(kp_seq)          # (T, AE_FEATURE_DIM)
            if ae_model.anomaly_threshold is None:
                raise RuntimeError("Autoencoder anomaly threshold is not set.")
            ae_score          = ae_model.anomaly_score_numpy(ae_seq)
            ae_threshold      = float(ae_model.anomaly_threshold) * AE_RUNTIME_THRESHOLD_MULTIPLIER
            ae_frame_errors   = ae_model.frame_errors_numpy(ae_seq)
            ae_feature_errors = ae_model.feature_errors_numpy(ae_seq)
            anomalous_frames  = ae_frame_errors > ae_threshold
            ae_longest_anomalous_run = longest_true_run(anomalous_frames)
            score_trigger = ae_score > ae_threshold
            frame_trigger = ae_longest_anomalous_run >= AE_MIN_CONSECUTIVE_ANOMALOUS_FRAMES
            ae_is_anomaly = bool(score_trigger or frame_trigger)
        except Exception as e:
            ae_error = f"JumpAutoencoder error: {e}"

    return {
        "metadata":          metadata,
        "ae_metadata":       ae_metadata,
        "replay_metadata":   replay_metadata,
        "replay_ae_start":   replay_ae_start,
        "replay_ae_skip":    replay_ae_skip,
        "pitch_plot_start":  pitch_plot_start,
        "pitch_plot_end":    pitch_plot_end,
        "kp_seq":            kp_seq if kp_seq is not None else None,
        "stickman_kp_seq":    stickman_kp_seq,
        "pitch_series":      pitch_series,
        "ae_pitch_series":   ae_pitch_series,
        "ae_score":          ae_score,
        "ae_is_anomaly":     ae_is_anomaly,
        "ae_threshold":      ae_threshold,
        "ae_frame_errors":   ae_frame_errors,
        "ae_feature_errors": ae_feature_errors,
        "ae_error":          ae_error,
    }


def open_video_capture(video_source: int | str) -> cv2.VideoCapture:
    """Open a webcam/video source with a macOS-friendly backend when possible."""

    if isinstance(video_source, int) and sys.platform == "darwin":
        return cv2.VideoCapture(video_source, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(video_source)


def can_retake_after_error(message: str) -> bool:
    """Return whether a failed capture can be retried from the GUI."""

    return any(message.startswith(prefix) for prefix in RETAKE_ERROR_PREFIXES)


def failed_protocol_rows(metadata: dict[str, int | float]) -> list[dict[str, str]]:
    """Build user-facing rows explaining failed protocol checks."""

    explanations = {
        "drop_started_from_height": "You must drop down from the box before landing.",
        "second_jump": "After landing, immediately perform the rebound jump.",
    }
    rows = []
    for key, passed in metadata.items():
        if not key.endswith("_passed") or key == "protocol_passed" or bool(passed):
            continue
        name = key.removesuffix("_passed")
        value = float(metadata.get(f"{name}_value", 0.0))
        threshold = float(metadata.get(f"{name}_threshold", 0.0))
        why = explanations.get(name, "This protocol condition was not met.")
        rows.append(
            {
                "check": name.replace("_", " "),
                "observed": f"{value:.2f}",
                "required": f"{threshold:.2f}",
                "why": why,
            }
        )
    return rows


def protocol_label(metadata: dict[str, int | float]) -> str:
    """Build a compact protocol summary."""

    checks = {
        key.replace("_passed", ""): value
        for key, value in metadata.items()
        if key.endswith("_passed") and key != "protocol_passed"
    }
    failed = [name.replace("_", " ") for name, value in checks.items() if not value]
    if not failed:
        return "The drop-jump sequence was detected correctly."
    return "Protocol issues detected: " + ", ".join(failed) + "."


KEYPOINT_NAMES = [
    "nose",
    "left eye",
    "right eye",
    "left ear",
    "right ear",
    "left shoulder",
    "right shoulder",
    "left elbow",
    "right elbow",
    "left wrist",
    "right wrist",
    "left hip",
    "right hip",
    "left knee",
    "right knee",
    "left ankle",
    "right ankle",
]
HEAD_KEYPOINT_INDICES = {0, 1, 2, 3, 4}
BODY_KEYPOINT_NAMES = [name for index, name in enumerate(KEYPOINT_NAMES) if index not in HEAD_KEYPOINT_INDICES]


def jump_phase(frame_index: int, metadata: dict[str, int | float]) -> str:
    """Map a frame index to a broad movement phase."""

    ic = int(metadata.get("ic_valid_frame", 0))
    if frame_index < ic:
        return "pre-landing descent"
    return "post-landing"


_FEATURE_MESSAGES: dict[str, str] = {
    "left_knee_flexion":           "The left knee bend depth is outside the normal range.",
    "right_knee_flexion":          "The right knee bend depth is outside the normal range.",
    "left_hip_flexion":            "The left hip flexion is outside the normal range.",
    "right_hip_flexion":           "The right hip flexion is outside the normal range.",
    "trunk_lateral_lean":          "The trunk tilts to one side.",
    "knee_width_ratio":            "The knees are unusually close together or wide apart.",
    "left_knee_valgus":            "The left knee alignment is outside the normal range.",
    "right_knee_valgus":           "The right knee alignment is outside the normal range.",
    "knee_center_vs_ankle_center": "The knees are not aligned over the ankles.",
    "body_lean_over_ankles":       "The body is leaning too far forward or backward.",
    "left_leg_length_ratio":       "The left leg segment proportions are unusual.",
    "right_leg_length_ratio":      "The right leg segment proportions are unusual.",
    "knee_flexion_asymmetry":      "One knee bends significantly more than the other.",
    "hip_flexion_asymmetry":       "Hip flexion is uneven between left and right.",
    "leg_ratio_asymmetry":         "There is a noticeable asymmetry in leg proportions.",
    "shoulder_tilt":               "The shoulders are not level.",
}


def detailed_anomaly_lines(
    *,
    pitch_series: np.ndarray,
    ae_feature_errors: np.ndarray | None,
    ae_threshold: float | None,
    replay_ae_start: int = 0,
) -> list[str]:
    """Return up to 3 plain-English sentences describing the anomaly,
    each referencing the stickman frame where it is most visible."""

    lines: list[str] = []

    # ── Feature-level messages ────────────────────────────────────────────────
    if ae_feature_errors is not None and len(ae_feature_errors):
        feature_mean = ae_feature_errors.mean(axis=0)
        threshold = float(ae_threshold) if ae_threshold is not None else 0.0
        ranked = np.argsort(feature_mean)[::-1]
        for fi in ranked:
            if len(lines) >= 3:
                break
            if fi >= len(AE_FEATURE_NAMES):
                continue
            if feature_mean[fi] <= threshold:
                break
            name = AE_FEATURE_NAMES[fi]
            msg = _FEATURE_MESSAGES.get(name)
            if msg:
                worst_ae_frame = int(np.argmax(ae_feature_errors[:, fi]))
                stickman_frame = worst_ae_frame + replay_ae_start
                lines.append(f"{msg} <span style='color:var(--muted);font-size:0.85em;'>(frame {stickman_frame} in the replay)</span>")

    # ── Pitch asymmetry ───────────────────────────────────────────────────────
    if len(lines) < 3:
        asym_series = np.abs(pitch_series[:, 0] - pitch_series[:, 1])
        mean_asym = float(asym_series.mean())
        max_asym = float(asym_series.max())
        if max_asym >= 8 or mean_asym >= 5:
            worst_frame = int(np.argmax(asym_series)) + replay_ae_start
            lines.append(
                f"The left and right legs rotate very differently from each other. "
                f"<span style='color:var(--muted);font-size:0.85em;'>(frame {worst_frame} in the replay)</span>"
            )

    return lines



def build_stickman_html(
    kp_seq: "np.ndarray",
    ae_frame_errors: "np.ndarray",
    ae_feature_errors: "np.ndarray | None",
    ae_threshold: float,
    metadata: dict,
    ae_start_in_replay: int = 0,
    title: str = "Movement replay",
    subtitle: str = "",
) -> str:
    """Build a self-contained HTML widget with an interactive stick-figure viewer."""
    import json

    T = len(kp_seq)
    ic = int(metadata.get("ic_valid_frame", 0))

    # Per-joint error: map feature errors onto the 12 body joints
    # Body-only joint indices: 0=LS,1=RS,2=LE,3=RE,4=LW,5=RW,6=LH,7=RH,8=LK,9=RK,10=LA,11=RA
    FEAT_TO_JOINTS = [
        [8, 6, 10], [9, 7, 11], [6, 0], [7, 1],      # knee/hip flex
        [0, 1, 6, 7], [8, 9],                         # trunk, knee_width
        [8], [9], [8, 9, 10, 11], [6, 7, 10, 11],     # valgus, balance
        [8, 6], [9, 7],                                # leg ratios
        [8, 9, 6, 7], [6, 7, 0, 1], [8, 9, 6, 7],    # asymmetries
        [0, 1],                                        # shoulder_tilt
    ]
    joint_errors = np.zeros((T, 12), dtype=np.float32)
    if ae_feature_errors is not None:
        for fi, joints in enumerate(FEAT_TO_JOINTS):
            if fi < ae_feature_errors.shape[1]:
                for j in joints:
                    joint_errors[:, j] = np.maximum(joint_errors[:, j], ae_feature_errors[:, fi])

    # Frames data: list of 24 floats (x0..x11, y0..y11)
    frames_data = kp_seq.tolist()
    jerr_data   = joint_errors.tolist()
    ferr_data   = ae_frame_errors.tolist()

    phases = [jump_phase(t, metadata) for t in range(T)]

    payload = json.dumps({
        "frames": frames_data,
        "jointErrors": jerr_data,
        "frameErrors": ferr_data,
        "phases": phases,
        "threshold": float(ae_threshold),
        "ic": ic, "aeStart": ae_start_in_replay, "T": T,
        "title": title,
        "subtitle": subtitle,
    }, separators=(",", ":"))

    return f"""
<h2 style="position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)">Interactive jump movement visualizer</h2>
<div id="jv-root" style="font-family:sans-serif;">
<style>
#jv-root canvas{{display:block;}}
#jv-title{{font-weight:700;color:#f8faff;text-align:center;margin:0 0 2px;}}
#jv-subtitle{{font-size:12px;color:#9ca3af;text-align:center;margin:0 0 8px;min-height:16px;}}
#jv-phase{{font-size:13px;color:#a78bfa;margin:4px 0 2px;text-align:center;}}
#jv-fnum{{font-size:12px;color:#6b7280;text-align:center;margin-bottom:6px;}}
#jv-play{{border:0;border-radius:8px;background:#7b61ff;color:white;padding:7px 13px;font-weight:700;cursor:pointer;}}
</style>

<div style="display:flex;flex-direction:column;align-items:center;gap:4px;">
  <div id="jv-title"></div>
  <div id="jv-subtitle"></div>
  <canvas id="jv-main" width="220" height="280" style="border-radius:10px;background:rgba(255,255,255,0.05);"></canvas>
  <button id="jv-play" type="button">Play</button>
  <div id="jv-phase">—</div>
  <div id="jv-fnum">frame 0</div>
  <canvas id="jv-strip" height="20" style="border-radius:4px;cursor:pointer;width:100%;max-width:480px;"></canvas>
  <canvas id="jv-scrub" height="28" style="cursor:pointer;width:100%;max-width:480px;"></canvas>
</div>
</div>

<script>
(function(){{
const D = {payload};
const T=D.T, ic=D.ic, aeStart=D.aeStart, thr=D.threshold;
const BONES=[[0,1],[0,2],[2,4],[1,3],[3,5],[0,6],[1,7],[6,7],[6,8],[8,10],[7,9],[9,11]];
const LABELS=["L shoulder","R shoulder","L elbow","R elbow","L wrist","R wrist","L hip","R hip","L knee","R knee","L ankle","R ankle"];

function errColor(e){{
  if(e <= thr) return `rgb(180,180,195)`;
  const t=Math.min((e-thr)/Math.max(thr,1e-6),1);
  return `rgb(${{Math.round(255)}},${{Math.round(80-40*t)}},${{Math.round(65-45*t)}})`;
}}

function getKps(fi){{
  const f=D.frames[fi]; const n=12;
  const pts=Array.from({{length:n}},(_,i)=>([f[i],f[i+n]]));
  const shoulderY=(pts[0][1]+pts[1][1])/2;
  const ankleY=(pts[10][1]+pts[11][1])/2;
  if(shoulderY > ankleY){{
    pts.forEach(p=>p[1]*=-1);
  }}
  return pts;
}}

function fitKps(kps,w,h,pad){{
  pad=pad||16;
  let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;
  kps.forEach(([x,y])=>{{xmin=Math.min(xmin,x);xmax=Math.max(xmax,x);ymin=Math.min(ymin,y);ymax=Math.max(ymax,y);}});
  const range=Math.max(xmax-xmin,ymax-ymin,0.01);
  const scale=Math.min(w-pad*2,h-pad*2)/range;
  const cx=(xmin+xmax)/2, cy=(ymin+ymax)/2;
  return kps.map(([x,y])=>[w/2+(x-cx)*scale, h/2+(y-cy)*scale]);
}}

function drawFrame(ctx,fi,w,h,pad){{
  ctx.clearRect(0,0,w,h);
  const raw=getKps(fi);
  const kps=fitKps(raw,w,h,pad);
  const je=D.jointErrors[fi];
  const frameIsAnomaly=D.frameErrors[fi] > thr;
  BONES.forEach(([a,b])=>{{
    const ec=Math.max(je[a],je[b]);
    ctx.strokeStyle=frameIsAnomaly ? errColor(Math.max(ec, D.frameErrors[fi])) : 'rgb(180,180,195)';
    ctx.lineWidth=w>150?3:2;
    ctx.beginPath(); ctx.moveTo(kps[a][0],kps[a][1]); ctx.lineTo(kps[b][0],kps[b][1]); ctx.stroke();
  }});
  kps.forEach(([x,y],i)=>{{
    ctx.fillStyle=frameIsAnomaly ? errColor(Math.max(je[i], D.frameErrors[fi])) : 'rgb(180,180,195)';
    ctx.beginPath(); ctx.arc(x,y,w>150?5:3,0,Math.PI*2); ctx.fill();
  }});
}}

function drawStrip(ctx,w){{
  for(let t=0;t<T;t++){{
    const e=D.frameErrors[t];
    const x=Math.round(t/T*w), nx=Math.round((t+1)/T*w);
    ctx.fillStyle=errColor(e);
    ctx.fillRect(x,0,nx-x,20);
  }}
  [[aeStart,'#00ff88'],[ic,'#ffe000']].forEach(([f,c])=>{{
    const x=f/T*w;
    ctx.strokeStyle=c; ctx.lineWidth=3;
    ctx.shadowColor=c; ctx.shadowBlur=6;
    ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,20); ctx.stroke();
    ctx.shadowBlur=0;
  }});
}}

function drawScrub(ctx,w,cur){{
  ctx.clearRect(0,0,w,28);
  ctx.strokeStyle='rgba(255,255,255,0.15)'; ctx.lineWidth=2;
  ctx.beginPath(); ctx.moveTo(10,14); ctx.lineTo(w-10,14); ctx.stroke();
  const x=10+(w-20)*(cur/Math.max(T-1,1));
  ctx.fillStyle='#a78bfa';
  ctx.beginPath(); ctx.arc(x,14,8,0,Math.PI*2); ctx.fill();
}}

let cur=0;
const mainC=document.getElementById('jv-main');
const mainCtx=mainC.getContext('2d');
const stripC=document.getElementById('jv-strip');
const stripCtx=stripC.getContext('2d');
const scrubC=document.getElementById('jv-scrub');
const scrubCtx=scrubC.getContext('2d');
const phaseEl=document.getElementById('jv-phase');
const fnumEl=document.getElementById('jv-fnum');
const playBtn=document.getElementById('jv-play');
document.getElementById('jv-title').textContent=D.title||'';
document.getElementById('jv-subtitle').textContent=D.subtitle||'';
let playing=false;
let timer=null;

function setCanvasWidth(canvas, w){{
  canvas.width=w; canvas.style.width=w+'px';
}}

function initSizes(){{
  const maxW=Math.min(480, (document.getElementById('jv-root').offsetWidth||480)-24);
  setCanvasWidth(stripC,maxW); setCanvasWidth(scrubC,maxW);
}}

function render(){{
  initSizes();
  drawFrame(mainCtx,cur,220,280,20);
  drawStrip(stripCtx,stripC.width);
  drawScrub(scrubCtx,scrubC.width,cur);
  phaseEl.textContent=D.phases[cur]||'';
  fnumEl.textContent='frame '+cur+' / '+(T-1);
}}

function stop(){{
  playing=false;
  if(timer) window.clearInterval(timer);
  timer=null;
  playBtn.textContent='Play';
}}

function start(){{
  playing=true;
  playBtn.textContent='Pause';
  timer=window.setInterval(()=>{{
    cur=(cur+1)%T;
    render();
    if(cur===T-1) stop();
  }}, 99);
}}

function scrubTo(e,canvas){{
  const rect=canvas.getBoundingClientRect();
  const x=(e.touches?e.touches[0].clientX:e.clientX)-rect.left;
  const w=canvas.width;
  cur=Math.round(Math.max(0,Math.min(T-1,(x-10)/(w-20)*(T-1))));
  render();
}}

[scrubC,stripC].forEach(c=>{{
  let drag=false;
  c.addEventListener('mousedown',e=>{{stop();drag=true;scrubTo(e,c);}});
  c.addEventListener('mousemove',e=>{{if(drag)scrubTo(e,c);}});
  c.addEventListener('mouseup',()=>drag=false);
  c.addEventListener('touchstart',e=>{{e.preventDefault();stop();scrubTo(e,c);}},{{passive:false}});
  c.addEventListener('touchmove',e=>{{e.preventDefault();scrubTo(e,c);}},{{passive:false}});
}});
playBtn.addEventListener('click',()=>playing?stop():start());

render();
}})();
</script>
"""


def show_empty_state() -> None:
    """Render the initial GUI state before an analysis is available."""

    st.markdown(
        """
        <div class="report-panel">
          <h3>Ready for analysis</h3>
          <p>
            Enter the user's height, check that the camera can see the full body,
            then start the guided drop-jump pipeline. The live camera preview
            appears in this dashboard during setup and capture.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Streamlit entry point."""

    logo_path = ASSETS / "image.png"
    logo_html = ""
    if logo_path.exists():
        logo_data = base64.b64encode(logo_path.read_bytes()).decode("ascii")
        logo_html = (
            f'<img class="hero-logo" src="data:image/png;base64,{logo_data}" '
            'alt="JumpGuard logo" />'
        )
    st.markdown(
        f"""
        <div class="hero">
          {logo_html}
          <div class="hero-copy">
            <h1>JumpGuard</h1>
            <p>Capture frontal drop jumps, validate protocol quality, and analyze the full movement pattern.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        height_text = st.text_input(
            "Height (cm)",
            value="183",
            max_chars=3,
        )
        audio_enabled = st.toggle("Voice guidance", value=True)
        start = st.button("Start Analysis", type="primary")
        st.markdown(
            '<p class="hint">The guided camera preview will appear here after you start the analysis.</p>',
            unsafe_allow_html=True,
        )

    source = "0"
    model_name = str(ROOT / "models" / "yolo26n-pose.pt")
    seconds = 8.0
    prepare_seconds = 2.0
    max_wait_seconds = 30.0
    min_drop_ratio = 0.06
    show_windows = False

    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "retake_requested" not in st.session_state:
        st.session_state.retake_requested = False

    run_requested = start or st.session_state.retake_requested
    if run_requested:
        st.session_state.retake_requested = False
        try:
            height_cm = float(height_text.strip().replace(",", "."))
        except ValueError:
            st.error("Enter height in centimeters, for example 183.")
            return
        if not 80.0 <= height_cm <= 250.0:
            st.error("Height must be between 80 and 250 cm.")
            return

        result = None
        error_message = None
        protocol_failure_metadata = None
        preview_placeholder = st.empty()
        with st.status("Running guided capture...", expanded=True) as status:
            st.write("Loading YOLO and opening the camera.")
            st.write("Follow the voice guidance and the live preview below.")
            try:
                result = run_jump_pipeline(
                    height_cm=height_cm,
                    source=source,
                    model_name=model_name,
                    seconds=seconds,
                    prepare_seconds=prepare_seconds,
                    max_wait_seconds=max_wait_seconds,
                    min_drop_ratio=min_drop_ratio,
                    audio_enabled=audio_enabled,
                    show_windows=show_windows,
                    preview_placeholder=preview_placeholder,
                )
            except ProtocolValidationError as exc:
                error_message = str(exc)
                protocol_failure_metadata = exc.metadata
                status.update(label="Analysis failed", state="error", expanded=True)
            except Exception as exc:
                error_message = str(exc)
                status.update(label="Analysis failed", state="error", expanded=True)
            else:
                status.update(label="Analysis complete", state="complete")
        if error_message is not None:
            st.error(error_message)
            if protocol_failure_metadata is not None:
                rows = failed_protocol_rows(protocol_failure_metadata)
                if rows:
                    st.markdown("### Why it failed")
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            if can_retake_after_error(error_message):
                if st.button("Retake", type="primary"):
                    st.session_state.retake_requested = True
                    st.rerun()
            return
        if result is not None:
            st.session_state.last_result = result

    if st.session_state.last_result is None:
        show_empty_state()
        return

    res             = st.session_state.last_result
    metadata        = res["metadata"]
    ae_metadata     = res.get("ae_metadata", metadata)
    pitch_series    = res.get("pitch_series")
    ae_pitch_series = res.get("ae_pitch_series")
    ae_score        = res.get("ae_score")
    ae_is_anomaly   = res.get("ae_is_anomaly")
    ae_threshold    = res.get("ae_threshold")
    ae_frame_errors = res.get("ae_frame_errors")
    ae_feature_errors = res.get("ae_feature_errors")
    ae_error        = res.get("ae_error")
    replay_ae_start = int(res.get("replay_ae_start", 0))
    protocol_ok     = bool(metadata["protocol_passed"])
    ae_anomalous    = ae_is_anomaly is True
    is_ok           = protocol_ok and not ae_anomalous

    if ae_error:
        st.warning(f"⚠ Movement pattern model unavailable: {ae_error}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    if ae_score is not None:
        col_a, col_b = st.columns(2)
        col_a.metric("Protocol",         "✓ OK" if protocol_ok else "⚠ CHECK")
        col_b.metric("Movement pattern", "⚠ ANOMALY" if ae_anomalous else "✓ NORMAL")
    else:
        st.metric("Protocol", "✓ OK" if protocol_ok else "⚠ CHECK")

    # ── Summary ───────────────────────────────────────────────────────────────
    st.markdown("### Summary")
    status_class = "status-ok" if is_ok else "status-warn"
    status_text  = "Movement looks normal." if is_ok else "Unusual movement pattern detected."
    proto_text   = protocol_label(metadata)

    explanation_lines = []
    if ae_anomalous and ae_pitch_series is not None:
        explanation_lines = detailed_anomaly_lines(
            pitch_series=ae_pitch_series,
            ae_feature_errors=ae_feature_errors,
            ae_threshold=ae_threshold,
            replay_ae_start=replay_ae_start,
        )

    explanation_html = "".join(f"<p style='margin:6px 0 0; color:var(--ink); font-size:0.92rem; line-height:1.5;'>{l}</p>" for l in explanation_lines)

    st.markdown(
        f"""
        <div class="report-panel">
          <p class="{status_class}">{status_text}</p>
          <p style="color:var(--muted); margin:4px 0 0;">{proto_text}</p>
          {explanation_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Pitch timeseries ──────────────────────────────────────────────────────
    if pitch_series is not None:
        import matplotlib.pyplot as plt, matplotlib
        matplotlib.rcParams.update({"font.family": "sans-serif"})

        ic = metadata["ic_valid_frame"]
        ae_start_plot = int(res.get("pitch_plot_start", 0))
        T  = len(pitch_series)
        plot_start = ae_start_plot
        plot_end = int(res.get("pitch_plot_end", T - 1))
        plot_start = max(0, min(plot_start, T - 1))
        plot_end = max(plot_start, min(plot_end, T - 1))
        display_len = plot_end - plot_start + 1

        pitch_window = pitch_series[plot_start:plot_end + 1].copy()
        if len(pitch_window):
            pitch_window = pitch_window - pitch_window[0]

        st.markdown("### Knee pitch over time")
        frames = np.arange(display_len)

        fig, ax = plt.subplots(figsize=(7, 2.0))
        fig.patch.set_facecolor("#1a1f3a")
        ax.set_facecolor("#1a1f3a")

        ax.plot(frames, pitch_window[:, 0], color="#4a90d9", lw=2, label="Left")
        ax.plot(frames, pitch_window[:, 1], color="#e05c5c", lw=2, label="Right")
        ax.axvline(0, color="#34d399", lw=1.2, linestyle="--", label="Start")
        ax.axvline(ic - plot_start, color="#fbbf24", lw=1.2, linestyle="--", label="Landing")
        ax.set_xlim(-0.5, display_len - 0.5)

        # Y axis: scale tightly to actual data so small differences look small
        y_all = np.concatenate([pitch_window[:, 0], pitch_window[:, 1]])
        pad = max((float(y_all.max()) - float(y_all.min())) * 0.12, 1.5)
        ax.set_ylim(float(y_all.min()) - pad, float(y_all.max()) + pad)

        if ae_frame_errors is not None and ae_threshold is not None:
            bad = ae_frame_errors[:display_len] > ae_threshold
            for t in np.where(bad)[0]:
                ax.axvspan(t - 0.5, t + 0.5, color="#ff6b6b", alpha=0.12, linewidth=0)

        ax.set_ylabel("Pitch (°)", color="#9ca3af", fontsize=8)
        ax.set_xlabel("Frame", color="#9ca3af", fontsize=8)
        ax.tick_params(colors="#9ca3af", labelsize=8)
        ax.spines[:].set_visible(False)
        ax.legend(fontsize=8, facecolor="#1f2937", labelcolor="#e5e7eb", edgecolor="#374151")
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Interactive stick figure ───────────────────────────────────────────────
    kp_seq_res       = res.get("kp_seq")
    stickman_kp_seq_res = res.get("stickman_kp_seq")
    ae_frame_errors_res  = res.get("ae_frame_errors")
    ae_feature_errors_res = res.get("ae_feature_errors")
    ae_threshold_res = res.get("ae_threshold")
    replay_metadata = res.get("replay_metadata", ae_metadata)
    replay_ae_start = int(res.get("replay_ae_start", 0))
    replay_ae_skip = int(res.get("replay_ae_skip", 0))

    if stickman_kp_seq_res is not None:
        st.markdown("### Movement replay")
        import streamlit.components.v1 as components

        display_len = len(stickman_kp_seq_res)
        display_threshold = float(ae_threshold_res) if ae_threshold_res is not None else 1.0
        display_frame_errors = np.zeros(display_len, dtype=np.float32)
        display_feature_errors = None
        if ae_frame_errors_res is not None:
            n = min(len(ae_frame_errors_res) - replay_ae_skip, max(0, display_len - replay_ae_start))
            if n > 0:
                display_frame_errors[replay_ae_start:replay_ae_start + n] = ae_frame_errors_res[replay_ae_skip:replay_ae_skip + n]

        if ae_feature_errors_res is not None:
            display_feature_errors = np.zeros((display_len, ae_feature_errors_res.shape[1]), dtype=np.float32)
            n = min(len(ae_feature_errors_res) - replay_ae_skip, max(0, display_len - replay_ae_start))
            if n > 0:
                display_feature_errors[replay_ae_start:replay_ae_start + n] = ae_feature_errors_res[replay_ae_skip:replay_ae_skip + n]

        html_widget = build_stickman_html(
            kp_seq=stickman_kp_seq_res,
            ae_frame_errors=display_frame_errors,
            ae_feature_errors=display_feature_errors,
            ae_threshold=display_threshold,
            metadata=replay_metadata,
            ae_start_in_replay=replay_ae_start,
            title="Your movement",
            subtitle="From drop trigger context to the end of the autoencoder window",
        )
        components.html(html_widget, height=470, scrolling=False)


if __name__ == "__main__":
    main()
