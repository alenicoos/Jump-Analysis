from __future__ import annotations

# User-facing Streamlit GUI for the jump-analysis pipeline.

import base64
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


class ProtocolValidationError(RuntimeError):
    """Raised when movement capture succeeded but protocol checks failed."""

    def __init__(self, metadata: dict[str, int | float]) -> None:
        super().__init__(PROTOCOL_ERROR_MESSAGE)
        self.metadata = metadata

from jump_analysis.feedback import AudioFeedback
from jump_analysis.video import (
    analyze_yolo_pose_frames,
    capture_yolo_pose_frames_with_open_capture,
    frames_to_transformer_input,
    run_floor_box_setup_with_open_capture,
)



st.set_page_config(
    page_title="AirPose",
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

    # ── PitchTransformer ──────────────────────────────────────────────────────
    pitch_series  = None
    pitch_model   = None
    pitch_error   = None
    pitch_model_path = ROOT / "models" / "pitch_transformer.pt"
    if not pitch_model_path.exists():
        pitch_error = f"pitch_transformer.pt not found in {pitch_model_path.parent}"
    else:
        try:
            from jump_analysis.models.pitch_transformer import PitchTransformer
            pitch_model  = PitchTransformer.load(str(pitch_model_path))
            kp_seq       = frames_to_transformer_input(frames)
            pitch_series = pitch_model.predict_numpy(kp_seq)
        except Exception as e:
            pitch_error = f"PitchTransformer error: {e}"

    # ── JumpAutoencoder ───────────────────────────────────────────────────────
    ae_score       = None
    ae_is_anomaly  = None
    ae_threshold   = None
    ae_frame_errors= None
    ae_error       = None
    ae_model_path  = ROOT / "models" / "jump_autoencoder.pt"
    if pitch_series is None:
        ae_error = pitch_error or "Pitch model not available"
    elif not ae_model_path.exists():
        ae_error = f"jump_autoencoder.pt not found in {ae_model_path.parent}"
    else:
        try:
            from jump_analysis.models.jump_autoencoder import JumpAutoencoder
            ae_model       = JumpAutoencoder.load(str(ae_model_path))
            kp_seq         = frames_to_transformer_input(frames)
            seq36          = np.concatenate([kp_seq, pitch_series], axis=1)
            ae_is_anomaly, ae_score = ae_model.is_anomaly(seq36)
            ae_threshold   = ae_model.anomaly_threshold
            ae_frame_errors= ae_model.frame_errors_numpy(seq36)   # (T,)
        except Exception as e:
            ae_error = f"JumpAutoencoder error: {e}"

    return {
        "metadata":        metadata,
        "pitch_series":    pitch_series,
        "ae_score":        ae_score,
        "ae_is_anomaly":   ae_is_anomaly,
        "ae_threshold":    ae_threshold,
        "ae_frame_errors": ae_frame_errors,
        "ae_error":        ae_error,
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

    logo_data = base64.b64encode((ASSETS / "image.png").read_bytes()).decode("ascii")
    st.markdown(
        f"""
        <div class="hero">
          <img class="hero-logo" src="data:image/png;base64,{logo_data}" alt="AirPose logo" />
          <div class="hero-copy">
            <h1>AirPose</h1>
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
    pitch_series    = res.get("pitch_series")
    ae_score        = res.get("ae_score")
    ae_is_anomaly   = res.get("ae_is_anomaly")
    ae_threshold    = res.get("ae_threshold")
    ae_frame_errors = res.get("ae_frame_errors")
    ae_error        = res.get("ae_error")
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

    # Spiegazione basata su frame errors e pitch (analisi dell'intero movimento)
    explanation_lines = []
    if ae_anomalous and ae_frame_errors is not None and pitch_series is not None:
        ic = metadata["ic_valid_frame"]
        kf = metadata["kfmax_valid_frame"]
        T  = len(ae_frame_errors)

        # ── Fase con più errore ───────────────────────────────────────────────
        phase_slices = {
            "drop from box":            ae_frame_errors[:max(1, ic)],
            "landing and flexion":      ae_frame_errors[ic:max(ic+1, kf)],
            "push-off and rebound":     ae_frame_errors[kf:],
        }
        worst_phase = max(phase_slices, key=lambda p: phase_slices[p].mean() if len(phase_slices[p]) > 0 else 0)
        explanation_lines.append(f"The most unusual part of the movement is the <b>{worst_phase}</b> phase.")

        # ── Asimmetria sinistra/destra sull'intero movimento ─────────────────
        asym_series = np.abs(pitch_series[:, 0] - pitch_series[:, 1])  # (T,)
        mean_asym   = float(asym_series.mean())
        max_asym    = float(asym_series.max())
        max_asym_t  = int(np.argmax(asym_series))
        if mean_asym > 6:
            # Quale lato è in media più in flessione?
            mean_l = float(np.abs(pitch_series[:, 0]).mean())
            mean_r = float(np.abs(pitch_series[:, 1]).mean())
            dominant = "left" if mean_l > mean_r else "right"
            explanation_lines.append(
                f"There is a persistent left-right asymmetry throughout the movement "
                f"(average {mean_asym:.0f}°, peak {max_asym:.0f}°). "
                f"The <b>{dominant}</b> side consistently shows greater knee pitch."
            )
        elif max_asym > 10:
            # Asimmetria transitoria
            if max_asym_t < ic:
                when = "during the descent"
            elif max_asym_t < kf:
                when = "during the flexion"
            else:
                when = "during the push-off"
            explanation_lines.append(
                f"A transient left-right asymmetry peaks at {max_asym:.0f}° {when}."
            )

        # ── Profondità media di flessione sull'intero movimento ───────────────
        flexion_window = pitch_series[ic:kf+1] if kf > ic else pitch_series[ic:ic+1]
        avg_depth_l = float(np.abs(flexion_window[:, 0]).mean())
        avg_depth_r = float(np.abs(flexion_window[:, 1]).mean())
        avg_depth   = (avg_depth_l + avg_depth_r) / 2
        if avg_depth < 12:
            explanation_lines.append("Knee flexion throughout the landing phase is very shallow — the landing is stiff.")
        elif avg_depth > 55:
            explanation_lines.append("Knee flexion throughout the landing phase is unusually deep.")

        # ── Smoothness: variabilità dell'errore ───────────────────────────────
        err_std = float(ae_frame_errors.std())
        if err_std > ae_frame_errors.mean() * 0.8:
            explanation_lines.append(
                "The reconstruction error is highly variable across frames, suggesting an abrupt or jerky movement rather than a smooth, controlled pattern."
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
        kf = metadata["kfmax_valid_frame"]
        T  = len(pitch_series)

        st.markdown("### Knee pitch over time")
        fig, ax = plt.subplots(figsize=(7, 2.8))
        fig.patch.set_facecolor("#1a1f3a")
        ax.set_facecolor("#1a1f3a")

        frames = np.arange(T)
        ax.plot(frames, pitch_series[:, 0], color="#4a90d9", lw=2, label="Left")
        ax.plot(frames, pitch_series[:, 1], color="#e05c5c", lw=2, label="Right")
        ax.axvline(ic, color="#fbbf24", lw=1.2, linestyle="--", label="Landing")
        ax.axvline(kf, color="#34d399", lw=1.2, linestyle="--", label="Deepest point")
        ax.axhline(0, color="#555577", lw=0.7)

        # Shade anomalous frames if available
        if ae_frame_errors is not None and ae_threshold is not None:
            bad = ae_frame_errors > ae_threshold * 0.5
            for t in np.where(bad)[0]:
                ax.axvspan(t - 0.5, t + 0.5, color="#ff6b6b", alpha=0.12, linewidth=0)

        ax.set_ylabel("Δ pitch (°)", color="#9ca3af", fontsize=8)
        ax.set_xlabel("Frame", color="#9ca3af", fontsize=8)
        ax.tick_params(colors="#9ca3af", labelsize=8)
        ax.spines[:].set_visible(False)
        ax.legend(fontsize=8, facecolor="#1f2937", labelcolor="#e5e7eb", edgecolor="#374151")
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


if __name__ == "__main__":
    main()
