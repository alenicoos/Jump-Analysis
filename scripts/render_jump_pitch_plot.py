from __future__ import annotations

"""Render the Streamlit knee-pitch plot for one jump video.

This script reproduces the plot block from ``scripts/streamlit_app.py``:
- left/right pitch over time
- dashed markers for landing and deepest flexion
- optional anomaly-frame shading when a compatible autoencoder is available

Usage
-----
    env PYTHONPATH=src .venv/bin/python scripts/render_jump_pitch_plot.py \
        --video path/to/jump.mp4 \
        --height-cm 180 \
        --output outputs/plots/jump_pitch.png
"""

import argparse
import os
import sys
from pathlib import Path

MPLCONFIGDIR = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jump_analysis.features.front_2d_features import AE_FEATURE_DIM, extract_ae_features
from jump_analysis.service import VideoAnalysisService
from jump_analysis.video import extract_front_features_from_yolo_frames, frames_to_transformer_input


def _compatible_frame_errors(
    service: VideoAnalysisService,
    frames,
    pitch_series: np.ndarray,
) -> tuple[np.ndarray | None, float | None]:
    """Return frame errors when the available AE checkpoint matches a known input path."""

    try:
        model = service._get_jump_model()
    except Exception:
        return None, None

    input_dim = int(model.input_dim)
    threshold = float(model.anomaly_threshold or 0.0)

    if input_dim == AE_FEATURE_DIM:
        ae_input = frames_to_transformer_input(frames, input_dim=24)
        sequence = extract_ae_features(ae_input)
    else:
        kp_sequence = frames_to_transformer_input(frames)
        concatenated = np.concatenate([kp_sequence, pitch_series], axis=1)
        if input_dim != concatenated.shape[1]:
            return None, None
        sequence = concatenated

    errors = model.frame_errors_numpy(sequence, device=service.inference_device)
    return errors, threshold


def render_plot(
    video_path: Path,
    height_cm: float,
    output_path: Path,
    pitch_model_path: Path | None,
    jump_model_path: Path | None,
) -> None:
    service = VideoAnalysisService(
        pitch_model_path=pitch_model_path,
        jump_model_path=jump_model_path,
    )
    frames, _fps, _shoulder_width_m = service._extract_pose_frames(video_path, height_cm)
    _features, metadata = extract_front_features_from_yolo_frames(frames)
    if not metadata["protocol_passed"]:
        raise RuntimeError("The movement did not pass the drop-jump protocol checks.")

    kp_seq = frames_to_transformer_input(frames)
    pitch_series = service._predict_pitch_series(kp_seq)
    ae_frame_errors, ae_threshold = _compatible_frame_errors(service, frames, pitch_series)

    ic = int(metadata["ic_valid_frame"])
    kf = int(metadata["kfmax_valid_frame"])
    frame_index = np.arange(len(pitch_series))

    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_facecolor("#1a1f3a")
    ax.set_facecolor("#1a1f3a")

    ax.plot(frame_index, pitch_series[:, 0], color="#4a90d9", lw=2, label="Left")
    ax.plot(frame_index, pitch_series[:, 1], color="#e05c5c", lw=2, label="Right")
    ax.axvline(ic, color="#fbbf24", lw=1.2, linestyle="--", label="Landing")
    ax.axvline(kf, color="#34d399", lw=1.2, linestyle="--", label="Deepest point")
    ax.axhline(0, color="#555577", lw=0.7)

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to the recorded jump video.")
    parser.add_argument("--height-cm", required=True, type=float, help="Athlete height in centimeters.")
    parser.add_argument("--output", default="outputs/plots/jump_pitch_plot.png", help="Output PNG path.")
    parser.add_argument("--pitch-model", default=None, help="Optional override for pitch_transformer.pt.")
    parser.add_argument("--jump-model", default=None, help="Optional override for autoencoder checkpoint.")
    args = parser.parse_args()

    render_plot(
        video_path=Path(args.video),
        height_cm=float(args.height_cm),
        output_path=Path(args.output),
        pitch_model_path=Path(args.pitch_model) if args.pitch_model else None,
        jump_model_path=Path(args.jump_model) if args.jump_model else None,
    )
    print(f"Jump pitch plot saved to {args.output}")


if __name__ == "__main__":
    main()
