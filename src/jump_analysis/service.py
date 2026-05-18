from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, UTC
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from jump_analysis.models import RobustAnomalyModel
from jump_analysis.video import compare_to_reference, extract_front_features_from_yolo_frames
from jump_analysis.video.yolo_video import (
    YoloPoseFrame,
    estimate_person_height_px,
    normalize_keypoints_to_mocap_scale,
    required_visible,
    select_pose,
)


@dataclass
class JumpAnalysisResult:
    timestamp: datetime
    protocol_passed: bool
    prediction: str
    anomaly_score: float
    outlier_feature_count: int
    analyzed_feature_count: int
    max_abs_robust_z: float
    worst_feature: str
    worst_feature_z: float
    worst_feature_value: float
    worst_feature_reference_median: float
    valid_pose_frames: int
    initial_contact_frame: int
    max_knee_flexion_frame: int
    video_fps: float
    estimated_shoulder_width_cm: float
    summary: str
    initial_contact_left_knee_angle_deg: float
    initial_contact_right_knee_angle_deg: float
    max_knee_flexion_left_knee_angle_deg: float
    max_knee_flexion_right_knee_angle_deg: float
    landing_asymmetry_ratio: float
    knee_asymmetry_ratio: float

    def as_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


class VideoAnalysisService:
    def __init__(
        self,
        model_path: str | Path = "yolo26n-pose.pt",
        reference_csv: str | Path = "mocap_front_37_features.csv",
        z_threshold: float = 4.0,
        max_outlier_features: int = 8,
    ) -> None:
        self.model_path = str(model_path)
        self.reference_csv = Path(reference_csv)
        self.z_threshold = z_threshold
        self.max_outlier_features = max_outlier_features
        self._model: YOLO | None = None

    def analyze_video(self, video_path: str | Path, height_cm: float) -> JumpAnalysisResult:
        if height_cm <= 0:
            raise ValueError("height_cm must be positive.")

        video_path = Path(video_path)
        frames, fps, estimated_shoulder_width_m = self._extract_pose_frames(video_path, height_cm)
        features, metadata = extract_front_features_from_yolo_frames(frames)
        comparison = compare_to_reference(features, self.reference_csv)
        reference = pd.read_csv(self.reference_csv)
        anomaly_model = RobustAnomalyModel.fit_reference(
            reference,
            excluded_features=["crop_length_frames"],
            z_threshold=self.z_threshold,
            max_outlier_features=self.max_outlier_features,
        )
        analysis = anomaly_model.predict(pd.DataFrame([features])).iloc[0]

        summary = self._build_summary(metadata, analysis)

        return JumpAnalysisResult(
            timestamp=datetime.now(UTC),
            protocol_passed=bool(metadata["protocol_passed"]),
            prediction=str(analysis["prediction"]),
            anomaly_score=float(analysis["anomaly_score"]),
            outlier_feature_count=int(analysis["outlier_feature_count"]),
            analyzed_feature_count=int(analysis["analyzed_feature_count"]),
            max_abs_robust_z=float(analysis["max_abs_robust_z"]),
            worst_feature=str(analysis["worst_feature"]),
            worst_feature_z=float(analysis["worst_feature_z"]),
            worst_feature_value=float(analysis["worst_feature_value"]),
            worst_feature_reference_median=float(analysis["worst_feature_reference_median"]),
            valid_pose_frames=int(metadata["valid_pose_frames"]),
            initial_contact_frame=int(metadata["ic_raw_frame"]),
            max_knee_flexion_frame=int(metadata["kfmax_raw_frame"]),
            video_fps=float(fps),
            estimated_shoulder_width_cm=float(estimated_shoulder_width_m * 100.0),
            summary=summary,
            initial_contact_left_knee_angle_deg=float(features["ic_left_hip_knee_ankle_frontal_angle"]),
            initial_contact_right_knee_angle_deg=float(features["ic_right_hip_knee_ankle_frontal_angle"]),
            max_knee_flexion_left_knee_angle_deg=float(features["kfmax_left_hip_knee_ankle_frontal_angle"]),
            max_knee_flexion_right_knee_angle_deg=float(features["kfmax_right_hip_knee_ankle_frontal_angle"]),
            landing_asymmetry_ratio=float(features["ic_left_right_ankle_y_difference_ratio"]),
            knee_asymmetry_ratio=float(features["kfmax_left_right_knee_y_difference_ratio"]),
        )

    def _extract_pose_frames(self, video_path: Path, height_cm: float) -> tuple[list[YoloPoseFrame], float, float]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0

        model = self._get_model()
        raw_frames: list[tuple[int, np.ndarray, np.ndarray, np.ndarray | None, float, float]] = []
        locked_track_id = None
        frame_index = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                result = model.track(frame, stream=False, persist=True, verbose=False)[0]
                selection = select_pose(result, locked_track_id)
                if selection is None:
                    frame_index += 1
                    continue

                _, detected_track_id, kpts_xy, kpts_conf, box = selection
                if locked_track_id is None and detected_track_id is not None:
                    locked_track_id = detected_track_id
                if not required_visible(kpts_conf):
                    frame_index += 1
                    continue

                body_height_px = estimate_person_height_px(box, kpts_xy)
                if body_height_px <= 1e-6:
                    frame_index += 1
                    continue

                shoulder_width_px = float(np.linalg.norm(kpts_xy[5] - kpts_xy[6]))
                meters_per_pixel = (height_cm / 100.0) / body_height_px
                shoulder_width_m = shoulder_width_px * meters_per_pixel
                raw_frames.append((frame_index, kpts_xy.copy(), kpts_conf.copy(), box, body_height_px, shoulder_width_m))
                frame_index += 1
        finally:
            cap.release()

        if len(raw_frames) < 10:
            raise RuntimeError(
                f"Too few valid pose frames: {len(raw_frames)}. Make sure the full body stays visible for the whole jump."
            )

        estimated_shoulder_width_m = float(np.median([frame[5] for frame in raw_frames]))
        frames = [
            YoloPoseFrame(
                frame_index=raw_frame[0],
                keypoints_xy=normalize_keypoints_to_mocap_scale(raw_frame[1], estimated_shoulder_width_m),
                keypoints_conf=raw_frame[2],
                box_xyxy=raw_frame[3],
            )
            for raw_frame in raw_frames
        ]
        return frames, fps, estimated_shoulder_width_m

    def _get_model(self) -> YOLO:
        if self._model is None:
            self._model = YOLO(self.model_path)
        return self._model

    def _build_summary(self, metadata: dict[str, int], analysis: pd.Series) -> str:
        if not metadata["protocol_passed"]:
            return (
                "The movement did not pass the drop-jump protocol checks. "
                "Repeat the trial starting from a raised position, land on two feet, and perform the rebound jump immediately."
            )

        prediction = str(analysis["prediction"])
        worst_feature = str(analysis["worst_feature"]).replace("_", " ")
        if prediction == "normal":
            return (
                f"The jump stayed close to the reference dataset overall. "
                f"The largest deviation was {worst_feature}, but it remained within an acceptable range."
            )

        return (
            f"The jump differs from the reference dataset and should be reviewed. "
            f"The strongest deviation was {worst_feature}, with {int(analysis['outlier_feature_count'])} features outside the normal range."
        )
