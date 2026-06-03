from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, UTC
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from jump_analysis.imu import IMURecordingSummary, WitMotionRecordingFinder
from jump_analysis.models import RobustAnomalyModel
from jump_analysis.video import compare_to_reference, extract_front_features_from_yolo_frames
from jump_analysis.video.yolo_video import (
    YoloPoseFrame,
    ankle_mean_y,
    estimate_person_height_px,
    knee_flexion_proxy,
    normalize_keypoints_to_mocap_scale,
    required_visible,
    select_pose,
)
from jump_analysis.features.front_2d_features import body_keypoint

logger = logging.getLogger("jump_analysis.service")


def _protocol_check_names(metadata: dict[str, int | float]) -> list[str]:
    """Return protocol-check names in metadata insertion order."""

    names: list[str] = []
    for key in metadata:
        if key == "protocol_passed" or not key.endswith("_passed"):
            continue
        names.append(key.removesuffix("_passed"))
    return names


@dataclass
class ProtocolCheckResult:
    name: str
    passed: bool
    value: float
    threshold: float


@dataclass
class JumpGraphPoint:
    elapsed_time_s: float
    ankle_height_px: float
    body_height_px: float
    knee_flexion_proxy_deg: float


@dataclass
class JumpGraph:
    initial_contact_time_s: float
    max_knee_flexion_time_s: float
    points: list[JumpGraphPoint]


@dataclass
class JumpAnalysisResult:
    timestamp: datetime
    protocol_passed: bool
    protocol_checks: list[ProtocolCheckResult]
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
    jump_graph: JumpGraph | None = None
    imu_recording: IMURecordingSummary | None = None

    def as_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        if self.imu_recording is not None:
            payload["imu_recording"] = self.imu_recording.as_json()
        return payload


class VideoAnalysisService:
    def __init__(
        self,
        model_path: str | Path = "yolo26n-pose.pt",
        reference_csv: str | Path = "mocap_front_37_features.csv",
        z_threshold: float = 4.0,
        max_outlier_features: int = 8,
        witmotion_recordings_root: str | Path | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self.reference_csv = Path(reference_csv)
        self.z_threshold = z_threshold
        self.max_outlier_features = max_outlier_features
        self._model: YOLO | None = None
        self.recording_finder = WitMotionRecordingFinder(recordings_root=witmotion_recordings_root)

    def analyze_video(self, video_path: str | Path, height_cm: float) -> JumpAnalysisResult:
        return self.analyze_video_with_timestamp(video_path, height_cm=height_cm, recording_started_at=None)

    def analyze_video_with_timestamp(
        self,
        video_path: str | Path,
        height_cm: float,
        recording_started_at: datetime | None,
    ) -> JumpAnalysisResult:
        if height_cm <= 0:
            raise ValueError("height_cm must be positive.")

        analysis_started_at = datetime.now(UTC)
        video_path = Path(video_path)
        logger.info(
            "Analysis started video_path=%s height_cm=%.2f analysis_started_at=%s recording_started_at=%s",
            video_path,
            height_cm,
            analysis_started_at.isoformat(),
            recording_started_at.astimezone().isoformat() if recording_started_at is not None else "none",
        )
        frames, fps, estimated_shoulder_width_m = self._extract_pose_frames(video_path, height_cm)
        logger.info(
            "Video features: extracted valid_frames=%s fps=%.3f estimated_shoulder_width_cm=%.2f",
            len(frames),
            fps,
            estimated_shoulder_width_m * 100.0,
        )
        return self.analyze_pose_frames(
            frames,
            fps=fps,
            estimated_shoulder_width_m=estimated_shoulder_width_m,
            recording_started_at=recording_started_at,
            analysis_started_at=analysis_started_at,
        )

    def analyze_pose_frames(
        self,
        frames: list[YoloPoseFrame],
        fps: float,
        estimated_shoulder_width_m: float,
        recording_started_at: datetime | None,
        analysis_started_at: datetime | None = None,
    ) -> JumpAnalysisResult:
        if analysis_started_at is None:
            analysis_started_at = datetime.now(UTC)

        features, metadata = extract_front_features_from_yolo_frames(frames)
        compare_to_reference(features, self.reference_csv)
        reference = pd.read_csv(self.reference_csv)
        anomaly_model = RobustAnomalyModel.fit_reference(
            reference,
            excluded_features=["crop_length_frames"],
            z_threshold=self.z_threshold,
            max_outlier_features=self.max_outlier_features,
        )
        analysis = anomaly_model.predict(pd.DataFrame([features])).iloc[0]
        imu_reference_time = (recording_started_at or analysis_started_at).astimezone()
        logger.info("IMU lookup: using reference_time=%s", imu_reference_time.isoformat())
        imu_recording = self.recording_finder.find_matching_recording(imu_reference_time)
        if imu_recording is not None:
            logger.info(
                "IMU attach: matched_file=%s device_count=%s total_samples=%s offset_seconds=%.3f",
                imu_recording.matched_file,
                imu_recording.device_count,
                imu_recording.total_samples,
                imu_recording.time_offset_seconds,
            )
            logger.info(
                "IMU merge step: no feature-level fusion yet, attaching IMU summary metadata to analysis result",
            )
        else:
            logger.info("IMU attach: no matching recording found, analysis will be returned with video-only metrics")

        summary = self._build_summary(metadata, analysis, imu_recording)
        protocol_check_names = _protocol_check_names(metadata)
        for check_name in protocol_check_names:
            logger.info(
                "Protocol check %s: passed=%s value=%.3f threshold=%.3f",
                check_name,
                bool(metadata.get(f"{check_name}_passed", 0)),
                float(metadata.get(f"{check_name}_value", 0.0)),
                float(metadata.get(f"{check_name}_threshold", 0.0)),
            )
        logger.info(
            "Analysis completed prediction=%s protocol_passed=%s anomaly_score=%.3f worst_feature=%s",
            analysis["prediction"],
            bool(metadata["protocol_passed"]),
            float(analysis["anomaly_score"]),
            analysis["worst_feature"],
        )
        jump_graph = self._build_jump_graph(
            frames,
            initial_contact_index=int(metadata["ic_valid_frame"]),
            max_knee_flexion_index=int(metadata["kfmax_valid_frame"]),
        )

        return JumpAnalysisResult(
            timestamp=analysis_started_at,
            protocol_passed=bool(metadata["protocol_passed"]),
            protocol_checks=[
                ProtocolCheckResult(
                    name=check_name,
                    passed=bool(metadata.get(f"{check_name}_passed", 0)),
                    value=float(metadata.get(f"{check_name}_value", 0.0)),
                    threshold=float(metadata.get(f"{check_name}_threshold", 0.0)),
                )
                for check_name in protocol_check_names
            ],
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
            jump_graph=jump_graph,
            imu_recording=imu_recording,
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
                timestamp_s=float(raw_frame[0]) / max(fps, 1e-6),
            )
            for raw_frame in raw_frames
        ]
        return frames, fps, estimated_shoulder_width_m

    def _build_jump_graph(
        self,
        frames: list[YoloPoseFrame],
        initial_contact_index: int,
        max_knee_flexion_index: int,
    ) -> JumpGraph | None:
        if not frames:
            return None

        start_time_s = float(frames[0].timestamp_s)
        landing_ankle_y = ankle_mean_y(frames[initial_contact_index])
        landing_body_y = float(body_keypoint(frames[initial_contact_index].keypoints_xy)[1])
        points = [
            JumpGraphPoint(
                elapsed_time_s=max(float(frame.timestamp_s) - start_time_s, 0.0),
                ankle_height_px=landing_ankle_y - ankle_mean_y(frame),
                body_height_px=landing_body_y - float(body_keypoint(frame.keypoints_xy)[1]),
                knee_flexion_proxy_deg=knee_flexion_proxy(frame),
            )
            for frame in frames
        ]
        return JumpGraph(
            initial_contact_time_s=points[initial_contact_index].elapsed_time_s,
            max_knee_flexion_time_s=points[max_knee_flexion_index].elapsed_time_s,
            points=points,
        )

    def _get_model(self) -> YOLO:
        if self._model is None:
            self._model = YOLO(self.model_path)
        return self._model

    def _build_summary(
        self,
        metadata: dict[str, int],
        analysis: pd.Series,
        imu_recording: IMURecordingSummary | None,
    ) -> str:
        if not metadata["protocol_passed"]:
            summary = (
                "The movement did not pass the drop-jump protocol checks. "
                "Repeat the trial starting from a raised position, land on two feet, and perform the rebound jump immediately."
            )
            if imu_recording is not None:
                summary += f" {imu_recording.short_summary()}"
            return summary

        prediction = str(analysis["prediction"])
        worst_feature = str(analysis["worst_feature"]).replace("_", " ")
        if prediction == "normal":
            summary = (
                f"The jump stayed close to the reference dataset overall. "
                f"The largest deviation was {worst_feature}, but it remained within an acceptable range."
            )
            if imu_recording is not None:
                summary += f" {imu_recording.short_summary()}"
            return summary

        summary = (
            f"The jump differs from the reference dataset and should be reviewed. "
            f"The strongest deviation was {worst_feature}, with {int(analysis['outlier_feature_count'])} features outside the normal range."
        )
        if imu_recording is not None:
            summary += f" {imu_recording.short_summary()}"
        return summary
