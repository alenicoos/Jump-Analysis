from __future__ import annotations

import unittest
from datetime import UTC, datetime

from jump_analysis.imu import IMUDeviceSummary, IMURecordingSummary
from jump_analysis.service import JumpAnalysisResult, ProtocolCheckResult, _protocol_check_names


class ServiceTests(unittest.TestCase):
    def test_protocol_check_names_preserve_metadata_order(self) -> None:
        metadata = {
            "valid_pose_frames": 12,
            "protocol_passed": 1,
            "drop_started_from_height_passed": 1,
            "drop_started_from_height_value": 1.0,
            "two_foot_contact_passed": 0,
            "second_jump_passed": 1,
            "stable_after_second_landing_passed": 1,
        }

        self.assertEqual(
            _protocol_check_names(metadata),
            [
                "drop_started_from_height",
                "two_foot_contact",
                "second_jump",
                "stable_after_second_landing",
            ],
        )

    def test_analysis_result_as_json_keeps_nested_imu_payload(self) -> None:
        imu = IMURecordingSummary(
            matched_file="recording.txt",
            matched_folder="/tmp",
            recording_start_time=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
            recording_end_time=datetime(2026, 6, 1, 12, 0, 5, tzinfo=UTC),
            time_offset_seconds=0.25,
            device_count=1,
            total_samples=42,
            device_summaries=[
                IMUDeviceSummary(
                    device_name="Left",
                    sample_count=42,
                    start_time=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
                    end_time=datetime(2026, 6, 1, 12, 0, 5, tzinfo=UTC),
                    duration_seconds=5.0,
                    mean_acceleration_g=1.1,
                    peak_acceleration_g=2.2,
                    mean_angular_velocity_dps=3.3,
                    peak_angular_velocity_dps=4.4,
                    mean_roll_deg=5.5,
                    mean_pitch_deg=6.6,
                    mean_yaw_deg=7.7,
                    mean_temperature_c=20.0,
                    battery_start_percent=90,
                    battery_end_percent=88,
                )
            ],
        )
        result = JumpAnalysisResult(
            timestamp=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
            protocol_passed=True,
            protocol_checks=[ProtocolCheckResult(name="second_jump", passed=True, value=1.0, threshold=0.5)],
            prediction="normal",
            anomaly_score=1.0,
            outlier_feature_count=0,
            analyzed_feature_count=36,
            max_abs_robust_z=1.2,
            worst_feature="feature",
            worst_feature_z=0.3,
            worst_feature_value=0.4,
            worst_feature_reference_median=0.5,
            valid_pose_frames=10,
            initial_contact_frame=2,
            max_knee_flexion_frame=5,
            video_fps=30.0,
            estimated_shoulder_width_cm=40.0,
            summary="ok",
            initial_contact_left_knee_angle_deg=170.0,
            initial_contact_right_knee_angle_deg=171.0,
            max_knee_flexion_left_knee_angle_deg=120.0,
            max_knee_flexion_right_knee_angle_deg=121.0,
            landing_asymmetry_ratio=0.1,
            knee_asymmetry_ratio=0.2,
            imu_recording=imu,
        )

        payload = result.as_json()

        self.assertEqual(payload["imu_recording"]["matched_file"], "recording.txt")
        self.assertEqual(payload["imu_recording"]["device_summaries"][0]["device_name"], "Left")


if __name__ == "__main__":
    unittest.main()
