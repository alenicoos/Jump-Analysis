import Foundation

struct JumpAnalysisResponse: Codable, Equatable {
    struct JumpGraphPoint: Codable, Equatable, Identifiable {
        let elapsedTimeS: Double
        let ankleHeightPx: Double
        let bodyHeightPx: Double
        let kneeFlexionProxyDeg: Double

        var id: Double { elapsedTimeS }

        private enum CodingKeys: String, CodingKey {
            case elapsedTimeS = "elapsed_time_s"
            case ankleHeightPx = "ankle_height_px"
            case bodyHeightPx = "body_height_px"
            case kneeFlexionProxyDeg = "knee_flexion_proxy_deg"
        }
    }

    struct JumpGraph: Codable, Equatable {
        let initialContactTimeS: Double
        let maxKneeFlexionTimeS: Double
        let points: [JumpGraphPoint]

        private enum CodingKeys: String, CodingKey {
            case initialContactTimeS = "initial_contact_time_s"
            case maxKneeFlexionTimeS = "max_knee_flexion_time_s"
            case points
        }
    }

    struct ProtocolCheckSummary: Codable, Equatable {
        let name: String
        let passed: Bool
        let value: Double
        let threshold: Double
    }

    struct IMUDeviceSummary: Codable, Equatable {
        let deviceName: String
        let sampleCount: Int
        let startTime: Date
        let endTime: Date
        let durationSeconds: Double
        let meanAccelerationG: Double?
        let peakAccelerationG: Double?
        let meanAngularVelocityDps: Double?
        let peakAngularVelocityDps: Double?
        let meanRollDeg: Double?
        let meanPitchDeg: Double?
        let meanYawDeg: Double?
        let meanTemperatureC: Double?
        let batteryStartPercent: Int?
        let batteryEndPercent: Int?

        private enum CodingKeys: String, CodingKey {
            case deviceName = "device_name"
            case sampleCount = "sample_count"
            case startTime = "start_time"
            case endTime = "end_time"
            case durationSeconds = "duration_seconds"
            case meanAccelerationG = "mean_acceleration_g"
            case peakAccelerationG = "peak_acceleration_g"
            case meanAngularVelocityDps = "mean_angular_velocity_dps"
            case peakAngularVelocityDps = "peak_angular_velocity_dps"
            case meanRollDeg = "mean_roll_deg"
            case meanPitchDeg = "mean_pitch_deg"
            case meanYawDeg = "mean_yaw_deg"
            case meanTemperatureC = "mean_temperature_c"
            case batteryStartPercent = "battery_start_percent"
            case batteryEndPercent = "battery_end_percent"
        }
    }

    struct IMURecordingSummary: Codable, Equatable {
        let matchedFile: String
        let matchedFolder: String
        let recordingStartTime: Date
        let recordingEndTime: Date
        let timeOffsetSeconds: Double
        let deviceCount: Int
        let totalSamples: Int
        let deviceSummaries: [IMUDeviceSummary]

        private enum CodingKeys: String, CodingKey {
            case matchedFile = "matched_file"
            case matchedFolder = "matched_folder"
            case recordingStartTime = "recording_start_time"
            case recordingEndTime = "recording_end_time"
            case timeOffsetSeconds = "time_offset_seconds"
            case deviceCount = "device_count"
            case totalSamples = "total_samples"
            case deviceSummaries = "device_summaries"
        }
    }

    let timestamp: Date
    let protocolPassed: Bool
    let protocolChecks: [ProtocolCheckSummary]
    let prediction: String
    let anomalyScore: Double
    let outlierFeatureCount: Int
    let analyzedFeatureCount: Int
    let maxAbsRobustZ: Double
    let worstFeature: String
    let worstFeatureZ: Double
    let worstFeatureValue: Double
    let worstFeatureReferenceMedian: Double
    let validPoseFrames: Int
    let initialContactFrame: Int
    let maxKneeFlexionFrame: Int
    let videoFPS: Double
    let estimatedShoulderWidthCm: Double
    let summary: String
    let initialContactLeftKneeAngleDeg: Double
    let initialContactRightKneeAngleDeg: Double
    let maxKneeFlexionLeftKneeAngleDeg: Double
    let maxKneeFlexionRightKneeAngleDeg: Double
    let landingAsymmetryRatio: Double
    let kneeAsymmetryRatio: Double
    let jumpGraph: JumpGraph?
    let imuRecording: IMURecordingSummary?

    private enum CodingKeys: String, CodingKey {
        case timestamp
        case protocolPassed = "protocol_passed"
        case protocolChecks = "protocol_checks"
        case prediction
        case anomalyScore = "anomaly_score"
        case outlierFeatureCount = "outlier_feature_count"
        case analyzedFeatureCount = "analyzed_feature_count"
        case maxAbsRobustZ = "max_abs_robust_z"
        case worstFeature = "worst_feature"
        case worstFeatureZ = "worst_feature_z"
        case worstFeatureValue = "worst_feature_value"
        case worstFeatureReferenceMedian = "worst_feature_reference_median"
        case validPoseFrames = "valid_pose_frames"
        case initialContactFrame = "initial_contact_frame"
        case maxKneeFlexionFrame = "max_knee_flexion_frame"
        case videoFPS = "video_fps"
        case estimatedShoulderWidthCm = "estimated_shoulder_width_cm"
        case summary
        case initialContactLeftKneeAngleDeg = "initial_contact_left_knee_angle_deg"
        case initialContactRightKneeAngleDeg = "initial_contact_right_knee_angle_deg"
        case maxKneeFlexionLeftKneeAngleDeg = "max_knee_flexion_left_knee_angle_deg"
        case maxKneeFlexionRightKneeAngleDeg = "max_knee_flexion_right_knee_angle_deg"
        case landingAsymmetryRatio = "landing_asymmetry_ratio"
        case kneeAsymmetryRatio = "knee_asymmetry_ratio"
        case jumpGraph = "jump_graph"
        case imuRecording = "imu_recording"
    }
}
