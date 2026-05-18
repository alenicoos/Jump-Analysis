import Foundation

struct JumpAnalysisResponse: Codable, Equatable {
    let timestamp: Date
    let protocolPassed: Bool
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

    private enum CodingKeys: String, CodingKey {
        case timestamp
        case protocolPassed = "protocol_passed"
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
    }
}
