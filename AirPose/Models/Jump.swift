import Foundation

struct Jump: Identifiable, Codable, Equatable {
    let id: UUID
    let date: Date
    let videoURL: URL?
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
    let analysisSummary: String
    let initialContactLeftKneeAngleDeg: Double
    let initialContactRightKneeAngleDeg: Double
    let maxKneeFlexionLeftKneeAngleDeg: Double
    let maxKneeFlexionRightKneeAngleDeg: Double
    let landingAsymmetryRatio: Double
    let kneeAsymmetryRatio: Double

    init(
        id: UUID = UUID(),
        date: Date,
        videoURL: URL?,
        protocolPassed: Bool,
        prediction: String,
        anomalyScore: Double,
        outlierFeatureCount: Int,
        analyzedFeatureCount: Int,
        maxAbsRobustZ: Double,
        worstFeature: String,
        worstFeatureZ: Double,
        worstFeatureValue: Double,
        worstFeatureReferenceMedian: Double,
        validPoseFrames: Int,
        initialContactFrame: Int,
        maxKneeFlexionFrame: Int,
        videoFPS: Double,
        estimatedShoulderWidthCm: Double,
        analysisSummary: String,
        initialContactLeftKneeAngleDeg: Double,
        initialContactRightKneeAngleDeg: Double,
        maxKneeFlexionLeftKneeAngleDeg: Double,
        maxKneeFlexionRightKneeAngleDeg: Double,
        landingAsymmetryRatio: Double,
        kneeAsymmetryRatio: Double
    ) {
        self.id = id
        self.date = date
        self.videoURL = videoURL
        self.protocolPassed = protocolPassed
        self.prediction = prediction
        self.anomalyScore = anomalyScore
        self.outlierFeatureCount = outlierFeatureCount
        self.analyzedFeatureCount = analyzedFeatureCount
        self.maxAbsRobustZ = maxAbsRobustZ
        self.worstFeature = worstFeature
        self.worstFeatureZ = worstFeatureZ
        self.worstFeatureValue = worstFeatureValue
        self.worstFeatureReferenceMedian = worstFeatureReferenceMedian
        self.validPoseFrames = validPoseFrames
        self.initialContactFrame = initialContactFrame
        self.maxKneeFlexionFrame = maxKneeFlexionFrame
        self.videoFPS = videoFPS
        self.estimatedShoulderWidthCm = estimatedShoulderWidthCm
        self.analysisSummary = analysisSummary
        self.initialContactLeftKneeAngleDeg = initialContactLeftKneeAngleDeg
        self.initialContactRightKneeAngleDeg = initialContactRightKneeAngleDeg
        self.maxKneeFlexionLeftKneeAngleDeg = maxKneeFlexionLeftKneeAngleDeg
        self.maxKneeFlexionRightKneeAngleDeg = maxKneeFlexionRightKneeAngleDeg
        self.landingAsymmetryRatio = landingAsymmetryRatio
        self.kneeAsymmetryRatio = kneeAsymmetryRatio
    }

    var averageInitialContactKneeAngleDeg: Double {
        (initialContactLeftKneeAngleDeg + initialContactRightKneeAngleDeg) / 2.0
    }

    var averageMaxKneeFlexionKneeAngleDeg: Double {
        (maxKneeFlexionLeftKneeAngleDeg + maxKneeFlexionRightKneeAngleDeg) / 2.0
    }

    var isNormalPrediction: Bool {
        prediction.caseInsensitiveCompare("normal") == .orderedSame
    }

    var displayPrediction: String {
        isNormalPrediction ? "Normal" : prediction.capitalized
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case date
        case videoURL
        case protocolPassed
        case prediction
        case anomalyScore
        case outlierFeatureCount
        case analyzedFeatureCount
        case maxAbsRobustZ
        case worstFeature
        case worstFeatureZ
        case worstFeatureValue
        case worstFeatureReferenceMedian
        case validPoseFrames
        case initialContactFrame
        case maxKneeFlexionFrame
        case videoFPS
        case estimatedShoulderWidthCm
        case analysisSummary
        case initialContactLeftKneeAngleDeg
        case initialContactRightKneeAngleDeg
        case maxKneeFlexionLeftKneeAngleDeg
        case maxKneeFlexionRightKneeAngleDeg
        case landingAsymmetryRatio
        case kneeAsymmetryRatio
    }

    private enum LegacyCodingKeys: String, CodingKey {
        case llmFeedback
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let legacyContainer = try decoder.container(keyedBy: LegacyCodingKeys.self)

        id = try container.decodeIfPresent(UUID.self, forKey: .id) ?? UUID()
        date = try container.decodeIfPresent(Date.self, forKey: .date) ?? .now
        videoURL = try container.decodeIfPresent(URL.self, forKey: .videoURL)
        protocolPassed = try container.decodeIfPresent(Bool.self, forKey: .protocolPassed) ?? true
        prediction = try container.decodeIfPresent(String.self, forKey: .prediction) ?? "legacy"
        anomalyScore = try container.decodeIfPresent(Double.self, forKey: .anomalyScore) ?? 0
        outlierFeatureCount = try container.decodeIfPresent(Int.self, forKey: .outlierFeatureCount) ?? 0
        analyzedFeatureCount = try container.decodeIfPresent(Int.self, forKey: .analyzedFeatureCount) ?? 0
        maxAbsRobustZ = try container.decodeIfPresent(Double.self, forKey: .maxAbsRobustZ) ?? 0
        worstFeature = try container.decodeIfPresent(String.self, forKey: .worstFeature) ?? "n/a"
        worstFeatureZ = try container.decodeIfPresent(Double.self, forKey: .worstFeatureZ) ?? 0
        worstFeatureValue = try container.decodeIfPresent(Double.self, forKey: .worstFeatureValue) ?? 0
        worstFeatureReferenceMedian = try container.decodeIfPresent(Double.self, forKey: .worstFeatureReferenceMedian) ?? 0
        validPoseFrames = try container.decodeIfPresent(Int.self, forKey: .validPoseFrames) ?? 0
        initialContactFrame = try container.decodeIfPresent(Int.self, forKey: .initialContactFrame) ?? 0
        maxKneeFlexionFrame = try container.decodeIfPresent(Int.self, forKey: .maxKneeFlexionFrame) ?? 0
        videoFPS = try container.decodeIfPresent(Double.self, forKey: .videoFPS) ?? 0
        estimatedShoulderWidthCm = try container.decodeIfPresent(Double.self, forKey: .estimatedShoulderWidthCm) ?? 0
        analysisSummary = try container.decodeIfPresent(String.self, forKey: .analysisSummary)
            ?? legacyContainer.decodeIfPresent(String.self, forKey: .llmFeedback)
            ?? ""
        initialContactLeftKneeAngleDeg = try container.decodeIfPresent(Double.self, forKey: .initialContactLeftKneeAngleDeg) ?? 0
        initialContactRightKneeAngleDeg = try container.decodeIfPresent(Double.self, forKey: .initialContactRightKneeAngleDeg) ?? 0
        maxKneeFlexionLeftKneeAngleDeg = try container.decodeIfPresent(Double.self, forKey: .maxKneeFlexionLeftKneeAngleDeg) ?? 0
        maxKneeFlexionRightKneeAngleDeg = try container.decodeIfPresent(Double.self, forKey: .maxKneeFlexionRightKneeAngleDeg) ?? 0
        landingAsymmetryRatio = try container.decodeIfPresent(Double.self, forKey: .landingAsymmetryRatio) ?? 0
        kneeAsymmetryRatio = try container.decodeIfPresent(Double.self, forKey: .kneeAsymmetryRatio) ?? 0
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(date, forKey: .date)
        try container.encodeIfPresent(videoURL, forKey: .videoURL)
        try container.encode(protocolPassed, forKey: .protocolPassed)
        try container.encode(prediction, forKey: .prediction)
        try container.encode(anomalyScore, forKey: .anomalyScore)
        try container.encode(outlierFeatureCount, forKey: .outlierFeatureCount)
        try container.encode(analyzedFeatureCount, forKey: .analyzedFeatureCount)
        try container.encode(maxAbsRobustZ, forKey: .maxAbsRobustZ)
        try container.encode(worstFeature, forKey: .worstFeature)
        try container.encode(worstFeatureZ, forKey: .worstFeatureZ)
        try container.encode(worstFeatureValue, forKey: .worstFeatureValue)
        try container.encode(worstFeatureReferenceMedian, forKey: .worstFeatureReferenceMedian)
        try container.encode(validPoseFrames, forKey: .validPoseFrames)
        try container.encode(initialContactFrame, forKey: .initialContactFrame)
        try container.encode(maxKneeFlexionFrame, forKey: .maxKneeFlexionFrame)
        try container.encode(videoFPS, forKey: .videoFPS)
        try container.encode(estimatedShoulderWidthCm, forKey: .estimatedShoulderWidthCm)
        try container.encode(analysisSummary, forKey: .analysisSummary)
        try container.encode(initialContactLeftKneeAngleDeg, forKey: .initialContactLeftKneeAngleDeg)
        try container.encode(initialContactRightKneeAngleDeg, forKey: .initialContactRightKneeAngleDeg)
        try container.encode(maxKneeFlexionLeftKneeAngleDeg, forKey: .maxKneeFlexionLeftKneeAngleDeg)
        try container.encode(maxKneeFlexionRightKneeAngleDeg, forKey: .maxKneeFlexionRightKneeAngleDeg)
        try container.encode(landingAsymmetryRatio, forKey: .landingAsymmetryRatio)
        try container.encode(kneeAsymmetryRatio, forKey: .kneeAsymmetryRatio)
    }
}
