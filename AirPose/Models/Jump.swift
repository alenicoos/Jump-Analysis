import Foundation

struct JumpProtocolCheck: Codable, Equatable, Identifiable {
    let name: String
    let passed: Bool
    let value: Double
    let threshold: Double

    var id: String { name }

    var title: String {
        switch name {
        case "drop_started_from_height":
            return "Started From Box"
        case "two_foot_contact":
            return "Two-Foot Landing"
        case "second_jump":
            return "Rebound Jump"
        case "stable_after_second_landing":
            return "Stable Second Landing"
        default:
            return name.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }

    var detailText: String {
        switch name {
        case "drop_started_from_height":
            return "Drop \(value.airPoseProtocolValueString) vs threshold \(threshold.airPoseProtocolValueString)"
        case "two_foot_contact":
            return "Ankle difference \(value.airPoseProtocolValueString) vs max \(threshold.airPoseProtocolValueString)"
        case "second_jump":
            return "Lift \(value.airPoseProtocolValueString) vs threshold \(threshold.airPoseProtocolValueString)"
        case "stable_after_second_landing":
            return "Post-landing drop \(value.airPoseProtocolValueString) vs max \(threshold.airPoseProtocolValueString)"
        default:
            return "Value \(value.airPoseProtocolValueString) vs threshold \(threshold.airPoseProtocolValueString)"
        }
    }
}

struct JumpAthleteProfile: Codable, Equatable {
    let name: String
    let age: String
    let height: String
    let weight: String
    let dominantLeg: DominantLeg
    let sport: String
    let experienceLevel: ExperienceLevel
    let source: Source

    enum Source: String, Codable {
        case accountProfile
        case guestAthlete
    }

    init(profile: UserProfile, source: Source) {
        name = profile.name
        age = profile.age
        height = profile.height
        weight = profile.weight
        dominantLeg = profile.dominantLeg
        sport = profile.sport
        experienceLevel = profile.experienceLevel
        self.source = source
    }

    var displayName: String {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? "Unnamed Athlete" : trimmed
    }

    var summaryText: String {
        let trimmedSport = sport.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedSport.isEmpty {
            return experienceLevel.rawValue
        }
        return "\(experienceLevel.rawValue) \(trimmedSport)"
    }
}

struct Jump: Identifiable, Codable, Equatable {
    let id: UUID
    let date: Date
    let videoURL: URL?
    let athleteProfile: JumpAthleteProfile?
    let protocolPassed: Bool
    let protocolChecks: [JumpProtocolCheck]
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
    let llmNarratedSummary: Bool
    let initialContactLeftKneeAngleDeg: Double
    let initialContactRightKneeAngleDeg: Double
    let maxKneeFlexionLeftKneeAngleDeg: Double
    let maxKneeFlexionRightKneeAngleDeg: Double
    let landingAsymmetryRatio: Double
    let kneeAsymmetryRatio: Double
    let jumpGraph: JumpAnalysisResponse.JumpGraph?
    let imuRecording: JumpAnalysisResponse.IMURecordingSummary?

    init(
        id: UUID = UUID(),
        date: Date,
        videoURL: URL?,
        athleteProfile: JumpAthleteProfile?,
        protocolPassed: Bool,
        protocolChecks: [JumpProtocolCheck] = [],
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
        llmNarratedSummary: Bool = false,
        initialContactLeftKneeAngleDeg: Double,
        initialContactRightKneeAngleDeg: Double,
        maxKneeFlexionLeftKneeAngleDeg: Double,
        maxKneeFlexionRightKneeAngleDeg: Double,
        landingAsymmetryRatio: Double,
        kneeAsymmetryRatio: Double,
        jumpGraph: JumpAnalysisResponse.JumpGraph? = nil,
        imuRecording: JumpAnalysisResponse.IMURecordingSummary? = nil
    ) {
        self.id = id
        self.date = date
        self.videoURL = videoURL
        self.athleteProfile = athleteProfile
        self.protocolPassed = protocolPassed
        self.protocolChecks = protocolChecks
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
        self.llmNarratedSummary = llmNarratedSummary
        self.initialContactLeftKneeAngleDeg = initialContactLeftKneeAngleDeg
        self.initialContactRightKneeAngleDeg = initialContactRightKneeAngleDeg
        self.maxKneeFlexionLeftKneeAngleDeg = maxKneeFlexionLeftKneeAngleDeg
        self.maxKneeFlexionRightKneeAngleDeg = maxKneeFlexionRightKneeAngleDeg
        self.landingAsymmetryRatio = landingAsymmetryRatio
        self.kneeAsymmetryRatio = kneeAsymmetryRatio
        self.jumpGraph = jumpGraph
        self.imuRecording = imuRecording
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

    var failedProtocolChecks: [JumpProtocolCheck] {
        protocolChecks.filter { !$0.passed }
    }

    var displayPrediction: String {
        isNormalPrediction ? "Normal" : prediction.capitalized
    }

    var analysisResponse: JumpAnalysisResponse {
        JumpAnalysisResponse(
            timestamp: date,
            protocolPassed: protocolPassed,
            protocolChecks: protocolChecks.map {
                JumpAnalysisResponse.ProtocolCheckSummary(
                    name: $0.name,
                    passed: $0.passed,
                    value: $0.value,
                    threshold: $0.threshold
                )
            },
            prediction: prediction,
            anomalyScore: anomalyScore,
            outlierFeatureCount: outlierFeatureCount,
            analyzedFeatureCount: analyzedFeatureCount,
            maxAbsRobustZ: maxAbsRobustZ,
            worstFeature: worstFeature,
            worstFeatureZ: worstFeatureZ,
            worstFeatureValue: worstFeatureValue,
            worstFeatureReferenceMedian: worstFeatureReferenceMedian,
            validPoseFrames: validPoseFrames,
            initialContactFrame: initialContactFrame,
            maxKneeFlexionFrame: maxKneeFlexionFrame,
            videoFPS: videoFPS,
            estimatedShoulderWidthCm: estimatedShoulderWidthCm,
            summary: analysisSummary,
            initialContactLeftKneeAngleDeg: initialContactLeftKneeAngleDeg,
            initialContactRightKneeAngleDeg: initialContactRightKneeAngleDeg,
            maxKneeFlexionLeftKneeAngleDeg: maxKneeFlexionLeftKneeAngleDeg,
            maxKneeFlexionRightKneeAngleDeg: maxKneeFlexionRightKneeAngleDeg,
            landingAsymmetryRatio: landingAsymmetryRatio,
            kneeAsymmetryRatio: kneeAsymmetryRatio,
            jumpGraph: jumpGraph,
            imuRecording: imuRecording
        )
    }

    func updatingNarration(summary: String, llmNarratedSummary: Bool) -> Jump {
        Jump(
            id: id,
            date: date,
            videoURL: videoURL,
            athleteProfile: athleteProfile,
            protocolPassed: protocolPassed,
            protocolChecks: protocolChecks,
            prediction: prediction,
            anomalyScore: anomalyScore,
            outlierFeatureCount: outlierFeatureCount,
            analyzedFeatureCount: analyzedFeatureCount,
            maxAbsRobustZ: maxAbsRobustZ,
            worstFeature: worstFeature,
            worstFeatureZ: worstFeatureZ,
            worstFeatureValue: worstFeatureValue,
            worstFeatureReferenceMedian: worstFeatureReferenceMedian,
            validPoseFrames: validPoseFrames,
            initialContactFrame: initialContactFrame,
            maxKneeFlexionFrame: maxKneeFlexionFrame,
            videoFPS: videoFPS,
            estimatedShoulderWidthCm: estimatedShoulderWidthCm,
            analysisSummary: summary,
            llmNarratedSummary: llmNarratedSummary,
            initialContactLeftKneeAngleDeg: initialContactLeftKneeAngleDeg,
            initialContactRightKneeAngleDeg: initialContactRightKneeAngleDeg,
            maxKneeFlexionLeftKneeAngleDeg: maxKneeFlexionLeftKneeAngleDeg,
            maxKneeFlexionRightKneeAngleDeg: maxKneeFlexionRightKneeAngleDeg,
            landingAsymmetryRatio: landingAsymmetryRatio,
            kneeAsymmetryRatio: kneeAsymmetryRatio,
            jumpGraph: jumpGraph,
            imuRecording: imuRecording
        )
    }

    func resettingNarrationStatus() -> Jump {
        Jump(
            id: id,
            date: date,
            videoURL: videoURL,
            athleteProfile: athleteProfile,
            protocolPassed: protocolPassed,
            protocolChecks: protocolChecks,
            prediction: prediction,
            anomalyScore: anomalyScore,
            outlierFeatureCount: outlierFeatureCount,
            analyzedFeatureCount: analyzedFeatureCount,
            maxAbsRobustZ: maxAbsRobustZ,
            worstFeature: worstFeature,
            worstFeatureZ: worstFeatureZ,
            worstFeatureValue: worstFeatureValue,
            worstFeatureReferenceMedian: worstFeatureReferenceMedian,
            validPoseFrames: validPoseFrames,
            initialContactFrame: initialContactFrame,
            maxKneeFlexionFrame: maxKneeFlexionFrame,
            videoFPS: videoFPS,
            estimatedShoulderWidthCm: estimatedShoulderWidthCm,
            analysisSummary: analysisSummary,
            llmNarratedSummary: false,
            initialContactLeftKneeAngleDeg: initialContactLeftKneeAngleDeg,
            initialContactRightKneeAngleDeg: initialContactRightKneeAngleDeg,
            maxKneeFlexionLeftKneeAngleDeg: maxKneeFlexionLeftKneeAngleDeg,
            maxKneeFlexionRightKneeAngleDeg: maxKneeFlexionRightKneeAngleDeg,
            landingAsymmetryRatio: landingAsymmetryRatio,
            kneeAsymmetryRatio: kneeAsymmetryRatio,
            jumpGraph: jumpGraph,
            imuRecording: imuRecording
        )
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case date
        case videoURL
        case athleteProfile
        case protocolPassed
        case protocolChecks
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
        case llmNarratedSummary
        case initialContactLeftKneeAngleDeg
        case initialContactRightKneeAngleDeg
        case maxKneeFlexionLeftKneeAngleDeg
        case maxKneeFlexionRightKneeAngleDeg
        case landingAsymmetryRatio
        case kneeAsymmetryRatio
        case jumpGraph
        case imuRecording
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
        athleteProfile = try container.decodeIfPresent(JumpAthleteProfile.self, forKey: .athleteProfile)
        protocolPassed = try container.decodeIfPresent(Bool.self, forKey: .protocolPassed) ?? true
        protocolChecks = try container.decodeIfPresent([JumpProtocolCheck].self, forKey: .protocolChecks) ?? []
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
        llmNarratedSummary = try container.decodeIfPresent(Bool.self, forKey: .llmNarratedSummary) ?? false
        initialContactLeftKneeAngleDeg = try container.decodeIfPresent(Double.self, forKey: .initialContactLeftKneeAngleDeg) ?? 0
        initialContactRightKneeAngleDeg = try container.decodeIfPresent(Double.self, forKey: .initialContactRightKneeAngleDeg) ?? 0
        maxKneeFlexionLeftKneeAngleDeg = try container.decodeIfPresent(Double.self, forKey: .maxKneeFlexionLeftKneeAngleDeg) ?? 0
        maxKneeFlexionRightKneeAngleDeg = try container.decodeIfPresent(Double.self, forKey: .maxKneeFlexionRightKneeAngleDeg) ?? 0
        landingAsymmetryRatio = try container.decodeIfPresent(Double.self, forKey: .landingAsymmetryRatio) ?? 0
        kneeAsymmetryRatio = try container.decodeIfPresent(Double.self, forKey: .kneeAsymmetryRatio) ?? 0
        jumpGraph = try container.decodeIfPresent(JumpAnalysisResponse.JumpGraph.self, forKey: .jumpGraph)
        imuRecording = try container.decodeIfPresent(JumpAnalysisResponse.IMURecordingSummary.self, forKey: .imuRecording)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(date, forKey: .date)
        try container.encodeIfPresent(videoURL, forKey: .videoURL)
        try container.encodeIfPresent(athleteProfile, forKey: .athleteProfile)
        try container.encode(protocolPassed, forKey: .protocolPassed)
        try container.encode(protocolChecks, forKey: .protocolChecks)
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
        try container.encode(llmNarratedSummary, forKey: .llmNarratedSummary)
        try container.encode(initialContactLeftKneeAngleDeg, forKey: .initialContactLeftKneeAngleDeg)
        try container.encode(initialContactRightKneeAngleDeg, forKey: .initialContactRightKneeAngleDeg)
        try container.encode(maxKneeFlexionLeftKneeAngleDeg, forKey: .maxKneeFlexionLeftKneeAngleDeg)
        try container.encode(maxKneeFlexionRightKneeAngleDeg, forKey: .maxKneeFlexionRightKneeAngleDeg)
        try container.encode(landingAsymmetryRatio, forKey: .landingAsymmetryRatio)
        try container.encode(kneeAsymmetryRatio, forKey: .kneeAsymmetryRatio)
        try container.encodeIfPresent(jumpGraph, forKey: .jumpGraph)
        try container.encodeIfPresent(imuRecording, forKey: .imuRecording)
    }
}
