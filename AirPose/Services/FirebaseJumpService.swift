import FirebaseFirestore
import Foundation

struct FirebaseJumpService {
    private let database = Firestore.firestore()

    func fetchJumps(for userID: String) async throws -> [Jump] {
        let snapshot = try await database
            .collection("users")
            .document(userID)
            .collection("jumps")
            .order(by: "date", descending: true)
            .getDocuments()

        return snapshot.documents.compactMap { document in
            FirestoreJumpRecord(document: document)?.toJump()
        }
    }

    func save(_ jump: Jump, for userID: String) async throws {
        try await database
            .collection("users")
            .document(userID)
            .collection("jumps")
            .document(jump.id.uuidString)
            .setData(FirestoreJumpRecord(jump: jump).dictionary)
    }

    func delete(_ jump: Jump, for userID: String) async throws {
        try await database
            .collection("users")
            .document(userID)
            .collection("jumps")
            .document(jump.id.uuidString)
            .delete()
    }
}

private struct FirestoreJumpRecord: Codable {
    let id: UUID
    let date: Date
    let videoURLString: String?
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

    init(jump: Jump) {
        id = jump.id
        date = jump.date
        videoURLString = jump.videoURL?.absoluteString
        protocolPassed = jump.protocolPassed
        prediction = jump.prediction
        anomalyScore = jump.anomalyScore
        outlierFeatureCount = jump.outlierFeatureCount
        analyzedFeatureCount = jump.analyzedFeatureCount
        maxAbsRobustZ = jump.maxAbsRobustZ
        worstFeature = jump.worstFeature
        worstFeatureZ = jump.worstFeatureZ
        worstFeatureValue = jump.worstFeatureValue
        worstFeatureReferenceMedian = jump.worstFeatureReferenceMedian
        validPoseFrames = jump.validPoseFrames
        initialContactFrame = jump.initialContactFrame
        maxKneeFlexionFrame = jump.maxKneeFlexionFrame
        videoFPS = jump.videoFPS
        estimatedShoulderWidthCm = jump.estimatedShoulderWidthCm
        analysisSummary = jump.analysisSummary
        initialContactLeftKneeAngleDeg = jump.initialContactLeftKneeAngleDeg
        initialContactRightKneeAngleDeg = jump.initialContactRightKneeAngleDeg
        maxKneeFlexionLeftKneeAngleDeg = jump.maxKneeFlexionLeftKneeAngleDeg
        maxKneeFlexionRightKneeAngleDeg = jump.maxKneeFlexionRightKneeAngleDeg
        landingAsymmetryRatio = jump.landingAsymmetryRatio
        kneeAsymmetryRatio = jump.kneeAsymmetryRatio
    }

    init?(document: QueryDocumentSnapshot) {
        let data = document.data()

        guard
            let id = UUID(uuidString: data["id"] as? String ?? document.documentID),
            let timestamp = data["date"] as? Timestamp
        else {
            return nil
        }

        self.id = id
        self.date = timestamp.dateValue()
        self.videoURLString = data["videoURLString"] as? String
        self.protocolPassed = data["protocolPassed"] as? Bool ?? true
        self.prediction = data["prediction"] as? String ?? "legacy"
        self.anomalyScore = data["anomalyScore"] as? Double ?? 0
        self.outlierFeatureCount = data["outlierFeatureCount"] as? Int ?? 0
        self.analyzedFeatureCount = data["analyzedFeatureCount"] as? Int ?? 0
        self.maxAbsRobustZ = data["maxAbsRobustZ"] as? Double ?? 0
        self.worstFeature = data["worstFeature"] as? String ?? "n/a"
        self.worstFeatureZ = data["worstFeatureZ"] as? Double ?? 0
        self.worstFeatureValue = data["worstFeatureValue"] as? Double ?? 0
        self.worstFeatureReferenceMedian = data["worstFeatureReferenceMedian"] as? Double ?? 0
        self.validPoseFrames = data["validPoseFrames"] as? Int ?? 0
        self.initialContactFrame = data["initialContactFrame"] as? Int ?? 0
        self.maxKneeFlexionFrame = data["maxKneeFlexionFrame"] as? Int ?? 0
        self.videoFPS = data["videoFPS"] as? Double ?? 0
        self.estimatedShoulderWidthCm = data["estimatedShoulderWidthCm"] as? Double ?? 0
        self.analysisSummary = (data["analysisSummary"] as? String) ?? (data["llmFeedback"] as? String) ?? ""
        self.initialContactLeftKneeAngleDeg = data["initialContactLeftKneeAngleDeg"] as? Double ?? 0
        self.initialContactRightKneeAngleDeg = data["initialContactRightKneeAngleDeg"] as? Double ?? 0
        self.maxKneeFlexionLeftKneeAngleDeg = data["maxKneeFlexionLeftKneeAngleDeg"] as? Double ?? 0
        self.maxKneeFlexionRightKneeAngleDeg = data["maxKneeFlexionRightKneeAngleDeg"] as? Double ?? 0
        self.landingAsymmetryRatio = data["landingAsymmetryRatio"] as? Double ?? 0
        self.kneeAsymmetryRatio = data["kneeAsymmetryRatio"] as? Double ?? 0
    }

    var dictionary: [String: Any] {
        [
            "id": id.uuidString,
            "date": Timestamp(date: date),
            "videoURLString": videoURLString as Any,
            "protocolPassed": protocolPassed,
            "prediction": prediction,
            "anomalyScore": anomalyScore,
            "outlierFeatureCount": outlierFeatureCount,
            "analyzedFeatureCount": analyzedFeatureCount,
            "maxAbsRobustZ": maxAbsRobustZ,
            "worstFeature": worstFeature,
            "worstFeatureZ": worstFeatureZ,
            "worstFeatureValue": worstFeatureValue,
            "worstFeatureReferenceMedian": worstFeatureReferenceMedian,
            "validPoseFrames": validPoseFrames,
            "initialContactFrame": initialContactFrame,
            "maxKneeFlexionFrame": maxKneeFlexionFrame,
            "videoFPS": videoFPS,
            "estimatedShoulderWidthCm": estimatedShoulderWidthCm,
            "analysisSummary": analysisSummary,
            "initialContactLeftKneeAngleDeg": initialContactLeftKneeAngleDeg,
            "initialContactRightKneeAngleDeg": initialContactRightKneeAngleDeg,
            "maxKneeFlexionLeftKneeAngleDeg": maxKneeFlexionLeftKneeAngleDeg,
            "maxKneeFlexionRightKneeAngleDeg": maxKneeFlexionRightKneeAngleDeg,
            "landingAsymmetryRatio": landingAsymmetryRatio,
            "kneeAsymmetryRatio": kneeAsymmetryRatio,
        ]
    }

    func toJump() -> Jump {
        Jump(
            id: id,
            date: date,
            videoURL: videoURLString.flatMap(URL.init(string:)),
            protocolPassed: protocolPassed,
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
            initialContactLeftKneeAngleDeg: initialContactLeftKneeAngleDeg,
            initialContactRightKneeAngleDeg: initialContactRightKneeAngleDeg,
            maxKneeFlexionLeftKneeAngleDeg: maxKneeFlexionLeftKneeAngleDeg,
            maxKneeFlexionRightKneeAngleDeg: maxKneeFlexionRightKneeAngleDeg,
            landingAsymmetryRatio: landingAsymmetryRatio,
            kneeAsymmetryRatio: kneeAsymmetryRatio
        )
    }
}
