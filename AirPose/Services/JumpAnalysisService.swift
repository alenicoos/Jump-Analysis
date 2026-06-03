import Foundation

enum JumpAnalysisServiceError: LocalizedError {
    case invalidServerURL
    case missingVideo
    case missingAthleteHeight
    case invalidResponse
    case serverError(String)

    var errorDescription: String? {
        switch self {
        case .invalidServerURL:
            "The backend server URL is invalid."
        case .missingVideo:
            "Record a jump first, or keep mock mode enabled to simulate analysis."
        case .missingAthleteHeight:
            "Enter the athlete's height in the Profile tab before using live server analysis."
        case .invalidResponse:
            "The jump analysis response could not be parsed."
        case .serverError(let message):
            message
        }
    }
}

private struct BackendErrorPayload: Decodable {
    let detail: String
}

struct JumpAnalysisService {
    func analyzeJump(
        videoURL: URL?,
        recordingStartedAt: Date?,
        settings: AppSettings,
        athleteHeightCm: Double?
    ) async throws -> JumpAnalysisResponse {
        if settings.mockModeEnabled {
            return mockResponse()
        }

        guard let endpoint = URL(string: settings.backendServerURL) else {
            throw JumpAnalysisServiceError.invalidServerURL
        }

        guard let videoURL else {
            throw JumpAnalysisServiceError.missingVideo
        }

        guard let athleteHeightCm, athleteHeightCm > 0 else {
            throw JumpAnalysisServiceError.missingAthleteHeight
        }

        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"

        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        let videoData = try Data(contentsOf: videoURL)
        request.httpBody = makeMultipartBody(
            videoData: videoData,
            filename: videoURL.lastPathComponent,
            heightCm: athleteHeightCm,
            recordingStartedAt: recordingStartedAt,
            boundary: boundary
        )

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw JumpAnalysisServiceError.invalidResponse
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            let message = parseServerErrorMessage(from: data)
            throw JumpAnalysisServiceError.serverError(message)
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        do {
            return try decoder.decode(JumpAnalysisResponse.self, from: data)
        } catch {
            throw JumpAnalysisServiceError.invalidResponse
        }
    }

    private func makeMultipartBody(
        videoData: Data,
        filename: String,
        heightCm: Double,
        recordingStartedAt: Date?,
        boundary: String
    ) -> Data {
        var body = Data()
        body.append("--\(boundary)\r\n")
        body.append("Content-Disposition: form-data; name=\"height_cm\"\r\n\r\n")
        body.append("\(heightCm)\r\n")
        if let recordingStartedAt {
            let formatter = ISO8601DateFormatter()
            formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            body.append("--\(boundary)\r\n")
            body.append("Content-Disposition: form-data; name=\"recording_started_at\"\r\n\r\n")
            body.append("\(formatter.string(from: recordingStartedAt))\r\n")
        }
        body.append("--\(boundary)\r\n")
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"\(filename)\"\r\n")
        body.append("Content-Type: video/quicktime\r\n\r\n")
        body.append(videoData)
        body.append("\r\n--\(boundary)--\r\n")
        return body
    }

    private func mockResponse() -> JumpAnalysisResponse {
        JumpAnalysisResponse(
            timestamp: .now,
            protocolPassed: Bool.random(),
            protocolChecks: [
                .init(name: "drop_started_from_height", passed: true, value: 0.23, threshold: 0.15),
                .init(name: "two_foot_contact", passed: false, value: 0.18, threshold: 0.10),
                .init(name: "second_jump", passed: true, value: 0.21, threshold: 0.12),
            ],
            prediction: Bool.random() ? "normal" : "anomaly",
            anomalyScore: Double.random(in: 1.2...5.8),
            outlierFeatureCount: Int.random(in: 1...9),
            analyzedFeatureCount: 36,
            maxAbsRobustZ: Double.random(in: 2.4...6.8),
            worstFeature: "kfmax_left_knee_medial_offset_ratio",
            worstFeatureZ: Double.random(in: 2.0...6.0),
            worstFeatureValue: Double.random(in: -0.6...0.6),
            worstFeatureReferenceMedian: Double.random(in: -0.2...0.2),
            validPoseFrames: Int.random(in: 18...42),
            initialContactFrame: Int.random(in: 12...24),
            maxKneeFlexionFrame: Int.random(in: 25...40),
            videoFPS: 30,
            estimatedShoulderWidthCm: Double.random(in: 34...48),
            summary: "Mock analysis complete. The largest deviation was knee alignment during maximum flexion, so review frontal knee tracking on landing and rebound.",
            initialContactLeftKneeAngleDeg: Double.random(in: 146...170),
            initialContactRightKneeAngleDeg: Double.random(in: 146...170),
            maxKneeFlexionLeftKneeAngleDeg: Double.random(in: 98...132),
            maxKneeFlexionRightKneeAngleDeg: Double.random(in: 98...132),
            landingAsymmetryRatio: Double.random(in: 0.02...0.18),
            kneeAsymmetryRatio: Double.random(in: 0.01...0.14),
            jumpGraph: nil,
            imuRecording: nil
        )
    }

    private func parseServerErrorMessage(from data: Data) -> String {
        if let payload = try? JSONDecoder().decode(BackendErrorPayload.self, from: data) {
            return payload.detail
        }

        if let body = String(data: data, encoding: .utf8), !body.isEmpty {
            return body
        }

        return "Unexpected server response."
    }
}

private extension Data {
    mutating func append(_ string: String) {
        if let data = string.data(using: .utf8) {
            append(data)
        }
    }
}
