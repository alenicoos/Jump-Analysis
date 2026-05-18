import Foundation

enum LLMFeedbackServiceError: LocalizedError {
    case invalidURL
    case invalidResponse

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            "The LLM API URL is invalid."
        case .invalidResponse:
            "The LLM service returned an unexpected response."
        }
    }
}

private struct LLMFeedbackPayload: Encodable {
    let prompt: String
}

private struct LLMFeedbackResponse: Decodable {
    let feedback: String
}

struct LLMFeedbackService {
    func generateFeedback(for analysis: JumpAnalysisResponse, settings: AppSettings) async throws -> String {
        if settings.mockModeEnabled {
            return analysis.summary
        }

        guard let endpoint = URL(string: settings.llmAPIURL) else {
            throw LLMFeedbackServiceError.invalidURL
        }

        let prompt = """
        You are a jump performance coach. Provide a concise 2 sentence explanation of this jump:
        prediction: \(analysis.prediction)
        anomalyScore: \(analysis.anomalyScore)
        protocolPassed: \(analysis.protocolPassed)
        worstFeature: \(analysis.worstFeature)
        summary: \(analysis.summary)
        Focus on actionable coaching feedback.
        """

        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        if !settings.llmAPIKey.isEmpty {
            request.setValue("Bearer \(settings.llmAPIKey)", forHTTPHeaderField: "Authorization")
        }

        request.httpBody = try JSONEncoder().encode(LLMFeedbackPayload(prompt: prompt))

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse, (200...299).contains(httpResponse.statusCode) else {
            throw LLMFeedbackServiceError.invalidResponse
        }

        if let decoded = try? JSONDecoder().decode(LLMFeedbackResponse.self, from: data) {
            return decoded.feedback
        }

        if let rawText = String(data: data, encoding: .utf8), !rawText.isEmpty {
            return rawText
        }

        throw LLMFeedbackServiceError.invalidResponse
    }
}
