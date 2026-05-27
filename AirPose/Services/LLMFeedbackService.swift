import Foundation

enum LLMFeedbackServiceError: LocalizedError {
    case invalidURL
    case missingAPIKey
    case invalidResponse
    case serverError(String)

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            "The OpenAI API URL is invalid."
        case .missingAPIKey:
            "Add a valid OpenAI API key in Settings before requesting narrated feedback."
        case .invalidResponse:
            "The OpenAI response could not be parsed."
        case .serverError(let message):
            message
        }
    }
}

private struct OpenAIResponsesPayload: Encodable {
    struct TextConfiguration: Encodable {
        struct FormatConfiguration: Encodable {
            let type: String
        }

        let format: FormatConfiguration
    }

    let model: String
    let instructions: String
    let input: String
    let maxOutputTokens: Int
    let text: TextConfiguration

    enum CodingKeys: String, CodingKey {
        case model
        case instructions
        case input
        case maxOutputTokens = "max_output_tokens"
        case text
    }
}

private struct OpenAIResponsesResult: Decodable {
    struct OutputItem: Decodable {
        struct ContentItem: Decodable {
            let type: String?
            let text: String?
        }

        let content: [ContentItem]?
    }

    let outputText: String?
    let output: [OutputItem]?

    enum CodingKeys: String, CodingKey {
        case outputText = "output_text"
        case output
    }
}

private enum JumpPostureInterpretationScript {
    static let prompt = """
    AirPose posture interpretation guide

    Use this guide to convert AirPose jump analytics into plain-language posture feedback.
    The analysis comes from a front-view 2D pose pipeline:
    - YOLO pose estimates COCO body keypoints from the frontal video.
    - The backend keeps shoulder, hip, knee, and ankle landmarks.
    - It extracts 37 frontal features from two keyframes:
      1. `ic_*` = initial contact
      2. `kfmax_*` = maximum knee flexion
      3. `crop_length_frames` = frame distance between those keyframes
    - The model compares those features against a converted 183-athlete motion-capture reference dataset.
    - `worst_feature` is the feature with the largest robust deviation from that reference.

    Important interpretation limits:
    - This is a frontal 2D system, not a side-view or 3D system.
    - You may describe left-right symmetry, knee tracking, ankle height differences, hip tilt, shoulder tilt, and lateral trunk shift.
    - Do not claim true forward/backward trunk lean, lumbar flexion, rotation in depth, or foot pronation/supination unless a feature explicitly supports only a cautious frontal-view proxy.
    - Do not make medical diagnoses.
    - If the data is ambiguous, say the movement "may suggest" or "looks consistent with" instead of stating certainty.

    Phase meaning:
    - `ic_` features describe landing at first ground contact.
    - `kfmax_` features describe the deepest part of the landing absorption phase.
    - If the issue appears at `ic_*`, describe it as a landing/contact issue.
    - If the issue appears at `kfmax_*`, describe it as an absorption/braking/deepest-landing issue.

    Global result interpretation:
    - `protocol_passed = false`: explain that the trial does not match the intended drop-jump sequence, so posture interpretation is less reliable.
    - `prediction = anomaly`: explain that the movement differs noticeably from the reference dataset.
    - `anomaly_score`, `max_abs_robust_z`, and `outlier_feature_count` indicate how strongly the jump deviates from the reference population.
    - Higher values justify stronger wording like "clear asymmetry" or "marked deviation."

    Feature families and suggested language:

    1. Width and stance features
    - `knee_distance`, `ankle_distance`, `hip_distance`, `shoulder_distance`
    - `knee_shoulder_width_ratio`, `ankle_shoulder_width_ratio`, `knee_ankle_width_ratio`
    Interpretation:
    - Describe how narrow or wide the stance appears relative to the athlete's upper-body width.
    - If knees are much narrower than ankles, this can support language like knees moving inward relative to the feet.
    - If knees are much wider than ankles, this can support language like knees staying pushed outward or feet landing relatively narrow.
    - Use cautious wording unless reinforced by medial-offset or knee-angle features.

    2. Knee tracking / valgus-varus proxy features
    - `left_knee_medial_offset_ratio`
    - `right_knee_medial_offset_ratio`
    - `knee_center_ankle_center_offset_ratio`
    Interpretation:
    - These are the most direct frontal-view proxies for knee tracking over the feet.
    - Higher medial-offset deviations can support phrases such as:
      "the knee tracks inward relative to the foot"
      "the knees do not stay centered over the ankles"
      "one knee collapses inward more than the other"
    - Center-offset deviations can support language about the lower-limb line shifting left or right as a whole.

    3. Frontal knee angle features
    - `left_hip_knee_ankle_frontal_angle`
    - `right_hip_knee_ankle_frontal_angle`
    Interpretation:
    - These describe frontal-plane knee alignment.
    - Differences between left and right support asymmetry language.
    - Large deviations from the reference can support phrases such as:
      "the left and right knees are not aligning the same way during landing"
      "one side shows a less stable knee line through the landing phase"
    - Do not over-interpret these as pure flexion-extension; keep the explanation tied to frontal alignment quality.

    4. Shoulder / pelvis / trunk alignment features
    - `shoulder_tilt_degrees`
    - `hip_tilt_degrees`
    - `trunk_lateral_lean_degrees`
    - `body_center_x_over_ankle_center_offset_ratio`
    Interpretation:
    - These support language about upper-body or pelvic alignment from the front view.
    - `shoulder_tilt_degrees`: shoulders look uneven or one shoulder sits higher during landing.
    - `hip_tilt_degrees`: pelvis looks uneven or one hip drops/rises relative to the other.
    - `trunk_lateral_lean_degrees`: torso drifts or leans to one side.
    - `body_center_x_over_ankle_center_offset_ratio`: body mass appears shifted to one side relative to the base of support.
    - Because the camera is frontal, keep this to side-to-side alignment. Do not convert it into forward/backward lean.

    5. Left-right vertical asymmetry features
    - `left_right_ankle_y_difference_ratio`
    - `left_right_knee_y_difference_ratio`
    Interpretation:
    - These support language about uneven landing height, uneven loading, or one side reaching the ground / absorbing force differently.
    - Good phrasing includes:
      "the landing looks uneven from left to right"
      "one leg appears to take load earlier or more strongly"
      "the knees do not descend symmetrically in the absorption phase"

    6. Timing/context features
    - `valid_pose_frames`
    - `initial_contact_frame`
    - `max_knee_flexion_frame`
    - `video_fps`
    - `crop_length_frames`
    Interpretation:
    - Use only as context for confidence or tempo, not as the main coaching conclusion.
    - Low valid-pose coverage should reduce certainty.

    Writing rules:
    - Start with the overall movement quality relative to the reference dataset.
    - Then describe the 2 or 3 clearest posture findings supported by the strongest features.
    - Mention whether the issue is more visible at landing (`ic`) or at deepest absorption (`kfmax`).
    - Finish with one concrete coaching correction.
    - Prefer movement language over raw metric language.
    - Example phrases:
      "the knees appear to drift inward relative to the feet on landing"
      "the torso shifts slightly to one side during the braking phase"
      "the landing is asymmetrical, with one leg appearing to absorb more load"
      "the shoulders and pelvis do not stay level through contact"

    Forbidden or unsupported claims:
    - Do not say the shoulders are too far forward or too far backward.
    - Do not claim ankle twisting in depth or foot pronation unless you clearly frame it as a frontal-view stability impression only.
    - Do not infer pain, injury, or pathology.
    """
}

struct LLMFeedbackService {
    private let model = "gpt-4.1-mini"
    private let datasetContext = """
    The jump was analyzed against AirPose's reference dataset: a converted frontal 2D, 37-feature representation of a 183-athlete motion-capture drop-jump dataset. The model compares the recorded jump against that reference distribution and flags deviations as anomaly-style differences from the dataset, not as a medical diagnosis.
    """

    func generateFeedback(for analysis: JumpAnalysisResponse, settings: AppSettings) async throws -> String {
        if settings.mockModeEnabled {
            return analysis.summary
        }

        guard let endpoint = URL(string: settings.llmAPIURL) else {
            throw LLMFeedbackServiceError.invalidURL
        }

        let apiKey = settings.llmAPIKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !apiKey.isEmpty, apiKey != AppSettings.placeholderLLMAPIKey else {
            throw LLMFeedbackServiceError.missingAPIKey
        }

        let instructions = """
        You are a jump performance coach writing athlete-facing posture feedback. Write one detailed paragraph of 4 to 6 sentences in plain language, do not mention being an AI model, and do not make medical claims. Focus on how the jump was performed from a posture and alignment point of view: trunk or shoulder position if supported by the feature names, knee tracking and alignment, left-right symmetry at initial contact and maximum flexion, and ankle or foot control if the strongest deviation suggests it. Translate the analytics into clear movement language such as knees collapsing inward, one side absorbing more load, torso drifting too far forward, or ankles looking unstable, but only when the provided metrics or feature names support that interpretation. Use the numeric values naturally to justify the explanation, compare the movement with the AirPose reference dataset, and finish with a concrete coaching correction.
        """

        let input = """
        \(datasetContext)

        Interpretation script:
        \(JumpPostureInterpretationScript.prompt)

        Jump result:
        - prediction: \(analysis.prediction)
        - protocol passed: \(analysis.protocolPassed)
        - anomaly score: \(analysis.anomalyScore)
        - outlier features: \(analysis.outlierFeatureCount) out of \(analysis.analyzedFeatureCount)
        - strongest deviation feature: \(humanReadableFeatureName(analysis.worstFeature))
        - strongest deviation z-score: \(analysis.worstFeatureZ)
        - measured feature value: \(analysis.worstFeatureValue)
        - reference median for that feature: \(analysis.worstFeatureReferenceMedian)
        - valid pose frames: \(analysis.validPoseFrames)
        - initial contact frame: \(analysis.initialContactFrame)
        - maximum knee flexion frame: \(analysis.maxKneeFlexionFrame)
        - video fps: \(analysis.videoFPS)
        - estimated shoulder width cm: \(analysis.estimatedShoulderWidthCm)
        - initial contact left knee angle deg: \(analysis.initialContactLeftKneeAngleDeg)
        - initial contact right knee angle deg: \(analysis.initialContactRightKneeAngleDeg)
        - max flexion left knee angle deg: \(analysis.maxKneeFlexionLeftKneeAngleDeg)
        - max flexion right knee angle deg: \(analysis.maxKneeFlexionRightKneeAngleDeg)
        - landing asymmetry ratio: \(analysis.landingAsymmetryRatio)
        - knee asymmetry ratio: \(analysis.kneeAsymmetryRatio)
        - backend summary: \(analysis.summary)

        Explain what these metrics mean in words for the athlete, with posture quality as the main focus.
        """

        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

        let payload = OpenAIResponsesPayload(
            model: model,
            instructions: instructions,
            input: input,
            maxOutputTokens: 260,
            text: .init(format: .init(type: "text"))
        )
        request.httpBody = try JSONEncoder().encode(payload)

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw LLMFeedbackServiceError.invalidResponse
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            let body = String(data: data, encoding: .utf8) ?? "Unexpected OpenAI server response."
            throw LLMFeedbackServiceError.serverError(body)
        }

        if let decoded = try? JSONDecoder().decode(OpenAIResponsesResult.self, from: data) {
            if let outputText = decoded.outputText?.trimmingCharacters(in: .whitespacesAndNewlines), !outputText.isEmpty {
                return outputText
            }

            let nestedText = decoded.output?
                .flatMap { $0.content ?? [] }
                .compactMap(\.text)
                .joined(separator: "\n")
                .trimmingCharacters(in: .whitespacesAndNewlines)

            if let nestedText, !nestedText.isEmpty {
                return nestedText
            }
        }

        if let rawText = String(data: data, encoding: .utf8), !rawText.isEmpty {
            return rawText
        }

        throw LLMFeedbackServiceError.invalidResponse
    }

    private func humanReadableFeatureName(_ feature: String) -> String {
        feature
            .replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: "kfmax", with: "max knee flexion")
            .replacingOccurrences(of: "ic", with: "initial contact")
            .replacingOccurrences(of: "lr", with: "left-right")
    }
}
