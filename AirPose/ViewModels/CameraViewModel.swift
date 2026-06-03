import Combine
import AVFoundation
import Foundation

@MainActor
final class CameraViewModel: ObservableObject {
    enum AthleteSelection: String, CaseIterable, Identifiable {
        case accountProfile = "My Profile"
        case guestAthlete = "Other Athlete"

        var id: String { rawValue }
    }

    enum AnalysisState: Equatable {
        case idle
        case uploading
        case completed
    }

    enum LiveGuidanceState: Equatable {
        case idle
        case connecting
        case active
        case completed
        case failed
    }

    @Published var analysisState: AnalysisState = .idle
    @Published var liveGuidanceState: LiveGuidanceState = .idle
    @Published var liveGuidancePhase: String = "idle"
    @Published var liveGuidanceMessage: String?
    @Published var liveAnalysisSummary: String?
    @Published var athleteSelection: AthleteSelection = .accountProfile
    @Published var guestAthleteProfile = UserProfile()
    @Published var errorMessage: String?
    @Published var successMessage: String?
    @Published var latestJump: Jump?

    let cameraManager = CameraManager()

    private let jumpStore: JumpStore
    private let profileStore: UserProfileStore
    private let settingsStore: AppSettingsStore
    private let cloudSyncCoordinator: CloudSyncCoordinator
    private let analysisService: JumpAnalysisService
    private let llmFeedbackService: LLMFeedbackService
    private let speechSynthesizer = AVSpeechSynthesizer()
    private let liveResultDecoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()
    private var cancellables = Set<AnyCancellable>()
    private var liveGuidanceSocket: URLSessionWebSocketTask?
    private var lastSpokenGuidance: String?
    private var lastSpokenGuidanceAt: Date?

    init(
        jumpStore: JumpStore,
        profileStore: UserProfileStore,
        settingsStore: AppSettingsStore,
        cloudSyncCoordinator: CloudSyncCoordinator,
        analysisService: JumpAnalysisService,
        llmFeedbackService: LLMFeedbackService
    ) {
        self.jumpStore = jumpStore
        self.profileStore = profileStore
        self.settingsStore = settingsStore
        self.cloudSyncCoordinator = cloudSyncCoordinator
        self.analysisService = analysisService
        self.llmFeedbackService = llmFeedbackService

        cameraManager.$errorMessage
            .receive(on: DispatchQueue.main)
            .sink { [weak self] message in
                guard let message else { return }
                self?.errorMessage = message
            }
            .store(in: &cancellables)

        cameraManager.objectWillChange
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.objectWillChange.send()
            }
            .store(in: &cancellables)
    }

    var recordedVideoURL: URL? {
        cameraManager.recordedVideoURL
    }

    var canSendForAnalysis: Bool {
        recordedVideoURL != nil || settingsStore.settings.mockModeEnabled
    }

    var canUseRecordedFallback: Bool {
        AppPlatform.isDesktopDemo || canSendForAnalysis
    }

    var accountProfile: UserProfile {
        profileStore.profile
    }

    var selectedAthleteProfile: UserProfile {
        athleteSelection == .accountProfile ? profileStore.profile : guestAthleteProfile
    }

    func onAppear() {
        guard AppPlatform.supportsLiveCameraCapture else { return }
        cameraManager.requestPermissionsAndConfigureIfNeeded()
    }

    func onDisappear() {
        stopLiveGuidance()
        guard AppPlatform.supportsLiveCameraCapture else { return }
        cameraManager.stopSession()
    }

    func startRecording() {
        successMessage = nil
        errorMessage = nil
        cameraManager.startRecording()
    }

    func stopRecording() {
        cameraManager.stopRecording()
    }

    func importVideo(from selectedURL: URL) {
        successMessage = nil
        errorMessage = nil

        do {
            try cameraManager.attachRecordedVideo(from: selectedURL)
        } catch {
            errorMessage = "Unable to import the selected video."
        }
    }

    func switchCamera() {
        cameraManager.switchCamera()
    }

    var canStartLiveGuidance: Bool {
        AppPlatform.supportsLiveCameraCapture
            && cameraManager.isConfigured
            && liveGuidanceState != .connecting
            && liveGuidanceState != .active
            && athleteHeightCm(for: settingsStore.settings.units) != nil
    }

    func startLiveGuidance() {
        guard canStartLiveGuidance else {
            if athleteHeightCm(for: settingsStore.settings.units) == nil {
                errorMessage = "Enter the athlete's height before starting live guidance."
            }
            return
        }
        guard let websocketURL = liveGuidanceURL(heightCm: athleteHeightCm(for: settingsStore.settings.units) ?? 0) else {
            errorMessage = "The backend server URL is invalid."
            liveGuidanceState = .failed
            return
        }

        stopLiveGuidance()
        errorMessage = nil
        successMessage = nil
        liveAnalysisSummary = nil
        liveGuidanceState = .connecting
        liveGuidancePhase = "connecting"
        liveGuidanceMessage = "Connecting live guidance..."
        lastSpokenGuidance = nil

        let task = URLSession.shared.webSocketTask(with: websocketURL)
        liveGuidanceSocket = task
        task.resume()
        receiveLiveGuidanceMessages()
        cameraManager.startPreviewFrameDelivery(frameInterval: 0.2) { [weak self] data in
            guard let self else { return }
            self.sendLiveFrame(data)
        }
    }

    func stopLiveGuidance() {
        cameraManager.stopPreviewFrameDelivery()
        if let liveGuidanceSocket {
            let stopMessage = ["type": "stop"]
            if let data = try? JSONSerialization.data(withJSONObject: stopMessage),
               let text = String(data: data, encoding: .utf8) {
                liveGuidanceSocket.send(.string(text)) { _ in }
            }
            liveGuidanceSocket.cancel(with: .normalClosure, reason: nil)
            self.liveGuidanceSocket = nil
        }
        speechSynthesizer.stopSpeaking(at: .immediate)
        if liveGuidanceState == .active || liveGuidanceState == .connecting {
            liveGuidanceState = .idle
            liveGuidancePhase = "idle"
            liveGuidanceMessage = "Live guidance stopped."
        }
    }

    func analyzeRecordedJump() async {
        errorMessage = nil
        successMessage = nil
        analysisState = .uploading

        do {
            let settings = settingsStore.settings
            let analysis = try await analysisService.analyzeJump(
                videoURL: recordedVideoURL,
                recordingStartedAt: cameraManager.recordedVideoStartedAt,
                settings: settings,
                athleteHeightCm: athleteHeightCm(for: settings.units)
            )

            let jump = await persistAnalysisResult(
                analysis,
                videoURL: recordedVideoURL,
                settings: settings
            )
            analysisState = .completed
            successMessage = jump.llmNarratedSummary
                ? "Jump analyzed, narrated, and saved."
                : "Jump analyzed and saved."
        } catch {
            analysisState = .idle
            errorMessage = error.localizedDescription
        }
    }

    private func narratedSummary(for analysis: JumpAnalysisResponse, settings: AppSettings) async -> (summary: String, didUseLLM: Bool) {
        guard settings.hasConfiguredLLMEndpoint else {
            return (analysis.summary, false)
        }

        do {
            let feedback = try await llmFeedbackService.generateFeedback(for: analysis, settings: settings)
            let trimmedFeedback = feedback.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedFeedback.isEmpty else {
                return (analysis.summary, false)
            }

            return (trimmedFeedback, true)
        } catch {
            return (analysis.summary, false)
        }
    }

    private func makeJump(
        from analysis: JumpAnalysisResponse,
        videoURL: URL?,
        summary: String,
        llmNarratedSummary: Bool
    ) -> Jump {
        let athleteProfile = JumpAthleteProfile(
            profile: selectedAthleteProfile,
            source: athleteSelection == .accountProfile ? .accountProfile : .guestAthlete
        )
        return Jump(
            date: analysis.timestamp,
            videoURL: videoURL,
            athleteProfile: athleteProfile,
            protocolPassed: analysis.protocolPassed,
            protocolChecks: analysis.protocolChecks.map {
                JumpProtocolCheck(
                    name: $0.name,
                    passed: $0.passed,
                    value: $0.value,
                    threshold: $0.threshold
                )
            },
            prediction: analysis.prediction,
            anomalyScore: analysis.anomalyScore,
            outlierFeatureCount: analysis.outlierFeatureCount,
            analyzedFeatureCount: analysis.analyzedFeatureCount,
            maxAbsRobustZ: analysis.maxAbsRobustZ,
            worstFeature: analysis.worstFeature,
            worstFeatureZ: analysis.worstFeatureZ,
            worstFeatureValue: analysis.worstFeatureValue,
            worstFeatureReferenceMedian: analysis.worstFeatureReferenceMedian,
            validPoseFrames: analysis.validPoseFrames,
            initialContactFrame: analysis.initialContactFrame,
            maxKneeFlexionFrame: analysis.maxKneeFlexionFrame,
            videoFPS: analysis.videoFPS,
            estimatedShoulderWidthCm: analysis.estimatedShoulderWidthCm,
            analysisSummary: summary,
            llmNarratedSummary: llmNarratedSummary,
            initialContactLeftKneeAngleDeg: analysis.initialContactLeftKneeAngleDeg,
            initialContactRightKneeAngleDeg: analysis.initialContactRightKneeAngleDeg,
            maxKneeFlexionLeftKneeAngleDeg: analysis.maxKneeFlexionLeftKneeAngleDeg,
            maxKneeFlexionRightKneeAngleDeg: analysis.maxKneeFlexionRightKneeAngleDeg,
            landingAsymmetryRatio: analysis.landingAsymmetryRatio,
            kneeAsymmetryRatio: analysis.kneeAsymmetryRatio,
            imuRecording: analysis.imuRecording
        )
    }

    private func persistAnalysisResult(
        _ analysis: JumpAnalysisResponse,
        videoURL: URL?,
        settings: AppSettings
    ) async -> Jump {
        let narration = await narratedSummary(for: analysis, settings: settings)
        let jump = makeJump(
            from: analysis,
            videoURL: videoURL,
            summary: narration.summary,
            llmNarratedSummary: narration.didUseLLM
        )
        jumpStore.add(jump)
        cloudSyncCoordinator.saveJump(jump)
        latestJump = jump
        return jump
    }

    private func athleteHeightCm(for units: MeasurementUnits) -> Double? {
        let sanitized = selectedAthleteProfile.height
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: ",", with: ".")
        guard let rawHeight = Double(sanitized), rawHeight > 0 else {
            return nil
        }

        switch units {
        case .metric:
            return rawHeight
        case .imperial:
            return rawHeight * 2.54
        }
    }

    private func receiveLiveGuidanceMessages() {
        guard let liveGuidanceSocket else { return }
        liveGuidanceSocket.receive { [weak self] result in
            guard let self else { return }
            switch result {
            case .failure(let error):
                Task { @MainActor in
                    if self.liveGuidanceState == .active || self.liveGuidanceState == .connecting {
                        self.liveGuidanceState = .failed
                        self.liveGuidanceMessage = error.localizedDescription
                        self.errorMessage = "Live guidance disconnected: \(error.localizedDescription)"
                    }
                    self.cameraManager.stopPreviewFrameDelivery()
                }
            case .success(let message):
                Task { @MainActor in
                    self.handleLiveGuidanceMessage(message)
                    self.receiveLiveGuidanceMessages()
                }
            }
        }
    }

    private func handleLiveGuidanceMessage(_ message: URLSessionWebSocketTask.Message) {
        let data: Data
        switch message {
        case .string(let text):
            guard let messageData = text.data(using: .utf8) else { return }
            data = messageData
        case .data(let binaryData):
            data = binaryData
        @unknown default:
            return
        }

        guard
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let type = object["type"] as? String
        else {
            return
        }

        let phase = object["phase"] as? String ?? liveGuidancePhase
        let text = object["text"] as? String ?? ""
        let speak = object["speak"] as? Bool ?? false
        let level = object["level"] as? String ?? "info"

        liveGuidancePhase = phase
        if !text.isEmpty {
            liveGuidanceMessage = text
        }

        switch type {
        case "status", "guidance":
            if liveGuidanceState != .completed {
                liveGuidanceState = .active
            }
        case "stop_streaming":
            liveGuidanceState = .active
            liveGuidanceMessage = text
            cameraManager.stopPreviewFrameDelivery()
        case "analysis_result":
            liveGuidanceState = .completed
            liveAnalysisSummary = text
            cameraManager.stopPreviewFrameDelivery()
            liveGuidanceSocket?.cancel(with: .normalClosure, reason: nil)
            liveGuidanceSocket = nil
            if let analysis = decodeLiveAnalysisResponse(from: object) {
                Task { @MainActor in
                    let jump = await self.persistAnalysisResult(
                        analysis,
                        videoURL: nil,
                        settings: self.settingsStore.settings
                    )
                    self.liveAnalysisSummary = jump.analysisSummary
                    self.successMessage = jump.llmNarratedSummary
                        ? "Live jump analyzed, narrated, and saved."
                        : "Live jump analyzed and saved."
                }
            } else {
                successMessage = "Live guided capture completed."
            }
        case "error":
            liveGuidanceState = .failed
            errorMessage = text
            cameraManager.stopPreviewFrameDelivery()
        default:
            break
        }

        if speak, !text.isEmpty {
            let now = Date()
            let isUrgent = level == "warning" || level == "error"
            let recentlySpokenSameText =
                text == lastSpokenGuidance &&
                (lastSpokenGuidanceAt.map { now.timeIntervalSince($0) < 5.0 } ?? false)

            guard !recentlySpokenSameText else { return }

            lastSpokenGuidance = text
            lastSpokenGuidanceAt = now
            let utterance = AVSpeechUtterance(string: text)
            utterance.rate = 0.5
            if isUrgent {
                speechSynthesizer.stopSpeaking(at: .immediate)
            } else if speechSynthesizer.isSpeaking {
                return
            }
            speechSynthesizer.speak(utterance)
        }
    }

    private func decodeLiveAnalysisResponse(from object: [String: Any]) -> JumpAnalysisResponse? {
        guard let payload = object["payload"] else { return nil }
        guard JSONSerialization.isValidJSONObject(payload) else { return nil }
        guard let data = try? JSONSerialization.data(withJSONObject: payload) else { return nil }
        return try? liveResultDecoder.decode(JumpAnalysisResponse.self, from: data)
    }

    private func sendLiveFrame(_ data: Data) {
        guard let liveGuidanceSocket else { return }
        let payload: [String: Any] = [
            "type": "frame",
            "image_base64": data.base64EncodedString(),
            "timestamp_ms": Int(Date().timeIntervalSince1970 * 1000)
        ]
        guard
            let jsonData = try? JSONSerialization.data(withJSONObject: payload),
            let text = String(data: jsonData, encoding: .utf8)
        else {
            return
        }

        liveGuidanceSocket.send(.string(text)) { [weak self] error in
            guard let self, let error else { return }
            Task { @MainActor in
                self.liveGuidanceState = .failed
                self.errorMessage = "Live guidance upload failed: \(error.localizedDescription)"
                self.cameraManager.stopPreviewFrameDelivery()
            }
        }
    }

    private func liveGuidanceURL(heightCm: Double) -> URL? {
        guard let baseURL = URL(string: settingsStore.settings.backendServerURL) else {
            return nil
        }

        var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false)
        components?.scheme = baseURL.scheme == "https" ? "wss" : "ws"
        components?.path = "/live-session"
        components?.queryItems = [
            URLQueryItem(name: "height_cm", value: String(heightCm))
        ]
        return components?.url
    }
}
