import Combine
import Foundation

@MainActor
final class CameraViewModel: ObservableObject {
    enum AnalysisState: Equatable {
        case idle
        case uploading
        case completed
    }

    @Published var analysisState: AnalysisState = .idle
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
    private var cancellables = Set<AnyCancellable>()

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

    func onAppear() {
        guard AppPlatform.supportsLiveCameraCapture else { return }
        cameraManager.requestPermissionsAndConfigureIfNeeded()
    }

    func onDisappear() {
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

    func analyzeRecordedJump() async {
        errorMessage = nil
        successMessage = nil
        analysisState = .uploading

        do {
            let settings = settingsStore.settings
            let analysis = try await analysisService.analyzeJump(
                videoURL: recordedVideoURL,
                settings: settings,
                athleteHeightCm: athleteHeightCm(for: settings.units)
            )

            let narration = await narratedSummary(for: analysis, settings: settings)
            let jump = makeJump(from: analysis, summary: narration.summary, llmNarratedSummary: narration.didUseLLM)

            jumpStore.add(jump)
            cloudSyncCoordinator.saveJump(jump)
            latestJump = jump
            analysisState = .completed
            successMessage = narration.didUseLLM
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

    private func makeJump(from analysis: JumpAnalysisResponse, summary: String, llmNarratedSummary: Bool) -> Jump {
        Jump(
            date: analysis.timestamp,
            videoURL: recordedVideoURL,
            protocolPassed: analysis.protocolPassed,
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
            kneeAsymmetryRatio: analysis.kneeAsymmetryRatio
        )
    }

    private func athleteHeightCm(for units: MeasurementUnits) -> Double? {
        let sanitized = profileStore.profile.height
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
}
