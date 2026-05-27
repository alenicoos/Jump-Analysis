import Combine
import Foundation

@MainActor
final class JumpNarrationCoordinator: ObservableObject {
    private let jumpStore: JumpStore
    private let settingsStore: AppSettingsStore
    private let cloudSyncCoordinator: CloudSyncCoordinator
    private let llmFeedbackService: LLMFeedbackService
    private var cancellables = Set<AnyCancellable>()
    private var attemptedJumpIDs = Set<UUID>()
    private var activeTask: Task<Void, Never>?

    init(
        jumpStore: JumpStore,
        settingsStore: AppSettingsStore,
        cloudSyncCoordinator: CloudSyncCoordinator,
        llmFeedbackService: LLMFeedbackService
    ) {
        self.jumpStore = jumpStore
        self.settingsStore = settingsStore
        self.cloudSyncCoordinator = cloudSyncCoordinator
        self.llmFeedbackService = llmFeedbackService

        jumpStore.$jumps
            .sink { [weak self] _ in
                self?.scheduleNextBackfillIfNeeded()
            }
            .store(in: &cancellables)

        settingsStore.$settings
            .dropFirst()
            .sink { [weak self] _ in
                self?.attemptedJumpIDs.removeAll()
                self?.scheduleNextBackfillIfNeeded()
            }
            .store(in: &cancellables)
    }

    func backfillExistingJumpsIfNeeded() {
        scheduleNextBackfillIfNeeded()
    }

    func reprocessExistingJumps() {
        attemptedJumpIDs.removeAll()
        activeTask?.cancel()
        activeTask = nil

        let refreshedJumps = jumpStore.currentJumps.map { jump in
            jump.llmNarratedSummary ? jump.resettingNarrationStatus() : jump
        }
        jumpStore.replaceAll(with: refreshedJumps)
        scheduleNextBackfillIfNeeded()
    }

    private func scheduleNextBackfillIfNeeded() {
        guard activeTask == nil else { return }
        guard isLLMConfigurationUsable(settingsStore.settings) else { return }
        guard let jump = nextPendingJump() else { return }

        attemptedJumpIDs.insert(jump.id)
        let settings = settingsStore.settings

        activeTask = Task { [weak self] in
            guard let self else { return }
            defer {
                Task { @MainActor [weak self] in
                    self?.activeTask = nil
                    self?.scheduleNextBackfillIfNeeded()
                }
            }

            do {
                let feedback = try await llmFeedbackService.generateFeedback(for: jump.analysisResponse, settings: settings)
                let trimmed = feedback.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty else { return }

                let updatedJump = jump.updatingNarration(summary: trimmed, llmNarratedSummary: true)
                await MainActor.run {
                    self.jumpStore.update(updatedJump)
                    self.cloudSyncCoordinator.saveJump(updatedJump)
                }
            } catch {
                return
            }
        }
    }

    private func nextPendingJump() -> Jump? {
        jumpStore.currentJumps.first { jump in
            !jump.llmNarratedSummary && !attemptedJumpIDs.contains(jump.id)
        }
    }

    private func isLLMConfigurationUsable(_ settings: AppSettings) -> Bool {
        settings.hasConfiguredLLMEndpoint
    }
}
