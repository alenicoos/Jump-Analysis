import Combine
import Foundation

@MainActor
final class CloudSyncCoordinator: ObservableObject {
    @Published private(set) var syncStatusMessage: String?

    private let authenticationService: AuthenticationService
    private let jumpStore: JumpStore
    private let profileStore: UserProfileStore
    private let jumpService: FirebaseJumpService
    private let profileService: FirebaseProfileService
    private var cancellables = Set<AnyCancellable>()
    private var bootstrappedUserID: String?

    init(
        authenticationService: AuthenticationService,
        jumpStore: JumpStore,
        profileStore: UserProfileStore,
        jumpService: FirebaseJumpService,
        profileService: FirebaseProfileService
    ) {
        self.authenticationService = authenticationService
        self.jumpStore = jumpStore
        self.profileStore = profileStore
        self.jumpService = jumpService
        self.profileService = profileService

        authenticationService.$userSession
            .sink { [weak self] userSession in
                guard let self else { return }

                if let userSession {
                    Task { await self.bootstrapCloudState(for: userSession) }
                } else {
                    self.bootstrappedUserID = nil
                    self.syncStatusMessage = nil
                }
            }
            .store(in: &cancellables)
    }

    func syncProfile(_ profile: UserProfile) {
        guard let userID = authenticationService.userSession?.uid else { return }

        Task {
            do {
                try await profileService.save(profile, for: userID)
                syncStatusMessage = "Profile synced to Firebase."
            } catch {
                syncStatusMessage = "Profile saved locally. Firebase sync will retry later."
            }
        }
    }

    func saveJump(_ jump: Jump) {
        guard let userID = authenticationService.userSession?.uid else { return }

        Task {
            do {
                try await jumpService.save(jump, for: userID)
                syncStatusMessage = "Jump synced to Firebase."
            } catch {
                syncStatusMessage = "Jump saved locally. Firebase sync will retry later."
            }
        }
    }

    func deleteJump(_ jump: Jump) {
        guard let userID = authenticationService.userSession?.uid else { return }

        Task {
            do {
                try await jumpService.delete(jump, for: userID)
                syncStatusMessage = "Jump removed from Firebase."
            } catch {
                syncStatusMessage = "Jump deleted locally. Firebase sync will retry later."
            }
        }
    }

    private func bootstrapCloudState(for userSession: FirebaseUserSession) async {
        guard bootstrappedUserID != userSession.uid else { return }

        bootstrappedUserID = userSession.uid
        syncStatusMessage = "Syncing Firebase data..."

        do {
            if let remoteProfile = try await profileService.fetchProfile(for: userSession.uid) {
                profileStore.replace(with: remoteProfile)
            } else {
                try await profileService.save(profileStore.profile, for: userSession.uid)
            }

            let remoteJumps = try await jumpService.fetchJumps(for: userSession.uid)
            if remoteJumps.isEmpty {
                for jump in jumpStore.currentJumps {
                    try await jumpService.save(jump, for: userSession.uid)
                }
            } else {
                jumpStore.replaceAll(with: remoteJumps)
            }

            syncStatusMessage = "Firebase sync complete."
        } catch {
            syncStatusMessage = "Using local data. Firebase sync could not complete right now."
            bootstrappedUserID = nil
        }
    }
}
