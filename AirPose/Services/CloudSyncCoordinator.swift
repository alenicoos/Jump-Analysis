import Combine
import Foundation

private enum PendingCloudMutationKind: String, Codable {
    case profileUpsert
    case jumpUpsert
    case jumpDelete
}

private struct PendingCloudMutation: Codable, Identifiable {
    let id: String
    let userID: String
    let kind: PendingCloudMutationKind
    let profile: UserProfile?
    let jump: Jump?
    let jumpID: UUID?
    let createdAt: Date

    static func profile(userID: String, profile: UserProfile) -> PendingCloudMutation {
        PendingCloudMutation(
            id: "\(userID):profile",
            userID: userID,
            kind: .profileUpsert,
            profile: profile,
            jump: nil,
            jumpID: nil,
            createdAt: Date()
        )
    }

    static func jumpUpsert(userID: String, jump: Jump) -> PendingCloudMutation {
        PendingCloudMutation(
            id: "\(userID):jump:\(jump.id.uuidString):upsert",
            userID: userID,
            kind: .jumpUpsert,
            profile: nil,
            jump: jump,
            jumpID: jump.id,
            createdAt: Date()
        )
    }

    static func jumpDelete(userID: String, jumpID: UUID) -> PendingCloudMutation {
        PendingCloudMutation(
            id: "\(userID):jump:\(jumpID.uuidString):delete",
            userID: userID,
            kind: .jumpDelete,
            profile: nil,
            jump: nil,
            jumpID: jumpID,
            createdAt: Date()
        )
    }
}

@MainActor
final class CloudSyncCoordinator: ObservableObject {
    @Published private(set) var syncStatusMessage: String?

    private let authenticationService: AuthenticationService
    private let jumpStore: JumpStore
    private let profileStore: UserProfileStore
    private let jumpService: FirebaseJumpService
    private let profileService: FirebaseProfileService
    private let pendingMutationStore = LocalJSONFileStore<[PendingCloudMutation]>(filename: "pending-cloud-sync.json")
    private var cancellables = Set<AnyCancellable>()
    private var bootstrappedUserID: String?
    private var pendingMutations: [PendingCloudMutation]

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
        self.pendingMutations = pendingMutationStore.load(defaultValue: [])

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

    func retryCloudSyncIfPossible() {
        guard let userSession = authenticationService.userSession else { return }
        Task { await bootstrapCloudState(for: userSession, forceRefresh: true) }
    }

    func syncProfile(_ profile: UserProfile) {
        guard let userID = authenticationService.userSession?.uid else { return }

        Task {
            do {
                try await profileService.save(profile, for: userID)
                clearPendingProfileMutation(for: userID)
                syncStatusMessage = "Profile synced to Firebase."
            } catch {
                queueProfileMutation(profile, for: userID)
                syncStatusMessage = "Profile saved locally. Firebase sync will retry later."
            }
        }
    }

    func saveJump(_ jump: Jump) {
        guard let userID = authenticationService.userSession?.uid else { return }

        Task {
            do {
                try await jumpService.save(jump, for: userID)
                clearPendingJumpMutation(for: jump.id, userID: userID)
                syncStatusMessage = "Jump synced to Firebase."
            } catch {
                queueJumpUpsertMutation(jump, for: userID)
                syncStatusMessage = "Jump saved locally. Firebase sync will retry later."
            }
        }
    }

    func deleteJump(_ jump: Jump) {
        guard let userID = authenticationService.userSession?.uid else { return }

        Task {
            do {
                try await jumpService.delete(jump, for: userID)
                clearPendingJumpMutation(for: jump.id, userID: userID)
                syncStatusMessage = "Jump removed from Firebase."
            } catch {
                queueJumpDeleteMutation(jump.id, for: userID)
                syncStatusMessage = "Jump deleted locally. Firebase sync will retry later."
            }
        }
    }

    private func mergedJumps(local localJumps: [Jump], remote remoteJumps: [Jump]) -> [Jump] {
        var mergedByID = Dictionary(uniqueKeysWithValues: remoteJumps.map { ($0.id, $0) })
        for jump in localJumps {
            if let remoteJump = mergedByID[jump.id], jump.cloudDocumentID == nil {
                mergedByID[jump.id] = jump.withCloudDocumentID(remoteJump.cloudDocumentID)
            } else {
                mergedByID[jump.id] = jump
            }
        }
        return mergedByID.values.sorted { $0.date > $1.date }
    }

    private func bootstrapCloudState(for userSession: FirebaseUserSession, forceRefresh: Bool = false) async {
        guard forceRefresh || bootstrappedUserID != userSession.uid || hasPendingMutations(for: userSession.uid) else { return }

        bootstrappedUserID = userSession.uid
        syncStatusMessage = "Syncing Firebase data..."

        do {
            try await flushPendingMutations(for: userSession.uid)

            if let remoteProfile = try await profileService.fetchProfile(for: userSession.uid) {
                profileStore.replace(with: remoteProfile)
            } else {
                try await profileService.save(profileStore.profile, for: userSession.uid)
            }

            let remoteJumps = try await jumpService.fetchJumps(for: userSession.uid)
            let localJumps = jumpStore.currentJumps
            if remoteJumps.isEmpty {
                for jump in localJumps {
                    try await jumpService.save(jump, for: userSession.uid)
                }
            } else {
                let mergedJumps = mergedJumps(local: localJumps, remote: remoteJumps)
                jumpStore.replaceAll(with: mergedJumps)

                let remoteJumpIDs = Set(remoteJumps.map(\.id))
                for jump in localJumps where !remoteJumpIDs.contains(jump.id) {
                    try await jumpService.save(jump, for: userSession.uid)
                }
            }

            syncStatusMessage = "Firebase sync complete."
        } catch {
            syncStatusMessage = "Using local data. Firebase sync could not complete right now."
            bootstrappedUserID = nil
        }
    }

    private func hasPendingMutations(for userID: String) -> Bool {
        pendingMutations.contains { $0.userID == userID }
    }

    private func queueProfileMutation(_ profile: UserProfile, for userID: String) {
        pendingMutations.removeAll { $0.userID == userID && $0.kind == .profileUpsert }
        pendingMutations.append(.profile(userID: userID, profile: profile))
        savePendingMutations()
    }

    private func queueJumpUpsertMutation(_ jump: Jump, for userID: String) {
        pendingMutations.removeAll {
            $0.userID == userID && $0.jumpID == jump.id && ($0.kind == .jumpUpsert || $0.kind == .jumpDelete)
        }
        pendingMutations.append(.jumpUpsert(userID: userID, jump: jump))
        savePendingMutations()
    }

    private func queueJumpDeleteMutation(_ jumpID: UUID, for userID: String) {
        pendingMutations.removeAll {
            $0.userID == userID && $0.jumpID == jumpID && ($0.kind == .jumpUpsert || $0.kind == .jumpDelete)
        }
        pendingMutations.append(.jumpDelete(userID: userID, jumpID: jumpID))
        savePendingMutations()
    }

    private func clearPendingProfileMutation(for userID: String) {
        pendingMutations.removeAll { $0.userID == userID && $0.kind == .profileUpsert }
        savePendingMutations()
    }

    private func clearPendingJumpMutation(for jumpID: UUID, userID: String) {
        pendingMutations.removeAll {
            $0.userID == userID && $0.jumpID == jumpID && ($0.kind == .jumpUpsert || $0.kind == .jumpDelete)
        }
        savePendingMutations()
    }

    private func flushPendingMutations(for userID: String) async throws {
        let operations = pendingMutations
            .filter { $0.userID == userID }
            .sorted { $0.createdAt < $1.createdAt }

        for operation in operations {
            switch operation.kind {
            case .profileUpsert:
                guard let profile = operation.profile else { continue }
                try await profileService.save(profile, for: userID)
                clearPendingProfileMutation(for: userID)
            case .jumpUpsert:
                guard let jump = operation.jump else { continue }
                try await jumpService.save(jump, for: userID)
                clearPendingJumpMutation(for: jump.id, userID: userID)
            case .jumpDelete:
                guard let jumpID = operation.jumpID else { continue }
                try await jumpService.deleteJump(withID: jumpID, for: userID)
                clearPendingJumpMutation(for: jumpID, userID: userID)
            }
        }
    }

    private func savePendingMutations() {
        try? pendingMutationStore.save(pendingMutations)
    }
}
