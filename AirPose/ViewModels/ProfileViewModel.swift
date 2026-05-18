import Foundation

@MainActor
final class ProfileViewModel: ObservableObject {
    @Published var profile: UserProfile

    private let profileStore: UserProfileStore
    private let cloudSyncCoordinator: CloudSyncCoordinator

    init(profileStore: UserProfileStore, cloudSyncCoordinator: CloudSyncCoordinator) {
        self.profileStore = profileStore
        self.cloudSyncCoordinator = cloudSyncCoordinator
        profile = profileStore.profile
    }

    func persist() {
        profileStore.profile = profile
        cloudSyncCoordinator.syncProfile(profile)
    }
}
