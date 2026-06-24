import Combine
import Foundation

@MainActor
final class ProfileViewModel: ObservableObject {
    @Published var profile: UserProfile

    private let profileStore: UserProfileStore
    private let cloudSyncCoordinator: CloudSyncCoordinator
    private var cancellables = Set<AnyCancellable>()

    init(profileStore: UserProfileStore, cloudSyncCoordinator: CloudSyncCoordinator) {
        self.profileStore = profileStore
        self.cloudSyncCoordinator = cloudSyncCoordinator
        profile = profileStore.profile

        profileStore.$profile
            .sink { [weak self] updatedProfile in
                guard let self else { return }
                if self.profile != updatedProfile {
                    self.profile = updatedProfile
                }
            }
            .store(in: &cancellables)
    }

    func persist() {
        profileStore.profile = profile
        cloudSyncCoordinator.syncProfile(profile)
    }
}
