import Foundation

@MainActor
final class UserProfileStore: ObservableObject {
    @Published var profile: UserProfile {
        didSet { save() }
    }

    private let fileStore = LocalJSONFileStore<UserProfile>(filename: "user-profile.json")

    init() {
        profile = fileStore.load(defaultValue: UserProfile())
    }

    func replace(with profile: UserProfile) {
        self.profile = profile
    }

    private func save() {
        try? fileStore.save(profile)
    }
}
