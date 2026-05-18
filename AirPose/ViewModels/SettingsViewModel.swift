import Foundation

@MainActor
final class SettingsViewModel: ObservableObject {
    @Published var settings: AppSettings

    private let settingsStore: AppSettingsStore

    init(settingsStore: AppSettingsStore) {
        self.settingsStore = settingsStore
        settings = settingsStore.settings
    }

    func persist() {
        settingsStore.settings = settings
    }
}
