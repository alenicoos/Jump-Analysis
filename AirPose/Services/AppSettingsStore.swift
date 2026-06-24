import Foundation

@MainActor
final class AppSettingsStore: ObservableObject {
    @Published var settings: AppSettings {
        didSet { save() }
    }

    private let fileStore = LocalJSONFileStore<AppSettings>(filename: "app-settings.json")

    init() {
        let loadedSettings = fileStore.load(defaultValue: AppSettings())
        let migratedSettings = Self.migrateIfNeeded(loadedSettings)
        settings = migratedSettings

        if migratedSettings != loadedSettings {
            try? fileStore.save(migratedSettings)
        }
    }

    private func save() {
        try? fileStore.save(settings)
    }

    private static func migrateIfNeeded(_ settings: AppSettings) -> AppSettings {
        var migrated = settings
        let backendURL = settings.backendServerURL.trimmingCharacters(in: .whitespacesAndNewlines)

        if backendURL == AppPlatform.legacyBackendURL {
            migrated.backendServerURL = AppPlatform.currentBackendURL
        }

        return migrated
    }
}
