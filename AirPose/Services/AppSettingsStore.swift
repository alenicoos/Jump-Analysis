import Foundation

@MainActor
final class AppSettingsStore: ObservableObject {
    @Published var settings: AppSettings {
        didSet { save() }
    }

    private let fileStore = LocalJSONFileStore<AppSettings>(filename: "app-settings.json")

    init() {
        settings = fileStore.load(defaultValue: AppSettings())
    }

    private func save() {
        try? fileStore.save(settings)
    }
}
