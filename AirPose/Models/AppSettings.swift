import SwiftUI

struct AppSettings: Codable, Equatable {
    static let defaultLLMAPIURL = "https://api.openai.com/v1/responses"
    static let placeholderLLMAPIKey = "paste-openai-api-key-here"

    // The desktop build can use loopback directly, while iPhone builds should
    // point at the Mac over LAN.
    var backendServerURL: String = AppPlatform.defaultBackendURL
    var llmAPIURL: String = AppSettings.defaultLLMAPIURL
    var llmAPIKey: String = AppSettings.placeholderLLMAPIKey
    var units: MeasurementUnits = .metric
    var mockModeEnabled: Bool = true
    var themePreference: ThemePreference = .system

    var hasConfiguredLLMEndpoint: Bool {
        let endpoint = llmAPIURL.trimmingCharacters(in: .whitespacesAndNewlines)
        let key = llmAPIKey.trimmingCharacters(in: .whitespacesAndNewlines)
        return !mockModeEnabled
            && !endpoint.isEmpty
            && endpoint == AppSettings.defaultLLMAPIURL
            && !key.isEmpty
            && key != AppSettings.placeholderLLMAPIKey
    }
}

enum MeasurementUnits: String, Codable, CaseIterable, Identifiable {
    case metric = "Metric"
    case imperial = "Imperial"

    var id: String { rawValue }

    var heightLabel: String {
        switch self {
        case .metric: "Height (cm)"
        case .imperial: "Height (in)"
        }
    }

    var weightLabel: String {
        switch self {
        case .metric: "Weight (kg)"
        case .imperial: "Weight (lb)"
        }
    }
}

enum ThemePreference: String, Codable, CaseIterable, Identifiable {
    case system = "System"
    case light = "Light"
    case dark = "Dark"

    var id: String { rawValue }

    var colorScheme: ColorScheme? {
        switch self {
        case .system: nil
        case .light: .light
        case .dark: .dark
        }
    }
}
