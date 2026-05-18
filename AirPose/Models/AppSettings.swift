import SwiftUI

struct AppSettings: Codable, Equatable {
    // The desktop build can use loopback directly, while iPhone builds should
    // point at the Mac over LAN.
    var backendServerURL: String = AppPlatform.defaultBackendURL
    var llmAPIURL: String = "https://api.example.com/v1/feedback"
    var llmAPIKey: String = "sk-placeholder"
    var units: MeasurementUnits = .metric
    var mockModeEnabled: Bool = true
    var themePreference: ThemePreference = .system
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
