import Foundation

enum AppPlatform {
    static let legacyBackendURL = "http://airpose.duckdns.org:8000/analyze-jump"
    static let currentBackendURL = "http://jumpguard.ddns.net:8000/analyze-jump"

    static var isDesktopDemo: Bool {
        #if targetEnvironment(macCatalyst)
        return true
        #else
        return ProcessInfo.processInfo.isiOSAppOnMac
        #endif
    }

    static var supportsLiveCameraCapture: Bool {
        true
    }

    static var defaultBackendURL: String {
        currentBackendURL
    }

    static var backendPlaceholder: String {
        currentBackendURL
    }

    static var backendHelpText: String {
        "JumpGuard is configured to use the public DDNS backend by default."
    }

    static var analysisTipText: String {
        return "Tip: the app uses `\(currentBackendURL)` by default for both recorded analysis and live guidance."
    }
}
