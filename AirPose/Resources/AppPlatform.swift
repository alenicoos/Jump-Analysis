import Foundation

enum AppPlatform {
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
        if isDesktopDemo {
            return "http://127.0.0.1:8000/analyze-jump"
        }
        return "http://localhost:8000/analyze-jump"
    }

    static var backendPlaceholder: String {
        if isDesktopDemo {
            return "http://127.0.0.1:8000/analyze-jump"
        }
        return "http://192.168.1.X:8000/analyze-jump"
    }

    static var backendHelpText: String {
        if isDesktopDemo {
            return "On the Mac app, use 127.0.0.1 so AirPose can reach the local analysis server directly."
        }
        return "On a physical iPhone, use your Mac's LAN IP instead of localhost."
    }

    static var analysisTipText: String {
        if isDesktopDemo {
            return "Tip: this desktop build can talk to your local backend directly with `http://127.0.0.1:8000/analyze-jump`."
        }
        return "Tip: `localhost` on a physical iPhone points to the phone itself, not your Mac. Use your Mac's LAN IP instead, such as `http://192.168.1.X:8000/analyze-jump`."
    }
}
