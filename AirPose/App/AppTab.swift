import Foundation

enum AppTab: Hashable {
    case home
    case camera
    case jumps
    case profile
    case settings

    var title: String {
        switch self {
        case .home: "Home"
        case .camera: "Camera"
        case .jumps: "Jumps"
        case .profile: "Profile"
        case .settings: "Settings"
        }
    }

    var systemImage: String {
        switch self {
        case .home: "house.fill"
        case .camera: "video.fill"
        case .jumps: "figure.run"
        case .profile: "person.fill"
        case .settings: "gearshape.fill"
        }
    }
}
