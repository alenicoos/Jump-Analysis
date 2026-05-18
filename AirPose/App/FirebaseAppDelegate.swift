import FirebaseCore
import UIKit

final class FirebaseAppDelegate: NSObject, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil
    ) -> Bool {
        FirebaseBootstrap.configureIfNeeded()
        return true
    }
}

enum FirebaseBootstrap {
    static func configureIfNeeded() {
        if FirebaseApp.app() == nil {
            FirebaseApp.configure()
        }
    }
}
