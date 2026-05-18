import SwiftUI

@main
struct AirPoseApp: App {
    @UIApplicationDelegateAdaptor(FirebaseAppDelegate.self) private var appDelegate
    @StateObject private var jumpStore: JumpStore
    @StateObject private var profileStore: UserProfileStore
    @StateObject private var settingsStore: AppSettingsStore
    @StateObject private var authenticationService: AuthenticationService
    @StateObject private var cloudSyncCoordinator: CloudSyncCoordinator
    @StateObject private var cameraViewModel: CameraViewModel
    @StateObject private var authViewModel: AuthViewModel
    @StateObject private var tabRouter = TabRouter()

    init() {
        FirebaseBootstrap.configureIfNeeded()

        let jumpStore = JumpStore()
        let profileStore = UserProfileStore()
        let settingsStore = AppSettingsStore()
        let authenticationService = AuthenticationService()
        let cloudSyncCoordinator = CloudSyncCoordinator(
            authenticationService: authenticationService,
            jumpStore: jumpStore,
            profileStore: profileStore,
            jumpService: FirebaseJumpService(),
            profileService: FirebaseProfileService()
        )

        _jumpStore = StateObject(wrappedValue: jumpStore)
        _profileStore = StateObject(wrappedValue: profileStore)
        _settingsStore = StateObject(wrappedValue: settingsStore)
        _authenticationService = StateObject(wrappedValue: authenticationService)
        _cloudSyncCoordinator = StateObject(wrappedValue: cloudSyncCoordinator)
        _cameraViewModel = StateObject(
            wrappedValue: CameraViewModel(
                jumpStore: jumpStore,
                profileStore: profileStore,
                settingsStore: settingsStore,
                cloudSyncCoordinator: cloudSyncCoordinator,
                analysisService: JumpAnalysisService()
            )
        )
        _authViewModel = StateObject(wrappedValue: AuthViewModel(authenticationService: authenticationService))
    }

    var body: some Scene {
        WindowGroup {
            Group {
                if authenticationService.canAccessApp {
                    RootTabView(cameraViewModel: cameraViewModel)
                } else {
                    NavigationStack {
                        AuthenticationView(viewModel: authViewModel)
                    }
                }
            }
            .environmentObject(jumpStore)
            .environmentObject(profileStore)
            .environmentObject(settingsStore)
            .environmentObject(authenticationService)
            .environmentObject(cloudSyncCoordinator)
            .environmentObject(tabRouter)
            .preferredColorScheme(settingsStore.settings.themePreference.colorScheme)
        }
    }
}
