import SwiftUI

@main
struct AirPoseApp: App {
    @Environment(\.scenePhase) private var scenePhase
    @UIApplicationDelegateAdaptor(FirebaseAppDelegate.self) private var appDelegate
    @StateObject private var jumpStore: JumpStore
    @StateObject private var profileStore: UserProfileStore
    @StateObject private var settingsStore: AppSettingsStore
    @StateObject private var authenticationService: AuthenticationService
    @StateObject private var cloudSyncCoordinator: CloudSyncCoordinator
    @StateObject private var jumpNarrationCoordinator: JumpNarrationCoordinator
    @StateObject private var cameraViewModel: CameraViewModel
    @StateObject private var authViewModel: AuthViewModel
    @StateObject private var tabRouter = TabRouter()

    init() {
        FirebaseBootstrap.configureIfNeeded()

        let jumpStore = JumpStore()
        let profileStore = UserProfileStore()
        let settingsStore = AppSettingsStore()
        let authenticationService = AuthenticationService()
        let llmFeedbackService = LLMFeedbackService()
        let cloudSyncCoordinator = CloudSyncCoordinator(
            authenticationService: authenticationService,
            jumpStore: jumpStore,
            profileStore: profileStore,
            jumpService: FirebaseJumpService(),
            profileService: FirebaseProfileService()
        )
        let jumpNarrationCoordinator = JumpNarrationCoordinator(
            jumpStore: jumpStore,
            settingsStore: settingsStore,
            cloudSyncCoordinator: cloudSyncCoordinator,
            llmFeedbackService: llmFeedbackService
        )

        _jumpStore = StateObject(wrappedValue: jumpStore)
        _profileStore = StateObject(wrappedValue: profileStore)
        _settingsStore = StateObject(wrappedValue: settingsStore)
        _authenticationService = StateObject(wrappedValue: authenticationService)
        _cloudSyncCoordinator = StateObject(wrappedValue: cloudSyncCoordinator)
        _jumpNarrationCoordinator = StateObject(wrappedValue: jumpNarrationCoordinator)
        _cameraViewModel = StateObject(
            wrappedValue: CameraViewModel(
                jumpStore: jumpStore,
                profileStore: profileStore,
                settingsStore: settingsStore,
                cloudSyncCoordinator: cloudSyncCoordinator,
                analysisService: JumpAnalysisService(),
                llmFeedbackService: llmFeedbackService
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
            .environmentObject(jumpNarrationCoordinator)
            .environmentObject(tabRouter)
            .preferredColorScheme(settingsStore.settings.themePreference.colorScheme)
            .task {
                cloudSyncCoordinator.retryCloudSyncIfPossible()
                jumpNarrationCoordinator.backfillExistingJumpsIfNeeded()
            }
            .onChange(of: scenePhase) { _, newPhase in
                guard newPhase == .active else { return }
                cloudSyncCoordinator.retryCloudSyncIfPossible()
            }
        }
    }
}
