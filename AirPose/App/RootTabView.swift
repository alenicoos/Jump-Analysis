import SwiftUI

struct RootTabView: View {
    @ObservedObject var cameraViewModel: CameraViewModel
    @EnvironmentObject private var tabRouter: TabRouter
    @EnvironmentObject private var jumpStore: JumpStore
    @EnvironmentObject private var profileStore: UserProfileStore
    @EnvironmentObject private var settingsStore: AppSettingsStore
    @EnvironmentObject private var cloudSyncCoordinator: CloudSyncCoordinator

    var body: some View {
        TabView(selection: $tabRouter.selectedTab) {
            NavigationStack {
                DashboardView(viewModel: DashboardViewModel(jumpStore: jumpStore))
            }
            .tabItem {
                Label(AppTab.home.title, systemImage: AppTab.home.systemImage)
            }
            .tag(AppTab.home)

            NavigationStack {
                CameraView(viewModel: cameraViewModel)
            }
            .tabItem {
                Label(AppTab.camera.title, systemImage: AppTab.camera.systemImage)
            }
            .tag(AppTab.camera)

            NavigationStack {
                JumpsView(viewModel: JumpsViewModel(jumpStore: jumpStore, cloudSyncCoordinator: cloudSyncCoordinator))
            }
            .tabItem {
                Label(AppTab.jumps.title, systemImage: AppTab.jumps.systemImage)
            }
            .tag(AppTab.jumps)

            NavigationStack {
                ProfileView(viewModel: ProfileViewModel(profileStore: profileStore, cloudSyncCoordinator: cloudSyncCoordinator))
            }
            .tabItem {
                Label(AppTab.profile.title, systemImage: AppTab.profile.systemImage)
            }
            .tag(AppTab.profile)

            NavigationStack {
                SettingsView(viewModel: SettingsViewModel(settingsStore: settingsStore))
            }
            .tabItem {
                Label(AppTab.settings.title, systemImage: AppTab.settings.systemImage)
            }
            .tag(AppTab.settings)
        }
        .tint(.airPoseElectricBlue)
    }
}
