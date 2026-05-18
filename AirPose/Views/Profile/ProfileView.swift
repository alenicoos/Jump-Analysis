import SwiftUI

struct ProfileView: View {
    @ObservedObject var viewModel: ProfileViewModel
    @EnvironmentObject private var settingsStore: AppSettingsStore
    @EnvironmentObject private var authenticationService: AuthenticationService

    var body: some View {
        GeometryReader { proxy in
            Form {
                Section("Account") {
                    if let userSession = authenticationService.userSession {
                        LabeledContent("Signed In") {
                            Text(userSession.email ?? "Firebase User")
                        }

                        Button("Sign Out", role: .destructive) {
                            authenticationService.signOut()
                        }
                    } else {
                        Text("You are using AirPose in offline mode.")
                            .foregroundStyle(.secondary)

                        Button("Sign In to Sync") {
                            authenticationService.leaveGuestMode()
                        }
                    }
                }

                Section("Athlete") {
                    TextField("Name", text: $viewModel.profile.name)
                    TextField("Age", text: $viewModel.profile.age)
                        .keyboardType(.numberPad)
                    TextField(settingsStore.settings.units.heightLabel, text: $viewModel.profile.height)
                        .keyboardType(.decimalPad)
                    TextField(settingsStore.settings.units.weightLabel, text: $viewModel.profile.weight)
                        .keyboardType(.decimalPad)
                }

                Section("Performance") {
                    Picker("Dominant Leg", selection: $viewModel.profile.dominantLeg) {
                        ForEach(DominantLeg.allCases) { leg in
                            Text(leg.rawValue).tag(leg)
                        }
                    }

                    TextField("Sport", text: $viewModel.profile.sport)

                    Picker("Experience Level", selection: $viewModel.profile.experienceLevel) {
                        ForEach(ExperienceLevel.allCases) { level in
                            Text(level.rawValue).tag(level)
                        }
                    }
                }
            }
            .frame(maxWidth: AirPoseLayout.contentMaxWidth(for: proxy.size))
            .frame(maxWidth: .infinity)
            .scrollContentBackground(.hidden)
            .background(BrandBackground())
            .padding(.horizontal, AirPoseLayout.horizontalPadding(for: proxy.size))
            .padding(.top, 20)
            .padding(.bottom, 36)
        }
        .navigationTitle("Profile")
        .navigationBarTitleDisplayMode(.inline)
        .onChange(of: viewModel.profile) { _, _ in
            viewModel.persist()
        }
    }
}
