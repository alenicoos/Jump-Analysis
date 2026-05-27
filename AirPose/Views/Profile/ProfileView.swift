import SwiftUI
import UIKit

struct ProfileView: View {
    @ObservedObject var viewModel: ProfileViewModel
    @EnvironmentObject private var settingsStore: AppSettingsStore
    @EnvironmentObject private var authenticationService: AuthenticationService

    var body: some View {
        AirPoseScrollCanvas { size in
            VStack(alignment: .leading, spacing: 20) {
                headerCard
                accountCard
                athleteDetailsSection(for: size)
            }
        }
        .navigationTitle("Profile")
        .navigationBarTitleDisplayMode(.inline)
        .onChange(of: viewModel.profile) { _, _ in
            viewModel.persist()
        }
    }

    private var headerCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 14) {
                SectionHeader("Athlete Profile", subtitle: "Keep your personal and performance details current so every analysis stays contextual.")

                HStack(alignment: .center, spacing: 14) {
                    ZStack {
                        Circle()
                            .fill(LinearGradient.airPoseBluePurple)
                            .frame(width: 58, height: 58)

                        Image(systemName: "figure.strengthtraining.functional")
                            .font(.title2.weight(.semibold))
                            .foregroundStyle(.white)
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        Text(displayName)
                            .font(.title2.weight(.bold))
                            .foregroundStyle(Color.airPosePrimaryText)

                        Text(profileSummary)
                            .font(.subheadline)
                            .foregroundStyle(Color.airPoseSecondaryText)
                    }
                }
            }
        }
    }

    private var accountCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 16) {
                SectionHeader("Account", subtitle: "Manage sign-in and cloud sync for your AirPose history.")

                if let userSession = authenticationService.userSession {
                    VStack(alignment: .leading, spacing: 12) {
                        infoRow(title: "Signed In", value: userSession.email ?? "Firebase User")
                        infoRow(title: "Sync Status", value: "Cloud sync active")

                        Button("Sign Out", role: .destructive) {
                            authenticationService.signOut()
                        }
                        .buttonStyle(.bordered)
                    }
                } else {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("You are using AirPose in offline mode.")
                            .font(.subheadline)
                            .foregroundStyle(Color.airPoseSecondaryText)

                        Button("Sign In to Sync") {
                            authenticationService.leaveGuestMode()
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.airPoseElectricBlue)
                    }
                }
            }
        }
    }

    private func athleteDetailsSection(for size: CGSize) -> some View {
        let columns = athleteDetailColumns(for: size)

        return LazyVGrid(columns: columns, alignment: .leading, spacing: 16) {
            GlassCard {
                VStack(alignment: .leading, spacing: 18) {
                    SectionHeader("Athlete", subtitle: "Basic identity and body metrics.")

                    profileField("Name", text: $viewModel.profile.name, textContentType: .name, textInputAutocapitalization: .words)
                    profileField("Age", text: $viewModel.profile.age, keyboardType: .numberPad, textInputAutocapitalization: .never)
                    profileField(settingsStore.settings.units.heightLabel, text: $viewModel.profile.height, keyboardType: .decimalPad, textInputAutocapitalization: .never)
                    profileField(settingsStore.settings.units.weightLabel, text: $viewModel.profile.weight, keyboardType: .decimalPad, textInputAutocapitalization: .never)
                }
            }

            GlassCard {
                VStack(alignment: .leading, spacing: 18) {
                    SectionHeader("Performance", subtitle: "Configure sport context and movement preferences.")

                    VStack(alignment: .leading, spacing: 8) {
                        fieldLabel("Dominant Leg")

                        Picker("Dominant Leg", selection: $viewModel.profile.dominantLeg) {
                            ForEach(DominantLeg.allCases) { leg in
                                Text(leg.rawValue).tag(leg)
                            }
                        }
                        .pickerStyle(.segmented)
                    }

                    profileField("Sport", text: $viewModel.profile.sport, textInputAutocapitalization: .words)

                    VStack(alignment: .leading, spacing: 8) {
                        fieldLabel("Experience Level")

                        Picker("Experience Level", selection: $viewModel.profile.experienceLevel) {
                            ForEach(ExperienceLevel.allCases) { level in
                                Text(level.rawValue).tag(level)
                            }
                        }
                        .pickerStyle(.menu)
                    }
                }
            }
        }
    }

    private func athleteDetailColumns(for size: CGSize) -> [GridItem] {
        if AppPlatform.isDesktopDemo && size.width >= 1180 {
            return Array(repeating: GridItem(.flexible(), spacing: 16, alignment: .top), count: 2)
        }

        return [GridItem(.flexible(), spacing: 16)]
    }

    private func profileField(
        _ title: String,
        text: Binding<String>,
        keyboardType: UIKeyboardType = .default,
        textContentType: UITextContentType? = nil,
        textInputAutocapitalization: TextInputAutocapitalization = .sentences
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            fieldLabel(title)

            TextField(title, text: text)
                .keyboardType(keyboardType)
                .textContentType(textContentType)
                .textInputAutocapitalization(textInputAutocapitalization)
                .autocorrectionDisabled()
                .padding(.horizontal, 14)
                .padding(.vertical, 12)
                .background(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .fill(Color.airPoseTileFill)
                )
                .overlay {
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .stroke(Color.airPoseCardStroke, lineWidth: 1)
                }
        }
    }

    private func fieldLabel(_ title: String) -> some View {
        Text(title)
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(Color.airPoseSecondaryText)
    }

    private func infoRow(title: String, value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(title)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(Color.airPoseSecondaryText)

            Spacer(minLength: 16)

            Text(value)
                .font(.subheadline)
                .foregroundStyle(Color.airPosePrimaryText)
                .multilineTextAlignment(.trailing)
        }
    }

    private var displayName: String {
        let trimmedName = viewModel.profile.name.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmedName.isEmpty ? "Your AirPose profile" : trimmedName
    }

    private var profileSummary: String {
        let sport = viewModel.profile.sport.trimmingCharacters(in: .whitespacesAndNewlines)
        let level = viewModel.profile.experienceLevel.rawValue

        if sport.isEmpty {
            return "\(level) athlete profile"
        }

        return "\(level) \(sport) athlete"
    }
}
