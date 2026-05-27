import SwiftUI

struct SettingsView: View {
    @ObservedObject var viewModel: SettingsViewModel
    @EnvironmentObject private var jumpNarrationCoordinator: JumpNarrationCoordinator

    var body: some View {
        AirPoseScrollCanvas { _ in
            VStack(spacing: 20) {
                GlassCard {
                    VStack(alignment: .leading, spacing: 18) {
                        SectionHeader("Experience", subtitle: "Tune units, theme, backend endpoint, and mock flow.")

                        VStack(alignment: .leading, spacing: 10) {
                            Text("Units")
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(.secondary)

                            Picker("Units", selection: $viewModel.settings.units) {
                                ForEach(MeasurementUnits.allCases) { unit in
                                    Text(unit.rawValue).tag(unit)
                                }
                            }
                            .pickerStyle(.segmented)
                        }

                        Divider()

                        VStack(alignment: .leading, spacing: 10) {
                            Text("Theme")
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(.secondary)

                            Picker("Theme", selection: $viewModel.settings.themePreference) {
                                ForEach(ThemePreference.allCases) { theme in
                                    Text(theme.rawValue).tag(theme)
                                }
                            }
                            .pickerStyle(.segmented)
                        }

                        Divider()

                        Toggle(isOn: $viewModel.settings.mockModeEnabled) {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Enable Mock Mode")
                                    .font(.headline)
                                Text("Use simulated analysis when you do not want to hit the backend.")
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .tint(.airPoseElectricBlue)

                        Divider()

                        VStack(alignment: .leading, spacing: 10) {
                            Text("Backend URL")
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(.secondary)

                            TextField(AppPlatform.backendPlaceholder, text: $viewModel.settings.backendServerURL)
                                .textInputAutocapitalization(.never)
                                .autocorrectionDisabled()
                                .keyboardType(.URL)

                            Text(AppPlatform.backendHelpText)
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }

                        Divider()

                        VStack(alignment: .leading, spacing: 10) {
                            Text("OpenAI Feedback")
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(.secondary)

                            TextField(AppSettings.defaultLLMAPIURL, text: $viewModel.settings.llmAPIURL)
                                .textInputAutocapitalization(.never)
                                .autocorrectionDisabled()
                                .keyboardType(.URL)

                            SecureField("OpenAI API key", text: $viewModel.settings.llmAPIKey)
                                .textInputAutocapitalization(.never)
                                .autocorrectionDisabled()

                            Text("AirPose sends the jump metrics, not the video, to OpenAI's Responses API and stores the short coaching explanation it gets back.")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }

                        Divider()

                        VStack(alignment: .leading, spacing: 12) {
                            Text("Existing Jumps")
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(.secondary)

                            Text("Use this after configuring OpenAI to regenerate posture feedback for older saved jumps.")
                                .font(.footnote)
                                .foregroundStyle(.secondary)

                            Button {
                                jumpNarrationCoordinator.reprocessExistingJumps()
                            } label: {
                                Label("Reprocess Existing Jumps", systemImage: "arrow.triangle.2.circlepath")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(.airPoseElectricBlue)
                        }
                    }
                }
            }
        }
        .navigationTitle("Settings")
        .navigationBarTitleDisplayMode(.inline)
        .onChange(of: viewModel.settings) { _, _ in
            viewModel.persist()
        }
    }
}
