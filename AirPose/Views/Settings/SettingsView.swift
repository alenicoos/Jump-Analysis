import SwiftUI

struct SettingsView: View {
    @ObservedObject var viewModel: SettingsViewModel

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
