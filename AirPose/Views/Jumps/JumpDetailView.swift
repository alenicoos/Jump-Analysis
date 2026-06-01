import SwiftUI

struct JumpDetailView: View {
    let jump: Jump
    let onDelete: () -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var isShowingDeleteConfirmation = false

    var body: some View {
        AirPoseScrollCanvas { size in
            VStack(spacing: 20) {
                JumpCardView(jump: jump, showsFeedback: true)
                    .frame(maxWidth: min(AirPoseLayout.contentMaxWidth(for: size), 920))

                detailSection

                if let videoURL = jump.videoURL {
                    videoSection(videoURL)
                }
            }
        }
        .navigationTitle("Jump Detail")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button(role: .destructive) {
                    isShowingDeleteConfirmation = true
                } label: {
                    Image(systemName: "trash")
                }
            }
        }
        .confirmationDialog("Delete this jump?", isPresented: $isShowingDeleteConfirmation, titleVisibility: .visible) {
            Button("Delete", role: .destructive) {
                onDelete()
                dismiss()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This removes the jump and its saved analysis from local storage.")
        }
    }

    private var detailSection: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                Text("Analysis Details")
                    .font(.headline)

                if let athleteProfile = jump.athleteProfile {
                    detailRow(title: "Athlete", value: athleteProfile.displayName)
                    detailRow(
                        title: "Profile Source",
                        value: athleteProfile.source == .accountProfile ? "Account Profile" : "Other Athlete"
                    )
                    detailRow(title: "Sport / Level", value: athleteProfile.summaryText)
                }
                detailRow(title: "Worst Feature", value: jump.worstFeature.replacingOccurrences(of: "_", with: " "))
                detailRow(title: "Worst Feature Z", value: jump.worstFeatureZ.airPoseOneDecimalString)
                detailRow(title: "Pose Frames", value: "\(jump.validPoseFrames)")
                detailRow(title: "Video FPS", value: jump.videoFPS.airPoseOneDecimalString)
                detailRow(title: "Estimated Shoulder Width", value: jump.estimatedShoulderWidthCm.airPoseCentimeterString)
                detailRow(title: "Knee Asymmetry", value: jump.kneeAsymmetryRatio.airPoseRatioString)

                if !jump.protocolChecks.isEmpty {
                    Divider()
                        .padding(.vertical, 4)

                    Text("Protocol Checks")
                        .font(.subheadline.weight(.semibold))

                    ForEach(jump.protocolChecks) { check in
                        HStack(alignment: .top, spacing: 12) {
                            Image(systemName: check.passed ? "checkmark.circle.fill" : "xmark.circle.fill")
                                .foregroundStyle(check.passed ? .green : .red)

                            VStack(alignment: .leading, spacing: 2) {
                                Text(check.title)
                                    .font(.subheadline.weight(.medium))
                                Text(check.detailText)
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                            }

                            Spacer()

                            Text(check.passed.airPosePassFailString)
                                .font(.footnote.weight(.semibold))
                                .foregroundStyle(check.passed ? .green : .red)
                        }
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func videoSection(_ videoURL: URL) -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                Text("Recorded Video")
                    .font(.headline)

                Text(videoURL.absoluteString)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func detailRow(title: String, value: String) -> some View {
        HStack {
            Text(title)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .multilineTextAlignment(.trailing)
        }
        .font(.subheadline)
    }
}
