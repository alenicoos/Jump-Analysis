import SwiftUI

struct JumpDetailView: View {
    let jump: Jump
    let onDelete: () -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var isShowingDeleteConfirmation = false

    var body: some View {
        AirPoseScrollCanvas { size in
            VStack(spacing: 20) {
                JumpCardView(jump: jump)
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

                detailRow(title: "Worst Feature", value: jump.worstFeature.replacingOccurrences(of: "_", with: " "))
                detailRow(title: "Worst Feature Z", value: jump.worstFeatureZ.airPoseOneDecimalString)
                detailRow(title: "Pose Frames", value: "\(jump.validPoseFrames)")
                detailRow(title: "Video FPS", value: jump.videoFPS.airPoseOneDecimalString)
                detailRow(title: "Estimated Shoulder Width", value: jump.estimatedShoulderWidthCm.airPoseCentimeterString)
                detailRow(title: "Knee Asymmetry", value: jump.kneeAsymmetryRatio.airPoseRatioString)
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
