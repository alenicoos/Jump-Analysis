import Charts
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

                if let jumpGraph = jump.jumpGraph, !jumpGraph.points.isEmpty {
                    jumpGraphSection(jumpGraph)
                }

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

                if let imuRecording = jump.imuRecording {
                    Divider()
                        .padding(.vertical, 4)

                    Text("IMU Recording")
                        .font(.subheadline.weight(.semibold))

                    detailRow(title: "Matched File", value: imuRecording.matchedFile)
                    detailRow(title: "Matched Folder", value: imuRecording.matchedFolder)
                    detailRow(title: "Sensors", value: "\(imuRecording.deviceCount)")
                    detailRow(title: "Total Samples", value: "\(imuRecording.totalSamples)")
                    detailRow(title: "Time Offset", value: "\(imuRecording.timeOffsetSeconds.airPoseOneDecimalString) s")

                    ForEach(Array(imuRecording.deviceSummaries.enumerated()), id: \.offset) { _, device in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(device.deviceName)
                                .font(.subheadline.weight(.medium))

                            Text(imuDeviceSummary(device))
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        .padding(.top, 4)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func jumpGraphSection(_ jumpGraph: JumpAnalysisResponse.JumpGraph) -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 12) {
                Text("Jump Graph")
                    .font(.headline)

                Chart {
                    ForEach(jumpGraph.points) { point in
                        LineMark(
                            x: .value("Time", point.elapsedTimeS),
                            y: .value("Ankle Height", point.ankleHeightPx)
                        )
                        .foregroundStyle(.blue)
                        .interpolationMethod(.catmullRom)

                        LineMark(
                            x: .value("Time", point.elapsedTimeS),
                            y: .value("Body Height", point.bodyHeightPx)
                        )
                        .foregroundStyle(.green)
                        .interpolationMethod(.catmullRom)
                    }

                    RuleMark(x: .value("Initial Contact", jumpGraph.initialContactTimeS))
                        .foregroundStyle(.orange)
                        .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [5, 4]))

                    RuleMark(x: .value("Max Knee Flexion", jumpGraph.maxKneeFlexionTimeS))
                        .foregroundStyle(.red)
                        .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [5, 4]))
                }
                .frame(height: 220)
                .chartXAxisLabel("Seconds")
                .chartYAxisLabel("Height Change")

                HStack(spacing: 14) {
                    graphLegendLabel("Ankles", color: .blue)
                    graphLegendLabel("Body", color: .green)
                    graphLegendLabel("Initial Contact", color: .orange)
                    graphLegendLabel("Max Flexion", color: .red)
                }
                .font(.footnote)

                Text("The graph shows how the ankles and body move relative to landing. Positive values mean the athlete is higher than the initial-contact position.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
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

    private func graphLegendLabel(_ title: String, color: Color) -> some View {
        HStack(spacing: 6) {
            Circle()
                .fill(color)
                .frame(width: 8, height: 8)
            Text(title)
                .foregroundStyle(.secondary)
        }
    }

    private func imuDeviceSummary(_ device: JumpAnalysisResponse.IMUDeviceSummary) -> String {
        [
            "Samples: \(device.sampleCount)",
            "Duration: \(device.durationSeconds.airPoseOneDecimalString) s",
            "Peak Accel: \(optionalValue(device.peakAccelerationG, suffix: " g"))",
            "Peak Gyro: \(optionalValue(device.peakAngularVelocityDps, suffix: " dps"))",
        ].joined(separator: " • ")
    }

    private func optionalValue(_ value: Double?, suffix: String) -> String {
        guard let value else { return "n/a" }
        return "\(value.airPoseOneDecimalString)\(suffix)"
    }
}
