import SwiftUI

struct JumpCardView: View {
    let jump: Jump
    let showsFeedback: Bool

    init(jump: Jump, showsFeedback: Bool = false) {
        self.jump = jump
        self.showsFeedback = showsFeedback
    }

    var body: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 16) {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(jump.date.airPoseMediumString)
                            .font(.headline)

                        if let athleteProfile = jump.athleteProfile {
                            Text(athleteProfile.displayName)
                                .font(.subheadline.weight(.medium))
                                .foregroundStyle(Color.airPoseSecondaryText)
                        }

                        Text(jump.displayPrediction)
                            .font(.title3.weight(.semibold))
                            .foregroundStyle(LinearGradient.airPoseBluePurple)
                    }

                    Spacer()

                    Image(systemName: "figure.highintensity.intervaltraining")
                        .font(.title2)
                        .foregroundStyle(LinearGradient.airPoseAccent)
                }

                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                    MetricTile(title: "Protocol", value: jump.protocolPassed.airPosePassFailString, systemImage: "checkmark.shield")
                    MetricTile(title: "Anomaly", value: jump.anomalyScore.airPoseOneDecimalString, systemImage: "exclamationmark.magnifyingglass")
                    MetricTile(title: "Outliers", value: "\(jump.outlierFeatureCount)", systemImage: "list.number")
                    MetricTile(title: "IC Knee", value: jump.averageInitialContactKneeAngleDeg.airPoseDegreeString, systemImage: "angle")
                    MetricTile(title: "KF Knee", value: jump.averageMaxKneeFlexionKneeAngleDeg.airPoseDegreeString, systemImage: "figure.strengthtraining.traditional")
                    MetricTile(title: "Landing Asym.", value: jump.landingAsymmetryRatio.airPoseRatioString, systemImage: "arrow.left.and.right")
                }

                if !jump.protocolPassed, !jump.failedProtocolChecks.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Protocol Checks")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(Color.airPosePrimaryText)

                        ForEach(jump.failedProtocolChecks) { check in
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Failed: \(check.title)")
                                    .font(.subheadline.weight(.medium))
                                    .foregroundStyle(.red)
                                Text(check.detailText)
                                    .font(.footnote)
                                    .foregroundStyle(Color.airPoseSecondaryText)
                            }
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                if showsFeedback {
                    Divider()

                    VStack(alignment: .leading, spacing: 8) {
                        Text("LLM Feedback")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(Color.airPosePrimaryText)

                        Text(summaryText)
                            .font(.subheadline)
                            .foregroundStyle(Color.airPoseSecondaryText)
                            .fixedSize(horizontal: false, vertical: true)

                        Text(footnoteText)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
    }

    private var summaryText: String {
        let trimmed = jump.analysisSummary.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return "No narrated feedback has been generated for this jump yet."
        }

        return trimmed
    }

    private var footnoteText: String {
        jump.llmNarratedSummary
            ? "Generated from the LLM prompt using the 183-athlete motion-capture reference dataset context."
            : "Showing the saved summary for this jump. Configure OpenAI and reprocess older jumps if you want narrated feedback here."
    }
}
