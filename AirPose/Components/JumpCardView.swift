import SwiftUI

struct JumpCardView: View {
    let jump: Jump

    var body: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 16) {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(jump.date.airPoseMediumString)
                            .font(.headline)

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

                Text(jump.analysisSummary)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}
