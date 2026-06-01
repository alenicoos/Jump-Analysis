import SwiftUI

struct DashboardView: View {
    @ObservedObject var viewModel: DashboardViewModel
    @EnvironmentObject private var tabRouter: TabRouter

    var body: some View {
        AirPoseScrollCanvas { size in
            VStack(alignment: .leading, spacing: 22) {
                header
                latestScoreCard
                statsGrid(for: size)
                ctaSection
            }
        }
        .navigationTitle("Dashboard")
        .navigationBarTitleDisplayMode(.inline)
    }

    private var header: some View {
        GlassCard {
            HStack(spacing: 16) {
                Image("AirPoseLogo")
                    .resizable()
                    .scaledToFit()
                    .frame(width: 76, height: 76)
                    .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))

                VStack(alignment: .leading, spacing: 6) {
                    Text("AirPose")
                        .font(.largeTitle.weight(.bold))
                        .foregroundStyle(Color.airPosePrimaryText)

                    Text("Capture frontal drop jumps, validate protocol quality, and compare each trial against the reference dataset.")
                        .font(.subheadline)
                        .foregroundStyle(Color.airPoseSecondaryText)
                }
            }
        }
    }

    private var latestScoreCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 14) {
                Text("Latest Analysis")
                    .font(.headline)
                    .foregroundStyle(Color.airPoseSecondaryText)

                Text(viewModel.latestJump?.displayPrediction ?? "--")
                    .font(.system(size: 42, weight: .bold, design: .rounded))
                    .foregroundStyle(LinearGradient.airPoseBluePurple)

                Text(viewModel.latestJump?.analysisSummary ?? "Record your first jump to unlock protocol validation and anomaly analysis.")
                    .font(.subheadline)
                    .foregroundStyle(Color.airPoseSecondaryText)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func statsGrid(for size: CGSize) -> some View {
        let columns = AirPoseLayout.usesWideDesktopColumns(for: size)
            ? Array(repeating: GridItem(.flexible(), spacing: 14), count: 4)
            : Array(repeating: GridItem(.flexible(), spacing: 14), count: 2)

        return LazyVGrid(columns: columns, spacing: 14) {
            MetricTile(title: "Total Jumps", value: "\(viewModel.totalJumps)", systemImage: "number")
            MetricTile(title: "Protocol Passes", value: "\(viewModel.protocolPassCount)", systemImage: "checkmark.shield")
            MetricTile(title: "Normal Analyses", value: "\(viewModel.normalJumpCount)", systemImage: "waveform.path.ecg")
            MetricTile(title: "Avg Anomaly", value: viewModel.averageAnomalyScore.airPoseOneDecimalString, systemImage: "chart.bar.fill")
        }
    }

    private var ctaSection: some View {
        PrimaryActionButton(title: "Record New Jump", systemImage: "video.badge.plus") {
            tabRouter.selectedTab = .camera
        }
    }
}
