import Combine
import Foundation

@MainActor
final class DashboardViewModel: ObservableObject {
    @Published private(set) var jumps: [Jump] = []

    private var cancellables = Set<AnyCancellable>()

    init(jumpStore: JumpStore) {
        jumpStore.$jumps
            .assign(to: &$jumps)
    }

    var latestJump: Jump? {
        jumps.sorted { $0.date > $1.date }.first
    }

    var totalJumps: Int {
        jumps.count
    }

    var protocolPassCount: Int {
        jumps.filter(\.protocolPassed).count
    }

    var normalJumpCount: Int {
        jumps.filter(\.isNormalPrediction).count
    }

    var averageAnomalyScore: Double {
        guard !jumps.isEmpty else { return 0 }
        return jumps.map(\.anomalyScore).reduce(0, +) / Double(jumps.count)
    }
}
