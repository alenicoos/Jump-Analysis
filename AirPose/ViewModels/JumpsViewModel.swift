import Combine
import Foundation

@MainActor
final class JumpsViewModel: ObservableObject {
    enum SortOption: String, CaseIterable, Identifiable {
        case date = "Date"
        case anomalyScore = "Anomaly"
        case protocolStatus = "Protocol"

        var id: String { rawValue }
    }

    @Published var sortOption: SortOption = .date
    @Published private(set) var sortedJumps: [Jump] = []

    private let jumpStore: JumpStore
    private let cloudSyncCoordinator: CloudSyncCoordinator
    private var cancellables = Set<AnyCancellable>()

    init(jumpStore: JumpStore, cloudSyncCoordinator: CloudSyncCoordinator) {
        self.jumpStore = jumpStore
        self.cloudSyncCoordinator = cloudSyncCoordinator

        Publishers.CombineLatest($sortOption, jumpStore.$jumps)
            .map { option, jumps in
                switch option {
                case .date:
                    jumps.sorted { $0.date > $1.date }
                case .anomalyScore:
                    jumps.sorted { $0.anomalyScore > $1.anomalyScore }
                case .protocolStatus:
                    jumps.sorted { lhs, rhs in
                        if lhs.protocolPassed == rhs.protocolPassed {
                            return lhs.date > rhs.date
                        }
                        return lhs.protocolPassed && !rhs.protocolPassed
                    }
                }
            }
            .assign(to: &$sortedJumps)
    }

    func delete(_ jump: Jump) {
        jumpStore.delete(jump)
        cloudSyncCoordinator.deleteJump(jump)
    }
}
