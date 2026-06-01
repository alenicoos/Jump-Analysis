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
    @Published var selectedAthleteFilter: String = "All Athletes"
    @Published private(set) var athleteFilters: [String] = ["All Athletes"]
    @Published private(set) var sortedJumps: [Jump] = []

    private let jumpStore: JumpStore
    private let cloudSyncCoordinator: CloudSyncCoordinator
    private var cancellables = Set<AnyCancellable>()

    static let allAthletesFilter = "All Athletes"

    init(jumpStore: JumpStore, cloudSyncCoordinator: CloudSyncCoordinator) {
        self.jumpStore = jumpStore
        self.cloudSyncCoordinator = cloudSyncCoordinator

        jumpStore.$jumps
            .map { jumps in
                let names = Set(jumps.map { $0.athleteProfile?.displayName ?? "Unnamed Athlete" })
                return [Self.allAthletesFilter] + names.sorted()
            }
            .assign(to: &$athleteFilters)

        Publishers.CombineLatest3($sortOption, $selectedAthleteFilter, jumpStore.$jumps)
            .map { option, athleteFilter, jumps in
                let filteredJumps = athleteFilter == Self.allAthletesFilter
                    ? jumps
                    : jumps.filter { ($0.athleteProfile?.displayName ?? "Unnamed Athlete") == athleteFilter }
                switch option {
                case .date:
                    return filteredJumps.sorted { $0.date > $1.date }
                case .anomalyScore:
                    return filteredJumps.sorted { $0.anomalyScore > $1.anomalyScore }
                case .protocolStatus:
                    return filteredJumps.sorted { lhs, rhs in
                        if lhs.protocolPassed == rhs.protocolPassed {
                            return lhs.date > rhs.date
                        }
                        return lhs.protocolPassed && !rhs.protocolPassed
                    }
                }
            }
            .assign(to: &$sortedJumps)

        $selectedAthleteFilter
            .combineLatest($athleteFilters)
            .sink { [weak self] selectedFilter, filters in
                guard let self else { return }
                guard filters.contains(selectedFilter) else {
                    self.selectedAthleteFilter = Self.allAthletesFilter
                    return
                }
            }
            .store(in: &cancellables)
    }

    func delete(_ jump: Jump) {
        jumpStore.delete(jump)
        cloudSyncCoordinator.deleteJump(jump)
    }
}
