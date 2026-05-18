import Foundation

@MainActor
final class TabRouter: ObservableObject {
    @Published var selectedTab: AppTab = .home
}
