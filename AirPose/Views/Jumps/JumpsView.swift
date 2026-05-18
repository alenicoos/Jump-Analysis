import SwiftUI

struct JumpsView: View {
    @ObservedObject var viewModel: JumpsViewModel
    @State private var jumpPendingDeletion: Jump?

    var body: some View {
        AirPoseScrollCanvas { _ in
            VStack(alignment: .leading, spacing: 18) {
                sortBar

                if viewModel.sortedJumps.isEmpty {
                    GlassCard {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("No jumps yet")
                                .font(.headline)

                            Text("Once you analyze a jump, it will appear here with full metrics and coaching feedback.")
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                } else {
                    ForEach(viewModel.sortedJumps) { jump in
                        NavigationLink {
                            JumpDetailView(jump: jump, onDelete: {
                                viewModel.delete(jump)
                            })
                        } label: {
                            JumpCardView(jump: jump)
                        }
                        .buttonStyle(.plain)
                        .contextMenu {
                            Button(role: .destructive) {
                                jumpPendingDeletion = jump
                            } label: {
                                Label("Delete Jump", systemImage: "trash")
                            }
                        }
                    }
                }
            }
        }
        .navigationTitle("Jumps")
        .navigationBarTitleDisplayMode(.inline)
        .confirmationDialog(
            "Delete this jump?",
            isPresented: Binding(
                get: { jumpPendingDeletion != nil },
                set: { if !$0 { jumpPendingDeletion = nil } }
            ),
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) {
                if let jumpPendingDeletion {
                    viewModel.delete(jumpPendingDeletion)
                    self.jumpPendingDeletion = nil
                }
            }
            Button("Cancel", role: .cancel) {
                jumpPendingDeletion = nil
            }
        } message: {
            Text("This removes the jump and its saved feedback from local storage, and from Firebase if sync is active.")
        }
    }

    private var sortBar: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 14) {
                SectionHeader("Sort & Filter", subtitle: "Placeholder controls ready for more advanced filtering later.")

                Picker("Sort By", selection: $viewModel.sortOption) {
                    ForEach(JumpsViewModel.SortOption.allCases) { option in
                        Text(option.rawValue).tag(option)
                    }
                }
                .pickerStyle(.segmented)
            }
        }
    }
}
