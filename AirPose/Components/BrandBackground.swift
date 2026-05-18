import SwiftUI

struct BrandBackground: View {
    var body: some View {
        LinearGradient(
            colors: [.airPoseBackgroundTop, .airPoseBackgroundBottom],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
        .ignoresSafeArea()
    }
}
