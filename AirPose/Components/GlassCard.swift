import SwiftUI

struct GlassCard<Content: View>: View {
    let content: Content

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    var body: some View {
        content
            .padding(20)
            .background {
                RoundedRectangle(cornerRadius: 28, style: .continuous)
                    .fill(.ultraThinMaterial)
                    .overlay {
                        RoundedRectangle(cornerRadius: 28, style: .continuous)
                            .fill(Color.airPoseGlassFill)
                    }
                    .overlay(alignment: .topLeading) {
                        Circle()
                            .fill(Color.airPoseBlueGlow)
                            .frame(width: 220, height: 220)
                            .blur(radius: 18)
                            .offset(x: -44, y: -54)
                    }
                    .overlay(alignment: .topTrailing) {
                        Circle()
                            .fill(Color.airPoseVioletGlow)
                            .frame(width: 200, height: 200)
                            .blur(radius: 20)
                            .offset(x: 52, y: -70)
                    }
                    .overlay(alignment: .bottomLeading) {
                        Circle()
                            .fill(Color.airPoseCoralGlow)
                            .frame(width: 180, height: 180)
                            .blur(radius: 24)
                            .offset(x: -48, y: 72)
                    }
            }
            .overlay {
                RoundedRectangle(cornerRadius: 28, style: .continuous)
                    .stroke(Color.airPoseCardStroke, lineWidth: 1)
            }
            .overlay(alignment: .top) {
                RoundedRectangle(cornerRadius: 28, style: .continuous)
                    .stroke(Color.airPoseGlassHighlight, lineWidth: 0.8)
                    .blur(radius: 0.6)
                    .mask(
                        LinearGradient(
                            colors: [.white, .white.opacity(0)],
                            startPoint: .top,
                            endPoint: .center
                        )
                    )
            }
            .shadow(color: .airPoseShadow, radius: 18, x: 0, y: 12)
    }
}
