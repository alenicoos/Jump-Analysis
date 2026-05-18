import SwiftUI

struct MetricTile: View {
    let title: String
    let value: String
    let systemImage: String

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Image(systemName: systemImage)
                .font(.headline)
                .foregroundStyle(LinearGradient.airPoseBluePurple)

            Text(title)
                .font(.subheadline)
                .foregroundStyle(.secondary)

            Text(value)
                .font(.title3.weight(.semibold))
                .foregroundStyle(.primary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background {
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .fill(.thinMaterial)
                .overlay {
                    RoundedRectangle(cornerRadius: 22, style: .continuous)
                        .fill(Color.airPoseTileFill.opacity(0.38))
                }
                .overlay(alignment: .topLeading) {
                    Circle()
                        .fill(Color.airPoseBlueGlow)
                        .frame(width: 120, height: 120)
                        .blur(radius: 14)
                        .offset(x: -18, y: -30)
                }
                .overlay(alignment: .bottomTrailing) {
                    Circle()
                        .fill(Color.airPoseVioletGlow)
                        .frame(width: 110, height: 110)
                        .blur(radius: 16)
                        .offset(x: 22, y: 24)
                }
                .overlay {
                    RoundedRectangle(cornerRadius: 22, style: .continuous)
                        .stroke(Color.airPoseCardStroke.opacity(0.75), lineWidth: 1)
                }
        }
    }
}
