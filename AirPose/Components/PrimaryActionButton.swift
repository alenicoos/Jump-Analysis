import SwiftUI

struct PrimaryActionButton: View {
    let title: String
    let systemImage: String
    let isDisabled: Bool
    let action: () -> Void

    init(title: String, systemImage: String, isDisabled: Bool = false, action: @escaping () -> Void) {
        self.title = title
        self.systemImage = systemImage
        self.isDisabled = isDisabled
        self.action = action
    }

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Image(systemName: systemImage)
                Text(title)
                    .fontWeight(.semibold)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 16)
            .background(
                LinearGradient(
                    colors: [Color.airPoseElectricBlue, Color.airPoseViolet],
                    startPoint: .leading,
                    endPoint: .trailing
                ),
                in: RoundedRectangle(cornerRadius: 18, style: .continuous)
            )
            .foregroundStyle(.white)
            .opacity(isDisabled ? 0.45 : 1)
        }
        .disabled(isDisabled)
    }
}
