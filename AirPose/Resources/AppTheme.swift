import SwiftUI
import UIKit

private extension Color {
    init(light: UIColor, dark: UIColor) {
        self.init(uiColor: UIColor { traitCollection in
            traitCollection.userInterfaceStyle == .dark ? dark : light
        })
    }
}

extension Color {
    static let airPoseBackgroundTop = Color(
        light: UIColor(red: 0.95, green: 0.96, blue: 0.99, alpha: 1),
        dark: UIColor(red: 0.07, green: 0.08, blue: 0.18, alpha: 1)
    )
    static let airPoseBackgroundBottom = Color(
        light: UIColor(red: 0.91, green: 0.92, blue: 0.98, alpha: 1),
        dark: UIColor(red: 0.18, green: 0.11, blue: 0.28, alpha: 1)
    )
    static let airPoseElectricBlue = Color(red: 0.20, green: 0.44, blue: 1.00)
    static let airPoseViolet = Color(red: 0.56, green: 0.28, blue: 0.94)
    static let airPoseCoral = Color(red: 1.00, green: 0.33, blue: 0.39)
    static let airPoseCardStroke = Color(
        light: UIColor.white.withAlphaComponent(0.55),
        dark: UIColor.white.withAlphaComponent(0.10)
    )
    static let airPoseGlassFill = Color(
        light: UIColor.white.withAlphaComponent(0.34),
        dark: UIColor(red: 0.12, green: 0.13, blue: 0.20, alpha: 0.52)
    )
    static let airPoseGlassHighlight = Color(
        light: UIColor.white.withAlphaComponent(0.52),
        dark: UIColor.white.withAlphaComponent(0.16)
    )
    static let airPoseBlueGlow = Color(
        light: UIColor(red: 0.41, green: 0.60, blue: 1.00, alpha: 0.22),
        dark: UIColor(red: 0.24, green: 0.43, blue: 1.00, alpha: 0.28)
    )
    static let airPoseVioletGlow = Color(
        light: UIColor(red: 0.67, green: 0.45, blue: 0.98, alpha: 0.18),
        dark: UIColor(red: 0.58, green: 0.34, blue: 0.94, alpha: 0.24)
    )
    static let airPoseCoralGlow = Color(
        light: UIColor(red: 1.00, green: 0.48, blue: 0.54, alpha: 0.14),
        dark: UIColor(red: 1.00, green: 0.34, blue: 0.39, alpha: 0.18)
    )
    static let airPoseTileFill = Color(
        light: UIColor.white.withAlphaComponent(0.70),
        dark: UIColor.white.withAlphaComponent(0.08)
    )
    static let airPosePrimaryText = Color(
        light: UIColor(red: 0.10, green: 0.12, blue: 0.22, alpha: 1),
        dark: UIColor.white.withAlphaComponent(0.98)
    )
    static let airPoseSecondaryText = Color(
        light: UIColor(red: 0.28, green: 0.31, blue: 0.44, alpha: 1),
        dark: UIColor.white.withAlphaComponent(0.82)
    )
    static let airPoseShadow = Color(
        light: UIColor(red: 0.22, green: 0.26, blue: 0.45, alpha: 0.12),
        dark: UIColor.black.withAlphaComponent(0.22)
    )
}

extension LinearGradient {
    static let airPoseBluePurple = LinearGradient(
        colors: [.airPoseElectricBlue, .airPoseViolet],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    static let airPoseAccent = LinearGradient(
        colors: [.airPoseCoral, .airPoseViolet],
        startPoint: .leading,
        endPoint: .trailing
    )
}
