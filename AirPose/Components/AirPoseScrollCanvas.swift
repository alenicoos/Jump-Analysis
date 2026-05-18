import SwiftUI

enum AirPoseLayout {
    static func contentMaxWidth(for size: CGSize) -> CGFloat {
        guard AppPlatform.isDesktopDemo else {
            return min(size.width - 32, 980)
        }

        switch size.width {
        case 0..<1100:
            return min(size.width - 32, 980)
        case 1100..<1450:
            return 1120
        default:
            return 1360
        }
    }

    static func horizontalPadding(for size: CGSize) -> CGFloat {
        AppPlatform.isDesktopDemo ? 24 : 16
    }

    static func usesWideDesktopColumns(for size: CGSize) -> Bool {
        AppPlatform.isDesktopDemo && size.width >= 1280
    }
}

struct AirPoseScrollCanvas<Content: View>: View {
    let content: (CGSize) -> Content

    init(@ViewBuilder content: @escaping (CGSize) -> Content) {
        self.content = content
    }

    var body: some View {
        GeometryReader { proxy in
            ZStack {
                BrandBackground()

                ScrollView {
                    content(proxy.size)
                        .frame(maxWidth: AirPoseLayout.contentMaxWidth(for: proxy.size))
                        .padding(.horizontal, AirPoseLayout.horizontalPadding(for: proxy.size))
                        .padding(.top, 20)
                        .padding(.bottom, 36)
                        .frame(maxWidth: .infinity)
                }
            }
        }
    }
}
