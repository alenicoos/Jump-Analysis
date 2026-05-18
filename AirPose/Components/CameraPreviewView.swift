import AVFoundation
import SwiftUI
import UIKit

struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.videoPreviewLayer.session = session
        view.videoPreviewLayer.videoGravity = .resizeAspect
        view.backgroundColor = .black
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        uiView.videoPreviewLayer.session = session
        uiView.videoPreviewLayer.videoGravity = .resizeAspect
    }
}

final class PreviewView: UIView {
    override class var layerClass: AnyClass {
        AVCaptureVideoPreviewLayer.self
    }

    var videoPreviewLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        videoPreviewLayer.frame = bounds
        updatePreviewOrientation()
    }

    private func updatePreviewOrientation() {
        #if targetEnvironment(macCatalyst)
        return
        #else
        guard let connection = videoPreviewLayer.connection else { return }

        if #available(iOS 17.0, *) {
            guard let interfaceOrientation = window?.windowScene?.interfaceOrientation else { return }
            let rotationAngle = rotationAngle(for: interfaceOrientation)
            if connection.isVideoRotationAngleSupported(rotationAngle) {
                connection.videoRotationAngle = rotationAngle
            }
        } else if connection.isVideoOrientationSupported,
                  let interfaceOrientation = window?.windowScene?.interfaceOrientation,
                  let videoOrientation = AVCaptureVideoOrientation(interfaceOrientation: interfaceOrientation) {
            connection.videoOrientation = videoOrientation
        }
        #endif
    }

    private func rotationAngle(for interfaceOrientation: UIInterfaceOrientation) -> CGFloat {
        switch interfaceOrientation {
        case .portrait:
            return 90
        case .portraitUpsideDown:
            return 270
        case .landscapeLeft:
            return 0
        case .landscapeRight:
            return 180
        default:
            return 90
        }
    }
}

private extension AVCaptureVideoOrientation {
    init?(interfaceOrientation: UIInterfaceOrientation) {
        switch interfaceOrientation {
        case .portrait:
            self = .portrait
        case .portraitUpsideDown:
            self = .portraitUpsideDown
        case .landscapeLeft:
            self = .landscapeRight
        case .landscapeRight:
            self = .landscapeLeft
        default:
            return nil
        }
    }
}
