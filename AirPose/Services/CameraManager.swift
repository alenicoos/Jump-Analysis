import AVFoundation
import Foundation
import ImageIO
import UIKit

final class CameraManager: NSObject, ObservableObject {
    @Published var authorizationStatus: AVAuthorizationStatus = AVCaptureDevice.authorizationStatus(for: .video)
    @Published var isConfigured = false
    @Published var isRecording = false
    @Published var isPreparingRecording = false
    @Published var recordedVideoURL: URL?
    @Published var recordedVideoStartedAt: Date?
    @Published var errorMessage: String?
    @Published var currentCameraPosition: AVCaptureDevice.Position = .back

    let session = AVCaptureSession()

    private let sessionQueue = DispatchQueue(label: "com.airpose.camera.session")
    private let videoDataOutputQueue = DispatchQueue(label: "com.airpose.camera.previewFrames")
    private let movieOutput = AVCaptureMovieFileOutput()
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let ciContext = CIContext()
    private var videoInput: AVCaptureDeviceInput?
    private var audioInput: AVCaptureDeviceInput?
    private var didRequestSetup = false
    private var isSettingUp = false
    private var isStartingRecording = false
    private var stopRequestedWhileStarting = false
    private var liveFrameHandler: ((Data) -> Void)?
    private var liveFrameInterval: TimeInterval = 0.2
    private var lastLiveFrameSentAt = Date.distantPast
    private let liveStreamCompressionQuality: CGFloat = 0.38
    private let liveStreamMaxDimension: CGFloat = 640

    private var supportsCameraSwitching: Bool {
        #if targetEnvironment(macCatalyst)
        return false
        #else
        return true
        #endif
    }

    func requestPermissionsAndConfigureIfNeeded() {
        if isConfigured {
            startSession()
            return
        }

        guard !didRequestSetup, !isSettingUp else { return }
        didRequestSetup = true
        isSettingUp = true

        Task {
            let granted = await requestCameraPermission()
            guard granted else {
                await MainActor.run {
                    self.isSettingUp = false
                }
                return
            }
            configureSessionIfNeeded()
            await requestMicrophonePermissionIfNeeded()
        }
    }

    func openAppSettings() {
        guard let settingsURL = URL(string: UIApplication.openSettingsURLString) else { return }
        UIApplication.shared.open(settingsURL)
    }

    func startSession() {
        sessionQueue.async { [weak self] in
            guard let self, self.isConfigured, !self.session.isRunning else { return }
            self.session.startRunning()
        }
    }

    func stopSession() {
        sessionQueue.async { [weak self] in
            guard let self, self.session.isRunning else { return }
            self.session.stopRunning()
        }
    }

    func startRecording() {
        sessionQueue.async { [weak self] in
            guard let self, self.isConfigured, !self.movieOutput.isRecording, !self.isStartingRecording else { return }
            guard self.session.isRunning else {
                self.session.startRunning()
                if !self.session.isRunning {
                    DispatchQueue.main.async {
                        self.errorMessage = "Camera session is not running yet. Please try again."
                    }
                    return
                }
                return self.startRecording()
            }

            let outputURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("airpose-\(UUID().uuidString)")
                .appendingPathExtension("mov")

            try? FileManager.default.removeItem(at: outputURL)
            self.isStartingRecording = true
            self.stopRequestedWhileStarting = false
            DispatchQueue.main.async {
                self.isPreparingRecording = true
                self.recordedVideoURL = nil
                self.recordedVideoStartedAt = nil
                self.errorMessage = nil
            }
            self.movieOutput.startRecording(to: outputURL, recordingDelegate: self)
        }
    }

    func stopRecording() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            if self.movieOutput.isRecording {
                self.movieOutput.stopRecording()
                return
            }
            if self.isStartingRecording {
                self.stopRequestedWhileStarting = true
            }
        }
    }

    func attachRecordedVideo(from selectedURL: URL) throws {
        let destinationURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("airpose-import-\(UUID().uuidString)")
            .appendingPathExtension(selectedURL.pathExtension.isEmpty ? "mov" : selectedURL.pathExtension)

        try? FileManager.default.removeItem(at: destinationURL)

        let needsScopedAccess = selectedURL.startAccessingSecurityScopedResource()
        defer {
            if needsScopedAccess {
                selectedURL.stopAccessingSecurityScopedResource()
            }
        }

        if FileManager.default.fileExists(atPath: destinationURL.path) {
            try FileManager.default.removeItem(at: destinationURL)
        }
        try FileManager.default.copyItem(at: selectedURL, to: destinationURL)
        let recordingStartedAt = Self.recordingTimestamp(for: selectedURL) ?? Self.recordingTimestamp(for: destinationURL)

        DispatchQueue.main.async {
            self.isPreparingRecording = false
            self.isRecording = false
            self.recordedVideoURL = destinationURL
            self.recordedVideoStartedAt = recordingStartedAt
            self.errorMessage = nil
        }
    }

    private static func recordingTimestamp(for url: URL) -> Date? {
        let values = try? url.resourceValues(forKeys: [.creationDateKey, .contentModificationDateKey])
        return values?.creationDate ?? values?.contentModificationDate
    }

    private func stopRecordingIfNeededAfterStart() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            guard self.stopRequestedWhileStarting else { return }
            self.stopRequestedWhileStarting = false
            guard self.movieOutput.isRecording else { return }
            self.movieOutput.stopRecording()
        }
    }

    func switchCamera() {
        sessionQueue.async { [weak self] in
            guard let self, self.supportsCameraSwitching, !self.isRecording else { return }
            self.currentCameraPosition = self.currentCameraPosition == .back ? .front : .back
            self.reconfigureVideoInput()
        }
    }

    func startPreviewFrameDelivery(frameInterval: TimeInterval = 0.2, handler: @escaping (Data) -> Void) {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.liveFrameInterval = frameInterval
            self.lastLiveFrameSentAt = .distantPast
            self.liveFrameHandler = handler
        }
    }

    func stopPreviewFrameDelivery() {
        sessionQueue.async { [weak self] in
            self?.liveFrameHandler = nil
        }
    }

    private func configureSessionIfNeeded() {
        sessionQueue.async { [weak self] in
            guard let self, !self.isConfigured else { return }

            self.session.beginConfiguration()
            self.session.sessionPreset = .high

            defer {
                self.session.commitConfiguration()
            }

            guard
                let videoDevice = self.makeVideoDevice(for: self.currentCameraPosition),
                let videoInput = try? AVCaptureDeviceInput(device: videoDevice),
                self.session.canAddInput(videoInput)
            else {
                DispatchQueue.main.async {
                    self.errorMessage = "Unable to access the camera hardware."
                }
                return
            }

            self.session.addInput(videoInput)
            self.videoInput = videoInput

            if let audioDevice = AVCaptureDevice.default(for: .audio),
               let audioInput = try? AVCaptureDeviceInput(device: audioDevice),
               self.session.canAddInput(audioInput) {
                self.session.addInput(audioInput)
                self.audioInput = audioInput
            }

            guard self.session.canAddOutput(self.movieOutput) else {
                DispatchQueue.main.async {
                    self.errorMessage = "Unable to configure video recording."
                }
                return
            }

            self.session.addOutput(self.movieOutput)
            if let videoConnection = self.movieOutput.connection(with: .video),
               videoConnection.isVideoStabilizationSupported {
                videoConnection.preferredVideoStabilizationMode = .auto
            }

            if self.session.canAddOutput(self.videoDataOutput) {
                self.videoDataOutput.alwaysDiscardsLateVideoFrames = true
                self.videoDataOutput.videoSettings = [
                    kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
                ]
                self.videoDataOutput.setSampleBufferDelegate(self, queue: self.videoDataOutputQueue)
                self.session.addOutput(self.videoDataOutput)
                if let previewConnection = self.videoDataOutput.connection(with: .video),
                   previewConnection.isVideoOrientationSupported {
                    previewConnection.videoOrientation = .portrait
                }
            }

            DispatchQueue.main.async {
                self.isConfigured = true
                self.isSettingUp = false
            }

            self.startSession()
        }
    }

    private func reconfigureVideoInput() {
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        if let videoInput {
            session.removeInput(videoInput)
            self.videoInput = nil
        }

        guard
            let newDevice = makeVideoDevice(for: currentCameraPosition),
            let newInput = try? AVCaptureDeviceInput(device: newDevice),
            session.canAddInput(newInput)
        else {
            DispatchQueue.main.async {
                self.errorMessage = "Unable to switch cameras on this device."
            }
            return
        }

        session.addInput(newInput)
        videoInput = newInput
    }

    private func requestCameraPermission() async -> Bool {
        let cameraGranted = await requestAccess(for: .video)
        await MainActor.run {
            authorizationStatus = AVCaptureDevice.authorizationStatus(for: .video)
            if !cameraGranted {
                errorMessage = "Camera access is required to record jumps."
            }
        }

        return cameraGranted
    }

    private func requestMicrophonePermissionIfNeeded() async {
        let microphoneGranted = await requestAccess(for: .audio)
        guard !microphoneGranted else { return }

        await MainActor.run {
            if self.errorMessage == nil {
                self.errorMessage = "Microphone access is recommended for video capture."
            }
        }
    }

    private func makeVideoDevice(for position: AVCaptureDevice.Position) -> AVCaptureDevice? {
        #if targetEnvironment(macCatalyst)
        if let device = AVCaptureDevice.default(for: .video) {
            return device
        }

        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .external],
            mediaType: .video,
            position: .unspecified
        )
        return discovery.devices.first
        #else
        if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position) {
            return device
        }

        let fallbackPosition: AVCaptureDevice.Position = position == .back ? .front : .back
        return AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: fallbackPosition)
        #endif
    }

    private func requestAccess(for mediaType: AVMediaType) async -> Bool {
        let status = AVCaptureDevice.authorizationStatus(for: mediaType)

        switch status {
        case .authorized:
            return true
        case .notDetermined:
            return await AVCaptureDevice.requestAccess(for: mediaType)
        case .denied, .restricted:
            await MainActor.run {
                if mediaType == .video {
                    authorizationStatus = status
                }
            }
            return false
        @unknown default:
            return false
        }
    }
}

extension CameraManager: AVCaptureFileOutputRecordingDelegate {
    func fileOutput(_ output: AVCaptureFileOutput, didStartRecordingTo fileURL: URL, from connections: [AVCaptureConnection]) {
        isStartingRecording = false
        DispatchQueue.main.async {
            self.isPreparingRecording = false
            self.isRecording = true
            self.recordedVideoStartedAt = Date()
        }
        stopRecordingIfNeededAfterStart()
    }

    func fileOutput(_ output: AVCaptureFileOutput, didFinishRecordingTo outputFileURL: URL, from connections: [AVCaptureConnection], error: Error?) {
        isStartingRecording = false
        stopRequestedWhileStarting = false
        DispatchQueue.main.async {
            self.isPreparingRecording = false
            self.isRecording = false

            if let error {
                self.errorMessage = error.localizedDescription
            } else {
                self.recordedVideoURL = outputFileURL
                if self.recordedVideoStartedAt == nil {
                    self.recordedVideoStartedAt = Date()
                }
            }
        }
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard output === videoDataOutput else { return }
        guard let handler = liveFrameHandler else { return }

        let now = Date()
        guard now.timeIntervalSince(lastLiveFrameSentAt) >= liveFrameInterval else { return }
        lastLiveFrameSentAt = now

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let image = CIImage(cvPixelBuffer: pixelBuffer).oriented(liveStreamOrientation())
        let extent = image.extent
        let maxDimension = max(extent.width, extent.height)
        let scale = min(1.0, liveStreamMaxDimension / max(maxDimension, 1.0))
        let scaledImage = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return }
        guard
            let jpegData = ciContext.jpegRepresentation(
                of: scaledImage,
                colorSpace: colorSpace,
                options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: liveStreamCompressionQuality]
            )
        else { return }
        handler(jpegData)
    }

    private func liveStreamOrientation() -> CGImagePropertyOrientation {
        #if targetEnvironment(macCatalyst)
        return .up
        #else
        let interfaceOrientation = DispatchQueue.main.sync {
            UIApplication.shared.connectedScenes
                .compactMap { ($0 as? UIWindowScene)?.interfaceOrientation }
                .first ?? .portrait
        }

        switch interfaceOrientation {
        case .portrait:
            return currentCameraPosition == .front ? .leftMirrored : .right
        case .portraitUpsideDown:
            return currentCameraPosition == .front ? .rightMirrored : .left
        case .landscapeLeft:
            return currentCameraPosition == .front ? .downMirrored : .up
        case .landscapeRight:
            return currentCameraPosition == .front ? .upMirrored : .down
        default:
            return currentCameraPosition == .front ? .leftMirrored : .right
        }
        #endif
    }
}
