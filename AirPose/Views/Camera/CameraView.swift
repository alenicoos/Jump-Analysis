import SwiftUI
import UniformTypeIdentifiers

struct CameraView: View {
    @ObservedObject var viewModel: CameraViewModel
    @EnvironmentObject private var settingsStore: AppSettingsStore
    @EnvironmentObject private var tabRouter: TabRouter
    @State private var isImportingVideo = false

    var body: some View {
        AirPoseScrollCanvas { size in
            VStack(spacing: 24) {
                cameraSurface(for: size)
                bottomSection(for: size)
            }
        }
        .navigationTitle("Camera")
        .navigationBarTitleDisplayMode(.inline)
        .task {
            viewModel.onAppear()
        }
        .onDisappear {
            viewModel.onDisappear()
        }
        .onChange(of: viewModel.latestJump) { _, jump in
            guard jump != nil else { return }
            tabRouter.selectedTab = .jumps
        }
        .fileImporter(
            isPresented: $isImportingVideo,
            allowedContentTypes: [.movie, .mpeg4Movie, .quickTimeMovie],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                guard let selectedURL = urls.first else { return }
                viewModel.importVideo(from: selectedURL)
            case .failure:
                viewModel.errorMessage = "Unable to open the selected video."
            }
        }
    }

    private func cameraSurface(for size: CGSize) -> some View {
        let previewHeight = previewHeight(for: size)

        return GlassCard {
            ZStack(alignment: .bottom) {
                Group {
                    if !AppPlatform.supportsLiveCameraCapture {
                        desktopCaptureState
                    } else if viewModel.cameraManager.isConfigured {
                        CameraPreviewView(session: viewModel.cameraManager.session)
                    } else if viewModel.cameraManager.authorizationStatus == .denied || viewModel.cameraManager.authorizationStatus == .restricted {
                        permissionDeniedState
                    } else {
                        preparingState
                    }
                }
                .frame(height: previewHeight)
                .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))

                VStack(spacing: 16) {
                    HStack(alignment: .top) {
                        Label(recordingStatusTitle, systemImage: recordingStatusIcon)
                            .font(isCompactPhonePreview(for: size) ? .footnote.weight(.semibold) : .subheadline.weight(.semibold))
                            .padding(.horizontal, isCompactPhonePreview(for: size) ? 12 : 14)
                            .padding(.vertical, isCompactPhonePreview(for: size) ? 8 : 10)
                            .background(.regularMaterial, in: Capsule())
                            .foregroundStyle(recordingStatusColor)

                        Spacer()

                        HStack(spacing: 10) {
                            if !isCompactPhonePreview(for: size) {
                                Button {
                                    viewModel.switchCamera()
                                } label: {
                                    Image(systemName: "arrow.triangle.2.circlepath.camera")
                                        .font(.headline)
                                        .padding(10)
                                        .background(.regularMaterial, in: Circle())
                                }
                                .disabled(!AppPlatform.supportsLiveCameraCapture || viewModel.cameraManager.isRecording || !viewModel.cameraManager.isConfigured)
                            }

                            if settingsStore.settings.mockModeEnabled {
                                Text("Mock Mode")
                                    .font(.caption.weight(.semibold))
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 8)
                                    .background(Color.airPoseCoral.opacity(0.22), in: Capsule())
                            }
                        }
                    }

                    Spacer()

                    cameraActionBar(for: size)
                }
                .padding(isCompactPhonePreview(for: size) ? 12 : 16)
                .frame(maxWidth: .infinity, maxHeight: previewHeight, alignment: .bottom)
            }
        }
    }

    private var preparingState: some View {
        RoundedRectangle(cornerRadius: 24, style: .continuous)
            .fill(Color.white.opacity(0.06))
            .overlay(
                VStack(spacing: 12) {
                    Image(systemName: "camera.viewfinder")
                        .font(.system(size: 42))
                        .foregroundStyle(LinearGradient.airPoseBluePurple)

                    Text("Preparing camera...")
                        .font(.headline)

                    Text("If no permission prompt appears, check iPhone Settings and allow camera access for AirPose.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
                .padding(24)
            )
    }

    private var desktopCaptureState: some View {
        RoundedRectangle(cornerRadius: 24, style: .continuous)
            .fill(Color.white.opacity(0.06))
            .overlay(
                VStack(spacing: 14) {
                    Image(systemName: "desktopcomputer.and.arrow.down")
                        .font(.system(size: 42))
                        .foregroundStyle(LinearGradient.airPoseBluePurple)

                    Text("Desktop Demo Mode")
                        .font(.headline)

                    Text("On Mac, import a local jump video and send it straight to the analysis server on `127.0.0.1`.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
                .padding(24)
            )
    }

    private var permissionDeniedState: some View {
        RoundedRectangle(cornerRadius: 24, style: .continuous)
            .fill(Color.white.opacity(0.06))
            .overlay(
                VStack(spacing: 14) {
                    Image(systemName: "camera.slash.fill")
                        .font(.system(size: 42))
                        .foregroundStyle(Color.airPoseCoral)

                    Text("Camera access is off")
                        .font(.headline)

                    Text("Allow camera permission in Settings to use the iPhone camera for jump recording.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)

                    Button("Open Settings") {
                        viewModel.cameraManager.openAppSettings()
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.airPoseElectricBlue)
                }
                .padding(24)
            )
    }

    private var controlsCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 16) {
                SectionHeader("Capture Actions", subtitle: "Record from the camera preview, then manage the clip here.")

                VStack(spacing: 12) {
                    PrimaryActionButton(
                        title: "Send for Analysis",
                        systemImage: "paperplane.fill",
                        isDisabled: !viewModel.canSendForAnalysis || viewModel.analysisState == .uploading
                    ) {
                        Task {
                            await viewModel.analyzeRecordedJump()
                        }
                    }

                    if AppPlatform.isDesktopDemo {
                        PrimaryActionButton(
                            title: "Import Video",
                            systemImage: "square.and.arrow.down",
                            isDisabled: viewModel.analysisState == .uploading
                        ) {
                            isImportingVideo = true
                        }
                    }
                }

                if let recordedURL = viewModel.recordedVideoURL {
                    Text("Saved video: \(recordedURL.lastPathComponent)")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                } else if AppPlatform.supportsLiveCameraCapture {
                    Text("The live preview shows the full camera frame.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                } else if settingsStore.settings.mockModeEnabled {
                    Text("No video recorded yet. In mock mode, Send for Analysis can simulate a jump without a clip.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                if AppPlatform.supportsLiveCameraCapture && viewModel.cameraManager.isConfigured {
                    Text("Using the \(viewModel.cameraManager.currentCameraPosition == .back ? "rear" : "front") phone camera.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                } else if AppPlatform.isDesktopDemo {
                    Text("Desktop mode uses imported videos instead of live camera recording.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var statusCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 14) {
                SectionHeader("Analysis Status", subtitle: "Uploads to your local Mac server, then requests concise coaching feedback.")

                Text(statusText)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                if let successMessage = viewModel.successMessage {
                    Label(successMessage, systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                        .font(.subheadline.weight(.medium))
                }

                if let errorMessage = viewModel.errorMessage {
                    Label(errorMessage, systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                        .font(.subheadline.weight(.medium))
                }

                Text(AppPlatform.analysisTipText)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    @ViewBuilder
    private func cameraActionBar(for size: CGSize) -> some View {
        if isCompactPhonePreview(for: size) {
            compactPhoneCameraBar
        } else {
            wideCameraBar
        }
    }

    private var wideCameraBar: some View {
        HStack(alignment: .center, spacing: 18) {
            if AppPlatform.isDesktopDemo {
                iconOverlayButton(
                    systemImage: "square.and.arrow.down",
                    isDisabled: viewModel.analysisState == .uploading
                ) {
                    isImportingVideo = true
                }
            }

            Spacer(minLength: 0)

            if AppPlatform.supportsLiveCameraCapture {
                recordOverlayButton
            }

            Spacer(minLength: 0)

            if AppPlatform.isDesktopDemo {
                Circle()
                    .fill(Color.clear)
                    .frame(width: 48, height: 48)
            }
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 14)
        .background(.ultraThinMaterial, in: Capsule())
    }

    private var compactPhoneCameraBar: some View {
        HStack(alignment: .center, spacing: 28) {
            iconOverlayButton(
                systemImage: "paperplane.fill",
                isDisabled: !viewModel.canSendForAnalysis || viewModel.analysisState == .uploading
            ) {
                Task {
                    await viewModel.analyzeRecordedJump()
                }
            }

            recordOverlayButton

            iconOverlayButton(
                systemImage: "arrow.triangle.2.circlepath.camera",
                isDisabled: viewModel.cameraManager.isRecording || !viewModel.cameraManager.isConfigured
            ) {
                viewModel.switchCamera()
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 4)
    }

    private var recordOverlayButton: some View {
        Button {
            if viewModel.cameraManager.isRecording || viewModel.cameraManager.isPreparingRecording {
                viewModel.stopRecording()
            } else {
                viewModel.startRecording()
            }
        } label: {
            ZStack {
                Circle()
                    .fill(Color.white.opacity(recordButtonDisabled ? 0.35 : 0.96))
                    .frame(width: 76, height: 76)

                if viewModel.cameraManager.isRecording || viewModel.cameraManager.isPreparingRecording {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(Color.red)
                        .frame(width: 26, height: 26)
                } else {
                    Circle()
                        .fill(
                            LinearGradient(
                                colors: [Color.airPoseElectricBlue, Color.airPoseViolet],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .frame(width: 58, height: 58)
                        .overlay(
                            Circle()
                                .stroke(Color.white.opacity(0.18), lineWidth: 1)
                        )
                }
            }
            .shadow(color: .black.opacity(0.18), radius: 18, y: 8)
        }
        .disabled(recordButtonDisabled)
        .accessibilityLabel(viewModel.cameraManager.isRecording || viewModel.cameraManager.isPreparingRecording ? "Stop Recording" : "Start Recording")
    }

    private func iconOverlayButton(systemImage: String, isDisabled: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 20, weight: .semibold))
                .foregroundStyle(isDisabled ? Color.white.opacity(0.42) : .white)
                .frame(width: 48, height: 48)
                .background(Color.white.opacity(isDisabled ? 0.08 : 0.14), in: Circle())
        }
        .disabled(isDisabled)
    }

    @ViewBuilder
    private func bottomSection(for size: CGSize) -> some View {
        if AirPoseLayout.usesWideDesktopColumns(for: size) {
            HStack(alignment: .top, spacing: 24) {
                controlsCard
                    .frame(maxWidth: .infinity)
                statusCard
                    .frame(width: min(max(size.width * 0.28, 320), 420))
            }
        } else {
            VStack(spacing: 20) {
                controlsCard
                statusCard
            }
        }
    }

    private func previewHeight(for size: CGSize) -> CGFloat {
        if isCompactPhonePreview(for: size) {
            let horizontalPadding = AirPoseLayout.horizontalPadding(for: size)
            let maxWidth = min(size.width - (horizontalPadding * 2), AirPoseLayout.contentMaxWidth(for: size))
            let previewWidth = max(maxWidth - 40, 240)
            let portraitHeight = previewWidth * (16.0 / 9.0)
            return min(portraitHeight, size.height * 0.58)
        }

        if AirPoseLayout.usesWideDesktopColumns(for: size) {
            return min(max(size.height * 0.42, 440), 620)
        }
        return min(max(size.height * 0.34, 360), 480)
    }

    private var statusText: String {
        switch viewModel.analysisState {
        case .idle:
            "Waiting for a recording or mock submission."
        case .uploading:
            "Uploading video and waiting for analysis..."
        case .completed:
            "Analysis complete. Your jump has been saved to the Jumps tab."
        }
    }

    private var recordingStatusTitle: String {
        if viewModel.cameraManager.isPreparingRecording {
            return "Starting"
        }
        return viewModel.cameraManager.isRecording ? "Recording" : "Ready"
    }

    private var recordingStatusIcon: String {
        if viewModel.cameraManager.isPreparingRecording || viewModel.cameraManager.isRecording {
            return "record.circle.fill"
        }
        return "checkmark.circle.fill"
    }

    private var recordingStatusColor: Color {
        if viewModel.cameraManager.isPreparingRecording || viewModel.cameraManager.isRecording {
            return .red
        }
        return .primary
    }

    private var recordButtonDisabled: Bool {
        if viewModel.cameraManager.isRecording || viewModel.cameraManager.isPreparingRecording {
            return false
        }
        return !viewModel.cameraManager.isConfigured
    }

    private func isCompactPhonePreview(for size: CGSize) -> Bool {
        !AppPlatform.isDesktopDemo && size.height > size.width
    }
}
