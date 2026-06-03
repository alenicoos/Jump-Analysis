import SwiftUI
import UIKit
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
                SectionHeader("Live Jump Session", subtitle: "Use the phone as a live camera source while the Python server handles setup, capture, and analysis.")

                athleteAssignmentSection

                Divider()

                VStack(spacing: 12) {
                    if AppPlatform.supportsLiveCameraCapture {
                        PrimaryActionButton(
                            title: viewModel.liveGuidanceState == .active || viewModel.liveGuidanceState == .connecting
                                ? "Stop Live Guidance"
                                : "Start Live Guidance",
                            systemImage: viewModel.liveGuidanceState == .active || viewModel.liveGuidanceState == .connecting
                                ? "stop.circle.fill"
                                : "waveform.and.mic"
                        ) {
                            if viewModel.liveGuidanceState == .active || viewModel.liveGuidanceState == .connecting {
                                viewModel.stopLiveGuidance()
                            } else {
                                viewModel.startLiveGuidance()
                            }
                        }
                    }

                    if viewModel.canUseRecordedFallback {
                        PrimaryActionButton(
                            title: "Analyze Recorded Fallback",
                            systemImage: "paperplane.fill",
                            isDisabled: !viewModel.canSendForAnalysis || viewModel.analysisState == .uploading
                        ) {
                            Task {
                                await viewModel.analyzeRecordedJump()
                            }
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
                    Text("The live preview is streamed to the server, which decides when setup is valid and when the jump starts.")
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

    private var athleteAssignmentSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Athlete for This Jump")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.secondary)

            Picker("Athlete Source", selection: $viewModel.athleteSelection) {
                ForEach(CameraViewModel.AthleteSelection.allCases) { selection in
                    Text(selection.rawValue).tag(selection)
                }
            }
            .pickerStyle(.segmented)

            if viewModel.athleteSelection == .accountProfile {
                VStack(alignment: .leading, spacing: 6) {
                    Text(viewModel.accountProfile.displayName)
                        .font(.headline)

                    Text(viewModel.accountProfile.summaryText)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 14)
                .padding(.vertical, 12)
                .background(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .fill(Color.airPoseTileFill)
                )
            } else {
                VStack(alignment: .leading, spacing: 12) {
                    athleteField("Name", text: $viewModel.guestAthleteProfile.name, textInputAutocapitalization: .words)
                    athleteField("Age", text: $viewModel.guestAthleteProfile.age, keyboardType: .numberPad, textInputAutocapitalization: .never)
                    athleteField(settingsStore.settings.units.heightLabel, text: $viewModel.guestAthleteProfile.height, keyboardType: .decimalPad, textInputAutocapitalization: .never)
                    athleteField(settingsStore.settings.units.weightLabel, text: $viewModel.guestAthleteProfile.weight, keyboardType: .decimalPad, textInputAutocapitalization: .never)

                    VStack(alignment: .leading, spacing: 8) {
                        athleteFieldLabel("Dominant Leg")

                        Picker("Dominant Leg", selection: $viewModel.guestAthleteProfile.dominantLeg) {
                            ForEach(DominantLeg.allCases) { leg in
                                Text(leg.rawValue).tag(leg)
                            }
                        }
                        .pickerStyle(.segmented)
                    }

                    athleteField("Sport", text: $viewModel.guestAthleteProfile.sport, textInputAutocapitalization: .words)

                    VStack(alignment: .leading, spacing: 8) {
                        athleteFieldLabel("Experience Level")

                        Picker("Experience Level", selection: $viewModel.guestAthleteProfile.experienceLevel) {
                            ForEach(ExperienceLevel.allCases) { level in
                                Text(level.rawValue).tag(level)
                            }
                        }
                        .pickerStyle(.menu)
                    }
                }
            }
        }
    }

    private var statusCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 14) {
                SectionHeader("Analysis Status", subtitle: "The Python server drives setup guidance, jump capture, and final analysis during the live session.")

                Text(statusText)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                if let liveGuidanceMessage = viewModel.liveGuidanceMessage {
                    Label(liveGuidanceMessage, systemImage: "waveform.and.mic")
                        .foregroundStyle(viewModel.liveGuidanceState == .failed ? .orange : .blue)
                        .font(.subheadline.weight(.medium))
                }

                if let liveAnalysisSummary = viewModel.liveAnalysisSummary {
                    Text(liveAnalysisSummary)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

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
                systemImage: viewModel.liveGuidanceState == .active || viewModel.liveGuidanceState == .connecting
                    ? "stop.fill"
                    : "waveform.and.mic",
                isDisabled: !viewModel.canStartLiveGuidance && viewModel.liveGuidanceState == .idle
            ) {
                if viewModel.liveGuidanceState == .active || viewModel.liveGuidanceState == .connecting {
                    viewModel.stopLiveGuidance()
                } else {
                    viewModel.startLiveGuidance()
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
        switch viewModel.liveGuidanceState {
        case .connecting:
            return "Connecting to the live guidance server..."
        case .active:
            return "Streaming live camera frames while the server guides setup and captures the jump."
        case .completed:
            return "Live analysis complete. The jump has been saved to the Jumps tab."
        case .failed:
            return "Live guidance stopped because the streaming session failed."
        case .idle:
            break
        }

        switch viewModel.analysisState {
        case .idle:
            return "Ready to start a live guided jump session."
        case .uploading:
            return "Analyzing the recorded fallback clip..."
        case .completed:
            return "Recorded fallback analysis complete. Your jump has been saved to the Jumps tab."
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

    private func athleteField(
        _ title: String,
        text: Binding<String>,
        keyboardType: UIKeyboardType = .default,
        textInputAutocapitalization: TextInputAutocapitalization = .sentences
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            athleteFieldLabel(title)

            TextField(title, text: text)
                .keyboardType(keyboardType)
                .textInputAutocapitalization(textInputAutocapitalization)
                .autocorrectionDisabled()
                .padding(.horizontal, 14)
                .padding(.vertical, 12)
                .background(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .fill(Color.airPoseTileFill)
                )
                .overlay {
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .stroke(Color.airPoseCardStroke, lineWidth: 1)
                }
        }
    }

    private func athleteFieldLabel(_ title: String) -> some View {
        Text(title)
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(Color.airPoseSecondaryText)
    }
}
