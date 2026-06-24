import SwiftUI

struct AuthenticationView: View {
    @ObservedObject var viewModel: AuthViewModel

    var body: some View {
        AirPoseScrollCanvas { size in
            VStack(spacing: 20) {
                headerCard
                authCard
                    .frame(maxWidth: min(AirPoseLayout.contentMaxWidth(for: size), 920))
            }
        }
        .toolbar(.hidden, for: .navigationBar)
    }

    private var headerCard: some View {
        GlassCard {
            HStack(spacing: 16) {
                Image("AirPoseLogo")
                    .resizable()
                    .scaledToFit()
                    .frame(width: 72, height: 72)
                    .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))

                VStack(alignment: .leading, spacing: 6) {
                    Text("JumpGuard")
                        .font(.largeTitle.weight(.bold))

                    Text("Sign in to sync your athlete profile and jump history across devices with Firebase.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var authCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 18) {
                Picker("Mode", selection: $viewModel.isCreatingAccount) {
                    Text("Sign In").tag(false)
                    Text("Create Account").tag(true)
                }
                .pickerStyle(.segmented)

                TextField("Email", text: $viewModel.email)
                    .keyboardType(.emailAddress)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .padding()
                    .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))

                SecureField("Password", text: $viewModel.password)
                    .padding()
                    .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))

                if viewModel.isCreatingAccount {
                    SecureField("Confirm Password", text: $viewModel.confirmPassword)
                        .padding()
                        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                }

                if let errorMessage = viewModel.errorMessage {
                    Text(errorMessage)
                        .font(.footnote)
                        .foregroundStyle(.red)
                }

                Button {
                    Task { await viewModel.submit() }
                } label: {
                    HStack {
                        if viewModel.isLoading {
                            ProgressView()
                                .tint(.white)
                        }

                        Text(viewModel.submitButtonTitle)
                            .font(.headline.weight(.semibold))
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
                    .background(LinearGradient.airPoseBluePurple, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
                    .foregroundStyle(.white)
                }
                .disabled(viewModel.isLoading)

                Button("Continue Offline") {
                    viewModel.continueAsGuest()
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
                .background(Color.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            }
        }
    }
}
