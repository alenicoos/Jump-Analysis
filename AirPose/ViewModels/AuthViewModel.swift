import Foundation
import Combine

@MainActor
final class AuthViewModel: ObservableObject {
    @Published var email = ""
    @Published var password = ""
    @Published var confirmPassword = ""
    @Published var isCreatingAccount = false

    private let authenticationService: AuthenticationService
    private var cancellables = Set<AnyCancellable>()

    init(authenticationService: AuthenticationService) {
        self.authenticationService = authenticationService

        authenticationService.objectWillChange
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.objectWillChange.send()
            }
            .store(in: &cancellables)
    }

    var isLoading: Bool {
        authenticationService.isLoading
    }

    var errorMessage: String? {
        authenticationService.errorMessage
    }

    var submitButtonTitle: String {
        isCreatingAccount ? "Create Account" : "Sign In"
    }

    func submit() async {
        let normalizedEmail = email.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !normalizedEmail.isEmpty, !password.isEmpty else {
            authenticationService.errorMessage = "Enter your email and password to continue."
            return
        }

        if isCreatingAccount {
            guard password == confirmPassword else {
                authenticationService.errorMessage = "Passwords do not match."
                return
            }

            await authenticationService.createAccount(email: normalizedEmail, password: password)
        } else {
            await authenticationService.signIn(email: normalizedEmail, password: password)
        }
    }

    func continueAsGuest() {
        authenticationService.continueAsGuest()
    }

    func leaveGuestMode() {
        authenticationService.leaveGuestMode()
    }
}
