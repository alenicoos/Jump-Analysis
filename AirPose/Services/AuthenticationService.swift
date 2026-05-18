import FirebaseAuth
import Foundation

@MainActor
final class AuthenticationService: ObservableObject {
    @Published private(set) var userSession: FirebaseUserSession?
    @Published private(set) var isLoading = false
    @Published var errorMessage: String?
    @Published var isGuestModeEnabled = false

    private var authStateListenerHandle: AuthStateDidChangeListenerHandle?

    init() {
        authStateListenerHandle = Auth.auth().addStateDidChangeListener { [weak self] _, user in
            Task { @MainActor in
                self?.userSession = user.map {
                    FirebaseUserSession(uid: $0.uid, email: $0.email)
                }
            }
        }
    }

    deinit {
        if let authStateListenerHandle {
            Auth.auth().removeStateDidChangeListener(authStateListenerHandle)
        }
    }

    var canAccessApp: Bool {
        isGuestModeEnabled || userSession != nil
    }

    func createAccount(email: String, password: String) async {
        await runAuthTask {
            _ = try await Auth.auth().createUser(withEmail: email, password: password)
            self.isGuestModeEnabled = false
        }
    }

    func signIn(email: String, password: String) async {
        await runAuthTask {
            _ = try await Auth.auth().signIn(withEmail: email, password: password)
            self.isGuestModeEnabled = false
        }
    }

    func signOut() {
        do {
            try Auth.auth().signOut()
            isGuestModeEnabled = false
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func continueAsGuest() {
        errorMessage = nil
        isGuestModeEnabled = true
    }

    func leaveGuestMode() {
        isGuestModeEnabled = false
    }

    private func runAuthTask(_ operation: @escaping @MainActor () async throws -> Void) async {
        isLoading = true
        errorMessage = nil

        do {
            try await operation()
        } catch {
            errorMessage = error.localizedDescription
        }

        isLoading = false
    }
}
