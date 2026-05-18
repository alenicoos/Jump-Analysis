import FirebaseFirestore
import Foundation

struct FirebaseProfileService {
    private let database = Firestore.firestore()

    func fetchProfile(for userID: String) async throws -> UserProfile? {
        let snapshot = try await database.collection("users").document(userID).getDocument()
        guard snapshot.exists else { return nil }
        let data = snapshot.data() ?? [:]

        return UserProfile(
            name: data["name"] as? String ?? "",
            age: data["age"] as? String ?? "",
            height: data["height"] as? String ?? "",
            weight: data["weight"] as? String ?? "",
            dominantLeg: DominantLeg(rawValue: data["dominantLeg"] as? String ?? "") ?? .right,
            sport: data["sport"] as? String ?? "",
            experienceLevel: ExperienceLevel(rawValue: data["experienceLevel"] as? String ?? "") ?? .intermediate
        )
    }

    func save(_ profile: UserProfile, for userID: String) async throws {
        try await database.collection("users").document(userID).setData([
            "name": profile.name,
            "age": profile.age,
            "height": profile.height,
            "weight": profile.weight,
            "dominantLeg": profile.dominantLeg.rawValue,
            "sport": profile.sport,
            "experienceLevel": profile.experienceLevel.rawValue,
        ], merge: true)
    }
}
