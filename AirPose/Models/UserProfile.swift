import Foundation

struct UserProfile: Codable, Equatable {
    var name: String = ""
    var age: String = ""
    var height: String = ""
    var weight: String = ""
    var dominantLeg: DominantLeg = .right
    var sport: String = ""
    var experienceLevel: ExperienceLevel = .intermediate
}

enum DominantLeg: String, Codable, CaseIterable, Identifiable {
    case left = "Left"
    case right = "Right"
    case both = "Both"

    var id: String { rawValue }
}

enum ExperienceLevel: String, Codable, CaseIterable, Identifiable {
    case beginner = "Beginner"
    case intermediate = "Intermediate"
    case advanced = "Advanced"
    case elite = "Elite"

    var id: String { rawValue }
}
