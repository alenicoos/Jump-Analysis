import Foundation

struct UserProfile: Codable, Equatable {
    var name: String = ""
    var age: String = ""
    var height: String = ""
    var weight: String = ""
    var dominantLeg: DominantLeg = .right
    var sport: String = ""
    var experienceLevel: ExperienceLevel = .intermediate

    var trimmedName: String {
        name.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var displayName: String {
        trimmedName.isEmpty ? "Unnamed Athlete" : trimmedName
    }

    var summaryText: String {
        let sport = sport.trimmingCharacters(in: .whitespacesAndNewlines)
        let level = experienceLevel.rawValue
        if sport.isEmpty {
            return "\(level) athlete"
        }
        return "\(level) \(sport) athlete"
    }
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
