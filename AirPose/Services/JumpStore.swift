import Foundation

final class LocalJSONFileStore<Value: Codable> {
    private let fileURL: URL
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    init(filename: String) {
        let baseURL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        let directoryURL = baseURL.appendingPathComponent("AirPose", isDirectory: true)
        fileURL = directoryURL.appendingPathComponent(filename)

        encoder = JSONEncoder()
        decoder = JSONDecoder()
        encoder.dateEncodingStrategy = .iso8601
        decoder.dateDecodingStrategy = .iso8601

        try? FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)
    }

    func load(defaultValue: Value) -> Value {
        guard let data = try? Data(contentsOf: fileURL) else { return defaultValue }
        return (try? decoder.decode(Value.self, from: data)) ?? defaultValue
    }

    func save(_ value: Value) throws {
        let data = try encoder.encode(value)
        try data.write(to: fileURL, options: [.atomic])
    }
}

@MainActor
final class JumpStore: ObservableObject {
    @Published private(set) var jumps: [Jump] = []

    private let fileStore = LocalJSONFileStore<[Jump]>(filename: "jumps.json")

    init() {
        load()
    }

    func add(_ jump: Jump) {
        jumps.insert(jump, at: 0)
        save()
    }

    func delete(_ jump: Jump) {
        jumps.removeAll { $0.id == jump.id }
        save()
    }

    func update(_ jump: Jump) {
        guard let index = jumps.firstIndex(where: { $0.id == jump.id }) else { return }
        jumps[index] = jump
        save()
    }

    func replaceAll(with jumps: [Jump]) {
        self.jumps = jumps
        save()
    }

    var currentJumps: [Jump] {
        jumps
    }

    private func load() {
        jumps = fileStore.load(defaultValue: [])
    }

    private func save() {
        do {
            try fileStore.save(jumps)
        } catch {
            assertionFailure("Failed to save jumps: \(error)")
        }
    }
}
