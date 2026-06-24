import Foundation

final class LocalJSONFileStore<Value: Codable> {
    private let fileURL: URL
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder
    private let fileManager = FileManager.default

    init(filename: String) {
        let baseURL = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? fileManager.temporaryDirectory
        let directoryURL = baseURL.appendingPathComponent("JumpGuard", isDirectory: true)
        let legacyDirectoryURL = baseURL.appendingPathComponent("AirPose", isDirectory: true)
        fileURL = directoryURL.appendingPathComponent(filename)
        let legacyFileURL = legacyDirectoryURL.appendingPathComponent(filename)

        encoder = JSONEncoder()
        decoder = JSONDecoder()
        encoder.dateEncodingStrategy = .iso8601
        decoder.dateDecodingStrategy = .iso8601

        try? fileManager.createDirectory(at: directoryURL, withIntermediateDirectories: true)

        if !fileManager.fileExists(atPath: fileURL.path),
           fileManager.fileExists(atPath: legacyFileURL.path) {
            try? fileManager.copyItem(at: legacyFileURL, to: fileURL)
        }
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
