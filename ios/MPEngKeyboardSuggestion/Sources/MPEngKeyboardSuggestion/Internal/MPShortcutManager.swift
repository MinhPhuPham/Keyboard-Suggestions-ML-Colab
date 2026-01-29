// ============================================================
// MPShortcutManager.swift
// ============================================================
// Custom user shortcuts (input → suggestions)
// NO SINGLETON - initialize through MPKeyboardSuggestion
// ============================================================

import Foundation

final class MPShortcutManager {
    
    private var shortcuts: [String: [String]] = [:]
    private let saveQueue = DispatchQueue(label: "com.mp.keyboard.shortcuts", qos: .utility)
    private let storageURL: URL
    
    var count: Int { shortcuts.count }
    
    init(storageDirectory: URL? = nil) {
        let directory = storageDirectory ?? FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        storageURL = directory.appendingPathComponent("mp_keyboard_shortcuts.json")
        load()
    }
    
    // MARK: - Public API
    
    func addShortcut(_ input: String, _ suggestion: String) {
        let key = input.lowercased()
        if shortcuts[key] == nil {
            shortcuts[key] = []
        }
        if !shortcuts[key]!.contains(suggestion) {
            shortcuts[key]!.append(suggestion)
        }
        save()
    }
    
    func addShortcuts(_ dict: [String: String]) {
        for (input, suggestion) in dict {
            addShortcut(input, suggestion)
        }
    }
    
    func removeShortcut(_ input: String) {
        shortcuts.removeValue(forKey: input.lowercased())
        save()
    }
    
    func getSuggestions(for input: String) -> [String]? {
        return shortcuts[input.lowercased()]
    }
    
    func getPartialMatches(for query: String) -> [(key: String, suggestions: [String])] {
        let lowerQuery = query.lowercased()
        return shortcuts
            .filter { $0.key.hasPrefix(lowerQuery) && $0.key != lowerQuery }
            .map { ($0.key, $0.value) }
    }
    
    func hasShortcut(_ input: String) -> Bool {
        return shortcuts[input.lowercased()] != nil
    }
    
    func getAllShortcuts() -> [String: [String]] {
        return shortcuts
    }
    
    func clear() {
        shortcuts.removeAll()
        try? FileManager.default.removeItem(at: storageURL)
    }
    
    // MARK: - Persistence
    
    private func load() {
        guard let data = try? Data(contentsOf: storageURL),
              let decoded = try? JSONDecoder().decode([String: [String]].self, from: data) else {
            return
        }
        shortcuts = decoded
    }
    
    private func save() {
        saveQueue.async { [weak self] in
            guard let self = self else { return }
            if let encoded = try? JSONEncoder().encode(self.shortcuts) {
                try? encoded.write(to: self.storageURL)
            }
        }
    }
}
