// ============================================================
// MPLearningManager.swift
// ============================================================
// User behavior learning: unigram boost, bigram context, recency
// NO SINGLETON - initialize through MPKeyboardSuggestion
// ============================================================

import Foundation

public struct MPLearningStats: Sendable {
    public let words: Int
    public let bigrams: Int
}

final class MPLearningManager {
    
    private struct LearningData: Codable {
        var unigramCounts: [String: Int] = [:]
        var bigramCounts: [String: [String: Int]] = [:]
        var lastUsed: [String: Date] = [:]
    }
    
    private var data = LearningData()
    private var lastContext = ""
    private var isDirty = false
    private var saveWorkItem: DispatchWorkItem?
    
    private let saveQueue = DispatchQueue(label: "com.mp.keyboard.learning", qos: .utility)
    private let storageURL: URL
    
    var recencyDecayDays = 30.0
    var maxEntries = 5000
    
    init(storageDirectory: URL? = nil) {
        let directory = storageDirectory ?? FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        storageURL = directory.appendingPathComponent("mp_keyboard_learning.json")
        load()
    }
    
    // MARK: - Public API
    
    func recordSelection(_ word: String, context: String? = nil) {
        let lowerWord = word.lowercased()
        let contextWord = (context ?? lastContext).lowercased()
            .split(separator: " ").last.map(String.init) ?? ""
        
        data.unigramCounts[lowerWord, default: 0] += 1
        
        if !contextWord.isEmpty {
            if data.bigramCounts[contextWord] == nil {
                data.bigramCounts[contextWord] = [:]
            }
            data.bigramCounts[contextWord]![lowerWord, default: 0] += 1
        }
        
        data.lastUsed[lowerWord] = Date()
        lastContext = lowerWord
        
        isDirty = true
        scheduleSave()
        
        if data.unigramCounts.count > maxEntries {
            prune()
        }
    }
    
    func getBoostScore(for word: String, context: String) -> Double {
        let lowerWord = word.lowercased()
        let contextWord = context.lowercased().split(separator: " ").last.map(String.init) ?? ""
        
        var score = 0.0
        
        if let count = data.unigramCounts[lowerWord] {
            score += Double(count) * 10
        }
        
        if !contextWord.isEmpty,
           let bigramMap = data.bigramCounts[contextWord],
           let count = bigramMap[lowerWord] {
            score += Double(count) * 50
        }
        
        if let lastUsedDate = data.lastUsed[lowerWord] {
            let daysSince = Date().timeIntervalSince(lastUsedDate) / 86400
            let decay = pow(0.5, daysSince / recencyDecayDays)
            score *= (0.5 + 0.5 * decay)
        }
        
        return score
    }
    
    func getBigramSuggestions(for contextWord: String, limit: Int = 5) -> [(word: String, count: Int)] {
        guard let bigramMap = data.bigramCounts[contextWord.lowercased()] else {
            return []
        }
        return bigramMap.sorted { $0.value > $1.value }
            .prefix(limit)
            .map { ($0.key, $0.value) }
    }
    
    func updateContext(_ context: String) {
        lastContext = context
    }
    
    func getStats() -> MPLearningStats {
        let bigramCount = data.bigramCounts.values.reduce(0) { $0 + $1.count }
        return MPLearningStats(words: data.unigramCounts.count, bigrams: bigramCount)
    }
    
    func clear() {
        data = LearningData()
        try? FileManager.default.removeItem(at: storageURL)
    }
    
    // MARK: - Persistence
    
    private func load() {
        guard let jsonData = try? Data(contentsOf: storageURL),
              let decoded = try? JSONDecoder().decode(LearningData.self, from: jsonData) else {
            return
        }
        data = decoded
    }
    
    private func save() {
        guard isDirty else { return }
        saveQueue.async { [weak self] in
            guard let self = self else { return }
            if let encoded = try? JSONEncoder().encode(self.data) {
                try? encoded.write(to: self.storageURL)
            }
            self.isDirty = false
        }
    }
    
    private func scheduleSave() {
        saveWorkItem?.cancel()
        let workItem = DispatchWorkItem { [weak self] in
            self?.save()
        }
        saveWorkItem = workItem
        DispatchQueue.main.asyncAfter(deadline: .now() + 5, execute: workItem)
    }
    
    private func prune() {
        let cutoffDate = Date().addingTimeInterval(-90 * 86400)
        
        let toRemove = data.lastUsed.filter { $0.value < cutoffDate && (data.unigramCounts[$0.key] ?? 0) < 3 }.keys
        for word in toRemove {
            data.unigramCounts.removeValue(forKey: word)
            data.lastUsed.removeValue(forKey: word)
        }
        
        for (context, words) in data.bigramCounts {
            data.bigramCounts[context] = words.filter { $0.value >= 2 }
            if data.bigramCounts[context]?.isEmpty == true {
                data.bigramCounts.removeValue(forKey: context)
            }
        }
    }
}
