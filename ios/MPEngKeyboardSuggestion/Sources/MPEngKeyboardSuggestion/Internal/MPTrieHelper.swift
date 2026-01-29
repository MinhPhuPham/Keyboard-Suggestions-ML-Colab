// ============================================================
// MPTrieHelper.swift
// ============================================================
// Fast prefix search using flat array index
// NO SINGLETON - initialize through MPKeyboardSuggestion
// ============================================================

import Foundation

final class MPTrieHelper {
    
    private weak var resourceLoader: MPResourceLoader?
    
    init(resourceLoader: MPResourceLoader) {
        self.resourceLoader = resourceLoader
    }
    
    func searchPrefix(_ prefix: String, limit: Int = 10) -> [(word: String, score: Double)] {
        guard !prefix.isEmpty, let loader = resourceLoader else { return [] }
        
        let completions = loader.getCompletions(for: prefix.lowercased(), limit: limit * 2)
        
        return completions.enumerated().map { (idx, item) in
            let chunkBonus: Double = {
                switch item.chunk {
                case .high: return 100.0
                case .medium: return 50.0
                case .low: return 10.0
                }
            }()
            
            let positionPenalty = Double(idx) * 0.5
            let score = chunkBonus - positionPenalty + (100.0 / Double(item.localIndex + 1))
            return (item.word, score)
        }.sorted { $0.score > $1.score }.prefix(limit).map { $0 }
    }
    
    func searchPrefixWithLength(_ prefix: String, minLength: Int, maxLength: Int, limit: Int = 20) -> [(word: String, score: Double)] {
        return searchPrefix(prefix, limit: limit * 2)
            .filter { $0.word.count >= minLength && $0.word.count <= maxLength }
            .prefix(limit)
            .map { $0 }
    }
    
    func wordExists(_ word: String) -> Bool {
        return resourceLoader?.getWordIndex(word) != nil
    }
    
    func getWordRank(_ word: String) -> Int? {
        return resourceLoader?.getWordIndex(word)
    }
    
    func fuzzySearch(_ query: String, tolerance: Int = 2, limit: Int = 10) -> [(word: String, score: Double)] {
        guard query.count >= 2 else { return [] }
        
        for prefixLen in stride(from: query.count, through: max(1, query.count - tolerance), by: -1) {
            let prefix = String(query.prefix(prefixLen))
            let results = searchPrefix(prefix, limit: limit)
            if !results.isEmpty { return results }
        }
        
        return []
    }
}
