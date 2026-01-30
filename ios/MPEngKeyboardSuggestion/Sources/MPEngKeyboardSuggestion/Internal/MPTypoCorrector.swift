// ============================================================
// MPTypoCorrector.swift
// ============================================================
// BK-Tree based typo correction - memory efficient
// Calculates edit distance at runtime with O(log n) tree pruning
// NO SINGLETON - initialize through MPKeyboardSuggestion
// ============================================================

import Foundation

/// BK-Tree based typo corrector
/// Memory efficient: ~200KB vs 15MB for SymSpell
final class MPTypoCorrector {
    
    private weak var resourceLoader: MPResourceLoader?
    
    init(resourceLoader: MPResourceLoader) {
        self.resourceLoader = resourceLoader
    }
    
    // MARK: - Public API
    
    /// Get typo corrections using BK-Tree search
    /// - Parameters:
    ///   - typo: The potentially misspelled word
    ///   - limit: Maximum corrections to return
    /// - Returns: Array of (corrected word, confidence score)
    func getCorrections(for typo: String, limit: Int = 5) -> [(word: String, score: Double)] {
        guard !typo.isEmpty, let loader = resourceLoader else { return [] }
        
        let lowerTypo = typo.lowercased()
        var candidates: [(word: String, score: Double, distance: Int)] = []
        var seen = Set<String>()
        
        // If word exists in vocabulary, it's not a typo
        if let exactIdx = loader.getWordIndex(lowerTypo) {
            let word = loader.getWordByIndex(exactIdx) ?? lowerTypo
            candidates.append((word, 200.0, 0))
            seen.insert(lowerTypo)
        }
        
        // Method 1: BK-Tree search (edit distance 1-2)
        let bkResults = loader.searchBKTree(query: lowerTypo, maxDistance: 2)
        for (word, idx, dist) in bkResults {
            if !seen.contains(word.lowercased()) && dist > 0 {
                let chunkBonus: Double = idx < 7000 ? 2.0 : 1.0
                let score = (100.0 / Double(dist + 1)) * chunkBonus
                candidates.append((word, score, dist))
                seen.insert(word.lowercased())
            }
        }
        
        // Method 2: Keyboard proximity fallback
        if candidates.count < limit {
            let keyboardResults = findKeyboardProximityMatches(typo: lowerTypo, loader: loader, seen: seen)
            for (word, idx, dist) in keyboardResults {
                if !seen.contains(word.lowercased()) {
                    let chunkBonus: Double = idx < 7000 ? 1.5 : 1.0
                    let score = (80.0 / Double(dist + 1)) * chunkBonus
                    candidates.append((word, score, dist))
                    seen.insert(word.lowercased())
                }
            }
        }
        
        candidates.sort { $0.score > $1.score }
        return candidates.prefix(limit).map { ($0.word, $0.score) }
    }
    
    /// Check if word is likely a typo
    func isLikelyTypo(_ word: String) -> Bool {
        guard let loader = resourceLoader else { return false }
        if loader.getWordIndex(word.lowercased()) != nil { return false }
        return !getCorrections(for: word, limit: 1).isEmpty
    }
    
    // MARK: - Private Methods
    
    private func findKeyboardProximityMatches(typo: String, loader: MPResourceLoader, seen: Set<String>) -> [(word: String, index: Int, distance: Int)] {
        var matches: [(word: String, index: Int, distance: Int)] = []
        
        for i in typo.indices {
            let char = typo[i]
            let adjacent = loader.getAdjacentKeys(for: char)
            
            for adjChar in adjacent {
                var modified = typo
                modified.replaceSubrange(i...i, with: String(adjChar))
                
                if let idx = loader.getWordIndex(modified), !seen.contains(modified) {
                    let word = loader.getWordByIndex(idx) ?? modified
                    matches.append((word, idx, 1))
                }
            }
        }
        
        return matches
    }
}
