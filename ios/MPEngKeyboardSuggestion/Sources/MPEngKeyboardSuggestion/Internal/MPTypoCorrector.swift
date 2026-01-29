// ============================================================
// MPTypoCorrector.swift
// ============================================================
// SymSpell-based typo correction
// Pre-computed delete index for O(1) lookup
// Handles vowel swaps, transpositions, omissions
// NO SINGLETON - initialize through MPKeyboardSuggestion
// ============================================================

import Foundation

/// SymSpell-based typo corrector
/// Uses pre-computed delete index for fast lookup
final class MPTypoCorrector {
    
    private weak var resourceLoader: MPResourceLoader?
    
    init(resourceLoader: MPResourceLoader) {
        self.resourceLoader = resourceLoader
    }
    
    // MARK: - Public API
    
    /// Get typo corrections using SymSpell algorithm
    /// - Parameters:
    ///   - typo: The potentially misspelled word
    ///   - limit: Maximum corrections to return
    /// - Returns: Array of (corrected word, confidence score)
    func getCorrections(for typo: String, limit: Int = 5) -> [(word: String, score: Double)] {
        guard !typo.isEmpty, let loader = resourceLoader else { return [] }
        
        let lowerTypo = typo.lowercased()
        var candidates: [(word: String, score: Double, distance: Double)] = []
        var seen = Set<String>()
        
        // If word exists in vocabulary, prioritize it
        if let exactIdx = loader.getWordIndex(lowerTypo) {
            let word = loader.getWordByIndex(exactIdx) ?? lowerTypo
            candidates.append((word, 200.0, 0))
            seen.insert(lowerTypo)
        }
        
        // Method 1: SymSpell delete lookup
        let symspellResults = findSymSpellMatches(typo: lowerTypo, loader: loader)
        for (word, idx, dist) in symspellResults {
            if !seen.contains(word.lowercased()) {
                let chunkBonus: Double = idx < 7000 ? 2.0 : 1.0
                let score = (100.0 / (dist + 0.5)) * chunkBonus
                candidates.append((word, score, dist))
                seen.insert(word.lowercased())
            }
        }
        
        // Method 2: Keyboard proximity fallback (for adjacent key typos)
        if candidates.count < limit {
            let keyboardResults = findKeyboardProximityMatches(typo: lowerTypo, loader: loader, seen: seen)
            for (word, idx, dist) in keyboardResults {
                if !seen.contains(word.lowercased()) {
                    let chunkBonus: Double = idx < 7000 ? 1.5 : 1.0
                    let score = (80.0 / (dist + 0.5)) * chunkBonus
                    candidates.append((word, score, dist))
                    seen.insert(word.lowercased())
                }
            }
        }
        
        candidates.sort { $0.score > $1.score }
        return candidates.prefix(limit).map { ($0.word, $0.score) }
    }
    
    /// Check if word is likely a typo (not in vocabulary and has corrections)
    func isLikelyTypo(_ word: String) -> Bool {
        guard let loader = resourceLoader else { return false }
        
        // If word exists in vocabulary, it's not a typo
        if loader.getWordIndex(word.lowercased()) != nil { return false }
        
        // If we can find corrections, it's likely a typo
        return !getCorrections(for: word, limit: 1).isEmpty
    }
    
    // MARK: - SymSpell Implementation
    
    /// Find matches using SymSpell delete index
    private func findSymSpellMatches(typo: String, loader: MPResourceLoader) -> [(word: String, index: Int, distance: Double)] {
        var matches: [(word: String, index: Int, distance: Double)] = []
        
        // Generate deletes of the input typo
        let inputDeletes = generateDeletes(word: typo, maxDistance: 2)
        
        // Include the typo itself
        var searchTerms = inputDeletes
        searchTerms.insert(typo)
        
        // Look up each delete variant in the index
        for term in searchTerms {
            if let wordIndices = loader.getSymSpellMatches(for: term) {
                for idx in wordIndices.prefix(10) {
                    if let word = loader.getWordByIndex(idx) {
                        let dist = editDistance(typo, word.lowercased())
                        if dist <= 2.0 && word.lowercased() != typo {
                            matches.append((word, idx, dist))
                        }
                    }
                }
            }
        }
        
        return matches
    }
    
    /// Find matches using keyboard proximity
    private func findKeyboardProximityMatches(typo: String, loader: MPResourceLoader, seen: Set<String>) -> [(word: String, index: Int, distance: Double)] {
        var matches: [(word: String, index: Int, distance: Double)] = []
        
        // Try replacing each character with adjacent keys
        for i in typo.indices {
            let char = typo[i]
            let adjacent = loader.getAdjacentKeys(for: char)
            
            for adjChar in adjacent {
                var modified = typo
                modified.replaceSubrange(i...i, with: String(adjChar))
                
                if let idx = loader.getWordIndex(modified), !seen.contains(modified) {
                    let word = loader.getWordByIndex(idx) ?? modified
                    matches.append((word, idx, 1.0))
                }
            }
        }
        
        return matches
    }
    
    // MARK: - Helper Methods
    
    /// Generate all delete variants within edit distance
    private func generateDeletes(word: String, maxDistance: Int) -> Set<String> {
        var deletes = Set<String>()
        generateDeletesRecursive(word: word, depth: 0, maxDepth: maxDistance, deletes: &deletes)
        return deletes
    }
    
    private func generateDeletesRecursive(word: String, depth: Int, maxDepth: Int, deletes: inout Set<String>) {
        guard depth < maxDepth && word.count > 1 else { return }
        
        for i in word.indices {
            var deleted = word
            deleted.remove(at: i)
            
            if !deletes.contains(deleted) {
                deletes.insert(deleted)
                generateDeletesRecursive(word: deleted, depth: depth + 1, maxDepth: maxDepth, deletes: &deletes)
            }
        }
    }
    
    /// Calculate edit distance between two strings
    private func editDistance(_ s1: String, _ s2: String) -> Double {
        let m = s1.count
        let n = s2.count
        
        if m == 0 { return Double(n) }
        if n == 0 { return Double(m) }
        
        let s1Array = Array(s1)
        let s2Array = Array(s2)
        
        var prev = Array(0...n)
        var curr = [Int](repeating: 0, count: n + 1)
        
        for i in 1...m {
            curr[0] = i
            for j in 1...n {
                let cost = s1Array[i-1] == s2Array[j-1] ? 0 : 1
                curr[j] = min(
                    prev[j] + 1,      // deletion
                    curr[j-1] + 1,    // insertion
                    prev[j-1] + cost  // substitution
                )
                
                // Transposition check
                if i > 1 && j > 1 && s1Array[i-1] == s2Array[j-2] && s1Array[i-2] == s2Array[j-1] {
                    curr[j] = min(curr[j], prev[j-1] + cost)
                }
            }
            swap(&prev, &curr)
        }
        
        return Double(prev[n])
    }
}
