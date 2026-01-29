// ============================================================
// MPTypoCorrector.swift
// ============================================================
// Soundex phonetic matching + keyboard proximity typo detection
// NO SINGLETON - initialize through MPKeyboardSuggestion
// ============================================================

import Foundation

final class MPTypoCorrector {
    
    private weak var resourceLoader: MPResourceLoader?
    
    private static let soundexMapping: [Character: Character] = [
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6"
    ]
    
    init(resourceLoader: MPResourceLoader) {
        self.resourceLoader = resourceLoader
    }
    
    // MARK: - Public API
    
    func getCorrections(for typo: String, limit: Int = 5) -> [(word: String, score: Double)] {
        guard !typo.isEmpty, let loader = resourceLoader else { return [] }
        
        let lowerTypo = typo.lowercased()
        var candidates: [(word: String, score: Double, distance: Double)] = []
        var seen = Set<String>()
        
        // Method 1: Soundex matches
        let typoCode = soundex(lowerTypo)
        if let matches = loader.getSoundexMatches(for: typoCode) {
            for (chunkName, localIndices) in matches {
                guard let chunk = MPVocabChunk(rawValue: "vocab_\(chunkName)"),
                      let words = loader.getChunkWords(chunk) else {
                    continue
                }
                
                for localIdx in localIndices.prefix(20) {
                    guard localIdx < words.count else { continue }
                    let word = words[localIdx]
                    
                    if word.lowercased() != lowerTypo && !seen.contains(word.lowercased()) {
                        let dist = weightedEditDistance(lowerTypo, word.lowercased())
                        if dist <= 4.0 {
                            let score = 100.0 / (dist + 0.5)
                            candidates.append((word, score, dist))
                            seen.insert(word.lowercased())
                        }
                    }
                }
            }
        }
        
        // Method 2: Prefix + edit distance
        let prefix = String(lowerTypo.prefix(2))
        let prefixMatches = loader.getCompletions(for: prefix, limit: 50)
        
        for (word, _, chunk) in prefixMatches {
            if !seen.contains(word.lowercased()) && abs(word.count - lowerTypo.count) <= 3 {
                let dist = weightedEditDistance(lowerTypo, word.lowercased())
                if dist <= 2.5 && word.lowercased() != lowerTypo {
                    let chunkBonus: Double = chunk == .high ? 2.0 : 1.0
                    let score = (100.0 / (dist + 0.5)) * chunkBonus
                    candidates.append((word, score, dist))
                    seen.insert(word.lowercased())
                }
            }
        }
        
        candidates.sort { $0.score > $1.score }
        return candidates.prefix(limit).map { ($0.word, $0.score) }
    }
    
    func soundex(_ word: String) -> String {
        guard !word.isEmpty else { return "" }
        
        let upper = word.uppercased()
        var result = String(upper.first!)
        var prevCode: Character = Self.soundexMapping[upper.first!] ?? "0"
        
        for char in upper.dropFirst() {
            let code = Self.soundexMapping[char] ?? "0"
            if code != "0" && code != prevCode {
                result.append(code)
            }
            if code != "0" { prevCode = code }
        }
        
        while result.count < 4 { result.append("0") }
        return String(result.prefix(4))
    }
    
    func isAdjacentKey(_ c1: Character, _ c2: Character) -> Bool {
        guard let loader = resourceLoader else { return false }
        let adjacent = loader.getAdjacentKeys(for: c1)
        return adjacent.contains(String(c2).lowercased())
    }
    
    func keyboardDistance(_ c1: Character, _ c2: Character) -> Double {
        if c1 == c2 { return 0 }
        return isAdjacentKey(c1, c2) ? 0.5 : 1.0
    }
    
    func weightedEditDistance(_ s1: String, _ s2: String, maxDist: Int = 4) -> Double {
        if abs(s1.count - s2.count) > maxDist { return Double(maxDist + 1) }
        
        let a1 = Array(s1)
        let a2 = Array(s2)
        let m = a1.count
        let n = a2.count
        
        if m == 0 { return Double(n) }
        if n == 0 { return Double(m) }
        
        var dp = [[Double]](repeating: [Double](repeating: 0, count: n + 1), count: m + 1)
        
        for i in 0...m { dp[i][0] = Double(i) }
        for j in 0...n { dp[0][j] = Double(j) }
        
        for i in 1...m {
            for j in 1...n {
                if a1[i-1] == a2[j-1] {
                    dp[i][j] = dp[i-1][j-1]
                } else {
                    let subCost = keyboardDistance(a1[i-1], a2[j-1])
                    dp[i][j] = min(
                        dp[i-1][j] + 1,
                        dp[i][j-1] + 1,
                        dp[i-1][j-1] + subCost
                    )
                }
            }
        }
        
        return dp[m][n]
    }
    
    func isLikelyTypo(_ word: String) -> Bool {
        guard let loader = resourceLoader else { return false }
        if loader.getWordIndex(word.lowercased()) != nil { return false }
        return !getCorrections(for: word, limit: 1).isEmpty
    }
}
