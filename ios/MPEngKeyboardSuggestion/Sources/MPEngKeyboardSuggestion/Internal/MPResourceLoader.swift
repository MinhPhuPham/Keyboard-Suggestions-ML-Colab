// ============================================================
// MPResourceLoader.swift
// ============================================================
// Handles loading vocabulary chunks, compact trie, and SymSpell index
// NO SINGLETON - initialize through MPKeyboardSuggestion
// ============================================================

import Foundation

public enum MPVocabChunk: String, CaseIterable {
    case high = "vocab_high"
    case medium = "vocab_medium"
    case low = "vocab_low"
}

/// Result of trie completion lookup
struct MPTrieResult {
    let word: String
    let globalIndex: Int
    let chunk: MPVocabChunk
}

final class MPResourceLoader {
    
    // MARK: - Properties
    
    private var loadedChunks: [MPVocabChunk: [String]] = [:]
    private var compactTrie: [String: Any]?  // Nested trie structure
    private var symspellIndex: [String: [Int]]?  // Delete variants -> word indices
    private var keyboardAdjacent: [String: String] = [:]
    private var wordToIndex: [String: Int] = [:]
    private var indexToWord: [Int: String] = [:]
    
    private let loadQueue = DispatchQueue(label: "com.mp.keyboard.loader", qos: .userInitiated)
    private var loadingChunks = Set<MPVocabChunk>()
    private let bundle: Bundle
    
    // MARK: - Initialization
    
    init(bundle: Bundle = .main) {
        self.bundle = bundle
        MPLog.debug("[MPResourceLoader] init - Loading resources from bundle: \(bundle.bundlePath)")
        loadWordMappings()
        loadCompactTrie()
        loadSymSpellIndex()
        loadKeyboardAdjacent()
        loadChunk(.high)
        MPLog.debug("[MPResourceLoader] init - Finished. vocabSize=\(vocabSize), trieLoaded=\(compactTrie != nil), symspellLoaded=\(symspellIndex != nil)")
    }
    
    // MARK: - Public API
    
    /// Get words from a specific chunk
    func getChunkWords(_ chunk: MPVocabChunk) -> [String]? {
        if loadedChunks[chunk] == nil {
            loadChunk(chunk)
        }
        return loadedChunks[chunk]
    }
    
    /// Get completions using compact trie traversal
    /// - Parameters:
    ///   - prefix: The prefix to search for
    ///   - limit: Maximum results to return
    /// - Returns: Array of matching words with metadata
    func getTrieCompletions(for prefix: String, limit: Int = 50) -> [MPTrieResult] {
        let lowerPrefix = prefix.lowercased()
        
        MPLog.debug("[MPResourceLoader] getTrieCompletions - prefix='\(lowerPrefix)', trieLoaded=\(compactTrie != nil)")
        
        guard let trie = compactTrie else {
            MPLog.debug("[MPResourceLoader] getTrieCompletions - No trie, falling back to linear search")
            return searchLoadedChunks(prefix: lowerPrefix, limit: limit)
        }
        
        // Traverse trie to find the node matching the prefix
        var node: [String: Any] = trie
        for char in lowerPrefix {
            let charStr = String(char)
            if let next = node[charStr] as? [String: Any] {
                node = next
            } else {
                MPLog.debug("[MPResourceLoader] getTrieCompletions - Prefix '\(lowerPrefix)' not in trie, trying fallback")
                // Try shorter prefixes as fallback
                return getTrieCompletionsFallback(prefix: lowerPrefix, limit: limit)
            }
        }
        
        // Collect all word indices from this node and descendants
        var indices: [Int] = []
        collectIndicesFromNode(node, into: &indices, limit: limit)
        
        // Convert indices to results
        var results: [MPTrieResult] = []
        for idx in indices {
            if let word = indexToWord[idx] {
                let chunk = getChunkForIndex(idx)
                results.append(MPTrieResult(word: word, globalIndex: idx, chunk: chunk))
                if results.count >= limit { break }
            }
        }
        
        MPLog.debug("[MPResourceLoader] getTrieCompletions - Found \(results.count) results for '\(lowerPrefix)'")
        return results
    }
    
    /// Get SymSpell matches for a term (exact or delete variant)
    func getSymSpellMatches(for term: String) -> [Int]? {
        return symspellIndex?[term.lowercased()]
    }
    
    /// Get word index from word
    func getWordIndex(_ word: String) -> Int? {
        return wordToIndex[word.lowercased()]
    }
    
    /// Get word from index
    func getWordByIndex(_ index: Int) -> String? {
        return indexToWord[index]
    }
    
    /// Get adjacent keys for typo correction
    func getAdjacentKeys(for key: Character) -> String {
        return keyboardAdjacent[String(key).lowercased()] ?? ""
    }
    
    /// Total vocabulary size
    var vocabSize: Int { wordToIndex.count }
    
    /// Async chunk loading
    func loadChunkAsync(_ chunk: MPVocabChunk, completion: (() -> Void)? = nil) {
        if loadedChunks[chunk] != nil { completion?(); return }
        loadQueue.async { [weak self] in
            self?.loadChunk(chunk)
            DispatchQueue.main.async { completion?() }
        }
    }
    
    /// Preload all chunks
    func preloadAllChunks(completion: (() -> Void)? = nil) {
        loadQueue.async { [weak self] in
            self?.loadChunk(.medium)
            self?.loadChunk(.low)
            DispatchQueue.main.async { completion?() }
        }
    }
    
    /// Release memory
    func releaseMemory(keepHigh: Bool = true) {
        if !keepHigh { loadedChunks.removeValue(forKey: .high) }
        loadedChunks.removeValue(forKey: .medium)
        loadedChunks.removeValue(forKey: .low)
    }
    
    // MARK: - Private Loading Methods
    
    private func loadCompactTrie() {
        guard let url = bundle.url(forResource: "compact_trie", withExtension: "json") else {
            MPLog.debug("[MPResourceLoader] loadCompactTrie - compact_trie.json not found, will use fallback")
            return
        }
        guard let data = try? Data(contentsOf: url, options: .mappedIfSafe),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            MPLog.error("[MPResourceLoader] loadCompactTrie - Failed to parse compact_trie.json")
            return
        }
        
        compactTrie = json
        MPLog.debug("[MPResourceLoader] loadCompactTrie - ✅ Loaded compact trie")
    }
    
    private func loadSymSpellIndex() {
        guard let url = bundle.url(forResource: "symspell_index", withExtension: "json") else {
            MPLog.debug("[MPResourceLoader] loadSymSpellIndex - symspell_index.json not found, will use fallback")
            return
        }
        guard let data = try? Data(contentsOf: url, options: .mappedIfSafe),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: [Int]] else {
            MPLog.error("[MPResourceLoader] loadSymSpellIndex - Failed to parse symspell_index.json")
            return
        }
        
        symspellIndex = json
        MPLog.debug("[MPResourceLoader] loadSymSpellIndex - ✅ Loaded \(json.count) symspell entries")
    }
    
    private func loadKeyboardAdjacent() {
        guard let url = bundle.url(forResource: "keyboard_adjacent", withExtension: "json"),
              let data = try? Data(contentsOf: url, options: .mappedIfSafe),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: String] else {
            // Use default keyboard layout
            keyboardAdjacent = [
                "q": "wa", "w": "qase", "e": "wsdr", "r": "edft", "t": "rfgy",
                "y": "tghu", "u": "yhji", "i": "ujko", "o": "iklp", "p": "ol",
                "a": "qwsz", "s": "awedxz", "d": "serfcx", "f": "drtgvc", "g": "ftyhbv",
                "h": "gyujnb", "j": "huikmn", "k": "jiolm", "l": "kop",
                "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn",
                "n": "bhjm", "m": "njk"
            ]
            return
        }
        keyboardAdjacent = json
    }
    
    private func loadWordMappings() {
        guard let url = bundle.url(forResource: "word_to_index", withExtension: "json") else {
            MPLog.error("[MPResourceLoader] loadWordMappings - ❌ word_to_index.json not found")
            return
        }
        guard let data = try? Data(contentsOf: url, options: .mappedIfSafe),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Int] else {
            MPLog.error("[MPResourceLoader] loadWordMappings - ❌ Failed to parse word_to_index.json")
            return
        }
        
        wordToIndex = json
        indexToWord = wordToIndex.reduce(into: [:]) { result, pair in
            result[pair.value] = pair.key
        }
        MPLog.debug("[MPResourceLoader] loadWordMappings - ✅ Loaded \(wordToIndex.count) word mappings")
    }
    
    private func loadChunk(_ chunk: MPVocabChunk) {
        guard loadedChunks[chunk] == nil, !loadingChunks.contains(chunk) else { return }
        loadingChunks.insert(chunk)
        
        guard let url = bundle.url(forResource: chunk.rawValue, withExtension: "json"),
              let data = try? Data(contentsOf: url, options: .mappedIfSafe),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let words = json["words"] as? [String] else {
            loadingChunks.remove(chunk)
            MPLog.error("[MPResourceLoader] loadChunk - ❌ Failed to load \(chunk.rawValue).json")
            return
        }
        
        loadedChunks[chunk] = words
        loadingChunks.remove(chunk)
        MPLog.debug("[MPResourceLoader] loadChunk - ✅ \(chunk.rawValue) (\(words.count) words)")
    }
    
    // MARK: - Trie Helper Methods
    
    /// Recursively collect word indices from trie node and descendants
    private func collectIndicesFromNode(_ node: [String: Any], into indices: inout [Int], limit: Int) {
        // Get indices at this node (terminal words)
        if let terminalIndices = node["$"] as? [Int] {
            indices.append(contentsOf: terminalIndices)
        }
        
        if indices.count >= limit { return }
        
        // Traverse children (in alphabetical order for consistency)
        for (key, value) in node.sorted(by: { $0.key < $1.key }) {
            if key == "$" { continue }
            if let childNode = value as? [String: Any] {
                collectIndicesFromNode(childNode, into: &indices, limit: limit)
                if indices.count >= limit { return }
            }
        }
    }
    
    /// Fallback: try shorter prefixes
    private func getTrieCompletionsFallback(prefix: String, limit: Int) -> [MPTrieResult] {
        // Try progressively shorter prefixes
        for prefixLen in stride(from: prefix.count - 1, through: 1, by: -1) {
            let shorterPrefix = String(prefix.prefix(prefixLen))
            if let trie = compactTrie {
                var node: [String: Any] = trie
                var found = true
                
                for char in shorterPrefix {
                    if let next = node[String(char)] as? [String: Any] {
                        node = next
                    } else {
                        found = false
                        break
                    }
                }
                
                if found {
                    var indices: [Int] = []
                    collectIndicesFromNode(node, into: &indices, limit: limit * 2)
                    
                    // Filter to only words that actually start with the original prefix
                    var results: [MPTrieResult] = []
                    for idx in indices {
                        if let word = indexToWord[idx], word.lowercased().hasPrefix(prefix) {
                            let chunk = getChunkForIndex(idx)
                            results.append(MPTrieResult(word: word, globalIndex: idx, chunk: chunk))
                            if results.count >= limit { break }
                        }
                    }
                    
                    if !results.isEmpty { return results }
                }
            }
        }
        
        // Final fallback: linear search
        return searchLoadedChunks(prefix: prefix, limit: limit)
    }
    
    /// Linear search through loaded chunks (fallback)
    private func searchLoadedChunks(prefix: String, limit: Int) -> [MPTrieResult] {
        var results: [MPTrieResult] = []
        let lowerPrefix = prefix.lowercased()
        
        for chunk in MPVocabChunk.allCases {
            guard let words = loadedChunks[chunk] else { continue }
            for (idx, word) in words.enumerated() {
                if word.lowercased().hasPrefix(lowerPrefix) {
                    let globalIdx = wordToIndex[word.lowercased()] ?? idx
                    results.append(MPTrieResult(word: word, globalIndex: globalIdx, chunk: chunk))
                    if results.count >= limit { return results }
                }
            }
        }
        return results
    }
    
    /// Determine chunk for a word index
    private func getChunkForIndex(_ index: Int) -> MPVocabChunk {
        // Based on Zipf ordering: lower indices = high frequency
        if index < 7000 { return .high }
        if index < 17000 { return .medium }
        return .low
    }
}
