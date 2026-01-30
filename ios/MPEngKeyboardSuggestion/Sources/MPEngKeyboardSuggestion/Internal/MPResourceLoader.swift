// ============================================================
// MPResourceLoader.swift
// ============================================================
// Handles loading vocabulary chunks, compact trie, and BK-Tree
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

/// BK-Tree node for typo correction
private class BKTreeNode {
    let wordIndex: Int
    var children: [Int: BKTreeNode] = [:]  // distance -> child
    
    init(wordIndex: Int) {
        self.wordIndex = wordIndex
    }
}

final class MPResourceLoader {
    
    // MARK: - Properties
    
    private var loadedChunks: [MPVocabChunk: [String]] = [:]
    private var compactTrie: [String: Any]?
    private var bkTreeRoot: BKTreeNode?
    private var keyboardAdjacent: [String: String] = [:]
    private var wordToIndex: [String: Int] = [:]
    private var indexToWord: [Int: String] = [:]
    
    private let loadQueue = DispatchQueue(label: "com.mp.keyboard.loader", qos: .userInitiated)
    private var loadingChunks = Set<MPVocabChunk>()
    private let bundle: Bundle
    
    // MARK: - Initialization
    
    init(bundle: Bundle = .main) {
        self.bundle = bundle
        MPLog.debug("[MPResourceLoader] init - Loading resources")
        loadWordMappings()
        loadCompactTrie()
        loadBKTree()
        loadKeyboardAdjacent()
        loadChunk(.high)
        MPLog.debug("[MPResourceLoader] init - Done. vocab=\(vocabSize), trie=\(compactTrie != nil), bkTree=\(bkTreeRoot != nil)")
    }
    
    // MARK: - Public API
    
    func getChunkWords(_ chunk: MPVocabChunk) -> [String]? {
        if loadedChunks[chunk] == nil { loadChunk(chunk) }
        return loadedChunks[chunk]
    }
    
    /// Get completions using compact trie traversal
    func getTrieCompletions(for prefix: String, limit: Int = 50) -> [MPTrieResult] {
        let lowerPrefix = prefix.lowercased()
        
        guard let trie = compactTrie else {
            return searchLoadedChunks(prefix: lowerPrefix, limit: limit)
        }
        
        // Traverse trie to find the node matching the prefix
        var node: [String: Any] = trie
        for char in lowerPrefix {
            if let next = node[String(char)] as? [String: Any] {
                node = next
            } else {
                return getTrieCompletionsFallback(prefix: lowerPrefix, limit: limit)
            }
        }
        
        // Collect word indices from this node
        var indices: [Int] = []
        collectIndicesFromNode(node, into: &indices, limit: limit)
        
        return indices.prefix(limit).compactMap { idx -> MPTrieResult? in
            guard let word = indexToWord[idx] else { return nil }
            return MPTrieResult(word: word, globalIndex: idx, chunk: getChunkForIndex(idx))
        }
    }
    
    /// Search BK-Tree for words within edit distance
    /// - Parameters:
    ///   - query: The word to find corrections for
    ///   - maxDistance: Maximum edit distance (default 2)
    ///   - limit: Maximum results to return (default 20)
    func searchBKTree(query: String, maxDistance: Int = 2, limit: Int = 20) -> [(word: String, index: Int, distance: Int)] {
        guard let root = bkTreeRoot else { return [] }
        
        var results: [(word: String, index: Int, distance: Int)] = []
        searchBKTreeNode(node: root, query: query.lowercased(), maxDistance: maxDistance, limit: limit, results: &results)
        
        return results.sorted { $0.distance < $1.distance }
    }
    
    func getWordIndex(_ word: String) -> Int? {
        return wordToIndex[word.lowercased()]
    }
    
    func getWordByIndex(_ index: Int) -> String? {
        return indexToWord[index]
    }
    
    func getAdjacentKeys(for key: Character) -> String {
        return keyboardAdjacent[String(key).lowercased()] ?? ""
    }
    
    var vocabSize: Int { wordToIndex.count }
    
    func loadChunkAsync(_ chunk: MPVocabChunk, completion: (() -> Void)? = nil) {
        if loadedChunks[chunk] != nil { completion?(); return }
        loadQueue.async { [weak self] in
            self?.loadChunk(chunk)
            DispatchQueue.main.async { completion?() }
        }
    }
    
    func preloadAllChunks(completion: (() -> Void)? = nil) {
        loadQueue.async { [weak self] in
            self?.loadChunk(.medium)
            self?.loadChunk(.low)
            DispatchQueue.main.async { completion?() }
        }
    }
    
    func releaseMemory(keepHigh: Bool = true) {
        if !keepHigh { loadedChunks.removeValue(forKey: .high) }
        loadedChunks.removeValue(forKey: .medium)
        loadedChunks.removeValue(forKey: .low)
    }
    
    // MARK: - Private Loading Methods
    
    private func loadCompactTrie() {
        guard let url = bundle.url(forResource: "compact_trie", withExtension: "json"),
              let data = try? Data(contentsOf: url, options: .mappedIfSafe),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return
        }
        compactTrie = json
    }
    
    private func loadBKTree() {
        guard let url = bundle.url(forResource: "bk_tree", withExtension: "json"),
              let data = try? Data(contentsOf: url, options: .mappedIfSafe),
              let json = try? JSONSerialization.jsonObject(with: data) as? [Any] else {
            MPLog.debug("[MPResourceLoader] loadBKTree - bk_tree.json not found")
            return
        }
        bkTreeRoot = parseBKTreeNode(json)
        MPLog.debug("[MPResourceLoader] loadBKTree - ✅ Loaded BK-Tree")
    }
    
    private func parseBKTreeNode(_ json: [Any]) -> BKTreeNode? {
        guard let wordIdx = json.first as? Int else { return nil }
        
        let node = BKTreeNode(wordIndex: wordIdx)
        
        if json.count > 1, let children = json[1] as? [String: [Any]] {
            for (distStr, childJson) in children {
                if let dist = Int(distStr), let child = parseBKTreeNode(childJson) {
                    node.children[dist] = child
                }
            }
        }
        
        return node
    }
    
    private func searchBKTreeNode(node: BKTreeNode, query: String, maxDistance: Int, limit: Int, results: inout [(word: String, index: Int, distance: Int)]) {
        // Early termination if we have enough results
        if results.count >= limit { return }
        
        guard let nodeWord = indexToWord[node.wordIndex] else { return }
        
        let dist = levenshteinDistance(query, nodeWord.lowercased())
        
        if dist <= maxDistance {
            results.append((nodeWord, node.wordIndex, dist))
        }
        
        // BK-Tree pruning: only check children with distances in range [dist - maxDistance, dist + maxDistance]
        // This is the triangle inequality optimization
        let minChildDist = max(0, dist - maxDistance)  // Fixed: was max(1, ...) which missed distance 0
        let maxChildDist = dist + maxDistance
        
        for childDist in minChildDist...maxChildDist {
            if results.count >= limit { return }
            if let child = node.children[childDist] {
                searchBKTreeNode(node: child, query: query, maxDistance: maxDistance, limit: limit, results: &results)
            }
        }
    }
    
    private func levenshteinDistance(_ s1: String, _ s2: String) -> Int {
        let m = s1.count, n = s2.count
        if m == 0 { return n }
        if n == 0 { return m }
        
        let s1 = Array(s1), s2 = Array(s2)
        var prev = Array(0...n)
        var curr = [Int](repeating: 0, count: n + 1)
        
        for i in 1...m {
            curr[0] = i
            for j in 1...n {
                let cost = s1[i-1] == s2[j-1] ? 0 : 1
                curr[j] = min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + cost)
            }
            swap(&prev, &curr)
        }
        return prev[n]
    }
    
    private func loadKeyboardAdjacent() {
        guard let url = bundle.url(forResource: "keyboard_adjacent", withExtension: "json"),
              let data = try? Data(contentsOf: url, options: .mappedIfSafe),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: String] else {
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
        guard let url = bundle.url(forResource: "word_to_index", withExtension: "json"),
              let data = try? Data(contentsOf: url, options: .mappedIfSafe),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Int] else {
            return
        }
        wordToIndex = json
        indexToWord = wordToIndex.reduce(into: [:]) { $0[$1.value] = $1.key }
    }
    
    private func loadChunk(_ chunk: MPVocabChunk) {
        guard loadedChunks[chunk] == nil, !loadingChunks.contains(chunk) else { return }
        loadingChunks.insert(chunk)
        
        guard let url = bundle.url(forResource: chunk.rawValue, withExtension: "json"),
              let data = try? Data(contentsOf: url, options: .mappedIfSafe),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let words = json["words"] as? [String] else {
            loadingChunks.remove(chunk)
            return
        }
        
        loadedChunks[chunk] = words
        loadingChunks.remove(chunk)
    }
    
    // MARK: - Trie Helpers
    
    private func collectIndicesFromNode(_ node: [String: Any], into indices: inout [Int], limit: Int) {
        if let terminalIndices = node["$"] as? [Int] {
            indices.append(contentsOf: terminalIndices)
        }
        if indices.count >= limit { return }
        
        for (key, value) in node.sorted(by: { $0.key < $1.key }) {
            if key == "$" { continue }
            if let childNode = value as? [String: Any] {
                collectIndicesFromNode(childNode, into: &indices, limit: limit)
                if indices.count >= limit { return }
            }
        }
    }
    
    private func getTrieCompletionsFallback(prefix: String, limit: Int) -> [MPTrieResult] {
        for prefixLen in stride(from: prefix.count - 1, through: 1, by: -1) {
            let shorter = String(prefix.prefix(prefixLen))
            if let trie = compactTrie {
                var node: [String: Any] = trie
                var found = true
                for char in shorter {
                    if let next = node[String(char)] as? [String: Any] {
                        node = next
                    } else { found = false; break }
                }
                if found {
                    var indices: [Int] = []
                    collectIndicesFromNode(node, into: &indices, limit: limit * 2)
                    let results = indices.compactMap { idx -> MPTrieResult? in
                        guard let word = indexToWord[idx], word.lowercased().hasPrefix(prefix) else { return nil }
                        return MPTrieResult(word: word, globalIndex: idx, chunk: getChunkForIndex(idx))
                    }
                    if !results.isEmpty { return Array(results.prefix(limit)) }
                }
            }
        }
        return searchLoadedChunks(prefix: prefix, limit: limit)
    }
    
    private func searchLoadedChunks(prefix: String, limit: Int) -> [MPTrieResult] {
        var results: [MPTrieResult] = []
        for chunk in MPVocabChunk.allCases {
            guard let words = loadedChunks[chunk] else { continue }
            for (idx, word) in words.enumerated() {
                if word.lowercased().hasPrefix(prefix) {
                    let globalIdx = wordToIndex[word.lowercased()] ?? idx
                    results.append(MPTrieResult(word: word, globalIndex: globalIdx, chunk: chunk))
                    if results.count >= limit { return results }
                }
            }
        }
        return results
    }
    
    private func getChunkForIndex(_ index: Int) -> MPVocabChunk {
        if index < 7000 { return .high }
        if index < 17000 { return .medium }
        return .low
    }
}
