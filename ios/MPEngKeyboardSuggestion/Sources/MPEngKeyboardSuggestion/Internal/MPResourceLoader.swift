// ============================================================
// MPResourceLoader.swift
// ============================================================
// Handles loading vocabulary chunks, indices, and config files
// NO SINGLETON - initialize through MPKeyboardSuggestion
// ============================================================

import Foundation

public enum MPVocabChunk: String, CaseIterable {
    case high = "vocab_high"
    case medium = "vocab_medium"
    case low = "vocab_low"
}

struct MPChunkInfo {
    let file: String
    let startIndex: Int
    let endIndex: Int
    let wordCount: Int
}

struct MPPrefixIndex {
    let version: String
    let chunks: [String: MPChunkInfo]
    let prefixes: [String: [String: [Int]]]
}

final class MPResourceLoader {
    
    private var loadedChunks: [MPVocabChunk: [String]] = [:]
    private var prefixIndex: MPPrefixIndex?
    private var soundexIndex: [String: [String: [Int]]]?
    private var keyboardAdjacent: [String: String] = [:]
    private var wordToIndex: [String: Int] = [:]
    private var indexToWord: [Int: String] = [:]
    
    private let loadQueue = DispatchQueue(label: "com.mp.keyboard.loader", qos: .userInitiated)
    private var loadingChunks = Set<MPVocabChunk>()
    private let bundle: Bundle
    
    init(bundle: Bundle = .main) {
        self.bundle = bundle
        loadPrefixIndex()
        loadKeyboardAdjacent()
        loadWordMappings()
        loadChunk(.high)
    }
    
    // MARK: - Public API
    
    func getChunkWords(_ chunk: MPVocabChunk) -> [String]? {
        if loadedChunks[chunk] == nil {
            loadChunk(chunk)
        }
        return loadedChunks[chunk]
    }
    
    func getCompletions(for prefix: String, chunks: [MPVocabChunk] = MPVocabChunk.allCases, limit: Int = 50) -> [(word: String, localIndex: Int, chunk: MPVocabChunk)] {
        guard let prefixData = prefixIndex?.prefixes[prefix.lowercased()] else {
            return searchLoadedChunks(prefix: prefix, limit: limit)
        }
        
        var results: [(word: String, localIndex: Int, chunk: MPVocabChunk)] = []
        
        for chunk in chunks {
            guard let localIndices = prefixData[chunk.rawValue.replacingOccurrences(of: "vocab_", with: "")],
                  let chunkWords = getChunkWords(chunk) else {
                continue
            }
            
            for localIdx in localIndices {
                guard localIdx < chunkWords.count else { continue }
                results.append((chunkWords[localIdx], localIdx, chunk))
                if results.count >= limit { return results }
            }
        }
        
        return results
    }
    
    func getSoundexMatches(for code: String) -> [String: [Int]]? {
        if soundexIndex == nil { loadSoundexIndex() }
        return soundexIndex?[code]
    }
    
    func getAdjacentKeys(for key: Character) -> String {
        return keyboardAdjacent[String(key).lowercased()] ?? ""
    }
    
    func getWordIndex(_ word: String) -> Int? {
        return wordToIndex[word.lowercased()]
    }
    
    func getWord(at index: Int) -> String? {
        return indexToWord[index]
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
            self?.loadSoundexIndex()
            DispatchQueue.main.async { completion?() }
        }
    }
    
    func releaseMemory(keepHigh: Bool = true) {
        if !keepHigh { loadedChunks.removeValue(forKey: .high) }
        loadedChunks.removeValue(forKey: .medium)
        loadedChunks.removeValue(forKey: .low)
        soundexIndex = nil
    }
    
    // MARK: - Private Loading
    
    private func loadPrefixIndex() {
        guard let url = bundle.url(forResource: "prefix_index", withExtension: "json"),
              let data = try? Data(contentsOf: url, options: .mappedIfSafe),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return
        }
        
        var chunks: [String: MPChunkInfo] = [:]
        if let chunksDict = json["chunks"] as? [String: [String: Any]] {
            for (name, info) in chunksDict {
                chunks[name] = MPChunkInfo(
                    file: info["file"] as? String ?? "",
                    startIndex: info["startIndex"] as? Int ?? 0,
                    endIndex: info["endIndex"] as? Int ?? 0,
                    wordCount: info["wordCount"] as? Int ?? 0
                )
            }
        }
        
        var prefixes: [String: [String: [Int]]] = [:]
        if let prefixDict = json["prefixes"] as? [String: [String: [Int]]] {
            prefixes = prefixDict
        }
        
        prefixIndex = MPPrefixIndex(
            version: json["version"] as? String ?? "3.0",
            chunks: chunks,
            prefixes: prefixes
        )
    }
    
    private func loadChunk(_ chunk: MPVocabChunk) {
        guard loadedChunks[chunk] == nil, !loadingChunks.contains(chunk) else { return }
        loadingChunks.insert(chunk)
        
        defer { loadingChunks.remove(chunk) }
        
        guard let url = bundle.url(forResource: chunk.rawValue, withExtension: "json"),
              let data = try? Data(contentsOf: url, options: .mappedIfSafe),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let words = json["words"] as? [String] else {
            return
        }
        
        loadedChunks[chunk] = words
    }
    
    private func loadSoundexIndex() {
        guard soundexIndex == nil,
              let url = bundle.url(forResource: "soundex_index", withExtension: "json"),
              let data = try? Data(contentsOf: url, options: .mappedIfSafe),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: [String: [Int]]] else {
            return
        }
        soundexIndex = json
    }
    
    private func loadKeyboardAdjacent() {
        guard let url = bundle.url(forResource: "keyboard_adjacent", withExtension: "json"),
              let data = try? Data(contentsOf: url, options: .mappedIfSafe),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: String] else {
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
        indexToWord = wordToIndex.reduce(into: [:]) { result, pair in
            result[pair.value] = pair.key
        }
    }
    
    private func searchLoadedChunks(prefix: String, limit: Int) -> [(word: String, localIndex: Int, chunk: MPVocabChunk)] {
        var results: [(word: String, localIndex: Int, chunk: MPVocabChunk)] = []
        let lowerPrefix = prefix.lowercased()
        
        for chunk in MPVocabChunk.allCases {
            guard let words = loadedChunks[chunk] else { continue }
            for (idx, word) in words.enumerated() {
                if word.lowercased().hasPrefix(lowerPrefix) {
                    results.append((word, idx, chunk))
                    if results.count >= limit { return results }
                }
            }
        }
        return results
    }
}
