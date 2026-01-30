// ============================================================
// MPKeyboardSuggestion.swift
// ============================================================
// Main public API for MPEngKeyboardSuggestion package
// NO INTERNAL SINGLETONS - user manages singleton if needed
// ============================================================

import Foundation
import CoreML

// MARK: - Public Models

/// Suggestion result with metadata
public struct MPSuggestion: Equatable, Sendable {
    public let word: String
    public let score: Double
    public let source: MPSuggestionSource
    
    public init(word: String, score: Double, source: MPSuggestionSource) {
        self.word = word
        self.score = score
        self.source = source
    }
}

/// Source of suggestion
public enum MPSuggestionSource: String, Sendable {
    case gru = "gru"
    case trie = "trie"
    case learning = "learning"
    case typo = "typo"
    case shortcut = "shortcut"
    case hybrid = "hybrid"
}

/// System statistics
public struct MPStats: Sendable {
    public let vocabSize: Int
    public let learnedWords: Int
    public let learnedBigrams: Int
    public let shortcuts: Int
    public let gruReady: Bool
}

// MARK: - Main API

/// High-performance English keyboard suggestion library
/// 
/// Uses hybrid ML + Trie approach for context-aware suggestions.
/// NO SINGLETON - create your own instance or manage singleton externally.
///
/// ## Example
/// ```swift
/// let keyboard = MPKeyboardSuggestion()
/// let suggestions = keyboard.getSuggestions(for: "how are y")
/// keyboard.recordSelection("you", context: "how are")
/// ```
public final class MPKeyboardSuggestion {
    
    // MARK: - Internal Components (owned, not singletons)
    
    private let resourceLoader: MPResourceLoader
    private let modelInference: MPModelInference
    private let trieHelper: MPTrieHelper
    private let learningManager: MPLearningManager
    private let typoCorrector: MPTypoCorrector
    private let shortcutManager: MPShortcutManager
    
    // MARK: - Configuration
    
    /// Weight for GRU model in reranking (0-1). Default: 0.7
    public var gruWeight: Double = 0.7
    
    /// Enable/disable GRU model. Default: true
    public var useGRU: Bool = true
    
    /// Enable/disable learning. Default: true
    public var useLearning: Bool = true
    
    // MARK: - Initialization
    
    /// Initialize keyboard suggestion engine
    /// - Parameters:
    ///   - bundle: Bundle containing resource files. Default uses package's Bundle.module.
    ///             Pass Bundle.main if resources are in your app bundle.
    ///   - storageDirectory: Directory for persisted data (default: Documents)
    public init(bundle: Bundle? = nil, storageDirectory: URL? = nil) {
        // Use Bundle.module for Swift Package resources, or provided bundle
        let resourceBundle = bundle ?? Bundle.module
        
        MPLog.debug("[MPKeyboardSuggestion] init - Starting with bundle: \(resourceBundle.bundlePath)")
        
        resourceLoader = MPResourceLoader(bundle: resourceBundle)
        modelInference = MPModelInference(resourceLoader: resourceLoader, bundle: resourceBundle)
        trieHelper = MPTrieHelper(resourceLoader: resourceLoader)
        learningManager = MPLearningManager(storageDirectory: storageDirectory)
        typoCorrector = MPTypoCorrector(resourceLoader: resourceLoader)
        shortcutManager = MPShortcutManager(storageDirectory: storageDirectory)
        
        MPLog.debug("[MPKeyboardSuggestion] init - Components initialized")
        MPLog.debug("[MPKeyboardSuggestion] init - vocabSize=\(resourceLoader.vocabSize), gruReady=\(modelInference.isReady)")
    }
    
    // MARK: - Main API
    
    /// Get suggestions for input text
    /// - Parameters:
    ///   - input: Current input text
    ///   - limit: Maximum suggestions to return (default: 5)
    /// - Returns: Array of suggestions sorted by relevance
    public func getSuggestions(for input: String, limit: Int = 5) -> [MPSuggestion] {
        guard !input.isEmpty else { return [] }
        
        // Next-word prediction (ends with space)
        if input.hasSuffix(" ") {
            return getNextWordSuggestions(context: input.trimmingCharacters(in: .whitespaces), limit: limit)
        }
        
        let words = input.split(separator: " ").map(String.init)
        let lastWord = words.last ?? input
        let context = words.dropLast().joined(separator: " ")
        
        if let shortcutSuggestions = getShortcutSuggestions(for: lastWord, limit: limit) {
            return shortcutSuggestions
        }
        
        if typoCorrector.isLikelyTypo(lastWord) {
            return getTypoSuggestions(typo: lastWord, context: context, limit: limit)
        }
        
        return getCompletionSuggestions(prefix: lastWord, context: context, limit: limit)
    }
    
    /// Get suggestions as simple string array
    public func getSuggestionWords(for input: String, limit: Int = 5) -> [String] {
        return getSuggestions(for: input, limit: limit).map { $0.word }
    }
    
    /// Record user selection for learning
    public func recordSelection(_ word: String, context: String? = nil) {
        learningManager.recordSelection(word, context: context)
    }
    
    // MARK: - Shortcuts API
    
    public func addShortcut(_ input: String, _ suggestion: String) {
        shortcutManager.addShortcut(input, suggestion)
    }
    
    public func addShortcuts(_ dict: [String: String]) {
        for (input, suggestion) in dict {
            shortcutManager.addShortcut(input, suggestion)
        }
    }
    
    public func removeShortcut(_ input: String) {
        shortcutManager.removeShortcut(input)
    }
    
    public func getAllShortcuts() -> [String: [String]] {
        return shortcutManager.getAllShortcuts()
    }
    
    // MARK: - Resource Management
    
    public func preload(completion: (() -> Void)? = nil) {
        resourceLoader.preloadAllChunks(completion: completion)
    }
    
    public func releaseMemory() {
        resourceLoader.releaseMemory()
    }
    
    public func clearLearningData() {
        learningManager.clear()
    }
    
    public func getStats() -> MPStats {
        let learningStats = learningManager.getStats()
        return MPStats(
            vocabSize: resourceLoader.vocabSize,
            learnedWords: learningStats.words,
            learnedBigrams: learningStats.bigrams,
            shortcuts: shortcutManager.count,
            gruReady: modelInference.isReady
        )
    }
    
    // MARK: - Private Methods
    
    private func getNextWordSuggestions(context: String, limit: Int) -> [MPSuggestion] {
        var candidates: [(word: String, score: Double, source: MPSuggestionSource)] = []
        
        if useGRU && modelInference.isReady {
            let gruPredictions = modelInference.predictNextWord(context: context, topK: limit * 2)
            for (word, prob) in gruPredictions {
                candidates.append((word, Double(prob) * 100, .gru))
            }
        }
        
        if useLearning {
            let contextWord = context.split(separator: " ").last.map(String.init) ?? ""
            let bigramSuggestions = learningManager.getBigramSuggestions(for: contextWord, limit: 5)
            for (word, count) in bigramSuggestions {
                candidates.append((word, Double(count) * 50, .learning))
            }
        }
        
        return combineCandidates(candidates, context: context, limit: limit)
    }
    
    private func getCompletionSuggestions(prefix: String, context: String, limit: Int) -> [MPSuggestion] {
        var candidates: [(word: String, score: Double, source: MPSuggestionSource)] = []
        let lowerPrefix = prefix.lowercased()
        
        // STRATEGY: Get GRU next-word predictions filtered by user's prefix
        // Example: "How d" -> GRU predicts "do, did, don't" from context "How", filtered by "d"
        // GRU predictions are CONTEXT-AWARE and should be prioritized!
        if useGRU && modelInference.isReady && !context.isEmpty {
            // Get predictions from cache or model, filtered by prefix
            let gruFiltered = modelInference.predictWithPrefix(context: context, prefix: lowerPrefix, topK: limit * 3)
            for (word, prob) in gruFiltered {
                // GRU + prefix match = HIGHEST priority
                // Add 200 point CONTEXT BONUS + multiplied probability
                // This ensures GRU beats trie (max ~120) when context matches
                let contextBonus: Double = 200.0
                let probScore = Double(prob) * 500
                candidates.append((word, contextBonus + probScore, .gru))
            }
            MPLog.debug("[getCompletionSuggestions] GRU candidates: \(candidates.map { ($0.word, $0.score) })")
        }
        
        // Trie completions as backup (already filtered during export - no nonsense words)
        let trieResults = trieHelper.searchPrefix(lowerPrefix, limit: limit * 2)
        for (word, score) in trieResults {
            // Skip single letters (except "a" and "i") - they should come from GRU context
            if word.count == 1 && word.lowercased() != "a" && word.lowercased() != "i" {
                continue
            }
            // If we have GRU candidates with same word, skip trie version
            if candidates.contains(where: { $0.word.lowercased() == word.lowercased() }) {
                continue
            }
            candidates.append((word, score, .trie))
        }
        
        // Learning-based suggestions with prefix filter
        if useLearning {
            let contextWord = context.split(separator: " ").last.map(String.init) ?? ""
            let bigramSuggestions = learningManager.getBigramSuggestions(for: contextWord, limit: 10)
            for (word, count) in bigramSuggestions {
                if word.lowercased().hasPrefix(lowerPrefix) {
                    candidates.append((word, Double(count) * 30, .learning))
                }
            }
        }
        
        return combineCandidates(candidates, context: context, limit: limit)
    }
    
    private func getTypoSuggestions(typo: String, context: String, limit: Int) -> [MPSuggestion] {
        let corrections = typoCorrector.getCorrections(for: typo, limit: limit * 2)
        if corrections.isEmpty { return [] }
        
        if useGRU && modelInference.isReady && !context.isEmpty {
            let words = corrections.map { $0.word }
            let originalScores = Dictionary(uniqueKeysWithValues: corrections.map { ($0.word, $0.score) })
            let reranked = modelInference.rerank(candidates: words, context: context, originalScores: originalScores, contextWeight: gruWeight)
            
            return reranked.prefix(limit).map { MPSuggestion(word: $0.word, score: $0.score, source: .typo) }
        }
        
        return corrections.prefix(limit).map { MPSuggestion(word: $0.word, score: $0.score, source: .typo) }
    }
    
    private func getShortcutSuggestions(for input: String, limit: Int) -> [MPSuggestion]? {
        if let suggestions = shortcutManager.getSuggestions(for: input) {
            return suggestions.prefix(limit).enumerated().map {
                MPSuggestion(word: $0.element, score: 1000 - Double($0.offset), source: .shortcut)
            }
        }
        
        let partials = shortcutManager.getPartialMatches(for: input)
        if !partials.isEmpty {
            var results: [MPSuggestion] = []
            for partial in partials.prefix(2) {
                for suggestion in partial.suggestions.prefix(2) {
                    results.append(MPSuggestion(word: suggestion, score: 500, source: .shortcut))
                }
            }
            if !results.isEmpty { return results }
        }
        
        return nil
    }
    
    private func combineCandidates(_ candidates: [(word: String, score: Double, source: MPSuggestionSource)], context: String, limit: Int) -> [MPSuggestion] {
        var scoreMap: [String: (score: Double, source: MPSuggestionSource)] = [:]
        
        for (word, score, source) in candidates {
            let key = word.lowercased()
            var finalScore = score
            
            if useLearning {
                finalScore += learningManager.getBoostScore(for: word, context: context)
            }
            
            if let existing = scoreMap[key] {
                scoreMap[key] = (existing.score + finalScore, .hybrid)
            } else {
                scoreMap[key] = (finalScore, source)
            }
        }
        
        let sorted = scoreMap.sorted { $0.value.score > $1.value.score }
        return sorted.prefix(limit).map {
            MPSuggestion(word: $0.key, score: $0.value.score, source: $0.value.source)
        }
    }
}
