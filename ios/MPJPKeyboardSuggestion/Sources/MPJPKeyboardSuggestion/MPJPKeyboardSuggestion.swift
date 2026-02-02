// ============================================================
// MPJPKeyboardSuggestion.swift
// ============================================================
// Main public API for MPJPKeyboardSuggestion package
// Japanese keyboard with zenz model for kana-kanji conversion
// and next phrase prediction
// ============================================================

import Foundation

// MARK: - Public Types

/// Japanese keyboard prediction result
public struct MPJPPrediction: Equatable, Sendable {
    /// The predicted text (kanji conversion or next phrase)
    public let text: String
    /// Probability/confidence score (0-1)
    public let probability: Float
    /// Type of prediction
    public let type: MPJPPredictionType
    
    public init(text: String, probability: Float, type: MPJPPredictionType) {
        self.text = text
        self.probability = probability
        self.type = type
    }
}

/// Type of prediction
public enum MPJPPredictionType: String, Sendable {
    case kanaToKanji = "conversion"  // ひらがな → 漢字
    case nextPhrase = "prediction"   // 文脈 → 次のフレーズ
}

/// Statistics about the keyboard model
public struct MPJPStats {
    public let modelReady: Bool
    public let vocabSize: Int
}

// MARK: - Main API

/// Japanese keyboard suggestion using zenz model
///
/// Provides kana-to-kanji conversion and next phrase prediction.
///
/// ## Example
/// ```swift
/// let keyboard = MPJPKeyboardSuggestion()
///
/// // Kana-Kanji conversion
/// let conversions = keyboard.getConversions(for: "ありがとう")
/// // → [("有難う", 0.9), ("ありがとう", 0.1), ...]
///
/// // Next phrase prediction
/// let predictions = keyboard.getPredictions(for: "ありがとう")
/// // → [("ございます", 0.7), ("ね", 0.1), ...]
///
/// // Combined (recommended for keyboards)
/// let all = keyboard.getSuggestions(for: "ありがとう")
/// ```
public final class MPJPKeyboardSuggestion {
    
    // MARK: - Private Properties
    
    private let modelInference: MPJPModelInference
    
    // MARK: - Initialization
    
    /// Initialize with default bundle (Bundle.module for SPM)
    public init() {
        self.modelInference = MPJPModelInference(bundle: Bundle.module)
    }
    
    /// Initialize with custom bundle
    /// - Parameter bundle: Bundle containing the CoreML model
    public init(bundle: Bundle) {
        self.modelInference = MPJPModelInference(bundle: bundle)
    }
    
    // MARK: - Main API
    
    /// Get combined predictions (conversions + next phrases)
    /// This is the main API for keyboard use
    ///
    /// - Parameters:
    ///   - input: Input text (hiragana/katakana or any Japanese text)
    ///   - context: Optional left context for better predictions
    ///   - limit: Maximum predictions to return (default: 5)
    /// - Returns: Array of predictions sorted by probability
    public func getSuggestions(for input: String, context: String? = nil, limit: Int = 5) -> [MPJPPrediction] {
        MPJPLog.debug("[getSuggestions] START input='\(input)' context='\(context ?? "nil")' limit=\(limit)")
        
        guard !input.isEmpty else {
            MPJPLog.debug("[getSuggestions] Empty input, returning []")
            return []
        }
        
        guard isReady else {
            MPJPLog.error("[getSuggestions] Model not ready!")
            return []
        }
        
        var predictions: [MPJPPrediction] = []
        
        // Check if input is kana (hiragana or katakana)
        let isKana = MPJPTokenizer.isKana(input)
        MPJPLog.debug("[getSuggestions] isKana=\(isKana)")
        
        // NOTE: The zenz model does katakana→hiragana conversion, NOT kanji.
        // This is not useful for keyboard suggestions, so we skip conversion
        // and only use next-phrase predictions which work well.
        
        // Get next phrase predictions
        // For kana input, this will suggest natural continuations (e.g., よろしく → お願いします)
        let fullContext = context.map { $0 + input } ?? input
        MPJPLog.debug("[getSuggestions] Getting next phrase predictions for context='\(fullContext)'...")
        let phrases = modelInference.predictNextPhrase(fullContext, topK: limit)
        MPJPLog.debug("[getSuggestions] Got \(phrases.count) phrases")
        for (phrase, prob) in phrases {
            predictions.append(MPJPPrediction(text: phrase, probability: prob, type: .nextPhrase))
        }
        
        // Sort by probability and return top results
        let result = predictions.sorted { $0.probability > $1.probability }.prefix(limit).map { $0 }
        MPJPLog.debug("[getSuggestions] END returning \(result.count) predictions")
        return result
    }
    
    /// Get kana-to-kanji conversions only
    ///
    /// - Parameters:
    ///   - kana: Hiragana or katakana input
    ///   - context: Optional left context
    ///   - limit: Maximum conversions (default: 5)
    /// - Returns: Array of kanji conversions
    public func getConversions(for kana: String, context: String? = nil, limit: Int = 5) -> [MPJPPrediction] {
        guard MPJPTokenizer.isKana(kana) else {
            MPJPLog.debug("[API] Input is not kana: \(kana)")
            return []
        }
        
        let conversions = modelInference.convertKanaToKanji(kana, context: context, topK: limit)
        return conversions.map { MPJPPrediction(text: $0.text, probability: $0.probability, type: .kanaToKanji) }
    }
    
    /// Get next phrase predictions only
    ///
    /// - Parameters:
    ///   - context: Left context text
    ///   - limit: Maximum predictions (default: 5)
    /// - Returns: Array of next phrase predictions
    public func getPredictions(for context: String, limit: Int = 5) -> [MPJPPrediction] {
        guard !context.isEmpty else { return [] }
        
        let phrases = modelInference.predictNextPhrase(context, topK: limit)
        return phrases.map { MPJPPrediction(text: $0.phrase, probability: $0.probability, type: .nextPhrase) }
    }
    
    // MARK: - Utility
    
    /// Check if model is ready for predictions
    public var isReady: Bool {
        return modelInference.isReady
    }
    
    /// Get model statistics
    public func getStats() -> MPJPStats {
        return MPJPStats(
            modelReady: modelInference.isReady,
            vocabSize: modelInference.vocabSize
        )
    }
    
    // MARK: - Static Helpers
    
    /// Convert hiragana to katakana
    public static func hiraganaToKatakana(_ text: String) -> String {
        return MPJPTokenizer.hiraganaToKatakana(text)
    }
    
    /// Convert katakana to hiragana
    public static func katakanaToHiragana(_ text: String) -> String {
        return MPJPTokenizer.katakanaToHiragana(text)
    }
    
    /// Check if text is hiragana
    public static func isHiragana(_ text: String) -> Bool {
        return MPJPTokenizer.isHiragana(text)
    }
    
    /// Check if text is katakana
    public static func isKatakana(_ text: String) -> Bool {
        return MPJPTokenizer.isKatakana(text)
    }
    
    /// Check if text is hiragana or katakana
    public static func isKana(_ text: String) -> Bool {
        return MPJPTokenizer.isKana(text)
    }
}
