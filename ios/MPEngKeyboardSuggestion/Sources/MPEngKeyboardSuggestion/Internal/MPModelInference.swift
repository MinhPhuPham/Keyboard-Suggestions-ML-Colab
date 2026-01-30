// ============================================================
// MPModelInference.swift
// ============================================================
// CoreML wrapper for GRU keyboard model inference
// NO SINGLETON - initialize through MPKeyboardSuggestion
// ============================================================

import Foundation
import CoreML

final class MPModelInference {
    
    private var model: MLModel?
    private let sequenceLength = 10
    private let loadQueue = DispatchQueue(label: "com.mp.keyboard.model", qos: .userInitiated)
    private weak var resourceLoader: MPResourceLoader?
    
    // MARK: - Prediction Cache (Optimized)
    // Cache 20 predictions per context for efficiency
    // "How " → Run model → Cache: [do, did, are, you, can...]
    // "How a" → Use cache → Filter by "a" → [are]
    // "How are" → "are" matches cache! → Pre-fetch next predictions
    private var cachedContext: String = ""
    private var cachedPredictions: [(word: String, probability: Float)] = []
    private var cachedWords: Set<String> = []  // For O(1) word lookup
    private let maxCacheSize = 20  // Only cache top 20 predictions
    
    var isReady: Bool { model != nil }
    
    init(resourceLoader: MPResourceLoader, bundle: Bundle = .main) {
        self.resourceLoader = resourceLoader
        MPLog.debug("[MPModelInference] init - Starting model load from bundle: \(bundle.bundlePath)")
        loadModel(from: bundle)
    }
    
    // MARK: - Model Loading
    
    private func loadModel(from bundle: Bundle) {
        loadQueue.async { [weak self] in
            guard let self = self else {
                MPLog.error("[MPModelInference] loadModel - self is nil")
                return
            }
            
            MPLog.debug("[MPModelInference] loadModel - Searching for model files...")
            
            // List all resources in bundle for debugging
            if let resourcePath = bundle.resourcePath {
                let files = (try? FileManager.default.contentsOfDirectory(atPath: resourcePath)) ?? []
                let modelFiles = files.filter { $0.contains("gru") || $0.contains(".mlmodel") || $0.contains(".mlpackage") }
                MPLog.debug("[MPModelInference] loadModel - Found model-related files: \(modelFiles)")
            }
            
            // Try compiled model first
            if let compiledURL = bundle.url(forResource: "gru_keyboard_ios", withExtension: "mlmodelc") {
                MPLog.debug("[MPModelInference] loadModel - Found .mlmodelc at: \(compiledURL.path)")
                do {
                    let config = MLModelConfiguration()
                    if #available(macOS 13.0, iOS 16.0, *) {
                        config.computeUnits = .cpuAndNeuralEngine
                    } else {
                        config.computeUnits = .cpuAndGPU
                    }
                    self.model = try MLModel(contentsOf: compiledURL, configuration: config)
                    MPLog.debug("[MPModelInference] loadModel - ✅ Successfully loaded .mlmodelc")
                    return
                } catch {
                    MPLog.error("[MPModelInference] loadModel - ❌ Failed to load .mlmodelc: \(error)")
                }
            } else {
                MPLog.debug("[MPModelInference] loadModel - No .mlmodelc found")
            }
            
            // Try mlpackage
            if let packageURL = bundle.url(forResource: "gru_keyboard_ios", withExtension: "mlpackage") {
                MPLog.debug("[MPModelInference] loadModel - Found .mlpackage at: \(packageURL.path)")
                do {
                    let compiledURL = try MLModel.compileModel(at: packageURL)
                    MPLog.debug("[MPModelInference] loadModel - Compiled to: \(compiledURL.path)")
                    let config = MLModelConfiguration()
                    if #available(macOS 13.0, iOS 16.0, *) {
                        config.computeUnits = .cpuAndNeuralEngine
                    } else {
                        config.computeUnits = .cpuAndGPU
                    }
                    self.model = try MLModel(contentsOf: compiledURL, configuration: config)
                    MPLog.debug("[MPModelInference] loadModel - ✅ Successfully loaded .mlpackage")
                } catch {
                    MPLog.error("[MPModelInference] loadModel - ❌ Failed to load .mlpackage: \(error)")
                }
            } else {
                MPLog.debug("[MPModelInference] loadModel - No .mlpackage found")
            }
            
            if self.model == nil {
                MPLog.error("[MPModelInference] loadModel - ❌ NO MODEL LOADED! Model files not found in bundle.")
            }
        }
    }
    
    // MARK: - Tokenization
    
    func tokenize(_ text: String) -> [Int] {
        guard let loader = resourceLoader else { return [] }
        let words = text.lowercased().split(separator: " ").map(String.init)
        return words.compactMap { loader.getWordIndex($0) }
    }
    
    // MARK: - Prediction
    
    /// Predict next word and update cache
    /// Cache is used by predictWithPrefix for efficient prefix filtering
    func predictNextWord(context: String, topK: Int = 10) -> [(word: String, probability: Float)] {
        guard let model = model, let loader = resourceLoader else { return [] }
        
        var sequence = tokenize(context)
        
        if sequence.count < sequenceLength {
            sequence = Array(repeating: 0, count: sequenceLength - sequence.count) + sequence
        } else if sequence.count > sequenceLength {
            sequence = Array(sequence.suffix(sequenceLength))
        }
        
        guard let inputArray = try? MLMultiArray(shape: [1, NSNumber(value: sequenceLength)], dataType: .int32) else {
            return []
        }
        for (i, idx) in sequence.enumerated() {
            inputArray[i] = NSNumber(value: idx)
        }
        
        let featureProvider = try? MLDictionaryFeatureProvider(dictionary: ["input": MLFeatureValue(multiArray: inputArray)])
        
        guard let provider = featureProvider,
              let output = try? model.prediction(from: provider),
              let outputArray = output.featureValue(for: "Identity")?.multiArrayValue else {
            return []
        }
        
        let vocabSize = min(outputArray.count, loader.vocabSize)
        var probs: [(Int, Float)] = []
        
        for i in 0..<vocabSize {
            let prob = outputArray[i].floatValue
            if prob > 0.001 {
                probs.append((i, prob))
            }
        }
        
        probs.sort { $0.1 > $1.1 }
        
        // Get all predictions for caching (top 20)
        let allPredictions = probs.prefix(maxCacheSize).compactMap { (idx, prob) in
            loader.getWordByIndex(idx).map { ($0, prob) }
        }
        
        // Update cache - so "How " populates cache for "How a" to use
        cachedContext = context
        cachedPredictions = allPredictions
        cachedWords = Set(allPredictions.map { $0.0.lowercased() })
        
        // Return requested topK
        return Array(allPredictions.prefix(topK))
    }
    
    /// Predict next word with prefix filter
    /// Uses cache if context matches, otherwise runs model and caches
    /// Smart pre-fetch: if typed word matches cache, get predictions for extended context
    func predictWithPrefix(context: String, prefix: String, topK: Int = 5) -> [(word: String, probability: Float)] {
        let lowerPrefix = prefix.lowercased()
        
        MPLog.debug("[predictWithPrefix] context='\(context)' prefix='\(prefix)' cachedContext='\(cachedContext)' cachedWords=\(cachedWords.prefix(5))")
        
        // Check if typed word exactly matches a cached prediction
        // "How are" where "are" was in cache for "How " → pre-fetch for "How are"
        if !lowerPrefix.isEmpty && cachedWords.contains(lowerPrefix) {
            let extendedContext = context.isEmpty ? prefix : "\(context) \(prefix)"
            MPLog.debug("[predictWithPrefix] ✅ Word '\(lowerPrefix)' matches cache! Querying model with context: '\(extendedContext)'")
            
            let predictions = predictNextWord(context: extendedContext, topK: maxCacheSize)
            MPLog.debug("[predictWithPrefix] Model predictions for '\(extendedContext)': \(predictions.prefix(10).map { $0.0 })")
            
            // Update cache for extended context
            cachedContext = extendedContext
            cachedPredictions = predictions
            cachedWords = Set(predictions.map { $0.0.lowercased() })
            
            // Keep matched word FIRST, then add ALL new predictions (no skipping)
            let matchedWordProb: Float = 1.0
            var result: [(word: String, probability: Float)] = [(prefix, matchedWordProb)]
            
            // Add new predictions (don't skip any - let the model decide)
            result += predictions.prefix(topK - 1).map { $0 }
            
            MPLog.debug("[predictWithPrefix] Final result: \(result.map { $0.0 })")
            return result
        }
        
        // If we have cached predictions for this context, use them (fast path)
        if context == cachedContext && !cachedPredictions.isEmpty {
            MPLog.debug("[predictWithPrefix] ✅ Cache HIT for context '\(context)'")
            let filtered = cachedPredictions.filter { 
                $0.word.lowercased().hasPrefix(lowerPrefix) 
            }
            MPLog.debug("[predictWithPrefix] Filtered by '\(lowerPrefix)': \(filtered.map { $0.word })")
            return Array(filtered.prefix(topK))
        }
        
        MPLog.debug("[predictWithPrefix] ❌ Cache MISS - running model")
        // Context changed or no cache - run model and cache results
        let predictions = predictNextWord(context: context, topK: maxCacheSize)
        
        // Update cache
        cachedContext = context
        cachedPredictions = predictions
        cachedWords = Set(predictions.map { $0.0.lowercased() })
        
        // Filter by prefix
        let filtered = predictions.filter { 
            $0.word.lowercased().hasPrefix(lowerPrefix) 
        }
        MPLog.debug("[predictWithPrefix] Model result filtered by '\(lowerPrefix)': \(filtered.map { $0.word })")
        return Array(filtered.prefix(topK))
    }
    
    /// Clear prediction cache (call when context changes significantly)
    func clearCache() {
        cachedContext = ""
        cachedPredictions = []
        cachedWords = []
    }
    
    func rerank(candidates: [String], context: String, originalScores: [String: Double]?, contextWeight: Double = 0.7) -> [(word: String, score: Double)] {
        guard let model = model, let loader = resourceLoader else {
            return candidates.map { ($0, originalScores?[$0] ?? 1.0) }
        }
        
        var sequence = tokenize(context)
        if sequence.count < sequenceLength {
            sequence = Array(repeating: 0, count: sequenceLength - sequence.count) + sequence
        } else {
            sequence = Array(sequence.suffix(sequenceLength))
        }
        
        guard let inputArray = try? MLMultiArray(shape: [1, NSNumber(value: sequenceLength)], dataType: .int32) else {
            return candidates.map { ($0, originalScores?[$0] ?? 1.0) }
        }
        for (i, idx) in sequence.enumerated() {
            inputArray[i] = NSNumber(value: idx)
        }
        
        guard let provider = try? MLDictionaryFeatureProvider(dictionary: ["input": MLFeatureValue(multiArray: inputArray)]),
              let output = try? model.prediction(from: provider),
              let outputArray = output.featureValue(for: "Identity")?.multiArrayValue else {
            return candidates.map { ($0, originalScores?[$0] ?? 1.0) }
        }
        
        let results: [(word: String, score: Double)] = candidates.map { word in
            let wordIdx = loader.getWordIndex(word.lowercased())
            let gruProb: Double = wordIdx.flatMap { idx in
                idx < outputArray.count ? Double(outputArray[idx].floatValue) * 100 : nil
            } ?? 0.0
            
            let origScore = originalScores?[word] ?? 1.0
            let combined = origScore * (1 - contextWeight) + gruProb * contextWeight
            return (word, combined)
        }
        
        return results.sorted { $0.score > $1.score }
    }
}
