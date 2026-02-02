// ============================================================
// MPJPModelInference.swift
// ============================================================
// CoreML inference handler for zenz GPT-2 model
// Optimized for fast keyboard suggestions
// ============================================================

import CoreML
import Foundation

final class MPJPModelInference {
    
    // MARK: - Properties
    
    private var model: MLModel?
    private let tokenizer = MPJPTokenizer()
    private(set) var vocabSize: Int = 0
    private(set) var isReady: Bool = false
    
    /// Max tokens to generate per candidate
    private let maxGenerationLength = 6  // Reduced for lower CPU usage
    
    private let loadQueue = DispatchQueue(label: "mpjp.model.load", qos: .userInitiated)
    
    // MARK: - Initialization
    
    init(bundle: Bundle) {
        loadModel(from: bundle)
    }
    
    private func loadModel(from bundle: Bundle) {
        loadQueue.async { [weak self] in
            guard let self = self else { return }
            
            let startTime = CFAbsoluteTimeGetCurrent()
            MPJPLog.info("[Model] Loading zenz model from bundle...")
            
            // Load vocab.json first (optimized: async-friendly)
            if let vocabURL = bundle.url(forResource: "vocab", withExtension: "json") {
                if self.tokenizer.loadVocab(from: vocabURL) {
                    self.vocabSize = self.tokenizer.vocabSize
                    MPJPLog.info("[Model] ✅ Loaded vocab: \(self.vocabSize) tokens")
                } else {
                    MPJPLog.error("[Model] Failed to load vocab.json, using default")
                    self.tokenizer.buildDefaultVocab()
                }
            } else {
                MPJPLog.error("[Model] vocab.json not found in bundle, using default")
                self.tokenizer.buildDefaultVocab()
            }
            
            // Load BPE merges (skip if not critical for inference)
            if let mergesURL = bundle.url(forResource: "merges", withExtension: "txt") {
                if self.tokenizer.loadMerges(from: mergesURL) {
                    MPJPLog.info("[Model] ✅ Loaded BPE merges")
                } else {
                    MPJPLog.warn("[Model] Failed to load merges.txt (BPE disabled)")
                }
            } else {
                MPJPLog.warn("[Model] merges.txt not found (BPE disabled)")
            }
            
            // Configure model for optimal performance
            let config = MLModelConfiguration()
            if #available(macOS 13.0, iOS 16.0, *) {
                config.computeUnits = .cpuAndNeuralEngine  // Use Neural Engine!
            } else {
                config.computeUnits = .cpuAndGPU
            }
            
            // Try compiled model first (.mlmodelc) - FASTEST
            if let compiledURL = bundle.url(forResource: "zenz-v3_1-xsmall_coreml", withExtension: "mlmodelc") {
                MPJPLog.debug("[Model] Found .mlmodelc at: \(compiledURL.path)")
                do {
                    self.model = try MLModel(contentsOf: compiledURL, configuration: config)
                    self.isReady = true
                    let loadTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
                    MPJPLog.info("[Model] ✅ Loaded compiled model in \(String(format: "%.0f", loadTime))ms")
                    return
                } catch {
                    MPJPLog.error("[Model] Failed to load .mlmodelc: \(error)")
                }
            }
            
            // Try mlpackage (slower - needs runtime compilation)
            if let packageURL = bundle.url(forResource: "zenz-v3_1-xsmall_coreml", withExtension: "mlpackage") {
                MPJPLog.debug("[Model] Found .mlpackage at: \(packageURL.path)")
                do {
                    let compileStart = CFAbsoluteTimeGetCurrent()
                    let compiledURL = try MLModel.compileModel(at: packageURL)
                    let compileTime = (CFAbsoluteTimeGetCurrent() - compileStart) * 1000
                    MPJPLog.debug("[Model] Compiled model in \(String(format: "%.0f", compileTime))ms")
                    
                    self.model = try MLModel(contentsOf: compiledURL, configuration: config)
                    self.isReady = true
                    let totalTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
                    MPJPLog.info("[Model] ✅ Compiled and loaded model in \(String(format: "%.0f", totalTime))ms")
                    return
                } catch {
                    MPJPLog.error("[Model] Failed to compile .mlpackage: \(error)")
                }
            }
            
            MPJPLog.error("[Model] ❌ No model found in bundle")
        }
    }
    
    // MARK: - Kana to Kanji Conversion (Optimized)
    
    /// Convert kana to kanji - FAST version
    /// Only get first-token predictions without generation loop
    func convertKanaToKanji(_ kana: String, context: String? = nil, topK: Int = 5) -> [(text: String, probability: Float)] {
        guard isReady, let _ = model else {
            MPJPLog.error("[Model] Model not ready")
            return []
        }
        
        let prompt = tokenizer.formatConversionPrompt(kana, context: context)
        let inputIds = tokenizer.encode(prompt)
        
        MPJPLog.debug("[Convert] Input: \(kana) -> Prompt: \(prompt)")
        
        // Single model call - get logits for prompt
        guard let probs = getNextTokenProbabilities(inputIds: inputIds) else {
            return []
        }
        
        // Get top-k first tokens (sorted by probability)
        let topTokens = probs.sorted { $0.value > $1.value }.prefix(topK * 3)
        
        var candidates: [(text: String, probability: Float)] = []
        
        for (tokenId, prob) in topTokens {
            guard let token = tokenizer.getToken(tokenId) else { continue }
            if isSpecialToken(token) { continue }
            
            // Decode single token to readable text
            let decodedText = tokenizer.decode([tokenId])
            if decodedText.isEmpty { continue }
            
            // Check if already have this candidate
            if candidates.contains(where: { $0.text == decodedText }) { continue }
            
            candidates.append((decodedText, prob))
            MPJPLog.debug("[Convert] Candidate: '\(decodedText)' prob=\(prob)")
            
            if candidates.count >= topK { break }
        }
        
        // Optionally extend best candidates with more tokens
        if candidates.count > 0 {
            candidates = extendCandidates(candidates, inputIds: inputIds, maxExtensions: 2)
        }
        
        MPJPLog.debug("[Convert] Final \(candidates.count) candidates")
        return candidates
    }
    
    // MARK: - Next Phrase Prediction (Optimized)
    
    /// Predict next phrase - FAST version
    func predictNextPhrase(_ context: String, topK: Int = 5) -> [(phrase: String, probability: Float)] {
        guard isReady, let _ = model else {
            MPJPLog.error("[Model] Model not ready")
            return []
        }
        
        let prompt = tokenizer.formatPredictionPrompt(context)
        let inputIds = tokenizer.encode(prompt)
        
        MPJPLog.debug("[Predict] Context: \(context) -> Prompt: \(prompt)")
        
        // Single model call
        guard let probs = getNextTokenProbabilities(inputIds: inputIds) else {
            return []
        }
        
        // Get top-k first tokens
        let topTokens = probs.sorted { $0.value > $1.value }.prefix(topK * 3)
        
        var candidates: [(phrase: String, probability: Float)] = []
        
        for (tokenId, prob) in topTokens {
            guard let token = tokenizer.getToken(tokenId) else { continue }
            if isSpecialToken(token) { continue }
            
            let decodedPhrase = tokenizer.decode([tokenId])
            if decodedPhrase.isEmpty { continue }
            if candidates.contains(where: { $0.phrase == decodedPhrase }) { continue }
            
            candidates.append((decodedPhrase, prob))
            MPJPLog.debug("[Predict] Candidate: '\(decodedPhrase)' prob=\(prob)")
            
            if candidates.count >= topK { break }
        }
        
        // Extend with more tokens for better phrases (limit extensions for CPU)
        if candidates.count > 0 {
            candidates = extendPhraseCandidates(candidates, inputIds: inputIds, maxExtensions: 3)  // Reduced from 6
        }
        
        MPJPLog.debug("[Predict] Final \(candidates.count) phrases")
        return candidates
    }
    
    // MARK: - Extension Helpers
    
    /// Extend top candidates with a few more tokens (limited calls)
    private func extendCandidates(_ candidates: [(text: String, probability: Float)], inputIds: [Int], maxExtensions: Int) -> [(text: String, probability: Float)] {
        
        var extended: [(text: String, probability: Float)] = []
        
        // Only extend top 2 candidates to save CPU/memory
        for (text, prob) in candidates.prefix(2) {
            let tokenIds = tokenizer.encode(text)
            var currentIds = inputIds + tokenIds
            var currentProb = prob
            
            for _ in 0..<maxExtensions {
                guard let nextProbs = getNextTokenProbabilities(inputIds: currentIds) else { break }
                guard let (nextId, nextProb) = nextProbs.max(by: { $0.value < $1.value }) else { break }
                guard let nextToken = tokenizer.getToken(nextId) else { break }
                
                // Stop at special tokens, ZENZ markers, punctuation
                if isZenzMarkerToken(nextId) { break }
                if isSpecialToken(nextToken) || nextToken == MPJPTokenizer.ZENZ_EOS { break }
                if isPunctuation(nextToken) { break }
                
                currentIds.append(nextId)
                currentProb *= nextProb
            }
            
            let fullText = tokenizer.decode(currentIds.suffix(currentIds.count - inputIds.count).map { $0 })
            if !fullText.isEmpty {
                extended.append((fullText, currentProb))
            }
        }
        
        // Add remaining original candidates
        for candidate in candidates.dropFirst(3) {
            extended.append(candidate)
        }
        
        return extended.sorted { $0.probability > $1.probability }
    }
    
    /// Extend phrase candidates
    private func extendPhraseCandidates(_ candidates: [(phrase: String, probability: Float)], inputIds: [Int], maxExtensions: Int) -> [(phrase: String, probability: Float)] {
        
        var extended: [(phrase: String, probability: Float)] = []
                // Only extend top 2 candidates for CPU/memory efficiency
        for (phrase, prob) in candidates.prefix(2) {
            let tokenIds = tokenizer.encode(phrase)
            var currentIds = inputIds + tokenIds
            var currentProb = prob
            
            for _ in 0..<maxExtensions {
                guard let nextProbs = getNextTokenProbabilities(inputIds: currentIds) else { break }
                guard let (nextId, nextProb) = nextProbs.max(by: { $0.value < $1.value }) else { break }
                guard let nextToken = tokenizer.getToken(nextId) else { break }
                
                // Stop at special tokens, ZENZ markers, punctuation
                if isZenzMarkerToken(nextId) { break }
                if isSpecialToken(nextToken) || nextToken == MPJPTokenizer.ZENZ_EOS { break }
                if isPunctuation(nextToken) { break }
                
                currentIds.append(nextId)
                currentProb *= nextProb
            }
            
            let fullPhrase = tokenizer.decode(currentIds.suffix(currentIds.count - inputIds.count).map { $0 })
            if !fullPhrase.isEmpty {
                extended.append((fullPhrase, currentProb))
            }
        }
        
        for candidate in candidates.dropFirst(3) {
            extended.append(candidate)
        }
        
        return extended.sorted { $0.probability > $1.probability }
    }
    
    // MARK: - Model Inference
    
    /// Get next token probabilities (softmax)
    private func getNextTokenProbabilities(inputIds: [Int]) -> [Int: Float]? {
        guard let model = model else { return nil }
        
        do {
            // Create input array
            let inputArray = try MLMultiArray(shape: [1, NSNumber(value: inputIds.count)], dataType: .int32)
            for (i, id) in inputIds.enumerated() {
                inputArray[i] = NSNumber(value: id)
            }
            
            // Run inference
            let input = try MLDictionaryFeatureProvider(dictionary: ["input_ids": inputArray])
            let output = try model.prediction(from: input)
            
            // Get logits
            guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
                MPJPLog.error("[Model] No logits in output")
                return nil
            }
            
            // Extract probabilities for last token position
            let seqLength = logits.shape[1].intValue
            let vocabSize = logits.shape[2].intValue
            let lastPosition = seqLength - 1
            
            // Apply softmax and collect top tokens
            var maxLogit: Float = -.infinity
            var probs: [Int: Float] = [:]
            
            // Find max for numerical stability
            for v in 0..<vocabSize {
                let idx = lastPosition * vocabSize + v
                let logit = logits[idx].floatValue
                if logit > maxLogit { maxLogit = logit }
            }
            
            // Compute softmax
            var sumExp: Float = 0
            for v in 0..<vocabSize {
                let idx = lastPosition * vocabSize + v
                let logit = logits[idx].floatValue
                sumExp += exp(logit - maxLogit)
            }
            
            // Get top-k (only store high probability tokens to save memory)
            let threshold: Float = 0.001  // Only keep tokens with >0.1% probability
            for v in 0..<vocabSize {
                let idx = lastPosition * vocabSize + v
                let logit = logits[idx].floatValue
                let prob = exp(logit - maxLogit) / sumExp
                if prob > threshold {
                    probs[v] = prob
                }
            }
            
            return probs
            
        } catch {
            MPJPLog.error("[Model] Inference failed: \(error)")
            return nil
        }
    }
    
    // MARK: - Token Validation
    
    /// Check if token ID is part of ZENZ special markers (byte-level tokens)
    /// ZENZ markers \uEE00-\uEE03 encode to byte tokens starting with 172, 120
    private func isZenzMarkerToken(_ tokenId: Int) -> Bool {
        // Token 172 is 'î' byte (0xEE first byte of \uEE00-\uEE03)
        // Token 120 is '¸' byte (0xB8 second byte)
        // Tokens 202-205 are the third bytes for EE00-EE03
        return tokenId == 172 || tokenId == 120 || 
               (tokenId >= 202 && tokenId <= 205)
    }
    
    private func isSpecialToken(_ token: String) -> Bool {
        return token.hasPrefix("<") || token.hasPrefix("[") ||
               token.hasPrefix("\u{EE00}") || token.hasPrefix("\u{EE01}") || 
               token.hasPrefix("\u{EE02}") || token.hasPrefix("\u{EE03}") ||
               token == "î" || token == "¸" || token == "Ģ"  // Byte tokens for ZENZ markers
    }
    
    private func isPunctuation(_ token: String) -> Bool {
        let punctuation = Set(["。", "、", "！", "？", "…", "「", "」", "『", "』", "（", "）", "・"])
        return punctuation.contains(token) || token.unicodeScalars.allSatisfy { CharacterSet.punctuationCharacters.contains($0) }
    }
}
