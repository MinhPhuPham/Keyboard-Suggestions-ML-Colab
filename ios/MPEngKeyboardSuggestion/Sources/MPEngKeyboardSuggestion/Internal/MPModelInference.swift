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
    
    var isReady: Bool { model != nil }
    
    init(resourceLoader: MPResourceLoader, bundle: Bundle = .main) {
        self.resourceLoader = resourceLoader
        loadModel(from: bundle)
    }
    
    // MARK: - Model Loading
    
    private func loadModel(from bundle: Bundle) {
        loadQueue.async { [weak self] in
            guard let self = self else { return }
            
            // Try compiled model first
            if let compiledURL = bundle.url(forResource: "gru_keyboard_ios", withExtension: "mlmodelc") {
                do {
                    let config = MLModelConfiguration()
                    config.computeUnits = .cpuAndNeuralEngine
                    self.model = try MLModel(contentsOf: compiledURL, configuration: config)
                    return
                } catch { }
            }
            
            // Try mlpackage
            if let packageURL = bundle.url(forResource: "gru_keyboard_ios", withExtension: "mlpackage") {
                do {
                    let compiledURL = try MLModel.compileModel(at: packageURL)
                    let config = MLModelConfiguration()
                    config.computeUnits = .cpuAndNeuralEngine
                    self.model = try MLModel(contentsOf: compiledURL, configuration: config)
                } catch { }
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
              let outputArray = output.featureValue(for: "output")?.multiArrayValue else {
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
        
        return probs.prefix(topK).compactMap { (idx, prob) in
            loader.getWord(at: idx).map { ($0, prob) }
        }
    }
    
    func predictWithPrefix(context: String, prefix: String, topK: Int = 5) -> [(word: String, probability: Float)] {
        let predictions = predictNextWord(context: context, topK: topK * 40)
        let lowerPrefix = prefix.lowercased()
        return predictions.filter { $0.word.lowercased().hasPrefix(lowerPrefix) }.prefix(topK).map { $0 }
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
              let outputArray = output.featureValue(for: "output")?.multiArrayValue else {
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
