# iOS Integration Guide - GRU Keyboard (2026)

**Model:** GRU Hybrid (Word-level + Edit Distance)  
**Format:** TensorFlow Lite (.tflite) OR CoreML (.mlpackage)  
**Size:** ~10-19MB  
**Latency:** <10ms (TFLite uses XNNPACK/CoreML Delegate)

---

## 📦 Model Files

**Location:** `models/gru_keyboard/`

```
gru_keyboard/
├── gru_model_fp16.tflite       # TFLite Model (Recommended for Stability) ✅
├── gru_model_optimized.tflite  # Smaller TFLite Model
├── gru_keyboard_ios.mlpackage  # CoreML Model (Requires conversion)
└── word_index.json             # Vocabulary file
```

---

## 🚀 Integration Options

### ✅ Option 1: TensorFlow Lite (Recommended)
**Best for:** Stability, guaranteed support for GRU (Flex) ops.  
**Performance:** Excellent (~8ms) using XNNPACK (default).

**1. Add Dependencies (CocoaPods)**
Add to your `Podfile`:

```ruby
# Core TFLite
pod 'TensorFlowLiteSwift', '~> 2.14.0'

# Required for GRU layers (Flex Ops)
pod 'TensorFlowLiteSelectTfOps', '~> 2.14.0'
```

**2. Add Files to Xcode**
- Drag `gru_model_fp16.tflite` to Xcode (Check "Add to targets").
- Drag `word_index.json` to Xcode.

---

## 🏗️ Swift Implementation (TFLite)

### Step 1: Vocabulary Helper
Create `Vocabulary.swift`:

```swift
import Foundation

class Vocabulary {
    private var wordToIndex: [String: Int] = [:]
    private var indexToWord: [Int: String] = [:]
    
    init() { loadVocabulary() }
    
    private func loadVocabulary() {
        guard let url = Bundle.main.url(forResource: "word_index", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Int] else {
            print("✗ Failed to load vocabulary")
            return
        }
        wordToIndex = dict
        indexToWord = Dictionary(uniqueKeysWithValues: dict.map { ($1, $0) })
        print("✓ Vocabulary loaded: \(wordToIndex.count) words")
    }
    
    func tokenize(_ text: String) -> [Int] {
        // Remove punctuation/symbols (keep layout)
        let allowed = CharacterSet.alphanumerics.union(.whitespaces)
        let cleaned = text.unicodeScalars
            .filter { allowed.contains($0) }
            .map(String.init)
            .joined()
            
        return cleaned.lowercased()
            .components(separatedBy: .whitespaces)
            .filter { !$0.isEmpty }
            .compactMap { wordToIndex[$0] ?? 1 } // 1=OOV
    }
    
    func decode(_ indices: [Int]) -> [String] {
        return indices.compactMap { indexToWord[$0] }
    }
    
    func getIndex(_ word: String) -> Int? { wordToIndex[word.lowercased()] }
    func allWords() -> [String] { Array(wordToIndex.keys) }
}
```

### Step 2: Edit Distance Helper
Create `EditDistance.swift` (Critical for typo correction):

```swift
import Foundation

class EditDistance {
    /// Calculate Levenshtein distance
    static func distance(_ s1: String, _ s2: String) -> Int {
        let s1 = Array(s1)
        let s2 = Array(s2)
        var dist = [[Int]](repeating: [Int](repeating: 0, count: s2.count + 1), count: s1.count + 1)
        
        for i in 0...s1.count { dist[i][0] = i }
        for j in 0...s2.count { dist[0][j] = j }
        
        for i in 1...s1.count {
            for j in 1...s2.count {
                if s1[i-1] == s2[j-1] {
                    dist[i][j] = dist[i-1][j-1]
                } else {
                    dist[i][j] = min(dist[i-1][j], dist[i][j-1], dist[i-1][j-1]) + 1
                }
            }
        }
        return dist[s1.count][s2.count]
    }
    
    /// Find similar words in vocabulary
    static func findSimilar(_ word: String, vocabulary: [String], maxDistance: Int = 2, topK: Int = 5) -> [(String, Int)] {
        let word = word.lowercased()
        var candidates: [(String, Int)] = []
        
        // Optimization: Iterate only top frequent words first if possible, 
        // but here we iterate all (limit if vocab > 50k)
        for vocabWord in vocabulary.prefix(15000) { 
            guard abs(vocabWord.count - word.count) <= maxDistance else { continue }
            let d = distance(word, vocabWord)
            if d <= maxDistance && d > 0 {
                candidates.append((vocabWord, d))
            }
        }
        return candidates.sorted { $0.1 < $1.1 }.prefix(topK).map { $0 }
    }
}
```

### Step 3: GRU Model Wrapper (Full Logic)
Create `GRUKeyboardModel.swift`. This matches the Python Hybrid Logic.

```swift
import Foundation
import TensorFlowLite
import TensorFlowLiteSelectTfOps // Required for Flex/Select Ops registration

class GRUKeyboardModel {
    private var interpreter: Interpreter?
    private let vocabulary = Vocabulary()
    private let sequenceLength = 10
    private let vocabSize = 25000
    
    init() {
        loadModel()
    }
    
    private func loadModel() {
        guard let modelPath = Bundle.main.path(forResource: "gru_model_fp16", ofType: "tflite") else {
            print("✗ Model not found")
            return
        }
        do {
            var options = Interpreter.Options()
            options.threadCount = 2
            interpreter = try Interpreter(modelPath: modelPath, options: options)
            try interpreter?.allocateTensors()
            print("✓ TFLite Model loaded")
        } catch {
            print("✗ Failed to load: \(error)")
        }
    }
    
    // MARK: - Main Prediction (Hybrid)
    
    /// Logic matches Python notebook:
    /// 1. Next Word: If input ends with space -> GRU
    /// 2. Completion: If partial word -> Prefix match
    /// 3. Typo: If no completion -> Edit distance
    /// 4. Rerank: If context exists -> Rerank candidates with GRU
    func predict(inputText: String, topK: Int = 3) -> [(String, Float, String)] {
        let text = inputText.lowercased()
        
        if text.hasSuffix(" ") {
            // 1. Next Word
            let context = text.trimmingCharacters(in: .whitespaces)
            return predictNextWord(context: context, topK: topK)
        }
        
        let words = text.components(separatedBy: .whitespaces)
        let partial = words.last ?? ""
        let context = words.dropLast().joined(separator: " ")
        
        // 2. Completion
        var results = completeWord(partial: partial, topK: topK)
        
        // 3. Typo Correction
        if results.isEmpty {
            let typos = correctTypo(typo: partial, topK: topK)
            results.append(contentsOf: typos)
        }
        
        // 4. GRU Reranking
        if !context.isEmpty && !results.isEmpty {
            return rerankWithGRU(candidates: results, context: context, topK: topK)
        }
        
        return results.sorted { $0.1 > $1.1 }.prefix(topK).map { $0 }
    }
    
    // MARK: - Components
    
    private func predictNextWord(context: String, topK: Int) -> [(String, Float, String)] {
        guard let interpreter = interpreter else { return [] }
        
        var tokens = vocabulary.tokenize(context)
        if tokens.count < sequenceLength {
            tokens = Array(repeating: 0, count: sequenceLength - tokens.count) + tokens
        } else {
            tokens = Array(tokens.suffix(sequenceLength))
        }
        
        let inputData = Data(copyingBufferOf: tokens.map { Float($0) })
        
        do {
            try interpreter.copy(inputData, toInputAt: 0)
            try interpreter.invoke()
            guard let outputTensor = try interpreter.output(at: 0) else { return [] }
            
            let probabilities = outputTensor.data.withUnsafeBytes {
                Array($0.bindMemory(to: Float.self))
            }
            
            let topIndices = probabilities.enumerated()
                .sorted { $0.element > $1.element }
                .prefix(topK)
            
            return topIndices.compactMap { (idx, prob) in
                guard let word = vocabulary.decode([idx]).first else { return nil }
                return (word, prob * 100, "next_word")
            }
        } catch {
            return []
        }
    }
    
    private func completeWord(partial: String, topK: Int) -> [(String, Float, String)] {
        let allWords = vocabulary.allWords()
        return allWords
            .filter { $0.hasPrefix(partial) && $0 != partial }
            .map { word -> (String, Float, String) in
                let idx = vocabulary.getIndex(word) ?? 99999
                let score = 100.0 / Float(idx + 1)
                return (word, score, "completion")
            }
            .sorted { $0.1 > $1.1 }
            .prefix(topK)
            .map { $0 }
    }
    
    private func correctTypo(typo: String, topK: Int) -> [(String, Float, String)] {
        let allWords = vocabulary.allWords()
        let similar = EditDistance.findSimilar(typo, vocabulary: allWords, maxDistance: 2, topK: topK)
        
        return similar.map { (word, dist) in
            let idx = vocabulary.getIndex(word) ?? 99999
            // Score = Distance penalty * Frequency penalty
            let score = (100.0 / Float(dist + 1)) * (100.0 / Float(idx + 1))
            return (word, score, "typo")
        }
    }
    
    private func rerankWithGRU(candidates: [(String, Float, String)], context: String, topK: Int) -> [(String, Float, String)] {
        // Get GRU probability for next word
        let gruPredictions = predictNextWord(context: context, topK: 100)
        let gruScores = Dictionary(uniqueKeysWithValues: gruPredictions.map { ($0.0, $0.1) })
        
        let reranked = candidates.map { (word, score, task) -> (String, Float, String) in
            let gruScore = gruScores[word] ?? 0
            // Hybrid Score: 30% Edit/Completion Score + 70% Context Likelihood
            let combined = score * 0.3 + gruScore * 0.7
            return (word, combined, task)
        }
        
        return reranked.sorted { $0.1 > $1.1 }.prefix(topK).map { $0 }
    }
    
    func testPredictions() {
        print("--- Testing Hybrid Logic ---")
        print("1. Next: 'how are ' -> \(predict(inputText: "how are "))")
        print("2. Comp: 'hel'      -> \(predict(inputText: "hel"))")
        print("3. Typo: 'helo'     -> \(predict(inputText: "helo"))")
        print("4. Hybr: 'how are yuo' -> \(predict(inputText: "how are yuo"))")
    }
}

// Helper
extension Data {
    init<T>(copyingBufferOf values: [T]) {
        self = values.withUnsafeBufferPointer { Data(buffer: $0) }
    }
}
```

### Step 4: Keyboard View Controller
(Standard UI implementation)

```swift
class KeyboardViewController: UIInputViewController {
    var predictor: GRUKeyboardModel!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        DispatchQueue.global().async { self.predictor = GRUKeyboardModel() }
    }
    
    override func textDidChange(_ textInput: UITextInput?) {
        guard let proxy = textDocumentProxy as? UITextDocumentProxy,
              let text = proxy.documentContextBeforeInput else { return }
        
        DispatchQueue.global(qos: .userInteractive).async {
            let suggestions = self.predictor.predict(inputText: text)
            print("Suggestions: \(suggestions)")
        }
    }
}
```

---

## ⚡ Performance Tuning (iOS specific)
1. **Threads:** `options.threadCount = 2`
2. **XNNPACK:** Enabled by default in TFLite Swift.
3. **Vocab Limit:** For `EditDistance`, only check top 15k words to keep UI responsive (<10ms).

---

## 🔧 Troubleshooting

### Error: "Select TensorFlow op(s) not supported"
If you see this error despite adding the `TensorFlowLiteSelectTfOps` pod:
1. **Ensure Import:** Make sure `import TensorFlowLiteSelectTfOps` is in your `GRUKeyboardModel.swift`.
2. **Force Load:** If using CocoaPods, the linker might strip the ops. Add the following to your target's **Build Settings** > **Other Linker Flags**:
   ```
   -force_load $(PODS_CONFIGURATION_BUILD_DIR)/TensorFlowLiteSelectTfOps/TensorFlowLiteSelectTfOps.framework/TensorFlowLiteSelectTfOps
   ```
   Or try `-all_load` (use cautiously as it increases binary size).

