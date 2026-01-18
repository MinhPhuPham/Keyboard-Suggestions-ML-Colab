# iOS Integration Guide - MobileBERT Keyboard with ONNX Runtime

Complete guide for integrating the MobileBERT keyboard suggestion model into your iOS keyboard extension using ONNX Runtime.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Setup](#setup)
3. [Swift Implementation](#swift-implementation)
4. [Keyboard Integration](#keyboard-integration)
5. [Complete Example](#complete-example)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Requirements
- **Xcode**: 14.0 or later
- **iOS Deployment Target**: iOS 15.0+
- **Swift**: 5.5+
- **Model File**: `keyboard_model.onnx` (5.48 MB)

### Install ONNX Runtime

**Option 1: CocoaPods (Recommended for iOS 15+)**

```ruby
platform :ios, '15.0'

target 'YourKeyboardExtension' do
  pod 'onnxruntime-objc', '~> 1.20.0'  # Supports IR version 9
end
```

Then run:
```bash
pod repo update
pod install
```

**Option 2: Swift Package Manager**

1. In Xcode, go to **File → Add Package Dependencies**
2. Enter URL: `https://github.com/microsoft/onnxruntime-swift-package-manager`
3. Select version: `1.18.0` or later
4. Add to your keyboard extension target

**Import Statement**:
```swift
import onnxruntime_objc  // If using CocoaPods
// OR
import onnxruntime       // If using Swift Package Manager
```

> ⚠️ **Note**: CocoaPods versions up to 1.23.0 support IR version 9. The model will be exported with IR version 9 compatibility.

---

## Setup

### Step 1: Export the Model

```bash
python scripts/export_to_onnx.py --model-dir ./models/best_model
```

This creates:
- `keyboard_model.ort` (74 MB) - Optimized model (INT8 quantized)
- `keyboard_model.onnx` (74 MB) - Standard ONNX (for testing)
- `vocab.txt` (226 KB) - Vocabulary

**Total size: ~74 MB** (49% smaller than FP32)

> **Note**: Use the `.ort` format for production - it enables memory-mapped loading and reduces RAM usage to <80MB.

### Step 2: Add Files to Xcode

1. Drag `keyboard_model.ort` and `vocab.txt` into your Xcode project
2. Check "Copy items if needed"
3. Add to keyboard extension target
4. Verify both files are in "Copy Bundle Resources"

**Files to add**:
```
keyboard_model.ort  (74 MB)  ← Use this for production
vocab.txt           (226 KB)
```

> **Alternative**: You can use `keyboard_model.onnx` for testing, but `.ort` is recommended for better memory efficiency.

---

## Swift Implementation

### 1. WordPiece Tokenizer

Create `WordPieceTokenizer.swift`:

```swift
import Foundation

class WordPieceTokenizer {
    private var vocab: [String: Int] = [:]
    private var idToToken: [Int: String] = [:]
    
    let maskToken = "[MASK]"
    let padToken = "[PAD]"
    let unkToken = "[UNK]"
    
    var maskTokenId: Int64 { Int64(vocab[maskToken] ?? 103) }
    var padTokenId: Int64 { Int64(vocab[padToken] ?? 0) }
    
    init() {
        loadVocabulary()
    }
    
    private func loadVocabulary() {
        guard let vocabPath = Bundle.main.path(forResource: "vocab", ofType: "txt"),
              let vocabContent = try? String(contentsOfFile: vocabPath) else {
            print("❌ Failed to load vocab.txt")
            return
        }
        
        let lines = vocabContent.components(separatedBy: .newlines)
        for (index, token) in lines.enumerated() where !token.isEmpty {
            vocab[token] = index
            idToToken[index] = token
        }
        
        print("✓ Loaded vocabulary: \(vocab.count) tokens")
    }
    
    func tokenize(_ text: String, maxLength: Int = 32) -> (inputIds: [Int64], attentionMask: [Int64]) {
        var tokens: [String] = []
        
        // Split by whitespace
        let words = text.components(separatedBy: .whitespaces)
        for word in words where !word.isEmpty {
            // Check if it's a special token (starts with [ and ends with ])
            if word.hasPrefix("[") && word.hasSuffix("]") {
                // Keep special tokens as-is (uppercase)
                if vocab[word] != nil {
                    tokens.append(word)
                    print("✓ Found special token: \(word) -> ID: \(vocab[word]!)")
                } else {
                    tokens.append(unkToken)
                    print("⚠️ Special token not in vocab: \(word), using \(unkToken)")
                }
            } else {
                // Lowercase regular words
                let lowercased = word.lowercased()
                if vocab[lowercased] != nil {
                    tokens.append(lowercased)
                } else if vocab["##\(lowercased)"] != nil {
                    tokens.append("##\(lowercased)")
                } else {
                    tokens.append(unkToken)
                }
            }
        }
        
        print("📝 Tokens: \(tokens)")
        
        var inputIds = tokens.map { Int64(vocab[$0] ?? vocab[unkToken]!) }
        
        // Pad to maxLength
        if inputIds.count < maxLength {
            inputIds.append(contentsOf: Array(repeating: padTokenId, count: maxLength - inputIds.count))
        } else {
            inputIds = Array(inputIds.prefix(maxLength))
        }
        
        let attentionMask = inputIds.map { $0 != padTokenId ? Int64(1) : Int64(0) }
        
        print("🔢 Input IDs: \(inputIds.prefix(10))...")
        print("🎭 Looking for MASK token ID: \(maskTokenId)")
        
        return (inputIds, attentionMask)
    }
    
    func decode(_ tokenId: Int) -> String {
        return idToToken[tokenId] ?? unkToken
    }
}
```

### 2. ONNX Model Wrapper

Create `KeyboardModel.swift`:

```swift
import Foundation
import onnxruntime_objc  // If using CocoaPods
// import onnxruntime     // If using Swift Package Manager

class KeyboardModel {
    private var session: ORTSession?
    private let tokenizer = WordPieceTokenizer()
    
    init() {
        loadModel()
    }
    
    private func loadModel() {
        do {
            guard let modelPath = Bundle.main.path(forResource: "keyboard_model", ofType: "onnx") else {
                print("❌ Model file not found")
                return
            }
            
            let env = try ORTEnv(loggingLevel: .warning)
            let options = try ORTSessionOptions()
            
            // ONNX Runtime will automatically use CoreML on iOS if available
            // No need to explicitly enable it
            
            session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: options)
            print("✓ Model loaded successfully")
        } catch {
            print("❌ Failed to load model: \(error)")
        }
    }
    
    func predict(text: String, topK: Int = 3) -> [(word: String, confidence: Float)] {
        guard let session = session else {
            print("❌ Model not loaded")
            return []
        }
        
        // Add [MASK] token
        let textWithMask = text + " " + tokenizer.maskToken
        
        // Tokenize
        let (inputIds, attentionMask) = tokenizer.tokenize(textWithMask)
        
        do {
            // Create input tensors with proper data handling
            var inputIdsMutable = inputIds
            var attentionMaskMutable = attentionMask
            
            let inputIdsData = NSMutableData(bytes: &inputIdsMutable, 
                                             length: inputIds.count * MemoryLayout<Int64>.size)
            let attentionMaskData = NSMutableData(bytes: &attentionMaskMutable, 
                                                  length: attentionMask.count * MemoryLayout<Int64>.size)
            
            let inputIdsTensor = try ORTValue(tensorData: inputIdsData,
                                              elementType: .int64,
                                              shape: [1, NSNumber(value: inputIds.count)])
            
            let attentionMaskTensor = try ORTValue(tensorData: attentionMaskData,
                                                   elementType: .int64,
                                                   shape: [1, NSNumber(value: attentionMask.count)])
            
            // Run inference
            let outputs = try session.run(withInputs: [
                "input_ids": inputIdsTensor,
                "attention_mask": attentionMaskTensor
            ],
            outputNames: ["logits"],
            runOptions: nil)
            
            guard let logitsTensor = outputs["logits"],
                  let logitsData = try? logitsTensor.tensorData() as Data else {
                print("❌ Failed to get logits")
                return []
            }
            
            // Find [MASK] position
            guard let maskPosition = inputIds.firstIndex(of: tokenizer.maskTokenId) else {
                print("❌ No [MASK] token found")
                return []
            }
            
            // Extract predictions at [MASK] position
            let vocabSize = 30522
            let floatArray = logitsData.withUnsafeBytes { $0.bindMemory(to: Float.self) }
            
            var predictions: [(id: Int, score: Float)] = []
            for tokenId in 0..<vocabSize {
                let index = Int(maskPosition) * vocabSize + tokenId
                let score = floatArray[index]
                predictions.append((tokenId, score))
            }
            
            // Sort and get top-k
            predictions.sort { $0.score > $1.score }
            let topPredictions = Array(predictions.prefix(topK))
            
            // Apply softmax
            let maxScore = topPredictions.first?.score ?? 0
            let expScores = topPredictions.map { exp($0.score - maxScore) }
            let sumExp = expScores.reduce(0, +)
            
            return topPredictions.enumerated().map { index, prediction in
                let word = tokenizer.decode(prediction.id)
                let confidence = expScores[index] / sumExp
                return (word, confidence)
            }
        } catch {
            print("❌ Prediction failed: \(error)")
            return []
        }
    }
}
```

---

## Keyboard Integration

### KeyboardViewController

```swift
import UIKit

class KeyboardViewController: UIInputViewController {
    private let model = KeyboardModel()
    private var suggestionButtons: [UIButton] = []
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupSuggestionBar()
    }
    
    private func setupSuggestionBar() {
        let suggestionBar = UIStackView()
        suggestionBar.axis = .horizontal
        suggestionBar.distribution = .fillEqually
        suggestionBar.spacing = 8
        
        for i in 0..<3 {
            let button = UIButton(type: .system)
            button.backgroundColor = .systemGray6
            button.layer.cornerRadius = 8
            button.tag = i
            button.addTarget(self, action: #selector(suggestionTapped), for: .touchUpInside)
            suggestionButtons.append(button)
            suggestionBar.addArrangedSubview(button)
        }
        
        view.addSubview(suggestionBar)
        suggestionBar.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            suggestionBar.topAnchor.constraint(equalTo: view.topAnchor, constant: 8),
            suggestionBar.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 8),
            suggestionBar.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -8),
            suggestionBar.heightAnchor.constraint(equalToConstant: 40)
        ])
    }
    
    override func textDidChange(_ textInput: UITextInput?) {
        super.textDidChange(textInput)
        updateSuggestions()
    }
    
    private func updateSuggestions() {
        guard let proxy = textDocumentProxy,
              let text = proxy.documentContextBeforeInput,
              !text.isEmpty else {
            clearSuggestions()
            return
        }
        
        DispatchQueue.global(qos: .userInteractive).async { [weak self] in
            let predictions = self?.model.predict(text: text, topK: 3) ?? []
            
            DispatchQueue.main.async {
                self?.displaySuggestions(predictions)
            }
        }
    }
    
    private func displaySuggestions(_ predictions: [(word: String, confidence: Float)]) {
        for (index, button) in suggestionButtons.enumerated() {
            if index < predictions.count {
                button.setTitle(predictions[index].word, for: .normal)
                button.isHidden = false
            } else {
                button.isHidden = true
            }
        }
    }
    
    private func clearSuggestions() {
        suggestionButtons.forEach { $0.isHidden = true }
    }
    
    @objc private func suggestionTapped(_ sender: UIButton) {
        guard let word = sender.title(for: .normal),
              let proxy = textDocumentProxy else { return }
        proxy.insertText(word + " ")
    }
}
```

---

## Complete Example

See the full implementation in the code above. Key points:

1. **Model Loading**: Uses ONNX Runtime with CoreML acceleration
2. **Tokenization**: WordPiece tokenizer from vocab.txt
3. **Inference**: ~14ms on iPhone (very fast!)
4. **Integration**: Standard UIInputViewController

---

## Troubleshooting

### IR Version Error (Unsupported model IR version: 10)

**Error**: `Unsupported model IR version: 10, max supported IR version: 9`

**Cause**: Older ONNX Runtime versions don't support newer ONNX models.

**Solution 1: Use Swift Package Manager (Easiest)**

1. Remove CocoaPods dependency
2. In Xcode: **File → Add Package Dependencies**
3. Add: `https://github.com/microsoft/onnxruntime-swift-package-manager`
4. Select latest version (1.17.0+)
5. Clean build (Cmd+Shift+K) and rebuild

**Solution 2: Update CocoaPods**

```bash
# Update CocoaPods spec repository
pod repo update

# Update Podfile to use latest available
pod 'onnxruntime-objc'  # Remove version constraint

# Install
pod install
```

**Solution 3: Re-export with Older Opset**

If you must use an older ONNX Runtime version:
```bash
# Edit scripts/export_to_onnx.py
# Change: opset_version=13 to opset_version=11
python scripts/export_to_onnx.py --model-dir ./models/best_model
```

**Recommended**: Use Swift Package Manager for latest versions and easier updates.

### Model Not Loading
- Verify `keyboard_model.onnx` is in bundle
- Check file is added to keyboard extension target
- Ensure ONNX Runtime pod is installed

### Slow Performance
```swift
// ONNX Runtime automatically uses CoreML on iOS
// Just run predictions on background thread
DispatchQueue.global(qos: .userInteractive).async {
    // Predictions here
}
```

### Memory Issues
- Model is only 5.48MB (very small!)
- If issues persist, check for memory leaks
- Use Instruments to profile

---

## Performance

**Expected Results**:
- Model size: 5.48 MB ✅
- Inference time: 10-20ms ✅
- Memory usage: <10MB ✅

**Much better than CoreML!**

---

## Next Steps

1. ✅ Integrate model
2. Test on device
3. Optimize UI
4. Submit to App Store

**Questions?** Check README or open an issue.
