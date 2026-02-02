// ============================================================
// MPJPTokenizer.swift
// ============================================================
// ByteLevel BPE tokenizer for zenz GPT-2 model
// Compatible with Hugging Face tokenizers ByteLevel encoding
// ============================================================

import Foundation

/// ByteLevel BPE tokenizer for zenz model
final class MPJPTokenizer {
    
    // MARK: - Special Markers (zenz format)
    
    static let ZENZ_START = "\u{EE00}"
    static let ZENZ_OUTPUT = "\u{EE01}"
    static let ZENZ_CONTEXT = "\u{EE02}"
    static let ZENZ_EOS = "</s>"
    static let ZENZ_BOS = "<s>"
    static let ZENZ_UNK = "[UNK]"
    
    // MARK: - Properties
    
    private var tokenToId: [String: Int] = [:]
    private var idToToken: [Int: String] = [:]
    private(set) var vocabSize: Int = 0
    
    /// Byte-to-Unicode mapping (GPT-2 style)
    private var byteEncoder: [UInt8: Character] = [:]
    private var byteDecoder: [Character: UInt8] = [:]
    
    /// BPE merge rules: (pair) -> merged token rank
    private var bpeRanks: [String: Int] = [:]
    
    // MARK: - Initialization
    
    init() {
        buildByteMapping()
    }
    
    /// Build the byte-to-unicode mapping used by GPT-2 ByteLevel BPE
    private func buildByteMapping() {
        var bs: [Int] = []
        
        // Printable ASCII and extended Latin
        for i in 33...126 { bs.append(i) }
        for i in 161...172 { bs.append(i) }
        for i in 174...255 { bs.append(i) }
        
        var cs = bs.map { $0 }
        var n = 0
        
        for b in 0...255 {
            if !bs.contains(b) {
                bs.append(b)
                cs.append(256 + n)
                n += 1
            }
        }
        
        for (b, c) in zip(bs, cs) {
            if let scalar = Unicode.Scalar(c) {
                byteEncoder[UInt8(b)] = Character(scalar)
                byteDecoder[Character(scalar)] = UInt8(b)
            }
        }
        
        MPJPLog.debug("[Tokenizer] Built byte mapping: \(byteEncoder.count) bytes")
        
        // Debug: verify key mappings for Japanese
        let testChars: [Character] = ["ã", "ģ", "Ĥ"]
        for char in testChars {
            if let byte = byteDecoder[char] {
                MPJPLog.debug("[Tokenizer] Decoder: '\(char)' (U+\(String(format: "%04X", char.unicodeScalars.first!.value))) -> \(byte)")
            } else {
                MPJPLog.warn("[Tokenizer] Decoder: '\(char)' NOT FOUND!")
            }
        }
    }
    
    /// Load vocabulary from vocab.json
    func loadVocab(from url: URL) -> Bool {
        do {
            let data = try Data(contentsOf: url)
            if let vocab = try JSONSerialization.jsonObject(with: data) as? [String: Int] {
                tokenToId = vocab
                idToToken = Dictionary(uniqueKeysWithValues: vocab.map { ($1, $0) })
                vocabSize = vocab.count
                MPJPLog.info("[Tokenizer] Loaded vocab: \(vocabSize) tokens")
                return true
            }
        } catch {
            MPJPLog.error("[Tokenizer] Failed to load vocab: \(error)")
        }
        return false
    }
    
    /// Load BPE merges from merges.txt
    func loadMerges(from url: URL) -> Bool {
        do {
            let content = try String(contentsOf: url, encoding: .utf8)
            let lines = content.components(separatedBy: .newlines)
            
            var rank = 0
            for line in lines {
                // Skip header and empty lines
                if line.isEmpty || line.hasPrefix("#") { continue }
                
                let parts = line.components(separatedBy: " ")
                if parts.count == 2 {
                    let pair = parts[0] + " " + parts[1]
                    bpeRanks[pair] = rank
                    rank += 1
                }
            }
            
            MPJPLog.info("[Tokenizer] Loaded \(bpeRanks.count) BPE merges")
            return true
        } catch {
            MPJPLog.error("[Tokenizer] Failed to load merges: \(error)")
        }
        return false
    }
    
    /// Build default vocabulary (fallback)
    func buildDefaultVocab() {
        tokenToId = [Self.ZENZ_UNK: 0, "[PAD]": 1, Self.ZENZ_BOS: 2, Self.ZENZ_EOS: 3]
        idToToken = Dictionary(uniqueKeysWithValues: tokenToId.map { ($1, $0) })
        vocabSize = tokenToId.count
        MPJPLog.info("[Tokenizer] Built default vocab: \(vocabSize) tokens")
    }
    
    // MARK: - ByteLevel Encoding
    
    /// Encode text to byte-level characters
    private func textToByteChars(_ text: String) -> [Character] {
        let utf8Bytes = Array(text.utf8)
        return utf8Bytes.compactMap { byteEncoder[$0] }
    }
    
    /// Decode byte-level characters to text
    private func byteCharsToText(_ chars: [Character]) -> String {
        var bytes: [UInt8] = []
        var missingChars: [Character] = []
        
        for char in chars {
            if let byte = byteDecoder[char] {
                bytes.append(byte)
            } else {
                missingChars.append(char)
            }
        }
        
        if !missingChars.isEmpty {
            MPJPLog.warn("[byteCharsToText] Missing decoder for: \(missingChars.map { "'\($0)' U+\(String(format: "%04X", $0.unicodeScalars.first!.value))" })")
        }
        
        let result = String(bytes: bytes, encoding: .utf8) ?? ""
        MPJPLog.debug("[byteCharsToText] \(chars.count) chars -> \(bytes.count) bytes -> '\(result)'")
        return result
    }
    
    // MARK: - BPE Algorithm
    
    /// Apply BPE merges to a list of tokens
    private func applyBPE(_ tokens: [String]) -> [String] {
        var word = tokens
        
        while word.count > 1 {
            // Find the best pair to merge
            var bestPair: (Int, String)? = nil  // (index, merged)
            var bestRank = Int.max
            
            for i in 0..<(word.count - 1) {
                let pair = word[i] + " " + word[i + 1]
                if let rank = bpeRanks[pair], rank < bestRank {
                    bestRank = rank
                    bestPair = (i, word[i] + word[i + 1])
                }
            }
            
            guard let (idx, merged) = bestPair else {
                break  // No more merges possible
            }
            
            // Apply the merge
            var newWord: [String] = []
            var i = 0
            while i < word.count {
                if i == idx {
                    newWord.append(merged)
                    i += 2  // Skip both merged tokens
                } else {
                    newWord.append(word[i])
                    i += 1
                }
            }
            word = newWord
        }
        
        return word
    }
    
    // MARK: - Tokenization
    
    /// Tokenize text into token IDs
    func encode(_ text: String) -> [Int] {
        // Step 1: Convert to byte-level characters
        let byteChars = textToByteChars(text)
        
        // Step 2: Start with individual characters as tokens
        var tokens = byteChars.map { String($0) }
        
        // Step 3: Apply BPE merges
        if !bpeRanks.isEmpty {
            tokens = applyBPE(tokens)
        }
        
        // Step 4: Convert tokens to IDs
        var ids: [Int] = []
        var unknownCount = 0
        
        for token in tokens {
            if let id = tokenToId[token] {
                ids.append(id)
            } else {
                unknownCount += 1
                ids.append(tokenToId[Self.ZENZ_UNK] ?? 0)
            }
        }
        
        MPJPLog.debug("[Tokenizer.encode] '\(text)' -> \(ids.count) tokens, \(unknownCount) unknown")
        return ids
    }
    
    /// Decode token IDs back to text
    func decode(_ ids: [Int]) -> String {
        var byteChars: [Character] = []
        var tokenStrings: [String] = []
        
        for id in ids {
            guard let token = idToToken[id] else { 
                MPJPLog.debug("[Decode] ID \(id) -> not found")
                continue 
            }
            
            tokenStrings.append(token)
            
            // Skip special tokens
            if token.hasPrefix("<") || token.hasPrefix("[") { 
                MPJPLog.debug("[Decode] ID \(id) -> '\(token)' (special, skipped)")
                continue 
            }
            
            for char in token {
                byteChars.append(char)
            }
        }
        
        MPJPLog.debug("[Decode] IDs: \(ids) -> tokens: \(tokenStrings)")
        MPJPLog.debug("[Decode] ByteChars: \(byteChars.map { String($0) })")
        
        let result = byteCharsToText(byteChars)
        MPJPLog.debug("[Decode] Result: '\(result)'")
        return result
    }
    
    /// Get token ID for a token string
    func getTokenId(_ token: String) -> Int? {
        return tokenToId[token]
    }
    
    /// Get token string for an ID
    func getToken(_ id: Int) -> String? {
        return idToToken[id]
    }
    
    // MARK: - Hiragana/Katakana Conversion
    
    static func hiraganaToKatakana(_ text: String) -> String {
        var result = ""
        for scalar in text.unicodeScalars {
            if scalar.value >= 0x3040 && scalar.value <= 0x309F {
                if let katakana = Unicode.Scalar(scalar.value + 0x60) {
                    result += String(Character(katakana))
                } else {
                    result += String(Character(scalar))
                }
            } else {
                result += String(Character(scalar))
            }
        }
        return result
    }
    
    static func katakanaToHiragana(_ text: String) -> String {
        var result = ""
        for scalar in text.unicodeScalars {
            if scalar.value >= 0x30A0 && scalar.value <= 0x30FF {
                if let hiragana = Unicode.Scalar(scalar.value - 0x60) {
                    result += String(Character(hiragana))
                } else {
                    result += String(Character(scalar))
                }
            } else {
                result += String(Character(scalar))
            }
        }
        return result
    }
    
    static func isHiragana(_ text: String) -> Bool {
        for scalar in text.unicodeScalars {
            if scalar.value < 0x3040 || scalar.value > 0x309F {
                if scalar.value != 0x30FC && scalar.value != 0x3000 &&
                   scalar.value != 0x3001 && scalar.value != 0x3002 {
                    return false
                }
            }
        }
        return !text.isEmpty
    }
    
    static func isKatakana(_ text: String) -> Bool {
        for scalar in text.unicodeScalars {
            if scalar.value < 0x30A0 || scalar.value > 0x30FF {
                if scalar.value != 0x3000 && scalar.value != 0x3001 && scalar.value != 0x3002 {
                    return false
                }
            }
        }
        return !text.isEmpty
    }
    
    static func isKana(_ text: String) -> Bool {
        return isHiragana(text) || isKatakana(text)
    }
    
    // MARK: - Prompt Formatting
    
    func formatConversionPrompt(_ kana: String, context: String? = nil) -> String {
        let katakana = Self.hiraganaToKatakana(kana)
        if let ctx = context, !ctx.isEmpty {
            return "\(Self.ZENZ_START)\(katakana)\(Self.ZENZ_CONTEXT)\(ctx)\(Self.ZENZ_OUTPUT)"
        } else {
            return "\(Self.ZENZ_START)\(katakana)\(Self.ZENZ_OUTPUT)"
        }
    }
    
    func formatPredictionPrompt(_ context: String) -> String {
        // Python format: f"{ZENZ_START}。{ZENZ_CONTEXT}{left_context}"
        // This format produces much better predictions (63% vs 92%)
        return "\(Self.ZENZ_START)。\(Self.ZENZ_CONTEXT)\(context)"
    }
}
