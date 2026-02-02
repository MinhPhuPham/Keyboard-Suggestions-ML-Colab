import XCTest
@testable import MPJPKeyboardSuggestion

final class MPJPKeyboardSuggestionTests: XCTestCase {
    
    func testTokenizerHiraganaToKatakana() {
        let result = MPJPTokenizer.hiraganaToKatakana("ありがとう")
        XCTAssertEqual(result, "アリガトウ")
    }
    
    func testTokenizerKatakanaToHiragana() {
        let result = MPJPTokenizer.katakanaToHiragana("アリガトウ")
        XCTAssertEqual(result, "ありがとう")
    }
    
    func testIsHiragana() {
        XCTAssertTrue(MPJPTokenizer.isHiragana("ありがとう"))
        XCTAssertFalse(MPJPTokenizer.isHiragana("アリガトウ"))
        XCTAssertFalse(MPJPTokenizer.isHiragana("有難う"))
    }
    
    func testIsKatakana() {
        XCTAssertTrue(MPJPTokenizer.isKatakana("アリガトウ"))
        XCTAssertFalse(MPJPTokenizer.isKatakana("ありがとう"))
    }
    
    func testIsKana() {
        XCTAssertTrue(MPJPTokenizer.isKana("ありがとう"))
        XCTAssertTrue(MPJPTokenizer.isKana("アリガトウ"))
        XCTAssertFalse(MPJPTokenizer.isKana("有難う"))
    }
    
    func testFormatConversionPrompt() {
        let tokenizer = MPJPTokenizer()
        
        // Without context
        let prompt1 = tokenizer.formatConversionPrompt("ありがとう")
        XCTAssertTrue(prompt1.contains("\u{EE00}"))
        XCTAssertTrue(prompt1.contains("アリガトウ"))
        XCTAssertTrue(prompt1.contains("\u{EE01}"))
        
        // With context
        let prompt2 = tokenizer.formatConversionPrompt("ありがとう", context: "どうも")
        XCTAssertTrue(prompt2.contains("\u{EE02}"))
        XCTAssertTrue(prompt2.contains("どうも"))
    }
    
    func testFormatPredictionPrompt() {
        let tokenizer = MPJPTokenizer()
        let prompt = tokenizer.formatPredictionPrompt("ありがとう")
        XCTAssertTrue(prompt.contains("\u{EE00}"))
        XCTAssertTrue(prompt.contains("。"))
        XCTAssertTrue(prompt.contains("\u{EE02}"))
        XCTAssertTrue(prompt.contains("ありがとう"))
    }
    
    func testKeyboardInitialization() {
        // Test that keyboard can be initialized
        // Model may not load in test environment
        let keyboard = MPJPKeyboardSuggestion()
        let stats = keyboard.getStats()
        XCTAssertEqual(stats.vocabSize, 6000)
    }
    
    func testStaticHelpers() {
        XCTAssertEqual(MPJPKeyboardSuggestion.hiraganaToKatakana("こんにちは"), "コンニチハ")
        XCTAssertTrue(MPJPKeyboardSuggestion.isHiragana("あいうえお"))
        XCTAssertTrue(MPJPKeyboardSuggestion.isKatakana("アイウエオ"))
    }
}
