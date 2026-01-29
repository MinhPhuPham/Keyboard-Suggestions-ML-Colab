// ============================================================
// MPEngKeyboardSuggestionTests.swift
// ============================================================

import XCTest
@testable import MPEngKeyboardSuggestion

final class MPEngKeyboardSuggestionTests: XCTestCase {
    
    var keyboard: MPKeyboardSuggestion!
    
    override func setUp() {
        super.setUp()
        keyboard = MPKeyboardSuggestion.shared
    }
    
    // MARK: - Basic Tests
    
    func testEmptyInput() {
        let suggestions = keyboard.getSuggestions(for: "")
        XCTAssertTrue(suggestions.isEmpty)
    }
    
    func testPrefixCompletion() {
        let suggestions = keyboard.getSuggestions(for: "hel")
        XCTAssertFalse(suggestions.isEmpty)
        XCTAssertTrue(suggestions.contains { $0.word.lowercased().hasPrefix("hel") })
    }
    
    func testNextWordPrediction() {
        let suggestions = keyboard.getSuggestions(for: "how are ")
        XCTAssertFalse(suggestions.isEmpty)
    }
    
    // MARK: - Shortcut Tests
    
    func testAddShortcut() {
        keyboard.addShortcut("brb", "be right back")
        let suggestions = keyboard.getSuggestions(for: "brb")
        XCTAssertTrue(suggestions.contains { $0.word == "be right back" })
        keyboard.removeShortcut("brb")
    }
    
    // MARK: - Learning Tests
    
    func testRecordSelection() {
        keyboard.recordSelection("hello", context: "say")
        let stats = keyboard.getStats()
        XCTAssertGreaterThan(stats.learnedWords, 0)
    }
    
    // MARK: - Stats Tests
    
    func testGetStats() {
        let stats = keyboard.getStats()
        XCTAssertGreaterThanOrEqual(stats.vocabSize, 0)
    }
}
