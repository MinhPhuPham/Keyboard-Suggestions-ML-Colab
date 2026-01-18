//
//  KeyboardSuggestionTest.swift
//  EnglishKeyboardAI
//
//  Created by Phu Pham on 26/12/25.
//

import SwiftUI

/// SwiftUI test view for keyboard suggestions
/// Supports manual input and automatic testing with performance logging
@available(iOS 15.0, macOS 12.0, *)
struct KeyboardSuggestionsTest: View {
    @StateObject private var viewModel = TestViewModel()
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Mode Picker
                Picker("Test Mode", selection: $viewModel.testMode) {
                    Text("Manual").tag(TestMode.manual)
                    Text("Automatic").tag(TestMode.automatic)
                }
                .pickerStyle(.segmented)
                .padding()
                
                if viewModel.testMode == .manual {
                    manualTestView
                } else {
                    automaticTestView
                }
                
                Spacer()
            }
            .navigationTitle("English Keyboard AI Test")
        }
    }
    
    // MARK: - Manual Test View
    
    private var manualTestView: some View {
        VStack(spacing: 16) {
            // Input Field
            TextField("Type in English", text: $viewModel.manualInput)
                .textFieldStyle(.roundedBorder)
                .padding(.horizontal)
                .onChange(of: viewModel.manualInput) { _ in
                    viewModel.getManualSuggestions()
                }
            
            // Performance Info
            if let responseTime = viewModel.lastResponseTime {
                Text("Response: \(String(format: "%.2f", responseTime))ms")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            // Suggestions List
            VStack(alignment: .leading, spacing: 8) {
                Text("Suggestions:")
                    .font(.headline)
                    .padding(.horizontal)
                
                if viewModel.suggestions.isEmpty {
                    Text("Type something to see suggestions...")
                        .foregroundColor(.secondary)
                        .padding()
                } else {
                    ForEach(Array(viewModel.suggestions.enumerated()), id: \.offset) { index, suggestion in
                        HStack {
                            Text("\(index + 1).")
                                .foregroundColor(.secondary)
                            Text(suggestion.word)
                                .font(.body)
                            Spacer()
                            Text("\(String(format: "%.1f", suggestion.confidence * 100))%")
                                .font(.caption)
                                .foregroundColor(.blue)
                        }
                        .padding(.horizontal)
                        .padding(.vertical, 8)
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(8)
                        .padding(.horizontal)
                    }
                }
            }
        }
    }
    
    // MARK: - Automatic Test View
    
    private var automaticTestView: some View {
        VStack(spacing: 16) {
            // Test Progress
            if viewModel.isRunningTest {
                ProgressView("Testing... \(viewModel.testProgress)/\(viewModel.testCases.count)")
                    .padding()
            }
            
            // Start Test Button
            Button(action: {
                viewModel.runAutomaticTest()
            }) {
                Text(viewModel.isRunningTest ? "Testing..." : "Start Automatic Test")
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(viewModel.isRunningTest ? Color.gray : Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
            .disabled(viewModel.isRunningTest)
            .padding(.horizontal)
            
            // Results Summary
            if !viewModel.testResults.isEmpty {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Test Results")
                        .font(.headline)
                    
                    Text("Total Tests: \(viewModel.testResults.count)")
                    Text("Avg Response: \(String(format: "%.2f", viewModel.averageResponseTime))ms")
                    Text("Min: \(String(format: "%.2f", viewModel.minResponseTime))ms")
                    Text("Max: \(String(format: "%.2f", viewModel.maxResponseTime))ms")
                }
                .padding()
                .background(Color.green.opacity(0.1))
                .cornerRadius(10)
                .padding(.horizontal)
                
                // Export Button
                Button(action: {
                    viewModel.exportResults()
                }) {
                    HStack {
                        Image(systemName: "square.and.arrow.down")
                        Text("Export Results")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.green)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
                .padding(.horizontal)
            }
            
            // Results List
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 8) {
                    ForEach(viewModel.testResults) { result in
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text(result.input)
                                    .font(.body)
                                Spacer()
                                Text("\(String(format: "%.2f", result.speedResponse))ms")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            ForEach(Array(result.suggestions.enumerated()), id: \.offset) { idx, suggestion in
                                HStack {
                                    Text(suggestion)
                                        .font(.caption)
                                    if idx < result.confidences.count {
                                        Text("(\(String(format: "%.1f", result.confidences[idx] * 100))%)")
                                            .font(.caption2)
                                            .foregroundColor(.blue)
                                    }
                                }
                            }
                        }
                        .padding()
                        .background(Color.gray.opacity(0.05))
                        .cornerRadius(8)
                    }
                }
                .padding(.horizontal)
            }
        }
    }
}

// MARK: - View Model

enum TestMode {
    case manual
    case automatic
}

struct TestResult: Identifiable, Codable {
    let id = UUID()
    let speedResponse: Double  // in milliseconds
    let input: String
    let suggestions: [String]
    let confidences: [Float]  // Confidence scores for each suggestion
    
    enum CodingKeys: String, CodingKey {
        case speedResponse, input, suggestions, confidences
    }
}

@available(iOS 15.0, macOS 12.0, *)
class TestViewModel: ObservableObject {
    @Published var testMode: TestMode = .manual
    @Published var manualInput: String = ""
    @Published var suggestions: [(word: String, confidence: Float)] = []
    @Published var lastResponseTime: Double?
    
    @Published var isRunningTest = false
    @Published var testProgress = 0
    @Published var testResults: [TestResult] = []
    
    private var model: KeyboardModel?
    
    // Test cases for automatic testing - English phrases
    let testCases = [
        "hello",
        "thank",
        "good morning",
        "how are",
        "I am",
        "what is",
        "where is",
        "can you",
        "please",
        "sorry",
        "yes",
        "no",
        "maybe",
        "today",
        "tomorrow"
    ]
    
    init() {
        model = KeyboardModel()
        print("✓ English Keyboard Model initialized")
    }
    
    // MARK: - Manual Testing
    
    func getManualSuggestions() {
        guard !manualInput.isEmpty else {
            suggestions = []
            lastResponseTime = nil
            return
        }
        
        // Run predictions on background thread as recommended in IOS_INTEGRATION.md
        DispatchQueue.global(qos: .userInteractive).async { [weak self] in
            guard let self = self else { return }
            
            let startTime = CFAbsoluteTimeGetCurrent()
            
            // Get predictions from ONNX model
            let predictions = self.model?.predict(text: self.manualInput, topK: 3) ?? []
            
            let endTime = CFAbsoluteTimeGetCurrent()
            let responseTime = (endTime - startTime) * 1000 // Convert to ms
            
            DispatchQueue.main.async {
                self.suggestions = predictions
                self.lastResponseTime = responseTime
            }
        }
    }
    
    // MARK: - Automatic Testing
    
    func runAutomaticTest() {
        guard let model = model else {
            print("❌ Model not initialized")
            return
        }
        
        isRunningTest = true
        testResults = []
        testProgress = 0
        
        // Run tests asynchronously
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            for (index, testCase) in self.testCases.enumerated() {
                let startTime = CFAbsoluteTimeGetCurrent()
                
                let predictions = model.predict(text: testCase, topK: 3)
                let suggestions = predictions.map { $0.word }
                let confidences = predictions.map { $0.confidence }
                
                let endTime = CFAbsoluteTimeGetCurrent()
                let responseTime = (endTime - startTime) * 1000 // ms
                
                let result = TestResult(
                    speedResponse: responseTime,
                    input: testCase,
                    suggestions: suggestions,
                    confidences: confidences
                )
                
                DispatchQueue.main.async {
                    self.testResults.append(result)
                    self.testProgress = index + 1
                }
                
                // Small delay between tests
                Thread.sleep(forTimeInterval: 0.1)
            }
            
            DispatchQueue.main.async {
                self.isRunningTest = false
            }
        }
    }
    
    // MARK: - Results Export
    
    func exportResults() {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        
        guard let jsonData = try? encoder.encode(testResults),
              let jsonString = String(data: jsonData, encoding: .utf8) else {
            print("❌ Failed to encode results")
            return
        }
        
        // Save to Documents directory
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let fileURL = documentsURL.appendingPathComponent("english_keyboard_test_results.json")
        
        do {
            try jsonString.write(to: fileURL, atomically: true, encoding: .utf8)
            print("✓ Results exported to: \(fileURL.path)")
            
            // Also print to console for easy copying
            print("\n=== ENGLISH KEYBOARD TEST RESULTS ===")
            print(jsonString)
            print("=====================================\n")
            
        } catch {
            print("❌ Failed to save results: \(error)")
        }
    }
    
    // MARK: - Computed Properties
    
    var averageResponseTime: Double {
        guard !testResults.isEmpty else { return 0 }
        return testResults.map { $0.speedResponse }.reduce(0, +) / Double(testResults.count)
    }
    
    var minResponseTime: Double {
        testResults.map { $0.speedResponse }.min() ?? 0
    }
    
    var maxResponseTime: Double {
        testResults.map { $0.speedResponse }.max() ?? 0
    }
}
