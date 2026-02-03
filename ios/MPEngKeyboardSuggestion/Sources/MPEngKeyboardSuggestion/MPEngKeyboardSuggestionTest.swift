//
//  MPEngKeyboardSuggestionTest.swift
//  MPEngKeyboardSuggestion
//
//  Created by MinhPhuPham on 03/02/26.
//

import SwiftUI
import Combine

/// SwiftUI test view for English keyboard suggestions
/// Supports manual input and automatic testing with performance logging
@available(iOS 14.0, macOS 11.0, *)
public struct MPEngKeyboardSuggestionTest: View {
    @State var testMode: EngTestMode = .manual
    @StateObject var sharedVM = EngSharedTestViewModel()
    
    public init() {}
    
    public var body: some View {
        VStack(spacing: 0) {
            // Header
            Text("🇬🇧 English Keyboard Test")
                .font(.headline)
                .padding()
            
            // Model Status
            HStack {
                Circle()
                    .fill(sharedVM.isModelReady ? Color.green : Color.red)
                    .frame(width: 10, height: 10)
                Text(sharedVM.isModelReady ? "Model Ready" : "Loading model...")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                if let vocabSize = sharedVM.vocabSize {
                    Text("Vocab: \(vocabSize)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .padding(.horizontal)
            .padding(.bottom, 8)
            
            // Mode Picker
            Picker("Test Mode", selection: $testMode) {
                Text("Manual").tag(EngTestMode.manual)
                Text("Automatic").tag(EngTestMode.automatic)
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)
            .padding(.bottom, 16)
            
            // Content based on mode
            engContentView
                .environmentObject(sharedVM)
                .id(testMode)
        }
    }
    
    @ViewBuilder
    var engContentView: some View {
        switch testMode {
        case .manual:
            EngManualTestView()
        case .automatic:
            EngAutomaticTestView()
        }
    }
}

// MARK: - Test Mode Enum

enum EngTestMode: Int, Hashable {
    case manual = 0
    case automatic = 1
}

// MARK: - Manual Test View

@available(iOS 14.0, macOS 11.0, *)
struct EngManualTestView: View {
    @EnvironmentObject var viewModel: EngSharedTestViewModel
    
    var body: some View {
        VStack(spacing: 16) {
            // Input Field with Search Button
            HStack {
                TextField("Type in English", text: $viewModel.inputText)
                    .textFieldStyle(.roundedBorder)
                
                Button(action: {
                    viewModel.fetchSuggestions()
                }) {
                    Image(systemName: "magnifyingglass")
                        .foregroundColor(.white)
                        .padding(10)
                        .background(viewModel.inputText.isEmpty ? Color.gray : Color.blue)
                        .cornerRadius(8)
                }
                .disabled(viewModel.inputText.isEmpty)
            }
            .padding(.horizontal)
            
            // Response time
            if let time = viewModel.responseTime {
                HStack {
                    Spacer()
                    Text("⏱️ \(String(format: "%.1f", time))ms")
                        .font(.caption)
                        .foregroundColor(.blue)
                }
                .padding(.horizontal)
            }
            
            // Suggestions List
            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    if viewModel.isLoading {
                        HStack {
                            ProgressView()
                            Text("Getting suggestions...")
                                .font(.caption)
                        }
                        .padding()
                    } else if viewModel.suggestions.isEmpty {
                        Text("Type something and tap 🔍 to see suggestions")
                            .foregroundColor(.secondary)
                            .padding()
                    } else {
                        ForEach(Array(viewModel.suggestions.enumerated()), id: \.offset) { index, suggestion in
                            HStack {
                                Text("\(index + 1).")
                                    .foregroundColor(.secondary)
                                    .frame(width: 24)
                                Text(suggestion.word)
                                    .font(.body)
                                Spacer()
                                Text(suggestion.source.rawValue)
                                    .font(.caption2)
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(sourceColor(suggestion.source).opacity(0.2))
                                    .foregroundColor(sourceColor(suggestion.source))
                                    .cornerRadius(4)
                                Text("\(String(format: "%.0f", suggestion.score))pt")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            .padding(.horizontal)
                            .padding(.vertical, 10)
                            .background(Color.gray.opacity(0.1))
                            .cornerRadius(8)
                            .padding(.horizontal)
                        }
                    }
                }
            }
            
            Spacer()
        }
    }
    
    private func sourceColor(_ source: MPSuggestionSource) -> Color {
        switch source {
        case .gru: return .blue
        case .trie: return .purple
        case .learning: return .green
        case .typo: return .orange
        case .shortcut: return .pink
        case .hybrid: return Color(red: 0.0, green: 0.8, blue: 0.8) // Cyan-like color for macOS 11 compatibility
        }
    }
}

// MARK: - Automatic Test View

@available(iOS 14.0, macOS 11.0, *)
struct EngAutomaticTestView: View {
    @EnvironmentObject var viewModel: EngSharedTestViewModel
    
    var body: some View {
        VStack(spacing: 16) {
            // Start Button
            Button(action: {
                viewModel.startAutomaticTest()
            }) {
                HStack {
                    if viewModel.isAutoTestRunning {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                        Text("Testing \(viewModel.autoProgress)/\(viewModel.testCases.count)...")
                    } else {
                        Image(systemName: "play.fill")
                        Text("Run \(viewModel.testCases.count) Test Cases")
                    }
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(viewModel.isAutoTestRunning ? Color.gray : Color.blue)
                .foregroundColor(.white)
                .cornerRadius(10)
            }
            .disabled(viewModel.isAutoTestRunning || !viewModel.isModelReady)
            .padding(.horizontal)
            
            // Results Summary
            if !viewModel.autoResults.isEmpty {
                HStack(spacing: 16) {
                    EngStatBox(value: "\(viewModel.autoResults.count)", label: "Tests", color: .blue)
                    EngStatBox(value: "\(String(format: "%.0f", viewModel.avgTime))ms", label: "Avg", color: .purple)
                    EngStatBox(value: "\(String(format: "%.0f", viewModel.minTime))ms", label: "Min", color: .green)
                    EngStatBox(value: "\(String(format: "%.0f", viewModel.maxTime))ms", label: "Max", color: .red)
                }
                .padding(.horizontal)
            }
            
            // Results List
            ScrollView {
                LazyVStack(spacing: 8) {
                    ForEach(viewModel.autoResults) { result in
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Text(result.input)
                                    .font(.system(.body, design: .rounded).bold())
                                Spacer()
                                Text("\(String(format: "%.0f", result.time))ms")
                                    .font(.caption)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 3)
                                    .background(result.time < 100 ? Color.green.opacity(0.2) : Color.orange.opacity(0.2))
                                    .foregroundColor(result.time < 100 ? .green : .orange)
                                    .cornerRadius(6)
                            }
                            
                            if result.outputs.isEmpty {
                                Text("No results")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            } else {
                                ScrollView(.horizontal, showsIndicators: false) {
                                    HStack(spacing: 6) {
                                        ForEach(result.outputs, id: \.self) { output in
                                            Text(output)
                                                .font(.caption)
                                                .padding(.horizontal, 8)
                                                .padding(.vertical, 4)
                                                .background(Color.blue.opacity(0.1))
                                                .cornerRadius(6)
                                        }
                                    }
                                }
                            }
                        }
                        .padding()
                        .background(Color.gray.opacity(0.05))
                        .cornerRadius(10)
                    }
                }
                .padding(.horizontal)
            }
        }
    }
}

// MARK: - Stat Box

@available(iOS 14.0, macOS 11.0, *)
struct EngStatBox: View {
    let value: String
    let label: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 4) {
            Text(value)
                .font(.system(.title3, design: .rounded).bold())
                .foregroundColor(color)
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(color.opacity(0.1))
        .cornerRadius(8)
    }
}

// MARK: - Shared ViewModel (Single keyboard instance)

@available(iOS 14.0, macOS 11.0, *)
class EngSharedTestViewModel: ObservableObject {
    // Manual test state
    @Published var inputText = ""
    @Published var suggestions: [MPSuggestion] = []
    @Published var responseTime: Double?
    @Published var isLoading = false
    
    // Automatic test state
    @Published var autoResults: [EngTestResult] = []
    @Published var isAutoTestRunning = false
    @Published var autoProgress = 0
    
    // Shared model state
    @Published var isModelReady = false
    @Published var vocabSize: Int?
    
    // Single keyboard instance
    var keyboard: MPKeyboardSuggestion?
    
    // Debounce support
    private var cancellables = Set<AnyCancellable>()
    
    // Request cancellation
    private var currentFetchTask: DispatchWorkItem?
    
    // Test cases for English
    let testCases = [
        // Next-word prediction (ends with space)
        "How are ",
        "I want to ",
        "Thank you for ",
        "What is the ",
        "Can you ",
        "Please let me ",
        "Would you like ",
        "I think that ",
        "We should ",
        "They said ",
        // Word completion (partial words)
        "hel",
        "tha",
        "beau",
        "imp",
        "com",
        "prog",
        "dev",
        "app",
        "key",
        "sug",
        // Typo correction
        "thers",
        "becuase",
        "definately",
        "occured",
        "recieve",
        "seperate",
        "occassion",
        "accomodate",
        "neccessary",
        "enviroment"
    ]
    
    var avgTime: Double {
        guard !autoResults.isEmpty else { return 0 }
        return autoResults.map(\.time).reduce(0, +) / Double(autoResults.count)
    }
    var minTime: Double { autoResults.map(\.time).min() ?? 0 }
    var maxTime: Double { autoResults.map(\.time).max() ?? 0 }
    
    init() {
        // Initialize keyboard once
        DispatchQueue.main.async { [weak self] in
            self?.keyboard = MPKeyboardSuggestion()
            print("✅ MPKeyboardSuggestion (English) initialized")
            
            // Check ready state
            self?.checkModelReady()
        }
        
        // Setup debounce for input text
        setupDebounce()
    }
    
    /// Setup 100ms debounce for inputText changes
    private func setupDebounce() {
        $inputText
            .debounce(for: .milliseconds(100), scheduler: DispatchQueue.main)
            .removeDuplicates()
            .sink { [weak self] text in
                guard !text.isEmpty else {
                    self?.suggestions = []
                    self?.responseTime = nil
                    return
                }
                self?.fetchSuggestions()
            }
            .store(in: &cancellables)
    }
    
    func checkModelReady() {
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            guard let self = self, let keyboard = self.keyboard else { return }
            
            let stats = keyboard.getStats()
            self.isModelReady = stats.gruReady
            self.vocabSize = stats.vocabSize
            
            if !self.isModelReady {
                self.checkModelReady() // Retry
            } else {
                print("✅ English Model is ready! Vocab: \(stats.vocabSize)")
            }
        }
    }
    
    // MARK: - Manual Test
    
    func fetchSuggestions() {
        guard !inputText.isEmpty, let keyboard = keyboard else {
            suggestions = []
            isLoading = false
            return
        }
        
        // Cancel any previous in-flight request
        currentFetchTask?.cancel()
        
        isLoading = true
        let input = inputText
        
        let workItem = DispatchWorkItem { [weak self] in
            guard !(self?.currentFetchTask?.isCancelled ?? true) else {
                DispatchQueue.main.async {
                    self?.isLoading = false
                }
                return
            }
            
            let start = CFAbsoluteTimeGetCurrent()
            let results = keyboard.getSuggestions(for: input, limit: 5)
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            guard !(self?.currentFetchTask?.isCancelled ?? true) else { return }
            
            DispatchQueue.main.async {
                self?.suggestions = results
                self?.responseTime = time
                self?.isLoading = false
            }
        }
        
        currentFetchTask = workItem
        DispatchQueue.global(qos: .userInteractive).async(execute: workItem)
    }
    
    // MARK: - Automatic Test
    
    func startAutomaticTest() {
        guard let keyboard = keyboard, isModelReady else {
            print("❌ Keyboard not ready")
            return
        }
        
        isAutoTestRunning = true
        autoResults = []
        autoProgress = 0
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            for (idx, testCase) in self.testCases.enumerated() {
                let start = CFAbsoluteTimeGetCurrent()
                let predictions = keyboard.getSuggestions(for: testCase, limit: 5)
                let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
                
                let result = EngTestResult(
                    input: testCase,
                    outputs: predictions.map(\.word),
                    time: time
                )
                
                DispatchQueue.main.async {
                    self.autoResults.append(result)
                    self.autoProgress = idx + 1
                }
                
                Thread.sleep(forTimeInterval: 0.03)
            }
            
            DispatchQueue.main.async {
                self.isAutoTestRunning = false
            }
        }
    }
}

// MARK: - Test Result Model

struct EngTestResult: Identifiable {
    let id = UUID()
    let input: String
    let outputs: [String]
    let time: Double
}
