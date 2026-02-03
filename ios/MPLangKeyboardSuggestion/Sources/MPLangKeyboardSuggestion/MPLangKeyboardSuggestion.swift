//
//  MPLangKeyboardSuggestion.swift
//  MPLangKeyboardSuggestion
//
//  Created by MinhPhuPham on 03/02/26.
//

import SwiftUI
import MPEngKeyboardSuggestion
import MPJPKeyboardSuggestion

/// Supported language models for keyboard suggestions
public enum KeyboardLanguage: String, CaseIterable, Identifiable {
    case english = "English"
    case japanese = "Japanese"
    
    public var id: String { rawValue }
    
    public var flag: String {
        switch self {
        case .english: return "🇬🇧"
        case .japanese: return "🇯🇵"
        }
    }
    
    public var displayName: String {
        "\(flag) \(rawValue)"
    }
}

/// Multi-language keyboard suggestion test view
/// Provides a dropdown to switch between English and Japanese keyboard models
public struct MPLangKeyboardSuggestion: View {
    @State private var selectedLanguage: KeyboardLanguage = .english
    @State private var isDropdownExpanded = false
    
    public init() {}
    
    public init(defaultLanguage: KeyboardLanguage) {
        _selectedLanguage = State(initialValue: defaultLanguage)
    }
    
    public var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Language Selector Header
                languageSelector
                
                Divider()
                
                // Content based on selected language
                languageTestView
            }
            .navigationTitle("Keyboard ML Test")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
        }
        #if os(iOS)
        .navigationViewStyle(.stack)
        #endif
    }
    
    // MARK: - Language Selector
    
    private var languageSelector: some View {
        HStack {
            Text("Language Model:")
                .font(.subheadline)
                .foregroundColor(.secondary)
            
            Spacer()
            
            Menu {
                ForEach(KeyboardLanguage.allCases) { language in
                    Button(action: {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            selectedLanguage = language
                        }
                    }) {
                        HStack {
                            Text(language.displayName)
                            if language == selectedLanguage {
                                Image(systemName: "checkmark")
                            }
                        }
                    }
                }
            } label: {
                HStack(spacing: 8) {
                    Text(selectedLanguage.displayName)
                        .font(.headline)
                    Image(systemName: "chevron.down")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .background(Color.blue.opacity(0.1))
                .foregroundColor(.blue)
                .cornerRadius(10)
            }
        }
        .padding()
        .background(Color.clear)
    }
    
    // MARK: - Language Test Views
    
    @ViewBuilder
    private var languageTestView: some View {
        switch selectedLanguage {
        case .english:
            MPEngKeyboardSuggestionTest()
                .transition(.asymmetric(
                    insertion: .move(edge: .leading),
                    removal: .move(edge: .trailing)
                ))
        case .japanese:
            MPJPKeyboardSuggestionTest()
                .transition(.asymmetric(
                    insertion: .move(edge: .trailing),
                    removal: .move(edge: .leading)
                ))
        }
    }
}

// MARK: - Preview

#if DEBUG
struct MPLangKeyboardSuggestion_Previews: PreviewProvider {
    static var previews: some View {
        MPLangKeyboardSuggestion()
    }
}
#endif
