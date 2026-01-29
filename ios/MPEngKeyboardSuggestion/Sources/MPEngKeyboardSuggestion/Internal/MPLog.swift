// ============================================================
// MPLog.swift
// ============================================================
// Debug logging helper for MPEngKeyboardSuggestion package
// Only logs in DEBUG builds
// ============================================================

import Foundation

/// Thread-safe debug logger for MPEngKeyboardSuggestion
/// Disabled in Release builds for performance
public enum MPLog {
    
    /// Enable/disable logging at runtime (default: true in DEBUG)
    #if DEBUG
    public static var isEnabled: Bool = true
    #else
    public static var isEnabled: Bool = false
    #endif
    
    /// Log debug message
    public static func debug(_ message: @autoclosure () -> String) {
        guard isEnabled else { return }
        print("🐛 [MP] DEBUG:", message())
    }
    
    /// Log info message
    public static func info(_ message: @autoclosure () -> String) {
        guard isEnabled else { return }
        print("ℹ️ [MP] INFO:", message())
    }
    
    /// Log warning message
    public static func warning(_ message: @autoclosure () -> String) {
        guard isEnabled else { return }
        print("⚠️ [MP] WARN:", message())
    }
    
    /// Log error message
    public static func error(_ message: @autoclosure () -> String) {
        guard isEnabled else { return }
        print("❌ [MP] ERROR:", message())
    }
    
    /// Log success message
    public static func success(_ message: @autoclosure () -> String) {
        guard isEnabled else { return }
        print("✅ [MP] SUCCESS:", message())
    }
}
