// ============================================================
// MPJPLog.swift
// ============================================================
// Simple logger for MPJPKeyboardSuggestion
// ============================================================

import Foundation
import os.log

public enum MPJPLog {
    private static let subsystem = "com.mp.jp.keyboard"
    private static let logger = Logger(subsystem: subsystem, category: "MPJP")
    
    /// Enable debug logging (default: true for diagnostics)
    public static var isDebugEnabled = true
    
    public static func debug(_ message: String) {
        if isDebugEnabled {
            print("🔍 [MPJP] \(message)")
            logger.debug("\(message)")
        }
    }
    
    public static func info(_ message: String) {
        print("ℹ️ [MPJP] \(message)")
        logger.info("\(message)")
    }
    
    public static func error(_ message: String) {
        print("❌ [MPJP] \(message)")
        logger.error("\(message)")
    }
    
    public static func warn(_ message: String) {
        print("⚠️ [MPJP] \(message)")
        logger.warning("\(message)")
    }
}
