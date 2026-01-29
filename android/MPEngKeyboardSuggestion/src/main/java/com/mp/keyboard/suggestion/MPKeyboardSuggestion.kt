// ============================================================
// MPKeyboardSuggestion.kt
// ============================================================
// Main public API for MPEngKeyboardSuggestion package
// NO INTERNAL SINGLETONS - user manages singleton if needed
// ============================================================

package com.mp.keyboard.suggestion

import android.content.Context
import com.mp.keyboard.suggestion.internal.*

// MARK: - Public Models

enum class MPSuggestionSource {
    GRU, TRIE, LEARNING, TYPO, SHORTCUT, HYBRID
}

data class MPSuggestion(
    val word: String,
    val score: Double,
    val source: MPSuggestionSource
)

data class MPStats(
    val vocabSize: Int,
    val learnedWords: Int,
    val learnedBigrams: Int,
    val shortcuts: Int,
    val gruReady: Boolean
)

/**
 * High-performance English keyboard suggestion library
 * 
 * Uses hybrid ML + Trie approach for context-aware suggestions.
 * NO SINGLETON - create your own instance or manage singleton externally.
 *
 * ## Example
 * ```kotlin
 * val keyboard = MPKeyboardSuggestion(context)
 * val suggestions = keyboard.getSuggestions("how are y")
 * keyboard.recordSelection("you", "how are")
 * ```
 */
class MPKeyboardSuggestion(context: Context) {
    
    // Internal components (owned, not singletons)
    private val resourceLoader: MPResourceLoader
    private val modelInference: MPModelInference
    private val trieHelper: MPTrieHelper
    private val learningManager: MPLearningManager
    private val typoCorrector: MPTypoCorrector
    private val shortcutManager: MPShortcutManager
    
    // Configuration
    var gruWeight: Double = 0.7
    var useGRU: Boolean = true
    var useLearning: Boolean = true
    
    init {
        resourceLoader = MPResourceLoader(context)
        modelInference = MPModelInference(context, resourceLoader)
        trieHelper = MPTrieHelper(resourceLoader)
        learningManager = MPLearningManager(context)
        typoCorrector = MPTypoCorrector(resourceLoader)
        shortcutManager = MPShortcutManager(context)
    }
    
    // MARK: - Main API
    
    fun getSuggestions(input: String, limit: Int = 5): List<MPSuggestion> {
        if (input.isEmpty()) return emptyList()
        
        // Next-word prediction
        if (input.endsWith(" ")) {
            return getNextWordSuggestions(input.trim(), limit)
        }
        
        val words = input.split(" ")
        val lastWord = words.lastOrNull() ?: input
        val context = words.dropLast(1).joinToString(" ")
        
        // Shortcuts first
        getShortcutSuggestions(lastWord, limit)?.let { return it }
        
        // Typo correction
        if (typoCorrector.isLikelyTypo(lastWord)) {
            return getTypoSuggestions(lastWord, context, limit)
        }
        
        // Normal completion
        return getCompletionSuggestions(lastWord, context, limit)
    }
    
    fun getSuggestionWords(input: String, limit: Int = 5): List<String> {
        return getSuggestions(input, limit).map { it.word }
    }
    
    fun recordSelection(word: String, context: String? = null) {
        learningManager.recordSelection(word, context)
    }
    
    // MARK: - Shortcuts API
    
    fun addShortcut(input: String, suggestion: String) = shortcutManager.addShortcut(input, suggestion)
    fun addShortcuts(dict: Map<String, String>) = shortcutManager.addShortcuts(dict)
    fun removeShortcut(input: String) = shortcutManager.removeShortcut(input)
    fun getAllShortcuts(): Map<String, List<String>> = shortcutManager.getAllShortcuts()
    
    // MARK: - Resource Management
    
    fun preload(onComplete: (() -> Unit)? = null) = resourceLoader.preloadAllChunks(onComplete)
    fun releaseMemory() = resourceLoader.releaseMemory()
    fun clearLearningData() = learningManager.clear()
    
    fun getStats(): MPStats {
        val learningStats = learningManager.getStats()
        return MPStats(
            resourceLoader.vocabSize,
            learningStats.words,
            learningStats.bigrams,
            shortcutManager.count,
            modelInference.isReady
        )
    }
    
    fun destroy() {
        resourceLoader.destroy()
        modelInference.destroy()
        learningManager.destroy()
        shortcutManager.destroy()
    }
    
    // MARK: - Private Methods
    
    private fun getNextWordSuggestions(context: String, limit: Int): List<MPSuggestion> {
        val candidates = mutableListOf<Triple<String, Double, MPSuggestionSource>>()
        
        if (useGRU && modelInference.isReady) {
            modelInference.predictNextWord(context, limit * 2).forEach { (word, prob) ->
                candidates.add(Triple(word, prob.toDouble() * 100, MPSuggestionSource.GRU))
            }
        }
        
        if (useLearning) {
            val contextWord = context.split(" ").lastOrNull() ?: ""
            learningManager.getBigramSuggestions(contextWord, 5).forEach { (word, count) ->
                candidates.add(Triple(word, count.toDouble() * 50, MPSuggestionSource.LEARNING))
            }
        }
        
        return combineCandidates(candidates, context, limit)
    }
    
    private fun getCompletionSuggestions(prefix: String, context: String, limit: Int): List<MPSuggestion> {
        val candidates = mutableListOf<Triple<String, Double, MPSuggestionSource>>()
        
        if (useGRU && modelInference.isReady && context.isNotEmpty()) {
            modelInference.predictWithPrefix(context, prefix, limit * 2).forEach { (word, prob) ->
                candidates.add(Triple(word, prob.toDouble() * 100, MPSuggestionSource.GRU))
            }
        }
        
        trieHelper.searchPrefix(prefix, limit * 2).forEach { (word, score) ->
            candidates.add(Triple(word, score, MPSuggestionSource.TRIE))
        }
        
        if (useLearning) {
            val contextWord = context.split(" ").lastOrNull() ?: ""
            learningManager.getBigramSuggestions(contextWord, 10)
                .filter { it.first.lowercase().startsWith(prefix.lowercase()) }
                .forEach { (word, count) ->
                    candidates.add(Triple(word, count.toDouble() * 30, MPSuggestionSource.LEARNING))
                }
        }
        
        return combineCandidates(candidates, context, limit)
    }
    
    private fun getTypoSuggestions(typo: String, context: String, limit: Int): List<MPSuggestion> {
        val corrections = typoCorrector.getCorrections(typo, limit * 2)
        if (corrections.isEmpty()) return emptyList()
        
        if (useGRU && modelInference.isReady && context.isNotEmpty()) {
            val words = corrections.map { it.first }
            val origScores = corrections.toMap()
            val reranked = modelInference.rerank(words, context, origScores, gruWeight)
            
            return reranked.take(limit).map { MPSuggestion(it.first, it.second, MPSuggestionSource.TYPO) }
        }
        
        return corrections.take(limit).map { MPSuggestion(it.first, it.second, MPSuggestionSource.TYPO) }
    }
    
    private fun getShortcutSuggestions(input: String, limit: Int): List<MPSuggestion>? {
        shortcutManager.getSuggestions(input)?.let { suggestions ->
            return suggestions.take(limit).mapIndexed { idx, text ->
                MPSuggestion(text, 1000 - idx.toDouble(), MPSuggestionSource.SHORTCUT)
            }
        }
        
        val partials = shortcutManager.getPartialMatches(input)
        if (partials.isNotEmpty()) {
            val results = mutableListOf<MPSuggestion>()
            for ((_, sgs) in partials.take(2)) {
                for (s in sgs.take(2)) {
                    results.add(MPSuggestion(s, 500.0, MPSuggestionSource.SHORTCUT))
                }
            }
            if (results.isNotEmpty()) return results
        }
        
        return null
    }
    
    private fun combineCandidates(candidates: List<Triple<String, Double, MPSuggestionSource>>, context: String, limit: Int): List<MPSuggestion> {
        val scoreMap = mutableMapOf<String, Pair<Double, MPSuggestionSource>>()
        
        for ((word, score, source) in candidates) {
            val key = word.lowercase()
            var finalScore = score
            
            if (useLearning) {
                finalScore += learningManager.getBoostScore(word, context)
            }
            
            if (scoreMap.containsKey(key)) {
                val existing = scoreMap[key]!!
                scoreMap[key] = (existing.first + finalScore) to MPSuggestionSource.HYBRID
            } else {
                scoreMap[key] = finalScore to source
            }
        }
        
        return scoreMap.entries
            .sortedByDescending { it.value.first }
            .take(limit)
            .map { MPSuggestion(it.key, it.value.first, it.value.second) }
    }
}
