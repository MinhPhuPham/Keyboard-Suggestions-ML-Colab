// ============================================================
// MPLearningManager.kt
// ============================================================
// User behavior learning: unigram boost, bigram context, recency
// NO SINGLETON - initialize through MPKeyboardSuggestion
// ============================================================

package com.mp.keyboard.suggestion.internal

import android.content.Context
import android.content.SharedPreferences
import kotlinx.coroutines.*
import org.json.JSONObject
import kotlin.math.pow

data class MPLearningStats(
    val words: Int,
    val bigrams: Int
)

class MPLearningManager(context: Context, prefsName: String = "mp_keyboard_learning") {
    
    companion object {
        private const val UNIGRAM_KEY = "unigram"
        private const val BIGRAM_KEY = "bigram"
        private const val LAST_USED_KEY = "last_used"
    }
    
    private val prefs: SharedPreferences = context.getSharedPreferences(prefsName, Context.MODE_PRIVATE)
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    
    private val unigramCounts = mutableMapOf<String, Int>()
    private val bigramCounts = mutableMapOf<String, MutableMap<String, Int>>()
    private val lastUsed = mutableMapOf<String, Long>()
    
    private var lastContext = ""
    private var isDirty = false
    private var saveJob: Job? = null
    
    var recencyDecayDays = 30.0
    var maxEntries = 5000
    
    init { load() }
    
    // Public API
    
    fun recordSelection(word: String, context: String? = null) {
        val lowerWord = word.lowercase()
        val contextWord = (context ?: lastContext).lowercase().split(" ").lastOrNull() ?: ""
        
        unigramCounts[lowerWord] = (unigramCounts[lowerWord] ?: 0) + 1
        
        if (contextWord.isNotEmpty()) {
            if (!bigramCounts.containsKey(contextWord)) bigramCounts[contextWord] = mutableMapOf()
            bigramCounts[contextWord]!![lowerWord] = (bigramCounts[contextWord]!![lowerWord] ?: 0) + 1
        }
        
        lastUsed[lowerWord] = System.currentTimeMillis()
        lastContext = lowerWord
        
        isDirty = true
        scheduleSave()
        
        if (unigramCounts.size > maxEntries) prune()
    }
    
    fun getBoostScore(word: String, context: String): Double {
        val lowerWord = word.lowercase()
        val contextWord = context.lowercase().split(" ").lastOrNull() ?: ""
        
        var score = 0.0
        
        unigramCounts[lowerWord]?.let { score += it * 10.0 }
        
        if (contextWord.isNotEmpty()) {
            bigramCounts[contextWord]?.get(lowerWord)?.let { score += it * 50.0 }
        }
        
        lastUsed[lowerWord]?.let { timestamp ->
            val daysSince = (System.currentTimeMillis() - timestamp) / 86400000.0
            val decay = 0.5.pow(daysSince / recencyDecayDays)
            score *= (0.5 + 0.5 * decay)
        }
        
        return score
    }
    
    fun getBigramSuggestions(contextWord: String, limit: Int = 5): List<Pair<String, Int>> {
        return bigramCounts[contextWord.lowercase()]
            ?.entries
            ?.sortedByDescending { it.value }
            ?.take(limit)
            ?.map { it.key to it.value }
            ?: emptyList()
    }
    
    fun updateContext(context: String) { lastContext = context }
    
    fun getStats(): MPLearningStats {
        val bigramCount = bigramCounts.values.sumOf { it.size }
        return MPLearningStats(unigramCounts.size, bigramCount)
    }
    
    fun clear() {
        unigramCounts.clear()
        bigramCounts.clear()
        lastUsed.clear()
        prefs.edit().clear().apply()
    }
    
    fun destroy() { scope.cancel() }
    
    // Private
    
    private fun load() {
        try {
            prefs.getString(UNIGRAM_KEY, null)?.let { json ->
                val obj = JSONObject(json)
                obj.keys().forEach { unigramCounts[it] = obj.getInt(it) }
            }
            
            prefs.getString(BIGRAM_KEY, null)?.let { json ->
                val obj = JSONObject(json)
                obj.keys().forEach { ctx ->
                    bigramCounts[ctx] = mutableMapOf()
                    val inner = obj.getJSONObject(ctx)
                    inner.keys().forEach { word -> bigramCounts[ctx]!![word] = inner.getInt(word) }
                }
            }
            
            prefs.getString(LAST_USED_KEY, null)?.let { json ->
                val obj = JSONObject(json)
                obj.keys().forEach { lastUsed[it] = obj.getLong(it) }
            }
        } catch (e: Exception) { }
    }
    
    private fun save() {
        if (!isDirty) return
        scope.launch {
            try {
                prefs.edit().apply {
                    putString(UNIGRAM_KEY, JSONObject(unigramCounts as Map<*, *>).toString())
                    
                    val bigramJson = JSONObject()
                    bigramCounts.forEach { (ctx, words) -> bigramJson.put(ctx, JSONObject(words as Map<*, *>)) }
                    putString(BIGRAM_KEY, bigramJson.toString())
                    
                    putString(LAST_USED_KEY, JSONObject(lastUsed as Map<*, *>).toString())
                    apply()
                }
                isDirty = false
            } catch (e: Exception) { }
        }
    }
    
    private fun scheduleSave() {
        saveJob?.cancel()
        saveJob = scope.launch {
            delay(5000)
            save()
        }
    }
    
    private fun prune() {
        val cutoff = System.currentTimeMillis() - 90 * 86400000L
        val toRemove = lastUsed.filter { it.value < cutoff && (unigramCounts[it.key] ?: 0) < 3 }.keys
        toRemove.forEach { unigramCounts.remove(it); lastUsed.remove(it) }
        
        bigramCounts.entries.toList().forEach { (ctx, words) ->
            bigramCounts[ctx] = words.filter { it.value >= 2 }.toMutableMap()
            if (bigramCounts[ctx]?.isEmpty() == true) bigramCounts.remove(ctx)
        }
    }
}
