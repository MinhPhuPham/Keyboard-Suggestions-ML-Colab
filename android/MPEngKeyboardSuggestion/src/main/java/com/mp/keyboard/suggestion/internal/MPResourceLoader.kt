// ============================================================
// MPResourceLoader.kt
// ============================================================
// Handles loading vocabulary chunks, indices, and config files
// NO SINGLETON - initialize through MPKeyboardSuggestion
// ============================================================

package com.mp.keyboard.suggestion.internal

import android.content.Context
import kotlinx.coroutines.*
import org.json.JSONObject

enum class MPVocabChunk(val fileName: String) {
    HIGH("vocab_high.json"),
    MEDIUM("vocab_medium.json"),
    LOW("vocab_low.json")
}

data class MPChunkInfo(
    val file: String,
    val startIndex: Int,
    val endIndex: Int,
    val wordCount: Int
)

data class MPPrefixIndex(
    val version: String,
    val chunks: Map<String, MPChunkInfo>,
    val prefixes: Map<String, Map<String, List<Int>>>
)

class MPResourceLoader(private val context: Context) {
    
    private val loadedChunks = mutableMapOf<MPVocabChunk, List<String>>()
    private var prefixIndex: MPPrefixIndex? = null
    private var soundexIndex: Map<String, Map<String, List<Int>>>? = null
    private var keyboardAdjacent = mapOf<String, String>()
    private var wordToIndex = mapOf<String, Int>()
    private var indexToWord = mapOf<Int, String>()
    
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private val loadingChunks = mutableSetOf<MPVocabChunk>()
    
    init {
        loadPrefixIndex()
        loadKeyboardAdjacent()
        loadWordMappings()
        loadChunk(MPVocabChunk.HIGH)
    }
    
    // Public API
    
    fun getChunkWords(chunk: MPVocabChunk): List<String>? {
        if (loadedChunks[chunk] == null) {
            loadChunk(chunk)
        }
        return loadedChunks[chunk]
    }
    
    fun getCompletions(prefix: String, chunks: List<MPVocabChunk> = MPVocabChunk.values().toList(), limit: Int = 50): List<Triple<String, Int, MPVocabChunk>> {
        val prefixData = prefixIndex?.prefixes?.get(prefix.lowercase()) ?: return searchLoadedChunks(prefix, limit)
        
        val results = mutableListOf<Triple<String, Int, MPVocabChunk>>()
        
        for (chunk in chunks) {
            val localIndices = prefixData[chunk.name.lowercase()] ?: continue
            val chunkWords = getChunkWords(chunk) ?: continue
            
            for (localIdx in localIndices) {
                if (localIdx < chunkWords.size) {
                    results.add(Triple(chunkWords[localIdx], localIdx, chunk))
                    if (results.size >= limit) return results
                }
            }
        }
        
        return results
    }
    
    fun getSoundexMatches(code: String): Map<String, List<Int>>? {
        if (soundexIndex == null) loadSoundexIndex()
        return soundexIndex?.get(code)
    }
    
    fun getAdjacentKeys(key: Char): String = keyboardAdjacent[key.lowercaseChar().toString()] ?: ""
    
    fun getWordIndex(word: String): Int? = wordToIndex[word.lowercase()]
    
    fun getWord(index: Int): String? = indexToWord[index]
    
    val vocabSize: Int get() = wordToIndex.size
    
    fun loadChunkAsync(chunk: MPVocabChunk, onComplete: (() -> Unit)? = null) {
        if (loadedChunks.containsKey(chunk)) { onComplete?.invoke(); return }
        scope.launch {
            loadChunk(chunk)
            withContext(Dispatchers.Main) { onComplete?.invoke() }
        }
    }
    
    fun preloadAllChunks(onComplete: (() -> Unit)? = null) {
        scope.launch {
            loadChunk(MPVocabChunk.MEDIUM)
            loadChunk(MPVocabChunk.LOW)
            loadSoundexIndex()
            withContext(Dispatchers.Main) { onComplete?.invoke() }
        }
    }
    
    fun releaseMemory(keepHigh: Boolean = true) {
        if (!keepHigh) loadedChunks.remove(MPVocabChunk.HIGH)
        loadedChunks.remove(MPVocabChunk.MEDIUM)
        loadedChunks.remove(MPVocabChunk.LOW)
        soundexIndex = null
    }
    
    fun destroy() { scope.cancel() }
    
    // Private loading
    
    private fun loadPrefixIndex() {
        try {
            val json = context.assets.open("prefix_index.json").bufferedReader().use { it.readText() }
            val obj = JSONObject(json)
            
            val chunks = mutableMapOf<String, MPChunkInfo>()
            val chunksObj = obj.optJSONObject("chunks")
            chunksObj?.keys()?.forEach { key ->
                val c = chunksObj.getJSONObject(key)
                chunks[key] = MPChunkInfo(c.getString("file"), c.getInt("startIndex"), c.getInt("endIndex"), c.getInt("wordCount"))
            }
            
            val prefixes = mutableMapOf<String, Map<String, List<Int>>>()
            val prefixesObj = obj.optJSONObject("prefixes")
            prefixesObj?.keys()?.forEach { prefix ->
                val chunkData = mutableMapOf<String, List<Int>>()
                val prefixChunks = prefixesObj.getJSONObject(prefix)
                prefixChunks.keys().forEach { chunkName ->
                    val arr = prefixChunks.getJSONArray(chunkName)
                    chunkData[chunkName] = (0 until arr.length()).map { arr.getInt(it) }
                }
                prefixes[prefix] = chunkData
            }
            
            prefixIndex = MPPrefixIndex(obj.optString("version", "3.0"), chunks, prefixes)
        } catch (e: Exception) { }
    }
    
    private fun loadChunk(chunk: MPVocabChunk) {
        if (loadedChunks.containsKey(chunk) || loadingChunks.contains(chunk)) return
        loadingChunks.add(chunk)
        
        try {
            val json = context.assets.open(chunk.fileName).bufferedReader().use { it.readText() }
            val obj = JSONObject(json)
            val arr = obj.getJSONArray("words")
            loadedChunks[chunk] = (0 until arr.length()).map { arr.getString(it) }
        } catch (e: Exception) { }
        
        loadingChunks.remove(chunk)
    }
    
    private fun loadSoundexIndex() {
        try {
            val json = context.assets.open("soundex_index.json").bufferedReader().use { it.readText() }
            val obj = JSONObject(json)
            soundexIndex = obj.keys().asSequence().associate { code ->
                code to obj.getJSONObject(code).keys().asSequence().associate { chunk ->
                    val arr = obj.getJSONObject(code).getJSONArray(chunk)
                    chunk to (0 until arr.length()).map { arr.getInt(it) }
                }
            }
        } catch (e: Exception) { }
    }
    
    private fun loadKeyboardAdjacent() {
        try {
            val json = context.assets.open("keyboard_adjacent.json").bufferedReader().use { it.readText() }
            val obj = JSONObject(json)
            keyboardAdjacent = obj.keys().asSequence().associateWith { obj.getString(it) }
        } catch (e: Exception) { }
    }
    
    private fun loadWordMappings() {
        try {
            val json = context.assets.open("word_to_index.json").bufferedReader().use { it.readText() }
            val obj = JSONObject(json)
            wordToIndex = obj.keys().asSequence().associateWith { obj.getInt(it) }
            indexToWord = wordToIndex.entries.associate { it.value to it.key }
        } catch (e: Exception) { }
    }
    
    private fun searchLoadedChunks(prefix: String, limit: Int): List<Triple<String, Int, MPVocabChunk>> {
        val results = mutableListOf<Triple<String, Int, MPVocabChunk>>()
        val lowerPrefix = prefix.lowercase()
        
        for (chunk in MPVocabChunk.values()) {
            loadedChunks[chunk]?.forEachIndexed { idx, word ->
                if (word.lowercase().startsWith(lowerPrefix)) {
                    results.add(Triple(word, idx, chunk))
                    if (results.size >= limit) return results
                }
            }
        }
        return results
    }
}
