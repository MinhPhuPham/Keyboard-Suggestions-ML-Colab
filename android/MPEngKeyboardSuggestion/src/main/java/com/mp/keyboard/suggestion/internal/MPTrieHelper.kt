// ============================================================
// MPTrieHelper.kt
// ============================================================
// Fast prefix search using flat array index
// NO SINGLETON - initialize through MPKeyboardSuggestion
// ============================================================

package com.mp.keyboard.suggestion.internal

class MPTrieHelper(private val resourceLoader: MPResourceLoader) {
    
    fun searchPrefix(prefix: String, limit: Int = 10): List<Pair<String, Double>> {
        if (prefix.isEmpty()) return emptyList()
        
        val completions = resourceLoader.getCompletions(prefix.lowercase(), limit = limit * 2)
        
        return completions.mapIndexed { idx, (word, localIdx, chunk) ->
            val chunkBonus = when (chunk) {
                MPVocabChunk.HIGH -> 100.0
                MPVocabChunk.MEDIUM -> 50.0
                MPVocabChunk.LOW -> 10.0
            }
            val positionPenalty = idx * 0.5
            val score = chunkBonus - positionPenalty + (100.0 / (localIdx + 1))
            word to score
        }.sortedByDescending { it.second }.take(limit)
    }
    
    fun searchPrefixWithLength(prefix: String, minLength: Int, maxLength: Int, limit: Int = 20): List<Pair<String, Double>> {
        return searchPrefix(prefix, limit * 2)
            .filter { it.first.length in minLength..maxLength }
            .take(limit)
    }
    
    fun wordExists(word: String): Boolean {
        return resourceLoader.getWordIndex(word.lowercase()) != null
    }
    
    fun getWordRank(word: String): Int? {
        return resourceLoader.getWordIndex(word.lowercase())
    }
    
    fun fuzzySearch(query: String, tolerance: Int = 2, limit: Int = 10): List<Pair<String, Double>> {
        if (query.length < 2) return emptyList()
        
        for (prefixLen in query.length downTo maxOf(1, query.length - tolerance)) {
            val prefix = query.take(prefixLen)
            val results = searchPrefix(prefix, limit)
            if (results.isNotEmpty()) return results
        }
        
        return emptyList()
    }
}
