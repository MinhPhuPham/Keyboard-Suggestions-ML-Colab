// ============================================================
// MPTypoCorrector.kt
// ============================================================
// Soundex phonetic matching + keyboard proximity typo detection
// NO SINGLETON - initialize through MPKeyboardSuggestion
// ============================================================

package com.mp.keyboard.suggestion.internal

import kotlin.math.abs
import kotlin.math.min

class MPTypoCorrector(private val resourceLoader: MPResourceLoader) {
    
    companion object {
        private val SOUNDEX_MAPPING = mapOf(
            'B' to '1', 'F' to '1', 'P' to '1', 'V' to '1',
            'C' to '2', 'G' to '2', 'J' to '2', 'K' to '2', 'Q' to '2', 'S' to '2', 'X' to '2', 'Z' to '2',
            'D' to '3', 'T' to '3',
            'L' to '4',
            'M' to '5', 'N' to '5',
            'R' to '6'
        )
    }
    
    fun getCorrections(typo: String, limit: Int = 5): List<Pair<String, Double>> {
        if (typo.isEmpty()) return emptyList()
        
        val lowerTypo = typo.lowercase()
        val candidates = mutableListOf<Triple<String, Double, Double>>()
        val seen = mutableSetOf<String>()
        
        // Method 1: Soundex
        val typoCode = soundex(lowerTypo)
        resourceLoader.getSoundexMatches(typoCode)?.forEach { (chunkName, localIndices) ->
            val chunk = MPVocabChunk.values().find { it.name.equals(chunkName, true) } ?: return@forEach
            val words = resourceLoader.getChunkWords(chunk) ?: return@forEach
            
            for (localIdx in localIndices.take(20)) {
                if (localIdx >= words.size) continue
                val word = words[localIdx]
                if (word.lowercase() != lowerTypo && !seen.contains(word.lowercase())) {
                    val dist = weightedEditDistance(lowerTypo, word.lowercase())
                    if (dist <= 4.0) {
                        val score = 100.0 / (dist + 0.5)
                        candidates.add(Triple(word, score, dist))
                        seen.add(word.lowercase())
                    }
                }
            }
        }
        
        // Method 2: Prefix + edit distance
        val prefix = lowerTypo.take(2)
        val prefixMatches = resourceLoader.getCompletions(prefix, limit = 50)
        
        for ((word, _, chunk) in prefixMatches) {
            if (!seen.contains(word.lowercase()) && abs(word.length - lowerTypo.length) <= 3) {
                val dist = weightedEditDistance(lowerTypo, word.lowercase())
                if (dist <= 2.5 && word.lowercase() != lowerTypo) {
                    val chunkBonus = if (chunk == MPVocabChunk.HIGH) 2.0 else 1.0
                    val score = (100.0 / (dist + 0.5)) * chunkBonus
                    candidates.add(Triple(word, score, dist))
                    seen.add(word.lowercase())
                }
            }
        }
        
        candidates.sortByDescending { it.second }
        return candidates.take(limit).map { it.first to it.second }
    }
    
    fun soundex(word: String): String {
        if (word.isEmpty()) return ""
        
        val upper = word.uppercase()
        val result = StringBuilder()
        result.append(upper[0])
        
        var prevCode = SOUNDEX_MAPPING[upper[0]] ?: '0'
        
        for (char in upper.drop(1)) {
            val code = SOUNDEX_MAPPING[char] ?: '0'
            if (code != '0' && code != prevCode) {
                result.append(code)
            }
            if (code != '0') prevCode = code
        }
        
        while (result.length < 4) result.append('0')
        return result.take(4).toString()
    }
    
    fun isAdjacentKey(c1: Char, c2: Char): Boolean {
        val adjacent = resourceLoader.getAdjacentKeys(c1)
        return adjacent.contains(c2.lowercaseChar().toString())
    }
    
    fun keyboardDistance(c1: Char, c2: Char): Double {
        if (c1 == c2) return 0.0
        return if (isAdjacentKey(c1, c2)) 0.5 else 1.0
    }
    
    fun weightedEditDistance(s1: String, s2: String, maxDist: Int = 4): Double {
        if (abs(s1.length - s2.length) > maxDist) return (maxDist + 1).toDouble()
        
        val m = s1.length
        val n = s2.length
        if (m == 0) return n.toDouble()
        if (n == 0) return m.toDouble()
        
        val dp = Array(m + 1) { DoubleArray(n + 1) }
        for (i in 0..m) dp[i][0] = i.toDouble()
        for (j in 0..n) dp[0][j] = j.toDouble()
        
        for (i in 1..m) {
            for (j in 1..n) {
                dp[i][j] = if (s1[i - 1] == s2[j - 1]) {
                    dp[i - 1][j - 1]
                } else {
                    val subCost = keyboardDistance(s1[i - 1], s2[j - 1])
                    min(min(dp[i - 1][j] + 1, dp[i][j - 1] + 1), dp[i - 1][j - 1] + subCost)
                }
            }
        }
        
        return dp[m][n]
    }
    
    fun isLikelyTypo(word: String): Boolean {
        if (resourceLoader.getWordIndex(word.lowercase()) != null) return false
        return getCorrections(word, 1).isNotEmpty()
    }
}
