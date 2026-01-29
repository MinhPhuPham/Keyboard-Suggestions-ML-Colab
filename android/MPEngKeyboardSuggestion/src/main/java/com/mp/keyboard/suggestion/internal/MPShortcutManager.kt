// ============================================================
// MPShortcutManager.kt
// ============================================================
// Custom user shortcuts (input → suggestions)
// NO SINGLETON - initialize through MPKeyboardSuggestion
// ============================================================

package com.mp.keyboard.suggestion.internal

import android.content.Context
import android.content.SharedPreferences
import kotlinx.coroutines.*
import org.json.JSONObject

class MPShortcutManager(context: Context, prefsName: String = "mp_keyboard_shortcuts") {
    
    companion object {
        private const val DATA_KEY = "shortcuts"
    }
    
    private val prefs: SharedPreferences = context.getSharedPreferences(prefsName, Context.MODE_PRIVATE)
    private val shortcuts = mutableMapOf<String, MutableList<String>>()
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    
    val count: Int get() = shortcuts.size
    
    init { load() }
    
    // Public API
    
    fun addShortcut(input: String, suggestion: String) {
        val key = input.lowercase()
        if (!shortcuts.containsKey(key)) shortcuts[key] = mutableListOf()
        if (!shortcuts[key]!!.contains(suggestion)) shortcuts[key]!!.add(suggestion)
        save()
    }
    
    fun addShortcuts(dict: Map<String, String>) {
        dict.forEach { (input, suggestion) -> addShortcut(input, suggestion) }
    }
    
    fun addShortcut(input: String, suggestions: List<String>) {
        shortcuts[input.lowercase()] = suggestions.toMutableList()
        save()
    }
    
    fun removeShortcut(input: String) {
        shortcuts.remove(input.lowercase())
        save()
    }
    
    fun getSuggestions(input: String): List<String>? = shortcuts[input.lowercase()]
    
    fun getPartialMatches(query: String): List<Pair<String, List<String>>> {
        val lowerQuery = query.lowercase()
        return shortcuts.filter { it.key.startsWith(lowerQuery) && it.key != lowerQuery }
            .map { it.key to it.value.toList() }
    }
    
    fun hasShortcut(input: String): Boolean = shortcuts.containsKey(input.lowercase())
    
    fun getAllShortcuts(): Map<String, List<String>> = shortcuts.mapValues { it.value.toList() }
    
    fun clear() {
        shortcuts.clear()
        prefs.edit().clear().apply()
    }
    
    fun destroy() { scope.cancel() }
    
    // Private
    
    private fun load() {
        try {
            prefs.getString(DATA_KEY, null)?.let { json ->
                val obj = JSONObject(json)
                obj.keys().forEach { key ->
                    val arr = obj.getJSONArray(key)
                    shortcuts[key] = (0 until arr.length()).map { arr.getString(it) }.toMutableList()
                }
            }
        } catch (e: Exception) { }
    }
    
    private fun save() {
        scope.launch {
            try {
                val obj = JSONObject()
                shortcuts.forEach { (key, list) ->
                    obj.put(key, org.json.JSONArray(list))
                }
                prefs.edit().putString(DATA_KEY, obj.toString()).apply()
            } catch (e: Exception) { }
        }
    }
}
