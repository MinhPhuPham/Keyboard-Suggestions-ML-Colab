// ============================================================
// MPModelInference.kt
// ============================================================
// TFLite wrapper for GRU keyboard model inference
// NO SINGLETON - initialize through MPKeyboardSuggestion
// ============================================================

package com.mp.keyboard.suggestion.internal

import android.content.Context
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MPModelInference(
    context: Context,
    private val resourceLoader: MPResourceLoader
) {
    
    companion object {
        private const val MODEL_FILE = "gru_model_fp16.tflite"
        private const val SEQUENCE_LENGTH = 10
    }
    
    private var interpreter: Interpreter? = null
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    
    val isReady: Boolean get() = interpreter != null
    
    init {
        loadModel(context)
    }
    
    private fun loadModel(context: Context) {
        scope.launch {
            try {
                val modelBuffer = loadModelFile(context)
                val options = Interpreter.Options().apply {
                    setNumThreads(4)
                    setUseXNNPACK(true)
                }
                interpreter = Interpreter(modelBuffer, options)
            } catch (e: Exception) {
                android.util.Log.e("MPModelInference", "Model load failed: ${e.message}")
            }
        }
    }
    
    private fun loadModelFile(context: Context): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(MODEL_FILE)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    fun tokenize(text: String): List<Int> {
        val words = text.lowercase().split(" ")
        return words.mapNotNull { resourceLoader.getWordIndex(it) }
    }
    
    fun predictNextWord(context: String, topK: Int = 10): List<Pair<String, Float>> {
        val interp = interpreter ?: return emptyList()
        
        var sequence = tokenize(context)
        if (sequence.size < SEQUENCE_LENGTH) {
            sequence = List(SEQUENCE_LENGTH - sequence.size) { 0 } + sequence
        } else if (sequence.size > SEQUENCE_LENGTH) {
            sequence = sequence.takeLast(SEQUENCE_LENGTH)
        }
        
        val inputBuffer = ByteBuffer.allocateDirect(SEQUENCE_LENGTH * 4).apply {
            order(ByteOrder.nativeOrder())
            sequence.forEach { putInt(it) }
            rewind()
        }
        
        val vocabSize = resourceLoader.vocabSize.coerceAtLeast(25000)
        val outputBuffer = ByteBuffer.allocateDirect(vocabSize * 4).apply {
            order(ByteOrder.nativeOrder())
        }
        
        try {
            interp.run(inputBuffer, outputBuffer)
            
            outputBuffer.rewind()
            val probs = mutableListOf<Pair<Int, Float>>()
            for (i in 0 until vocabSize) {
                val prob = outputBuffer.float
                if (prob > 0.001f) {
                    probs.add(i to prob)
                }
            }
            
            probs.sortByDescending { it.second }
            
            return probs.take(topK).mapNotNull { (idx, prob) ->
                resourceLoader.getWord(idx)?.let { it to prob }
            }
            
        } catch (e: Exception) {
            return emptyList()
        }
    }
    
    fun predictWithPrefix(context: String, prefix: String, topK: Int = 5): List<Pair<String, Float>> {
        val predictions = predictNextWord(context, topK * 40)
        val lowerPrefix = prefix.lowercase()
        return predictions.filter { it.first.lowercase().startsWith(lowerPrefix) }.take(topK)
    }
    
    fun rerank(
        candidates: List<String>,
        context: String,
        originalScores: Map<String, Double>? = null,
        contextWeight: Double = 0.7
    ): List<Pair<String, Double>> {
        val interp = interpreter ?: return candidates.map { it to (originalScores?.get(it) ?: 1.0) }
        
        var sequence = tokenize(context)
        if (sequence.size < SEQUENCE_LENGTH) {
            sequence = List(SEQUENCE_LENGTH - sequence.size) { 0 } + sequence
        } else {
            sequence = sequence.takeLast(SEQUENCE_LENGTH)
        }
        
        val inputBuffer = ByteBuffer.allocateDirect(SEQUENCE_LENGTH * 4).apply {
            order(ByteOrder.nativeOrder())
            sequence.forEach { putInt(it) }
            rewind()
        }
        
        val vocabSize = resourceLoader.vocabSize.coerceAtLeast(25000)
        val outputBuffer = ByteBuffer.allocateDirect(vocabSize * 4).apply {
            order(ByteOrder.nativeOrder())
        }
        
        try {
            interp.run(inputBuffer, outputBuffer)
            outputBuffer.rewind()
            
            return candidates.map { word ->
                val wordIdx = resourceLoader.getWordIndex(word.lowercase())
                val gruProb = if (wordIdx != null && wordIdx < vocabSize) {
                    outputBuffer.position(wordIdx * 4)
                    outputBuffer.float.toDouble() * 100
                } else 0.0
                
                val origScore = originalScores?.get(word) ?: 1.0
                val combined = origScore * (1 - contextWeight) + gruProb * contextWeight
                word to combined
            }.sortedByDescending { it.second }
            
        } catch (e: Exception) {
            return candidates.map { it to (originalScores?.get(it) ?: 1.0) }
        }
    }
    
    fun destroy() {
        scope.cancel()
        interpreter?.close()
        interpreter = null
    }
}
