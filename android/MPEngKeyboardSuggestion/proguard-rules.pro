# Add project specific ProGuard rules here.

# Keep TensorFlow Lite classes
-keep class org.tensorflow.** { *; }
-dontwarn org.tensorflow.**

# Keep our public API
-keep class com.mp.keyboard.suggestion.MPKeyboardSuggestion { *; }
-keep class com.mp.keyboard.suggestion.MPSuggestion { *; }
-keep class com.mp.keyboard.suggestion.MPSuggestionSource { *; }
-keep class com.mp.keyboard.suggestion.MPStats { *; }
