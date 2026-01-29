#!/usr/bin/env python3
"""
Interactive Model Testing Script for GRU Keyboard Suggestions

Usage:
    python test_model_prediction.py [--model-dir path/to/models]

Features:
- Word completion: "hel" → ["hello", "help", "helping"]
- Next-word prediction: "how are " → ["you", "they", "we"]
- Configurable top-k (3 or 30 suggestions)
- Real-time timing information
- Type ':q' to quit
"""

import tensorflow as tf
import keras
import numpy as np
import json
import time
import sys
import os
import argparse


# Custom Perplexity metric (needed to load model that was saved with this)
@keras.saving.register_keras_serializable()
class Perplexity(tf.keras.metrics.Metric):
    """Custom perplexity metric."""
    
    def __init__(self, name='perplexity', **kwargs):
        super().__init__(name=name, **kwargs)
        self.cross_entropy = self.add_weight(name='cross_entropy', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        self.cross_entropy.assign_add(tf.reduce_sum(loss))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
    
    def result(self):
        return tf.exp(self.cross_entropy / self.count)
    
    def reset_state(self):
        self.cross_entropy.assign(0.)
        self.count.assign(0.)


class GRUKeyboardTester:
    """Interactive tester for GRU keyboard suggestion model."""
    
    def __init__(self, model_dir: str = "./models"):
        """
        Initialize the tester with a trained GRU model.
        
        Args:
            model_dir: Directory containing the model and word_index.json
        """
        print(f"Loading GRU model from: {model_dir}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Load model (compile=False since we only need inference)
        model_path = os.path.join(model_dir, "gru_model.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = tf.keras.models.load_model(model_path, compile=False)
        
        # Load word_to_index mapping
        w2i_path = os.path.join(model_dir, "word_index.json")
        if not os.path.exists(w2i_path):
            raise FileNotFoundError(f"Word index not found: {w2i_path}")
        
        with open(w2i_path, 'r') as f:
            self.word_to_index = json.load(f)
        
        # Create reverse mapping (index to word)
        self.index_to_word = {v: k for k, v in self.word_to_index.items()}
        
        # Get model config
        self.sequence_length = self.model.input_shape[1]  # Usually 10
        self.vocab_size = self.model.output_shape[-1]
        
        load_time = (time.time() - start_time) * 1000
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Load time: {load_time:.2f} ms")
        print(f"✓ Vocab size: {self.vocab_size:,}")
        print(f"✓ Sequence length: {self.sequence_length}")
        print(f"✓ Word mappings: {len(self.word_to_index):,}")
        print("=" * 60)
        print()
    
    def tokenize(self, text: str) -> list:
        """Convert text to token indices."""
        words = text.lower().strip().split()
        return [self.word_to_index.get(w, 0) for w in words]
    
    def predict(self, input_text: str, top_k: int = 3) -> tuple:
        """
        Generate predictions using GRU model.
        
        Args:
            input_text: The input text
            top_k: Number of suggestions to return
            
        Returns:
            Tuple of (predictions_list, confidences, inference_time_ms, is_completion)
        """
        start_time = time.time()
        
        # Detect if this is word completion or next-word prediction
        words = input_text.strip().split()
        is_completion = not input_text.endswith(' ')
        prefix_to_complete = ""
        
        if is_completion and words:
            prefix_to_complete = words[-1].lower()
            # For completion, use context without the incomplete word
            context_words = words[:-1]
        else:
            context_words = words
        
        # Tokenize context
        sequence = self.tokenize(' '.join(context_words))
        
        # Pad/truncate to sequence_length
        if len(sequence) < self.sequence_length:
            sequence = [0] * (self.sequence_length - len(sequence)) + sequence
        else:
            sequence = sequence[-self.sequence_length:]
        
        # Predict
        input_array = np.array([sequence], dtype=np.float32)
        predictions = self.model.predict(input_array, verbose=0)[0]
        
        # Get top predictions
        top_indices = np.argsort(predictions)[::-1]
        
        # Filter and collect results
        results = []
        for idx in top_indices:
            if idx == 0:  # Skip padding token
                continue
            
            word = self.index_to_word.get(idx, None)
            if word is None:
                continue
            
            prob = predictions[idx] * 100
            
            # For completion, filter by prefix
            if is_completion and prefix_to_complete:
                if word.lower().startswith(prefix_to_complete):
                    results.append((word, prob))
            else:
                results.append((word, prob))
            
            if len(results) >= top_k:
                break
        
        # If no prefix matches found, return top predictions anyway
        if is_completion and not results:
            for idx in top_indices[:top_k * 2]:
                if idx == 0:
                    continue
                word = self.index_to_word.get(idx, None)
                if word:
                    results.append((word, predictions[idx] * 100))
                if len(results) >= top_k:
                    break
        
        top_words = [r[0] for r in results]
        top_probs = [r[1] for r in results]
        
        inference_time = (time.time() - start_time) * 1000
        
        return top_words, top_probs, inference_time, is_completion
    
    def format_predictions(self, predictions: list, confidences: list, input_text: str, is_completion: bool = False) -> str:
        """Format predictions for display."""
        if not predictions:
            return "  No predictions generated"
        
        result = []
        
        if is_completion:
            words = input_text.strip().split()
            context = ' '.join(words[:-1]) if words else ""
            for i, (pred, conf) in enumerate(zip(predictions, confidences), 1):
                if context:
                    full_text = f"{context} {pred}"
                else:
                    full_text = pred
                result.append(f"  {i}. \"{full_text}\" ({conf:.2f}%)")
        else:
            base_text = input_text.strip()
            for i, (pred, conf) in enumerate(zip(predictions, confidences), 1):
                result.append(f"  {i}. \"{base_text} {pred}\" ({conf:.2f}%)")
        
        return "\n".join(result)
    
    def run_interactive(self):
        """Run the interactive testing loop."""
        print("🎹 GRU Keyboard Suggestion Model - Interactive Tester")
        print("=" * 60)
        print("Instructions:")
        print("  - Type text and press Enter to get suggestions")
        print("  - End with SPACE for next-word prediction")
        print("  - No space at end for word completion")
        print("  - Type ':q' or 'quit' to exit")
        print("  - Type ':k30' for 30 suggestions (default is 3)")
        print("=" * 60)
        print()
        
        current_top_k = 3
        
        while True:
            try:
                user_input = input("📝 Input: ")
                
                # Handle commands
                if user_input.strip().lower() in [':q', 'quit', 'exit']:
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.strip().lower() == ':help':
                    print("\nCommands:")
                    print("  :q, quit    - Quit")
                    print("  :help       - Show help")
                    print("  :stats      - Show model stats")
                    print("  :k3         - 3 suggestions")
                    print("  :k30        - 30 suggestions")
                    print()
                    continue
                
                if user_input.strip().lower() == ':stats':
                    print(f"\nModel Statistics:")
                    print(f"  Architecture: GRU")
                    print(f"  Vocab size: {self.vocab_size:,}")
                    print(f"  Sequence length: {self.sequence_length}")
                    print(f"  Word mappings: {len(self.word_to_index):,}")
                    print(f"  Current top-k: {current_top_k}")
                    print()
                    continue
                
                if user_input.strip().lower() == ':k3':
                    current_top_k = 3
                    print(f"✓ Set to {current_top_k} suggestions\n")
                    continue
                
                if user_input.strip().lower() == ':k30':
                    current_top_k = 30
                    print(f"✓ Set to {current_top_k} suggestions\n")
                    continue
                
                if not user_input.strip():
                    print("⚠ Please enter some text\n")
                    continue
                
                # Generate predictions
                predictions, confidences, inference_time, is_completion = self.predict(
                    user_input, top_k=current_top_k
                )
                
                # Display results
                print()
                task_type = "Word Completion" if is_completion else "Next-Word Prediction"
                print(f"🔮 Predictions ({task_type}):")
                print(self.format_predictions(predictions, confidences, user_input, is_completion))
                print()
                print(f"⏱️  Inference: {inference_time:.2f} ms")
                print("=" * 60)
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again.\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive testing for GRU keyboard suggestion model"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models",
        help="Path to the models directory (default: ./models)"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = os.path.join(args.model_dir, "gru_model.keras")
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found: {model_path}")
        print("\nExpected files in models directory:")
        print("  - gru_model.keras")
        print("  - word_index.json")
        print("\n💡 Train a model first using the Colab notebook!")
        sys.exit(1)
    
    # Create tester and run
    try:
        tester = GRUKeyboardTester(model_dir=args.model_dir)
        tester.run_interactive()
    except Exception as e:
        print(f"❌ Failed to initialize tester: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
