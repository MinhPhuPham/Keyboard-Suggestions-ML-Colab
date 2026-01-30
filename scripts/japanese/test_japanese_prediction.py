#!/usr/bin/env python3
"""
Interactive Japanese Model Testing Script (zenz-v2.5-small)

Usage:
    python test_japanese_prediction.py [--model-dir path/to/model]

Features:
- Character-by-character prediction (no word splitting for Japanese)
- Next token prediction from context
- Configurable top-k (3 or 30 suggestions)
- Real-time timing information
- Type ':q' to quit

Model: GPT-2 based Japanese model for kana-kanji conversion
"""

import torch
import numpy as np
import time
import sys
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


class JapaneseKeyboardTester:
    """Interactive tester for Japanese GPT-2 keyboard prediction model."""
    
    def __init__(self, model_dir: str = "./models/japanese/zenz-v2.5-small"):
        """
        Initialize the tester with a Japanese model.
        
        Args:
            model_dir: Directory containing the model and tokenizer files
        """
        print(f"Loading Japanese model from: {model_dir}")
        print("=" * 60)
        
        start_time = time.time()
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.model.eval()
        
        # Model info
        self.vocab_size = self.tokenizer.vocab_size
        self.model_name = "zenz-v2.5-small"
        
        load_time = (time.time() - start_time) * 1000
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Load time: {load_time:.2f} ms")
        print(f"✓ Vocab size: {self.vocab_size:,}")
        print(f"✓ Model: {self.model_name}")
        print("=" * 60)
        print()
    
    def predict(self, input_text: str, top_k: int = 3) -> tuple:
        """
        Generate predictions for next token.
        
        Args:
            input_text: The input Japanese text
            top_k: Number of suggestions to return
            
        Returns:
            Tuple of (predictions_list, probabilities, inference_time_ms)
        """
        start_time = time.time()
        
        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_ids)
            # Get logits for the last position
            logits = outputs.logits[0, -1, :]
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probs, top_k)
        
        # Decode tokens
        predictions = []
        probabilities = []
        
        for idx, prob in zip(top_indices.tolist(), top_probs.tolist()):
            token = self.tokenizer.decode([idx])
            predictions.append(token)
            probabilities.append(prob * 100)  # Convert to percentage
        
        inference_time = (time.time() - start_time) * 1000
        
        return predictions, probabilities, inference_time
    
    def format_predictions(self, predictions: list, probabilities: list, input_text: str) -> str:
        """Format predictions for display."""
        if not predictions:
            return "  No predictions generated"
        
        result = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities), 1):
            # Show what the full text would be with this prediction
            full_text = f"{input_text}{pred}"
            # Handle display of special characters
            display_pred = repr(pred) if pred.strip() == "" else pred
            result.append(f"  {i}. \"{full_text}\" [+{display_pred}] ({prob:.2f}%)")
        
        return "\n".join(result)
    
    def run_interactive(self):
        """Run the interactive testing loop."""
        print("🇯🇵 Japanese Keyboard Prediction - Interactive Tester")
        print("=" * 60)
        print("Instructions:")
        print("  - Type Japanese text and press Enter to get predictions")
        print("  - Each prediction shows the next character/token")
        print("  - Type ':q' or 'quit' to exit")
        print("  - Type ':k30' for 30 suggestions (default is 3)")
        print("  - Type ':clear' to clear context")
        print("=" * 60)
        print()
        
        current_top_k = 3
        context = ""
        
        while True:
            try:
                # Show current context if any
                if context:
                    print(f"📝 Context: {context}")
                
                user_input = input("📝 Input: ")
                
                # Handle commands
                if user_input.strip().lower() in [':q', 'quit', 'exit']:
                    print("\n👋 さようなら!")
                    break
                
                if user_input.strip().lower() == ':clear':
                    context = ""
                    print("✓ Context cleared\n")
                    continue
                
                if user_input.strip().lower() == ':help':
                    print("\nCommands:")
                    print("  :q, quit    - Quit")
                    print("  :help       - Show help")
                    print("  :stats      - Show model stats")
                    print("  :k3         - 3 suggestions")
                    print("  :k30        - 30 suggestions")
                    print("  :clear      - Clear context")
                    print()
                    continue
                
                if user_input.strip().lower() == ':stats':
                    print(f"\nModel Statistics:")
                    print(f"  Model: {self.model_name}")
                    print(f"  Architecture: GPT-2")
                    print(f"  Vocab size: {self.vocab_size:,}")
                    print(f"  Current top-k: {current_top_k}")
                    print(f"  Current context: {len(context)} chars")
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
                
                # Add to context (Japanese: no space between characters)
                context += user_input
                
                # Generate predictions
                predictions, probabilities, inference_time = self.predict(
                    context, top_k=current_top_k
                )
                
                # Display results
                print()
                print(f"🔮 Next token predictions for: \"{context}\"")
                print(self.format_predictions(predictions, probabilities, context))
                print()
                print(f"⏱️  Inference: {inference_time:.2f} ms")
                print("=" * 60)
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 さようなら!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
                print("Please try again.\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive testing for Japanese keyboard prediction model"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models/japanese/zenz-v2.5-small",
        help="Path to the model directory (default: ./models/japanese/zenz-v2.5-small)"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_dir):
        print(f"❌ Error: Model not found: {args.model_dir}")
        print("\nExpected files in model directory:")
        print("  - config.json")
        print("  - model.safetensors or pytorch_model.bin")
        print("  - tokenizer.json / vocab.json")
        print("\n💡 Run download_convert_zenz_coreml.py first to download the model!")
        sys.exit(1)
    
    # Create tester and run
    try:
        tester = JapaneseKeyboardTester(model_dir=args.model_dir)
        tester.run_interactive()
    except Exception as e:
        print(f"❌ Failed to initialize tester: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
