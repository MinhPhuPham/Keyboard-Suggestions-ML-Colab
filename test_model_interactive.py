#!/usr/bin/env python3
"""
Interactive Model Testing Script for MobileBERT Keyboard Suggestions

Usage:
    python test_model_interactive.py [--model-dir path/to/model]

Features:
- Word completion: "Hel" → ["Hello", "Help", "Helping"]
- Next-word prediction: "How are" → ["you", "they", "we"]
- Typo correction: "Thers" → ["There", "Theirs", "Therapy"]
- Configurable top-k (3 or 30 suggestions)
- Real-time timing information
- Type ':q' to quit
"""

import torch
import time
import sys
import os
from transformers import MobileBertForMaskedLM, MobileBertTokenizer
import argparse


class MobileBERTKeyboardTester:
    """Interactive tester for MobileBERT keyboard suggestion model."""
    
    def __init__(self, model_dir: str = "./models/best_model"):
        """
        Initialize the tester with a trained MobileBERT model.
        
        Args:
            model_dir: Directory containing the model and tokenizer
        """
        print(f"Loading MobileBERT model from: {model_dir}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Load tokenizer and model
        self.tokenizer = MobileBertTokenizer.from_pretrained(model_dir)
        self.model = MobileBertForMaskedLM.from_pretrained(model_dir)
        self.model.eval()
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        load_time = (time.time() - start_time) * 1000
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Load time: {load_time:.2f} ms")
        print(f"✓ Device: {self.device}")
        print(f"✓ Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 60)
        print()
    
    def predict(self, input_text: str, top_k: int = 3) -> tuple:
        """
        Generate predictions using MLM.
        
        Args:
            input_text: The input text (will add [MASK] automatically)
            top_k: Number of suggestions to return (3 or 30)
            
        Returns:
            Tuple of (predictions_list, confidences, inference_time_ms, is_completion)
        """
        start_time = time.time()
        
        # Detect if this is word completion (last word is incomplete)
        words = input_text.strip().split()
        is_completion = False
        prefix_to_complete = ""
        
        if words and not input_text.endswith(' '):
            # Last word is incomplete - this is word completion
            is_completion = True
            prefix_to_complete = words[-1].lower()
            # Keep the incomplete word in context, add [MASK] after it
            # This helps the model understand what we're trying to complete
            text_with_mask = f"{input_text} {self.tokenizer.mask_token}"
        else:
            # Next-word prediction
            text_with_mask = f"{input_text} {self.tokenizer.mask_token}"
        
        # Tokenize
        inputs = self.tokenizer(
            text_with_mask,
            return_tensors='pt',
            padding='max_length',
            max_length=32,
            truncation=True
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.logits
        
        # Get predictions at [MASK] position
        mask_positions = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        
        if len(mask_positions[1]) > 0:
            mask_pos = mask_positions[1][0]
            mask_predictions = predictions[0, mask_pos]
            
            # Get top-k predictions
            top_k_results = torch.topk(mask_predictions, k=min(top_k, mask_predictions.size(0)))
            all_words = [self.tokenizer.decode([idx]).strip() for idx in top_k_results.indices]
            all_probs = torch.softmax(mask_predictions, dim=0)
            all_confidences = [all_probs[idx].item() * 100 for idx in top_k_results.indices]
            
            # Filter for word completion
            if is_completion and prefix_to_complete:
                filtered_words = []
                filtered_confidences = []
                for word, conf in zip(all_words, all_confidences):
                    # Check if word starts with the prefix
                    if word.lower().startswith(prefix_to_complete):
                        filtered_words.append(word)
                        filtered_confidences.append(conf)
                
                # If we have matches, use them; otherwise fall back to all predictions
                if filtered_words:
                    top_words = filtered_words[:top_k]
                    top_probs = filtered_confidences[:top_k]
                else:
                    # No matches - show all predictions anyway
                    top_words = all_words
                    top_probs = all_confidences
            else:
                top_words = all_words
                top_probs = all_confidences
        else:
            top_words = []
            top_probs = []
            is_completion = False
        
        inference_time = (time.time() - start_time) * 1000
        
        return top_words, top_probs, inference_time, is_completion
    
    def format_predictions(self, predictions: list, confidences: list, input_text: str, is_completion: bool = False) -> str:
        """
        Format predictions for display.
        
        Args:
            predictions: List of prediction strings
            confidences: List of confidence scores
            input_text: Original input text
            is_completion: Whether this is word completion (replace last word) or next-word (append)
            
        Returns:
            Formatted string
        """
        if not predictions:
            return "  No predictions generated"
        
        result = []
        
        if is_completion:
            # Word completion: replace the incomplete last word
            words = input_text.strip().split()
            if words:
                context = ' '.join(words[:-1])
                for i, (pred, conf) in enumerate(zip(predictions, confidences), 1):
                    if context:
                        full_text = f"{context} {pred}"
                    else:
                        full_text = pred
                    result.append(f"  {i}. \"{full_text}\" (confidence: {conf:.1f}%)")
            else:
                for i, (pred, conf) in enumerate(zip(predictions, confidences), 1):
                    result.append(f"  {i}. \"{pred}\" (confidence: {conf:.1f}%)")
        else:
            # Next-word prediction: append to input
            for i, (pred, conf) in enumerate(zip(predictions, confidences), 1):
                result.append(f"  {i}. \"{input_text} {pred}\" (confidence: {conf:.1f}%)")
        
        return "\n".join(result)
    
    def detect_gibberish(self, confidences: list, threshold: float = 10.0) -> bool:
        """
        Heuristic gibberish detection based on confidence.
        
        Args:
            confidences: List of confidence scores
            threshold: Minimum confidence threshold
            
        Returns:
            True if likely gibberish, False otherwise
        """
        if not confidences:
            return True
        
        # If top prediction has very low confidence, likely gibberish
        return confidences[0] < threshold
    
    def run_interactive(self):
        """Run the interactive testing loop."""
        print("🎹 MobileBERT Keyboard Suggestion Model - Interactive Tester")
        print("=" * 60)
        print("Instructions:")
        print("  - Type your text and press Enter to get suggestions")
        print("  - Type ':q' or 'quit' to exit")
        print("  - Type ':help' for more options")
        print("  - Type ':k30' to get 30 suggestions (default is 3)")
        print("=" * 60)
        print()
        
        current_top_k = 3
        
        while True:
            try:
                # Get input
                user_input = input("📝 Input: ").strip()
                
                # Handle commands
                if user_input.lower() in [':q', 'quit', 'exit']:
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == ':help':
                    print("\nCommands:")
                    print("  :q, quit, exit - Quit the program")
                    print("  :help          - Show this help message")
                    print("  :stats         - Show model statistics")
                    print("  :k3            - Get 3 suggestions (quick)")
                    print("  :k30           - Get 30 suggestions (extended)")
                    print()
                    continue
                
                if user_input.lower() == ':stats':
                    print("\nModel Statistics:")
                    print(f"  Model: MobileBERT for Masked Language Modeling")
                    print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
                    print(f"  Vocab size: {self.tokenizer.vocab_size:,}")
                    print(f"  Device: {self.device}")
                    print(f"  Current top-k: {current_top_k}")
                    print()
                    continue
                
                if user_input.lower() == ':k3':
                    current_top_k = 3
                    print(f"✓ Set to {current_top_k} suggestions\n")
                    continue
                
                if user_input.lower() == ':k30':
                    current_top_k = 30
                    print(f"✓ Set to {current_top_k} suggestions\n")
                    continue
                
                if not user_input:
                    print("⚠ Please enter some text\n")
                    continue
                
                # Generate predictions
                predictions, confidences, inference_time, is_completion = self.predict(user_input, top_k=current_top_k)
                
                # Check for gibberish
                is_gibberish = self.detect_gibberish(confidences)
                
                # Display results
                print()
                if is_gibberish:
                    print("⚠️  Gibberish detected! (very low confidence)")
                    print("🔮 Predictions (may not be accurate):")
                else:
                    task_type = "Word Completion" if is_completion else "Next-Word Prediction"
                    print(f"🔮 Predictions ({task_type}):")
                
                print(self.format_predictions(predictions, confidences, user_input, is_completion))
                print()
                print(f"⏱️  Inference time: {inference_time:.2f} ms")
                print(f"📊 Top-k: {current_top_k}")
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
        description="Interactive testing for MobileBERT keyboard suggestion model"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models/best_model",
        help="Path to the model directory (default: ./models/best_model)"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_dir):
        print(f"❌ Error: Model directory not found: {args.model_dir}")
        print("\nAvailable models:")
        models_dir = "./models"
        if os.path.exists(models_dir):
            for item in os.listdir(models_dir):
                item_path = os.path.join(models_dir, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    print(f"  - {item_path}")
        print("\n💡 Train a model first using the Colab notebook!")
        sys.exit(1)
    
    # Create tester and run
    try:
        tester = MobileBERTKeyboardTester(model_dir=args.model_dir)
        tester.run_interactive()
    except Exception as e:
        print(f"❌ Failed to initialize tester: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
