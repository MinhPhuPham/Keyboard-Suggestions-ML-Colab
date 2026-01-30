#!/usr/bin/env python3
"""
Interactive Japanese Model Testing Script (zenz GPT-2)

Usage:
    python test_japanese_prediction.py [--model-dir path/to/model]

Features:
- Kana-to-Kanji conversion (かな漢字変換)
- Uses correct zenz input format with special markers
- Configurable top-k suggestions
- Real-time timing information
- Type ':q' to quit

Model: GPT-2 based Japanese model for kana-kanji conversion
Format: \uEE00<input_katakana>\uEE01<output></s>
"""

import torch
import numpy as np
import time
import sys
import os
import argparse
import unicodedata
from transformers import AutoModelForCausalLM, AutoTokenizer


# Zenz special markers
ZENZ_START = '\uEE00'  # Start of input
ZENZ_OUTPUT = '\uEE01'  # Start of output
ZENZ_CONTEXT = '\uEE02'  # Context marker (optional)
ZENZ_EOS = '</s>'  # End of sequence


def hiragana_to_katakana(text: str) -> str:
    """Convert hiragana to katakana (zenz expects katakana input)."""
    result = []
    for char in text:
        code = ord(char)
        # Hiragana range: \u3041-\u3096, Katakana range: \u30A1-\u30F6
        if 0x3041 <= code <= 0x3096:
            result.append(chr(code + 0x60))  # Convert hiragana to katakana
        else:
            result.append(char)
    return ''.join(result)


def is_hiragana_or_katakana(text: str) -> bool:
    """Check if text contains mainly hiragana/katakana."""
    kana_count = 0
    for char in text:
        if '\u3041' <= char <= '\u3096' or '\u30A1' <= char <= '\u30F6':
            kana_count += 1
    return kana_count > 0


class JapaneseKeyboardTester:
    """Interactive tester for Japanese GPT-2 kana-kanji conversion model."""
    
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
        self.model_name = os.path.basename(model_dir)
        
        load_time = (time.time() - start_time) * 1000
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Load time: {load_time:.2f} ms")
        print(f"✓ Vocab size: {self.vocab_size:,}")
        print(f"✓ Model: {self.model_name}")
        print("=" * 60)
        print()
    
    def format_zenz_input(self, kana_input: str, context: str = "") -> str:
        """
        Format input for zenz model.
        
        zenz-v2 format: \uEE00<input_katakana>\uEE01<output>
        zenz-v2 with context: \uEE00<input_katakana>\uEE02<context>\uEE01<output>
        
        Args:
            kana_input: Hiragana or katakana input
            context: Optional left context
        """
        # Convert hiragana to katakana (zenz expects katakana)
        katakana_input = hiragana_to_katakana(kana_input)
        
        if context:
            return f"{ZENZ_START}{katakana_input}{ZENZ_CONTEXT}{context}{ZENZ_OUTPUT}"
        else:
            return f"{ZENZ_START}{katakana_input}{ZENZ_OUTPUT}"
    
    def predict_conversion(self, kana_input: str, context: str = "", top_k: int = 5, max_tokens: int = 20) -> tuple:
        """
        Generate kana-to-kanji conversion predictions.
        
        Args:
            kana_input: Hiragana or katakana input to convert
            context: Optional left context for better conversion
            top_k: Number of conversion candidates
            max_tokens: Maximum tokens to generate per candidate
            
        Returns:
            Tuple of (conversions_list, probabilities, inference_time_ms)
        """
        start_time = time.time()
        
        # Format input for zenz
        formatted_input = self.format_zenz_input(kana_input, context)
        
        # Tokenize
        inputs = self.tokenizer(formatted_input, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        conversions = []
        probs_list = []
        
        # Generate multiple candidates using beam search-like approach
        with torch.no_grad():
            # Get initial predictions after the output marker
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k * 2)
            
            for idx, prob in zip(top_indices.tolist(), top_probs.tolist()):
                if len(conversions) >= top_k:
                    break
                    
                # Start building conversion
                token = self.tokenizer.decode([idx])
                
                # Skip invalid tokens
                if not self.is_valid_token(token):
                    continue
                
                # Skip if it's a special marker or EOS
                if token in [ZENZ_START, ZENZ_OUTPUT, ZENZ_CONTEXT, ZENZ_EOS, '</s>', '<s>']:
                    continue
                
                conversion = token
                combined_prob = prob
                current_ids = torch.cat([input_ids, torch.tensor([[idx]])], dim=1)
                
                # Continue generating until EOS or max tokens
                for _ in range(max_tokens - 1):
                    outputs = self.model(current_ids)
                    next_logits = outputs.logits[0, -1, :]
                    next_probs = torch.softmax(next_logits, dim=-1)
                    next_idx = torch.argmax(next_probs).item()
                    next_prob = next_probs[next_idx].item()
                    
                    next_token = self.tokenizer.decode([next_idx])
                    
                    # Stop at EOS
                    if next_token in ['</s>', '<s>'] or ZENZ_EOS in next_token:
                        break
                    
                    # Stop at special markers
                    if next_token in [ZENZ_START, ZENZ_OUTPUT, ZENZ_CONTEXT]:
                        break
                    
                    # Skip invalid tokens
                    if not self.is_valid_token(next_token):
                        break
                    
                    conversion += next_token
                    combined_prob *= next_prob
                    current_ids = torch.cat([current_ids, torch.tensor([[next_idx]])], dim=1)
                
                if conversion.strip():
                    conversions.append(conversion)
                    probs_list.append(combined_prob * 100)
        
        inference_time = (time.time() - start_time) * 1000
        
        return conversions, probs_list, inference_time
    
    def is_valid_token(self, token: str) -> bool:
        """
        Check if a token is valid and displayable.
        Filters out broken byte-level BPE tokens that appear as ���
        """
        if not token or token.strip() == "":
            return False
        
        # Check for replacement character (appears when bytes can't be decoded)
        if '\ufffd' in token:  # Unicode replacement character
            return False
        
        # Check for other common problematic patterns
        try:
            token.encode('utf-8').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            return False
        
        return True
    
    def format_predictions(self, conversions: list, probabilities: list, input_text: str) -> str:
        """Format predictions for display."""
        if not conversions:
            return "  No conversions generated"
        
        result = []
        for i, (conv, prob) in enumerate(zip(conversions, probabilities), 1):
            result.append(f"  {i}. {conv} ({prob:.2f}%)")
        
        return "\n".join(result)
    
    def run_interactive(self):
        """Run the interactive testing loop."""
        print("🇯🇵 Japanese Kana-Kanji Conversion - Interactive Tester")
        print("=" * 60)
        print("Instructions:")
        print("  - Type hiragana/katakana and press Enter to get conversions")
        print("  - Example: 'ありがとう' → '有難う', '有り難う'")
        print("  - Type ':q' or 'quit' to exit")
        print("  - Type ':k10' for 10 suggestions (default is 5)")
        print("  - Type ':ctx <text>' to set left context")
        print("  - Type ':clear' to clear context")
        print("=" * 60)
        print()
        
        current_top_k = 5
        context = ""
        
        while True:
            try:
                # Show current context if any
                if context:
                    print(f"📝 Context: {context}")
                
                user_input = input("📝 ひらがな入力: ")
                
                # Handle commands
                if user_input.strip().lower() in [':q', 'quit', 'exit']:
                    print("\n👋 さようなら!")
                    break
                
                if user_input.strip().lower() == ':clear':
                    context = ""
                    print("✓ Context cleared\n")
                    continue
                
                if user_input.strip().lower().startswith(':ctx '):
                    context = user_input[5:].strip()
                    print(f"✓ Context set to: {context}\n")
                    continue
                
                if user_input.strip().lower() == ':help':
                    print("\nCommands:")
                    print("  :q, quit    - Quit")
                    print("  :help       - Show help")
                    print("  :stats      - Show model stats")
                    print("  :k5         - 5 suggestions (default)")
                    print("  :k10        - 10 suggestions")
                    print("  :ctx <text> - Set left context")
                    print("  :clear      - Clear context")
                    print("\nExamples:")
                    print("  ありがとう → 有難う, 有り難う")
                    print("  おはよう → お早う, おはよう")
                    print("  にほんご → 日本語")
                    print()
                    continue
                
                if user_input.strip().lower() == ':stats':
                    print(f"\nModel Statistics:")
                    print(f"  Model: {self.model_name}")
                    print(f"  Type: Kana-Kanji Conversion (zenz)")
                    print(f"  Architecture: GPT-2")
                    print(f"  Vocab size: {self.vocab_size:,}")
                    print(f"  Current top-k: {current_top_k}")
                    print(f"  Current context: '{context}' ({len(context)} chars)")
                    print()
                    continue
                
                if user_input.strip().lower() == ':k5':
                    current_top_k = 5
                    print(f"✓ Set to {current_top_k} suggestions\n")
                    continue
                
                if user_input.strip().lower() == ':k10':
                    current_top_k = 10
                    print(f"✓ Set to {current_top_k} suggestions\n")
                    continue
                
                if not user_input.strip():
                    print("⚠ Please enter some text\n")
                    continue
                
                # Check if input contains kana
                if not is_hiragana_or_katakana(user_input):
                    print("⚠ Please enter hiragana or katakana\n")
                    continue
                
                # Generate conversions
                conversions, probabilities, inference_time = self.predict_conversion(
                    user_input, context=context, top_k=current_top_k
                )
                
                # Display results
                katakana_input = hiragana_to_katakana(user_input)
                print()
                print(f"🔮 Kanji conversions for: \"{user_input}\" (→ {katakana_input})")
                print(self.format_predictions(conversions, probabilities, user_input))
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
        description="Interactive testing for Japanese kana-kanji conversion model"
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
