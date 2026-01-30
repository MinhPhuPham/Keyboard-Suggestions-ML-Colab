#!/usr/bin/env python3
"""
Interactive Japanese Model Testing Script (zenz GPT-2)

Supports TWO modes:
1. Kana-Kanji Conversion (かな漢字変換)
   Format: \uEE00<katakana>\uEE01<output>
   
2. Next Character Prediction (次の文字予測)
   Format: \uEE00。\uEE02<leftSideContext>

Usage:
    python test_japanese_prediction.py [--model-dir path/to/model]
"""

import torch
import numpy as np
import time
import sys
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


# Zenz special markers
ZENZ_START = '\uEE00'      # Start of input
ZENZ_OUTPUT = '\uEE01'     # Start of output
ZENZ_CONTEXT = '\uEE02'    # Context marker
ZENZ_PROFILE = '\uEE03'    # Profile (v3)
ZENZ_TOPIC = '\uEE04'      # Topic (v3)
ZENZ_STYLE = '\uEE05'      # Style (v3)
ZENZ_EOS = '</s>'


def hiragana_to_katakana(text: str) -> str:
    """Convert hiragana to katakana."""
    result = []
    for char in text:
        code = ord(char)
        if 0x3041 <= code <= 0x3096:
            result.append(chr(code + 0x60))
        else:
            result.append(char)
    return ''.join(result)


def is_hiragana_or_katakana(text: str) -> bool:
    """Check if text contains hiragana/katakana."""
    for char in text:
        if '\u3041' <= char <= '\u3096' or '\u30A1' <= char <= '\u30F6':
            return True
    return False


class JapaneseKeyboardTester:
    """Interactive tester for Japanese GPT-2 model."""
    
    def __init__(self, model_dir: str):
        print(f"Loading Japanese model from: {model_dir}")
        print("=" * 60)
        
        start_time = time.time()
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.model.eval()
        
        self.vocab_size = self.tokenizer.vocab_size
        self.model_name = os.path.basename(model_dir)
        
        load_time = (time.time() - start_time) * 1000
        
        print(f"✓ Model loaded: {self.model_name}")
        print(f"✓ Load time: {load_time:.2f} ms")
        print(f"✓ Vocab size: {self.vocab_size:,}")
        print("=" * 60)
        print()
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for zenz tokenizer."""
        # Replace space with ideographic space, remove newlines
        return text.replace(" ", "\u3000").replace("\n", "")
    
    def is_valid_token(self, token: str) -> bool:
        """Filter out invalid tokens."""
        if not token or token.strip() == "":
            return False
        if '\ufffd' in token:
            return False
        try:
            token.encode('utf-8').decode('utf-8')
        except:
            return False
        return True
    
    # ========================================
    # Mode 1: Kana-Kanji Conversion
    # ========================================
    def convert_kana_to_kanji(self, kana_input: str, context: str = "", top_k: int = 5, max_tokens: int = 20) -> tuple:
        """
        Convert kana to kanji.
        Format: \uEE00<katakana>\uEE01<output> or with context: \uEE00<katakana>\uEE02<context>\uEE01<output>
        """
        start_time = time.time()
        
        katakana = hiragana_to_katakana(kana_input)
        
        if context:
            prompt = f"{ZENZ_START}{katakana}{ZENZ_CONTEXT}{context}{ZENZ_OUTPUT}"
        else:
            prompt = f"{ZENZ_START}{katakana}{ZENZ_OUTPUT}"
        
        prompt = self.preprocess_text(prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        conversions = []
        probs_list = []
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k * 3)
            
            for idx, prob in zip(top_indices.tolist(), top_probs.tolist()):
                if len(conversions) >= top_k:
                    break
                
                token = self.tokenizer.decode([idx])
                if not self.is_valid_token(token):
                    continue
                if token in [ZENZ_START, ZENZ_OUTPUT, ZENZ_CONTEXT, ZENZ_EOS, '</s>', '<s>']:
                    continue
                
                # Generate full conversion
                conversion = token
                combined_prob = prob
                current_ids = torch.cat([input_ids, torch.tensor([[idx]])], dim=1)
                
                for _ in range(max_tokens - 1):
                    outputs = self.model(current_ids)
                    next_logits = outputs.logits[0, -1, :]
                    next_probs = torch.softmax(next_logits, dim=-1)
                    next_idx = torch.argmax(next_probs).item()
                    next_prob = next_probs[next_idx].item()
                    next_token = self.tokenizer.decode([next_idx])
                    
                    if next_token in ['</s>', '<s>'] or ZENZ_EOS in next_token:
                        break
                    if next_token in [ZENZ_START, ZENZ_OUTPUT, ZENZ_CONTEXT]:
                        break
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
    
    # ========================================
    # Mode 2: Next Character Prediction
    # ========================================
    def predict_next_character(self, left_context: str, top_k: int = 10) -> tuple:
        """
        Predict next character given left context.
        Format: \uEE00。\uEE02<leftSideContext>
        
        This is the format used for next-character prediction in zenz.
        """
        start_time = time.time()
        
        # Format for next character prediction (from ZENZ_USAGE_EX.md line 285)
        prompt = f"{ZENZ_START}。{ZENZ_CONTEXT}{left_context}"
        prompt = self.preprocess_text(prompt)
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            
            # Apply repeat penalty (similar to original implementation)
            token_counts = {}
            for token_id in input_ids[0].tolist():
                token_counts[token_id] = token_counts.get(token_id, 0) + 1
            
            # Collect predictions
            predictions = []
            seen_chars = set()
            
            top_probs, top_indices = torch.topk(probs, min(top_k * 5, self.vocab_size))
            
            for idx, prob in zip(top_indices.tolist(), top_probs.tolist()):
                token = self.tokenizer.decode([idx])
                
                if not self.is_valid_token(token):
                    continue
                if token in [ZENZ_START, ZENZ_OUTPUT, ZENZ_CONTEXT, ZENZ_EOS, '</s>', '<s>']:
                    continue
                
                # Get first character (like original implementation)
                first_char = token[0] if token else None
                if first_char and first_char not in seen_chars:
                    seen_chars.add(first_char)
                    predictions.append((first_char, prob * 100))
                    
                    if len(predictions) >= top_k:
                        break
        
        inference_time = (time.time() - start_time) * 1000
        return predictions, inference_time
    
    # ========================================
    # Mode 3: Greedy Text Generation
    # ========================================
    def generate_text(self, context: str, max_tokens: int = 20) -> tuple:
        """
        Generate text greedily from context.
        """
        start_time = time.time()
        
        prompt = self.preprocess_text(context)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        initial_len = input_ids.shape[1]
        
        generated = ""
        
        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]
                next_idx = torch.argmax(logits).item()
                next_token = self.tokenizer.decode([next_idx])
                
                if next_token in ['</s>', '<s>'] or not self.is_valid_token(next_token):
                    break
                
                generated += next_token
                input_ids = torch.cat([input_ids, torch.tensor([[next_idx]])], dim=1)
        
        inference_time = (time.time() - start_time) * 1000
        return generated, inference_time
    
    def run_interactive(self):
        """Run the interactive testing loop."""
        print("🇯🇵 Japanese Model Tester - Dual Mode")
        print("=" * 60)
        print("Modes:")
        print("  :convert   - Kana-Kanji conversion (default)")
        print("  :predict   - Next character prediction")
        print("  :generate  - Free text generation")
        print("")
        print("Commands:")
        print("  :q         - Quit")
        print("  :ctx <txt> - Set left context")
        print("  :clear     - Clear context")
        print("  :k<N>      - Set top-k (e.g. :k10)")
        print("=" * 60)
        print()
        
        current_mode = "convert"
        current_top_k = 5
        context = ""
        
        while True:
            try:
                if context:
                    print(f"📝 Context: {context}")
                
                mode_display = {"convert": "変換", "predict": "予測", "generate": "生成"}[current_mode]
                user_input = input(f"📝 [{mode_display}]: ")
                
                # Commands
                if user_input.strip().lower() in [':q', 'quit', 'exit']:
                    print("\n👋 さようなら!")
                    break
                
                if user_input.strip().lower() == ':convert':
                    current_mode = "convert"
                    print("✓ Mode: Kana-Kanji Conversion\n")
                    continue
                
                if user_input.strip().lower() == ':predict':
                    current_mode = "predict"
                    print("✓ Mode: Next Character Prediction\n")
                    continue
                
                if user_input.strip().lower() == ':generate':
                    current_mode = "generate"
                    print("✓ Mode: Text Generation\n")
                    continue
                
                if user_input.strip().lower() == ':clear':
                    context = ""
                    print("✓ Context cleared\n")
                    continue
                
                if user_input.strip().lower().startswith(':ctx '):
                    context = user_input[5:].strip()
                    print(f"✓ Context: {context}\n")
                    continue
                
                if user_input.strip().lower().startswith(':k'):
                    try:
                        current_top_k = int(user_input[2:])
                        print(f"✓ Top-k: {current_top_k}\n")
                    except:
                        print("⚠ Invalid number\n")
                    continue
                
                if user_input.strip().lower() == ':help':
                    print("\nModes:")
                    print("  :convert  - ひらがな → 漢字 (e.g. ありがとう → 有難う)")
                    print("  :predict  - 次の文字予測 (e.g. 私の名前は → '田', '山', ...)")
                    print("  :generate - テキスト生成")
                    print("\nExamples:")
                    print("  [変換] ありがとう → 有難う, ありがとう")
                    print("  [予測] 私の名前は → 田(30%), 山(15%), ...")
                    print()
                    continue
                
                if not user_input.strip():
                    continue
                
                # Execute based on mode
                print()
                
                if current_mode == "convert":
                    if not is_hiragana_or_katakana(user_input):
                        print("⚠ Please enter hiragana/katakana for conversion\n")
                        continue
                    
                    conversions, probs, time_ms = self.convert_kana_to_kanji(
                        user_input, context=context, top_k=current_top_k
                    )
                    katakana = hiragana_to_katakana(user_input)
                    print(f"🔮 Kanji conversions for: \"{user_input}\" (→ {katakana})")
                    for i, (conv, prob) in enumerate(zip(conversions, probs), 1):
                        print(f"  {i}. {conv} ({prob:.2f}%)")
                    print(f"\n⏱️  {time_ms:.2f} ms")
                
                elif current_mode == "predict":
                    # Use user input as context for prediction
                    full_context = (context + user_input) if context else user_input
                    predictions, time_ms = self.predict_next_character(
                        full_context, top_k=current_top_k
                    )
                    print(f"🔮 Next character predictions for: \"{full_context}\"")
                    for i, (char, prob) in enumerate(predictions, 1):
                        full_text = full_context + char
                        print(f"  {i}. {full_text} [+{char}] ({prob:.2f}%)")
                    print(f"\n⏱️  {time_ms:.2f} ms")
                
                elif current_mode == "generate":
                    full_context = (context + user_input) if context else user_input
                    generated, time_ms = self.generate_text(full_context, max_tokens=30)
                    print(f"🔮 Generated text from: \"{full_context}\"")
                    print(f"   → {full_context}{generated}")
                    print(f"\n⏱️  {time_ms:.2f} ms")
                
                print("=" * 60)
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 さようなら!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
                print()


def main():
    parser = argparse.ArgumentParser(description="Japanese model tester (kana-kanji + prediction)")
    parser.add_argument("--model-dir", type=str, default="./models/japanese/zenz-v2.5-small")
    args = parser.parse_args()
    
    if not os.path.exists(args.model_dir):
        print(f"❌ Model not found: {args.model_dir}")
        print("💡 Run ./run_download_and_test.sh first!")
        sys.exit(1)
    
    try:
        tester = JapaneseKeyboardTester(model_dir=args.model_dir)
        tester.run_interactive()
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
