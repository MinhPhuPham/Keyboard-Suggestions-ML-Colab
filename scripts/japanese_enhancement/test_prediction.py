#!/usr/bin/env python3
"""
Interactive BiGRU Model Testing Script (Shared Encoder v2)

Supports TWO modes:
1. Kana-Kanji Conversion (かな漢字変換)
   Input: hiragana → Output: kanji candidates

2. Next Word Prediction (次の単語予測)
   Input: context text → Output: predicted next words

Usage:
    python -m scripts.japanese_enhancement.test_prediction [--model-dir path]
"""

import time
import sys
import os
import json
import argparse
import numpy as np


def _lazy_import_tf():
    """Lazy import TensorFlow to speed up --help."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    return tf


class BiGRUKeyboardTester:
    """Interactive tester for BiGRU shared encoder model."""

    def __init__(self, model_dir: str):
        print(f"Loading BiGRU model from: {model_dir}")
        print("=" * 60)

        start_time = time.time()

        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        tf = _lazy_import_tf()

        # Load model
        model_path = os.path.join(model_dir, 'model.keras')
        if not os.path.exists(model_path):
            # Try best model
            model_path = os.path.join(model_dir, 'best_shared_multitask.keras')
        self.model = tf.keras.models.load_model(model_path, compile=False)

        # Load char vocab
        with open(os.path.join(model_dir, 'char_vocab.json'), 'r', encoding='utf-8') as f:
            self.char_to_idx = json.load(f)
        self.idx_to_char = {int(v): k for k, v in self.char_to_idx.items()}

        # Load word vocab
        with open(os.path.join(model_dir, 'word_vocab.json'), 'r', encoding='utf-8') as f:
            self.word_to_idx = json.load(f)
        self.idx_to_word = {int(v): k for k, v in self.word_to_idx.items()}

        # Load config
        config_path = os.path.join(model_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.model_config = json.load(f)
        else:
            self.model_config = {}

        self.max_encoder_len = self.model_config.get('max_encoder_len', 61)
        self.max_decoder_len = self.model_config.get('max_decoder_len', 32)

        # Special token IDs
        self.PAD = self.char_to_idx.get('<PAD>', 0)
        self.UNK = self.char_to_idx.get('<UNK>', 1)
        self.BOS = self.char_to_idx.get('<BOS>', 2)
        self.EOS = self.char_to_idx.get('<EOS>', 3)
        self.SEP = self.char_to_idx.get('<SEP>', 4)

        load_time = (time.time() - start_time) * 1000

        print(f"✓ Model loaded: {self.model.name}")
        print(f"✓ Load time: {load_time:.0f} ms")
        print(f"✓ Params: {self.model.count_params():,}")
        print(f"✓ Char vocab: {len(self.char_to_idx):,}")
        print(f"✓ Word vocab: {len(self.word_to_idx):,}")
        print("=" * 60)
        print()

    # ========================================
    # Encoding Helpers
    # ========================================

    def _encode_encoder_input(self, text: str) -> np.ndarray:
        """Encode text to encoder input array with <SEP> handling."""
        tokens = []
        i = 0
        while i < len(text):
            if text[i:i+5] == '<SEP>':
                tokens.append(self.SEP)
                i += 5
            else:
                tokens.append(self.char_to_idx.get(text[i], self.UNK))
                i += 1

        arr = np.zeros(self.max_encoder_len, dtype=np.int32)
        n = min(len(tokens), self.max_encoder_len)
        arr[:n] = tokens[:n]
        return arr

    def _encode_nwp_context(self, context_words: list) -> np.ndarray:
        """Encode context words to char IDs with <SEP> markers."""
        char_ids = []
        for i, word in enumerate(context_words):
            if i > 0:
                char_ids.append(self.SEP)
            for ch in word:
                char_ids.append(self.char_to_idx.get(ch, self.UNK))
                if len(char_ids) >= self.max_encoder_len:
                    break
            if len(char_ids) >= self.max_encoder_len:
                break

        arr = np.zeros(self.max_encoder_len, dtype=np.int32)
        n = min(len(char_ids), self.max_encoder_len)
        arr[:n] = char_ids[:n]
        return arr

    def _tokenize_words(self, text: str) -> list:
        """Word tokenization using fugashi (fallback: char-level split)."""
        if not text:
            return []
        try:
            import fugashi
            tagger = fugashi.Tagger()
            return [t.surface for t in tagger(text) if t.feature.pos1 != '空白']
        except (ImportError, RuntimeError):
            # Fallback: simple character-level split (no MeCab)
            # Split on common particles/punctuation as rough word boundaries
            import re
            parts = re.split(r'(は|が|を|に|で|と|の|も|へ|から|まで|より|、|。)', text)
            return [p for p in parts if p]

    # ========================================
    # Mode 1: Kana-Kanji Conversion
    # ========================================

    def convert_kana_to_kanji(self, kana_input: str, context: str = "",
                              top_k: int = 5) -> tuple:
        """Convert kana to kanji using greedy decoding.

        Uses encoder-decoder with attention to convert hiragana to kanji.
        Returns top-k candidates by beam search (simplified).
        """
        tf = _lazy_import_tf()
        start_time = time.time()

        # Encode: context<SEP>kana
        if context:
            enc_text = f"{context}<SEP>{kana_input}"
        else:
            enc_text = f"<SEP>{kana_input}"
        enc_input = self._encode_encoder_input(enc_text)
        enc_batch = np.expand_dims(enc_input, 0)  # (1, encoder_len)

        # Greedy decode: start with <BOS>, predict one char at a time
        dec_input = np.zeros((1, self.max_decoder_len), dtype=np.int32)
        dec_input[0, 0] = self.BOS

        result_chars = []

        for step in range(1, self.max_decoder_len):
            kkc_pred, _ = self.model.predict(
                [enc_batch, dec_input], verbose=0
            )
            # kkc_pred: (1, decoder_len, char_vocab_size)
            next_probs = kkc_pred[0, step - 1, :]
            next_char_id = int(np.argmax(next_probs))

            if next_char_id == self.EOS or next_char_id == self.PAD:
                break

            char = self.idx_to_char.get(next_char_id, '?')
            result_chars.append((char, float(next_probs[next_char_id])))

            # Feed predicted char as next decoder input
            if step < self.max_decoder_len:
                dec_input[0, step] = next_char_id

        # Build result
        conversion = ''.join(c for c, _ in result_chars)
        avg_prob = np.mean([p for _, p in result_chars]) * 100 if result_chars else 0

        # Get top-k alternatives from first position
        dec_first = np.zeros((1, self.max_decoder_len), dtype=np.int32)
        dec_first[0, 0] = self.BOS
        kkc_pred, _ = self.model.predict([enc_batch, dec_first], verbose=0)
        first_probs = kkc_pred[0, 0, :]
        top_indices = np.argsort(first_probs)[-top_k * 2:][::-1]

        alternatives = []
        for idx in top_indices:
            ch = self.idx_to_char.get(int(idx), None)
            if ch and ch not in ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<SEP>']:
                alternatives.append((ch, float(first_probs[idx]) * 100))
                if len(alternatives) >= top_k:
                    break

        inference_time = (time.time() - start_time) * 1000
        return conversion, avg_prob, alternatives, inference_time

    # ========================================
    # Mode 2: Next Word Prediction
    # ========================================

    def predict_next_word(self, context_text: str,
                          top_k: int = 5) -> tuple:
        """Predict next word given context.

        Tokenizes context into words, encodes as char sequence with
        <SEP> word boundaries, then runs through shared encoder → NWP head.
        """
        tf = _lazy_import_tf()
        start_time = time.time()

        # Tokenize context into words
        words = self._tokenize_words(context_text)
        if not words:
            return [], [], 0

        # Encode context words as char IDs with <SEP> markers
        enc_input = self._encode_nwp_context(words)
        enc_batch = np.expand_dims(enc_input, 0)

        # Dummy decoder input (NWP doesn't use it, but model needs 2 inputs)
        dec_dummy = np.zeros((1, self.max_decoder_len), dtype=np.int32)

        # Forward pass
        _, nwp_pred = self.model.predict([enc_batch, dec_dummy], verbose=0)
        # nwp_pred: (1, word_vocab_size)
        probs = nwp_pred[0]

        # Get top-k predictions
        top_indices = np.argsort(probs)[-top_k * 2:][::-1]

        predictions = []
        pred_probs = []
        for idx in top_indices:
            word = self.idx_to_word.get(int(idx), None)
            if word and word not in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
                predictions.append(word)
                pred_probs.append(float(probs[idx]) * 100)
                if len(predictions) >= top_k:
                    break

        inference_time = (time.time() - start_time) * 1000
        return predictions, pred_probs, inference_time

    # ========================================
    # Batch Test
    # ========================================

    def run_batch_test(self):
        """Run predefined test cases for both KKC and NWP."""
        print("\n🧪 Running batch tests...")
        print("=" * 60)

        # --- KKC Tests ---
        print("\n📝 KKC: Kana → Kanji Conversion")
        print("-" * 40)
        kkc_tests = [
            ("きょう", ""),
            ("ありがとう", ""),
            ("おはよう", ""),
            ("にほん", ""),
            ("とうきょう", ""),
            ("がっこう", ""),
            ("せんせい", ""),
            ("しごと", ""),
            ("でんしゃ", ""),
            ("てんき", "今日は"),
        ]

        for kana, ctx in kkc_tests:
            conversion, prob, alts, ms = self.convert_kana_to_kanji(
                kana, context=ctx, top_k=3
            )
            ctx_str = f" [ctx: {ctx}]" if ctx else ""
            print(f"  {kana}{ctx_str} → {conversion} ({prob:.1f}%)"
                  f"  [{', '.join(a for a, _ in alts[:3])}]"
                  f"  {ms:.0f}ms")

        # --- NWP Tests ---
        print(f"\n📝 NWP: Next Word Prediction")
        print("-" * 40)
        nwp_tests = [
            "ありがとう",
            "お願い",
            "今日は天気が",
            "東京に",
            "日本の",
            "よろしく",
            "すみません",
            "おはよう",
            "明日は",
            "私は",
        ]

        for ctx in nwp_tests:
            preds, probs, ms = self.predict_next_word(ctx, top_k=5)
            pred_str = ", ".join(
                f"{w}({p:.1f}%)" for w, p in zip(preds[:5], probs[:5])
            )
            print(f"  {ctx} → {pred_str}  {ms:.0f}ms")

        print(f"\n{'=' * 60}")
        print("✅ Batch test complete")

    # ========================================
    # Interactive Mode
    # ========================================

    def run_interactive(self):
        """Run the interactive testing loop."""
        print("🇯🇵 BiGRU Model Tester — Shared Encoder v2")
        print("=" * 60)
        print("Modes:")
        print("  :convert   - Kana→Kanji conversion (default)")
        print("  :predict   - Next word prediction")
        print("  :batch     - Run all test cases")
        print()
        print("Commands:")
        print("  :q         - Quit")
        print("  :ctx <txt> - Set left context")
        print("  :clear     - Clear context")
        print("  :k<N>      - Set top-k (e.g. :k10)")
        print("  :help      - Show help")
        print("=" * 60)
        print()

        current_mode = "convert"
        current_top_k = 5
        context = ""

        while True:
            try:
                if context:
                    print(f"📝 Context: {context}")

                mode_display = {
                    "convert": "変換",
                    "predict": "予測",
                }[current_mode]
                user_input = input(f"📝 [{mode_display}]: ")

                # Commands
                if user_input.strip().lower() in [':q', 'quit', 'exit']:
                    print("\n👋 さようなら!")
                    break

                if user_input.strip().lower() == ':convert':
                    current_mode = "convert"
                    print("✓ Mode: Kana→Kanji Conversion\n")
                    continue

                if user_input.strip().lower() == ':predict':
                    current_mode = "predict"
                    print("✓ Mode: Next Word Prediction\n")
                    continue

                if user_input.strip().lower() == ':batch':
                    self.run_batch_test()
                    print()
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
                    except ValueError:
                        print("⚠ Invalid number\n")
                    continue

                if user_input.strip().lower() == ':help':
                    print("\nModes:")
                    print("  :convert  - ひらがな → 漢字 (e.g. きょう → 今日)")
                    print("  :predict  - 次の単語予測 (e.g. ありがとう → ございます)")
                    print("\nExamples:")
                    print("  [変換] きょう → 今日")
                    print("  [予測] ありがとう → ございます")
                    print()
                    continue

                if not user_input.strip():
                    continue

                # Execute based on mode
                print()

                if current_mode == "convert":
                    conversion, prob, alts, ms = self.convert_kana_to_kanji(
                        user_input, context=context, top_k=current_top_k
                    )
                    print(f"🔮 Kanji conversion for: \"{user_input}\"")
                    print(f"  → {conversion} ({prob:.1f}%)")
                    print()
                    print(f"  Top first-char alternatives:")
                    for i, (ch, p) in enumerate(alts, 1):
                        print(f"    {i}. {ch} ({p:.1f}%)")
                    print(f"\n⏱️  {ms:.0f} ms")

                elif current_mode == "predict":
                    full_context = (context + user_input) if context else user_input
                    preds, probs, ms = self.predict_next_word(
                        full_context, top_k=current_top_k
                    )
                    print(f"🔮 Next word predictions for: \"{full_context}\"")
                    for i, (word, prob) in enumerate(zip(preds, probs), 1):
                        print(f"  {i}. {full_context}【{word}】 ({prob:.1f}%)")
                    print(f"\n⏱️  {ms:.0f} ms")

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
    parser = argparse.ArgumentParser(
        description="BiGRU Shared Encoder model tester (KKC + NWP)"
    )
    parser.add_argument(
        "--model-dir", type=str,
        default="./models/japanese/multitask_v2",
        help="Path to model directory"
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Run batch test (non-interactive)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        print(f"❌ Model not found: {args.model_dir}")
        print("💡 Train the model first using the notebook!")
        sys.exit(1)

    try:
        tester = BiGRUKeyboardTester(model_dir=args.model_dir)
        if args.batch:
            tester.run_batch_test()
        else:
            tester.run_interactive()
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
