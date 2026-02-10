"""
Verification for Multi-Task GRU.
Tests both KKC and NWP heads with real test cases.
"""
import os
import json
import numpy as np

from . import config
from .tokenizer import (
    encode_encoder_input, tokenize_words, encode_words,
)


# ===========================================================
# KKC VERIFICATION
# ===========================================================

def verify_kkc(model, char_to_idx, idx_to_char, cache_paths):
    """Test KKC head with saved test cases.

    Runs autoregressive decoding (greedy) and checks
    exact/partial match against expected output.
    """
    PAD = char_to_idx['<PAD>']
    BOS = char_to_idx['<BOS>']
    EOS = char_to_idx['<EOS>']
    UNK = char_to_idx['<UNK>']
    SEP = char_to_idx['<SEP>']

    def convert(context, kana):
        """Convert kana to kanji using context (greedy decode)."""
        enc_text = f"{context}<SEP>{kana}"
        enc_ids = encode_encoder_input(enc_text, char_to_idx, PAD, UNK)
        enc_in = np.array([enc_ids], dtype=np.int32)

        dec_in = np.zeros((1, config.MAX_DECODER_LEN), dtype=np.int32)
        dec_in[0, 0] = BOS

        result = []
        for i in range(config.MAX_DECODER_LEN - 1):
            # NWP input (zeros, not used during KKC inference)
            nwp_dummy = np.zeros((1, config.MAX_WORD_CONTEXT), dtype=np.int32)
            pred = model.predict(
                {
                    'encoder_input': enc_in,
                    'decoder_input': dec_in,
                    'nwp_input': nwp_dummy,
                },
                verbose=0
            )
            kkc_pred = pred[0]  # First output = KKC
            next_id = int(np.argmax(kkc_pred[0, i]))

            if next_id == EOS:
                break
            if next_id not in [PAD, BOS, EOS, UNK, SEP]:
                result.append(idx_to_char.get(next_id, ''))

            if i + 1 < config.MAX_DECODER_LEN:
                dec_in[0, i + 1] = next_id

        return ''.join(result)

    # Load test cases
    test_cases_path = cache_paths.get('kkc_test_cases', '')
    if os.path.exists(test_cases_path):
        with open(test_cases_path, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        print(f"✓ Loaded {len(test_cases)} KKC test cases")
    else:
        test_cases = [
            {'context': '今日はとても', 'kana': 'あつい', 'expected': '暑い'},
            {'context': 'お茶が', 'kana': 'あつい', 'expected': '熱い'},
            {'context': '川に', 'kana': 'はし', 'expected': '橋'},
            {'context': 'ご飯を', 'kana': 'はし', 'expected': '箸'},
        ]
        print("⚠️ Using default KKC test cases")

    # Run
    print(f"\n{'='*60}")
    print("KKC HEAD VERIFICATION")
    print(f"{'='*60}")

    test_subset = test_cases[:20]
    exact = 0
    partial = 0

    for tc in test_subset:
        result = convert(tc['context'], tc['kana'])
        expected = tc['expected']

        exact_match = result == expected
        partial_match = (expected in result or result in expected) and len(result) > 0

        if exact_match:
            exact += 1
            status = '✅'
        elif partial_match:
            partial += 1
            status = '🟡'
        else:
            status = '❌'

        ctx_short = tc['context'][:15] or '(none)'
        print(f"  {status} {ctx_short}<SEP>{tc['kana']}")
        print(f"       got: {result} | expected: {expected}")

    n = len(test_subset)
    print(f"\n📊 KKC Results:")
    print(f"  Exact match:   {exact}/{n} ({exact/n*100:.1f}%)")
    print(f"  Partial match: {partial}/{n} ({partial/n*100:.1f}%)")
    print(f"  Total useful:  {exact+partial}/{n} ({(exact+partial)/n*100:.1f}%)")

    return {'exact': exact, 'partial': partial, 'total': n}


# ===========================================================
# NWP VERIFICATION
# ===========================================================

def verify_nwp(model, word_to_idx, idx_to_word, cache_paths):
    """Test NWP head with saved test cases.

    Encodes word context as chars (padded to encoder length),
    feeds through model, checks top-K predictions.
    """
    PAD_W = word_to_idx['<PAD>']
    UNK_W = word_to_idx['<UNK>']
    nwp_special = {'<PAD>', '<UNK>', '<BOS>', '<EOS>'}

    def predict_top_k(context_words, top_k=5):
        """Predict top-K next words from word context."""
        # Encode words → left-padded IDs
        word_ids = encode_words(context_words, word_to_idx, PAD_W, UNK_W)
        nwp_input = np.array([word_ids], dtype=np.int32)

        # Dummy KKC inputs (not used by NWP head)
        enc_dummy = np.zeros((1, config.MAX_ENCODER_LEN), dtype=np.int32)
        dec_dummy = np.zeros((1, config.MAX_DECODER_LEN), dtype=np.int32)

        pred = model.predict(
            {
                'encoder_input': enc_dummy,
                'decoder_input': dec_dummy,
                'nwp_input': nwp_input,
            },
            verbose=0
        )
        nwp_probs = pred[1][0]  # Second output = NWP

        top_indices = np.argsort(nwp_probs)[-top_k*2:][::-1]
        results = []
        for idx in top_indices:
            word = idx_to_word.get(idx, '<UNK>')
            if word not in nwp_special:
                results.append((word, float(nwp_probs[idx])))
            if len(results) >= top_k:
                break
        return results

    # Load test cases
    test_cases_path = cache_paths.get('nwp_test_cases', '')
    if os.path.exists(test_cases_path):
        with open(test_cases_path, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        print(f"\n✓ Loaded {len(test_cases)} NWP test cases")
    else:
        print("\n⚠️ No NWP test cases found")
        return {'top1': 0, 'top5': 0, 'total': 0}

    # Run
    print(f"\n{'='*60}")
    print("NWP HEAD VERIFICATION")
    print(f"{'='*60}")

    test_subset = test_cases[:20]
    correct_top1 = 0
    correct_top5 = 0
    all_predictions = set()

    for tc in test_subset:
        preds = predict_top_k(tc['context'], top_k=5)
        pred_words = [w for w, _ in preds]
        all_predictions.update(pred_words)

        in_top1 = pred_words[0] == tc['expected'] if preds else False
        in_top5 = tc['expected'] in pred_words

        if in_top1:
            correct_top1 += 1
        if in_top5:
            correct_top5 += 1

        status = '✅' if in_top5 else '❌'
        ctx_str = ''.join(tc['context'][-5:])
        pred_str = ', '.join(pred_words[:5])
        print(f"  {status} {ctx_str} → expected: {tc['expected']}")
        print(f"       top5: [{pred_str}]")

    n = len(test_subset)
    print(f"\n📊 NWP Results:")
    print(f"  Top-1 accuracy: {correct_top1}/{n} ({correct_top1/n*100:.1f}%)")
    print(f"  Top-5 accuracy: {correct_top5}/{n} ({correct_top5/n*100:.1f}%)")
    print(f"  Unique predictions: {len(all_predictions)}")

    return {'top1': correct_top1, 'top5': correct_top5, 'total': n}


# ===========================================================
# COMBINED VERIFICATION
# ===========================================================

def verify_all(model, char_to_idx, idx_to_char, word_to_idx, idx_to_word, cache_paths):
    """Run verification for both KKC and NWP heads."""
    kkc_results = verify_kkc(model, char_to_idx, idx_to_char, cache_paths)
    nwp_results = verify_nwp(model, word_to_idx, idx_to_word, cache_paths)

    print(f"\n{'='*60}")
    print("MULTI-TASK VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"  KKC: {kkc_results['exact']+kkc_results['partial']}/{kkc_results['total']} useful")
    if nwp_results['total'] > 0:
        print(f"  NWP: {nwp_results['top5']}/{nwp_results['total']} in top-5")

    if config.TESTING_MODE:
        print("\n⚠️ TESTING MODE — accuracy expected to be lower.")
        print("  → Set TESTING_MODE = False for production training.")

    print(f"\n{'='*60}")
    print("✅ VERIFICATION COMPLETE")

    return kkc_results, nwp_results
