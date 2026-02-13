"""
Data loading and cache management for Multi-Task GRU.
Handles loading datasets (custom JSONL or zenz fallback),
building both KKC and NWP training arrays, and saving/loading from cache.
"""
import os
import gc
import re
import json
import numpy as np
from tqdm.auto import tqdm

from . import config
from .tokenizer import (
    tokenize_with_sep, encode_encoder_input, encode_decoder_seq,
    tokenize_words, encode_words,
    build_char_vocab, build_word_vocab,
)


# Regex to check if input contains kana (hiragana or katakana)
_KANA_PATTERN = re.compile(r'[\u3040-\u309F\u30A0-\u30FF]')


# ===========================================================
# CACHE CHECK
# ===========================================================

def check_cache(cache_paths):
    """Check if all cache files exist.

    Returns:
        (kkc_ready, nwp_ready): tuple of booleans
    """
    kkc_files = ['char_vocab', 'kkc_encoder', 'kkc_dec_in', 'kkc_dec_tgt']
    nwp_files = ['nwp_vocab', 'nwp_x', 'nwp_y']

    kkc_ready = all(os.path.exists(cache_paths[k]) for k in kkc_files)
    nwp_ready = all(os.path.exists(cache_paths[k]) for k in nwp_files)

    return kkc_ready, nwp_ready


# ===========================================================
# LOAD RAW DATASET
# ===========================================================

def load_raw_dataset(max_samples=None, source='auto'):
    """Load training dataset.

    Supports two data sources:
    - 'custom': Load from local JSONL file (ime_dataset_10m.jsonl on Drive)
    - 'zenz': Load from HuggingFace (Miwa-Keita/zenz-v2.5-dataset)
    - 'auto' (default): Try custom first, fallback to zenz

    Custom JSONL format (one JSON per line):
        {"left_context": "...", "input": "かな", "output": "漢字"}

    Returns:
        list of dicts with keys: input, output, left_context, raw_kana, input_len
    """
    if max_samples is None:
        max_samples = config.MAX_SAMPLES

    if source == 'auto':
        custom_path = f'{config.DATASET_DIR}/ime_dataset_10m.jsonl'
        if os.path.exists(custom_path):
            source = 'custom'
            print(f"📂 Found custom dataset: {custom_path}")
        else:
            source = 'zenz'
            print(f"⚠️ Custom dataset not found at {custom_path}")
            print(f"   Falling back to zenz dataset from HuggingFace")

    if source == 'custom':
        raw_items = _load_custom_jsonl(max_samples)
    else:
        raw_items = _load_zenz_hf(max_samples)

    # Filter and prepare
    training_data = _filter_items(raw_items, max_samples)

    # Sort by input length for bucketing (helps GRU training stability)
    training_data.sort(key=lambda x: x['input_len'])
    lengths = [d['input_len'] for d in training_data]
    print(f"  Bucketed: short(1-5)={sum(1 for l in lengths if l <= 5):,}, "
          f"med(6-15)={sum(1 for l in lengths if 5 < l <= 15):,}, "
          f"long(16+)={sum(1 for l in lengths if l > 15):,}")

    return training_data


def _load_custom_jsonl(max_samples):
    """Load from local JSONL file (ime_dataset_10m.jsonl).

    Expected path: {DATASET_DIR}/ime_dataset_10m.jsonl
    Each line: {"left_context": "...", "input": "かな", "output": "漢字"}
    """
    path = f'{config.DATASET_DIR}/ime_dataset_10m.jsonl'
    print(f"📥 Loading custom dataset: {path}")

    raw_items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Reading JSONL")):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                raw_items.append(item)
            except json.JSONDecodeError:
                continue  # Skip malformed lines
            if len(raw_items) >= max_samples * 2:
                # Load extra to account for filtering
                break

    print(f"  ✓ Read {len(raw_items):,} items from JSONL")
    return raw_items


def _load_zenz_hf(max_samples):
    """Load from HuggingFace zenz dataset (fallback)."""
    from datasets import load_dataset

    print("📥 Loading zenz dataset from HuggingFace...")
    dataset = load_dataset(
        "Miwa-Keita/zenz-v2.5-dataset",
        data_files="train_wikipedia.jsonl",
        split="train"
    )
    print(f"  ✓ Raw: {len(dataset):,} items")

    # Convert to list of dicts (same format as custom)
    raw_items = []
    for item in tqdm(dataset, desc="Processing"):
        raw_items.append({
            'left_context': item.get('left_context', '') or '',
            'input': item.get('input', '') or '',
            'output': item.get('output', '') or '',
        })
        if len(raw_items) >= max_samples * 2:
            break

    del dataset
    gc.collect()
    return raw_items


def _filter_items(raw_items, max_samples):
    """Filter and prepare training items.

    Applies:
    - Empty check
    - Kana content check (input must have hiragana/katakana)
    - Length limits
    - Context truncation (keep last N chars)
    """
    training_data = []
    skipped = {'empty': 0, 'no_kana': 0, 'too_long': 0}

    for item in tqdm(raw_items, desc="Filtering"):
        left_ctx = (item.get('left_context', '') or '').strip()
        kana_input = (item.get('input', '') or '').strip()
        kanji_output = (item.get('output', '') or '').strip()

        # Skip empty
        if not kana_input or not kanji_output:
            skipped['empty'] += 1
            continue

        # Check input contains actual kana (hiragana or katakana)
        if not _KANA_PATTERN.search(kana_input):
            skipped['no_kana'] += 1
            continue

        # Length limits
        if len(kana_input) > config.MAX_INPUT_LEN:
            skipped['too_long'] += 1
            continue
        if len(kanji_output) > config.MAX_OUTPUT_LEN:
            skipped['too_long'] += 1
            continue

        # Truncate context to max (keep last N chars)
        if len(left_ctx) > config.MAX_CONTEXT_LEN:
            left_ctx = left_ctx[-config.MAX_CONTEXT_LEN:]

        # Build encoder input: context<SEP>kana
        enc_input = f"{left_ctx}<SEP>{kana_input}" if left_ctx else f"<SEP>{kana_input}"

        training_data.append({
            'input': enc_input,
            'output': kanji_output,
            'left_context': left_ctx,
            'raw_kana': kana_input,
            'input_len': len(kana_input),
        })

        if len(training_data) >= max_samples:
            break

    print(f"  ✓ {len(training_data):,} valid training items")
    print(f"    Skipped: empty={skipped['empty']:,}, "
          f"no_kana={skipped['no_kana']:,}, too_long={skipped['too_long']:,}")

    # Show sample data
    print("\n📝 Sample data:")
    for d in training_data[:5]:
        ctx = d['left_context'][:20] or '(none)'
        print(f"  ctx: {ctx} | {d['raw_kana']} → {d['output']}")

    del raw_items
    gc.collect()
    return training_data


def _augment_with_prefixes(training_data, ratio=0.3):
    """Augment KKC data with prefix variants for partial-input prediction.

    For each selected sample (e.g., あつい→暑い with context 今日はとても),
    generates prefix variants:
        今日はとても<SEP>あ   → 暑い  (prefix 1 char)
        今日はとても<SEP>あつ  → 暑い  (prefix 2 chars)
        今日はとても<SEP>あつい → 暑い  (full word — original, already in data)

    Args:
        training_data: list of training dicts from _filter_items
        ratio: fraction of samples to augment (0.3 = 30%)

    Returns:
        training_data with prefix variants appended
    """
    # Only augment samples with kana length >= 2 (single chars can't have prefixes)
    candidates = [d for d in training_data if d['input_len'] >= 2]
    n_augment = int(len(candidates) * ratio)

    if n_augment == 0:
        print("  ⚠ No candidates for prefix augmentation")
        return training_data

    # Random sample without replacement
    selected = np.random.choice(len(candidates), size=n_augment, replace=False)

    augmented = []
    for idx in selected:
        d = candidates[idx]
        kana = d['raw_kana']
        ctx = d['left_context']
        output = d['output']

        # Generate prefix variants (all prefixes shorter than full kana)
        for plen in range(1, len(kana)):
            prefix = kana[:plen]
            enc_prefix = f"{ctx}<SEP>{prefix}" if ctx else f"<SEP>{prefix}"
            augmented.append({
                'input': enc_prefix,
                'output': output,
                'left_context': ctx,
                'raw_kana': prefix,
                'input_len': len(prefix),
            })

    training_data.extend(augmented)
    print(f"  ✓ Prefix augmentation: {n_augment:,} samples → {len(augmented):,} prefix variants added")
    print(f"    Total training data: {len(training_data):,}")

    # Show examples
    print("\n📝 Prefix augmentation examples:")
    shown = 0
    for idx in selected[:3]:  # Show first 3 augmented samples
        d = candidates[idx]
        kana = d['raw_kana']
        ctx_short = d['left_context'][:15] or '(none)'
        print(f"  Original: {ctx_short}... | {kana} → {d['output']}")
        for plen in range(1, len(kana)):
            prefix = kana[:plen]
            print(f"    prefix: {ctx_short}... | {prefix} → {d['output']}")
        shown += 1

    return training_data


# ===========================================================
# BUILD KKC CACHE
# ===========================================================

def build_kkc_cache(training_data, cache_paths):
    """Build and save KKC training arrays.

    Creates:
    - char_vocab.json: character vocabulary
    - kkc_encoder.npy: encoder input arrays
    - kkc_decoder_input.npy: decoder input (with BOS)
    - kkc_decoder_target.npy: decoder target (with EOS)
    - kkc_test_cases.json: meaningful test cases
    """
    print("\n🔨 Building KKC cache...")

    # Build char vocabulary from ORIGINAL data (before prefix augmentation)
    char_to_idx, idx_to_char = build_char_vocab(training_data)
    vocab_size = len(char_to_idx)
    print(f"  Char vocab: {vocab_size:,} characters")

    # Save vocab
    with open(cache_paths['char_vocab'], 'w', encoding='utf-8') as f:
        json.dump({
            'char_to_idx': char_to_idx,
            'idx_to_char': {str(k): v for k, v in idx_to_char.items()},
        }, f, ensure_ascii=False)

    PAD = char_to_idx['<PAD>']
    UNK = char_to_idx['<UNK>']

    # Save meaningful test cases from ORIGINAL data (before prefix augmentation)
    # Data is sorted by input_len ascending. Most items are len=1 particles,
    # so scan from the END (longest inputs) to find real kana→kanji conversions.
    test_cases = []
    seen_kana = set()  # Deduplicate by kana to get diverse test cases
    debug_count = 0
    for d in reversed(training_data):
        if len(test_cases) >= 50:
            break
        kana = d['raw_kana']
        output = d['output']
        context = d['left_context']

        # Skip if we already have a test case for this kana
        if kana in seen_kana:
            continue

        # Filter: all chars in vocab, meaningful conversion
        kana_missing = [c for c in kana if c not in char_to_idx]
        out_missing = [c for c in output if c not in char_to_idx]
        ctx_missing = [c for c in context if c not in char_to_idx] if context else []

        all_kana_in_vocab = len(kana_missing) == 0
        all_output_in_vocab = len(out_missing) == 0
        all_ctx_in_vocab = len(ctx_missing) == 0

        # Must be a real kana→kanji conversion (output differs from input)
        is_conversion = kana != output
        has_good_length = 2 <= len(kana) <= 15 and 1 <= len(output) <= 15

        if (all_kana_in_vocab and all_output_in_vocab and all_ctx_in_vocab
                and is_conversion and has_good_length):
            test_cases.append({
                'context': context,
                'kana': kana,
                'expected': output,
            })
            seen_kana.add(kana)
        elif debug_count < 5:
            # Show why this item was rejected
            reasons = []
            if not all_kana_in_vocab:
                reasons.append(f"kana_missing={kana_missing[:3]}")
            if not all_output_in_vocab:
                reasons.append(f"out_missing={out_missing[:3]}")
            if not all_ctx_in_vocab:
                reasons.append(f"ctx_missing={ctx_missing[:3]}")
            if not is_conversion:
                reasons.append("kana==output")
            if not has_good_length:
                reasons.append(f"bad_len(kana={len(kana)},out={len(output)})")
            print(f"    [SKIP] '{kana}'→'{output}' ctx='{context[:10]}...' | {', '.join(reasons)}")
            debug_count += 1

    with open(cache_paths['kkc_test_cases'], 'w', encoding='utf-8') as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Saved {len(test_cases)} KKC test cases")

    # --- Prefix augmentation AFTER test cases (KKC only, not NWP) ---
    prefix_ratio = getattr(config, 'PREFIX_AUG_RATIO', 0.3)
    if prefix_ratio > 0:
        training_data = _augment_with_prefixes(training_data, prefix_ratio)

    n = len(training_data)

    # --- Encoder input arrays ---
    print(f"  Encoding {n:,} samples...")
    arr = np.zeros((n, config.MAX_ENCODER_LEN), dtype=np.int32)
    for i, d in enumerate(tqdm(training_data, desc="Encoder")):
        arr[i] = encode_encoder_input(d['input'], char_to_idx, PAD, UNK)
    np.save(cache_paths['kkc_encoder'], arr)
    del arr; gc.collect()

    # --- Decoder input (with BOS) ---
    arr = np.zeros((n, config.MAX_DECODER_LEN), dtype=np.int32)
    for i, d in enumerate(tqdm(training_data, desc="Dec input")):
        arr[i] = encode_decoder_seq(d['output'], char_to_idx, PAD, UNK, add_bos=True)
    # Verify BOS is at position 0
    assert arr[0][0] == char_to_idx['<BOS>'], f"Expected BOS, got {arr[0][0]}"
    np.save(cache_paths['kkc_dec_in'], arr)
    del arr; gc.collect()

    # --- Decoder target (with EOS) ---
    arr = np.zeros((n, config.MAX_DECODER_LEN), dtype=np.int32)
    for i, d in enumerate(tqdm(training_data, desc="Dec target")):
        arr[i] = encode_decoder_seq(d['output'], char_to_idx, PAD, UNK, add_eos=True)
    # Verify EOS is present
    assert char_to_idx['<EOS>'] in list(arr[0]), "Decoder target should contain EOS"
    np.save(cache_paths['kkc_dec_tgt'], arr)
    del arr; gc.collect()

    print(f"  ✓ KKC cache saved → {cache_paths['kkc_encoder']}")
    return char_to_idx, idx_to_char


# ===========================================================
# BUILD NWP CACHE
# ===========================================================

def build_nwp_cache(training_data, cache_paths):
    """Build and save NWP training arrays.

    Uses left_context + output as full sentences, then creates
    word-level sliding window pairs: [w1,w2,w3] → w4

    Creates:
    - nwp_word_vocab.json: word vocabulary
    - nwp_x.npy: context word arrays
    - nwp_y.npy: target word IDs
    - nwp_test_cases.json: meaningful test cases
    """
    from collections import Counter

    print("\n🔨 Building NWP cache...")

    # Pass 1: tokenize sentences + count words
    print("  Pass 1: Tokenizing words...")
    word_counts = Counter()
    all_sentences = []

    for d in tqdm(training_data, desc="Word tokenize"):
        text = d['left_context'] + d['output']
        if not text.strip():
            continue
        words = tokenize_words(text)
        if len(words) < 3:  # Skip very short fragments
            continue
        word_counts.update(words)
        all_sentences.append((words, text))

    print(f"  ✓ {len(all_sentences):,} sentences, {len(word_counts):,} unique words")
    print(f"  Top 15: {[w for w, c in word_counts.most_common(15)]}")

    # Build word vocabulary
    word_to_idx, idx_to_word = build_word_vocab(all_sentences)

    # Save vocab
    with open(cache_paths['nwp_vocab'], 'w', encoding='utf-8') as f:
        json.dump({
            'word_to_idx': word_to_idx,
            'idx_to_word': {str(k): v for k, v in idx_to_word.items()},
        }, f, ensure_ascii=False)

    PAD = word_to_idx['<PAD>']
    UNK = word_to_idx['<UNK>']
    max_pairs = config.MAX_NWP_PAIRS

    # Pass 2: create sliding window pairs
    print(f"  Pass 2: Creating NWP pairs (max {max_pairs:,})...")
    X = np.zeros((max_pairs, config.MAX_WORD_CONTEXT), dtype=np.int32)
    y = np.zeros(max_pairs, dtype=np.int32)
    pair_idx = 0

    test_cases = []

    for words, original_text in tqdm(all_sentences, desc="NWP pairs"):
        if len(words) < 2:
            continue

        all_in_vocab = all(w in word_to_idx for w in words)

        # Create sliding window pairs: context → next_word
        for i in range(1, len(words)):
            next_word = words[i]
            if next_word not in word_to_idx:
                continue
            context = words[max(0, i - config.MAX_WORD_CONTEXT):i]
            X[pair_idx] = encode_words(context, word_to_idx, PAD, UNK)
            y[pair_idx] = word_to_idx[next_word]
            pair_idx += 1
            if pair_idx >= max_pairs:
                break

        # Save test case (clean sentence, >= 4 words, no punctuation target)
        if all_in_vocab and len(words) >= 4 and len(test_cases) < 50:
            for i in range(2, len(words)):  # Start from 3rd word
                nw = words[i]
                # Skip punctuation as target (we want real words)
                if nw in ['、', '。', '・', '（', '）', '「', '」', '！', '？']:
                    continue
                if nw not in word_to_idx:
                    continue
                context = words[max(0, i - config.MAX_WORD_CONTEXT):i]
                test_cases.append({
                    'context': context,
                    'expected': nw,
                    'sentence': ''.join(words),
                })
                break  # One test case per sentence

        if pair_idx >= max_pairs:
            break

    # Trim to actual size
    X = X[:pair_idx]
    y = y[:pair_idx]
    print(f"  ✓ {pair_idx:,} NWP pairs created")
    print(f"  Avg pairs/sentence: {pair_idx / max(len(all_sentences), 1):.1f}")

    # Show sample pairs
    print("\n📝 Sample NWP pairs:")
    idx_to_word_local = {v: k for k, v in word_to_idx.items()}
    for i in range(min(8, pair_idx)):
        ctx = [idx_to_word_local.get(int(idx), '?') for idx in X[i] if idx != PAD]
        tgt = idx_to_word_local.get(int(y[i]), '?')
        print(f"  [{', '.join(ctx)}] → {tgt}")

    # Save
    np.save(cache_paths['nwp_x'], X)
    np.save(cache_paths['nwp_y'], y)
    with open(cache_paths['nwp_test_cases'], 'w', encoding='utf-8') as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Saved {len(test_cases)} NWP test cases")

    del X, y, all_sentences, word_counts
    gc.collect()

    return word_to_idx, idx_to_word


# ===========================================================
# LOAD FROM CACHE
# ===========================================================

def load_kkc_cache(cache_paths):
    """Load KKC data from cache (memory-mapped).

    Returns:
        char_to_idx, idx_to_char, enc_mmap, dec_in_mmap, dec_tgt_mmap
    """
    with open(cache_paths['char_vocab'], 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)

    char_to_idx = vocab_data['char_to_idx']
    idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}

    enc_mmap = np.load(cache_paths['kkc_encoder'], mmap_mode='r')
    dec_in_mmap = np.load(cache_paths['kkc_dec_in'], mmap_mode='r')
    dec_tgt_mmap = np.load(cache_paths['kkc_dec_tgt'], mmap_mode='r')

    print(f"  ✓ KKC cache loaded: {len(enc_mmap):,} samples (mmap)")
    return char_to_idx, idx_to_char, enc_mmap, dec_in_mmap, dec_tgt_mmap


def load_nwp_cache(cache_paths):
    """Load NWP data from cache (memory-mapped).

    Returns:
        word_to_idx, idx_to_word, x_mmap, y_mmap
    """
    with open(cache_paths['nwp_vocab'], 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)

    word_to_idx = vocab_data['word_to_idx']
    idx_to_word = {int(k): v for k, v in vocab_data['idx_to_word'].items()}

    x_mmap = np.load(cache_paths['nwp_x'], mmap_mode='r')
    y_mmap = np.load(cache_paths['nwp_y'], mmap_mode='r')

    print(f"  ✓ NWP cache loaded: {len(x_mmap):,} pairs (mmap)")
    return word_to_idx, idx_to_word, x_mmap, y_mmap


# ===========================================================
# SHARED ENCODER: NWP CHAR-LEVEL CACHE (v2)
# ===========================================================

def _encode_nwp_context_chars(context_words, char_to_idx, max_len):
    """Convert context words to char IDs with <SEP> boundary markers.

    Example: ["今日", "は", "天気"] → [今_id, 日_id, SEP, は_id, SEP, 天_id, 気_id, 0, ...]

    The <SEP> markers help the shared encoder learn word boundaries,
    which is critical for NWP (next word prediction) since the target
    is a whole word.

    Args:
        context_words: list of word strings (e.g., ["今日", "は", "天気"])
        char_to_idx: character vocabulary mapping
        max_len: pad/truncate to this length (= MAX_ENCODER_LEN)

    Returns:
        np.array of shape (max_len,) with char IDs
    """
    char_ids = []
    for i, word in enumerate(context_words):
        if i > 0:
            char_ids.append(config.SEP_IDX)  # Word boundary marker
        for ch in word:
            char_ids.append(char_to_idx.get(ch, config.UNK_IDX))
            if len(char_ids) >= max_len:
                break
        if len(char_ids) >= max_len:
            break

    result = np.zeros(max_len, dtype=np.int32)
    n = min(len(char_ids), max_len)
    result[:n] = char_ids[:n]
    return result


def build_nwp_char_cache(training_data, cache_paths, char_to_idx):
    """Build NWP training arrays with char-level context (shared encoder v2).

    Instead of word IDs as input, converts context words to character
    sequences with <SEP> markers between words:
        "今日<SEP>は<SEP>天気" → char IDs (same shape as KKC encoder input)

    This enables the shared encoder to process both KKC and NWP data.

    Creates:
    - nwp_word_vocab.json: word vocabulary (same as v1)
    - nwp_char_x.npy: char-level context arrays (MAX_ENCODER_LEN,)
    - nwp_y.npy: target word IDs (same as v1)
    - nwp_test_cases.json: test cases
    """
    from collections import Counter

    print("\n🔨 Building NWP char cache (shared encoder v2)...")

    # Pass 1: tokenize + count words
    print("  Pass 1: Tokenizing words...")
    word_counts = Counter()
    all_sentences = []

    for d in tqdm(training_data, desc="Word tokenize"):
        text = d['left_context'] + d['output']
        if not text.strip():
            continue
        words = tokenize_words(text)
        if len(words) < 3:
            continue
        word_counts.update(words)
        all_sentences.append((words, text))

    print(f"  ✓ {len(all_sentences):,} sentences, {len(word_counts):,} unique words")
    print(f"  Top 15: {[w for w, c in word_counts.most_common(15)]}")

    # Build word vocabulary
    word_to_idx, idx_to_word = build_word_vocab(all_sentences)

    # Save vocab
    with open(cache_paths['nwp_vocab'], 'w', encoding='utf-8') as f:
        json.dump({
            'word_to_idx': word_to_idx,
            'idx_to_word': {str(k): v for k, v in idx_to_word.items()},
        }, f, ensure_ascii=False)

    max_pairs = config.MAX_NWP_PAIRS

    # Pass 2: create char-level NWP pairs
    # Shuffle to sample from full diversity (data is sorted by input_len)
    np.random.shuffle(all_sentences)
    print(f"  Pass 2: Creating NWP char pairs (max {max_pairs:,})...")
    X = np.zeros((max_pairs, config.MAX_ENCODER_LEN), dtype=np.int32)
    y = np.zeros(max_pairs, dtype=np.int32)
    pair_idx = 0
    test_cases = []

    import random
    MAX_PAIRS_PER_SENTENCE = 1  # 1 random pair per sentence → 100% coverage, halved steps
    sentences_used = 0

    for words, original_text in tqdm(all_sentences, desc="NWP char pairs"):
        if len(words) < 2:
            continue

        # Collect valid positions where next_word is in vocab
        valid_positions = []
        for i in range(1, len(words)):
            if words[i] in word_to_idx:
                valid_positions.append(i)

        if not valid_positions:
            continue

        # Randomly sample up to MAX_PAIRS_PER_SENTENCE positions
        sampled = random.sample(valid_positions, min(MAX_PAIRS_PER_SENTENCE, len(valid_positions)))
        sentences_used += 1

        for i in sampled:
            next_word = words[i]
            context = words[max(0, i - config.MAX_WORD_CONTEXT):i]
            X[pair_idx] = _encode_nwp_context_chars(
                context, char_to_idx, config.MAX_ENCODER_LEN
            )
            y[pair_idx] = word_to_idx[next_word]
            pair_idx += 1
            if pair_idx >= max_pairs:
                break

        # Save test case (clean sentence, >= 4 words)
        all_in_vocab = all(w in word_to_idx for w in words)
        if all_in_vocab and len(words) >= 4 and len(test_cases) < 50:
            for i in range(2, len(words)):
                nw = words[i]
                if nw in ['、', '。', '・', '（', '）', '「', '」', '！', '？']:
                    continue
                if nw not in word_to_idx:
                    continue
                context = words[max(0, i - config.MAX_WORD_CONTEXT):i]
                test_cases.append({
                    'context': context,
                    'expected': nw,
                    'sentence': ''.join(words),
                })
                break

        if pair_idx >= max_pairs:
            break

    # Trim to actual size
    X = X[:pair_idx]
    y = y[:pair_idx]
    coverage_pct = sentences_used / max(len(all_sentences), 1) * 100
    print(f"  ✓ {pair_idx:,} NWP char pairs from {sentences_used:,}/{len(all_sentences):,} sentences ({coverage_pct:.1f}% coverage)")

    # Show sample pairs
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    idx_to_word_local = {v: k for k, v in word_to_idx.items()}
    print("\n📝 Sample NWP char pairs (with <SEP> markers):")
    for i in range(min(5, pair_idx)):
        chars = [idx_to_char.get(int(c), '?') for c in X[i] if c != 0]
        tgt = idx_to_word_local.get(int(y[i]), '?')
        print(f"  {''.join(chars)} → {tgt}")

    # Save
    np.save(cache_paths['nwp_char_x'], X)
    np.save(cache_paths['nwp_y'], y)
    with open(cache_paths['nwp_test_cases'], 'w', encoding='utf-8') as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Saved {len(test_cases)} NWP test cases")

    del X, y, all_sentences, word_counts
    gc.collect()

    return word_to_idx, idx_to_word


def load_nwp_char_cache(cache_paths):
    """Load NWP char-level data from cache (shared encoder v2).

    Returns:
        word_to_idx, idx_to_word, char_x_mmap, y_mmap
    """
    with open(cache_paths['nwp_vocab'], 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)

    word_to_idx = vocab_data['word_to_idx']
    idx_to_word = {int(k): v for k, v in vocab_data['idx_to_word'].items()}

    char_x_mmap = np.load(cache_paths['nwp_char_x'], mmap_mode='r')
    y_mmap = np.load(cache_paths['nwp_y'], mmap_mode='r')

    print(f"  ✓ NWP char cache loaded: {len(char_x_mmap):,} pairs (mmap)")
    return word_to_idx, idx_to_word, char_x_mmap, y_mmap
