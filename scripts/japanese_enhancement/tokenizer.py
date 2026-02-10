"""
Tokenizers for Multi-Task GRU.
- Char-level tokenizer (shared encoder + KKC head)
- Word-level tokenizer (NWP head, using fugashi/MeCab)
"""
import re
from . import config


# ===========================================================
# CHARACTER-LEVEL TOKENIZER (Shared Encoder + KKC)
# ===========================================================

def tokenize_with_sep(text):
    """Tokenize text handling <SEP> as a single token.

    'abc<SEP>def' → ['a','b','c','<SEP>','d','e','f']
    """
    tokens = []
    i = 0
    while i < len(text):
        if text[i:i+5] == '<SEP>':
            tokens.append('<SEP>')
            i += 5
        else:
            tokens.append(text[i])
            i += 1
    return tokens


def encode_tokens(tokens, vocab, max_len, pad_id, unk_id):
    """Encode token list → padded int array (right-padded)."""
    ids = [vocab.get(t, unk_id) for t in tokens[:max_len]]
    ids += [pad_id] * (max_len - len(ids))
    return ids


def encode_encoder_input(text, vocab, pad_id, unk_id):
    """Encode 'context<SEP>kana' for the shared encoder."""
    tokens = tokenize_with_sep(text)
    return encode_tokens(tokens, vocab, config.MAX_ENCODER_LEN, pad_id, unk_id)


def encode_decoder_seq(text, vocab, pad_id, unk_id, add_bos=False, add_eos=False):
    """Encode decoder sequence with optional BOS/EOS."""
    tokens = []
    if add_bos:
        tokens.append('<BOS>')
    tokens.extend(list(text))
    if add_eos:
        tokens.append('<EOS>')
    return encode_tokens(tokens, vocab, config.MAX_DECODER_LEN, pad_id, unk_id)


# ===========================================================
# WORD-LEVEL TOKENIZER (NWP Head)
# ===========================================================

_tagger = None


def _get_tagger():
    """Lazy-init fugashi tagger (only when NWP is used)."""
    global _tagger
    if _tagger is None:
        import fugashi
        _tagger = fugashi.Tagger()
    return _tagger


def tokenize_words(text):
    """Word-level tokenization using fugashi (MeCab).

    Returns list of surface forms, skipping whitespace tokens.
    """
    if not text:
        return []
    tagger = _get_tagger()
    result = []
    for t in tagger(text):
        if t.feature.pos1 not in ['空白']:  # Skip whitespace
            result.append(t.surface)
    return result


def encode_words(words, vocab, pad_id, unk_id, max_len=None):
    """Encode word list → left-padded int array.

    Left-padding is standard for language models so the most recent
    context is closest to the output layer.
    """
    if max_len is None:
        max_len = config.MAX_WORD_CONTEXT
    ids = [vocab.get(w, unk_id) for w in words]
    if len(ids) < max_len:
        ids = [pad_id] * (max_len - len(ids)) + ids  # Left-pad
    return ids[-max_len:]  # Keep last N tokens


# ===========================================================
# VOCABULARY BUILDERS
# ===========================================================

def build_char_vocab(training_data, max_vocab=None):
    """Build character vocabulary from KKC training data.

    Args:
        training_data: list of dicts with 'input' and 'output' keys
        max_vocab: max vocabulary size (default from config)

    Returns:
        char_to_idx, idx_to_char
    """
    from collections import Counter

    if max_vocab is None:
        max_vocab = config.CHAR_VOCAB_SIZE

    char_counts = Counter()
    for d in training_data:
        # Count chars from encoder input (context<SEP>kana)
        for ch in tokenize_with_sep(d['input']):
            char_counts[ch] += 1
        # Count chars from decoder output (kanji)
        for ch in d['output']:
            char_counts[ch] += 1

    # Build vocab: special tokens first
    char_to_idx = {tok: i for i, tok in enumerate(config.SPECIAL_TOKENS)}
    for ch, _ in char_counts.most_common(max_vocab - len(config.SPECIAL_TOKENS)):
        if ch not in char_to_idx:
            char_to_idx[ch] = len(char_to_idx)

    idx_to_char = {v: k for k, v in char_to_idx.items()}
    return char_to_idx, idx_to_char


def build_word_vocab(sentences, max_vocab=None):
    """Build word vocabulary from tokenized sentences.

    Args:
        sentences: list of (word_list, text) tuples
        max_vocab: max vocabulary size

    Returns:
        word_to_idx, idx_to_word
    """
    from collections import Counter

    if max_vocab is None:
        max_vocab = config.WORD_VOCAB_SIZE

    word_counts = Counter()
    for words, _ in sentences:
        word_counts.update(words)

    # Build vocab: special tokens first (no <SEP> for NWP)
    nwp_special = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
    word_to_idx = {tok: i for i, tok in enumerate(nwp_special)}
    for word, _ in word_counts.most_common(max_vocab - len(nwp_special)):
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)

    idx_to_word = {v: k for k, v in word_to_idx.items()}

    # Stats
    total = sum(word_counts.values())
    covered = sum(c for w, c in word_counts.items() if w in word_to_idx)
    print(f"  Word vocab: {len(word_to_idx):,} words")
    print(f"  Coverage:   {covered/total*100:.1f}% of tokens")

    return word_to_idx, idx_to_word
