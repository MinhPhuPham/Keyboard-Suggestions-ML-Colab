"""
Custom Tokenizer for Keyboard Model
Builds a small vocabulary (10k words) from frequency data
"""

import csv
import json
import pickle
from typing import List, Dict, Optional
from pathlib import Path


class KeyboardTokenizer:
    """
    Custom tokenizer with dynamic vocabulary for efficient mobile deployment
    
    Features:
    - Dynamic vocabulary built from training data (100% coverage)
    - Or fixed vocabulary from frequency file
    - Special tokens: [PAD], [UNK], [MASK]
    - Fast encoding/decoding
    - Serializable for mobile export
    """
    
    def __init__(self, vocab_size: int = 10000):
        """
        Initialize tokenizer
        
        Args:
            vocab_size: Maximum vocabulary size (default: 10000)
        """
        self.vocab_size = vocab_size
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        
        # Special tokens
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.mask_token = "[MASK]"
        
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.mask_token_id = 2
        
        self._is_built = False
    
    def build_vocab(self, word_freq_path: str, min_freq: int = 10) -> None:
        """
        Build vocabulary from word frequency file
        
        Args:
            word_freq_path: Path to single_word_freq.csv
            min_freq: Minimum frequency threshold
        """
        print(f"Building vocabulary from {word_freq_path}...")
        
        # Read word frequencies
        words_with_freq = []
        with open(word_freq_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = row['word'].strip().lower()
                freq = int(row.get('count_frequency', row.get('count', 1)))
                
                # Filter: alphabetic, length >= 2, frequency >= min_freq
                if (len(word) >= 2 and 
                    word.isalpha() and 
                    freq >= min_freq):
                    words_with_freq.append((word, freq))
        
        # Sort by frequency (descending)
        words_with_freq.sort(key=lambda x: x[1], reverse=True)
        
        # Take top vocab_size - 3 (reserve for special tokens)
        top_words = [w for w, f in words_with_freq[:self.vocab_size - 3]]
        
        # Build vocabulary
        self.word2idx = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.mask_token: self.mask_token_id
        }
        
        for idx, word in enumerate(top_words, start=3):
            self.word2idx[word] = idx
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self._is_built = True
        
        print(f"✓ Vocabulary built: {len(self.word2idx):,} words")
        print(f"  - Special tokens: 3")
        print(f"  - Regular words: {len(self.word2idx) - 3:,}")
        print(f"  - Coverage: Top {len(top_words):,} most frequent words")
    
    def build_vocab_from_training_data(
        self, 
        train_data_path: str,
        max_vocab_size: int = 20000
    ) -> None:
        """
        Build vocabulary dynamically from training data (100% coverage)
        
        This ensures ALL target words in training data are in vocabulary,
        eliminating the [UNK] problem that causes high loss.
        
        Args:
            train_data_path: Path to train.jsonl file
            max_vocab_size: Maximum vocabulary size (safety limit)
        """
        from collections import Counter
        
        print(f"Building dynamic vocabulary from {train_data_path}...")
        
        # Collect all target words from training data
        target_words = []
        with open(train_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    target = sample['target'].strip().lower()
                    if target and target.isalpha():
                        target_words.append(target)
                except (json.JSONDecodeError, KeyError):
                    continue
        
        # Count frequencies
        word_freq = Counter(target_words)
        
        # Get most common words (up to max_vocab_size - 3 for special tokens)
        most_common = word_freq.most_common(max_vocab_size - 3)
        
        # Build vocabulary
        self.word2idx = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.mask_token: self.mask_token_id
        }
        
        for idx, (word, freq) in enumerate(most_common, start=3):
            self.word2idx[word] = idx
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self._is_built = True
        
        # Calculate coverage
        total_targets = len(target_words)
        covered_targets = sum(1 for w in target_words if w in self.word2idx)
        coverage = covered_targets / total_targets * 100 if total_targets > 0 else 0
        
        print(f"✓ Dynamic vocabulary built: {len(self.word2idx):,} words")
        print(f"  - Special tokens: 3")
        print(f"  - Regular words: {len(self.word2idx) - 3:,}")
        print(f"  - Unique targets in data: {len(word_freq):,}")
        print(f"  - Coverage: {coverage:.2f}% of training targets")
        
        if coverage < 99.0:
            print(f"  ⚠️  Warning: Coverage < 99%. Consider increasing max_vocab_size.")
    
    def encode(
        self, 
        text: str, 
        max_length: int = 16, 
        padding: bool = True,
        add_special_tokens: bool = True
    ) -> List[int]:
        """
        Encode text to token IDs
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            padding: Whether to pad to max_length
            add_special_tokens: Whether to add special tokens (not used for now)
            
        Returns:
            List of token IDs
        """
        if not self._is_built:
            raise ValueError("Vocabulary not built! Call build_vocab() first.")
        
        # Tokenize by whitespace
        words = text.lower().split()
        
        # Convert to IDs
        ids = [self.word2idx.get(w, self.unk_token_id) for w in words]
        
        # Truncate if needed
        if len(ids) > max_length:
            ids = ids[:max_length]
        
        # Pad if needed
        if padding and len(ids) < max_length:
            ids = ids + [self.pad_token_id] * (max_length - len(ids))
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text
        
        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        if not self._is_built:
            raise ValueError("Vocabulary not built! Call build_vocab() first.")
        
        words = []
        for idx in ids:
            word = self.idx2word.get(idx, self.unk_token)
            
            # Skip special tokens if requested
            if skip_special_tokens and word in [self.pad_token, self.unk_token, self.mask_token]:
                continue
            
            words.append(word)
        
        return ' '.join(words)
    
    def save(self, save_path: str) -> None:
        """
        Save tokenizer to file
        
        Args:
            save_path: Path to save tokenizer
        """
        save_dict = {
            'vocab_size': self.vocab_size,
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'special_tokens': {
                'pad_token': self.pad_token,
                'unk_token': self.unk_token,
                'mask_token': self.mask_token,
                'pad_token_id': self.pad_token_id,
                'unk_token_id': self.unk_token_id,
                'mask_token_id': self.mask_token_id
            }
        }
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"✓ Tokenizer saved to {save_path}")
    
    @classmethod
    def load(cls, load_path: str) -> 'KeyboardTokenizer':
        """
        Load tokenizer from file
        
        Args:
            load_path: Path to load tokenizer from
            
        Returns:
            Loaded tokenizer
        """
        with open(load_path, 'rb') as f:
            save_dict = pickle.load(f)
        
        tokenizer = cls(vocab_size=save_dict['vocab_size'])
        tokenizer.word2idx = save_dict['word2idx']
        tokenizer.idx2word = save_dict['idx2word']
        
        # Restore special tokens
        special = save_dict['special_tokens']
        tokenizer.pad_token = special['pad_token']
        tokenizer.unk_token = special['unk_token']
        tokenizer.mask_token = special['mask_token']
        tokenizer.pad_token_id = special['pad_token_id']
        tokenizer.unk_token_id = special['unk_token_id']
        tokenizer.mask_token_id = special['mask_token_id']
        
        tokenizer._is_built = True
        
        print(f"✓ Tokenizer loaded from {load_path}")
        print(f"  Vocabulary size: {len(tokenizer.word2idx):,}")
        
        return tokenizer
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary dictionary"""
        return self.word2idx.copy()
    
    def __len__(self) -> int:
        """Get vocabulary size"""
        return len(self.word2idx)
    
    def __repr__(self) -> str:
        return f"KeyboardTokenizer(vocab_size={len(self.word2idx):,}, built={self._is_built})"
