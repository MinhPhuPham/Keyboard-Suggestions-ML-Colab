"""
Data Preparation for Keyboard Suggestion Model

This module prepares training data for 3 tasks:
1. Word Completion: Partial words → Complete words
2. Next-Word Prediction: Context → Next word
3. Typo Correction: Typos → Corrections

Optimized for Colab training workflow.
"""

import os
import random
from typing import List, Tuple, Dict
from collections import Counter
import json


def prepare_word_completion_data(
    word_freq_path: str,
    output_path: str,
    max_samples: int = 500000
) -> str:
    """
    Generate word completion training pairs from word frequency list.
    
    Args:
        word_freq_path: Path to word frequency file (word per line or word\tfreq)
        output_path: Output JSON file path
        max_samples: Maximum number of samples to generate
    
    Returns:
        Path to generated dataset
    
    Example output:
        {"input": "hel", "target": "hello", "task": "completion"}
        {"input": "hel", "target": "help", "task": "completion"}
    """
    print(f"Preparing word completion data from {word_freq_path}...")
    
    # Load word frequencies
    words = []
    with open(word_freq_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            word = parts[0].lower()
            
            # Filter: only words 3-15 characters, alphabetic
            if 3 <= len(word) <= 15 and word.isalpha():
                words.append(word)
    
    print(f"  Loaded {len(words)} words")
    
    # Generate completion pairs
    completion_pairs = []
    for word in words:
        # Generate prefixes (1 to len-1 characters)
        for i in range(1, len(word)):
            prefix = word[:i]
            completion_pairs.append({
                "input": prefix,
                "target": word,
                "task": "completion"
            })
            
            # Limit samples
            if len(completion_pairs) >= max_samples:
                break
        
        if len(completion_pairs) >= max_samples:
            break
    
    # Shuffle
    random.shuffle(completion_pairs)
    completion_pairs = completion_pairs[:max_samples]
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in completion_pairs:
            f.write(json.dumps(pair) + '\n')
    
    print(f"✓ Generated {len(completion_pairs)} word completion pairs")
    print(f"  Saved to: {output_path}")
    
    return output_path


def prepare_nextword_data(
    corpus_path: str,
    output_path: str,
    max_samples: int = 1000000,
    context_length: int = 3
) -> str:
    """
    Generate next-word prediction pairs from text corpus.
    
    Args:
        corpus_path: Path to text corpus (one sentence per line)
        output_path: Output JSON file path
        max_samples: Maximum number of samples
        context_length: Number of context words (default: 3)
    
    Returns:
        Path to generated dataset
    
    Example output:
        {"input": "how are you", "target": "doing", "task": "next_word"}
    """
    print(f"Preparing next-word data from {corpus_path}...")
    
    nextword_pairs = []
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line_num % 10000 == 0:
                print(f"  Processed {line_num} lines, {len(nextword_pairs)} pairs...")
            
            # Tokenize
            words = line.strip().lower().split()
            
            # Skip short sentences
            if len(words) < 2:
                continue
            
            # Generate pairs
            for i in range(len(words) - 1):
                # Get context (last N words)
                start_idx = max(0, i - context_length + 1)
                context = ' '.join(words[start_idx:i+1])
                next_word = words[i + 1]
                
                # Filter: only alphabetic words
                if next_word.isalpha() and len(next_word) >= 2:
                    nextword_pairs.append({
                        "input": context,
                        "target": next_word,
                        "task": "next_word"
                    })
            
            # Limit samples
            if len(nextword_pairs) >= max_samples:
                break
    
    # Shuffle
    random.shuffle(nextword_pairs)
    nextword_pairs = nextword_pairs[:max_samples]
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in nextword_pairs:
            f.write(json.dumps(pair) + '\n')
    
    print(f"✓ Generated {len(nextword_pairs)} next-word pairs")
    print(f"  Saved to: {output_path}")
    
    return output_path


def prepare_typo_data(
    word_list_path: str,
    output_path: str,
    max_samples: int = 100000
) -> str:
    """
    Generate synthetic typo correction pairs.
    
    Args:
        word_list_path: Path to common words list
        output_path: Output JSON file path
        max_samples: Maximum number of samples
    
    Returns:
        Path to generated dataset
    
    Example output:
        {"input": "thers", "target": "there", "task": "typo"}
        {"input": "recieve", "target": "receive", "task": "typo"}
    """
    print(f"Preparing typo correction data from {word_list_path}...")
    
    # Load common words
    words = []
    with open(word_list_path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().split('\t')[0].lower()
            if 3 <= len(word) <= 12 and word.isalpha():
                words.append(word)
    
    print(f"  Loaded {len(words)} words")
    
    # Generate synthetic typos
    typo_pairs = []
    
    for word in words:
        # Generate 1-2 typos per word
        num_typos = random.randint(1, 2)
        
        for _ in range(num_typos):
            typo = generate_typo(word)
            if typo != word:  # Ensure typo is different
                typo_pairs.append({
                    "input": typo,
                    "target": word,
                    "task": "typo"
                })
        
        if len(typo_pairs) >= max_samples:
            break
    
    # Shuffle
    random.shuffle(typo_pairs)
    typo_pairs = typo_pairs[:max_samples]
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in typo_pairs:
            f.write(json.dumps(pair) + '\n')
    
    print(f"✓ Generated {len(typo_pairs)} typo correction pairs")
    print(f"  Saved to: {output_path}")
    
    return output_path


def generate_typo(word: str) -> str:
    """
    Generate a realistic typo for a word.
    
    Typo types:
    1. Character swap (adjacent)
    2. Missing character
    3. Extra character
    4. Wrong character (keyboard neighbors)
    """
    if len(word) < 3:
        return word
    
    typo_type = random.choice(['swap', 'missing', 'extra', 'wrong'])
    word_list = list(word)
    
    if typo_type == 'swap' and len(word) > 2:
        # Swap adjacent characters
        idx = random.randint(0, len(word) - 2)
        word_list[idx], word_list[idx + 1] = word_list[idx + 1], word_list[idx]
    
    elif typo_type == 'missing':
        # Remove a character
        idx = random.randint(0, len(word) - 1)
        word_list.pop(idx)
    
    elif typo_type == 'extra':
        # Add a random character
        idx = random.randint(0, len(word))
        char = random.choice('abcdefghijklmnopqrstuvwxyz')
        word_list.insert(idx, char)
    
    elif typo_type == 'wrong':
        # Replace with keyboard neighbor
        idx = random.randint(0, len(word) - 1)
        word_list[idx] = get_keyboard_neighbor(word_list[idx])
    
    return ''.join(word_list)


def get_keyboard_neighbor(char: str) -> str:
    """Get a keyboard neighbor for a character (QWERTY layout)."""
    keyboard_map = {
        'q': 'wa', 'w': 'qes', 'e': 'wrd', 'r': 'etf', 't': 'ryg',
        'y': 'tuh', 'u': 'yij', 'i': 'uok', 'o': 'ipl', 'p': 'ol',
        'a': 'qwsz', 's': 'awedxz', 'd': 'serfcx', 'f': 'drtgvc',
        'g': 'ftyhbv', 'h': 'gyujnb', 'j': 'huikmn', 'k': 'jiolm',
        'l': 'kop', 'z': 'asx', 'x': 'zsdc', 'c': 'xdfv',
        'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk'
    }
    
    neighbors = keyboard_map.get(char.lower(), 'abcdefghijklmnopqrstuvwxyz')
    return random.choice(neighbors)


def combine_datasets(
    completion_path: str,
    nextword_path: str,
    typo_path: str,
    output_path: str,
    train_ratio: float = 0.9
) -> Tuple[str, str]:
    """
    Combine all datasets and split into train/val.
    
    Args:
        completion_path: Word completion dataset
        nextword_path: Next-word prediction dataset
        typo_path: Typo correction dataset
        output_path: Output directory
        train_ratio: Training split ratio
    
    Returns:
        Tuple of (train_path, val_path)
    """
    print("Combining datasets...")
    
    # Load all datasets
    all_data = []
    
    for path in [completion_path, nextword_path, typo_path]:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                all_data.append(json.loads(line))
    
    print(f"  Total samples: {len(all_data)}")
    
    # Shuffle
    random.shuffle(all_data)
    
    # Split
    split_idx = int(len(all_data) * train_ratio)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    # Save
    os.makedirs(output_path, exist_ok=True)
    
    train_path = os.path.join(output_path, "train.jsonl")
    val_path = os.path.join(output_path, "val.jsonl")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    with open(val_path, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"✓ Train: {len(train_data)} samples → {train_path}")
    print(f"✓ Val: {len(val_data)} samples → {val_path}")
    
    return train_path, val_path


if __name__ == "__main__":
    print("Keyboard Data Preparation Utilities")
    print("="*60)
    print("\nThis module provides functions to prepare training data for:")
    print("1. Word completion")
    print("2. Next-word prediction")
    print("3. Typo correction")
    print("\nImport and use in your training notebook or scripts.")
