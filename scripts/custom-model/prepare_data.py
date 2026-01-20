#!/usr/bin/env python3
"""
Prepare training data for custom keyboard model
Generates train/val splits from raw datasets
"""

import sys
import argparse
import csv
import json
import random
from pathlib import Path
from typing import List, Dict

random.seed(42)


def prepare_word_completion_data(word_freq_path: str, max_samples: int = 50000) -> List[Dict]:
    """Generate word completion training pairs"""
    print(f"\nGenerating word completion data (max {max_samples:,})...")
    
    samples = []
    words_with_freq = []
    
    with open(word_freq_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row['word'].strip().lower()
            freq = int(row.get('count_frequency', row.get('count', 1)))
            if len(word) >= 3 and word.isalpha():
                words_with_freq.append((word, freq))
    
    words_with_freq.sort(key=lambda x: x[1], reverse=True)
    words_with_freq = words_with_freq[:15000]  # Top 15k words
    
    for word, freq in words_with_freq:
        if len(samples) >= max_samples:
            break
        
        num_samples = min(5, max(1, freq // 10000))
        
        for i in range(num_samples):
            if len(samples) >= max_samples:
                break
            
            prefix_ratio = 0.4 + (i * 0.1)
            prefix_len = max(1, int(len(word) * prefix_ratio))
            
            if prefix_len < len(word):
                samples.append({
                    'input': word[:prefix_len],
                    'target': word,
                    'task': 'completion'
                })
    
    print(f"  Generated {len(samples):,} completion pairs")
    return samples


def prepare_nextword_data(corpus_path: str, max_samples: int = 100000, context_length: int = 3) -> List[Dict]:
    """Generate next-word prediction pairs"""
    print(f"\nGenerating next-word prediction data (max {max_samples:,})...")
    
    samples = []
    seen_pairs = set()
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(samples) >= max_samples:
                break
            
            line = line.strip().lower()
            words = line.split()
            
            if len(words) < context_length + 1:
                continue
            
            for i in range(len(words) - context_length):
                if len(samples) >= max_samples:
                    break
                
                context = ' '.join(words[i:i+context_length])
                target = words[i+context_length]
                
                pair_key = f"{context}|{target}"
                if (target.isalpha() and 
                    len(target) > 1 and 
                    pair_key not in seen_pairs):
                    
                    samples.append({
                        'input': context,
                        'target': target,
                        'task': 'nextword'
                    })
                    seen_pairs.add(pair_key)
    
    print(f"  Generated {len(samples):,} next-word pairs")
    return samples


def prepare_typo_data(typo_path: str, max_samples: int = 20000) -> List[Dict]:
    """Generate typo correction pairs"""
    print(f"\nGenerating typo correction data (max {max_samples:,})...")
    
    samples = []
    
    with open(typo_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(samples) >= max_samples:
                break
            
            correct = row.get('correct_word', row.get('label', '')).strip().lower()
            misspelled_list = row.get('misspelled_words', row.get('input', '')).strip().lower()
            
            typos = [t.strip() for t in misspelled_list.replace(',', ' ').split() if t.strip()]
            
            for typo in typos:
                if len(samples) >= max_samples:
                    break
                
                if typo and typo != correct and len(typo) >= 2:
                    samples.append({
                        'input': typo,
                        'target': correct,
                        'task': 'typo'
                    })
    
    print(f"  Generated {len(samples):,} typo pairs")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for custom keyboard model")
    
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing raw datasets')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save processed data')
    parser.add_argument('--max-completion', type=int, default=50000,
                        help='Max word completion samples')
    parser.add_argument('--max-nextword', type=int, default=100000,
                        help='Max next-word samples')
    parser.add_argument('--max-typo', type=int, default=20000,
                        help='Max typo correction samples')
    parser.add_argument('--val-split', type=float, default=0.05,
                        help='Validation split ratio')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Data Preparation for Custom Keyboard Model")
    print(f"{'='*60}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # Generate data
    completion_samples = prepare_word_completion_data(
        str(data_dir / "single_word_freq.csv"),
        max_samples=args.max_completion
    )
    
    nextword_samples = prepare_nextword_data(
        str(data_dir / "keyboard_training_data.txt"),
        max_samples=args.max_nextword
    )
    
    typo_samples = prepare_typo_data(
        str(data_dir / "misspelled.csv"),
        max_samples=args.max_typo
    )
    
    # Combine and shuffle
    all_samples = completion_samples + nextword_samples + typo_samples
    random.shuffle(all_samples)
    
    print(f"\n{'='*60}")
    print(f"Dataset Summary:")
    print(f"  Completion: {len(completion_samples):,}")
    print(f"  Next-word:  {len(nextword_samples):,}")
    print(f"  Typo:       {len(typo_samples):,}")
    print(f"  TOTAL:      {len(all_samples):,}")
    print(f"{'='*60}\n")
    
    # Split train/val
    split_idx = int(len(all_samples) * (1 - args.val_split))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    print(f"Train/Val Split:")
    print(f"  Train: {len(train_samples):,} ({len(train_samples)/len(all_samples)*100:.1f}%)")
    print(f"  Val:   {len(val_samples):,} ({len(val_samples)/len(all_samples)*100:.1f}%)\n")
    
    # Save to JSONL
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + '\n')
    
    with open(val_path, 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"✓ Data saved:")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print(f"\n✅ Data preparation complete!\n")


if __name__ == "__main__":
    main()
