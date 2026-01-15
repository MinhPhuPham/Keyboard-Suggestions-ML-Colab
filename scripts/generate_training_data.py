"""
Generate training datasets for keyboard suggestion model.

This script uses the data preparation utilities to create training pairs
for all three tasks: word completion, next-word prediction, and typo correction.

Run after downloading datasets with download_datasets.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from keyboard_data_prep import (
    prepare_word_completion_data,
    prepare_nextword_data,
    prepare_typo_data,
    combine_datasets
)


def main():
    print("="*60)
    print("Generating Keyboard Suggestion Training Datasets")
    print("="*60)
    
    # Paths
    data_dir = "./data/keyboard"
    word_freq_path = os.path.join(data_dir, "word_freq.txt")
    corpus_path = os.path.join(data_dir, "corpus.txt")
    output_dir = os.path.join(data_dir, "processed")
    
    # Check if source data exists
    if not os.path.exists(word_freq_path):
        print(f"❌ Error: {word_freq_path} not found")
        print("Run: python scripts/download_datasets.py first")
        return
    
    if not os.path.exists(corpus_path):
        print(f"❌ Error: {corpus_path} not found")
        print("Run: python scripts/download_datasets.py first")
        return
    
    print("\n1. Generating word completion pairs...")
    completion_path = prepare_word_completion_data(
        word_freq_path=word_freq_path,
        output_path=os.path.join(output_dir, "completion.jsonl"),
        max_samples=10000  # Reduced for sample data
    )
    
    print("\n2. Generating next-word prediction pairs...")
    nextword_path = prepare_nextword_data(
        corpus_path=corpus_path,
        output_path=os.path.join(output_dir, "nextword.jsonl"),
        max_samples=5000,  # Reduced for sample data
        context_length=3
    )
    
    print("\n3. Generating typo correction pairs...")
    typo_path = prepare_typo_data(
        word_list_path=word_freq_path,
        output_path=os.path.join(output_dir, "typo.jsonl"),
        max_samples=3000  # Reduced for sample data
    )
    
    print("\n4. Combining and splitting datasets...")
    train_path, val_path = combine_datasets(
        completion_path=completion_path,
        nextword_path=nextword_path,
        typo_path=typo_path,
        output_path=output_dir,
        train_ratio=0.9
    )
    
    print("\n" + "="*60)
    print("✓ Dataset generation complete!")
    print("="*60)
    print("\nGenerated files:")
    print(f"  - {train_path}")
    print(f"  - {val_path}")
    print("\nNext steps:")
    print("1. Upload these files to Google Drive")
    print("2. Open train_mobilebert_keyboard.ipynb in Colab")
    print("3. Train the model!")
    print("\n⚠ NOTE: These are sample datasets for testing.")
    print("For production, use larger datasets:")
    print("  - Word completion: 500K+ pairs")
    print("  - Next-word: 1M+ pairs")
    print("  - Typo correction: 100K+ pairs")


if __name__ == "__main__":
    main()
