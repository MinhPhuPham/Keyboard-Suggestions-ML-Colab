"""
Data Preparation Utilities for Keyboard Suggestion Models

This module provides reusable functions for downloading, cleaning, and preparing
training data for both English and Japanese keyboard suggestion models.
"""

import os
import re
import urllib.request
import zipfile
from typing import List, Tuple
import sentencepiece as spm
from datasets import load_dataset


def download_swiftkey_corpus(output_dir: str = "./data/english") -> str:
    """
    Download and extract the SwiftKey corpus for English training.
    
    Args:
        output_dir: Directory to save the extracted corpus
        
    Returns:
        Path to the extracted corpus directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Note: You'll need to provide the actual URL or download manually
    # from Coursera SwiftKey dataset
    print("Please download the SwiftKey corpus manually from:")
    print("https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip")
    print(f"Extract to: {output_dir}")
    
    return output_dir


def clean_english_text(text: str) -> str:
    """
    Clean English text for training.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def augment_with_emojis(sentences: List[str], emoji_ratio: float = 0.1) -> List[str]:
    """
    Augment sentences with emojis.
    
    Args:
        sentences: List of text sentences
        emoji_ratio: Proportion of sentences to augment (0.0 to 1.0)
        
    Returns:
        Augmented sentences with emojis added
    """
    common_emojis = ['😊', '😂', '❤️', '👍', '🎉', '😍', '🔥', '✨', '💯', '🙏']
    
    augmented = []
    for i, sentence in enumerate(sentences):
        if i % int(1 / emoji_ratio) == 0:
            # Add emoji to this sentence
            emoji = common_emojis[i % len(common_emojis)]
            sentence = f"{sentence} {emoji}"
        augmented.append(sentence)
    
    return augmented


def prepare_japanese_data(output_dir: str = "./data/japanese", 
                         sample_size: str = "10%") -> str:
    """
    Download and prepare Japanese CC100 dataset.
    
    Args:
        output_dir: Directory to save processed data
        sample_size: Percentage of dataset to use (e.g., "10%")
        
    Returns:
        Path to the prepared data file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading CC100 Japanese dataset ({sample_size})...")
    dataset = load_dataset('cc100', lang='ja', split=f'train[:{sample_size}]', streaming=True)
    
    output_file = os.path.join(output_dir, "train.jsonl")
    
    print(f"Saving to {output_file}...")
    # Process and save dataset
    # Note: Actual implementation would iterate through streaming dataset
    
    return output_file


def clean_japanese_text(text: str) -> str:
    """
    Clean Japanese text for training.
    
    Args:
        text: Raw Japanese text string
        
    Returns:
        Cleaned text string
    """
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace (but preserve Japanese spacing)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Note: Do NOT lowercase Japanese text
    
    return text


def tokenize_with_sentencepiece(input_file: str, 
                                vocab_size: int = 20000,
                                model_prefix: str = "tokenizer") -> str:
    """
    Train a SentencePiece BPE tokenizer.
    
    Args:
        input_file: Path to training text file
        vocab_size: Size of vocabulary
        model_prefix: Prefix for output model files
        
    Returns:
        Path to the trained tokenizer model
    """
    print(f"Training SentencePiece tokenizer with vocab_size={vocab_size}...")
    
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=0.9995,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3
    )
    
    model_path = f"{model_prefix}.model"
    print(f"Tokenizer saved to {model_path}")
    
    return model_path


def split_dataset(input_file: str, 
                 output_dir: str,
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.1) -> Tuple[str, str, str]:
    """
    Split dataset into train/validation/test sets.
    
    Args:
        input_file: Path to input data file
        output_dir: Directory to save split files
        train_ratio: Proportion for training (default 0.8)
        val_ratio: Proportion for validation (default 0.1)
        test_ratio: Proportion for testing (default 0.1)
        
    Returns:
        Tuple of (train_path, val_path, test_path)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Read all lines
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total = len(lines)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    # Split data
    train_data = lines[:train_size]
    val_data = lines[train_size:train_size + val_size]
    test_data = lines[train_size + val_size:]
    
    # Save splits
    train_path = os.path.join(output_dir, "train.txt")
    val_path = os.path.join(output_dir, "val.txt")
    test_path = os.path.join(output_dir, "test.txt")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        f.writelines(train_data)
    
    with open(val_path, 'w', encoding='utf-8') as f:
        f.writelines(val_data)
    
    with open(test_path, 'w', encoding='utf-8') as f:
        f.writelines(test_data)
    
    print(f"Dataset split complete:")
    print(f"  Train: {len(train_data)} samples -> {train_path}")
    print(f"  Val: {len(val_data)} samples -> {val_path}")
    print(f"  Test: {len(test_data)} samples -> {test_path}")
    
    return train_path, val_path, test_path


def remove_duplicates(input_file: str, output_file: str) -> int:
    """
    Remove duplicate lines from a text file.
    
    Args:
        input_file: Path to input file
        output_file: Path to output file
        
    Returns:
        Number of duplicates removed
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    original_count = len(lines)
    unique_lines = list(set(lines))
    duplicates_removed = original_count - len(unique_lines)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(unique_lines)
    
    print(f"Removed {duplicates_removed} duplicates ({original_count} -> {len(unique_lines)})")
    
    return duplicates_removed


if __name__ == "__main__":
    # Example usage
    print("Data Preparation Utilities")
    print("Import this module in your notebooks or scripts")
    print("\nExample:")
    print("  from src.data_prep import clean_english_text, split_dataset")
