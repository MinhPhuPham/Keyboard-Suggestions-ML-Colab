"""
Keyboard Dataset Pipeline

Handles data loading and preprocessing for training the tiny transformer.
Supports multiple tasks:
1. Next-word prediction
2. Word completion
3. Typo correction

Uses OSCAR English dataset (streaming mode for memory efficiency).
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import List, Tuple, Dict
import random
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))
from typo_generator import TypoGenerator


class KeyboardDataset(Dataset):
    """
    Dataset for keyboard suggestion training.
    
    Supports three tasks:
    1. Next-word prediction: "How are" → "you"
    2. Word completion: "Hel" → "Hello"
    3. Typo correction: "Thers" → "There"
    """
    
    def __init__(
        self,
        tokenizer,
        max_length: int = 32,
        task_weights: Dict[str, float] = None,
        typo_prob: float = 0.15,
        completion_prob: float = 0.2
    ):
        """
        Args:
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            task_weights: Weights for each task (next_word, completion, typo)
            typo_prob: Probability of typo injection
            completion_prob: Probability of word completion task
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.typo_generator = TypoGenerator(typo_prob=typo_prob)
        self.completion_prob = completion_prob
        
        # Default task weights
        self.task_weights = task_weights or {
            'next_word': 0.5,
            'completion': 0.3,
            'typo': 0.2
        }
        
        # Data will be loaded in batches
        self.data = []
    
    def load_from_oscar(self, num_samples: int = 100000):
        """
        Load data from OSCAR English dataset.
        
        Args:
            num_samples: Number of samples to load
        """
        print(f"Loading {num_samples:,} samples from OSCAR...")
        
        try:
            # Load OSCAR dataset in streaming mode
            dataset = load_dataset(
                "oscar-corpus/OSCAR-2301",
                "en",
                split="train",
                streaming=True,
                trust_remote_code=True
            )
            
            count = 0
            for example in dataset:
                if count >= num_samples:
                    break
                
                text = example['text'].strip()
                if len(text) > 20:  # Skip very short texts
                    # Split into sentences
                    sentences = text.split('.')
                    for sent in sentences:
                        sent = sent.strip()
                        if 10 < len(sent) < 200:  # Reasonable length
                            self.data.append(sent)
                            count += 1
                            if count >= num_samples:
                                break
                
                if count % 10000 == 0:
                    print(f"  Loaded {count:,} samples...")
            
            print(f"✓ Loaded {len(self.data):,} samples from OSCAR")
            
        except Exception as e:
            print(f"⚠️  Failed to load OSCAR: {e}")
            print("  Using fallback sample data...")
            self._load_fallback_data()
    
    def _load_fallback_data(self):
        """Load fallback data if OSCAR fails."""
        # Common English phrases for testing
        fallback = [
            "Hello how are you today",
            "Thank you for your help",
            "Please let me know if you need anything",
            "I am looking forward to hearing from you",
            "Have a great day",
            "See you later",
            "Good morning everyone",
            "What time is the meeting",
            "Can you please send me the document",
            "I will be there in five minutes"
        ] * 100  # Repeat to have enough data
        
        self.data = fallback
        print(f"✓ Loaded {len(self.data)} fallback samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a training sample.
        
        Returns:
            Dict with input_ids, attention_mask, labels, task_type
        """
        text = self.data[idx]
        
        # Randomly select task
        task = random.choices(
            list(self.task_weights.keys()),
            weights=list(self.task_weights.values())
        )[0]
        
        if task == 'next_word':
            return self._create_next_word_sample(text)
        elif task == 'completion':
            return self._create_completion_sample(text)
        else:  # typo
            return self._create_typo_sample(text)
    
    def _create_next_word_sample(self, text: str):
        """Create next-word prediction sample."""
        words = text.split()
        if len(words) < 3:
            # Fallback to first task type
            return self._create_completion_sample(text)
        
        # Random position to mask
        mask_pos = random.randint(1, len(words) - 1)
        
        # Create input with [MASK]
        input_words = words[:mask_pos] + ['[MASK]'] + words[mask_pos+1:]
        input_text = ' '.join(input_words[:self.max_length])
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create labels (only predict masked token)
        labels = encoding['input_ids'].clone()
        labels[labels != self.tokenizer.mask_token_id] = -100  # Ignore non-masked
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
            'task_type': 'next_word'
        }
    
    def _create_completion_sample(self, text: str):
        """Create word completion sample."""
        words = text.split()
        if len(words) < 2:
            return self._create_next_word_sample(text)
        
        # Random word to complete
        complete_pos = random.randint(0, len(words) - 1)
        word = words[complete_pos]
        
        if len(word) < 3:
            return self._create_next_word_sample(text)
        
        # Truncate word randomly
        truncate_len = random.randint(2, len(word) - 1)
        partial = word[:truncate_len]
        
        # Replace with partial + [MASK]
        words[complete_pos] = partial + ' [MASK]'
        input_text = ' '.join(words[:self.max_length])
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = encoding['input_ids'].clone()
        labels[labels != self.tokenizer.mask_token_id] = -100
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
            'task_type': 'completion'
        }
    
    def _create_typo_sample(self, text: str):
        """Create typo correction sample."""
        # Inject typos
        typo_text = self.typo_generator.augment_text(text, typo_rate=0.3)
        
        # Add [MASK] at a random position
        words = typo_text.split()
        if len(words) < 2:
            return self._create_next_word_sample(text)
        
        mask_pos = random.randint(0, len(words) - 1)
        words[mask_pos] = '[MASK]'
        input_text = ' '.join(words[:self.max_length])
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = encoding['input_ids'].clone()
        labels[labels != self.tokenizer.mask_token_id] = -100
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
            'task_type': 'typo'
        }


def create_dataloaders(
    tokenizer,
    train_samples: int = 80000,
    val_samples: int = 10000,
    batch_size: int = 32,
    num_workers: int = 4
):
    """
    Create train and validation dataloaders.
    
    Args:
        tokenizer: Tokenizer instance
        train_samples: Number of training samples
        val_samples: Number of validation samples
        batch_size: Batch size
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = KeyboardDataset(tokenizer)
    train_dataset.load_from_oscar(train_samples)
    
    val_dataset = KeyboardDataset(tokenizer)
    val_dataset.load_from_oscar(val_samples)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    from transformers import BertTokenizer
    
    print("Testing Keyboard Dataset...")
    print("-" * 60)
    
    # Create tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create dataset
    dataset = KeyboardDataset(tokenizer)
    dataset.load_from_oscar(num_samples=100)
    
    # Test samples
    print(f"\nDataset size: {len(dataset)}")
    print("\nSample examples:")
    
    for i in range(3):
        sample = dataset[i]
        print(f"\nSample {i+1}:")
        print(f"  Task: {sample['task_type']}")
        print(f"  Input shape: {sample['input_ids'].shape}")
        print(f"  Attention mask shape: {sample['attention_mask'].shape}")
        print(f"  Labels shape: {sample['labels'].shape}")
