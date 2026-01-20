"""
Dataset class for keyboard model training
Handles word completion, next-word prediction, and typo correction
"""

import torch
from torch.utils.data import Dataset
import json
from typing import List, Dict, Optional
from pathlib import Path


class KeyboardDataset(Dataset):
    """
    Dataset for keyboard suggestion training
    
    Supports three tasks:
    1. Word completion: "hel" → "hello"
    2. Next-word prediction: "how are" → "you"
    3. Typo correction: "thers" → "there"
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 16,
        task_filter: Optional[str] = None
    ):
        """
        Initialize dataset
        
        Args:
            data_path: Path to JSONL file with training data
            tokenizer: KeyboardTokenizer instance
            max_length: Maximum sequence length
            task_filter: Filter by task type ('completion', 'nextword', 'typo', or None for all)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_filter = task_filter
        
        # Load data
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                
                # Filter by task if specified
                if task_filter is None or item.get('task') == task_filter:
                    self.data.append(item)
        
        print(f"✓ Loaded {len(self.data):,} samples from {data_path}")
        if task_filter:
            print(f"  Filtered to task: {task_filter}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample
        
        Returns:
            Dictionary with:
            - input_ids: Token IDs [max_length]
            - attention_mask: Attention mask [max_length]
            - labels: Target labels [max_length]
            - task: Task type (string)
        """
        item = self.data[idx]
        input_text = item['input']
        target_word = item['target']
        task = item.get('task', 'completion')
        
        # Encode input WITHOUT [MASK] first
        input_ids = self.tokenizer.encode(
            input_text,
            max_length=self.max_length - 1,  # Leave room for [MASK]
            padding=False  # Don't pad yet
        )
        
        # Manually add [MASK] token
        input_ids.append(self.tokenizer.mask_token_id)
        
        # Now pad to max_length
        while len(input_ids) < self.max_length:
            input_ids.append(self.tokenizer.pad_token_id)
        
        # Truncate if too long
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [
            1 if token_id != self.tokenizer.pad_token_id else 0
            for token_id in input_ids
        ]
        
        # Create labels (only predict at [MASK] position)
        labels = [-100] * len(input_ids)  # -100 = ignore in loss
        
        # Find [MASK] position (should be right after input)
        try:
            mask_pos = input_ids.index(self.tokenizer.mask_token_id)
            
            # Get target token ID
            target_id = self.tokenizer.word2idx.get(
                target_word.lower(),
                self.tokenizer.unk_token_id
            )
            
            # Set label at mask position
            labels[mask_pos] = target_id
            
        except ValueError:
            # [MASK] not found (shouldn't happen now, but handle gracefully)
            print(f"Warning: [MASK] not found in input_ids for sample {idx}")
            pass
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'task': task
        }
    
    def get_task_distribution(self) -> Dict[str, int]:
        """Get distribution of tasks in dataset"""
        from collections import Counter
        tasks = [item.get('task', 'unknown') for item in self.data]
        return dict(Counter(tasks))


def create_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 16,
    num_workers: int = 0
):
    """
    Create train and validation dataloaders
    
    Args:
        train_path: Path to training JSONL
        val_path: Path to validation JSONL
        tokenizer: KeyboardTokenizer instance
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of data loading workers
        
    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = KeyboardDataset(train_path, tokenizer, max_length)
    val_dataset = KeyboardDataset(val_path, tokenizer, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\n✓ Dataloaders created:")
    print(f"  Train: {len(train_dataset):,} samples, {len(train_loader):,} batches")
    print(f"  Val: {len(val_dataset):,} samples, {len(val_loader):,} batches")
    print(f"  Batch size: {batch_size}")
    
    # Show task distribution
    print(f"\nTask distribution (train):")
    for task, count in train_dataset.get_task_distribution().items():
        pct = count / len(train_dataset) * 100
        print(f"  {task}: {count:,} ({pct:.1f}%)")
    
    return train_loader, val_loader
