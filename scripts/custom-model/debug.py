#!/usr/bin/env python3
"""
Debug script to check data and model
"""

import sys
from pathlib import Path
import torch

# Import from current directory
from tokenizer import KeyboardTokenizer
from model import KeyboardTransformer
from dataset import KeyboardDataset

print("="*60)
print("Debugging Custom Model")
print("="*60)

# Load tokenizer
print("\n1. Loading tokenizer...")
tokenizer = KeyboardTokenizer.load("../../models/custom_keyboard/tokenizer.pkl")
print(f"✓ Vocab size: {len(tokenizer):,}")

# Load dataset
print("\n2. Loading dataset...")
dataset = KeyboardDataset("../../data/processed/train.jsonl", tokenizer, max_length=16)
print(f"✓ Dataset size: {len(dataset):,}")

# Check a sample
print("\n3. Checking sample data...")
sample = dataset[0]
print(f"Input IDs: {sample['input_ids']}")
print(f"Attention mask: {sample['attention_mask']}")
print(f"Labels: {sample['labels']}")
print(f"Decoded input: {tokenizer.decode(sample['input_ids'].tolist())}")

# Check for invalid values
print("\n4. Checking for NaN/Inf in data...")
has_nan = torch.isnan(sample['input_ids']).any()
has_inf = torch.isinf(sample['input_ids'].float()).any()
print(f"Has NaN: {has_nan}")
print(f"Has Inf: {has_inf}")

# Check labels
valid_labels = sample['labels'][sample['labels'] != -100]
print(f"Valid labels: {valid_labels}")
print(f"Label range: {valid_labels.min().item()} to {valid_labels.max().item()}")
print(f"Vocab size: {len(tokenizer)}")

if valid_labels.max() >= len(tokenizer):
    print("❌ ERROR: Label ID exceeds vocab size!")
else:
    print("✓ Labels are valid")

# Create model
print("\n5. Creating model...")
model = KeyboardTransformer(
    vocab_size=len(tokenizer),
    hidden_size=128,
    num_layers=6,
    num_heads=4,
    ff_dim=512,
    max_length=16,
    dropout=0.1
)
print(f"✓ Model created: {model.count_parameters():,} params")

# Test forward pass
print("\n6. Testing forward pass...")
try:
    with torch.no_grad():
        logits = model(
            sample['input_ids'].unsqueeze(0),
            sample['attention_mask'].unsqueeze(0)
        )
    print(f"✓ Forward pass successful")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits range: {logits.min().item():.4f} to {logits.max().item():.4f}")
    print(f"  Has NaN: {torch.isnan(logits).any()}")
    print(f"  Has Inf: {torch.isinf(logits).any()}")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()

# Test loss calculation
print("\n7. Testing loss calculation...")
try:
    import torch.nn.functional as F
    
    labels = sample['labels'].unsqueeze(0)
    loss = F.cross_entropy(
        logits.view(-1, len(tokenizer)),
        labels.view(-1),
        ignore_index=-100
    )
    print(f"✓ Loss calculation successful")
    print(f"  Loss: {loss.item()}")
    print(f"  Is NaN: {torch.isnan(loss)}")
    print(f"  Is Inf: {torch.isinf(loss)}")
except Exception as e:
    print(f"❌ Loss calculation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Debug complete!")
print("="*60)
