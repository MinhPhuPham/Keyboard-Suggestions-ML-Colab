#!/usr/bin/env python3
"""
Train Custom Keyboard Model
Main training script for the custom transformer model
"""

import sys
import argparse
from pathlib import Path

# Import from current directory
from tokenizer import KeyboardTokenizer
from model import KeyboardTransformer
from dataset import KeyboardDataset, create_dataloaders
from trainer import KeyboardTrainer

import torch


def main():
    parser = argparse.ArgumentParser(description="Train custom keyboard model")
    
    # Data paths
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing datasets')
    parser.add_argument('--save-dir', type=str, default='./models/custom_keyboard',
                        help='Directory to save model')
    
    # Model hyperparameters
    parser.add_argument('--vocab-size', type=int, default=20000,
                        help='Maximum vocabulary size (dynamic from training data)')
    parser.add_argument('--hidden-size', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--num-layers', type=int, default=6,
                        help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--ff-dim', type=int, default=512,
                        help='Feed-forward dimension')
    parser.add_argument('--max-length', type=int, default=16,
                        help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    
    # Training hyperparameters
    parser.add_argument('--num-epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4, safer than 3e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--early-stopping', type=int, default=None,
                        help='Early stopping patience (None = disabled)')
    
    # Other options
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/cpu/auto)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"\n{'='*60}")
    print(f"Custom Keyboard Model Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Save directory: {args.save_dir}")
    print(f"{'='*60}\n")
    
    # Build tokenizer from training data (dynamic vocabulary for 100% coverage)
    print("Step 1: Building tokenizer...")
    tokenizer = KeyboardTokenizer(vocab_size=args.vocab_size)
    
    # Use dynamic vocabulary from training data
    train_path = Path(args.data_dir) / "train.jsonl"
    
    if not train_path.exists():
        print(f"❌ Error: Training data not found: {train_path}")
        print(f"\n💡 Run prepare_data.py first to generate training data")
        sys.exit(1)
    
    tokenizer.build_vocab_from_training_data(
        str(train_path),
        max_vocab_size=args.vocab_size
    )
    
    # Save tokenizer
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(save_dir / "tokenizer.pkl")
    
    # Create model
    print("\nStep 2: Creating model...")
    model = KeyboardTransformer(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        max_length=args.max_length,
        dropout=args.dropout
    )
    print(model)
    
    # Create dataloaders
    print("\nStep 3: Loading data...")
    train_path = Path(args.data_dir) / "train.jsonl"
    val_path = Path(args.data_dir) / "val.jsonl"
    
    if not train_path.exists() or not val_path.exists():
        print(f"❌ Error: Training data not found!")
        print(f"  Expected: {train_path}")
        print(f"  Expected: {val_path}")
        print(f"\n💡 Run data preparation first to generate train/val splits")
        sys.exit(1)
    
    train_loader, val_loader = create_dataloaders(
        str(train_path),
        str(val_path),
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # Create trainer
    print("\nStep 4: Initializing trainer...")
    trainer = KeyboardTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        device=device
    )
    
    # Train
    print("\nStep 5: Training...")
    trainer.train(
        num_epochs=args.num_epochs,
        save_dir=args.save_dir,
        save_every=5,
        early_stopping_patience=args.early_stopping
    )
    
    print("\n✅ Training complete!")
    print(f"Model saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
