"""
Training script for Tiny Transformer keyboard model

Simple training loop for the tiny transformer with multi-task learning.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import BertTokenizer
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'src'))
from tiny_transformer import TinyTransformer
from keyboard_dataset import create_dataloaders


def train_tiny_model(
    output_dir: str = './models/tiny_transformer',
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    train_samples: int = 10000,  # Small for quick training
    val_samples: int = 1000
):
    """
    Train the tiny transformer model.
    
    Args:
        output_dir: Directory to save model
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        train_samples: Number of training samples
        val_samples: Number of validation samples
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create model
    print("\n2. Creating model...")
    model = TinyTransformer(vocab_size=tokenizer.vocab_size)
    model = model.to(device)
    
    print(f"   Parameters: {model.count_parameters():,}")
    print(f"   Size: {model.get_model_size():.2f} MB")
    
    # Create dataloaders
    print(f"\n3. Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        tokenizer,
        train_samples=train_samples,
        val_samples=val_samples,
        batch_size=batch_size
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Training loop
    print(f"\n4. Training for {epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            
            # Compute loss
            loss = criterion(logits.view(-1, tokenizer.vocab_size), labels.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                logits = model(input_ids, attention_mask)
                loss = criterion(logits.view(-1, tokenizer.vocab_size), labels.view(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\n   Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss:   {avg_val_loss:.4f}\n")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, output_path / 'best_model.pt')
            print(f"   ✓ Saved best model (val_loss: {avg_val_loss:.4f})")
    
    # Save final model
    torch.save(model.state_dict(), output_path / 'final_model.pt')
    tokenizer.save_pretrained(output_path)
    
    print(f"\n✅ Training complete!")
    print(f"   Models saved to: {output_path}")
    print(f"   Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train_tiny_model(
        epochs=2,  # Quick training for testing
        train_samples=1000,
        val_samples=100
    )
