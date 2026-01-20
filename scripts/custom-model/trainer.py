"""
Trainer for Custom Keyboard Model
Handles training loop, validation, checkpointing, and metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
from pathlib import Path
from typing import Optional, Dict, List
import json
import time


class KeyboardTrainer:
    """
    Trainer for keyboard transformer model
    
    Features:
    - Automatic mixed precision training (AMP)
    - Learning rate scheduling
    - Gradient clipping
    - Checkpointing
    - Metrics tracking
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_amp: bool = True
    ):
        """
        Initialize trainer
        
        Args:
            model: KeyboardTransformer model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            learning_rate: Learning rate
            weight_decay: Weight decay for AdamW
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to train on
            use_amp: Use automatic mixed precision
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp and device == 'cuda'
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        total_steps = len(train_loader) * 100  # Assume max 100 epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=learning_rate * 0.01
        )
        
        # AMP scaler
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Metrics
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')
        
        print(f"\n✓ Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  AMP: {self.use_amp}")
        print(f"  Model parameters: {model.count_parameters():,}")
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} [Train]",
            leave=False
        )
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with AMP
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(input_ids, attention_mask)
                    loss = F.cross_entropy(
                        logits.view(-1, self.model.vocab_size),
                        labels.view(-1),
                        ignore_index=-100
                    )
            else:
                logits = self.model(input_ids, attention_mask)
                loss = F.cross_entropy(
                    logits.view(-1, self.model.vocab_size),
                    labels.view(-1),
                    ignore_index=-100
                )
            
            # Backward pass
            self.optimizer.zero_grad()
            
            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n❌ ERROR: Loss is {loss.item()}!")
                print(f"This usually means:")
                print(f"  1. Learning rate too high (try --learning-rate 5e-5)")
                print(f"  2. Bad data (check for inf/nan in inputs)")
                print(f"  3. Gradient explosion (gradient clipping should prevent this)")
                print(f"\n💡 Stopping training. Please restart with lower learning rate.")
                raise ValueError(f"Loss became {loss.item()}")
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> tuple:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_predictions = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                
                # Calculate loss
                loss = F.cross_entropy(
                    logits.view(-1, self.model.vocab_size),
                    labels.view(-1),
                    ignore_index=-100
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                # Calculate accuracy (only at non-ignored positions)
                predictions = torch.argmax(logits, dim=-1)
                mask = labels != -100
                
                correct = (predictions == labels) & mask
                total_correct += correct.sum().item()
                total_predictions += mask.sum().item()
        
        avg_loss = total_loss / num_batches
        accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0
        
        return avg_loss, accuracy
    
    def train(
        self,
        num_epochs: int,
        save_dir: str = "./models/custom_keyboard",
        save_every: int = 5,
        early_stopping_patience: Optional[int] = None
    ):
        """
        Train the model
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Stop if no improvement for N epochs (None = no early stopping)
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}")
        print(f"Epochs: {num_epochs}")
        print(f"Save directory: {save_dir}")
        print(f"{'='*60}\n")
        
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_accuracy = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            self.history['learning_rate'].append(self.scheduler.get_last_lr()[0])
            
            # Print metrics
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch}/{num_epochs} ({epoch_time:.1f}s):")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val Acc:    {val_accuracy*100:.2f}%")
            print(f"  LR:         {self.scheduler.get_last_lr()[0]:.2e}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(save_dir / "best_model.pt", epoch, val_loss, val_accuracy)
                print(f"  ✓ Best model saved!")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save periodic checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(save_dir / f"checkpoint_epoch_{epoch}.pt", epoch, val_loss, val_accuracy)
                print(f"  ✓ Checkpoint saved!")
            
            # Early stopping
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                print(f"\n⚠️  Early stopping triggered (no improvement for {early_stopping_patience} epochs)")
                break
        
        # Training complete
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✓ Training Complete!")
        print(f"{'='*60}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Final val accuracy: {val_accuracy*100:.2f}%")
        print(f"{'='*60}\n")
        
        # Save training history
        self.save_history(save_dir / "training_history.json")
    
    def save_checkpoint(self, path: Path, epoch: int, val_loss: float, val_accuracy: float):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'history': self.history
        }, path)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        print(f"✓ Checkpoint loaded from {path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
        print(f"  Val Accuracy: {checkpoint['val_accuracy']*100:.2f}%")
    
    def save_history(self, path: Path):
        """Save training history to JSON"""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Training history saved to {path}")
