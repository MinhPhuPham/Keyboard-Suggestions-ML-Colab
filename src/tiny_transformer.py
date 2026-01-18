"""
Tiny Transformer Model for Keyboard Suggestions

A lightweight 2-layer transformer optimized for mobile deployment.
Supports: word completion, next-word prediction, typo correction.

Target specs:
- Size: <50 MB (INT8)
- RAM: <10 MB
- Latency: <50 ms
- Parameters: ~6-8M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """Add positional encoding to input embeddings."""
        return x + self.pe[:, :x.size(1)]


class TinyTransformer(nn.Module):
    """
    Tiny Transformer for keyboard suggestions.
    
    Architecture:
    - 2 transformer encoder layers
    - 256 hidden dimensions
    - 4 attention heads
    - ~6-8M parameters
    
    Features:
    1. Word completion: "Hel" → ["Hello", "Help"]
    2. Next-word prediction: "How are" → ["you", "they"]
    3. Typo correction: "Thers" → ["There", "Theirs"]
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_length: int = 32
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Embedding
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        
        # Output layer
        nn.init.normal_(self.fc_out.weight, mean=0, std=0.02)
        nn.init.zeros_(self.fc_out.bias)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass.
        
        Args:
            input_ids: (batch_size, seq_len) - Token IDs
            attention_mask: (batch_size, seq_len) - Attention mask (1=attend, 0=ignore)
        
        Returns:
            logits: (batch_size, seq_len, vocab_size) - Prediction logits
        """
        # Embed tokens
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # Create attention mask for transformer
        # Convert from (batch, seq) to (batch, seq, seq)
        if attention_mask is not None:
            # Transformer expects: True = ignore, False = attend
            # Our mask: 1 = attend, 0 = ignore
            # So we need to invert and expand
            mask = (attention_mask == 0)  # Invert
            # No need to expand - TransformerEncoder handles it
        else:
            mask = None
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Project to vocabulary
        logits = self.fc_out(x)
        
        return logits
    
    def predict_masked(self, input_ids, mask_positions, attention_mask=None, top_k=3):
        """
        Predict tokens at masked positions.
        
        Args:
            input_ids: (batch_size, seq_len)
            mask_positions: List of positions to predict
            attention_mask: (batch_size, seq_len)
            top_k: Number of top predictions to return
        
        Returns:
            predictions: List of (token_id, probability) tuples
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            
            # Get predictions at mask positions
            predictions = []
            for pos in mask_positions:
                pos_logits = logits[0, pos]  # Assuming batch_size=1
                probs = F.softmax(pos_logits, dim=-1)
                
                # Get top-k predictions
                top_probs, top_indices = torch.topk(probs, top_k)
                
                for prob, idx in zip(top_probs, top_indices):
                    predictions.append((idx.item(), prob.item()))
            
            return predictions
    
    def get_model_size(self):
        """Calculate model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        size_mb = (param_size + buffer_size) / (1024 * 1024)
        return size_mb
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_tiny_transformer(vocab_size=30000):
    """
    Create a tiny transformer model for keyboard suggestions.
    
    Returns:
        model: TinyTransformer instance
        param_count: Number of parameters
        size_mb: Estimated size in MB (FP32)
    """
    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=256,
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        max_seq_length=32
    )
    
    param_count = model.count_parameters()
    size_mb = model.get_model_size()
    
    print(f"✓ Tiny Transformer created")
    print(f"  Parameters: {param_count:,}")
    print(f"  Size (FP32): {size_mb:.2f} MB")
    print(f"  Expected (INT8): {size_mb/4:.2f} MB")
    
    return model, param_count, size_mb


if __name__ == "__main__":
    # Test model creation
    model, params, size = create_tiny_transformer()
    
    # Test forward pass
    batch_size = 2
    seq_len = 32
    
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    logits = model(input_ids, attention_mask)
    
    print(f"\n✓ Forward pass successful")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected: ({batch_size}, {seq_len}, 30000)")
