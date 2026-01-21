"""
Custom Keyboard Transformer Model
Lightweight transformer architecture optimized for keyboard suggestions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class KeyboardTransformer(nn.Module):
    """
    Custom lightweight transformer for keyboard suggestions
    
    Architecture:
    - Token + Position embeddings
    - 6-layer Transformer Encoder
    - Multi-head self-attention (4 heads)
    - Feed-forward networks
    - Output projection to vocabulary
    
    Parameters: ~12M
    Size: ~10-12MB (FP32), ~3-4MB (INT8)
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        hidden_size: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        ff_dim: int = 512,
        max_length: int = 16,
        dropout: float = 0.1
    ):
        """
        Initialize the transformer model
        
        Args:
            vocab_size: Size of vocabulary
            hidden_size: Hidden dimension size
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            max_length: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_length = max_length
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',  # GELU activation (better than ReLU)
            batch_first=True,
            norm_first=True  # Pre-norm (more stable training)
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_size),
            enable_nested_tensor=False  # Explicitly disable to suppress warning
        )
        
        # Output projection
        self.output = nn.Linear(hidden_size, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Create position IDs
        position_ids = torch.arange(
            seq_len,
            dtype=torch.long,
            device=input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        embeddings = self.dropout(token_embeds + position_embeds)
        
        # Create attention mask for transformer
        # PyTorch transformer expects: True = ignore, False = attend
        if attention_mask is not None:
            mask = (attention_mask == 0)
        else:
            mask = None
        
        # Pass through transformer
        hidden_states = self.transformer(
            embeddings,
            src_key_padding_mask=mask
        )
        
        # Project to vocabulary
        logits = self.output(hidden_states)
        
        return logits
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mask_position: Optional[int] = None,
        top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict top-k tokens at mask position
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            mask_position: Position to predict (if None, uses last position)
            top_k: Number of top predictions to return
            
        Returns:
            top_tokens: Top-k token IDs [batch_size, top_k]
            top_probs: Top-k probabilities [batch_size, top_k]
        """
        self.eval()
        
        with torch.no_grad():
            # Get logits
            logits = self.forward(input_ids, attention_mask)
            
            # Get predictions at mask position
            if mask_position is None:
                # Use last non-padding position
                if attention_mask is not None:
                    mask_position = attention_mask.sum(dim=1) - 1
                else:
                    mask_position = input_ids.size(1) - 1
            
            # Get logits at mask position
            if isinstance(mask_position, int):
                mask_logits = logits[:, mask_position, :]
            else:
                # Handle batch of positions
                batch_indices = torch.arange(logits.size(0), device=logits.device)
                mask_logits = logits[batch_indices, mask_position, :]
            
            # Get probabilities
            probs = F.softmax(mask_logits, dim=-1)
            
            # Get top-k
            top_probs, top_tokens = torch.topk(probs, k=top_k, dim=-1)
            
            return top_tokens, top_probs
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        """Estimate model size in MB (FP32)"""
        param_size = sum(p.numel() * 4 for p in self.parameters()) / (1024 ** 2)
        return param_size
    
    def __repr__(self) -> str:
        params = self.count_parameters()
        size_mb = self.get_model_size_mb()
        return (
            f"KeyboardTransformer(\n"
            f"  vocab_size={self.vocab_size:,},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  num_layers={self.num_layers},\n"
            f"  num_heads={self.num_heads},\n"
            f"  parameters={params:,},\n"
            f"  size={size_mb:.2f}MB\n"
            f")"
        )
