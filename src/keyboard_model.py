"""
Keyboard Suggestion Model using MobileBERT with Multi-Task Learning

Simplified architecture with 3 ML tasks + heuristic gibberish detection:
1. Word Completion: "Hel" → ["Hello", "Help", "Helping"]
2. Next-Word Prediction: "How are" → ["you", "they", "we"]
3. Typo Correction: "Thers" → ["There", "Theirs", "Therapy"]
4. Gibberish Detection: Heuristic (no suggestions after 7-8 keystrokes)

Optimized for:
- Colab GPU training
- Mobile deployment (CoreML/TFLite)
- <50ms inference on mobile devices
"""

import torch
import torch.nn as nn
from transformers import MobileBertModel
from typing import Optional, Tuple, List


class KeyboardSuggestionModel(nn.Module):
    """
    Multi-task keyboard suggestion model based on MobileBERT.
    
    Features:
    - Word completion from partial input
    - Next-word prediction from context
    - Typo correction
    - Gibberish detection (heuristic, not ML)
    """
    
    def __init__(
        self,
        base_model_name: str = "google/mobilebert-uncased",
        vocab_size: int = 50000,
        dropout: float = 0.1
    ):
        """
        Initialize the keyboard suggestion model.
        
        Args:
            base_model_name: Pretrained MobileBERT model name
            vocab_size: Size of vocabulary
            dropout: Dropout probability
        """
        super().__init__()
        
        # Load pretrained MobileBERT encoder
        print(f"Loading {base_model_name}...")
        self.encoder = MobileBertModel.from_pretrained(base_model_name)
        
        # Get hidden size from config
        self.hidden_size = self.encoder.config.hidden_size
        self.vocab_size = vocab_size
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Task-specific heads (3 tasks)
        self.word_completion_head = nn.Linear(self.hidden_size, vocab_size)
        self.next_word_head = nn.Linear(self.hidden_size, vocab_size)
        
        # Typo correction head with extra layer for edit distance learning
        self.typo_correction_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, vocab_size)
        )
        
        # Initialize weights
        self._init_weights()
        
        print(f"✓ Model initialized")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _init_weights(self):
        """Initialize task head weights."""
        for module in [self.word_completion_head, self.next_word_head]:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task: str = "completion"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            task: Task type ("completion", "next_word", "typo")
        
        Returns:
            Tuple of (top_k_indices, top_k_scores)
        """
        # Get encoder output
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use pooled output (CLS token representation)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)
        
        # Route to appropriate task head
        if task == "completion":
            logits = self.word_completion_head(pooled_output)
        elif task == "next_word":
            logits = self.next_word_head(pooled_output)
        elif task == "typo":
            logits = self.typo_correction_head(pooled_output)
        else:
            raise ValueError(f"Unknown task: {task}. Use 'completion', 'next_word', or 'typo'")
        
        # Return top-3 predictions
        top_k = torch.topk(logits, k=3, dim=-1)
        return top_k.indices, top_k.values
    
    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        # Save config
        config = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "model_type": "keyboard_suggestion",
            "base_model": "google/mobilebert-uncased"
        }
        
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, load_directory: str):
        """Load model from directory."""
        import os
        import json
        
        # Load config
        with open(os.path.join(load_directory, "config.json"), "r") as f:
            config = json.load(f)
        
        # Create model
        model = cls(vocab_size=config["vocab_size"])
        
        # Load weights
        state_dict = torch.load(
            os.path.join(load_directory, "pytorch_model.bin"),
            map_location="cpu"
        )
        model.load_state_dict(state_dict)
        
        print(f"✓ Model loaded from {load_directory}")
        return model


class GibberishDetector:
    """
    Heuristic-based gibberish detection.
    No ML model needed - uses simple rules.
    """
    
    def __init__(self, threshold: int = 7):
        """
        Initialize gibberish detector.
        
        Args:
            threshold: Number of failed suggestion attempts before flagging as gibberish
        """
        self.threshold = threshold
        self.failed_attempts = 0
    
    def check(self, has_suggestions: bool) -> bool:
        """
        Check if input is likely gibberish.
        
        Args:
            has_suggestions: Whether the model found any suggestions
        
        Returns:
            True if likely gibberish, False otherwise
        """
        if not has_suggestions:
            self.failed_attempts += 1
        else:
            self.failed_attempts = 0
        
        return self.failed_attempts >= self.threshold
    
    def reset(self):
        """Reset the failed attempts counter."""
        self.failed_attempts = 0


if __name__ == "__main__":
    print("Testing KeyboardSuggestionModel...")
    print("="*60)
    
    # Test model creation (will download MobileBERT if not cached)
    try:
        model = KeyboardSuggestionModel(vocab_size=10000)  # Smaller vocab for testing
        
        # Test forward pass
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        print("\nTesting forward pass...")
        for task in ["completion", "next_word", "typo"]:
            predictions, scores = model(input_ids, attention_mask, task=task)
            print(f"✓ {task:15s}: shape={predictions.shape}, top_score={scores[0,0]:.4f}")
        
        # Test gibberish detector
        print("\nTesting gibberish detector...")
        detector = GibberishDetector(threshold=3)
        
        print("  Simulating: 'hel' (has suggestions)")
        print(f"    Is gibberish? {detector.check(has_suggestions=True)}")
        
        print("  Simulating: 'acb' (no suggestions)")
        print(f"    Is gibberish? {detector.check(has_suggestions=False)}")
        
        print("  Simulating: 'xzq' (no suggestions)")
        print(f"    Is gibberish? {detector.check(has_suggestions=False)}")
        
        print("  Simulating: 'qwe' (no suggestions)")
        print(f"    Is gibberish? {detector.check(has_suggestions=False)}")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nNote: This requires transformers library and internet connection")
        print("to download MobileBERT. Run in Colab for best results.")
