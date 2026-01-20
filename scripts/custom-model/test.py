#!/usr/bin/env python3
"""
Interactive Testing Script for Custom Keyboard Model
Test the trained custom transformer model interactively
"""

import sys
import argparse
import torch
from pathlib import Path

# Import from current directory
from tokenizer import KeyboardTokenizer
from model import KeyboardTransformer


class CustomModelTester:
    """Interactive tester for custom keyboard model"""
    
    def __init__(self, model_dir: str):
        """
        Initialize tester
        
        Args:
            model_dir: Directory containing trained model and tokenizer
        """
        self.model_dir = Path(model_dir)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Loading custom keyboard model from: {model_dir}")
        print("=" * 60)
        
        # Load tokenizer
        tokenizer_path = self.model_dir / "tokenizer.pkl"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        
        self.tokenizer = KeyboardTokenizer.load(str(tokenizer_path))
        
        # Load model
        model_path = self.model_dir / "best_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model with same architecture
        self.model = KeyboardTransformer(
            vocab_size=len(self.tokenizer),
            hidden_size=128,
            num_layers=6,
            num_heads=4,
            ff_dim=512,
            max_length=16,
            dropout=0.1
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Device: {self.device}")
        print(f"✓ Vocabulary: {len(self.tokenizer):,} words")
        print(f"✓ Parameters: {self.model.count_parameters():,}")
        print(f"✓ Val Loss: {checkpoint.get('val_loss', 'N/A')}")
        print(f"✓ Val Accuracy: {checkpoint.get('val_accuracy', 0)*100:.2f}%")
        print("=" * 60)
    
    def predict(self, input_text: str, top_k: int = 5):
        """
        Predict suggestions for input text
        
        Args:
            input_text: Input text
            top_k: Number of suggestions
            
        Returns:
            List of (word, probability) tuples
        """
        # Add [MASK] token
        text_with_mask = f"{input_text} {self.tokenizer.mask_token}"
        
        # Encode
        input_ids = self.tokenizer.encode(text_with_mask, max_length=16, padding=True)
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        # Create attention mask
        attention_mask = torch.tensor(
            [[1 if idx != self.tokenizer.pad_token_id else 0 for idx in input_ids]],
            dtype=torch.long
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            top_tokens, top_probs = self.model.predict(
                input_ids_tensor,
                attention_mask,
                top_k=top_k * 2  # Get more for filtering
            )
        
        # Decode and filter
        predictions = []
        for token_id, prob in zip(top_tokens[0], top_probs[0]):
            word = self.tokenizer.idx2word.get(token_id.item(), self.tokenizer.unk_token)
            
            # Filter out special tokens
            if word not in [self.tokenizer.pad_token, self.tokenizer.unk_token, self.tokenizer.mask_token]:
                predictions.append((word, prob.item() * 100))
            
            if len(predictions) >= top_k:
                break
        
        return predictions
    
    def run_interactive(self):
        """Run interactive testing loop"""
        print("\n🎹 Custom Keyboard Model - Interactive Tester")
        print("=" * 60)
        print("Instructions:")
        print("  - Type your text and press Enter to get suggestions")
        print("  - Type ':q' or 'quit' to exit")
        print("  - Type ':help' for more options")
        print("  - Type ':k10' to get 10 suggestions (default is 5)")
        print("=" * 60)
        print()
        
        current_top_k = 5
        
        while True:
            try:
                user_input = input("📝 Input: ").strip()
                
                # Handle commands
                if user_input.lower() in [':q', 'quit', 'exit']:
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == ':help':
                    print("\nCommands:")
                    print("  :q, quit, exit - Quit the program")
                    print("  :help          - Show this help message")
                    print("  :stats         - Show model statistics")
                    print("  :k5            - Get 5 suggestions")
                    print("  :k10           - Get 10 suggestions")
                    print()
                    continue
                
                if user_input.lower() == ':stats':
                    print("\nModel Statistics:")
                    print(f"  Model: Custom Keyboard Transformer")
                    print(f"  Parameters: {self.model.count_parameters():,}")
                    print(f"  Vocab size: {len(self.tokenizer):,}")
                    print(f"  Architecture: 6-layer Transformer")
                    print(f"  Hidden size: 128")
                    print(f"  Attention heads: 4")
                    print(f"  Device: {self.device}")
                    print(f"  Current top-k: {current_top_k}")
                    print()
                    continue
                
                if user_input.lower().startswith(':k'):
                    try:
                        current_top_k = int(user_input[2:])
                        print(f"✓ Top-k set to {current_top_k}")
                    except:
                        print("❌ Invalid format. Use :k5 or :k10")
                    continue
                
                if not user_input:
                    continue
                
                # Get predictions
                predictions = self.predict(user_input, top_k=current_top_k)
                
                # Display results
                print(f"\n💡 Suggestions for '{user_input}':")
                if predictions:
                    for i, (word, prob) in enumerate(predictions, 1):
                        confidence = "🟢" if prob > 50 else "🟡" if prob > 20 else "🔴"
                        print(f"  {i}. {word:15s} {confidence} {prob:5.1f}%")
                else:
                    print("  (no suggestions)")
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Test custom keyboard model")
    parser.add_argument(
        '--model-dir',
        type=str,
        default='../../models/custom_keyboard',
        help='Path to model directory'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"❌ Error: Model directory not found: {model_dir}")
        print("\n💡 Train the model first using:")
        print(f"   python train.py --data-dir ../../data/processed --save-dir {model_dir}")
        sys.exit(1)
    
    # Create tester and run
    try:
        tester = CustomModelTester(model_dir=args.model_dir)
        tester.run_interactive()
    except Exception as e:
        print(f"❌ Failed to initialize tester: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
