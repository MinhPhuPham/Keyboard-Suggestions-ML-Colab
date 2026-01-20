"""
Custom Keyboard Transformer Model
A lightweight transformer designed specifically for keyboard suggestions

Features:
- Custom 10k vocabulary (most common English words)
- 6-layer transformer encoder with multi-head attention
- ~12M parameters
- Optimized for mobile deployment (<15MB)
"""

__version__ = "1.0.0"
__author__ = "MinhPhuPham"

from .model import KeyboardTransformer
from .tokenizer import KeyboardTokenizer
from .dataset import KeyboardDataset
from .trainer import KeyboardTrainer

__all__ = [
    'KeyboardTransformer',
    'KeyboardTokenizer',
    'KeyboardDataset',
    'KeyboardTrainer'
]
