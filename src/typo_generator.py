"""
Typo Generator for Keyboard Training Data Augmentation

Generates realistic typos to train the model for typo correction.
Supports common typo patterns:
- Adjacent key errors (QWERTY layout)
- Missing characters
- Extra characters  
- Swapped characters
- Similar sounding characters
"""

import random
from typing import List, Tuple


# QWERTY keyboard layout for adjacent key errors
QWERTY_NEIGHBORS = {
    'q': ['w', 'a'], 'w': ['q', 'e', 's'], 'e': ['w', 'r', 'd'],
    'r': ['e', 't', 'f'], 't': ['r', 'y', 'g'], 'y': ['t', 'u', 'h'],
    'u': ['y', 'i', 'j'], 'i': ['u', 'o', 'k'], 'o': ['i', 'p', 'l'],
    'p': ['o', 'l'],
    'a': ['q', 's', 'z'], 's': ['w', 'a', 'd', 'x'], 'd': ['e', 's', 'f', 'c'],
    'f': ['r', 'd', 'g', 'v'], 'g': ['t', 'f', 'h', 'b'], 'h': ['y', 'g', 'j', 'n'],
    'j': ['u', 'h', 'k', 'm'], 'k': ['i', 'j', 'l'], 'l': ['o', 'k', 'p'],
    'z': ['a', 'x'], 'x': ['z', 's', 'c'], 'c': ['x', 'd', 'v'],
    'v': ['c', 'f', 'b'], 'b': ['v', 'g', 'n'], 'n': ['b', 'h', 'm'],
    'm': ['n', 'j']
}

# Common phonetic substitutions
PHONETIC_SUBSTITUTIONS = {
    'c': ['k', 's'], 'k': ['c'], 's': ['c', 'z'], 'z': ['s'],
    'f': ['ph'], 'ph': ['f'], 'i': ['y'], 'y': ['i'],
    'er': ['or'], 'or': ['er']
}


class TypoGenerator:
    """Generate realistic typos for training data augmentation."""
    
    def __init__(self, typo_prob: float = 0.15):
        """
        Args:
            typo_prob: Probability of introducing a typo (default: 15%)
        """
        self.typo_prob = typo_prob
    
    def generate_typo(self, word: str) -> str:
        """
        Generate a single typo in a word.
        
        Args:
            word: Original word
        
        Returns:
            Word with typo (or original if no typo generated)
        """
        if len(word) < 3 or random.random() > self.typo_prob:
            return word
        
        # Choose typo type
        typo_types = [
            self._adjacent_key_error,
            self._missing_char,
            self._extra_char,
            self._swapped_chars,
            self._double_char
        ]
        
        typo_func = random.choice(typo_types)
        return typo_func(word)
    
    def _adjacent_key_error(self, word: str) -> str:
        """Replace a character with an adjacent key."""
        if len(word) < 2:
            return word
        
        pos = random.randint(0, len(word) - 1)
        char = word[pos].lower()
        
        if char in QWERTY_NEIGHBORS and QWERTY_NEIGHBORS[char]:
            replacement = random.choice(QWERTY_NEIGHBORS[char])
            return word[:pos] + replacement + word[pos+1:]
        
        return word
    
    def _missing_char(self, word: str) -> str:
        """Remove a random character."""
        if len(word) <= 3:
            return word
        
        pos = random.randint(1, len(word) - 2)  # Don't remove first/last
        return word[:pos] + word[pos+1:]
    
    def _extra_char(self, word: str) -> str:
        """Add an extra character."""
        pos = random.randint(0, len(word))
        char = word[pos-1] if pos > 0 else word[0]
        
        # Add adjacent key or duplicate
        if char.lower() in QWERTY_NEIGHBORS:
            extra = random.choice(QWERTY_NEIGHBORS[char.lower()])
        else:
            extra = char
        
        return word[:pos] + extra + word[pos:]
    
    def _swapped_chars(self, word: str) -> str:
        """Swap two adjacent characters."""
        if len(word) < 2:
            return word
        
        pos = random.randint(0, len(word) - 2)
        chars = list(word)
        chars[pos], chars[pos+1] = chars[pos+1], chars[pos]
        return ''.join(chars)
    
    def _double_char(self, word: str) -> str:
        """Double a random character."""
        if len(word) < 2:
            return word
        
        pos = random.randint(0, len(word) - 1)
        return word[:pos+1] + word[pos] + word[pos+1:]
    
    def augment_text(self, text: str, typo_rate: float = 0.15) -> str:
        """
        Augment text with typos.
        
        Args:
            text: Original text
            typo_rate: Probability of typo per word
        
        Returns:
            Text with typos
        """
        words = text.split()
        augmented = []
        
        for word in words:
            if random.random() < typo_rate and len(word) >= 3:
                augmented.append(self.generate_typo(word))
            else:
                augmented.append(word)
        
        return ' '.join(augmented)
    
    def generate_typo_pairs(self, words: List[str], n_pairs: int = 1000) -> List[Tuple[str, str]]:
        """
        Generate (typo, correct) pairs for training.
        
        Args:
            words: List of correct words
            n_pairs: Number of pairs to generate
        
        Returns:
            List of (typo, correct_word) tuples
        """
        pairs = []
        
        for _ in range(n_pairs):
            word = random.choice(words)
            if len(word) >= 3:
                typo = self.generate_typo(word)
                if typo != word:  # Only add if typo was actually generated
                    pairs.append((typo, word))
        
        return pairs


# Common typo patterns for testing
COMMON_TYPOS = {
    'the': ['teh', 'hte', 'th'],
    'there': ['thers', 'ther', 'thre'],
    'their': ['thier', 'thir', 'theyr'],
    'hello': ['helo', 'helllo', 'hwllo'],
    'world': ['wrold', 'worl', 'worrld'],
    'please': ['pleas', 'pleaes', 'plese'],
    'thank': ['thnk', 'thnak', 'thankk'],
    'you': ['yuo', 'yu', 'youu']
}


if __name__ == "__main__":
    # Test typo generator
    generator = TypoGenerator(typo_prob=1.0)  # Always generate typo for testing
    
    test_words = ['hello', 'world', 'there', 'thank', 'please']
    
    print("Typo Generation Examples:")
    print("-" * 50)
    
    for word in test_words:
        typos = [generator.generate_typo(word) for _ in range(5)]
        print(f"{word:10} → {typos}")
    
    print("\nText Augmentation:")
    print("-" * 50)
    
    text = "hello world thank you for your help"
    augmented = generator.augment_text(text, typo_rate=0.5)
    print(f"Original:   {text}")
    print(f"Augmented:  {augmented}")
    
    print("\nTypo Pairs Generation:")
    print("-" * 50)
    
    pairs = generator.generate_typo_pairs(test_words, n_pairs=10)
    for typo, correct in pairs[:5]:
        print(f"{typo:15} → {correct}")
