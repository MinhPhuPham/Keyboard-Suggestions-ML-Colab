"""
Gibberish Detector - Heuristic-based (No ML)

Fast pre-filter to detect gibberish before running ML model.
Uses linguistic heuristics:
- Vowel ratio
- Repeated characters
- Character sequence probability
- Dictionary lookup
"""

import re
from typing import Set


class GibberishDetector:
    """Heuristic-based gibberish detection."""
    
    def __init__(self, vocab_file: str = None):
        """
        Args:
            vocab_file: Path to vocabulary file (optional)
        """
        self.vowels = set('aeiouAEIOU')
        self.consonants = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')
        
        # Load vocabulary if provided
        self.vocab: Set[str] = set()
        if vocab_file:
            self.load_vocab(vocab_file)
    
    def load_vocab(self, vocab_file: str):
        """Load vocabulary from file."""
        try:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                self.vocab = {line.strip().lower() for line in f if line.strip()}
            print(f"✓ Loaded {len(self.vocab):,} words from vocabulary")
        except Exception as e:
            print(f"⚠️  Failed to load vocabulary: {e}")
    
    def is_gibberish(self, text: str) -> bool:
        """
        Check if text is gibberish using heuristics.
        
        Args:
            text: Input text to check
        
        Returns:
            True if gibberish, False otherwise
        """
        if not text or len(text) < 2:
            return True
        
        text = text.strip().lower()
        
        # Check 1: Too many non-alphabetic characters
        if self._has_too_many_non_alpha(text):
            return True
        
        # Check 2: Vowel ratio (should be 20-60%)
        if not self._has_valid_vowel_ratio(text):
            return True
        
        # Check 3: Too many repeated characters
        if self._has_excessive_repeats(text):
            return True
        
        # Check 4: Dictionary lookup (if vocab loaded)
        if self.vocab and not self._in_vocabulary(text):
            # Not in vocab, but might be a valid partial word
            # Check if it's a prefix of any vocab word
            if not self._is_valid_prefix(text):
                return True
        
        # Check 5: Consonant clusters (no more than 3 in a row)
        if self._has_invalid_consonant_clusters(text):
            return True
        
        return False
    
    def _has_too_many_non_alpha(self, text: str) -> bool:
        """Check if text has too many non-alphabetic characters."""
        alpha_count = sum(1 for c in text if c.isalpha())
        if len(text) == 0:
            return True
        alpha_ratio = alpha_count / len(text)
        return alpha_ratio < 0.5  # At least 50% should be letters
    
    def _has_valid_vowel_ratio(self, text: str) -> bool:
        """Check if vowel ratio is reasonable (20-60%)."""
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return False
        
        vowel_count = sum(1 for c in letters if c in self.vowels)
        vowel_ratio = vowel_count / len(letters)
        
        return 0.2 <= vowel_ratio <= 0.6
    
    def _has_excessive_repeats(self, text: str) -> bool:
        """Check for excessive character repetition."""
        # No more than 3 of the same character in a row
        pattern = r'(.)\1{3,}'
        return bool(re.search(pattern, text))
    
    def _in_vocabulary(self, text: str) -> bool:
        """Check if text is in vocabulary."""
        return text.lower() in self.vocab
    
    def _is_valid_prefix(self, text: str) -> bool:
        """Check if text is a valid prefix of any vocabulary word."""
        if not self.vocab:
            return True  # Can't check without vocab
        
        text_lower = text.lower()
        # Check if any vocab word starts with this text
        return any(word.startswith(text_lower) for word in self.vocab)
    
    def _has_invalid_consonant_clusters(self, text: str) -> bool:
        """Check for invalid consonant clusters."""
        # No more than 4 consonants in a row (except for some valid cases)
        consonant_run = 0
        max_run = 0
        
        for char in text:
            if char.lower() in self.consonants:
                consonant_run += 1
                max_run = max(max_run, consonant_run)
            else:
                consonant_run = 0
        
        return max_run > 4
    
    def filter_gibberish(self, words: list) -> list:
        """
        Filter out gibberish from a list of words.
        
        Args:
            words: List of words to filter
        
        Returns:
            List of non-gibberish words
        """
        return [word for word in words if not self.is_gibberish(word)]
    
    def get_confidence(self, text: str) -> float:
        """
        Get confidence score that text is NOT gibberish.
        
        Args:
            text: Input text
        
        Returns:
            Confidence score (0.0 = definitely gibberish, 1.0 = definitely valid)
        """
        if not text:
            return 0.0
        
        score = 1.0
        text = text.strip().lower()
        
        # Penalty for non-alpha characters
        alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
        score *= alpha_ratio
        
        # Penalty for bad vowel ratio
        letters = [c for c in text if c.isalpha()]
        if letters:
            vowel_ratio = sum(1 for c in letters if c in self.vowels) / len(letters)
            if vowel_ratio < 0.2 or vowel_ratio > 0.6:
                score *= 0.5
        
        # Penalty for repeats
        if self._has_excessive_repeats(text):
            score *= 0.3
        
        # Bonus for being in vocabulary
        if self.vocab and self._in_vocabulary(text):
            score = 1.0
        elif self.vocab and self._is_valid_prefix(text):
            score *= 0.9
        
        return max(0.0, min(1.0, score))


if __name__ == "__main__":
    # Test gibberish detector
    detector = GibberishDetector()
    
    test_cases = [
        # (text, expected_gibberish)
        ("hello", False),
        ("helo", False),  # Typo, but not gibberish
        ("asdfgh", True),  # Random keys
        ("zzzzz", True),  # Too many repeats
        ("bcdfg", True),  # No vowels
        ("aeiou", True),  # Only vowels
        ("th", False),  # Short but valid
        ("xyz123", True),  # Too many non-alpha
        ("world", False),
        ("wrld", False),  # Missing vowel but still recognizable
        ("qwerty", False),  # Valid word
        ("zxcvbn", True),  # Gibberish
    ]
    
    print("Gibberish Detection Tests:")
    print("-" * 60)
    print(f"{'Text':<15} {'Expected':<12} {'Detected':<12} {'Confidence':<12} {'Result'}")
    print("-" * 60)
    
    correct = 0
    for text, expected in test_cases:
        detected = detector.is_gibberish(text)
        confidence = detector.get_confidence(text)
        result = "✓" if detected == expected else "✗"
        
        if detected == expected:
            correct += 1
        
        print(f"{text:<15} {str(expected):<12} {str(detected):<12} {confidence:<12.2f} {result}")
    
    print("-" * 60)
    print(f"Accuracy: {correct}/{len(test_cases)} ({100*correct/len(test_cases):.1f}%)")
