"""
Local Test Script for Model Training Workflow

This script tests the training workflow locally before running in Colab.
It validates imports, data preparation, and model utilities without GPU training.
"""

import sys
import os

# Add src to path
sys.path.append('./src')

def test_imports():
    """Test that all required modules can be imported."""
    print("="*60)
    print("TEST 1: Importing Required Modules")
    print("="*60)
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("✓ Transformers")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        from datasets import Dataset
        print("✓ Datasets")
    except ImportError as e:
        print(f"✗ Datasets import failed: {e}")
        return False
    
    try:
        from data_prep import clean_english_text, clean_japanese_text, augment_with_emojis
        print("✓ data_prep module")
    except ImportError as e:
        print(f"✗ data_prep import failed: {e}")
        return False
    
    try:
        from model_utils import count_parameters
        print("✓ model_utils module")
    except ImportError as e:
        print(f"✗ model_utils import failed: {e}")
        return False
    
    try:
        from export_utils import verify_model_size
        print("✓ export_utils module")
    except ImportError as e:
        print(f"✗ export_utils import failed: {e}")
        return False
    
    print("\n✓ All imports successful!\n")
    return True


def test_data_preparation():
    """Test data preparation functions."""
    print("="*60)
    print("TEST 2: Data Preparation Functions")
    print("="*60)
    
    from data_prep import clean_english_text, clean_japanese_text, augment_with_emojis
    
    # Test English cleaning
    test_text = "Hello World! Check this URL: http://example.com"
    cleaned = clean_english_text(test_text)
    print(f"English cleaning:")
    print(f"  Input:  {test_text}")
    print(f"  Output: {cleaned}")
    
    # Test Japanese cleaning
    test_text_ja = "今日は良い天気ですね http://example.com"
    cleaned_ja = clean_japanese_text(test_text_ja)
    print(f"\nJapanese cleaning:")
    print(f"  Input:  {test_text_ja}")
    print(f"  Output: {cleaned_ja}")
    
    # Test emoji augmentation
    sentences = ["hello world", "good morning", "thank you"]
    augmented = augment_with_emojis(sentences, emoji_ratio=0.5)
    print(f"\nEmoji augmentation:")
    for orig, aug in zip(sentences, augmented):
        print(f"  {orig} → {aug}")
    
    print("\n✓ Data preparation tests passed!\n")
    return True


def test_dataset_creation():
    """Test dataset creation."""
    print("="*60)
    print("TEST 3: Dataset Creation")
    print("="*60)
    
    from datasets import Dataset
    
    # Create sample dataset
    sample_data = {
        'text': [
            "today is a beautiful day",
            "i love programming",
            "the weather is nice"
        ]
    }
    
    dataset = Dataset.from_dict(sample_data)
    print(f"✓ Created dataset with {len(dataset)} samples")
    print(f"  Sample: {dataset[0]}")
    
    print("\n✓ Dataset creation test passed!\n")
    return True


def test_tokenizer():
    """Test tokenizer loading."""
    print("="*60)
    print("TEST 4: Tokenizer Loading")
    print("="*60)
    
    try:
        from transformers import AutoTokenizer
        
        # Test with a small model
        print("Loading tokenizer (this may take a moment)...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Test tokenization
        test_text = "hello world"
        tokens = tokenizer(test_text, return_tensors="pt")
        
        print(f"✓ Tokenizer loaded successfully")
        print(f"  Test text: '{test_text}'")
        print(f"  Token IDs: {tokens['input_ids'].tolist()}")
        
        print("\n✓ Tokenizer test passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Tokenizer test failed: {e}")
        return False


def test_model_loading():
    """Test model loading (without GPU)."""
    print("="*60)
    print("TEST 5: Model Loading (CPU)")
    print("="*60)
    
    try:
        from transformers import AutoModelForCausalLM
        from model_utils import count_parameters
        
        print("Loading small test model (this may take a moment)...")
        # Use a very small model for testing
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        
        print(f"✓ Model loaded successfully")
        
        # Count parameters
        stats = count_parameters(model)
        
        print("\n✓ Model loading test passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Model loading test failed: {e}")
        print("  Note: This is expected if you don't have enough RAM")
        return False


def test_file_structure():
    """Test that required directories exist."""
    print("="*60)
    print("TEST 6: File Structure")
    print("="*60)
    
    required_dirs = ['src', 'notebooks', 'data', 'models']
    required_files = [
        'src/data_prep.py',
        'src/model_utils.py',
        'src/export_utils.py',
        'src/colab_data_manager.py',
        'requirements.txt',
        'README.md'
    ]
    
    all_good = True
    
    # Check directories
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"✓ Directory exists: {dir_name}/")
        else:
            print(f"✗ Directory missing: {dir_name}/")
            all_good = False
    
    # Check files
    for file_name in required_files:
        if os.path.isfile(file_name):
            print(f"✓ File exists: {file_name}")
        else:
            print(f"✗ File missing: {file_name}")
            all_good = False
    
    if all_good:
        print("\n✓ File structure test passed!\n")
    else:
        print("\n✗ Some files/directories are missing\n")
    
    return all_good


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("LOCAL WORKFLOW TEST SUITE")
    print("="*60 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Data Preparation", test_data_preparation),
        ("Dataset Creation", test_dataset_creation),
        ("Tokenizer", test_tokenizer),
        ("File Structure", test_file_structure),
        # Skip model loading by default (requires lots of RAM)
        # ("Model Loading", test_model_loading),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} test crashed: {e}\n")
            results[test_name] = False
    
    # Summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready for Colab training.")
        return 0
    else:
        print("\n⚠ Some tests failed. Please fix issues before running in Colab.")
        return 1


if __name__ == "__main__":
    exit(main())
