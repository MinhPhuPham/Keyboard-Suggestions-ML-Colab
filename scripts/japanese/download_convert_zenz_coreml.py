#!/usr/bin/env python3
"""
Download and Convert zenz-v2.5-small to CoreML

This script downloads the Japanese GPT-2 model from HuggingFace
and converts it to CoreML format for iOS deployment.

Usage:
    python download_convert_zenz_coreml.py [--force-download] [--hf-token TOKEN]

Model: https://huggingface.co/Miwa-Keita/zenz-v3.1-xsmall

Compatible versions:
- Python: 3.10 or 3.11
- torch: 2.2.x - 2.3.x  
- transformers: 4.38.x - 4.41.x
- coremltools: 7.2+
"""

import os
import sys
import json
import shutil
import argparse
import time

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', 'japanese')

# Default model (can be overridden by --model-name)
DEFAULT_MODEL_NAME = "Miwa-Keita/zenz-v3.1-xsmall"

# These will be set based on model name
MODEL_NAME = None
LOCAL_MODEL_DIR = None
COREML_MODEL_PATH = None

def setup_paths(model_name: str):
    """Setup global paths based on model name."""
    global MODEL_NAME, LOCAL_MODEL_DIR, COREML_MODEL_PATH
    MODEL_NAME = model_name
    # Extract short name from "Miwa-Keita/zenz-v3.1-xsmall" -> "zenz-v3.1-xsmall"
    short_name = model_name.split('/')[-1]
    LOCAL_MODEL_DIR = os.path.join(MODELS_DIR, short_name)
    COREML_MODEL_PATH = os.path.join(MODELS_DIR, f'{short_name.replace(".", "_")}_coreml.mlpackage')


def get_vocab_size_from_config() -> int:
    """Read vocab_size from downloaded config.json."""
    config_path = os.path.join(LOCAL_MODEL_DIR, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    vocab_size = config.get('vocab_size')
    if vocab_size is None:
        raise ValueError("vocab_size not found in config.json")
    
    print(f"   📋 vocab_size from config.json: {vocab_size}")
    return vocab_size


def model_exists() -> bool:
    """Check if model is already downloaded."""
    config_path = os.path.join(LOCAL_MODEL_DIR, 'config.json')
    model_file = os.path.join(LOCAL_MODEL_DIR, 'model.safetensors')
    return os.path.exists(config_path) and os.path.exists(model_file)


def step_1_download_model(hf_token: str = None):
    """Download model and tokenizer from HuggingFace."""
    print("\n" + "=" * 60)
    print("[1/2] Downloading Model from HuggingFace...")
    print("=" * 60)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    
    print(f"Model: {MODEL_NAME}")
    print(f"Saving to: {LOCAL_MODEL_DIR}")
    
    start_time = time.time()
    
    # Download tokenizer
    print("\n📥 Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(LOCAL_MODEL_DIR)
    print("   ✓ Tokenizer saved")
    
    # Download model
    print("\n📥 Downloading model (181 MB)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        trust_remote_code=True
    )
    model.save_pretrained(LOCAL_MODEL_DIR)
    
    elapsed = time.time() - start_time
    print(f"   ✓ Model saved ({elapsed:.1f}s)")
    
    return model, tokenizer


import torch.nn as nn


class GPT2TraceWrapper(nn.Module):
    """
    Wrapper to strip away the tuple output of GPT-2
    and return ONLY the logits tensor.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids):
        # The model returns (logits, past_key_values) when torchscript=True
        # We only want the first element [0] -> logits
        outputs = self.model(input_ids)
        return outputs[0]


def step_2_convert_to_coreml():
    """Convert PyTorch model to CoreML using torch.jit.trace."""
    print("\n" + "=" * 60)
    print("[2/2] Converting to CoreML...")
    print("=" * 60)
    
    import torch
    import coremltools as ct
    from transformers import GPT2LMHeadModel
    
    # Get vocab size from config
    vocab_size = get_vocab_size_from_config()
    
    # Load model with torchscript=True for compatibility
    print("Loading model with torchscript=True...")
    base_model = GPT2LMHeadModel.from_pretrained(LOCAL_MODEL_DIR, torchscript=True)
    base_model.eval()
    
    # Wrap the model to return only logits
    print("📦 Wrapping model for clean tracing...")
    model = GPT2TraceWrapper(base_model)
    
    # Create dummy input
    SEQ_LEN = 128
    dummy_input = torch.randint(0, vocab_size, (1, SEQ_LEN))
    
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Vocab size: {vocab_size}")
    
    # Trace the model
    print("\n📦 Tracing model with torch.jit.trace...")
    start_time = time.time()
    
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Convert to CoreML
    print("   Converting traced model to CoreML...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input_ids", shape=dummy_input.shape, dtype=int)],
        outputs=[ct.TensorType(name="logits")],  # Now matches wrapper output!
        minimum_deployment_target=ct.target.iOS16,
        convert_to='mlprogram'
    )
    
    # Add metadata
    mlmodel.short_description = "Zenz v2.5 Japanese Keyboard Suggestions (GPT-2)"
    mlmodel.author = "Miwa-Keita (HuggingFace) / Converted by MinhPhuPham"
    mlmodel.version = "2.5"
    
    # Save
    print(f"\n💾 Saving to: {COREML_MODEL_PATH}")
    mlmodel.save(COREML_MODEL_PATH)
    
    elapsed = time.time() - start_time
    print(f"   ✓ CoreML model saved ({elapsed:.1f}s)")
    
    return COREML_MODEL_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Download Japanese GPT-2 model and convert to CoreML"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"HuggingFace model name (default: {DEFAULT_MODEL_NAME})"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API token (optional, model is public)"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if model exists"
    )
    parser.add_argument(
        "--skip-coreml",
        action="store_true",
        help="Skip CoreML conversion (download only)"
    )
    
    args = parser.parse_args()
    
    # Setup paths based on model name
    setup_paths(args.model_name)
    
    short_name = args.model_name.split('/')[-1]
    print("\n" + "═" * 60)
    print("🇯🇵 Japanese Keyboard Model")
    print(f"   Model: {short_name} (GPT-2)")
    if args.skip_coreml:
        print("   Mode: Download only (skip CoreML)")
    else:
        print("   Mode: Download + CoreML conversion")
    print("═" * 60)
    
    # Smart download logic:
    # - exists + force-download => delete and re-download
    # - exists + no force-download => skip to convert
    # - not exists => download
    
    should_download = False
    
    if model_exists():
        if args.force_download:
            print(f"\n🔄 Force download requested. Removing existing model...")
            shutil.rmtree(LOCAL_MODEL_DIR)
            should_download = True
        else:
            print(f"\n✓ Model already exists: {LOCAL_MODEL_DIR}")
            print("   (Use --force-download to re-download)")
    else:
        should_download = True
    
    # Step 1: Download (if needed)
    if should_download:
        step_1_download_model(args.hf_token)
    
    # Step 2: CoreML Conversion (if not skipped)
    if not args.skip_coreml:
        step_2_convert_to_coreml()
    
    # Summary
    print("\n" + "═" * 60)
    if args.skip_coreml:
        print("✅ DOWNLOAD COMPLETE!")
        print("═" * 60)
        print(f"📁 Model files: {LOCAL_MODEL_DIR}")
        print("\n💡 To convert to CoreML, run without --skip-coreml")
    else:
        print("✅ CONVERSION COMPLETE!")
        print("═" * 60)
        print(f"📁 Model files: {LOCAL_MODEL_DIR}")
        print(f"📁 CoreML:      {COREML_MODEL_PATH}")
        print("\n👉 Open the .mlpackage in Xcode to verify the model")
    print("═" * 60)


if __name__ == "__main__":
    main()
