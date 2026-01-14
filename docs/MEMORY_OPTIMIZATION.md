"""
Memory Optimization Guide for Colab Free Tier

Colab Free Tier Limits:
- RAM: ~12-13 GB
- GPU: T4 with 15 GB VRAM
- Session: 12 hours max, 90 min idle timeout

Common Issues:
1. RAM crashes during model loading
2. OOM during training
3. Runtime restarts

Solutions implemented in this project.
"""

## Issue: Runtime Restarts Due to RAM

**Symptoms:**
```
kernel restarted
AsyncIOLoopKernelRestarter: restarting kernel (1/5)
```

**Root Cause:**
Phi-3 Mini (3.8B parameters) uses ~8-10GB RAM when loaded in float16, leaving little room for training.

## Solutions

### Solution 1: Aggressive Memory Management (Implemented)

**In notebooks - Add this cell BEFORE model loading:**

```python
# Aggressive memory management for free tier
import gc
import torch

# Clear any existing models/data
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    
# Set memory allocation strategy
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

print("✓ Memory optimizations applied")
```

### Solution 2: Reduce Batch Size (Implemented)

Current settings:
```python
batch_size=8  # May be too large for free tier
```

**Recommended for free tier:**
```python
batch_size=2  # or even 1
gradient_accumulation_steps=4  # Maintain effective batch size
```

### Solution 3: Use Smaller Model (Alternative)

**Instead of Phi-3 Mini (3.8B), use:**

```python
# Option A: Phi-3 Mini-128K (smaller variant)
MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"  # Smaller context

# Option B: TinyLlama (1.1B - much smaller)
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Option C: Qwen 0.5B (smallest, fastest)
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
```

### Solution 4: Skip Optimization Steps

For free tier, you can skip:
- ✅ Pruning (already optional)
- ✅ Some export formats (keep only ONNX)

### Solution 5: Use Colab Pro

If free tier keeps crashing:
- **Colab Pro**: $9.99/month
  - 25GB RAM (2x free tier)
  - Better GPUs (A100, V100)
  - Longer sessions (24 hours)

## Recommended Workflow for Free Tier

### Step 1: Start Small
```python
# Use smaller model first
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
batch_size = 4
num_epochs = 1  # Quick test
```

### Step 2: Monitor Memory
```python
# Add to notebook
import psutil
import GPUtil

def print_memory_usage():
    # RAM
    ram = psutil.virtual_memory()
    print(f"RAM: {ram.used/1e9:.1f}GB / {ram.total/1e9:.1f}GB ({ram.percent}%)")
    
    # GPU
    if torch.cuda.is_available():
        gpu = GPUtil.getGPUs()[0]
        print(f"GPU: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")

# Call after each major step
print_memory_usage()
```

### Step 3: Clear Memory Between Steps
```python
# After training, before optimization
del trainer
gc.collect()
torch.cuda.empty_cache()
print_memory_usage()
```

### Step 4: Minimal Export
```python
# Only export ONNX (skip Core ML if OOM)
onnx_path = export_to_onnx(model, tokenizer, output_path)
# Skip: export_to_coreml() if running low on memory
```

## Quick Fixes to Try Now

### Fix 1: Restart and Reduce Batch Size

In your current Colab session, change training cell:
```python
trainer = train_causal_lm(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    output_dir=checkpoint_dir,
    num_epochs=3,
    batch_size=2,  # ← Changed from 8 to 2
    learning_rate=1e-5,
    max_seq_length=128,
    save_steps=100
)
```

### Fix 2: Add Memory Clearing

Add this cell before model loading:
```python
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
```

### Fix 3: Use Gradient Checkpointing

Add to model loading:
```python
model.gradient_checkpointing_enable()  # Trades compute for memory
```

## When to Upgrade to Colab Pro

Upgrade if you experience:
- ✗ Frequent runtime restarts
- ✗ Can't load Phi-3 model
- ✗ Training takes >12 hours
- ✗ Need to train multiple models

## Alternative: Use Smaller Dataset

Reduce training data size:
```python
# Instead of full dataset
augmented = augmented[:1000]  # Use only 1000 samples for testing
```

This lets you validate the workflow before full training.
