"""
Model Training and Optimization Utilities

This module provides reusable functions for loading models with LoRA,
training, evaluation, pruning, and quantization.
"""

import torch
import torch.nn.utils.prune as prune
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, PeftModel
from typing import Optional, Dict, Any
import math


def load_model_with_lora(
    model_name: str,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: Optional[list] = None
) -> tuple:
    """
    Load a pretrained model and add LoRA adapters.
    
    Args:
        model_name: Hugging Face model identifier
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA (auto-detected if None)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with use_cache=False to avoid DynamicCache compatibility issues
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False  # Disable KV cache to avoid compatibility issues
    )
    
    # Auto-detect target modules if not specified
    if target_modules is None:
        # Check model architecture and set appropriate modules
        model_type = model.config.model_type.lower()
        
        if "phi" in model_type or "phi3" in model_name.lower():
            # Phi-3 uses: qkv_proj or o_proj
            target_modules = ["qkv_proj", "o_proj"]
            print(f"Detected Phi-3 model, using modules: {target_modules}")
        elif "qwen" in model_type or "qwen" in model_name.lower():
            # Qwen uses: c_attn or attn.c_proj
            target_modules = ["c_attn", "c_proj"]
            print(f"Detected Qwen model, using modules: {target_modules}")
        else:
            # Default for most models (LLaMA, Mistral, etc.)
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            print(f"Using default modules: {target_modules}")
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Add LoRA adapters
    try:
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    except ValueError as e:
        print(f"\n⚠ Error applying LoRA: {e}")
        print("\nAvailable modules in model:")
        for name, _ in model.named_modules():
            if 'attn' in name.lower() or 'proj' in name.lower():
                print(f"  - {name}")
        raise
    
    return model, tokenizer


def train_causal_lm(
    model,
    tokenizer,
    train_dataset,
    val_dataset=None,
    output_dir: str = "./checkpoints",
    num_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 1e-5,
    max_seq_length: int = 8,
    save_steps: int = 500
) -> Trainer:
    """
    Train a causal language model.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer for the model
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        max_seq_length: Maximum sequence length
        save_steps: Save checkpoint every N steps
        
    Returns:
        Trained Trainer object
    """
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_steps=50,
        save_steps=save_steps,
        save_total_limit=2,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=save_steps if val_dataset else None,
        fp16=True,  # Mixed precision training
        gradient_accumulation_steps=1,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",  # Disable W&B, tensorboard, etc.
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    return trainer


def evaluate_perplexity(model, tokenizer, eval_dataset) -> float:
    """
    Calculate perplexity on evaluation dataset.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        eval_dataset: Evaluation dataset
        
    Returns:
        Perplexity score
    """
    model.eval()
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
    )
    
    eval_results = trainer.evaluate(eval_dataset)
    perplexity = math.exp(eval_results['eval_loss'])
    
    print(f"Perplexity: {perplexity:.2f}")
    
    return perplexity


def prune_model(model, amount: float = 0.3) -> torch.nn.Module:
    """
    Apply L1 unstructured pruning to model.
    
    Args:
        model: Model to prune
        amount: Proportion of weights to prune (0.0 to 1.0)
        
    Returns:
        Pruned model
    """
    print(f"Pruning {amount*100}% of model weights...")
    
    # Get all Linear layers
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    # Apply global unstructured pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    
    # Make pruning permanent
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)
    
    print("Pruning complete")
    
    return model


def quantize_model(model, dtype=torch.qint8) -> torch.nn.Module:
    """
    Apply dynamic quantization to model.
    
    Args:
        model: Model to quantize
        dtype: Quantization data type (default: qint8)
        
    Returns:
        Quantized model
    """
    print(f"Quantizing model to {dtype}...")
    
    # Move to CPU for quantization
    model = model.cpu()
    
    # Apply dynamic quantization to Linear layers
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=dtype
    )
    
    print("Quantization complete")
    
    return quantized_model


def merge_lora_weights(model) -> torch.nn.Module:
    """
    Merge LoRA weights into base model.
    
    Args:
        model: PEFT model with LoRA adapters
        
    Returns:
        Model with merged weights
    """
    print("Merging LoRA weights into base model...")
    
    if isinstance(model, PeftModel):
        model = model.merge_and_unload()
        print("LoRA weights merged")
    else:
        print("Model is not a PEFT model, skipping merge")
    
    return model


def count_parameters(model) -> Dict[str, int]:
    """
    Count total and trainable parameters in model.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    stats = {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {stats['frozen']:,}")
    
    return stats


if __name__ == "__main__":
    print("Model Training and Optimization Utilities")
    print("Import this module in your notebooks or scripts")
    print("\nExample:")
    print("  from src.model_utils import load_model_with_lora, train_causal_lm")
