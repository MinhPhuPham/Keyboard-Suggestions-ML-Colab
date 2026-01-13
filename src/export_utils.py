"""
Model Export Utilities

This module provides functions for exporting trained models to ONNX,
Core ML, and other formats for deployment on mobile devices.
"""

import torch
import os
from typing import Tuple, Optional
import zipfile


def export_to_onnx(
    model,
    tokenizer,
    output_path: str,
    max_seq_length: int = 8,
    opset_version: int = 14
) -> str:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to export
        tokenizer: Tokenizer for the model
        output_path: Path to save ONNX model
        max_seq_length: Maximum sequence length
        opset_version: ONNX opset version
        
    Returns:
        Path to exported ONNX model
    """
    print(f"Exporting model to ONNX: {output_path}")
    
    # Ensure model is in eval mode and on CPU
    model.eval()
    model = model.cpu()
    
    # Create dummy input
    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, max_seq_length))
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'logits': {0: 'batch_size', 1: 'sequence'}
        }
    )
    
    print(f"ONNX export complete: {output_path}")
    
    return output_path


def export_to_coreml(
    onnx_path: str,
    output_path: str,
    model_name: str = "KeyboardSuggestionModel"
) -> str:
    """
    Convert ONNX model to Core ML format for iOS.
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Path to save Core ML model
        model_name: Name for the Core ML model
        
    Returns:
        Path to exported Core ML model
    """
    try:
        import coremltools as ct
    except ImportError:
        print("Error: coremltools not installed")
        print("Install with: pip install coremltools")
        return None
    
    print(f"Converting ONNX to Core ML: {output_path}")
    
    # Load ONNX model
    model = ct.converters.onnx.convert(
        model=onnx_path,
        minimum_deployment_target=ct.target.iOS15,
    )
    
    # Set metadata
    model.author = "KeyboardSuggestionsML"
    model.short_description = f"{model_name} for keyboard suggestions"
    model.version = "1.0"
    
    # Save Core ML model
    model.save(output_path)
    
    print(f"Core ML export complete: {output_path}")
    
    return output_path


def verify_model_size(model_path: str, max_size_mb: float) -> Tuple[float, bool]:
    """
    Verify that model file size meets requirements.
    
    Args:
        model_path: Path to model file
        max_size_mb: Maximum allowed size in MB
        
    Returns:
        Tuple of (actual_size_mb, meets_requirement)
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return 0.0, False
    
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    
    meets_requirement = size_mb <= max_size_mb
    
    status = "✓ PASS" if meets_requirement else "✗ FAIL"
    print(f"{status} Model size: {size_mb:.2f} MB (limit: {max_size_mb} MB)")
    
    return size_mb, meets_requirement


def benchmark_latency(
    model,
    tokenizer,
    num_runs: int = 100,
    seq_length: int = 8
) -> float:
    """
    Benchmark model inference latency.
    
    Args:
        model: Model to benchmark
        tokenizer: Tokenizer
        num_runs: Number of inference runs
        seq_length: Input sequence length
        
    Returns:
        Average latency in milliseconds
    """
    import time
    
    print(f"Benchmarking latency ({num_runs} runs)...")
    
    model.eval()
    device = next(model.parameters()).device
    
    # Create dummy input
    dummy_input = torch.randint(
        0, tokenizer.vocab_size, 
        (1, seq_length)
    ).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(dummy_input)
            end = time.time()
            latencies.append((end - start) * 1000)  # Convert to ms
    
    avg_latency = sum(latencies) / len(latencies)
    
    print(f"Average latency: {avg_latency:.2f} ms")
    
    return avg_latency


def package_for_download(
    model_dir: str,
    output_zip: str,
    include_patterns: Optional[list] = None
) -> str:
    """
    Package model files into a zip archive for download.
    
    Args:
        model_dir: Directory containing model files
        output_zip: Path to output zip file
        include_patterns: List of file patterns to include (default: all)
        
    Returns:
        Path to created zip file
    """
    print(f"Packaging model files: {output_zip}")
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, model_dir)
                
                # Check if file matches include patterns
                if include_patterns:
                    if not any(pattern in file for pattern in include_patterns):
                        continue
                
                zipf.write(file_path, arcname)
                print(f"  Added: {arcname}")
    
    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"Package complete: {output_zip} ({size_mb:.2f} MB)")
    
    return output_zip


def save_model_metadata(
    output_path: str,
    model_name: str,
    vocab_size: int,
    max_seq_length: int,
    model_size_mb: float,
    perplexity: float,
    latency_ms: float
) -> str:
    """
    Save model metadata to a JSON file.
    
    Args:
        output_path: Path to save metadata file
        model_name: Name of the model
        vocab_size: Vocabulary size
        max_seq_length: Maximum sequence length
        model_size_mb: Model file size in MB
        perplexity: Model perplexity score
        latency_ms: Average inference latency in ms
        
    Returns:
        Path to metadata file
    """
    import json
    from datetime import datetime
    
    metadata = {
        "model_name": model_name,
        "vocab_size": vocab_size,
        "max_seq_length": max_seq_length,
        "model_size_mb": model_size_mb,
        "perplexity": perplexity,
        "latency_ms": latency_ms,
        "created_at": datetime.now().isoformat(),
    }
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved: {output_path}")
    
    return output_path


if __name__ == "__main__":
    print("Model Export Utilities")
    print("Import this module in your notebooks or scripts")
    print("\nExample:")
    print("  from src.export_utils import export_to_onnx, export_to_coreml")
