#!/usr/bin/env python3
"""
Export MobileBERT to optimized ONNX for mobile deployment

This version includes:
- Aggressive INT8 quantization targeting Gather/Embeddings
- Disabled per-channel quantization to save RAM
- ORT format conversion for memory-mapped loading
- Target: ~25-30MB model size, <80MB RAM usage
"""

import argparse
import os
import sys
import torch
from pathlib import Path
from transformers import MobileBertForMaskedLM, MobileBertTokenizer

try:
    import onnx
    from onnx.external_data_helper import convert_model_from_external_data
    from onnxruntime.quantization import quantize_dynamic, QuantType
    import onnxruntime as ort
except ImportError:
    print("❌ Error: Required packages not installed")
    print("Install with: pip install onnx onnxruntime")
    sys.exit(1)


def export_mobile_optimized(model_dir: str, output_dir: str):
    """Export MobileBERT model optimized for mobile deployment."""
    print("="*70)
    print("📱 MobileBERT → Optimized Mobile ONNX Export")
    print("="*70)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    temp_fp32_path = output_path / "temp_fp32.onnx"
    final_onnx_path = output_path / "keyboard_model.onnx"
    final_ort_path = output_path / "keyboard_model.ort"
    
    # 1. Load model
    print(f"\n[1/5] Loading model from: {model_dir}")
    try:
        tokenizer = MobileBertTokenizer.from_pretrained(model_dir)
        model = MobileBertForMaskedLM.from_pretrained(model_dir)
        model.eval()
        model.cpu()
        print(f"   ✓ Model loaded successfully")
        print(f"   ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        sys.exit(1)
    
    # 2. Export to ONNX (FP32)
    print("\n[2/5] Exporting to ONNX (FP32)...")
    try:
        dummy_input_ids = torch.zeros(1, 32, dtype=torch.long)
        dummy_attention_mask = torch.ones(1, 32, dtype=torch.long)
        
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            str(temp_fp32_path),
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'attention_mask': {0: 'batch_size', 1: 'sequence'},
                'logits': {0: 'batch_size', 1: 'sequence'}
            },
            opset_version=17,  # Higher opset for better quantization support
            do_constant_folding=True,
            export_params=True
        )
        
        # Merge external data if any
        print("   → Merging external weights...")
        model_proto = onnx.load(str(temp_fp32_path))
        convert_model_from_external_data(model_proto)
        onnx.save(model_proto, str(temp_fp32_path))
        
        # Clean up .data file
        data_file = Path(str(temp_fp32_path) + ".data")
        if data_file.exists():
            data_file.unlink()
        
        fp32_size_mb = temp_fp32_path.stat().st_size / (1024 * 1024)
        print(f"   ✓ FP32 ONNX export successful")
        print(f"   ✓ FP32 size: {fp32_size_mb:.2f} MB")
        
    except Exception as e:
        print(f"   ❌ Failed to export to ONNX: {e}")
        sys.exit(1)
    
    # 3. Aggressive INT8 Quantization
    print("\n[3/5] Quantizing to INT8 (targeting Embeddings)...")
    try:
        # CRITICAL: Quantize Gather operator (Embeddings) to reduce size
        # This is where the vocabulary embeddings live (~60-80MB)
        op_types_to_quantize = [
            'MatMul',      # Matrix multiplications
            'Gemm',        # General matrix multiply
            'Gather',      # Embedding lookups (CRITICAL for size reduction)
            'Add',         # Addition operations
            'Mul'          # Multiplication operations
        ]
        
        print(f"   → Targeting operators: {', '.join(op_types_to_quantize)}")
        print(f"   → Using per_channel=False (saves RAM on mobile)")
        
        quantize_dynamic(
            model_input=str(temp_fp32_path),
            model_output=str(final_onnx_path),
            weight_type=QuantType.QUInt8,  # UINT8 safer for mobile (Android NNAPI)
            op_types_to_quantize=op_types_to_quantize,
            per_channel=False,   # Saves RAM, reduces metadata
            reduce_range=True    # Prevents overflow on mobile hardware
        )
        
        # Clean up temp file
        if temp_fp32_path.exists():
            temp_fp32_path.unlink()
        
        int8_size_mb = final_onnx_path.stat().st_size / (1024 * 1024)
        reduction = ((fp32_size_mb - int8_size_mb) / fp32_size_mb) * 100
        
        print(f"   ✓ INT8 quantization successful")
        print(f"   ✓ INT8 size: {int8_size_mb:.2f} MB")
        print(f"   ✓ Size reduction: {reduction:.1f}%")
        
    except Exception as e:
        print(f"   ⚠️  Quantization failed: {e}")
        print(f"   → Using FP32 model")
        if temp_fp32_path.exists():
            temp_fp32_path.rename(final_onnx_path)
        int8_size_mb = fp32_size_mb
    
    # 4. Verify ONNX model
    print("\n[4/5] Verifying ONNX model...")
    try:
        onnx_model = onnx.load(str(final_onnx_path))
        onnx.checker.check_model(onnx_model)
        print(f"   ✓ ONNX model is valid")
        print(f"   ✓ IR version: {onnx_model.ir_version}")
        print(f"   ✓ Opset version: {onnx_model.opset_import[0].version}")
    except Exception as e:
        print(f"   ⚠️  ONNX model validation warning: {e}")
    
    # 5. Convert to ORT format (for memory-mapped loading)
    print("\n[5/5] Converting to ORT format (memory-mapped loading)...")
    try:
        import subprocess
        
        # Use onnxruntime tools to convert to ORT format
        result = subprocess.run([
            sys.executable, "-m", 
            "onnxruntime.tools.convert_onnx_models_to_ort",
            str(final_onnx_path)
        ], capture_output=True, text=True)
        
        # The tool creates .ort file with same base name
        generated_ort = final_onnx_path.with_suffix('.ort')
        
        if generated_ort.exists():
            if generated_ort != final_ort_path:
                generated_ort.rename(final_ort_path)
            
            ort_size_mb = final_ort_path.stat().st_size / (1024 * 1024)
            print(f"   ✓ ORT conversion successful")
            print(f"   ✓ ORT size: {ort_size_mb:.2f} MB")
            print(f"   🚀 RAM usage should be <80MB with memory mapping")
        else:
            print(f"   ⚠️  ORT file not generated")
            if result.stderr:
                print(f"   Error: {result.stderr}")
            
    except Exception as e:
        print(f"   ⚠️  Could not convert to ORT format: {e}")
        print(f"   → Install with: pip install onnxruntime-tools")
        print(f"   → You can still use the .onnx file")
    
    # 6. Test inference
    print("\n[6/6] Testing ONNX Runtime inference...")
    try:
        # Test with .onnx file
        session = ort.InferenceSession(str(final_onnx_path))
        
        test_input = {
            'input_ids': dummy_input_ids.numpy(),
            'attention_mask': dummy_attention_mask.numpy()
        }
        
        import time
        start = time.time()
        outputs = session.run(None, test_input)
        inference_time = (time.time() - start) * 1000
        
        print(f"   ✓ Inference test successful")
        print(f"   ✓ Inference time: {inference_time:.2f} ms")
        print(f"   ✓ Output shape: {outputs[0].shape}")
    except Exception as e:
        print(f"   ⚠️  Inference test failed: {e}")
    
    # 7. Save vocabulary
    print("\n[7/7] Saving vocabulary...")
    vocab_path = output_path / "vocab.txt"
    try:
        tokenizer.save_vocabulary(str(output_path))
        vocab_size_kb = vocab_path.stat().st_size / 1024
        print(f"   ✓ Vocabulary saved to: {vocab_path}")
        print(f"   ✓ Vocab size: {vocab_size_kb:.2f} KB")
    except Exception as e:
        print(f"   ⚠️  Failed to save vocabulary: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("✅ Export Complete!")
    print("="*70)
    print(f"\nFiles created:")
    print(f"  • keyboard_model.onnx ({int8_size_mb:.2f} MB) - Use for testing")
    if final_ort_path.exists():
        print(f"  • keyboard_model.ort ({ort_size_mb:.2f} MB) - Use in production app")
    print(f"  • vocab.txt ({vocab_size_kb:.2f} KB)")
    
    print(f"\nOptimizations applied:")
    print(f"  ✓ INT8 quantization (Gather/Embeddings targeted)")
    print(f"  ✓ Per-channel disabled (RAM optimized)")
    if final_ort_path.exists():
        print(f"  ✓ ORT format (memory-mapped loading)")
    
    print(f"\nExpected mobile performance:")
    print(f"  • Model size: ~{int8_size_mb:.0f} MB")
    print(f"  • RAM usage: <80 MB (with ORT format)")
    print(f"  • Inference: ~{inference_time:.0f} ms")
    
    print(f"\nNext steps:")
    if final_ort_path.exists():
        print(f"  1. Use keyboard_model.ort for production")
        print(f"  2. Update app to load .ort file (see docs)")
    else:
        print(f"  1. Use keyboard_model.onnx")
        print(f"  2. Install onnxruntime-tools for ORT format")
    print(f"  3. iOS: Follow docs/IOS_INTEGRATION.md")
    print(f"  4. Android: Follow docs/ANDROID_INTEGRATION.md")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Export MobileBERT keyboard model optimized for mobile"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./exports/onnx",
        help="Output directory for ONNX model (default: ./exports/onnx)"
    )
    
    args = parser.parse_args()
    
    # Validate model directory
    if not os.path.exists(args.model_dir):
        print(f"❌ Error: Model directory not found: {args.model_dir}")
        sys.exit(1)
    
    # Export
    export_mobile_optimized(
        model_dir=args.model_dir,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
