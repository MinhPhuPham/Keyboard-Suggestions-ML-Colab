"""
Model export utilities for Multi-Task GRU.
Save Keras model, vocabularies, config, and export TFLite.
"""
import os
import json
import tensorflow as tf

from . import config


def save_model(model, char_to_idx, word_to_idx):
    """Save model + vocabularies + config to MODEL_DIR.

    Saves:
    - model.keras: full Keras model
    - char_vocab.json: character vocabulary
    - word_vocab.json: word vocabulary
    - config.json: model configuration
    """
    # Model
    model_path = f'{config.MODEL_DIR}/model.keras'
    model.save(model_path)
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✓ model.keras ({size_mb:.2f} MB)")

    # Char vocab
    with open(f'{config.MODEL_DIR}/char_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(char_to_idx, f, ensure_ascii=False)
    print(f"  ✓ char_vocab.json ({len(char_to_idx):,} chars)")

    # Word vocab
    with open(f'{config.MODEL_DIR}/word_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(word_to_idx, f, ensure_ascii=False)
    print(f"  ✓ word_vocab.json ({len(word_to_idx):,} words)")

    # Config
    with open(f'{config.MODEL_DIR}/config.json', 'w') as f:
        json.dump({
            'architecture': 'MultiTask_BiGRU_v1',
            'char_vocab_size': len(char_to_idx),
            'word_vocab_size': len(word_to_idx),
            'embedding_dim': config.EMBEDDING_DIM,
            'gru_units': config.GRU_UNITS,
            'num_encoder_layers': config.NUM_ENCODER_LAYERS,
            'num_decoder_layers': config.NUM_DECODER_LAYERS,
            'max_encoder_len': config.MAX_ENCODER_LEN,
            'max_decoder_len': config.MAX_DECODER_LEN,
            'max_word_context': config.MAX_WORD_CONTEXT,
            'nwp_dense_units': config.NWP_DENSE_UNITS,
            'kkc_loss_weight': config.KKC_LOSS_WEIGHT,
            'nwp_loss_weight': config.NWP_LOSS_WEIGHT,
            'special_tokens': config.SPECIAL_TOKENS,
            'heads': ['kkc', 'nwp'],
        }, f, indent=2)
    print("  ✓ config.json")


def export_tflite(model):
    """Export model to TFLite format (full + FP16).

    Creates:
    - model.tflite: full precision
    - model_fp16.tflite: FP16 quantized (smaller)
    """
    print("\n📦 Exporting TFLite...")

    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter._experimental_lower_tensor_list_ops = False

        # Full precision
        tflite_model = converter.convert()
        path_full = f'{config.MODEL_DIR}/model.tflite'
        with open(path_full, 'wb') as f:
            f.write(tflite_model)
        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"  ✓ model.tflite ({size_mb:.2f} MB)")

        # FP16 quantized
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_fp16 = converter.convert()
        path_fp16 = f'{config.MODEL_DIR}/model_fp16.tflite'
        with open(path_fp16, 'wb') as f:
            f.write(tflite_fp16)
        size_mb = len(tflite_fp16) / (1024 * 1024)
        print(f"  ✓ model_fp16.tflite ({size_mb:.2f} MB)")

    except Exception as e:
        print(f"  ⚠ TFLite export failed: {e}")


def list_saved_files():
    """List all files saved in MODEL_DIR with sizes."""
    print(f"\n📦 Files in {config.MODEL_DIR}:")
    if not os.path.exists(config.MODEL_DIR):
        print("  (empty)")
        return
    for f in sorted(os.listdir(config.MODEL_DIR)):
        p = os.path.join(config.MODEL_DIR, f)
        if os.path.isfile(p):
            s = os.path.getsize(p)
            if s > 1024 * 1024:
                print(f"  {f}: {s / (1024*1024):.2f} MB")
            else:
                print(f"  {f}: {s / 1024:.1f} KB")
