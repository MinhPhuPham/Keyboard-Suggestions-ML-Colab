"""
Local test script for Shared Encoder Multi-Task GRU model (v2).
Mocks small datasets to validate model build + training without Colab.

Usage:
    python -m scripts.japanese_enhancement.test_local
"""
import os
import sys
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info logs

import tensorflow as tf
from scripts.japanese_enhancement import config
from scripts.japanese_enhancement.model import (
    build_shared_multitask_model, masked_sparse_ce, masked_accuracy,
)

# ===========================================================
# CONFIG
# ===========================================================
BATCH_SIZE = 4
NUM_SAMPLES = 20  # Tiny dataset
CHAR_VOCAB = 100  # Small vocabs for fast test
WORD_VOCAB = 100


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_model_build():
    """Test 1: Build shared encoder model and print summary."""
    separator("TEST 1: Shared Encoder Model Build")

    model = build_shared_multitask_model(
        CHAR_VOCAB, WORD_VOCAB, strategy=None
    )
    model.summary()

    print(f"\n✅ Model built successfully")
    print(f"   Name:    {model.name}")
    print(f"   Inputs:  {[i.name + ' ' + str(i.shape) for i in model.inputs]}")
    print(f"   Outputs: {[o.name + ' ' + str(o.shape) for o in model.outputs]}")
    print(f"   Params:  {model.count_params():,}")

    # Verify 2 inputs (not 3!)
    assert len(model.inputs) == 2, \
        f"Expected 2 inputs (shared encoder), got {len(model.inputs)}"
    assert len(model.outputs) == 2, \
        f"Expected 2 outputs, got {len(model.outputs)}"

    return model


def test_forward_pass(model):
    """Test 2: Forward pass with mock data."""
    separator("TEST 2: Forward Pass")

    enc_input = np.random.randint(
        0, CHAR_VOCAB, (BATCH_SIZE, config.MAX_ENCODER_LEN)
    ).astype(np.int32)
    dec_input = np.random.randint(
        0, CHAR_VOCAB, (BATCH_SIZE, config.MAX_DECODER_LEN)
    ).astype(np.int32)

    print(f"  enc_input: {enc_input.shape}")
    print(f"  dec_input: {dec_input.shape}")
    print(f"  (No separate nwp_input — uses shared encoder!)")

    outputs = model.predict([enc_input, dec_input], verbose=0)

    kkc_out, nwp_out = outputs
    print(f"\n  kkc_output: {kkc_out.shape}  "
          f"(expect: ({BATCH_SIZE}, {config.MAX_DECODER_LEN}, {CHAR_VOCAB}))")
    print(f"  nwp_output: {nwp_out.shape}  "
          f"(expect: ({BATCH_SIZE}, {WORD_VOCAB}))")

    assert kkc_out.shape == (BATCH_SIZE, config.MAX_DECODER_LEN, CHAR_VOCAB), \
        f"KKC shape mismatch: {kkc_out.shape}"
    assert nwp_out.shape == (BATCH_SIZE, WORD_VOCAB), \
        f"NWP shape mismatch: {nwp_out.shape}"

    print(f"\n✅ Forward pass OK — all shapes correct")


def test_kkc_only(model):
    """Test 3: KKC-only forward (decoder input is real, NWP output ignored)."""
    separator("TEST 3: KKC-Only Forward")

    enc_input = np.random.randint(
        0, CHAR_VOCAB, (BATCH_SIZE, config.MAX_ENCODER_LEN)
    ).astype(np.int32)
    dec_input = np.random.randint(
        0, CHAR_VOCAB, (BATCH_SIZE, config.MAX_DECODER_LEN)
    ).astype(np.int32)

    kkc_out, nwp_out = model.predict([enc_input, dec_input], verbose=0)
    print(f"  kkc_output: {kkc_out.shape}")
    print(f"  nwp_output: {nwp_out.shape} (ignored in KKC mode)")
    print(f"✅ KKC-only forward OK")


def test_nwp_only(model):
    """Test 4: NWP-only forward (encoder gets NWP chars, decoder gets zeros)."""
    separator("TEST 4: NWP-Only Forward (decoder=zeros)")

    # NWP context chars (same shape as encoder input!)
    nwp_chars = np.random.randint(
        0, CHAR_VOCAB, (BATCH_SIZE, config.MAX_ENCODER_LEN)
    ).astype(np.int32)
    dec_dummy = np.zeros(
        (BATCH_SIZE, config.MAX_DECODER_LEN), dtype=np.int32
    )

    kkc_out, nwp_out = model.predict([nwp_chars, dec_dummy], verbose=0)
    print(f"  kkc_output: {kkc_out.shape} (ignored in NWP mode)")
    print(f"  nwp_output: {nwp_out.shape}")
    print(f"✅ NWP-only forward OK")


def test_gradient_tape(model):
    """Test 5: Two-forward-pass GradientTape step (shared encoder training)."""
    separator("TEST 5: GradientTape — Two Forward Passes")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    kkc_loss_fn = masked_sparse_ce
    nwp_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Mock KKC batch
    kkc_enc = tf.constant(np.random.randint(
        0, CHAR_VOCAB, (BATCH_SIZE, config.MAX_ENCODER_LEN)
    ), dtype=tf.int32)
    kkc_dec_in = tf.constant(np.random.randint(
        0, CHAR_VOCAB, (BATCH_SIZE, config.MAX_DECODER_LEN)
    ), dtype=tf.int32)
    kkc_dec_tgt = tf.constant(np.random.randint(
        0, CHAR_VOCAB, (BATCH_SIZE, config.MAX_DECODER_LEN)
    ), dtype=tf.int32)

    # Mock NWP batch (char-level context, same shape as encoder!)
    nwp_enc_chars = tf.constant(np.random.randint(
        0, CHAR_VOCAB, (BATCH_SIZE, config.MAX_ENCODER_LEN)
    ), dtype=tf.int32)
    nwp_y = tf.constant(np.random.randint(
        0, WORD_VOCAB, (BATCH_SIZE,)
    ), dtype=tf.int32)

    print(f"  KKC: enc={kkc_enc.shape}, dec_in={kkc_dec_in.shape}")
    print(f"  NWP: enc_chars={nwp_enc_chars.shape}, y={nwp_y.shape}")

    with tf.GradientTape() as tape:
        # Pass 1: KKC data through shared encoder
        kkc_pred, _ = model([kkc_enc, kkc_dec_in], training=True)
        kkc_loss = kkc_loss_fn(kkc_dec_tgt, kkc_pred)

        # Pass 2: NWP chars through shared encoder
        dec_dummy = tf.zeros((BATCH_SIZE, config.MAX_DECODER_LEN), dtype=tf.int32)
        _, nwp_pred = model([nwp_enc_chars, dec_dummy], training=True)
        nwp_loss = nwp_loss_fn(nwp_y, nwp_pred)

        total_loss = kkc_loss + nwp_loss

    print(f"  KKC loss: {kkc_loss.numpy():.4f}")
    print(f"  NWP loss: {nwp_loss.numpy():.4f}")
    print(f"  Total loss: {total_loss.numpy():.4f}")

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    n_grads = sum(1 for g in grads if g is not None)
    n_total = len(model.trainable_variables)
    print(f"\n  Gradients: {n_grads}/{n_total} non-None")

    # CRITICAL: all grads must be non-None (shared encoder gets from both!)
    assert n_grads == n_total, \
        f"Expected all {n_total} gradients non-None, got {n_grads}"
    print(f"✅ GradientTape OK — all {n_total} variables have gradients")


def test_multi_steps(model):
    """Test 6: Multiple training steps to check stability."""
    separator("TEST 6: Multi-Step Training (3 steps)")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    kkc_loss_fn = masked_sparse_ce
    nwp_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    for step in range(3):
        kkc_enc = tf.constant(np.random.randint(
            0, CHAR_VOCAB, (BATCH_SIZE, config.MAX_ENCODER_LEN)
        ), dtype=tf.int32)
        kkc_dec_in = tf.constant(np.random.randint(
            0, CHAR_VOCAB, (BATCH_SIZE, config.MAX_DECODER_LEN)
        ), dtype=tf.int32)
        kkc_dec_tgt = tf.constant(np.random.randint(
            0, CHAR_VOCAB, (BATCH_SIZE, config.MAX_DECODER_LEN)
        ), dtype=tf.int32)
        nwp_enc_chars = tf.constant(np.random.randint(
            0, CHAR_VOCAB, (BATCH_SIZE, config.MAX_ENCODER_LEN)
        ), dtype=tf.int32)
        nwp_y = tf.constant(np.random.randint(
            0, WORD_VOCAB, (BATCH_SIZE,)
        ), dtype=tf.int32)

        with tf.GradientTape() as tape:
            # Pass 1: KKC
            kkc_pred, _ = model([kkc_enc, kkc_dec_in], training=True)
            kkc_loss = kkc_loss_fn(kkc_dec_tgt, kkc_pred)

            # Pass 2: NWP
            dec_dummy = tf.zeros(
                (BATCH_SIZE, config.MAX_DECODER_LEN), dtype=tf.int32
            )
            _, nwp_pred = model([nwp_enc_chars, dec_dummy], training=True)
            nwp_loss = nwp_loss_fn(nwp_y, nwp_pred)

            total_loss = kkc_loss + nwp_loss

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(f"  Step {step+1}: loss={total_loss.numpy():.4f}  "
              f"kkc={kkc_loss.numpy():.4f}  nwp={nwp_loss.numpy():.4f}")

    print(f"✅ Multi-step training OK — no crashes")


def test_shared_encoder_grads():
    """Test 7: Verify shared encoder gets gradients from BOTH tasks."""
    separator("TEST 7: Shared Encoder Gradient Verification")

    model = build_shared_multitask_model(
        CHAR_VOCAB, WORD_VOCAB, strategy=None
    )
    kkc_loss_fn = masked_sparse_ce
    nwp_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    kkc_enc = tf.constant(np.random.randint(
        0, CHAR_VOCAB, (BATCH_SIZE, config.MAX_ENCODER_LEN)
    ), dtype=tf.int32)
    kkc_dec_in = tf.constant(np.random.randint(
        0, CHAR_VOCAB, (BATCH_SIZE, config.MAX_DECODER_LEN)
    ), dtype=tf.int32)
    kkc_dec_tgt = tf.constant(np.random.randint(
        0, CHAR_VOCAB, (BATCH_SIZE, config.MAX_DECODER_LEN)
    ), dtype=tf.int32)
    nwp_enc_chars = tf.constant(np.random.randint(
        0, CHAR_VOCAB, (BATCH_SIZE, config.MAX_ENCODER_LEN)
    ), dtype=tf.int32)
    nwp_y = tf.constant(np.random.randint(
        0, WORD_VOCAB, (BATCH_SIZE,)
    ), dtype=tf.int32)

    # Identify variables by LAYER names (Keras 3.x variable names are minimal)
    encoder_layer_prefixes = ('char_embedding', 'kkc_enc_')
    nwp_layer_prefixes = ('nwp_',)
    decoder_layer_prefixes = ('kkc_dec_', 'kkc_state_', 'kkc_attention',
                              'kkc_concat', 'kkc_norm', 'kkc_proj', 'kkc_output')

    encoder_var_ids = set()
    nwp_var_ids = set()
    decoder_var_ids = set()

    for layer in model.layers:
        for v in layer.trainable_variables:
            vid = id(v)
            if any(layer.name.startswith(p) for p in encoder_layer_prefixes):
                encoder_var_ids.add(vid)
            elif any(layer.name.startswith(p) for p in nwp_layer_prefixes):
                nwp_var_ids.add(vid)
            elif any(layer.name.startswith(p) for p in decoder_layer_prefixes):
                decoder_var_ids.add(vid)

    print(f"  Encoder vars: {len(encoder_var_ids)}")
    print(f"  NWP head vars: {len(nwp_var_ids)}")
    print(f"  KKC decoder vars: {len(decoder_var_ids)}")

    # Test: KKC-only gradients
    with tf.GradientTape() as tape:
        kkc_pred, _ = model([kkc_enc, kkc_dec_in], training=True)
        kkc_loss = kkc_loss_fn(kkc_dec_tgt, kkc_pred)
    kkc_grads = tape.gradient(kkc_loss, model.trainable_variables)
    kkc_encoder_grad_count = sum(
        1 for g, v in zip(kkc_grads, model.trainable_variables)
        if g is not None and id(v) in encoder_var_ids
    )

    # Test: NWP-only gradients
    dec_dummy = tf.zeros((BATCH_SIZE, config.MAX_DECODER_LEN), dtype=tf.int32)
    with tf.GradientTape() as tape:
        _, nwp_pred = model([nwp_enc_chars, dec_dummy], training=True)
        nwp_loss = nwp_loss_fn(nwp_y, nwp_pred)
    nwp_grads = tape.gradient(nwp_loss, model.trainable_variables)
    nwp_encoder_grad_count = sum(
        1 for g, v in zip(nwp_grads, model.trainable_variables)
        if g is not None and id(v) in encoder_var_ids
    )

    print(f"\n  KKC → encoder grads: {kkc_encoder_grad_count}")
    print(f"  NWP → encoder grads: {nwp_encoder_grad_count}")

    assert kkc_encoder_grad_count > 0, "KKC should produce encoder gradients"
    assert nwp_encoder_grad_count > 0, "NWP should produce encoder gradients"
    assert kkc_encoder_grad_count == nwp_encoder_grad_count, \
        "Both tasks should gradient ALL encoder vars"

    print(f"✅ Shared encoder gets gradients from BOTH KKC and NWP!")


# ===========================================================
# MAIN
# ===========================================================
if __name__ == '__main__':
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPU: {tf.config.list_physical_devices('GPU')}")
    print(f"Config: ENCODER_LEN={config.MAX_ENCODER_LEN}, "
          f"DECODER_LEN={config.MAX_DECODER_LEN}")
    print(f"Config: GRU_UNITS={config.GRU_UNITS}, "
          f"ENC_LAYERS={config.NUM_ENCODER_LAYERS}, "
          f"DEC_LAYERS={config.NUM_DECODER_LAYERS}")

    model = test_model_build()
    test_forward_pass(model)
    test_kkc_only(model)
    test_nwp_only(model)
    test_gradient_tape(model)
    test_multi_steps(model)
    test_shared_encoder_grads()

    separator("ALL 7 TESTS PASSED ✅")
