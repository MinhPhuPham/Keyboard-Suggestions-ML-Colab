"""
Training utilities for Multi-Task GRU.
Dataset creation, callbacks, and training orchestration.
"""
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
)

from . import config
from .model import masked_sparse_ce, masked_accuracy


# ===========================================================
# DATASET CREATION
# ===========================================================

def create_datasets(kkc_data, nwp_data, batch_size=None):
    """Create interleaved TF datasets for multi-task training.

    For each batch, both heads receive data:
    - KKC: encoder_input + decoder_input → decoder_target
    - NWP: nwp_input (word IDs) → word prediction

    The model has 3 inputs:
    - encoder_input: char IDs for KKC encoder
    - decoder_input: char IDs for KKC decoder
    - nwp_input: word IDs for NWP head

    Args:
        kkc_data: (enc_mmap, dec_in_mmap, dec_tgt_mmap)
        nwp_data: (x_mmap, y_mmap, word_to_idx)

    Returns:
        (kkc_train_ds, kkc_val_ds, nwp_train_ds, nwp_val_ds), info dict
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    enc_mmap, dec_in_mmap, dec_tgt_mmap = kkc_data
    nwp_x_mmap, nwp_y_mmap, _ = nwp_data

    n_kkc = len(enc_mmap)
    n_nwp = len(nwp_x_mmap)

    # Use the smaller dataset size to keep batches aligned
    n_shared = min(n_kkc, n_nwp)
    split = int(n_shared * 0.9)

    # Shuffle indices
    kkc_indices = np.random.permutation(n_kkc).astype(np.int32)
    nwp_indices = np.random.permutation(n_nwp).astype(np.int32)

    kkc_train_idx = kkc_indices[:int(n_kkc * 0.9)]
    kkc_val_idx = kkc_indices[int(n_kkc * 0.9):]
    nwp_train_idx = nwp_indices[:int(n_nwp * 0.9)]
    nwp_val_idx = nwp_indices[int(n_nwp * 0.9):]

    def make_kkc_generator(indices):
        """Generate KKC batches from memory-mapped arrays."""
        def gen():
            np.random.shuffle(indices)
            for i in indices:
                yield (
                    enc_mmap[i].astype(np.int32),
                    dec_in_mmap[i].astype(np.int32),
                    dec_tgt_mmap[i].astype(np.int32),
                )
        return gen

    def make_nwp_generator(indices):
        """Generate NWP batches from memory-mapped arrays."""
        def gen():
            np.random.shuffle(indices)
            for i in indices:
                yield (
                    nwp_x_mmap[i].astype(np.int32),
                    nwp_y_mmap[i].astype(np.int32),
                )
        return gen

    # KKC dataset
    kkc_output_sig = (
        tf.TensorSpec(shape=(config.MAX_ENCODER_LEN,), dtype=tf.int32),
        tf.TensorSpec(shape=(config.MAX_DECODER_LEN,), dtype=tf.int32),
        tf.TensorSpec(shape=(config.MAX_DECODER_LEN,), dtype=tf.int32),
    )
    kkc_train_ds = tf.data.Dataset.from_generator(
        make_kkc_generator(kkc_train_idx), output_signature=kkc_output_sig
    ).repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    kkc_val_ds = tf.data.Dataset.from_generator(
        make_kkc_generator(kkc_val_idx), output_signature=kkc_output_sig
    ).repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # NWP dataset
    nwp_output_sig = (
        tf.TensorSpec(shape=(config.MAX_WORD_CONTEXT,), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    nwp_train_ds = tf.data.Dataset.from_generator(
        make_nwp_generator(nwp_train_idx), output_signature=nwp_output_sig
    ).repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    nwp_val_ds = tf.data.Dataset.from_generator(
        make_nwp_generator(nwp_val_idx), output_signature=nwp_output_sig
    ).repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Steps per epoch
    kkc_train_steps = len(kkc_train_idx) // batch_size
    nwp_train_steps = len(nwp_train_idx) // batch_size
    kkc_val_steps = max(1, len(kkc_val_idx) // batch_size)
    nwp_val_steps = max(1, len(nwp_val_idx) // batch_size)

    # Use min steps so both heads contribute each epoch
    train_steps = min(kkc_train_steps, nwp_train_steps)
    val_steps = min(kkc_val_steps, nwp_val_steps)

    info = {
        'kkc_train': len(kkc_train_idx),
        'kkc_val': len(kkc_val_idx),
        'nwp_train': len(nwp_train_idx),
        'nwp_val': len(nwp_val_idx),
        'train_steps': train_steps,
        'val_steps': val_steps,
    }

    print(f"  KKC: train={info['kkc_train']:,}, val={info['kkc_val']:,}")
    print(f"  NWP: train={info['nwp_train']:,}, val={info['nwp_val']:,}")
    print(f"  Steps/epoch: {train_steps}, Val steps: {val_steps}")

    return (kkc_train_ds, kkc_val_ds, nwp_train_ds, nwp_val_ds), info


# ===========================================================
# CALLBACKS
# ===========================================================

def get_callbacks():
    """Standard training callbacks."""
    return [
        ModelCheckpoint(
            f'{config.MODEL_DIR}/best_multitask.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
    ]


# ===========================================================
# CUSTOM TRAINING LOOP
# ===========================================================

def _top_k_accuracy(y_true, y_pred, k=5):
    """Compute top-k accuracy (useful for large vocabs where top-1 is too strict)."""
    top_k = tf.math.in_top_k(
        targets=tf.cast(y_true, tf.int32),
        predictions=y_pred,
        k=k
    )
    return tf.reduce_mean(tf.cast(top_k, tf.float32))


def train_multitask(model, datasets, info, num_epochs=None):
    """Custom training loop for multi-task model.

    Uses SINGLE forward pass per step: feeds real KKC and NWP data
    simultaneously. Both heads are independent (no shared weights),
    so this is equivalent to training them separately but faster.

    IMPORTANT: Caller must wrap this function call inside
    strategy.scope() for GPU/TPU training.

    Args:
        model: compiled multi-task model
        datasets: (kkc_train_ds, kkc_val_ds, nwp_train_ds, nwp_val_ds)
        info: dict with train_steps, val_steps etc.
        num_epochs: override config.NUM_EPOCHS

    Returns:
        history dict with losses and metrics per epoch
    """
    if num_epochs is None:
        num_epochs = config.NUM_EPOCHS

    kkc_train_ds, kkc_val_ds, nwp_train_ds, nwp_val_ds = datasets
    train_steps = info['train_steps']
    val_steps = info['val_steps']

    kkc_train_iter = iter(kkc_train_ds)
    nwp_train_iter = iter(nwp_train_ds)
    kkc_val_iter = iter(kkc_val_ds)
    nwp_val_iter = iter(nwp_val_ds)

    optimizer = model.optimizer
    kkc_loss_fn = masked_sparse_ce
    nwp_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    history = {
        'loss': [], 'kkc_loss': [], 'nwp_loss': [],
        'kkc_accuracy': [], 'nwp_accuracy': [], 'nwp_top5_accuracy': [],
        'val_loss': [], 'val_kkc_loss': [], 'val_nwp_loss': [],
        'val_kkc_accuracy': [], 'val_nwp_accuracy': [], 'val_nwp_top5_accuracy': [],
    }

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5

    # Force unbuffered stdout (skip on Colab where OutStream lacks reconfigure)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass  # Colab/IPython OutStream — already unbuffered

    for epoch in range(num_epochs):
        print(f"\n{'='*60}", flush=True)
        print(f"  Epoch {epoch+1}/{num_epochs}", flush=True)
        print(f"{'='*60}", flush=True)
        # --- Training ---
        epoch_kkc_loss = tf.keras.metrics.Mean()
        epoch_nwp_loss = tf.keras.metrics.Mean()
        epoch_total_loss = tf.keras.metrics.Mean()
        epoch_kkc_acc = tf.keras.metrics.Mean()  # Masked accuracy (ignores PAD)
        epoch_nwp_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        epoch_nwp_top5 = tf.keras.metrics.Mean()

        for step in range(train_steps):
            # Get batches from both heads
            kkc_enc, kkc_dec_in, kkc_dec_tgt = next(kkc_train_iter)
            nwp_x, nwp_y = next(nwp_train_iter)

            # Align batch sizes (KKC and NWP may have different batch sizes)
            min_batch = tf.minimum(tf.shape(kkc_enc)[0], tf.shape(nwp_x)[0])
            kkc_enc = kkc_enc[:min_batch]
            kkc_dec_in = kkc_dec_in[:min_batch]
            kkc_dec_tgt = kkc_dec_tgt[:min_batch]
            nwp_x = nwp_x[:min_batch]
            nwp_y = nwp_y[:min_batch]

            with tf.GradientTape() as tape:
                # SINGLE forward pass with REAL data for both heads
                kkc_pred, nwp_pred = model(
                    {
                        'encoder_input': kkc_enc,
                        'decoder_input': kkc_dec_in,
                        'nwp_input': nwp_x,
                    },
                    training=True
                )

                kkc_loss = kkc_loss_fn(kkc_dec_tgt, kkc_pred)
                nwp_loss = nwp_loss_fn(nwp_y, nwp_pred)

                # Combined loss (equal weights)
                total_loss = (
                    config.KKC_LOSS_WEIGHT * kkc_loss +
                    config.NWP_LOSS_WEIGHT * nwp_loss
                )

            # Update weights (optimizer.clipnorm handles per-variable clipping)
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track metrics
            epoch_kkc_loss.update_state(kkc_loss)
            epoch_nwp_loss.update_state(nwp_loss)
            epoch_total_loss.update_state(total_loss)
            # Masked accuracy for KKC (ignores PAD tokens)
            epoch_kkc_acc.update_state(masked_accuracy(kkc_dec_tgt, kkc_pred))
            epoch_nwp_acc.update_state(nwp_y, nwp_pred)
            epoch_nwp_top5.update_state(_top_k_accuracy(nwp_y, nwp_pred, k=5))

            if step % 50 == 0:
                print(
                    f"  step {step}/{train_steps} | "
                    f"loss={total_loss:.4f} "
                    f"kkc={kkc_loss:.4f} nwp={nwp_loss:.4f}",
                    flush=True
                )

        # --- Validation ---
        val_kkc_loss = tf.keras.metrics.Mean()
        val_nwp_loss = tf.keras.metrics.Mean()
        val_total_loss = tf.keras.metrics.Mean()
        val_kkc_acc = tf.keras.metrics.Mean()  # Masked accuracy (ignores PAD)
        val_nwp_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        val_nwp_top5 = tf.keras.metrics.Mean()

        for _ in range(val_steps):
            kkc_enc, kkc_dec_in, kkc_dec_tgt = next(kkc_val_iter)
            nwp_x, nwp_y = next(nwp_val_iter)

            # Align batch sizes
            min_batch = tf.minimum(tf.shape(kkc_enc)[0], tf.shape(nwp_x)[0])
            kkc_enc = kkc_enc[:min_batch]
            kkc_dec_in = kkc_dec_in[:min_batch]
            kkc_dec_tgt = kkc_dec_tgt[:min_batch]
            nwp_x = nwp_x[:min_batch]
            nwp_y = nwp_y[:min_batch]

            # SINGLE forward pass for validation
            kkc_pred, nwp_pred = model(
                {
                    'encoder_input': kkc_enc,
                    'decoder_input': kkc_dec_in,
                    'nwp_input': nwp_x,
                },
                training=False
            )

            kkc_l = kkc_loss_fn(kkc_dec_tgt, kkc_pred)
            nwp_l = nwp_loss_fn(nwp_y, nwp_pred)
            total_l = config.KKC_LOSS_WEIGHT * kkc_l + config.NWP_LOSS_WEIGHT * nwp_l

            val_kkc_loss.update_state(kkc_l)
            val_nwp_loss.update_state(nwp_l)
            val_total_loss.update_state(total_l)
            val_kkc_acc.update_state(masked_accuracy(kkc_dec_tgt, kkc_pred))
            val_nwp_acc.update_state(nwp_y, nwp_pred)
            val_nwp_top5.update_state(_top_k_accuracy(nwp_y, nwp_pred, k=5))

        # Record history
        history['loss'].append(float(epoch_total_loss.result()))
        history['kkc_loss'].append(float(epoch_kkc_loss.result()))
        history['nwp_loss'].append(float(epoch_nwp_loss.result()))
        history['kkc_accuracy'].append(float(epoch_kkc_acc.result()))
        history['nwp_accuracy'].append(float(epoch_nwp_acc.result()))
        history['nwp_top5_accuracy'].append(float(epoch_nwp_top5.result()))
        history['val_loss'].append(float(val_total_loss.result()))
        history['val_kkc_loss'].append(float(val_kkc_loss.result()))
        history['val_nwp_loss'].append(float(val_nwp_loss.result()))
        history['val_kkc_accuracy'].append(float(val_kkc_acc.result()))
        history['val_nwp_accuracy'].append(float(val_nwp_acc.result()))
        history['val_nwp_top5_accuracy'].append(float(val_nwp_top5.result()))

        vl = history['val_loss'][-1]
        print(
            f"\n✅ Epoch {epoch+1}/{num_epochs}\n"
            f"  Train: loss={history['loss'][-1]:.4f} "
            f"kkc_loss={history['kkc_loss'][-1]:.4f} "
            f"nwp_loss={history['nwp_loss'][-1]:.4f}\n"
            f"  Val:   loss={vl:.4f} "
            f"kkc_loss={history['val_kkc_loss'][-1]:.4f} "
            f"nwp_loss={history['val_nwp_loss'][-1]:.4f}\n"
            f"  KKC:   acc={history['kkc_accuracy'][-1]*100:.1f}% "
            f"val_acc={history['val_kkc_accuracy'][-1]*100:.1f}%\n"
            f"  NWP:   acc={history['nwp_accuracy'][-1]*100:.2f}% "
            f"top5={history['nwp_top5_accuracy'][-1]*100:.1f}% "
            f"val_acc={history['val_nwp_accuracy'][-1]*100:.2f}% "
            f"val_top5={history['val_nwp_top5_accuracy'][-1]*100:.1f}%",
            flush=True
        )

        # Early stopping + save best
        if vl < best_val_loss:
            best_val_loss = vl
            patience_counter = 0
            model.save(f'{config.MODEL_DIR}/best_multitask.keras')
            print(f"  💾 Best model saved (val_loss={vl:.4f})", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹ Early stopping at epoch {epoch+1}", flush=True)
                # Load best weights
                model.load_weights(f'{config.MODEL_DIR}/best_multitask.keras')
                break

        # Reduce LR
        if patience_counter >= 2:
            old_lr = float(optimizer.learning_rate)
            new_lr = max(old_lr * 0.5, 1e-6)
            optimizer.learning_rate.assign(new_lr)
            if new_lr != old_lr:
                print(f"  📉 LR: {old_lr:.2e} → {new_lr:.2e}", flush=True)

    return history


# ===========================================================
# SHARED ENCODER TRAINING (v2)
# ===========================================================

def create_shared_datasets(kkc_data, nwp_char_data, batch_size=None):
    """Create TF datasets for shared encoder training.

    Uses from_tensor_slices (fast C++ pipeline) instead of
    from_generator (slow Python GIL bottleneck).

    Args:
        kkc_data: (enc_mmap, dec_in_mmap, dec_tgt_mmap)
        nwp_char_data: (char_x_mmap, y_mmap, word_to_idx)

    Returns:
        (kkc_train_ds, kkc_val_ds, nwp_train_ds, nwp_val_ds), info dict
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    enc_mmap, dec_in_mmap, dec_tgt_mmap = kkc_data
    nwp_char_x_mmap, nwp_y_mmap, _ = nwp_char_data

    n_kkc = len(enc_mmap)
    n_nwp = len(nwp_char_x_mmap)

    # Shuffle indices for train/val split
    kkc_indices = np.random.permutation(n_kkc)
    nwp_indices = np.random.permutation(n_nwp)

    kkc_split = int(n_kkc * 0.9)
    nwp_split = int(n_nwp * 0.9)

    kkc_train_idx = kkc_indices[:kkc_split]
    kkc_val_idx = kkc_indices[kkc_split:]
    nwp_train_idx = nwp_indices[:nwp_split]
    nwp_val_idx = nwp_indices[nwp_split:]

    # Shuffle buffer: large enough for good randomness, small enough for RAM
    SHUFFLE_BUFFER = min(100_000, n_kkc // 2)

    def make_kkc_ds(indices, shuffle=True):
        """Create KKC dataset from pre-shuffled indices using tensor slices."""
        ds = tf.data.Dataset.from_tensor_slices((
            enc_mmap[indices].astype(np.int32),
            dec_in_mmap[indices].astype(np.int32),
            dec_tgt_mmap[indices].astype(np.int32),
        ))
        if shuffle:
            ds = ds.shuffle(SHUFFLE_BUFFER, reshuffle_each_iteration=True)
        return ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE).repeat()

    def make_nwp_ds(indices, shuffle=True):
        """Create NWP dataset from pre-shuffled indices using tensor slices."""
        ds = tf.data.Dataset.from_tensor_slices((
            nwp_char_x_mmap[indices].astype(np.int32),
            nwp_y_mmap[indices].astype(np.int32),
        ))
        if shuffle:
            ds = ds.shuffle(SHUFFLE_BUFFER, reshuffle_each_iteration=True)
        return ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE).repeat()

    kkc_train_ds = make_kkc_ds(kkc_train_idx, shuffle=True)
    kkc_val_ds = make_kkc_ds(kkc_val_idx, shuffle=False)
    nwp_train_ds = make_nwp_ds(nwp_train_idx, shuffle=True)
    nwp_val_ds = make_nwp_ds(nwp_val_idx, shuffle=False)

    # Steps per epoch
    kkc_train_steps = len(kkc_train_idx) // batch_size
    nwp_train_steps = len(nwp_train_idx) // batch_size
    kkc_val_steps = max(1, len(kkc_val_idx) // batch_size)
    nwp_val_steps = max(1, len(nwp_val_idx) // batch_size)

    train_steps = min(kkc_train_steps, nwp_train_steps)
    val_steps = min(kkc_val_steps, nwp_val_steps)

    info = {
        'kkc_train': len(kkc_train_idx),
        'kkc_val': len(kkc_val_idx),
        'nwp_train': len(nwp_train_idx),
        'nwp_val': len(nwp_val_idx),
        'train_steps': train_steps,
        'val_steps': val_steps,
    }

    print(f"  KKC: train={info['kkc_train']:,}, val={info['kkc_val']:,}")
    print(f"  NWP: train={info['nwp_train']:,}, val={info['nwp_val']:,}")
    print(f"  Steps/epoch: {train_steps} (batch={batch_size}), Val steps: {val_steps}")

    return (kkc_train_ds, kkc_val_ds, nwp_train_ds, nwp_val_ds), info


def train_shared_multitask(model, datasets, info, num_epochs=None):
    """Custom training loop for shared encoder model (v2).

    Optimized with @tf.function compiled train/val steps for
    graph-mode execution (no Python overhead per step).

    Each step does TWO forward passes through the shared encoder:
    1. KKC data → shared encoder → KKC decoder → kkc_loss
    2. NWP char context → shared encoder → NWP head → nwp_loss

    Args:
        model: shared multitask model (2 inputs, 2 outputs)
        datasets: (kkc_train_ds, kkc_val_ds, nwp_train_ds, nwp_val_ds)
        info: dict with train_steps, val_steps etc.
        num_epochs: override config.NUM_EPOCHS
    """
    if num_epochs is None:
        num_epochs = config.NUM_EPOCHS

    kkc_train_ds, kkc_val_ds, nwp_train_ds, nwp_val_ds = datasets
    train_steps = info['train_steps']
    val_steps = info['val_steps']

    kkc_train_iter = iter(kkc_train_ds)
    nwp_train_iter = iter(nwp_train_ds)
    kkc_val_iter = iter(kkc_val_ds)
    nwp_val_iter = iter(nwp_val_ds)

    optimizer = model.optimizer
    kkc_loss_fn = masked_sparse_ce
    nwp_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Auto-detect mixed precision (FP16) and wrap optimizer for loss scaling
    use_fp16 = tf.keras.mixed_precision.global_policy().name == 'mixed_float16'
    if use_fp16:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        print("  🔧 Mixed precision: FP16 compute + FP32 output + loss scaling")

    # Pre-create decoder dummy (reused every NWP step)
    dec_dummy_len = config.MAX_DECODER_LEN
    kkc_w = config.KKC_LOSS_WEIGHT
    nwp_w = config.NWP_LOSS_WEIGHT

    # ------- Compiled train step (runs on GPU, no Python overhead) -------
    @tf.function
    def train_step(kkc_enc, kkc_dec_in, kkc_dec_tgt, nwp_enc_chars, nwp_y):
        with tf.GradientTape() as tape:
            # Pass 1: KKC
            kkc_pred, _ = model([kkc_enc, kkc_dec_in], training=True)
            kkc_loss = kkc_loss_fn(kkc_dec_tgt, kkc_pred)

            # Pass 2: NWP (decoder gets zeros — output ignored)
            nwp_batch = tf.shape(nwp_enc_chars)[0]
            dec_dummy = tf.zeros((nwp_batch, dec_dummy_len), dtype=tf.int32)
            _, nwp_pred = model([nwp_enc_chars, dec_dummy], training=True)
            nwp_loss = nwp_loss_fn(nwp_y, nwp_pred)

            total_loss = kkc_w * kkc_loss + nwp_w * nwp_loss

            # FP16: scale loss UP to prevent gradient underflow in float16
            grad_loss = optimizer.get_scaled_loss(total_loss) if use_fp16 else total_loss

        grads = tape.gradient(grad_loss, model.trainable_variables)
        # FP16: scale gradients back DOWN before applying
        if use_fp16:
            grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        kkc_acc = masked_accuracy(kkc_dec_tgt, kkc_pred)
        nwp_top5 = _top_k_accuracy(nwp_y, nwp_pred, k=5)
        return total_loss, kkc_loss, nwp_loss, kkc_acc, nwp_y, nwp_pred, nwp_top5

    # ------- Compiled validation step -------
    @tf.function
    def val_step(kkc_enc, kkc_dec_in, kkc_dec_tgt, nwp_enc_chars, nwp_y):
        kkc_pred, _ = model([kkc_enc, kkc_dec_in], training=False)
        kkc_loss = kkc_loss_fn(kkc_dec_tgt, kkc_pred)

        nwp_batch = tf.shape(nwp_enc_chars)[0]
        dec_dummy = tf.zeros((nwp_batch, dec_dummy_len), dtype=tf.int32)
        _, nwp_pred = model([nwp_enc_chars, dec_dummy], training=False)
        nwp_loss = nwp_loss_fn(nwp_y, nwp_pred)

        total_loss = kkc_w * kkc_loss + nwp_w * nwp_loss
        kkc_acc = masked_accuracy(kkc_dec_tgt, kkc_pred)
        nwp_top5 = _top_k_accuracy(nwp_y, nwp_pred, k=5)
        return total_loss, kkc_loss, nwp_loss, kkc_acc, nwp_y, nwp_pred, nwp_top5

    history = {
        'loss': [], 'kkc_loss': [], 'nwp_loss': [],
        'kkc_accuracy': [], 'nwp_accuracy': [], 'nwp_top5_accuracy': [],
        'val_loss': [], 'val_kkc_loss': [], 'val_nwp_loss': [],
        'val_kkc_accuracy': [], 'val_nwp_accuracy': [], 'val_nwp_top5_accuracy': [],
    }

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5

    # Force unbuffered stdout
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    print("  ⚡ First step will be slow (TF graph compilation)...", flush=True)

    for epoch in range(num_epochs):
        print(f"\n{'='*60}", flush=True)
        print(f"  Epoch {epoch+1}/{num_epochs}", flush=True)
        print(f"{'='*60}", flush=True)

        # --- Training ---
        epoch_kkc_loss = tf.keras.metrics.Mean()
        epoch_nwp_loss = tf.keras.metrics.Mean()
        epoch_total_loss = tf.keras.metrics.Mean()
        epoch_kkc_acc = tf.keras.metrics.Mean()
        epoch_nwp_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        epoch_nwp_top5 = tf.keras.metrics.Mean()

        for step in range(train_steps):
            kkc_enc, kkc_dec_in, kkc_dec_tgt = next(kkc_train_iter)
            nwp_enc_chars, nwp_y = next(nwp_train_iter)

            total_loss, kkc_loss, nwp_loss, kkc_acc, nwp_y_out, nwp_pred, nwp_top5 = \
                train_step(kkc_enc, kkc_dec_in, kkc_dec_tgt, nwp_enc_chars, nwp_y)

            # Track metrics
            epoch_kkc_loss.update_state(kkc_loss)
            epoch_nwp_loss.update_state(nwp_loss)
            epoch_total_loss.update_state(total_loss)
            epoch_kkc_acc.update_state(kkc_acc)
            epoch_nwp_acc.update_state(nwp_y_out, nwp_pred)
            epoch_nwp_top5.update_state(nwp_top5)

            if step % 50 == 0:
                print(
                    f"  step {step}/{train_steps} | "
                    f"loss={total_loss:.4f} "
                    f"kkc={kkc_loss:.4f} nwp={nwp_loss:.4f}",
                    flush=True
                )

        # --- Validation ---
        val_kkc_loss = tf.keras.metrics.Mean()
        val_nwp_loss = tf.keras.metrics.Mean()
        val_total_loss = tf.keras.metrics.Mean()
        val_kkc_acc = tf.keras.metrics.Mean()
        val_nwp_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        val_nwp_top5 = tf.keras.metrics.Mean()

        for _ in range(val_steps):
            kkc_enc, kkc_dec_in, kkc_dec_tgt = next(kkc_val_iter)
            nwp_enc_chars, nwp_y = next(nwp_val_iter)

            total_l, kkc_l, nwp_l, kkc_acc_v, nwp_y_out, nwp_pred, nwp_top5 = \
                val_step(kkc_enc, kkc_dec_in, kkc_dec_tgt, nwp_enc_chars, nwp_y)

            val_kkc_loss.update_state(kkc_l)
            val_nwp_loss.update_state(nwp_l)
            val_total_loss.update_state(total_l)
            val_kkc_acc.update_state(kkc_acc_v)
            val_nwp_acc.update_state(nwp_y_out, nwp_pred)
            val_nwp_top5.update_state(nwp_top5)

        # Record history
        history['loss'].append(float(epoch_total_loss.result()))
        history['kkc_loss'].append(float(epoch_kkc_loss.result()))
        history['nwp_loss'].append(float(epoch_nwp_loss.result()))
        history['kkc_accuracy'].append(float(epoch_kkc_acc.result()))
        history['nwp_accuracy'].append(float(epoch_nwp_acc.result()))
        history['nwp_top5_accuracy'].append(float(epoch_nwp_top5.result()))
        history['val_loss'].append(float(val_total_loss.result()))
        history['val_kkc_loss'].append(float(val_kkc_loss.result()))
        history['val_nwp_loss'].append(float(val_nwp_loss.result()))
        history['val_kkc_accuracy'].append(float(val_kkc_acc.result()))
        history['val_nwp_accuracy'].append(float(val_nwp_acc.result()))
        history['val_nwp_top5_accuracy'].append(float(val_nwp_top5.result()))

        vl = history['val_loss'][-1]
        print(
            f"\n✅ Epoch {epoch+1}/{num_epochs}\n"
            f"  Train: loss={history['loss'][-1]:.4f} "
            f"kkc_loss={history['kkc_loss'][-1]:.4f} "
            f"nwp_loss={history['nwp_loss'][-1]:.4f}\n"
            f"  Val:   loss={vl:.4f} "
            f"kkc_loss={history['val_kkc_loss'][-1]:.4f} "
            f"nwp_loss={history['val_nwp_loss'][-1]:.4f}\n"
            f"  KKC:   acc={history['kkc_accuracy'][-1]*100:.1f}% "
            f"val_acc={history['val_kkc_accuracy'][-1]*100:.1f}%\n"
            f"  NWP:   acc={history['nwp_accuracy'][-1]*100:.2f}% "
            f"top5={history['nwp_top5_accuracy'][-1]*100:.1f}% "
            f"val_acc={history['val_nwp_accuracy'][-1]*100:.2f}% "
            f"val_top5={history['val_nwp_top5_accuracy'][-1]*100:.1f}%",
            flush=True
        )

        # Early stopping + save best
        if vl < best_val_loss:
            best_val_loss = vl
            patience_counter = 0
            model.save(f'{config.MODEL_DIR}/best_shared_multitask.keras')
            print(f"  💾 Best model saved (val_loss={vl:.4f})", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹ Early stopping at epoch {epoch+1}", flush=True)
                model.load_weights(
                    f'{config.MODEL_DIR}/best_shared_multitask.keras'
                )
                break

        # Reduce LR
        if patience_counter >= 2:
            old_lr = float(optimizer.learning_rate)
            new_lr = max(old_lr * 0.5, 1e-6)
            optimizer.learning_rate.assign(new_lr)
            if new_lr != old_lr:
                print(f"  📉 LR: {old_lr:.2e} → {new_lr:.2e}", flush=True)

    return history
