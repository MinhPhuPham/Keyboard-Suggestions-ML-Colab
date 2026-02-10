"""
Training utilities for Multi-Task GRU.
Dataset creation, callbacks, and training orchestration.
"""
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

def train_multitask(model, datasets, info, num_epochs=None):
    """Custom training loop for multi-task model.

    Alternates between KKC and NWP batches each step,
    combined through shared encoder.

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
        'kkc_accuracy': [], 'nwp_accuracy': [],
        'val_loss': [], 'val_kkc_loss': [], 'val_nwp_loss': [],
        'val_kkc_accuracy': [], 'val_nwp_accuracy': [],
    }

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5

    for epoch in range(num_epochs):
        # --- Training ---
        epoch_kkc_loss = tf.keras.metrics.Mean()
        epoch_nwp_loss = tf.keras.metrics.Mean()
        epoch_total_loss = tf.keras.metrics.Mean()
        epoch_kkc_acc = tf.keras.metrics.Mean()  # Masked accuracy (ignores PAD)
        epoch_nwp_acc = tf.keras.metrics.SparseCategoricalAccuracy()

        for step in range(train_steps):
            # Get batches
            kkc_enc, kkc_dec_in, kkc_dec_tgt = next(kkc_train_iter)
            nwp_x, nwp_y = next(nwp_train_iter)

            with tf.GradientTape() as tape:
                # KKC forward pass (NWP input = zeros, not used by NWP head)
                nwp_dummy = tf.zeros(
                    (tf.shape(kkc_enc)[0], config.MAX_WORD_CONTEXT),
                    dtype=tf.int32
                )
                kkc_pred, _ = model(
                    {
                        'encoder_input': kkc_enc,
                        'decoder_input': kkc_dec_in,
                        'nwp_input': nwp_dummy,
                    },
                    training=True
                )
                kkc_loss = kkc_loss_fn(kkc_dec_tgt, kkc_pred)

                # NWP forward pass (KKC inputs = zeros, not used by KKC head)
                enc_dummy = tf.zeros(
                    (tf.shape(nwp_x)[0], config.MAX_ENCODER_LEN),
                    dtype=tf.int32
                )
                dec_dummy = tf.zeros(
                    (tf.shape(nwp_x)[0], config.MAX_DECODER_LEN),
                    dtype=tf.int32
                )
                _, nwp_pred = model(
                    {
                        'encoder_input': enc_dummy,
                        'decoder_input': dec_dummy,
                        'nwp_input': nwp_x,
                    },
                    training=True
                )
                nwp_loss = nwp_loss_fn(nwp_y, nwp_pred)

                # Combined loss
                total_loss = (
                    config.KKC_LOSS_WEIGHT * kkc_loss +
                    config.NWP_LOSS_WEIGHT * nwp_loss
                )

            # Update weights
            grads = tape.gradient(total_loss, model.trainable_variables)
            # Clip gradients
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track metrics
            epoch_kkc_loss.update_state(kkc_loss)
            epoch_nwp_loss.update_state(nwp_loss)
            epoch_total_loss.update_state(total_loss)
            # Masked accuracy for KKC (ignores PAD tokens)
            epoch_kkc_acc.update_state(masked_accuracy(kkc_dec_tgt, kkc_pred))
            epoch_nwp_acc.update_state(nwp_y, nwp_pred)

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

        for _ in range(val_steps):
            kkc_enc, kkc_dec_in, kkc_dec_tgt = next(kkc_val_iter)
            nwp_x, nwp_y = next(nwp_val_iter)

            # KKC validation
            nwp_dummy = tf.zeros(
                (tf.shape(kkc_enc)[0], config.MAX_WORD_CONTEXT),
                dtype=tf.int32
            )
            kkc_pred, _ = model(
                {
                    'encoder_input': kkc_enc,
                    'decoder_input': kkc_dec_in,
                    'nwp_input': nwp_dummy,
                },
                training=False
            )

            # NWP validation
            enc_dummy = tf.zeros(
                (tf.shape(nwp_x)[0], config.MAX_ENCODER_LEN),
                dtype=tf.int32
            )
            dec_dummy = tf.zeros(
                (tf.shape(nwp_x)[0], config.MAX_DECODER_LEN),
                dtype=tf.int32
            )
            _, nwp_pred = model(
                {
                    'encoder_input': enc_dummy,
                    'decoder_input': dec_dummy,
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

        # Record history
        history['loss'].append(float(epoch_total_loss.result()))
        history['kkc_loss'].append(float(epoch_kkc_loss.result()))
        history['nwp_loss'].append(float(epoch_nwp_loss.result()))
        history['kkc_accuracy'].append(float(epoch_kkc_acc.result()))
        history['nwp_accuracy'].append(float(epoch_nwp_acc.result()))
        history['val_loss'].append(float(val_total_loss.result()))
        history['val_kkc_loss'].append(float(val_kkc_loss.result()))
        history['val_nwp_loss'].append(float(val_nwp_loss.result()))
        history['val_kkc_accuracy'].append(float(val_kkc_acc.result()))
        history['val_nwp_accuracy'].append(float(val_nwp_acc.result()))

        vl = history['val_loss'][-1]
        print(
            f"\nEpoch {epoch+1}/{num_epochs} | "
            f"loss={history['loss'][-1]:.4f} val_loss={vl:.4f} | "
            f"kkc_acc={history['kkc_accuracy'][-1]*100:.1f}% "
            f"nwp_acc={history['nwp_accuracy'][-1]*100:.1f}% | "
            f"val_kkc_acc={history['val_kkc_accuracy'][-1]*100:.1f}% "
            f"val_nwp_acc={history['val_nwp_accuracy'][-1]*100:.1f}%"
        )

        # Early stopping + save best
        if vl < best_val_loss:
            best_val_loss = vl
            patience_counter = 0
            model.save(f'{config.MODEL_DIR}/best_multitask.keras')
            print(f"  💾 Best model saved (val_loss={vl:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹ Early stopping at epoch {epoch+1}")
                # Load best weights
                model.load_weights(f'{config.MODEL_DIR}/best_multitask.keras')
                break

        # Reduce LR
        if patience_counter >= 2:
            old_lr = float(optimizer.learning_rate)
            new_lr = max(old_lr * 0.5, 1e-6)
            optimizer.learning_rate.assign(new_lr)
            if new_lr != old_lr:
                print(f"  📉 LR: {old_lr:.2e} → {new_lr:.2e}")

    return history
