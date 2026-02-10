"""
Multi-Task GRU Model: KKC Head + NWP Head.

Architecture:
    KKC Path: Char Embedding → Bi-GRU Encoder → GRU Decoder + Attention → char vocab
    NWP Path: Word Embedding → Bi-GRU → Self-Attention → Context GRU → word vocab

Both heads are combined in a single Model with 3 inputs:
    - encoder_input: char IDs for KKC encoder (batch, MAX_ENCODER_LEN)
    - decoder_input: char IDs for KKC decoder (batch, MAX_DECODER_LEN)
    - nwp_input:     word IDs for NWP head    (batch, MAX_WORD_CONTEXT)

KKC architecture matches original KKC notebook (train_gru_kana_kanji.ipynb):
    - Shared embedding between encoder and decoder
    - LayerNorm after every Bi-GRU layer
    - LayerNorm after each decoder GRU layer
    - Dropout on decoder embedding
    - Dropout(0.2) before projection Dense

NWP architecture matches original NWP notebook (train_gru_next_word.ipynb):
    - Word embedding → Bi-GRU(return_sequences) → Self-Attention
    - Concatenate + LayerNorm → Context GRU → Dropout → Dense output
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, GRU, Dense, Dropout,
    Bidirectional, Attention, Concatenate,
    LayerNormalization, RepeatVector, GlobalAveragePooling1D,
)

from . import config


# ===========================================================
# KKC ENCODER (Char-level Bi-GRU)
# ===========================================================

def build_kkc_encoder(char_vocab_size, name_prefix='kkc'):
    """Build char-level Bi-GRU encoder for KKC head.

    Input:  (batch, MAX_ENCODER_LEN) — char IDs
    Output: (batch, MAX_ENCODER_LEN, GRU_UNITS*2) — encoder states

    Architecture matches original KKC notebook:
    - Shared embedding (also used by decoder)
    - Dropout after embedding
    - Stacked Bi-GRU with LayerNorm after EVERY layer

    Returns:
        encoder_input, encoder_output, encoder_state, char_embedding
    """
    encoder_input = Input(
        shape=(config.MAX_ENCODER_LEN,),
        name='encoder_input'
    )

    # Char embedding (shared between encoder and decoder — matches original)
    char_embedding = Embedding(
        char_vocab_size,
        config.EMBEDDING_DIM,
        mask_zero=False,
        name='char_embedding'
    )

    x = Dropout(config.DROPOUT_RATE, name=f'{name_prefix}_emb_drop')(
        char_embedding(encoder_input)
    )

    # Stacked Bi-GRU layers with LayerNorm after EVERY layer
    # NOTE: return_state=True is broken in Keras 3 / TF 2.16+ (produces
    # 1D state tensors). All layers use return_state=False instead.
    for i in range(config.NUM_ENCODER_LAYERS):
        x = Bidirectional(
            GRU(
                config.GRU_UNITS,
                return_sequences=True,
                name=f'{name_prefix}_enc_gru_{i}'
            ),
            name=f'{name_prefix}_enc_bigru_{i}'
        )(x)
        x = LayerNormalization(name=f'{name_prefix}_enc_norm_{i}')(x)

    encoder_output = x  # (batch, seq_len, GRU_UNITS*2)

    # Derive encoder state from encoder output using pooling.
    # GlobalAveragePooling1D is a proper Keras layer that guarantees
    # (batch, features) output — unlike return_state or tensor slicing
    # which break in Keras 3 / TF 2.16+ (produce 1D tensors).
    encoder_state = GlobalAveragePooling1D(
        name=f'{name_prefix}_enc_pool'
    )(encoder_output)  # (batch, GRU_UNITS*2)
    encoder_state = Dense(
        config.GRU_UNITS * 2,
        activation='tanh',
        name=f'{name_prefix}_state_proj'
    )(encoder_state)  # (batch, GRU_UNITS*2)

    return encoder_input, encoder_output, encoder_state, char_embedding


# ===========================================================
# KKC HEAD (GRU Decoder + Attention)
# ===========================================================

def build_kkc_head(encoder_output, encoder_state, char_vocab_size,
                   char_embedding):
    """Build KKC decoder head: GRU + Attention → char output.

    Input:  decoder_input (batch, MAX_DECODER_LEN) — char IDs with BOS
    Output: (batch, MAX_DECODER_LEN, char_vocab_size) — char probabilities

    Architecture matches original KKC notebook:
    - Uses SHARED char embedding (same layer as encoder)
    - Dropout on decoder embedding
    - LayerNorm after each decoder GRU
    - Only first GRU layer receives encoder state
    - Dropout(0.2) before projection Dense
    """
    decoder_input = Input(
        shape=(config.MAX_DECODER_LEN,),
        name='decoder_input'
    )

    # Shared char embedding + dropout (matches original: Dropout(0.1)(emb(dec_in)))
    dec_out = Dropout(
        config.DROPOUT_RATE, name='kkc_dec_emb_drop'
    )(char_embedding(decoder_input))

    # Inject encoder state via context injection (avoids initial_state bug
    # in Keras 3 / TF 2.16+ where GRU receives 1D state instead of 2D).
    # RepeatVector tiles encoder_state across all decoder timesteps,
    # then Concatenate merges it with decoder embeddings.
    tiled_state = RepeatVector(
        config.MAX_DECODER_LEN, name='kkc_state_tile'
    )(encoder_state)  # (batch, dec_len, GRU_UNITS*2)
    dec_out = Concatenate(
        name='kkc_dec_state_concat'
    )([dec_out, tiled_state])  # (batch, dec_len, EMBEDDING_DIM + GRU_UNITS*2)

    # Stacked GRU decoder with LayerNorm after each layer
    for i in range(config.NUM_DECODER_LAYERS):
        gru_layer = GRU(
            config.GRU_UNITS * 2,
            return_sequences=True,
            name=f'kkc_dec_gru_{i}',
        )
        dec_out = gru_layer(dec_out)

        # LayerNorm after each decoder GRU (matches original)
        dec_out = LayerNormalization(name=f'kkc_dec_norm_{i}')(dec_out)

    # Attention over encoder output
    attention_out = Attention(
        use_scale=True,
        name='kkc_attention'
    )([dec_out, encoder_output])

    combined = Concatenate(name='kkc_concat')([dec_out, attention_out])
    combined = LayerNormalization(name='kkc_norm')(combined)

    # Dropout before projection (matches original: Dropout(0.2)(combined))
    combined = Dropout(0.2, name='kkc_proj_drop')(combined)

    # Projection layer before output (matches original notebook pattern)
    combined = Dense(
        config.GRU_UNITS * 2,
        activation='relu',
        name='kkc_projection'
    )(combined)

    # Output projection
    kkc_output = Dense(
        char_vocab_size,
        activation='softmax',
        name='kkc_output',
        dtype='float32'  # FP32 for numerical stability under mixed precision
    )(combined)

    return decoder_input, kkc_output


# ===========================================================
# NWP HEAD (Word-level Bi-GRU + Self-Attention)
# Matches original NWP notebook architecture exactly
# ===========================================================

def build_nwp_head(word_vocab_size):
    """Build NWP head: word embedding → Bi-GRU → self-attention → Dense.

    Architecture matches original NWP notebook (train_gru_next_word.ipynb):
    - Word embedding (NOT shared with char embedding)
    - Bi-GRU(return_sequences=True, dropout=0.2)
    - Self-Attention(use_scale=True)
    - Concatenate + LayerNorm
    - Context GRU (compress to single vector)
    - Dropout(0.3) → Dense(softmax)

    Input:  nwp_input (batch, MAX_WORD_CONTEXT) — word IDs
    Output: (batch, word_vocab_size) — word probabilities
    """
    nwp_input = Input(
        shape=(config.MAX_WORD_CONTEXT,),
        name='nwp_input'
    )

    # Word embedding (separate from char embedding — matches original NWP)
    x = Embedding(
        word_vocab_size,
        config.EMBEDDING_DIM,
        name='word_embedding'
    )(nwp_input)

    # Bidirectional GRU (matches original: dropout=0.2)
    encoder_out = Bidirectional(
        GRU(config.GRU_UNITS, return_sequences=True, dropout=0.2,
            name='nwp_gru'),
        name='nwp_bigru'
    )(x)

    # Self-Attention (Luong-style, matches original)
    attention_out = Attention(
        use_scale=True,
        name='nwp_attention'
    )([encoder_out, encoder_out])

    # Combine encoder + attention (matches original)
    combined = Concatenate(name='nwp_concat')([encoder_out, attention_out])
    combined = LayerNormalization(name='nwp_norm')(combined)

    # Context GRU: compress to single vector (matches original)
    context = GRU(
        config.GRU_UNITS,
        name='nwp_context_gru'
    )(combined)
    context = Dropout(config.NWP_DROPOUT, name='nwp_dropout')(context)

    # Output: predict next word
    nwp_output = Dense(
        word_vocab_size,
        activation='softmax',
        name='nwp_output',
        dtype='float32'  # FP32 for numerical stability
    )(context)

    return nwp_input, nwp_output


# ===========================================================
# FULL MULTI-TASK MODEL
# ===========================================================

def build_multitask_model(char_vocab_size, word_vocab_size, strategy=None):
    """Build complete multi-task model with 3 inputs.

    The model combines two independent architectures:
    - KKC: char-level seq2seq (encoder-decoder with attention)
    - NWP: word-level predictor (Bi-GRU + self-attention + context GRU)

    Inputs:
        encoder_input: (batch, MAX_ENCODER_LEN) — char IDs for KKC
        decoder_input: (batch, MAX_DECODER_LEN) — char IDs for KKC decoder
        nwp_input:     (batch, MAX_WORD_CONTEXT) — word IDs for NWP

    Outputs:
        kkc_output: (batch, MAX_DECODER_LEN, char_vocab_size)
        nwp_output: (batch, word_vocab_size)

    Returns:
        model: Keras Model with inputs [encoder_input, decoder_input, nwp_input]
               and outputs [kkc_output, nwp_output]
    """
    def _build():
        # KKC encoder (char-level, returns shared embedding for decoder reuse)
        encoder_input, encoder_output, encoder_state, char_embedding = \
            build_kkc_encoder(char_vocab_size)

        # KKC head (uses shared char embedding — matches original KKC notebook)
        decoder_input, kkc_output = build_kkc_head(
            encoder_output, encoder_state, char_vocab_size,
            char_embedding
        )

        # NWP head (independent word-level path — matches original NWP notebook)
        nwp_input, nwp_output = build_nwp_head(word_vocab_size)

        # Combined model with 3 inputs
        model = Model(
            inputs=[encoder_input, decoder_input, nwp_input],
            outputs=[kkc_output, nwp_output],
            name='multitask_gru_v1'
        )
        return model

    if strategy is not None:
        with strategy.scope():
            model = _build()
    else:
        model = _build()

    return model


def compile_model(model, strategy=None):
    """Compile with combined losses.

    Note: This compile is for model.fit() usage only.
    The custom training loop in training.py manages losses directly.
    """
    def _compile():
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=config.LEARNING_RATE,
                clipnorm=1.0
            ),
            loss={
                'kkc_output': masked_sparse_ce,
                'nwp_output': 'sparse_categorical_crossentropy',
            },
            loss_weights={
                'kkc_output': config.KKC_LOSS_WEIGHT,
                'nwp_output': config.NWP_LOSS_WEIGHT,
            },
            metrics={
                'kkc_output': [masked_accuracy],
                'nwp_output': ['accuracy'],
            }
        )

    if strategy is not None:
        with strategy.scope():
            _compile()
    else:
        _compile()


def masked_sparse_ce(y_true, y_pred):
    """Sparse categorical crossentropy that ignores PAD tokens (idx=0)."""
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    masked_loss = loss * mask
    return tf.reduce_sum(masked_loss) / (tf.reduce_sum(mask) + 1e-8)


def masked_accuracy(y_true, y_pred):
    """Accuracy that ignores PAD tokens (idx=0)."""
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    preds = tf.cast(tf.argmax(y_pred, axis=-1), y_true.dtype)
    correct = tf.cast(tf.equal(y_true, preds), tf.float32) * mask
    return tf.reduce_sum(correct) / (tf.reduce_sum(mask) + 1e-8)
