"""
Configuration for Multi-Task GRU Model.
All hyperparameters, paths, and constants in one place.
"""
import os


# ===========================================================
# PLATFORM DETECTION
# ===========================================================
def detect_platform():
    """Detect running environment: colab, kaggle, or local."""
    if os.path.exists('/content/drive'):
        return 'colab'
    elif os.path.exists('/kaggle/working'):
        return 'kaggle'
    return 'local'


PLATFORM = detect_platform()

# ===========================================================
# DIRECTORY PATHS
# ===========================================================
if PLATFORM == 'colab':
    DRIVE_DIR = '/content/drive/MyDrive/Keyboard-Suggestions-ML-Colab'
elif PLATFORM == 'kaggle':
    DRIVE_DIR = '/kaggle/working'
else:
    DRIVE_DIR = os.path.expanduser('~/KeyboardSuggestionsML')

MODEL_DIR = f'{DRIVE_DIR}/models/multitask_v1'
CACHE_DIR = f'{DRIVE_DIR}/cache/multitask_v1'
DATASET_DIR = f'{DRIVE_DIR}/datasets'

# ===========================================================
# TRAINING MODE
# ===========================================================
TESTING_MODE = True

if TESTING_MODE:
    MAX_SAMPLES = 100_000
    MAX_NWP_PAIRS = 500_000
    NUM_EPOCHS = 10
    CACHE_SUFFIX = '_test'
else:
    MAX_SAMPLES = 8_000_000
    MAX_NWP_PAIRS = 8_000_000
    NUM_EPOCHS = 10
    CACHE_SUFFIX = ''

# ===========================================================
# MODEL ARCHITECTURE (Shared Encoder)
# ===========================================================
CHAR_VOCAB_SIZE = 6000        # Character vocabulary
WORD_VOCAB_SIZE = 6000        # Word vocabulary (NWP head)
EMBEDDING_DIM = 128           # Shared char embedding
GRU_UNITS = 192               # Bi-GRU units (output = 384)
NUM_ENCODER_LAYERS = 2        # Shared encoder depth
NUM_DECODER_LAYERS = 3        # KKC decoder depth
DROPOUT_RATE = 0.1            # Encoder/decoder dropout
NWP_DENSE_UNITS = 256         # NWP head hidden dim
NWP_DROPOUT = 0.3             # NWP head dropout

# ===========================================================
# SEQUENCE LENGTHS
# ===========================================================
MAX_CONTEXT_LEN = 30          # Left context (chars)
MAX_INPUT_LEN = 30            # Kana input (chars)
MAX_OUTPUT_LEN = 30           # Kanji output (chars)
MAX_ENCODER_LEN = MAX_CONTEXT_LEN + 1 + MAX_INPUT_LEN  # ctx + <SEP> + input
MAX_DECODER_LEN = MAX_OUTPUT_LEN + 2                    # BOS + content + EOS
MAX_WORD_CONTEXT = 10         # NWP: max words in context

# ===========================================================
# SPECIAL TOKENS
# ===========================================================
SPECIAL_TOKENS = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<SEP>']
PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3
SEP_IDX = 4
SEP_TOKEN = '<SEP>'

# ===========================================================
# TRAINING CONFIG
# ===========================================================
BATCH_SIZE = 512              # Per GPU — T4 (16GB) handles 512 for ~5M param GRU
FORCE_REBUILD_CACHE = False
PREFIX_AUG_RATIO = 0.3        # Fraction of KKC samples to augment with prefix variants
KKC_LOSS_WEIGHT = 1.0        # KKC task weight
NWP_LOSS_WEIGHT = 1.0        # NWP task weight (equal to KKC)
LEARNING_RATE = 1e-3

# ===========================================================
# CACHE FILE PATHS
# ===========================================================
def get_cache_paths(cache_dir, suffix=''):
    """Return all cache file paths."""
    return {
        # Char vocab (shared)
        'char_vocab': f'{cache_dir}/char_vocab{suffix}.json',

        # KKC arrays
        'kkc_encoder': f'{cache_dir}/kkc_encoder{suffix}.npy',
        'kkc_dec_in': f'{cache_dir}/kkc_decoder_input{suffix}.npy',
        'kkc_dec_tgt': f'{cache_dir}/kkc_decoder_target{suffix}.npy',
        'kkc_test_cases': f'{cache_dir}/kkc_test_cases{suffix}.json',

        # NWP arrays
        'nwp_vocab': f'{cache_dir}/nwp_word_vocab{suffix}.json',
        'nwp_x': f'{cache_dir}/nwp_x{suffix}.npy',
        'nwp_y': f'{cache_dir}/nwp_y{suffix}.npy',
        'nwp_char_x': f'{cache_dir}/nwp_char_x{suffix}.npy',  # Shared encoder
        'nwp_test_cases': f'{cache_dir}/nwp_test_cases{suffix}.json',
    }


def ensure_dirs():
    """Create output directories."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)


def print_config():
    """Print current configuration."""
    mode = "⚠️ TESTING" if TESTING_MODE else "🚀 FULL TRAINING"
    print(f"{'='*60}")
    print(f"  Multi-Task GRU Config")
    print(f"{'='*60}")
    print(f"  Mode:     {mode}")
    print(f"  Platform: {PLATFORM}")
    print(f"  Samples:  {MAX_SAMPLES:,}")
    print(f"  Epochs:   {NUM_EPOCHS}")
    print(f"  Batch:    {BATCH_SIZE}")
    print(f"")
    print(f"  Shared Encoder: Bi-GRU {NUM_ENCODER_LAYERS}×{GRU_UNITS}")
    print(f"  KKC Decoder:    GRU {NUM_DECODER_LAYERS}×{GRU_UNITS*2}")
    print(f"  NWP Head:       Dense {NWP_DENSE_UNITS}")
    print(f"  Loss weights:   KKC={KKC_LOSS_WEIGHT}, NWP={NWP_LOSS_WEIGHT}")
    print(f"")
    print(f"  Model:  {MODEL_DIR}")
    print(f"  Cache:  {CACHE_DIR}")
    print(f"{'='*60}")
