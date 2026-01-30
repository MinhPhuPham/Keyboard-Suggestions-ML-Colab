#!/bin/bash
# ============================================================
# Download & Test Japanese Model (Combined Script)
# ============================================================
# Downloads a zenz model from HuggingFace and runs interactive test.
# Does NOT convert to CoreML - use run_download.sh for that.
#
# Usage: 
#   ./run_download_and_test.sh [model_name]
#   ./run_download_and_test.sh Miwa-Keita/zenz-v2.5-small
#   ./run_download_and_test.sh   # Interactive selection
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Available models
MODELS=(
    "Miwa-Keita/zenz-v2.5-small"
    "Miwa-Keita/zenz-v2.5-xsmall"
    "Miwa-Keita/zenz-v2-small"
)

# Default model
DEFAULT_MODEL="Miwa-Keita/zenz-v2.5-small"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "🇯🇵 Japanese Model: Download & Test"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Step 0: Check/Setup virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Virtual environment not found. Running setup..."
    echo ""
    "$SCRIPT_DIR/setup.sh"
    echo ""
fi

# Activate venv
source "$VENV_DIR/bin/activate"
echo "📌 Using Python: $(which python)"
echo ""

# Parse model argument
MODEL_NAME="$1"

# If no model specified, show interactive selection
if [ -z "$MODEL_NAME" ]; then
    echo "📌 Select a model to download and test:"
    echo ""
    for i in "${!MODELS[@]}"; do
        if [ "${MODELS[$i]}" == "$DEFAULT_MODEL" ]; then
            echo "   $((i+1)). ${MODELS[$i]} (default)"
        else
            echo "   $((i+1)). ${MODELS[$i]}"
        fi
    done
    echo ""
    read -p "Enter choice (1-${#MODELS[@]}) or model name [default: 1]: " choice
    
    # Handle choice
    if [ -z "$choice" ]; then
        MODEL_NAME="$DEFAULT_MODEL"
    elif [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#MODELS[@]}" ]; then
        MODEL_NAME="${MODELS[$((choice-1))]}"
    elif [[ "$choice" == */* ]]; then
        # User entered full model name
        MODEL_NAME="$choice"
    else
        echo "❌ Invalid choice: $choice"
        exit 1
    fi
fi

echo ""
echo "📦 Selected model: $MODEL_NAME"
echo ""

# Extract model short name for directory
MODEL_SHORT_NAME=$(echo "$MODEL_NAME" | sed 's|.*/||')
MODELS_DIR="$PROJECT_ROOT/models/japanese"
MODEL_DIR="$MODELS_DIR/$MODEL_SHORT_NAME"

# Check if model already exists
NEED_DOWNLOAD=false
FORCE_FLAG=""

if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ]; then
    echo "✓ Model already exists: $MODEL_DIR"
    read -p "Re-download? (y/N): " redownload
    if [[ $redownload == [yY] ]]; then
        FORCE_FLAG="--force-download"
        NEED_DOWNLOAD=true
    else
        echo "   Skipping download, proceeding to test..."
    fi
else
    echo "📥 Model not found, will download..."
    NEED_DOWNLOAD=true
fi

# Step 1: Download model (if needed) - skip CoreML conversion
if [ "$NEED_DOWNLOAD" = true ]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "Step 1: Downloading model (skip CoreML)..."
    echo "═══════════════════════════════════════════════════════════"
    python "$SCRIPT_DIR/download_convert_zenz_coreml.py" $FORCE_FLAG --model-name "$MODEL_NAME" --skip-coreml
fi

# Step 2: Run test
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Step 2: Testing model..."
echo "═══════════════════════════════════════════════════════════"
python "$SCRIPT_DIR/test_japanese_prediction.py" --model-dir "$MODEL_DIR"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ Done!"
echo "═══════════════════════════════════════════════════════════"
