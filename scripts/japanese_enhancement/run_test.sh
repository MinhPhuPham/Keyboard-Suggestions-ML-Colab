#!/bin/bash
# ============================================================
# Setup & Run: BiGRU Shared Encoder Test (Local)
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "🔧 BiGRU Shared Encoder — Local Environment"
echo "============================================================"
echo "  Project: $PROJECT_DIR"
echo "  Venv:    $VENV_DIR"
echo "============================================================"

# Create venv if not exists
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "✓ venv created"
fi

# Activate
source "$VENV_DIR/bin/activate"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet tensorflow numpy fugashi[unidic-lite] tqdm

echo ""
echo "✅ Environment ready!"
echo "============================================================"
echo ""

# Run based on argument
cd "$PROJECT_DIR"

case "${1:-test}" in
    test)
        echo "🧪 Running unit tests..."
        python -m scripts.japanese_enhancement.test_local
        ;;
    predict)
        echo "🔮 Running prediction test..."
        python -m scripts.japanese_enhancement.test_prediction \
            --model-dir "${2:-./models/multitask_v2}"
        ;;
    batch)
        echo "🔮 Running batch prediction test..."
        python -m scripts.japanese_enhancement.test_prediction \
            --model-dir "${2:-./models/multitask_v2}" --batch
        ;;
    *)
        echo "Usage: ./run_test.sh [test|predict|batch] [model-dir]"
        echo ""
        echo "  test     Run unit tests (default)"
        echo "  predict  Interactive prediction test"
        echo "  batch    Batch prediction test (non-interactive)"
        ;;
esac
