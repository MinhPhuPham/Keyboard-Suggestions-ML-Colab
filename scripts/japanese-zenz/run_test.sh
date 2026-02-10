#!/bin/bash
# ============================================================
# Test Japanese Keyboard Predictions
# ============================================================
# Interactive testing script for next-token prediction.
#
# Usage: ./run_test.sh [--model-dir path/to/model]
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
VENV_DIR="$SCRIPT_DIR/.venv"
DEFAULT_MODEL_DIR="$PROJECT_ROOT/models/japanese/zenz-v3.1-xsmall"
# Check if venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Run ./setup.sh first"
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "🇯🇵 Japanese Keyboard Prediction Tester"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Activate venv and run
source "$VENV_DIR/bin/activate"
echo "📌 Using Python: $(which python)"
echo ""

# Use default model dir if not specified
if [[ "$*" != *"--model-dir"* ]]; then
    python "$SCRIPT_DIR/test_japanese_prediction.py" --model-dir "$DEFAULT_MODEL_DIR" "$@"
else
    python "$SCRIPT_DIR/test_japanese_prediction.py" "$@"
fi
