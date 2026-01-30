#!/bin/bash
# ============================================================
# Download & Convert Japanese Model to CoreML
# ============================================================
# Downloads zenz-v2.5-small from HuggingFace and converts
# to CoreML format for iOS deployment.
#
# Usage: 
#   ./run_download.sh              # Normal run (skip download if exists)
#   ./run_download.sh --force      # Force re-download
#   ./run_download.sh --hf-token YOUR_TOKEN
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Check if venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Run ./setup.sh first"
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "🇯🇵 Download & Convert: zenz-v2.5-small"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Activate venv and run
source "$VENV_DIR/bin/activate"
echo "📌 Using Python: $(which python)"
echo ""

# Parse --force shorthand
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--force" ]]; then
        ARGS+=("--force-download")
    else
        ARGS+=("$arg")
    fi
done

python "$SCRIPT_DIR/download_convert_zenz_coreml.py" "${ARGS[@]}"
