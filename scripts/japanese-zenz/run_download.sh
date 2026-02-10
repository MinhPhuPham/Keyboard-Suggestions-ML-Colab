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

# Step 2: Compile .mlpackage to .mlmodelc for faster loading
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "📦 Compiling to .mlmodelc (for fast iOS loading)..."
echo "═══════════════════════════════════════════════════════════"

PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODELS_DIR="$PROJECT_ROOT/models/japanese"

# Find all .mlpackage files and compile them
for mlpackage in "$MODELS_DIR"/*.mlpackage; do
    if [ -f "$mlpackage/Manifest.json" ]; then
        # Get basename without extension
        basename=$(basename "$mlpackage" .mlpackage)
        output_dir="$MODELS_DIR/${basename}.mlmodelc"
        
        echo "   Compiling: $(basename "$mlpackage")"
        
        # Remove old .mlmodelc if exists
        [ -d "$output_dir" ] && rm -rf "$output_dir"
        
        # Compile using xcrun
        if xcrun coremlcompiler compile "$mlpackage" "$MODELS_DIR" 2>/dev/null; then
            echo "   ✅ Created: $(basename "$output_dir")"
        else
            echo "   ⚠️ Compilation failed (xcrun not available or error)"
        fi
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ Done! Models ready for iOS deployment."
echo "═══════════════════════════════════════════════════════════"
