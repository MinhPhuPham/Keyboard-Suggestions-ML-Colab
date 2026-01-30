#!/bin/bash
# ============================================================
# Setup Script for Japanese Keyboard Model (zenz-v2.5-small)
# ============================================================
# This script creates a dedicated Python virtual environment
# and installs all required dependencies.
#
# Usage: ./setup.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "🇯🇵 Japanese Keyboard Model - Environment Setup"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1)
echo "📌 Python: $PYTHON_VERSION"

# Warning for version compatibility
PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
if [[ $PYTHON_MINOR -lt 10 || $PYTHON_MINOR -gt 11 ]]; then
    echo "⚠️  Warning: Python 3.10 or 3.11 recommended for CoreML conversion"
    echo "   Current: $PYTHON_VERSION"
    echo "   The conversion may fail with other versions."
    echo ""
fi

# Create virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "📁 Virtual environment already exists: $VENV_DIR"
    read -p "   Recreate? (y/N): " confirm
    if [[ $confirm == [yY] ]]; then
        echo "   Removing old venv..."
        rm -rf "$VENV_DIR"
    else
        echo "   Keeping existing venv."
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "   ✓ Created: $VENV_DIR"
fi

# Activate and install
echo ""
echo "📥 Installing dependencies..."
source "$VENV_DIR/bin/activate"

pip install --upgrade pip -q
pip install -r "$SCRIPT_DIR/requirements_japanese.txt"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ Setup Complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Download & convert model:  ./run_download.sh"
echo "  2. Test predictions:          ./run_test.sh"
echo ""
