#!/bin/bash
# ============================================================
# Setup Script for Japanese Keyboard Model (zenz-v2.5-small)
# ============================================================
# This script creates a dedicated Python virtual environment
# with Python 3.10 or 3.11 (required for CoreML conversion).
#
# Usage: ./setup.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
REQUIRED_PYTHON_MIN="3.10"
REQUIRED_PYTHON_MAX="3.11"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "🇯🇵 Japanese Keyboard Model - Environment Setup"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ============================================================
# Find correct Python version (3.10 or 3.11)
# ============================================================
find_python() {
    # Priority: python3.11 > python3.10 > python3 (if correct version)
    for cmd in python3.11 python3.10 python3; do
        if command -v "$cmd" &> /dev/null; then
            local version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
            local minor=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
            
            if [[ "$minor" == "10" || "$minor" == "11" ]]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

PYTHON_CMD=$(find_python)

if [ -z "$PYTHON_CMD" ]; then
    echo "❌ Error: Python 3.10 or 3.11 not found!"
    echo ""
    echo "Please install Python 3.10 or 3.11:"
    echo ""
    echo "  macOS (Homebrew):"
    echo "    brew install python@3.11"
    echo ""
    echo "  Ubuntu/Debian:"
    echo "    sudo apt update"
    echo "    sudo apt install python3.11 python3.11-venv"
    echo ""
    echo "  Windows:"
    echo "    Download from https://www.python.org/downloads/"
    echo ""
    echo "  pyenv (cross-platform):"
    echo "    pyenv install 3.11.9"
    echo "    pyenv local 3.11.9"
    echo ""
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo "📌 Found compatible Python: $PYTHON_VERSION"
echo "   Using command: $PYTHON_CMD"
echo ""

# ============================================================
# Create virtual environment
# ============================================================
if [ -d "$VENV_DIR" ]; then
    # Check if existing venv has correct Python version
    VENV_PYTHON_VERSION=$("$VENV_DIR/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "unknown")
    echo "📁 Virtual environment exists with Python $VENV_PYTHON_VERSION"
    
    # Check if venv Python matches required version
    VENV_MINOR=$("$VENV_DIR/bin/python" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
    if [[ "$VENV_MINOR" != "10" && "$VENV_MINOR" != "11" ]]; then
        echo "⚠️  Existing venv has incompatible Python version"
        echo "   Recreating with Python 3.$VENV_MINOR → $PYTHON_VERSION"
        rm -rf "$VENV_DIR"
    else
        read -p "   Recreate? (y/N): " confirm
        if [[ $confirm == [yY] ]]; then
            echo "   Removing old venv..."
            rm -rf "$VENV_DIR"
        else
            echo "   Keeping existing venv."
        fi
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "📦 Creating virtual environment with $PYTHON_CMD..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    echo "   ✓ Created: $VENV_DIR"
fi

# ============================================================
# Install dependencies
# ============================================================
echo ""
echo "📥 Installing dependencies..."
source "$VENV_DIR/bin/activate"

pip install --upgrade pip -q
pip install -r "$SCRIPT_DIR/requirements_japanese.txt"

# Verify installation
echo ""
echo "🔍 Verifying installation..."
python -c "import torch; print(f'   ✓ PyTorch {torch.__version__}')"
python -c "import transformers; print(f'   ✓ Transformers {transformers.__version__}')"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ Setup Complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Python: $(python --version)"
echo "Venv:   $VENV_DIR"
echo ""
echo "Next steps:"
echo "  1. Download & test model:  ./run_download_and_test.sh"
echo "  2. Download + CoreML:      ./run_download.sh"
echo ""
