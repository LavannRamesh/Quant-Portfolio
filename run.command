#!/bin/bash
# ============================================================
#  Quant & Data Science Master Portfolio — Mac Launcher
# ============================================================

set -e

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Quant & Data Science Master Portfolio           ║"
echo "║  Self-Configuring Launcher                       ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Navigate to script directory
cd "$(dirname "$0")"

# Check for Python 3
if command -v python3 &>/dev/null; then
    PY_CMD=python3
elif command -v python &>/dev/null; then
    PY_CMD=python
else
    echo "❌ Error: Python 3 is not installed or not in PATH."
    exit 1
fi

# Ensure virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment (.venv)..."
    $PY_CMD -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip and install dependencies
echo "📥 Verifying dependencies from requirements.txt..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

# macOS OpenMP runtime self-healing for LightGBM & XGBoost
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Search for an existing libomp.dylib if not already linked
    LIBOMP_SRC=$(find .venv/lib/python*/site-packages/sklearn/.dylibs/libomp.dylib /Library/Frameworks/R.framework/Versions/*/Resources/lib/libomp.dylib /opt/homebrew/opt/libomp/lib/libomp.dylib 2>/dev/null | head -1 || true)
    
    if [ -n "$LIBOMP_SRC" ] && [ -f "$LIBOMP_SRC" ]; then
        for PKG_DIR in $(find .venv/lib/python*/site-packages -maxdepth 1 -name "lightgbm" -o -name "xgboost" 2>/dev/null); do
            mkdir -p "$PKG_DIR/lib"
            cp -n "$LIBOMP_SRC" "$PKG_DIR/lib/libomp.dylib" 2>/dev/null || true
            for DYLIB in "$PKG_DIR/lib"/*.dylib; do
                if [ -f "$DYLIB" ] && command -v install_name_tool &>/dev/null; then
                    install_name_tool -add_rpath @loader_path "$DYLIB" 2>/dev/null || true
                fi
            done
        done
    fi
fi

echo "🚀 Launching Streamlit web portal..."
# Open browser after a brief delay
(sleep 2 && open http://localhost:8501 2>/dev/null || true) &

# Launch Streamlit server
streamlit run app.py --server.headless true --server.port 8501
