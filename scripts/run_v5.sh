#!/usr/bin/env bash
# run_v5.sh - One-shot script to generate data and train the v5 model
set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================================="
echo "          Akshara-OCR v5 Training Pipeline"
echo "=========================================================="

cd "$ROOT_DIR"

# 1. Activate venv
if [ -d "venv" ]; then
    echo "Activting virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activting virtual environment..."
    source .venv/bin/activate
else
    echo "WARNING: No venv found! Using system python."
fi

# 2. Check for tqdm and install it if missing (for nice progress bars)
if ! python -c "import tqdm" &> /dev/null; then
    echo "Installing tqdm for progress bars..."
    pip install tqdm
fi

# 3. Generate data
echo "----------------------------------------------------------"
echo "  Step 1: Generating Dataset"
echo "----------------------------------------------------------"
python data/generate_v5_dataset.py

# 4. Train
echo "----------------------------------------------------------"
echo "  Step 2: Training Model (v5)"
echo "----------------------------------------------------------"
python scripts/training/train_v5.py

echo "----------------------------------------------------------"
echo "  Pipeline finished!"
echo "----------------------------------------------------------"
