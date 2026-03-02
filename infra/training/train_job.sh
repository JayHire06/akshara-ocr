#!/bin/bash
set -e

echo "======================================"
echo " Starting Akshara OCR Training Job"
echo "======================================"

if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY environment variable is not set."
    echo "Please set it before starting the container."
    exit 1
fi

echo "Logging in to Weights & Biases..."
wandb login $WANDB_API_KEY

echo "Pulling dataset from DVC..."
if [ -d ".dvc" ]; then
    dvc pull
else
    echo "Warning: No .dvc directory found. Proceeding with local dataset..."
fi

echo "Calling PyTorch training script (model/train.py)..."
python model/train.py

echo "Training Completed successfully."
