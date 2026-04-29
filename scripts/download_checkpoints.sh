#!/usr/bin/env bash
# Download the public single-style checkpoint folder from the README.

set -euo pipefail

CHECKPOINT_DIR="${1:-checkpoints}"
FOLDER_URL="https://drive.google.com/drive/folders/1V8rHuhAQnSVW6hwQ8nMYwPSiPRqloHJR"

mkdir -p "$CHECKPOINT_DIR"
gdown --folder "$FOLDER_URL" -O "$CHECKPOINT_DIR"

if [ -f "$CHECKPOINT_DIR/single_style/model_final.pt" ]; then
    ln -sfn "single_style/model_final.pt" "$CHECKPOINT_DIR/model_final.pt"
fi

if [ -f "$CHECKPOINT_DIR/single_style/training_history.json" ]; then
    ln -sfn "single_style/training_history.json" "$CHECKPOINT_DIR/training_history.json"
fi
