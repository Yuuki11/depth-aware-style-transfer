#!/usr/bin/env bash
# Download the standard content/style datasets used by the training script.
#
# COCO train2017 is pulled from the official COCO image host. WikiArt is pulled
# through the Kaggle CLI, so run `kaggle config set -n path -v ~/.kaggle` or
# place your kaggle.json credentials before using --wikiart.

set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
DOWNLOAD_COCO=1
DOWNLOAD_WIKIART=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --coco-only)
            DOWNLOAD_WIKIART=0
            shift
            ;;
        --wikiart-only)
            DOWNLOAD_COCO=0
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--data-dir data] [--coco-only|--wikiart-only]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

mkdir -p "$DATA_DIR"

if [[ "$DOWNLOAD_COCO" -eq 1 ]]; then
    mkdir -p "$DATA_DIR/coco"
    if [[ ! -d "$DATA_DIR/coco/train2017" ]]; then
        wget -c http://images.cocodataset.org/zips/train2017.zip -O "$DATA_DIR/coco/train2017.zip"
        unzip -q "$DATA_DIR/coco/train2017.zip" -d "$DATA_DIR/coco"
    fi
fi

if [[ "$DOWNLOAD_WIKIART" -eq 1 ]]; then
    mkdir -p "$DATA_DIR/wikiart"
    if [[ -z "$(find "$DATA_DIR/wikiart" -type f \( -iname '*.jpg' -o -iname '*.png' \) -print -quit)" ]]; then
        kaggle datasets download -d steubk/wikiart -p "$DATA_DIR/wikiart" --unzip
        if [[ -d "$DATA_DIR/wikiart/wikiart" ]]; then
            find "$DATA_DIR/wikiart/wikiart" -mindepth 1 -maxdepth 1 -exec mv {} "$DATA_DIR/wikiart" \;
            rmdir "$DATA_DIR/wikiart/wikiart"
        fi
    fi
fi

echo "Training data is ready under $DATA_DIR"
