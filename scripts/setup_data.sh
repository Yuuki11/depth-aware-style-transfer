#!/usr/bin/env bash
# setup_data.sh — Download COCO 2017 + WikiArt datasets
#
# Usage:
#   bash scripts/setup_data.sh          # Downloads both datasets
#   bash scripts/setup_data.sh --coco   # COCO only
#   bash scripts/setup_data.sh --wiki   # WikiArt only

set -e

DATA_DIR="./data"
mkdir -p "$DATA_DIR"

download_coco() {
    echo "================================================"
    echo "Downloading COCO 2017 train images (~18GB)..."
    echo "================================================"
    cd "$DATA_DIR"
    if [ ! -d "coco/train2017" ]; then
        wget -q --show-progress http://images.cocodataset.org/zips/train2017.zip
        mkdir -p coco
        unzip -q train2017.zip -d coco/
        rm train2017.zip
        echo "COCO 2017: $(ls coco/train2017/*.jpg | wc -l) images"
    else
        echo "COCO 2017 already exists, skipping."
    fi
    cd ..
}

download_wikiart() {
    echo "================================================"
    echo "Downloading WikiArt from Kaggle (~25GB)..."
    echo "================================================"
    cd "$DATA_DIR"
    if [ ! -d "wikiart" ]; then
        # Requires kaggle CLI: pip install kaggle
        # And ~/.kaggle/kaggle.json with your API key
        if command -v kaggle &> /dev/null; then
            kaggle datasets download steubk/wikiart
            mkdir -p wikiart
            unzip -q wikiart.zip -d wikiart/
            rm wikiart.zip
        else
            echo "ERROR: kaggle CLI not found."
            echo "Install: pip install kaggle"
            echo "Then place your API key at ~/.kaggle/kaggle.json"
            echo ""
            echo "Alternative: download manually from"
            echo "https://www.kaggle.com/datasets/steubk/wikiart"
            exit 1
        fi
        echo "WikiArt: $(find wikiart -name '*.jpg' | wc -l) images"
    else
        echo "WikiArt already exists, skipping."
    fi
    cd ..
}

# Parse args
if [ "$1" == "--coco" ]; then
    download_coco
elif [ "$1" == "--wiki" ]; then
    download_wikiart
else
    download_coco
    download_wikiart
fi

echo ""
echo "Done! Data directory structure:"
find "$DATA_DIR" -maxdepth 2 -type d
