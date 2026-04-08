#!/bin/bash
# =============================================================================
# Download all downstream evaluation datasets
# =============================================================================
# Usage:
#   bash datasets/download_datasets.sh               # downloads to ./data
#   VISREG_DATA_DIR=/my/path bash datasets/download_datasets.sh
#
# Datasets handled here (require manual download from original sources):
#   1. DTD            - VGG Oxford
#   2. FGVC-Aircraft  - VGG Oxford
#   3. Oxford Flowers - VGG Oxford
#   4. ADE20K         - MIT CSAIL
#   5. AID            - HuggingFace (blanchon/AID)
#   6. MedMNIST       - auto-downloaded by medmnist library (optional pre-download)
#
# Datasets that auto-download via HuggingFace on first run (no action needed):
#   cifar10, cifar100, cars, galaxy10, food, pets, chestxray
# =============================================================================

set -euo pipefail

DATA_DIR="${VISREG_DATA_DIR:-$(cd "$(dirname "$0")/.." && pwd)/data}"
ZOO_DIR="${DATA_DIR}/imagenet-zoo"
mkdir -p "$ZOO_DIR"

echo "========================================="
echo "VISReg Dataset Downloader"
echo "Target directory: $ZOO_DIR"
echo "========================================="

# ---- 1. DTD (Describable Textures Dataset) - VGG Oxford ----
echo ""
echo "[1/5] DTD dataset..."
DTD_DIR="$ZOO_DIR/dtd_original"
mkdir -p "$DTD_DIR"
if [ ! -d "$DTD_DIR/dtd" ]; then
    echo "  Downloading from VGG Oxford..."
    wget -q --show-progress -P "$DTD_DIR" https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
    tar -xzf "$DTD_DIR/dtd-r1.0.1.tar.gz" -C "$DTD_DIR"
    echo "  Done."
else
    echo "  Already exists, skipping."
fi

# ---- 2. FGVC-Aircraft - VGG Oxford ----
echo ""
echo "[2/5] FGVC-Aircraft dataset..."
AIRCRAFT_DIR="$ZOO_DIR/fgvc-aircraft_original"
mkdir -p "$AIRCRAFT_DIR"
if [ ! -d "$AIRCRAFT_DIR/fgvc-aircraft-2013b" ]; then
    echo "  Downloading from VGG Oxford..."
    wget -q --show-progress -P "$AIRCRAFT_DIR" https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
    tar -xzf "$AIRCRAFT_DIR/fgvc-aircraft-2013b.tar.gz" -C "$AIRCRAFT_DIR"
    echo "  Done."
else
    echo "  Already exists, skipping."
fi

# ---- 3. Oxford Flowers 102 - VGG Oxford ----
echo ""
echo "[3/5] Oxford Flowers 102 dataset..."
FLOWERS_DIR="$ZOO_DIR/oxford-flowers_original"
mkdir -p "$FLOWERS_DIR"
if [ ! -f "$FLOWERS_DIR/imagelabels.mat" ]; then
    echo "  Downloading from VGG Oxford..."
    wget -q --show-progress -P "$FLOWERS_DIR" https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
    wget -q --show-progress -P "$FLOWERS_DIR" https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
    wget -q --show-progress -P "$FLOWERS_DIR" https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat
    tar -xzf "$FLOWERS_DIR/102flowers.tgz" -C "$FLOWERS_DIR"
    echo "  Done."
else
    echo "  Already exists, skipping."
fi

# ---- 4. ADE20K (Scene Parsing) - MIT CSAIL ----
echo ""
echo "[4/5] ADE20K dataset..."
ADE20K_DIR="$ZOO_DIR/ADEChallengeData2016"
if [ ! -d "$ADE20K_DIR/images" ]; then
    echo "  Downloading from MIT CSAIL..."
    wget -q --show-progress -P "$ZOO_DIR" http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
    unzip -q "$ZOO_DIR/ADEChallengeData2016.zip" -d "$ZOO_DIR"
    rm -f "$ZOO_DIR/ADEChallengeData2016.zip"
    echo "  Done."
else
    echo "  Already exists, skipping."
fi

# ---- 5. AID (Aerial Image Dataset) - via HuggingFace ----
echo ""
echo "[5/5] AID dataset..."
AID_DIR="$ZOO_DIR/AID"
if [ ! -d "$AID_DIR" ] || [ -z "$(ls -A "$AID_DIR" 2>/dev/null)" ]; then
    echo "  Downloading from HuggingFace (blanchon/AID)..."
    python3 -c "
import os, sys
from datasets import load_dataset
from PIL import Image
from collections import defaultdict

ds = load_dataset('blanchon/AID', split='train')
out = '$AID_DIR'
os.makedirs(out, exist_ok=True)

counts = defaultdict(int)
for item in ds:
    label = item['label']
    class_name = ds.features['label'].int2str(label)
    class_dir = os.path.join(out, class_name)
    os.makedirs(class_dir, exist_ok=True)
    counts[class_name] += 1
    img_path = os.path.join(class_dir, f'{class_name}_{counts[class_name]:04d}.jpg')
    item['image'].convert('RGB').save(img_path)

print(f'  Saved {sum(counts.values())} images across {len(counts)} classes.')
"
    echo "  Done."
else
    echo "  Already exists, skipping."
fi

# ---- Optional: Pre-download MedMNIST ----
echo ""
echo "========================================="
echo "Download complete!"
echo "========================================="
echo ""
echo "Downloaded to: $ZOO_DIR"
echo "  - DTD:             $ZOO_DIR/dtd_original/dtd"
echo "  - FGVC-Aircraft:   $ZOO_DIR/fgvc-aircraft_original/fgvc-aircraft-2013b/data"
echo "  - Oxford Flowers:  $ZOO_DIR/oxford-flowers_original"
echo "  - ADE20K:          $ZOO_DIR/ADEChallengeData2016"
echo "  - AID:             $ZOO_DIR/AID"
echo ""
echo "Auto-downloaded on first use (HuggingFace):"
echo "  cifar10, cifar100, stanford_cars, galaxy10, food101, pets, chestxray"
echo ""
echo "Auto-downloaded on first use (medmnist library):"
echo "  retinamnist, organamnist"
echo ""
echo "Set VISREG_DATA_DIR to point evaluation scripts to this directory:"
echo "  export VISREG_DATA_DIR=$DATA_DIR"
