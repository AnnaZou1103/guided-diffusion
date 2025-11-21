#!/bin/bash
# Convert ArtBench LSUN format to image directories
# Usage: bash convert_artbench_lsun.sh

# ArtBench 3 style names
STYLES=("impressionism" "romanticism" "surrealism")

# Configuration paths
LSUN_DIR="${1:-/path/to/artbench_lsun}"  # ArtBench LSUN data directory
OUTPUT_DIR="${2:-./artbench_images}"      # Output image directory
IMAGE_SIZE="${3:-256}"                     # Image size

echo "Converting ArtBench LSUN format to image directories..."
echo "LSUN directory: $LSUN_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Image size: $IMAGE_SIZE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Convert each style
for style in "${STYLES[@]}"; do
    LMDB_PATH="${LSUN_DIR}/${style}_train_lmdb"
    STYLE_OUTPUT="${OUTPUT_DIR}/${style}"
    
    if [ ! -d "$LMDB_PATH" ]; then
        echo "Warning: ${LMDB_PATH} not found, skipping ${style}..."
        continue
    fi
    
    echo "Converting ${style}..."
    python datasets/lsun_bedroom.py \
        --image-size "$IMAGE_SIZE" \
        --prefix "$style" \
        "$LMDB_PATH" \
        "$STYLE_OUTPUT"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully converted ${style}"
    else
        echo "✗ Failed to convert ${style}"
    fi
    echo ""
done

echo "Conversion completed!"
echo "Converted images are in: $OUTPUT_DIR"

