#!/bin/bash
# Create reference batch for ArtBench style evaluation using TEST SET only
# Usage: bash create_artbench_reference.sh <style_name> [artbench_images_dir] [num_images] [output_dir]
# Example: bash create_artbench_reference.sh impressionism ./datasets/artbench_images 1000
# Output: datasets/artbench_reference/reference_artbench_{style_name}.npz (default)
# Note: Automatically uses test set images from index 5000 onwards

STYLE_NAME=${1:-"impressionism"}
ARTBENCH_IMAGES_DIR=${2:-"./datasets/artbench_images"}
NUM_IMAGES=${3:-1000}  # ArtBench test set has 1000 images per style

# Default output path: datasets/artbench_reference/reference_artbench_{style_name}.npz
OUTPUT_DIR="${4:-./datasets/artbench_reference}"
OUTPUT_PATH="${OUTPUT_DIR}/reference_artbench_${STYLE_NAME}.npz"

# Use test/ subdirectory
STYLE_DIR="${ARTBENCH_IMAGES_DIR}/${STYLE_NAME}/test"

if [ ! -d "$STYLE_DIR" ]; then
    echo "Error: Style directory not found: $STYLE_DIR"
    echo "Available styles in $ARTBENCH_IMAGES_DIR:"
    ls -d "$ARTBENCH_IMAGES_DIR"/*/ 2>/dev/null | sed 's|.*/||' | sed 's|/$||' || echo "No styles found"
    exit 1
fi

echo "=========================================="
echo "Creating ArtBench Reference Batch from Test Set"
echo "=========================================="
echo "Style: $STYLE_NAME"
echo "Test set directory: $STYLE_DIR"
echo "Number of images: $NUM_IMAGES"
echo "Output directory: $OUTPUT_DIR"
echo "Output file: $OUTPUT_PATH"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Load all images from test/ directory
python utils/create_reference_batch.py \
    --data_dir "$STYLE_DIR" \
    --output "$OUTPUT_PATH" \
    --num_images "$NUM_IMAGES" \
    --image_size 256

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Reference batch created successfully!"
    echo "Output: $OUTPUT_PATH"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Failed to create reference batch!"
    echo "=========================================="
    exit 1
fi

