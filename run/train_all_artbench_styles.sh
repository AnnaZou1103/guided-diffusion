#!/bin/bash
# Batch training script for all ArtBench styles
# Usage: bash train_all_artbench_styles.sh [artbench_images_dir] [pretrained_model]

# ArtBench 3 style names
STYLES=("impressionism" "romanticism" "surrealism")

ARTBENCH_IMAGES_DIR="${1:-./artbench_images}"
PRETRAINED_MODEL="${2:-models/lsun_bedroom.pt}"

echo "=========================================="
echo "Batch Training ArtBench Styles (3 styles)"
echo "=========================================="
echo "Images directory: $ARTBENCH_IMAGES_DIR"
echo "Pretrained model: $PRETRAINED_MODEL"
echo "Number of styles: ${#STYLES[@]}"
echo ""

# Check data directory
if [ ! -d "$ARTBENCH_IMAGES_DIR" ]; then
    echo "Error: Images directory not found: $ARTBENCH_IMAGES_DIR"
    echo "Please run convert_artbench_lsun.sh first."
    exit 1
fi

# Check pretrained model
if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "Warning: Pretrained model not found: $PRETRAINED_MODEL"
    echo "Will train from scratch for all styles."
fi

# Submit training job for each style
for style in "${STYLES[@]}"; do
    STYLE_DIR="${ARTBENCH_IMAGES_DIR}/${style}"
    
    if [ ! -d "$STYLE_DIR" ]; then
        echo "Warning: Style directory not found: $STYLE_DIR, skipping..."
        continue
    fi
    
    echo "Submitting training job for ${style}..."
    sbatch train_artbench_style.sh "$style" "$ARTBENCH_IMAGES_DIR" "$PRETRAINED_MODEL"
    
    # Wait a bit to avoid submitting too many jobs at once
    sleep 2
done

echo ""
echo "=========================================="
echo "All training jobs submitted!"
echo "Check job status with: squeue -u \$USER"
echo "=========================================="

