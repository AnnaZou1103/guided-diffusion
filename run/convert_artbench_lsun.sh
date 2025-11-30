#!/bin/bash
# Convert ArtBench LSUN format to image directories
# Supports both tar files and extracted lmdb directories
# Usage: bash convert_artbench_lsun.sh [lsun_dir] [output_dir] [image_size] [temp_dir]
# Example: bash convert_artbench_lsun.sh datasets/artbench_lsun ./artbench_images 256
#
# Expected structure:
#   datasets/artbench_lsun/
#     ├── impressionism_lmdb.tar
#     ├── romanticism_lmdb.tar
#     └── surrealism_lmdb.tar

# ArtBench 3 style names
STYLES=("impressionism" "romanticism" "surrealism")

# Configuration paths
LSUN_DIR="${1:-datasets/artbench_lsun}"  # ArtBench LSUN data directory
OUTPUT_DIR="${2:-./artbench_images}"      # Output image directory
IMAGE_SIZE="${3:-256}"                     # Image size
TEMP_DIR="${4:-./temp_lmdb}"               # Temporary directory for extracted tar files

echo "Converting ArtBench LSUN format to image directories..."
echo "LSUN directory: $LSUN_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Image size: $IMAGE_SIZE"
echo ""

# Create output and temp directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR"

# Convert each style
for style in "${STYLES[@]}"; do
    TAR_FILE="${LSUN_DIR}/${style}_lmdb.tar"
    LMDB_DIR="${TEMP_DIR}/${style}_lmdb"
    STYLE_OUTPUT="${OUTPUT_DIR}/${style}"
    
    # Check if tar file exists
    if [ -f "$TAR_FILE" ]; then
        echo "Found tar file: $TAR_FILE"
        echo "Extracting ${style}..."
        
        # Extract tar file to a style-specific directory
        STYLE_TEMP_DIR="${TEMP_DIR}/${style}"
        mkdir -p "$STYLE_TEMP_DIR"
        
        tar -xf "$TAR_FILE" -C "$STYLE_TEMP_DIR" 2>/dev/null
        
        if [ $? -ne 0 ]; then
            echo "✗ Failed to extract ${TAR_FILE}"
            rm -rf "$STYLE_TEMP_DIR"
            continue
        fi
        
        # Find the extracted lmdb directory (tar might extract with different structure)
        # Try common patterns: style_lmdb, style_lmdb/style_lmdb, etc.
        if [ -d "${STYLE_TEMP_DIR}/${style}_lmdb" ]; then
            LMDB_DIR="${STYLE_TEMP_DIR}/${style}_lmdb"
        elif [ -d "${STYLE_TEMP_DIR}/lmdb" ]; then
            LMDB_DIR="${STYLE_TEMP_DIR}/lmdb"
        else
            # Try to find any lmdb directory in the extracted location
            EXTRACTED_LMDB=$(find "$STYLE_TEMP_DIR" -type d -name "*lmdb*" | head -1)
            if [ -n "$EXTRACTED_LMDB" ] && [ -d "$EXTRACTED_LMDB" ]; then
                LMDB_DIR="$EXTRACTED_LMDB"
            else
                echo "✗ Could not find extracted lmdb directory for ${style}"
                echo "  Contents of ${STYLE_TEMP_DIR}:"
                ls -la "$STYLE_TEMP_DIR" 2>/dev/null || true
                rm -rf "$STYLE_TEMP_DIR"
                continue
            fi
        fi
        
        echo "✓ Extracted to: $LMDB_DIR"
    elif [ -d "${LSUN_DIR}/${style}_lmdb" ]; then
        # If it's already a directory (not tar), use it directly
        LMDB_DIR="${LSUN_DIR}/${style}_lmdb"
        echo "Found lmdb directory: $LMDB_DIR"
    else
        echo "Warning: Neither ${TAR_FILE} nor ${LSUN_DIR}/${style}_lmdb found, skipping ${style}..."
        continue
    fi
    
    # Check if lmdb directory exists
    if [ ! -d "$LMDB_DIR" ]; then
        echo "✗ LMDB directory not found: $LMDB_DIR"
        continue
    fi
    
    # Create train and test subdirectories
    STYLE_TRAIN_OUTPUT="${OUTPUT_DIR}/${style}/train"
    STYLE_TEST_OUTPUT="${OUTPUT_DIR}/${style}/test"
    
    echo "Converting ${style} from $LMDB_DIR..."
    echo "  Training set: first 5000 images -> ${STYLE_TRAIN_OUTPUT}"
    echo "  Test set: next 1000 images -> ${STYLE_TEST_OUTPUT}"
    
    # Convert training set (first 5000 images, indices 0-4999)
    python datasets/lsun_bedroom.py \
        --image-size "$IMAGE_SIZE" \
        --prefix "$style" \
        --max-images 5000 \
        --start-index 0 \
        "$LMDB_DIR" \
        "$STYLE_TRAIN_OUTPUT"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully converted training set for ${style}"
    else
        echo "✗ Failed to convert training set for ${style}"
        continue
    fi
    
    # Convert test set (next 1000 images, indices 5000-5999)
    python datasets/lsun_bedroom.py \
        --image-size "$IMAGE_SIZE" \
        --prefix "$style" \
        --max-images 1000 \
        --start-index 5000 \
        "$LMDB_DIR" \
        "$STYLE_TEST_OUTPUT"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully converted test set for ${style}"
        echo "✓ Successfully converted ${style} (train: 5000, test: 1000)"
    else
        echo "✗ Failed to convert test set for ${style}"
    fi
    echo ""
done

# Clean up temporary extracted files (optional)
# Set CLEANUP_TEMP=1 to automatically clean up, or leave unset for manual cleanup
if [ "${CLEANUP_TEMP:-0}" = "1" ]; then
    echo "Cleaning up temporary files..."
    rm -rf "$TEMP_DIR"
    echo "✓ Cleanup completed"
else
    echo "Temporary files kept in: $TEMP_DIR"
    echo "To clean up automatically, set CLEANUP_TEMP=1 or run: rm -rf $TEMP_DIR"
fi

echo "Conversion completed!"
echo "Converted images are in: $OUTPUT_DIR"
echo ""
echo "Directory structure:"
echo "  $OUTPUT_DIR/"
echo "    ├── impressionism/"
echo "    │   ├── train/  (5000 training images)"
echo "    │   └── test/    (1000 test images)"
echo "    ├── romanticism/"
echo "    │   ├── train/"
echo "    │   └── test/"
echo "    └── surrealism/"
echo "        ├── train/"
echo "        └── test/"

