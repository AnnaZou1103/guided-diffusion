#!/bin/bash
# Auto-resubmit training script for ArtBench style
# This script automatically resubmits training jobs until completion
# Usage: bash train_artbench_style_loop.sh [style_name] [artbench_images_dir] [max_iterations]
# Example: bash train_artbench_style_loop.sh surrealism
# Example: bash train_artbench_style_loop.sh surrealism /path/to/artbench_images 50

STYLE_NAME=${1:-"surrealism"}
ARTBENCH_IMAGES_DIR_ARG=${2:-""}  # Optional: ArtBench images directory (if not provided, will auto-detect)
MAX_ITERATIONS=${3:-50}  # Maximum resubmissions (about 100 hours)

# Get project root from current working directory (assumes running from project root)
# When using sbatch, the working directory is preserved
PROJECT_ROOT="$(pwd)"
SCRIPT_DIR="${PROJECT_ROOT}/run"

# Verify script directory exists
if [ ! -d "$SCRIPT_DIR" ] || [ ! -f "$SCRIPT_DIR/train_artbench_style.sh" ]; then
    echo "Error: Cannot find train_artbench_style.sh in $SCRIPT_DIR"
    echo "Current directory: $PROJECT_ROOT"
    echo "Please run from project root directory: cd /path/to/guided-diffusion && sbatch run/train_artbench_style_loop.sh"
    exit 1
fi

LOG_DIR="${PROJECT_ROOT}/logs/artbench_${STYLE_NAME}"
TARGET_STEPS=200000

# Determine dataset directory (use absolute path)
# If provided as argument, use it; otherwise auto-detect
if [ -n "$ARTBENCH_IMAGES_DIR_ARG" ]; then
    # Use provided path (convert to absolute if relative)
    if [[ "$ARTBENCH_IMAGES_DIR_ARG" = /* ]]; then
        ARTBENCH_IMAGES_DIR="$ARTBENCH_IMAGES_DIR_ARG"
    else
        ARTBENCH_IMAGES_DIR="$(cd "$ARTBENCH_IMAGES_DIR_ARG" && pwd 2>/dev/null || echo "${PROJECT_ROOT}/${ARTBENCH_IMAGES_DIR_ARG}")"
    fi
else
    # Auto-detect: Try common locations (prioritize datasets/artbench_images)
    if [ -d "${PROJECT_ROOT}/datasets/artbench_images" ]; then
        ARTBENCH_IMAGES_DIR="${PROJECT_ROOT}/datasets/artbench_images"
    elif [ -d "${PROJECT_ROOT}/artbench_images" ]; then
        ARTBENCH_IMAGES_DIR="${PROJECT_ROOT}/artbench_images"
    elif [ -d "./datasets/artbench_images" ]; then
        ARTBENCH_IMAGES_DIR="$(cd ./datasets/artbench_images && pwd)"
    elif [ -d "./artbench_images" ]; then
        ARTBENCH_IMAGES_DIR="$(cd ./artbench_images && pwd)"
    else
        # Use default: datasets/artbench_images
        ARTBENCH_IMAGES_DIR="${PROJECT_ROOT}/datasets/artbench_images"
    fi
fi

# Verify dataset directory exists
if [ ! -d "$ARTBENCH_IMAGES_DIR" ]; then
    echo "Error: ArtBench images directory not found: $ARTBENCH_IMAGES_DIR"
    echo "Please run convert_artbench_lsun.sh first to convert LSUN format to images."
    echo "Or set ARTBENCH_IMAGES_DIR environment variable."
    exit 1
fi

# Verify style directory exists
STYLE_DATA_DIR="${ARTBENCH_IMAGES_DIR}/${STYLE_NAME}"
if [ ! -d "$STYLE_DATA_DIR" ]; then
    echo "Error: Style data directory not found: $STYLE_DATA_DIR"
    echo "Available styles in $ARTBENCH_IMAGES_DIR:"
    ls -d "$ARTBENCH_IMAGES_DIR"/*/ 2>/dev/null | sed 's|.*/||' | sed 's|/$||' || echo "No styles found"
    exit 1
fi

# Pretrained model path
PRETRAINED_MODEL="${PROJECT_ROOT}/models/lsun_bedroom.pt"

echo "=========================================="
echo "Auto-resubmit Training: ${STYLE_NAME}"
echo "Target steps: ${TARGET_STEPS}"
echo "Max iterations: ${MAX_ITERATIONS}"
echo "Log directory: ${LOG_DIR}"
echo "Dataset directory: ${ARTBENCH_IMAGES_DIR}"
echo "Style data directory: ${STYLE_DATA_DIR}"
echo "Pretrained model: ${PRETRAINED_MODEL}"
echo "=========================================="

# Ensure log directory exists
mkdir -p "$LOG_DIR"

iteration=0

while [ $iteration -lt $MAX_ITERATIONS ]; do
    iteration=$((iteration + 1))
    echo ""
    echo "--- Iteration $iteration ---"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # Check if training is completed
    LATEST_CHECKPOINT=$(ls -t "$LOG_DIR"/model*.pt 2>/dev/null | head -n 1)
    if [ -n "$LATEST_CHECKPOINT" ] && [ -f "$LATEST_CHECKPOINT" ]; then
        LATEST_STEP=$(basename "$LATEST_CHECKPOINT" | sed 's/model//; s/.pt//' | grep -o '[0-9]*')
        if [ -n "$LATEST_STEP" ] && [ "$LATEST_STEP" -ge "$TARGET_STEPS" ]; then
            echo ""
            echo "=========================================="
            echo "Training completed!"
            echo "Latest step: $LATEST_STEP / $TARGET_STEPS"
            echo "Final checkpoint: $LATEST_CHECKPOINT"
            echo "Total iterations: $iteration"
            echo "=========================================="
            exit 0
        fi
        echo "Current step: ${LATEST_STEP:-0} / $TARGET_STEPS"
        if command -v bc >/dev/null 2>&1; then
            PROGRESS=$(echo "scale=2; ${LATEST_STEP:-0} * 100 / $TARGET_STEPS" | bc)
            echo "Progress: ${PROGRESS}%"
        else
            PROGRESS=$(( ${LATEST_STEP:-0} * 100 / $TARGET_STEPS ))
            echo "Progress: ${PROGRESS}%"
        fi
    else
        echo "No checkpoint found yet, starting from beginning..."
    fi
    
    # Submit job
    echo ""
    echo "Submitting training job..."
    echo "Script directory: $SCRIPT_DIR"
    echo "Training script: $SCRIPT_DIR/train_artbench_style.sh"
    
    # Use absolute path for sbatch to avoid SLURM temp directory issues
    TRAIN_SCRIPT_PATH="${SCRIPT_DIR}/train_artbench_style.sh"
    if [ ! -f "$TRAIN_SCRIPT_PATH" ]; then
        echo "Error: Training script not found: $TRAIN_SCRIPT_PATH"
        exit 1
    fi
    
    # Pass style_name, dataset_dir, and pretrained_model to train_artbench_style.sh
    JOB_ID=$(sbatch --parsable "$TRAIN_SCRIPT_PATH" "$STYLE_NAME" "$ARTBENCH_IMAGES_DIR" "$PRETRAINED_MODEL")
    
    if [ $? -ne 0 ] || [ -z "$JOB_ID" ]; then
        echo "Error: Failed to submit job!"
        exit 1
    fi
    
    echo "Job ID: $JOB_ID"
    echo "Job output: artbench_train_${JOB_ID}.out"
    echo "Job error: artbench_train_${JOB_ID}.err"
    
    # Wait for job to complete
    echo ""
    echo "Waiting for job to complete..."
    while squeue -j "$JOB_ID" 2>/dev/null | grep -q "$JOB_ID"; do
        sleep 60  # Check every minute
        # Show progress (every 5 minutes)
        if [ $(($(date +%s) % 300)) -eq 0 ]; then
            echo "  [$(date '+%H:%M:%S')] Job $JOB_ID still running..."
        fi
    done
    
    echo "Job finished, checking status..."
    sleep 5  # Wait for filesystem sync
    
    # Create error log directory
    ERROR_LOG_DIR="${PROJECT_ROOT}/logs/artbench_${STYLE_NAME}_errors"
    mkdir -p "$ERROR_LOG_DIR"
    
    # Get job exit status from SLURM
    JOB_EXIT_CODE=""
    if command -v sacct >/dev/null 2>&1; then
        JOB_EXIT_CODE=$(sacct -j "$JOB_ID" --format=ExitCode --noheader --parsable2 2>/dev/null | head -1 | cut -d'|' -f1)
    fi
    
    # Check job output files
    JOB_OUT_FILE="artbench_train_${JOB_ID}.out"
    JOB_ERR_FILE="artbench_train_${JOB_ID}.err"
    
    # Record previous checkpoint
    PREV_CHECKPOINT="$LATEST_CHECKPOINT"
    LATEST_CHECKPOINT=$(ls -t "$LOG_DIR"/model*.pt 2>/dev/null | head -n 1)
    
    # Parse exit code (format: "EXIT_CODE:SIGNAL" or just "EXIT_CODE")
    EXIT_CODE_NUM=""
    if [ -n "$JOB_EXIT_CODE" ]; then
        # Extract exit code number (before colon)
        EXIT_CODE_NUM=$(echo "$JOB_EXIT_CODE" | cut -d':' -f1)
    fi
    
    # Check job exit status and log errors
    # Exit code 2 means time limit reached (normal, should continue)
    # Exit code 0 means success (check if training completed)
    # Other exit codes mean failure
    JOB_FAILED=false
    if [ -n "$EXIT_CODE_NUM" ] && [ "$EXIT_CODE_NUM" != "0" ] && [ "$EXIT_CODE_NUM" != "2" ]; then
        JOB_FAILED=true
        echo ""
        echo "=========================================="
        echo "Job $JOB_ID failed with exit code: $JOB_EXIT_CODE"
        echo "=========================================="
    elif [ "$EXIT_CODE_NUM" = "2" ]; then
        echo ""
        echo "Job $JOB_ID reached time limit (exit code: 2). This is normal, will continue..."
    fi
    
    # Check if new checkpoint or progress was made
    if [ -z "$LATEST_CHECKPOINT" ]; then
        JOB_FAILED=true
        echo ""
        echo "Warning: No checkpoint found after job completion."
    elif [ "$LATEST_CHECKPOINT" == "$PREV_CHECKPOINT" ]; then
        JOB_FAILED=true
        echo ""
        echo "Warning: No new checkpoint created. Training may have failed."
    else
        NEW_STEP=$(basename "$LATEST_CHECKPOINT" | sed 's/model//; s/.pt//' | grep -o '[0-9]*')
        echo "New checkpoint created: step $NEW_STEP"
    fi
    
    # If job failed, log detailed error information
    if [ "$JOB_FAILED" = true ]; then
        ERROR_LOG_FILE="${ERROR_LOG_DIR}/job_${JOB_ID}_error.log"
        echo "==========================================" > "$ERROR_LOG_FILE"
        echo "Error Log for Job $JOB_ID" >> "$ERROR_LOG_FILE"
        echo "Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$ERROR_LOG_FILE"
        echo "Iteration: $iteration" >> "$ERROR_LOG_FILE"
        echo "Style: $STYLE_NAME" >> "$ERROR_LOG_FILE"
        echo "==========================================" >> "$ERROR_LOG_FILE"
        echo "" >> "$ERROR_LOG_FILE"
        
        # Log job exit code
        if [ -n "$JOB_EXIT_CODE" ]; then
            echo "SLURM Exit Code: $JOB_EXIT_CODE" >> "$ERROR_LOG_FILE"
            echo "" >> "$ERROR_LOG_FILE"
        fi
        
        # Log standard output (last 100 lines)
        if [ -f "$JOB_OUT_FILE" ]; then
            echo "=== Standard Output (last 100 lines) ===" >> "$ERROR_LOG_FILE"
            tail -n 100 "$JOB_OUT_FILE" >> "$ERROR_LOG_FILE"
            echo "" >> "$ERROR_LOG_FILE"
        fi
        
        # Log error output (all content)
        if [ -f "$JOB_ERR_FILE" ]; then
            ERROR_SIZE=$(stat -f%z "$JOB_ERR_FILE" 2>/dev/null || stat -c%s "$JOB_ERR_FILE" 2>/dev/null)
            if [ "$ERROR_SIZE" -gt 0 ]; then
                echo "=== Error Output (all content) ===" >> "$ERROR_LOG_FILE"
                cat "$JOB_ERR_FILE" >> "$ERROR_LOG_FILE"
                echo "" >> "$ERROR_LOG_FILE"
            fi
        fi
        
        # Log training log file if exists
        if [ -f "${LOG_DIR}/log.txt" ]; then
            echo "=== Training Log (last 50 lines) ===" >> "$ERROR_LOG_FILE"
            tail -n 50 "${LOG_DIR}/log.txt" >> "$ERROR_LOG_FILE"
            echo "" >> "$ERROR_LOG_FILE"
        fi
        
        # Log training output log if exists
        if [ -f "${LOG_DIR}/training_output.log" ]; then
            echo "=== Training Output Log (last 100 lines) ===" >> "$ERROR_LOG_FILE"
            tail -n 100 "${LOG_DIR}/training_output.log" >> "$ERROR_LOG_FILE"
            echo "" >> "$ERROR_LOG_FILE"
        fi
        
        # Log progress CSV if exists
        if [ -f "${LOG_DIR}/progress.csv" ]; then
            echo "=== Progress CSV (last 20 lines) ===" >> "$ERROR_LOG_FILE"
            tail -n 20 "${LOG_DIR}/progress.csv" >> "$ERROR_LOG_FILE"
            echo "" >> "$ERROR_LOG_FILE"
        fi
        
        # Display error summary
        echo ""
        echo "=========================================="
        echo "Job $JOB_ID failed!"
        echo "Error log saved to: $ERROR_LOG_FILE"
        echo "=========================================="
        echo ""
        echo "Quick error summary:"
        echo "--- Last 20 lines of output ---"
        if [ -f "$JOB_OUT_FILE" ]; then
            tail -n 20 "$JOB_OUT_FILE"
        fi
        echo ""
        echo "--- Error output ---"
        if [ -f "$JOB_ERR_FILE" ] && [ "$ERROR_SIZE" -gt 0 ]; then
            cat "$JOB_ERR_FILE"
        else
            echo "(No error output file or file is empty)"
        fi
        echo ""
        echo "For full error details, see: $ERROR_LOG_FILE"
        echo "=========================================="
        echo ""
        echo "Do you want to continue to next iteration? (This will exit now for manual inspection)"
        echo "To continue manually, check the error log and fix the issue, then run:"
        echo "  sbatch run/train_artbench_style.sh $STYLE_NAME $ARTBENCH_IMAGES_DIR $PRETRAINED_MODEL"
        exit 1
    fi
    
    # Check for warnings in output (even if job succeeded)
    if [ -f "$JOB_ERR_FILE" ]; then
        ERROR_SIZE=$(stat -f%z "$JOB_ERR_FILE" 2>/dev/null || stat -c%s "$JOB_ERR_FILE" 2>/dev/null)
        if [ "$ERROR_SIZE" -gt 0 ]; then
            echo ""
            echo "Note: Job produced some error output (may be warnings):"
            tail -n 5 "artbench_train_${JOB_ID}.err"
        fi
    fi
    
    echo "Progress made. Continuing to next iteration..."
    echo ""
done

echo ""
echo "=========================================="
echo "Reached maximum iterations ($MAX_ITERATIONS)"
LATEST_CHECKPOINT=$(ls -t "$LOG_DIR"/model*.pt 2>/dev/null | head -n 1)
if [ -n "$LATEST_CHECKPOINT" ]; then
    LATEST_STEP=$(basename "$LATEST_CHECKPOINT" | sed 's/model//; s/.pt//' | grep -o '[0-9]*')
    echo "Current step: ${LATEST_STEP:-0} / $TARGET_STEPS"
    echo "Latest checkpoint: $LATEST_CHECKPOINT"
fi
echo "Please check training status manually or increase MAX_ITERATIONS."
echo "=========================================="
exit 0

