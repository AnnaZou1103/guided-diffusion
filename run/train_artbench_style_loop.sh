#!/bin/bash
# Auto-resubmit training script for ArtBench style
# This script automatically resubmits training jobs until completion
# Usage: bash train_artbench_style_loop.sh [style_name] [max_iterations]
# Example: bash train_artbench_style_loop.sh surrealism 50

STYLE_NAME=${1:-"surrealism"}
MAX_ITERATIONS=${2:-50}  # Maximum resubmissions (about 100 hours)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_ROOT}/logs/artbench_${STYLE_NAME}"
TARGET_STEPS=200000

echo "=========================================="
echo "Auto-resubmit Training: ${STYLE_NAME}"
echo "Target steps: ${TARGET_STEPS}"
echo "Max iterations: ${MAX_ITERATIONS}"
echo "Log directory: ${LOG_DIR}"
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
    JOB_ID=$(sbatch --parsable "$SCRIPT_DIR/train_artbench_style.sh" "$STYLE_NAME")
    
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
    
    # Record previous checkpoint
    PREV_CHECKPOINT="$LATEST_CHECKPOINT"
    LATEST_CHECKPOINT=$(ls -t "$LOG_DIR"/model*.pt 2>/dev/null | head -n 1)
    
    # Check if new checkpoint or progress was made
    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo ""
        echo "Warning: No checkpoint found after job completion."
        echo "Please check job logs:"
        echo "  tail -n 50 artbench_train_${JOB_ID}.out"
        echo "  tail -n 50 artbench_train_${JOB_ID}.err"
        exit 1
    elif [ "$LATEST_CHECKPOINT" == "$PREV_CHECKPOINT" ]; then
        echo ""
        echo "Warning: No new checkpoint created. Training may have failed."
        echo "Please check job logs:"
        echo "  tail -n 50 artbench_train_${JOB_ID}.out"
        echo "  tail -n 50 artbench_train_${JOB_ID}.err"
        exit 1
    else
        NEW_STEP=$(basename "$LATEST_CHECKPOINT" | sed 's/model//; s/.pt//' | grep -o '[0-9]*')
        echo "New checkpoint created: step $NEW_STEP"
    fi
    
    # Check job exit status (from output files)
    if [ -f "artbench_train_${JOB_ID}.err" ]; then
        ERROR_SIZE=$(stat -f%z "artbench_train_${JOB_ID}.err" 2>/dev/null || stat -c%s "artbench_train_${JOB_ID}.err" 2>/dev/null)
        if [ "$ERROR_SIZE" -gt 0 ]; then
            echo ""
            echo "Warning: Job produced error output. Last 10 lines:"
            tail -n 10 "artbench_train_${JOB_ID}.err"
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

