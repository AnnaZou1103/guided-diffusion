#!/bin/bash
#SBATCH --job-name=artbench_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --error=artbench_train_%j.err
#SBATCH --gres=gpu:1

# Train a single ArtBench style from LSUN format
# Usage: sbatch train_artbench_style.sh <style_name>
# Example: sbatch train_artbench_style.sh impressionism

module load anaconda3
module load cuda/12.1.1
eval "$(conda shell.bash hook)"
conda activate diffusion

export OMPI_MCA_mtl=^ofi
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,tcp

# Style name (passed as argument, uses default if not provided)
STYLE_NAME=${1:-"surrealism"}

# Get project root from current working directory (assumes running from project root)
# When using sbatch, the working directory is preserved
PROJECT_ROOT="$(pwd)"
SCRIPT_DIR="${PROJECT_ROOT}/run"

# Configuration paths
# Convert relative paths to absolute paths
if [ -n "$2" ]; then
    # Use provided path (convert to absolute if relative)
    if [[ "$2" = /* ]]; then
        ARTBENCH_IMAGES_DIR="$2"
    else
        ARTBENCH_IMAGES_DIR="$(cd "$2" && pwd 2>/dev/null || echo "${PROJECT_ROOT}/$2")"
    fi
else
    # Default: use datasets/artbench_images
    ARTBENCH_IMAGES_DIR="${PROJECT_ROOT}/datasets/artbench_images"
fi

if [ -n "$3" ]; then
    # Use provided pretrained model path (convert to absolute if relative)
    if [[ "$3" = /* ]]; then
        PRETRAINED_MODEL="$3"
    else
        PRETRAINED_MODEL="$(cd "$(dirname "$3")" && pwd)/$(basename "$3" 2>/dev/null || echo "${PROJECT_ROOT}/$3")"
    fi
else
    # Default: use project root relative path
    PRETRAINED_MODEL="${PROJECT_ROOT}/models/lsun_bedroom.pt"
fi

export OPENAI_LOGDIR="${PROJECT_ROOT}/logs/artbench_${STYLE_NAME}"

export OPENAI_LOG_FORMAT="stdout,log,csv"
export OPENAI_LOG_FORMAT_MPI="stdout,log,csv"

mkdir -p "$OPENAI_LOGDIR"
chmod 755 "$OPENAI_LOGDIR"

# Fine-tuning parameters: Reduce save interval to ensure at least one save within 2 hours
# Assuming ~5000 steps can be trained in 2 hours, set save_interval to 2000 steps (safe)
TRAIN_FLAGS="--lr_anneal_steps 200000 --batch_size 16 --microbatch 8 --lr 5e-5 --save_interval 2000 --weight_decay 0.0 --log_interval 10"

# Model parameters: unconditional model (same as LSUN)
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

# Data directory (single style image directory)
DATA_DIR="${ARTBENCH_IMAGES_DIR}/${STYLE_NAME}"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    echo "Please run convert_artbench_lsun.sh first to convert LSUN format to images."
    exit 1
fi

# Automatically find latest checkpoint
LATEST_CHECKPOINT=""
if [ -d "$OPENAI_LOGDIR" ]; then
    # Find latest model*.pt file
    LATEST_CHECKPOINT=$(ls -t "$OPENAI_LOGDIR"/model*.pt 2>/dev/null | head -n 1)
fi

# If checkpoint found, use it; otherwise use pretrained model
if [ -n "$LATEST_CHECKPOINT" ] && [ -f "$LATEST_CHECKPOINT" ]; then
    RESUME_CHECKPOINT="--resume_checkpoint $LATEST_CHECKPOINT"
    # Extract step number from filename
    CHECKPOINT_STEP=$(basename "$LATEST_CHECKPOINT" | sed 's/model//; s/.pt//' | grep -o '[0-9]*' || echo "0")
    echo "Found latest checkpoint: $LATEST_CHECKPOINT"
    echo "Resuming from step: $CHECKPOINT_STEP"
else
    # If no checkpoint found, use pretrained model (if exists)
    if [ -f "$PRETRAINED_MODEL" ]; then
        RESUME_CHECKPOINT="--resume_checkpoint $PRETRAINED_MODEL"
        echo "No checkpoint found, using pretrained model: $PRETRAINED_MODEL"
    else
        RESUME_CHECKPOINT=""
        echo "Warning: No checkpoint or pretrained model found, training from scratch"
    fi
fi

echo "=========================================="
echo "Training ArtBench Style: ${STYLE_NAME}"
echo "=========================================="
echo "Data directory: $DATA_DIR"
echo "Model Parameters: $MODEL_FLAGS"
echo "Training Parameters: $TRAIN_FLAGS"
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resuming from: $RESUME_CHECKPOINT"
    if [ -n "$CHECKPOINT_STEP" ]; then
        echo "Current step: $CHECKPOINT_STEP / 200000"
    fi
else
    echo "Training from scratch"
fi
echo "Log directory: $OPENAI_LOGDIR"
echo ""

cd "$PROJECT_ROOT"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"


python scripts/image_train.py \
    --data_dir "$DATA_DIR" \
    $RESUME_CHECKPOINT \
    $MODEL_FLAGS \
    $TRAIN_FLAGS

TRAIN_EXIT_CODE=$?

# Check if training is completed (reached target steps)
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    # Check latest checkpoint step number
    LATEST_CHECKPOINT=$(ls -t "$OPENAI_LOGDIR"/model*.pt 2>/dev/null | head -n 1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        LATEST_STEP=$(basename "$LATEST_CHECKPOINT" | sed 's/model//; s/.pt//' | grep -o '[0-9]*' || echo "0")
        if [ "$LATEST_STEP" -ge 200000 ]; then
            echo ""
            echo "=========================================="
            echo "Training completed for ${STYLE_NAME}!"
            echo "Final step: $LATEST_STEP / 200000"
            echo "Model saved in: $OPENAI_LOGDIR"
            echo "=========================================="
            exit 0
        else
            echo ""
            echo "=========================================="
            echo "Training paused (time limit reached)."
            echo "Current step: $LATEST_STEP / 200000"
            if command -v bc >/dev/null 2>&1; then
                PROGRESS=$(echo "scale=2; $LATEST_STEP * 100 / 200000" | bc)
                echo "Progress: ${PROGRESS}%"
            else
                PROGRESS=$(( $LATEST_STEP * 100 / 200000 ))
                echo "Progress: ${PROGRESS}%"
            fi
            echo "Please resubmit this script to continue training."
            echo "=========================================="
            exit 2  # Return special exit code to indicate continuation needed
        fi
    else
        echo ""
        echo "=========================================="
        echo "Training completed for ${STYLE_NAME}!"
        echo "Model saved in: $OPENAI_LOGDIR"
        echo "=========================================="
        exit 0
    fi
else
    echo ""
    echo "=========================================="
    echo "Training failed for ${STYLE_NAME}!"
    echo "Exit code: $TRAIN_EXIT_CODE"
    echo "=========================================="
    exit 1
fi

