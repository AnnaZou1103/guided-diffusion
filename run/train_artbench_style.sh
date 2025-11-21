#!/bin/bash
#SBATCH --job-name=artbench_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=artbench_train_%j.out
#SBATCH --error=artbench_train_%j.err
#SBATCH --gres=gpu:1

# Train a single ArtBench style from LSUN format
# Usage: sbatch train_artbench_style.sh <style_name>
# Example: sbatch train_artbench_style.sh impressionism

module load anaconda3
module load cuda/12.1.1

conda activate diffusion

export OMPI_MCA_mtl=^ofi
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,tcp

# Style name (passed as argument, uses default if not provided)
STYLE_NAME=${1:-"impressionism"}

# Configuration paths
ARTBENCH_IMAGES_DIR="${2:-./artbench_images}"  # ArtBench images directory
PRETRAINED_MODEL="${3:-models/lsun_bedroom.pt}"  # Pretrained model path

export OPENAI_LOGDIR=./logs/artbench_${STYLE_NAME}

# Fine-tuning parameters: start from pretrained model with smaller learning rate
TRAIN_FLAGS="--iterations 200000 --anneal_lr True --batch_size 32 --lr 5e-5 --save_interval 10000 --weight_decay 0.0 --log_interval 10"

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

# Check if pretrained model exists
if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "Warning: Pretrained model not found: $PRETRAINED_MODEL"
    echo "Training from scratch instead..."
    RESUME_CHECKPOINT=""
else
    RESUME_CHECKPOINT="--resume_checkpoint $PRETRAINED_MODEL"
fi

echo "=========================================="
echo "Training ArtBench Style: ${STYLE_NAME}"
echo "=========================================="
echo "Data directory: $DATA_DIR"
echo "Model Parameters: $MODEL_FLAGS"
echo "Training Parameters: $TRAIN_FLAGS"
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resuming from: $PRETRAINED_MODEL"
else
    echo "Training from scratch"
fi
echo "Log directory: $OPENAI_LOGDIR"
echo ""

python scripts/image_train.py \
    --data_dir "$DATA_DIR" \
    $RESUME_CHECKPOINT \
    $MODEL_FLAGS \
    $TRAIN_FLAGS

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed for ${STYLE_NAME}!"
    echo "Model saved in: $OPENAI_LOGDIR"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Training failed for ${STYLE_NAME}!"
    echo "=========================================="
    exit 1
fi

