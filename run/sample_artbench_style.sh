#!/bin/bash
#SBATCH --job-name=artbench_sample
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=artbench_sample_%j.out
#SBATCH --error=artbench_sample_%j.err
#SBATCH --gres=gpu:1

# Sample from a trained ArtBench style model
# Usage: sbatch sample_artbench_style.sh <style_name> [model_path] [num_samples]
# Example: sbatch sample_artbench_style.sh impressionism

module load anaconda3
module load cuda/12.1.1

conda activate diffusion

export OMPI_MCA_mtl=^ofi
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,tcp

# Parameters
STYLE_NAME=${1:-"impressionism"}
MODEL_PATH=${2:-"./logs/artbench_${STYLE_NAME}/model200000.pt"}
NUM_SAMPLES=${3:-100}

export OPENAI_LOGDIR=./results/artbench_${STYLE_NAME}

# Sampling parameters
SAMPLE_FLAGS="--batch_size 4 --num_samples ${NUM_SAMPLES} --timestep_respacing 1000"

# Model parameters: unconditional model (same as training)
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

# Check model file
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    echo "Available models in ./logs/artbench_${STYLE_NAME}/:"
    ls -lh ./logs/artbench_${STYLE_NAME}/model*.pt 2>/dev/null || echo "No models found"
    exit 1
fi

echo "=========================================="
echo "Sampling from ArtBench Style: ${STYLE_NAME}"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Number of samples: $NUM_SAMPLES"
echo "Model Parameters: $MODEL_FLAGS"
echo "Sampling Parameters: $SAMPLE_FLAGS"
echo "Output directory: $OPENAI_LOGDIR"
echo ""

python scripts/image_sample.py \
    $MODEL_FLAGS \
    --model_path "$MODEL_PATH" \
    $SAMPLE_FLAGS

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Sampling completed for ${STYLE_NAME}!"
    echo "Results saved in: $OPENAI_LOGDIR"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Sampling failed for ${STYLE_NAME}!"
    echo "=========================================="
    exit 1
fi

