#!/bin/bash
#SBATCH --job-name=diffusion_sampling
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=diffusion_sampling_%j.out
#SBATCH --error=diffusion_sampling_%j.err
#SBATCH --gres=gpu:1

module load anaconda3
module load cuda/12.1.1

conda activate diffusion

export OMPI_MCA_mtl=^ofi
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,tcp

export OPENAI_LOGDIR=./results/sample

SAMPLE_FLAGS="--batch_size 4 --num_samples 24 --timestep_respacing 250"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

echo "Starting sampling..."
echo "Model Parameters: $MODEL_FLAGS"
echo "Sampling Parameters: $SAMPLE_FLAGS"
echo ""

python scripts/classifier_sample.py \
    $MODEL_FLAGS \
    --classifier_scale 1.0 \
    --classifier_path models/256x256_classifier.pt \
    --model_path models/256x256_diffusion.pt \
    $SAMPLE_FLAGS

echo ""
echo "Sampling completed!"