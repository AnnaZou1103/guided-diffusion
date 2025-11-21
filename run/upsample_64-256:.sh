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
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"

python scripts/classifier_sample.py \
    $MODEL_FLAGS \
    --classifier_scale 1.0 \
    --classifier_path models/64x64_classifier.pt \
    --classifier_depth 4 \
    --model_path models/64x64_diffusion.pt \
    $SAMPLE_FLAGS


cp ./results/sample/samples_*64x64x3.npz ./results/sample/64_samples.npz

export OPENAI_LOGDIR=./results/upsample
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --large_size 256 --small_size 64 --learn_sigma True --noise_schedule linear --num_channels 192 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

python scripts/super_res_sample.py \
    $MODEL_FLAGS \
    --model_path models/64_256_upsampler.pt \
    --base_samples ./results/sample/64_samples.npz \
    $SAMPLE_FLAGS