# Run Scripts Documentation

This directory contains scripts for data conversion, training, and sampling with the guided-diffusion models.

## Table of Contents

1. [Data Preparation](#data-preparation)
2. [Pretrained Models](#pretrained-models)
3. [File Structure](#file-structure)
4. [Scripts Overview](#scripts-overview)

---

## Data Preparation

### ArtBench LSUN Format Data

Before training ArtBench style models, you need to convert LSUN format data to image directories.

**Expected input structure:**
```
datasets/artbench_lsun/
  ├── impressionism_lmdb.tar
  ├── romanticism_lmdb.tar
  └── surrealism_lmdb.tar
```

**Output structure:**
After conversion, images will be organized as:
```
artbench_images/  (or datasets/artbench_images/)
  ├── impressionism/
  │   └── impressionism_*.png
  ├── romanticism/
  │   └── romanticism_*.png
  └── surrealism/
      └── surrealism_*.png
```

---

## Pretrained Models

Download the following pretrained models and place them in the `models/` directory:

### For ArtBench Training (LSUN-based)
- **lsun_bedroom.pt** - LSUN bedroom pretrained model
  - Download: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_bedroom.pt
  - Used as pretrained checkpoint for ArtBench style training

### For ImageNet Fine-tuning
- **256x256_diffusion_uncond.pt** - ImageNet 256x256 unconditional model
  - Download: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
  - Used for fine-tuning on ArtBench styles

### For Sampling
- **256x256_classifier.pt** - Classifier for 256x256 guided sampling
  - Download: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt
- **256x256_diffusion.pt** - 256x256 diffusion model
  - Download: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt
- **64x64_classifier.pt** - Classifier for 64x64 guided sampling
  - Download: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt
- **64x64_diffusion.pt** - 64x64 diffusion model
  - Download: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_diffusion.pt
- **64_256_upsampler.pt** - Upsampler from 64x64 to 256x256
  - Download: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64_256_upsampler.pt

---

## File Structure

### Directory Layout

```
guided-diffusion/
├── models/                    # Pretrained model checkpoints
│   ├── lsun_bedroom.pt
│   ├── 256x256_diffusion_uncond.pt
│   └── ...
├── datasets/
│   ├── artbench_lsun/         # Input: LSUN format data (tar files or lmdb dirs)
│   └── artbench_images/        # Output: Converted image directories
├── logs/                      # Training logs and checkpoints
│   ├── artbench_impressionism/
│   ├── artbench_romanticism/
│   ├── artbench_surrealism/
│   └── artbench_imagenet_*/
├── results/
└── run/                       # This directory
    └── *.sh                   # Script files
```

---

## Scripts Overview

### Data Conversion Scripts

#### `convert_artbench_lsun.sh`

**Purpose:** Converts ArtBench LSUN format data (tar files or LMDB directories) to image directories.

**Usage:**
```bash
bash convert_artbench_lsun.sh [lsun_dir] [output_dir] [image_size] [temp_dir]
```

**Parameters:**
- `lsun_dir` (optional, default: `datasets/artbench_lsun`) - Directory containing LSUN format data
- `output_dir` (optional, default: `./artbench_images`) - Output directory for converted images
- `image_size` (optional, default: `256`) - Target image size
- `temp_dir` (optional, default: `./temp_lmdb`) - Temporary directory for extracted tar files

**Example:**
```bash
bash convert_artbench_lsun.sh datasets/artbench_lsun ./artbench_images 256
```

**Output:**
- Creates subdirectories for each style (impressionism, romanticism, surrealism) in the output directory
- Each subdirectory contains PNG images with the style prefix

**Environment Variables:**
- `CLEANUP_TEMP=1` - Set to automatically clean up temporary files after conversion

---

### Training Scripts

#### `train_artbench_style.sh`

**Purpose:** Trains a single ArtBench style model from LSUN pretrained checkpoint or from scratch.

**Usage:**
```bash
sbatch train_artbench_style.sh [style_name] [artbench_images_dir] [pretrained_model]
```

**Parameters:**
- `style_name` (optional, default: `surrealism`) - ArtBench style name: `impressionism`, `romanticism`, or `surrealism`
- `artbench_images_dir` (optional, default: `datasets/artbench_images`) - Directory containing style image subdirectories
- `pretrained_model` (optional, default: `models/lsun_bedroom.pt`) - Path to pretrained LSUN model

**Example:**
```bash
sbatch train_artbench_style.sh impressionism ./artbench_images models/lsun_bedroom.pt
```

**Training Configuration:**
- Target steps: 200,000
- Batch size: 8
- Microbatch: 4
- Learning rate: 1e-5
- Image size: 256x256
- Model: Unconditional diffusion model (same architecture as LSUN)
- FP16: Disabled (use_fp16=False)

**Output:**
- Training logs: `logs/artbench_{style_name}/`
- Checkpoints: `logs/artbench_{style_name}/model*.pt`
- Training log file: `logs/artbench_{style_name}/training_output.log`

**Note:** This script automatically resumes from the latest checkpoint if training was interrupted.

---

#### `train_imagenet_finetune.sh`

**Purpose:** Fine-tunes ImageNet 256x256 pretrained model on a specific ArtBench style.

**Usage:**
```bash
sbatch train_imagenet_finetune.sh [style_name] [artbench_images_dir] [pretrained_model_path]
```

**Parameters:**
- `style_name` (optional, default: `surrealism`) - ArtBench style name
- `artbench_images_dir` (optional, default: `datasets/artbench_images`) - Directory containing style image subdirectories
- `pretrained_model_path` (optional, default: `models/256x256_diffusion_uncond.pt`) - Path to ImageNet 256x256 unconditional model

**Example:**
```bash
sbatch train_imagenet_finetune.sh impressionism ./artbench_images models/256x256_diffusion_uncond.pt
```

**Training Configuration:**
- Target steps: 200,000
- Batch size: 16
- Learning rate: 5e-5
- Image size: 256x256
- Model: ImageNet 256x256 unconditional architecture (dropout 0.0)

**Output:**
- Training logs: `logs/artbench_imagenet_{style_name}/`
- Checkpoints: `logs/artbench_imagenet_{style_name}/model*.pt`

**Note:** This script automatically resumes from the latest checkpoint if training was interrupted.

---

### Sampling Scripts

#### `sample_artbench_style.sh`

**Purpose:** Generates samples from a trained ArtBench style model.

**Usage:**
```bash
sbatch sample_artbench_style.sh [style_name] [model_path] [num_samples]
```

**Parameters:**
- `style_name` (optional, default: `impressionism`) - ArtBench style name
- `model_path` (optional, default: `./logs/artbench_{style_name}/model200000.pt`) - Path to trained model checkpoint
- `num_samples` (optional, default: `100`) - Number of samples to generate

**Example:**
```bash
sbatch sample_artbench_style.sh impressionism ./logs/artbench_impressionism/model200000.pt 100
```

**Sampling Configuration:**
- Batch size: 4
- Timestep respacing: 1000 (full diffusion steps)
- Image size: 256x256
- Model: Unconditional diffusion model

**Output:**
- Samples: `results/artbench_{style_name}/samples_*.npz`
- Images: `results/artbench_{style_name}/image/image_*.png`

---

#### `sample_LSUN.sh`

**Purpose:** Generates samples from the LSUN bedroom pretrained model.

**Usage:**
```bash
sbatch sample_LSUN.sh
```

**Parameters:** None (uses hardcoded paths)

**Model Configuration:**
- Model path: `models/lsun_bedroom.pt`
- Image size: 256x256
- Unconditional model
- Dropout: 0.1

**Sampling Configuration:**
- Batch size: 4
- Number of samples: 24
- Timestep respacing: 250

**Output:**
- Samples: `results/LSUN/samples_*.npz`
- Images: `results/LSUN/image/image_*.png`

---

#### `sample_256x256.sh`

**Purpose:** Generates samples from the 256x256 classifier-guided ImageNet model.

**Usage:**
```bash
sbatch sample_256x256.sh
```

**Parameters:** None (uses hardcoded paths)

**Model Configuration:**
- Diffusion model: `models/256x256_diffusion.pt`
- Classifier: `models/256x256_classifier.pt`
- Classifier scale: 1.0
- Image size: 256x256
- Class conditional: True

**Sampling Configuration:**
- Batch size: 4
- Number of samples: 24
- Timestep respacing: 250

**Output:**
- Samples: `results/sample/samples_*.npz`
- Images: `results/sample/image/image_*.png`

---

#### `upsample_64-256:.sh`

**Purpose:** Generates 64x64 samples and upsamples them to 256x256 using the upsampler model.

**Usage:**
```bash
sbatch upsample_64-256:.sh
```

**Parameters:** None (uses hardcoded paths)

**Process:**
1. First generates 64x64 samples using classifier-guided model
2. Then upsamples to 256x256 using the upsampler model

**Model Configuration:**
- 64x64 diffusion: `models/64x64_diffusion.pt`
- 64x64 classifier: `models/64x64_classifier.pt`
- Upsampler: `models/64_256_upsampler.pt`
- Classifier scale: 1.0

**Sampling Configuration:**
- Batch size: 4
- Number of samples: 24
- Timestep respacing: 250

**Output:**
- 64x64 samples: `results/sample/samples_*.npz`
- 256x256 upsampled: `results/upsample/samples_*.npz`
- Images: `results/upsample/image/image_*.png`

---

## Common Parameters

### Model Flags

Most scripts use model configuration flags. Common parameters include:

- `--attention_resolutions` - Attention resolutions (e.g., `32,16,8`)
- `--class_cond` - Class conditional (True/False)
- `--diffusion_steps` - Number of diffusion steps (typically 1000)
- `--dropout` - Dropout rate (0.0, 0.1)
- `--image_size` - Image size (64, 128, 256, 512)
- `--learn_sigma` - Learn sigma (True/False)
- `--noise_schedule` - Noise schedule (`linear`, `cosine`)
- `--num_channels` - Number of channels (192, 256)
- `--num_head_channels` - Number of head channels (64)
- `--num_res_blocks` - Number of residual blocks (2, 3)
- `--resblock_updown` - Residual block up/down (True/False)
- `--use_fp16` - Use FP16 (True/False)
- `--use_scale_shift_norm` - Use scale shift norm (True/False)

### Training Flags

- `--batch_size` - Batch size
- `--microbatch` - Microbatch size (-1 to disable)
- `--lr` - Learning rate
- `--lr_anneal_steps` - Learning rate annealing steps
- `--save_interval` - Checkpoint save interval
- `--log_interval` - Logging interval
- `--weight_decay` - Weight decay
- `--resume_checkpoint` - Path to checkpoint to resume from

### Sampling Flags

- `--batch_size` - Batch size for sampling
- `--num_samples` - Total number of samples to generate
- `--timestep_respacing` - Timestep respacing (e.g., `250`, `1000`, `ddim25`)
- `--use_ddim` - Use DDIM (True/False)
- `--clip_denoised` - Clip denoised (True/False)
- `--classifier_scale` - Classifier guidance scale

---

## Environment Setup

All scripts assume the following environment:

- **SLURM** - For job scheduling (sbatch commands)
- **Conda environment** - Named `diffusion` with required packages
- **CUDA** - Version 12.1.1
- **Modules** - anaconda3 and cuda modules loaded

If not using SLURM, you can modify the scripts to run directly with `bash` instead of `sbatch`.

---

## Notes

1. **Checkpoint Resumption:** Training scripts automatically detect and resume from the latest checkpoint in the log directory.

2. **Time Limits:** SLURM scripts have time limits (typically 2 hours for training). If training is interrupted, simply resubmit the same script to continue.

3. **Output Directories:** All scripts create necessary output directories automatically.

4. **Model Paths:** Ensure pretrained models are downloaded and placed in the `models/` directory before running scripts.

5. **Data Format:** Training scripts expect image directories where each image is a PNG file. The `convert_artbench_lsun.sh` script handles the conversion from LSUN format.

