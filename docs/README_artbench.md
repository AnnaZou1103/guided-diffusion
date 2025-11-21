# ArtBench 训练指南

本目录包含使用 LSUN 格式的 ArtBench 数据集进行训练的脚本。

## ArtBench 数据集

本脚本使用 ArtBench 的 3 种艺术风格，每个风格约 5,000 张训练图像：
- **impressionism**（印象主义）
- **romanticism**（浪漫主义）
- **surrealism**（超现实主义）

## 使用步骤

### 1. 下载 ArtBench LSUN 格式数据

从 [ArtBench Kaggle](https://www.kaggle.com/datasets/artbench) 下载 original size LSUN, per-style 格式的数据。

目录结构应该是：
```
artbench_lsun/
├── impressionism_train_lmdb/
├── romanticism_train_lmdb/
└── surrealism_train_lmdb/
```

### 2. 转换 LSUN 格式为图像目录

```bash
bash convert_artbench_lsun.sh /path/to/artbench_lsun ./artbench_images 256
```

参数说明：
- 第一个参数：ArtBench LSUN 数据目录
- 第二个参数：输出图像目录（默认：./artbench_images）
- 第三个参数：图像尺寸（默认：256）

转换后的目录结构：
```
artbench_images/
├── impressionism/
│   ├── impressionism_0000000.png
│   ├── impressionism_0000001.png
│   └── ...
├── romanticism/
└── surrealism/
```

### 3. 训练单个风格

```bash
sbatch train_artbench_style.sh impressionism
# 或
sbatch train_artbench_style.sh romanticism
# 或
sbatch train_artbench_style.sh surrealism
```

或者指定自定义路径：
```bash
sbatch train_artbench_style.sh impressionism ./artbench_images models/lsun_bedroom.pt
```

参数说明：
- 第一个参数：风格名称
- 第二个参数：图像目录（默认：./artbench_images）
- 第三个参数：预训练模型路径（默认：models/lsun_bedroom.pt）

### 4. 批量训练所有风格

```bash
bash train_all_artbench_styles.sh ./artbench_images models/lsun_bedroom.pt
```

这会为所有 3 个风格提交训练任务。

### 5. 从训练好的模型采样

```bash
sbatch sample_artbench_style.sh impressionism
# 或
sbatch sample_artbench_style.sh romanticism
# 或
sbatch sample_artbench_style.sh surrealism
```

或者指定模型路径和采样数量：
```bash
sbatch sample_artbench_style.sh impressionism ./logs/artbench_impressionism/model200000.pt 200
```

## 训练配置

### 微调参数
- **迭代次数**: 200,000 步
- **批次大小**: 32
- **学习率**: 5e-5（微调，较小）
- **保存间隔**: 每 10,000 步
- **Dropout**: 0.1

### 模型架构
- **图像尺寸**: 256x256
- **分类条件**: False（无条件模型）
- **扩散步数**: 1000
- **噪声调度**: linear
- **FP16**: True

### 预训练模型
默认使用 `lsun_bedroom.pt` 作为预训练模型，因为：
- 艺术图像与自然图像有相似性
- LSUN bedroom 模型已学习良好的图像特征
- 可以加速收敛并提高生成质量

## 输出文件

### 训练输出
- 模型 checkpoint: `./logs/artbench_<style>/model<step>.pt`
- 训练日志: `./logs/artbench_<style>/progress.csv`
- 文本日志: `./logs/artbench_<style>/log.txt`

### 采样输出
- 采样结果: `./results/artbench_<style>/samples_<shape>.npz`

## 注意事项

1. **数据格式**: 确保 LSUN 数据库目录名称格式为 `{style}_train_lmdb`
2. **预训练模型**: 如果没有预训练模型，脚本会自动从头训练（需要更多时间）
3. **存储空间**: 每个风格训练需要约 50-100 GB 空间
4. **训练时间**: 在单个 GPU 上，每个风格约需 24-48 小时
5. **并行训练**: 可以同时训练多个风格（如果有多个 GPU）

## 故障排除

### 数据目录不存在
```
Error: Data directory not found
```
解决：先运行 `convert_artbench_lsun.sh` 转换数据

### 预训练模型不存在
```
Warning: Pretrained model not found
```
解决：下载预训练模型或从头训练（移除 `--resume_checkpoint` 参数）

### 内存不足
如果遇到 OOM 错误，可以：
- 减小 `batch_size`（在 `train_artbench_style.sh` 中）
- 使用 `microbatch` 参数

## 示例工作流

```bash
# 1. 转换数据
bash convert_artbench_lsun.sh /data/artbench_lsun ./artbench_images 256

# 2. 训练所有风格（批量提交）
bash train_all_artbench_styles.sh ./artbench_images models/lsun_bedroom.pt

# 3. 检查训练状态
squeue -u $USER

# 4. 训练完成后采样
sbatch sample_artbench_style.sh impressionism
sbatch sample_artbench_style.sh romanticism
sbatch sample_artbench_style.sh surrealism
```

## 引用

如果使用 ArtBench 数据集，请引用：
```bibtex
@article{liao2022artbench,
  title={The ArtBench Dataset: Benchmarking Generative Models with Artworks},
  author={Liao, Peiyuan and Li, Xiuyu and Liu, Xihui and Keutzer, Kurt},
  journal={arXiv preprint arXiv:2206.11404},
  year={2022}
}
```

