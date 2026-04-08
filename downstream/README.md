# Downstream Evaluation Suite

This folder contains scripts for evaluating pretrained models on various downstream tasks.

## Data Setup

All datasets are stored under a single root directory (default: `./data`).
Set `VISREG_DATA_DIR` to override:

```bash
export VISREG_DATA_DIR=/path/to/your/data
```

Run the download script to fetch datasets that are not auto-downloaded:

```bash
bash datasets/download_datasets.sh
```

This downloads **DTD**, **FGVC-Aircraft**, **Oxford Flowers 102**, **ADE20K**, and **AID**.
The remaining datasets (**CIFAR-10/100**, **Stanford Cars**, **Galaxy10**, **Food-101**, **Pets**, **ChestX-ray**, **RetinaMNIST**, **OrganAMNIST**) are auto-downloaded on first use via HuggingFace or the `medmnist` library.

---

## 1. Linear Probe Evaluation
Evaluates the model on a suite of 10+ datasets (DTD, CIFAR, Aircraft, etc.) with 1-shot, 10-shot, and All-shot protocols.

### Usage
```bash
# Evaluate a checkpoint on all datasets (All-shots)
python downstream/linear_prob/run_evaluation.py \
    --checkpoint /path/to/your/checkpoint.pt \
    --model vit_l \
    --shots -1

# Evaluate on 10-shot
python downstream/linear_prob/run_evaluation.py \
    --checkpoint /path/to/your/checkpoint.pt \
    --model vit_l \
    --shots 10
```

---

## 2. Semantic Segmentation (ADE20K)

### Linear Probe (Recommended)
This approach uses a simple linear layer on top of the frozen patch embeddings. It does **not** require `mmcv` or `mmsegmentation`, making it much easier to run on environments with specific CUDA versions (like CUDA 12.8).

ADE20K is loaded from disk at `$VISREG_DATA_DIR/imagenet-zoo/ADEChallengeData2016/`.
Make sure to run `bash datasets/download_datasets.sh` first.

```bash
uv run downstream/segmentation/run_linear_seg.py \
    --checkpoint /path/to/your/checkpoint.pt \
    --model vit_b \
    --img_size 512 \
    --bs 16
```
