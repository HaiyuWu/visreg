<div align="center">

# VISReg: Variance-Invariance-Sketching Regularization for JEPA training

[Haiyu Wu](https://haiyuwu.github.io/)<sup>1</sup> &emsp; [Randall Balestriero](https://randallbalestriero.github.io/)<sup>2</sup> &emsp; <!-- [Yann LeCun](http://yann.lecun.com/)<sup>3</sup> &emsp; -->[Morgan Levine](https://www.altoslabs.com/team/morgan-levine)<sup>1</sup>

<sup>1</sup>Altos Labs &emsp; <sup>2</sup>Brown University<!-- &emsp; <sup>3</sup>New York University -->

<a href='#'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<a href='https://haiyuwu.github.io/visreg/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
<a href='https://huggingface.co/BooBooWu/visreg'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow'></a>
<a href='https://creativecommons.org/licenses/by-nc/4.0/'><img src='https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey'></a>

</div>

This is the official implementation of **VISReg**, a heuristic-free and self-supervised learning method. It is more robust to model collapse and can learn more general representations.

&emsp;💪 **Strong collapse prevention**: High gradient when embedding collapse<br>
&emsp;⚡ **Friendly to scale training**: Linear complexity to scaling factors<br>
&emsp;🧩 **Easy to train**: Similar to LeJEPA, it is a heuristic-free method<br>
&emsp;🏆 **Best OOD performance**: Achieve the best accuracy on 6 OOD datasets<br>
&emsp;📉 **Data efficiency**: Achieving a similar average accuracy to DINOv2 with 90% less data<br>
&emsp;🧬 **Robust to low-quality datasets**: It is robust to long-tailed and sparse datasets<br>

**If you find VISReg useful for your research, please consider citing us 😄**
```bibtex
@inproceedings{wu2026visreg,
  title     = {VISReg: Variance-Invariance-Sketching Regularization for JEPA training},
  author    = {Wu, Haiyu and Balestriero, Randall and Levine, Morgan},
  booktitle = {arXiv},
  year      = {2026}
}
```

# News/Updates

- [2026/6] Paper released on arXiv!
- [2026/6] Added Kornia GPU augmentation pipeline and torch.compile support.
  - Note: the provided checkpoints were not trained with these two functions.
- [2026/3] Pretrained weights added to [HuggingFace](https://huggingface.co/BooBooWu/visreg).
- [2026/3] Code created.

# Table of Contents

| | Section | | Section |
|---|---------|---|---------|
| :wrench: | [Installation](#wrench-installation) | :arrow_down: | [Download pretrained weights](#arrow_down-download-pretrained-weights) |
| :floppy_disk: | [Download datasets](#floppy_disk-download-datasets) | :rocket: | [Pretraining](#rocket-pretraining) |
| :bar_chart: | [Downstream Evaluation](#bar_chart-downstream-evaluation) | :trophy: | [Results](#trophy-results) |
| :pray: | [Acknowledgements](#pray-acknowledgements) | :page_facing_up: | [License](#page_facing_up-license) |

# :wrench: Installation

**Requirements**: Python 3.12, CUDA-compatible GPU

```bash
git clone https://github.com/HaiyuWu/visreg.git
cd visreg

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

# :arrow_down: Download pretrained weights

Pretrained weights (backbone-only, no projection head/optimizer/probe) can be downloaded from [HuggingFace](https://huggingface.co/BooBooWu/visreg) or using python:

| Checkpoint | Backbone | Params | Size |
|------------|----------|--------|------|
| `visreg-vit-b-inet1k.pth` | ViT-B/16 | 86M | 328M |
| `visreg-vit-l-inet1k.pth` | ViT-L/14 | 304M | 1.2G |

```python
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="BooBooWu/visreg", filename="visreg-vit-b-inet1k.pth", local_dir="./")
hf_hub_download(repo_id="BooBooWu/visreg", filename="visreg-vit-l-inet1k.pth", local_dir="./")
```

Load into a timm model:

```python
import timm
import torch

model = timm.create_model("vit_base_patch16_224", pretrained=False)
model.load_state_dict(torch.load("visreg-vit-b-inet1k.pth", weights_only=True))
```

# :floppy_disk: Download datasets

## Pretraining data

ImageNet-1k is loaded from HuggingFace ([ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k)). You must accept the terms of use and authenticate:

```bash
huggingface-cli login
```

## Downstream evaluation data

HuggingFace-hosted datasets (CIFAR-10/100, Stanford Cars, Galaxy10, Food-101, Pets, ChestX-ray) and MedMNIST datasets (RetinaMNIST, OrganAMNIST) are auto-downloaded on first use. For the remaining datasets, run:

```bash
bash datasets/download_datasets.sh
```

This downloads **DTD**, **FGVC-Aircraft**, **Oxford Flowers 102**, **ADE20K**, and **AID** to `./data`. To use a custom location:

```bash
export VISREG_DATA_DIR=/path/to/your/data
bash datasets/download_datasets.sh
```

# :rocket: Pretraining

Training is configured via [Hydra](https://hydra.cc/) YAML configs in `configs/`.

### Single-GPU

```bash
uv run python train.py --config-name default
```

### Multi-GPU

```bash
uv run accelerate launch --num_processes 8 train.py --config-name default
```

### Kornia GPU augmentation

Move the color augmentations to the GPU with Kornia (DataLoader workers emit uint8 crops; ColorJitter/Grayscale/Blur/Solarize/Normalize run on-device). Set `aug.backend=kornia`:

```bash
uv run accelerate launch --num_processes 8 train.py --config-name default aug.backend=kornia
```

### torch.compile

Compile the model via Accelerate's dynamo backend:

```bash
uv run accelerate launch --num_processes 8 --dynamo_backend inductor train.py --config-name default
```

The two can be combined, and the released checkpoints were trained with neither.


# :bar_chart: Downstream Evaluation

## Linear probe

Evaluates frozen features on 10+ classification datasets using a DINOv2-style protocol (SGD, cosine LR, concatenated multi-layer features).

```bash
# All datasets, all shots
uv run python downstream/linear_prob/run_evaluation.py \
    --checkpoint /path/to/checkpoint.pt \
    --model vit_b \
    --datasets all \
    --shots -1

# 10-shot evaluation
uv run python downstream/linear_prob/run_evaluation.py \
    --checkpoint /path/to/checkpoint.pt \
    --model vit_b \
    --shots 10
```

**Dataset groups**: `default` (DTD, Aircraft, Cars, CIFAR-10/100, Flowers, Food, Pets), `ood` (DTD, Galaxy10, AID, ChestX-ray, RetinaMNIST, OrganAMNIST), `all`.

## Semantic segmentation (ADE20K)

Linear probe on frozen patch embeddings for 150-class segmentation.

```bash
uv run python downstream/segmentation/run_linear_seg.py \
    --checkpoint /path/to/checkpoint.pt \
    --model vit_b \
    --img_size 512 \
    --bs 16
```

## Fine-tuning

Full fine-tuning with DeiT-style augmentation (Mixup, CutMix, RandAugment, RandomErasing).

```bash
uv run accelerate launch --num_processes 8 \
    downstream/fine_tuning/train_finetune.py \
    --checkpoint /path/to/checkpoint.pt \
    --model vit_b \
    --dataset cifar100
```

## Comparing with pretrained baselines

The evaluation scripts support loading DINOv2, MoCo v3, MAE, iBOT, data2vec, I-JEPA, and DINOv1 out of the box:

```bash
# DINOv2
uv run python downstream/linear_prob/run_evaluation.py --model dinov2_vit_b

# MAE
uv run python downstream/linear_prob/run_evaluation.py --model mae_vit_l

# I-JEPA
uv run python downstream/linear_prob/run_evaluation.py --model vit_h --pretrained ijepa
```

# :trophy: Results

## In-domain linear probe

VISReg is competitive with heuristic-based methods and outperforms all heuristic-free methods. On the OOD column (DTD), VISReg outperforms all methods including those with heuristics.

| Methods | Backbone | Epochs | DTD | Aircraft | Cars | CIFAR-10 | CIFAR-100 | Flowers | Food | Pets | Avg. | Inet1K |
|---------|----------|--------|-----|----------|------|----------|-----------|---------|------|------|------|--------|
| *w/ heuristics* | | | | | | | | | | | | |
| MoCoV3 | ViT-B/16 | 300 | 73.7 | 57.9 | 67.5 | 96.9 | 85.2 | 91.5 | 81.8 | 89.8 | 80.5 | 76.7 |
| DINO | ViT-B/16 | 400 | 74.3 | 63.6 | 73.9 | 96.5 | 85.0 | **94.6** | 83.1 | 93.6 | 83.1 | 78.2 |
| data2vec | ViT-L/14 | 1600 | 69.7 | 43.9 | 38.7 | 96.9 | 83.7 | 81.4 | 79.6 | 83.0 | 72.1 | 77.3 |
| iBOT | ViT-B/16 | 400 | 74.1 | 63.5 | 73.8 | 97.1 | 85.9 | 93.7 | 84.2 | 93.6 | 83.2 | 79.8 |
| iBOT | ViT-L/16 | 250 | 75.3 | **66.0** | **76.1** | **97.5** | **87.2** | 94.0 | **86.1** | **94.0** | **84.5** | **81.0** |
| I-JEPA | ViT-H/14 | 300 | 69.9 | 55.4 | 59.2 | 97.2 | 85.5 | 86.8 | 83.3 | 92.8 | 78.7 | 79.3 |
| *w/o heuristics* | | | | | | | | | | | | |
| MAE | ViT-L/16 | 1600 | 72.8 | 61.9 | 61.5 | 93.3 | 78.0 | 85.4 | 78.6 | 91.3 | 77.8 | 75.1 |
| **VISReg** | ViT-B/16 | 400 | **75.7** | 57.1 | 64.8 | 94.6 | 78.8 | 90.4 | 82.9 | 88.3 | 79.1 | 75.7 |
| **VISReg** | ViT-L/14 | 400 | **76.5** | 56.6 | 66.2 | 94.1 | 71.9 | 90.2 | 83.3 | 89.2 | 78.5 | 77.0 |

## Out-of-distribution linear probe

VISReg achieves the best average OOD accuracy among all methods at ImageNet-1K scale. With ImageNet-22K pretraining, VISReg matches DINOv2 (trained on 10x more data) on OOD benchmarks.

| Methods | Backbone | DTD | Galaxy10 | AID | ChestXRay | Retina. | OrganA. | Avg. |
|---------|----------|-----|----------|-----|-----------|---------|---------|------|
| *w/ heuristics* | | | | | | | | |
| MoCoV3 | ViT-B/16 | 73.72 | 73.06 | 90.20 | 23.89 | 64.50 | 91.37 | 69.46 |
| DINO | ViT-B/16 | 74.26 | 72.77 | 91.52 | 24.63 | 62.50 | 91.70 | 69.56 |
| data2vec | ViT-L/16 | 69.68 | 65.73 | 85.98 | 22.45 | 63.75 | 89.04 | 66.10 |
| iBOT | ViT-B/16 | 74.10 | 71.65 | **91.73** | 24.80 | 63.75 | 91.84 | 69.64 |
| iBOT | ViT-L/16 | 75.27 | 72.66 | 91.24 | 23.94 | 63.00 | 90.82 | 69.49 |
| I-JEPA | ViT-H/14 | 69.89 | 71.31 | 88.68 | 23.46 | **65.75** | 92.19 | 68.55 |
| *w/o heuristics* | | | | | | | | |
| MAE | ViT-L/16 | 72.77 | 71.98 | 86.42 | 22.87 | 63.00 | 90.06 | 67.85 |
| **VISReg** | ViT-B/16 | 75.69 | 74.01 | 90.91 | **24.88** | 62.25 | **93.40** | 70.19 |
| **VISReg** | ViT-L/14 | **76.54** | **76.32** | 90.12 | 23.64 | 64.25 | 92.93 | **70.63** |
| *large scale* | | | | | | | | |
| DINOv2-LVD142M | ViT-L/14 | **82.23** | 76.72 | **94.27** | 23.83 | **69.50** | 91.01 | 72.93 |
| **VISReg**-Inet22K | ViT-L/14 | 80.74 | **79.82** | 92.81 | **24.46** | 66.50 | **93.33** | **72.94** |

## Transfer learning

VISReg outperforms both DINO and supervised pretraining after fine-tuning on all tested datasets (ViT-B/16).

| Methods | CIFAR-10 | CIFAR-100 | Flowers | Inet1K | Galaxy10 |
|---------|----------|-----------|---------|--------|----------|
| Sup. | 99.0 | 89.5 | 98.5 | 81.5 | - |
| DINO | 99.1 | 91.7 | 98.8 | 82.8 | 86.6 |
| **VISReg** | **99.2** | **91.8** | **99.0** | **83.0** | **87.0** |

## Image generation

Following iREPA, we train SiT-B/2 for 100K steps with guidance from DINO and VISReg features. VISReg achieves better results across all metrics.

| Methods | Backbone | IS | gFID | Precision | Recall |
|---------|----------|----|------|-----------|--------|
| DINO | ViT-B/16 | 33.47 | 41.15 | 50.51 | 60.70 |
| **VISReg** | ViT-B/16 | **33.48** | **40.36** | **51.38** | **61.26** |

# :pray: Acknowledgements

- Thanks [LeJEPA](https://github.com/galilai-group/lejepa) for providing such a good math-grounded theorem.
- Thanks to [Hugging Face](https://huggingface.co/) for dataset and model hosting.

# Star History

[![Star History Chart](https://api.star-history.com/svg?repos=HaiyuWu/visreg&type=Date)](https://star-history.com/#HaiyuWu/visreg&Date)

# :page_facing_up: License

This project (code and pretrained weights) is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) for non-commercial use only.
