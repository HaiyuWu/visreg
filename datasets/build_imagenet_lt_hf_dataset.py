"""
Build HuggingFace ImageNet-LT dataset from ImageNet-1K.
Saves pre-filtered train dataset to disk for fast loading.

Usage:
    uv run python build_imagenet_lt_hf_dataset.py

Output:
    {HF_CACHE}/imagenet_lt_hf/train/  (115K LT samples)
    
Test uses HF validation directly (same as ImageNet-1K).
"""

from datasets import load_dataset
from collections import Counter
import numpy as np
from pathlib import Path

np.random.seed(42)

HF_CACHE = "./data/imagenets"
OUTPUT_DIR = Path(HF_CACHE) / "imagenet_lt_hf"

def parse_lt_txt(txt_file):
    counts = Counter()
    with open(txt_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                label = int(parts[1])
                counts[label] += 1
    return counts

def sample_indices(class_counts, labels):
    labels = np.array(labels)
    indices = []
    
    for label in range(1000):
        count = class_counts.get(label, 0)
        if count == 0:
            continue
        available = np.where(labels == label)[0]
        if len(available) >= count:
            selected = np.random.choice(available, count, replace=False)
        else:
            selected = available
            print(f"Warning: Class {label} only has {len(available)} samples, need {count}")
        indices.extend(selected.tolist())
    
    return np.array(indices)


print("=" * 60)
print("Building ImageNet-LT HuggingFace Dataset")
print("=" * 60)

# Load full ImageNet-1K train
print("\n[1/3] Loading HuggingFace ImageNet-1K train...")
ds_train = load_dataset("ILSVRC/imagenet-1k", cache_dir=HF_CACHE, split="train")
print(f"  HF train: {len(ds_train)} samples")

# Get labels
print("\n[2/3] Parsing ImageNet-LT train.txt and sampling...")
train_labels = ds_train['label']

train_counts = parse_lt_txt("ImageNet_LT_train.txt")
print(f"  LT train distribution: {sum(train_counts.values())} samples across {len(train_counts)} classes")

train_indices = sample_indices(train_counts, train_labels)
print(f"  Sampled {len(train_indices)} indices")

# Save train dataset
print("\n[3/3] Saving train dataset...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

lt_train = ds_train.select(train_indices.tolist())
lt_train.save_to_disk(OUTPUT_DIR / "train")
print(f"  Saved {len(lt_train)} samples to {OUTPUT_DIR / 'train'}")

# Verify
print("\n" + "=" * 60)
print("Verification")
print("=" * 60)

from datasets import load_from_disk

ds = load_from_disk(OUTPUT_DIR / "train")
label_counts = Counter(ds['label'])
print(f"\nTrain:")
print(f"  Total samples: {len(ds)}")
print(f"  Classes: {len(label_counts)}")
print(f"  Max samples/class: {max(label_counts.values())}")
print(f"  Min samples/class: {min(label_counts.values())}")

print(f"\nTest: Use HuggingFace 'validation' split directly (50K samples)")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
print(f"""
Train: load_from_disk("{OUTPUT_DIR}/train")
Test:  load_dataset("ILSVRC/imagenet-1k", cache_dir="{HF_CACHE}", split="validation")
""")

