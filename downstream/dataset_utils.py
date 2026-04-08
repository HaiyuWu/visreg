import os
import random
import torch
import numpy as np
import scipy.io
import medmnist
from medmnist import INFO
from pathlib import Path
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms.v2 import InterpolationMode
from datasets import load_dataset as hf_load_dataset


# ============================================================================
# Cache Directories
# ============================================================================
# Override with environment variable VISREG_DATA_DIR (default: ./data)

_DATA_DIR = os.environ.get("VISREG_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
IMAGENET_ZOO_CACHE = os.environ.get("IMAGENET_ZOO_CACHE", os.path.join(_DATA_DIR, "imagenet-zoo"))
IMAGENET_CACHE = os.environ.get("IMAGENET_CACHE", os.path.join(_DATA_DIR, "imagenet-cache"))
CHESTXRAY_CACHE = os.environ.get("CHESTXRAY_CACHE", os.path.join(_DATA_DIR, "chest-x-ray"))


# ============================================================================
# Dataset Configurations
# ============================================================================

DATASET_CONFIGS = {
    # === HuggingFace Datasets ===
    "imagenet": {
        "type": "huggingface",
        "path": "ILSVRC/imagenet-1k",
        "cache": IMAGENET_CACHE,
        "num_classes": 1000,
        "train_split": "train",
        "val_split": "validation",
    },
    "cifar10": {
        "type": "huggingface",
        "path": "uoft-cs/cifar10",
        "config": "plain_text",
        "cache": IMAGENET_ZOO_CACHE,
        "num_classes": 10,
        "train_split": "train",
        "val_split": "test",
    },
    "cifar100": {
        "type": "huggingface",
        "path": "uoft-cs/cifar100",
        "config": "cifar100",
        "cache": IMAGENET_ZOO_CACHE,
        "num_classes": 100,
        "train_split": "train",
        "val_split": "test",
        "label_key": "fine_label",
    },
    "cars": {
        "type": "huggingface",
        "path": "tanganke/stanford_cars",
        "cache": IMAGENET_ZOO_CACHE,
        "num_classes": 196,
        "train_split": "train",
        "val_split": "test",
    },
    "galaxy10": {
        "type": "huggingface",
        "path": "matthieulel/galaxy10_decals",
        "cache": IMAGENET_ZOO_CACHE,
        "num_classes": 10,
        "train_split": "train",
        "val_split": "test",
    },
    "food": {
        "type": "huggingface",
        "path": "ethz/food101",
        "cache": IMAGENET_ZOO_CACHE,
        "num_classes": 101,
        "train_split": "train",
        "val_split": "validation",
    },
    "pets": {
        "type": "huggingface",
        "path": "timm/oxford-iiit-pet",
        "cache": IMAGENET_ZOO_CACHE,
        "num_classes": 37,
        "train_split": "train",
        "val_split": "test",
    },
    "chestxray": {
        "type": "huggingface",
        "path": "alkzar90/NIH-Chest-X-ray-dataset",
        "config": "image-classification",
        "cache": CHESTXRAY_CACHE,
        "num_classes": 14,
        "train_split": "train",
        "val_split": "test",
        "multi_label": True,
    },
    "retinamnist": {
        "type": "medmnist",
        "data_flag": "retinamnist",
        "num_classes": 5,  # 5 diabetic retinopathy grades
        "cache": IMAGENET_ZOO_CACHE,
        "size": 224,
    },
    "organamnist": {
        "type": "medmnist",
        "data_flag": "organamnist",
        "num_classes": 11,  # 11 body organ classes (CT axial)
        "cache": IMAGENET_ZOO_CACHE,
        "size": 224,
    },
    "flowers": {
        "type": "original",
        "root": f"{IMAGENET_ZOO_CACHE}/oxford-flowers_original",
        "num_classes": 102,
    },
    "dtd": {
        "type": "original",
        "root": f"{IMAGENET_ZOO_CACHE}/dtd_original/dtd",
        "num_classes": 47,
        "fold": 1,  # DTD has 10-fold CV, use fold 1 by default
    },
    "aircraft": {
        "type": "original",
        "root": f"{IMAGENET_ZOO_CACHE}/fgvc-aircraft_original/fgvc-aircraft-2013b/data",
        "num_classes": 100,  # 100 aircraft variants
    },
    "aid": {
        "type": "original",
        "root": f"{IMAGENET_ZOO_CACHE}/AID",
        "num_classes": 30,
        "train_ratio": 0.1,  # 10% train, 90% test (SSL evaluation setting)
    },
    "ade20k": {
        "type": "segmentation",
        "root": f"{IMAGENET_ZOO_CACHE}/ADEChallengeData2016",
        "num_classes": 150,
        "ignore_index": -1,
    },
}

# Convenience lists for different evaluation protocols
DEFAULT_EVAL_DATASETS = ["dtd", "aircraft", "cars", "cifar10", "cifar100", "flowers", "food", "pets"]

OOD_EVAL_DATASETS = ["dtd", "galaxy10", "aid", "chestxray", "retinamnist", "organamnist"]


# ============================================================================
# Transform Functions
# ============================================================================

def get_train_transform(img_size=224, use_deit_aug=False):
    if use_deit_aug:
        # DeiT-style augmentation: RandAugment + RandomErasing
        return v2.Compose([
            v2.RandomResizedCrop(img_size, scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC),
            v2.RandomHorizontalFlip(),
            v2.RandAugment(num_ops=2, magnitude=9),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.RandomErasing(p=0.25),
        ])
    else:
        # DINO-style: simple augmentation only
        return v2.Compose([
            v2.RandomResizedCrop(img_size, scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC),
            v2.RandomHorizontalFlip(),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def get_val_transform(img_size=224):
    resize_size = int(img_size / 0.875)  # 224 -> 256
    return v2.Compose([
        v2.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
        v2.CenterCrop(img_size),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ============================================================================
# Dataset Classes
# ============================================================================

class HuggingFaceDataset(torch.utils.data.Dataset):
    def __init__(self, ds, transform, label_key="label", multi_label=False, num_classes=None):
        self.ds = ds
        self.transform = transform
        self.label_key = label_key
        self.multi_label = multi_label
        self.num_classes = num_classes
        
        # Detect image key
        if 'image' in ds.features:
            self.image_key = 'image'
        elif 'img' in ds.features:
            self.image_key = 'img'
        else:
            raise ValueError(f"Cannot find image key in dataset features: {ds.features.keys()}")
        
        # Auto-detect label key if not found
        if label_key not in ds.features:
            for k in ['label', 'labels', 'fine_label', 'coarse_label', 'variant']:
                if k in ds.features:
                    self.label_key = k
                    break
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, i):
        item = self.ds[i]
        img = item[self.image_key].convert("RGB")
        
        if self.multi_label:
            # Multi-label: convert to multi-hot encoding
            label = torch.zeros(self.num_classes, dtype=torch.float32)
            raw_label = item[self.label_key]
            if isinstance(raw_label, (list, tuple)):
                for idx in raw_label:
                    if 0 <= idx < self.num_classes:
                        label[idx] = 1.0
            else:
                if 0 <= raw_label < self.num_classes:
                    label[raw_label] = 1.0
        else:
            # Single-label
            label = item[self.label_key]
            if isinstance(label, list):
                label = label[0]
            # Handle boolean labels (e.g., PCAM)
            if isinstance(label, bool):
                label = int(label)
        
        return self.transform(img), label


class FlowersDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transform, combine_train_val=False):
        self.root = Path(root)
        self.transform = transform
        
        # Load labels (1-indexed in the file)
        labels_mat = scipy.io.loadmat(self.root / "imagelabels.mat")
        all_labels = labels_mat['labels'].flatten() - 1  # Convert to 0-indexed
        
        # Load split indices
        setid_mat = scipy.io.loadmat(self.root / "setid.mat")
        if split == "train":
            if combine_train_val:
                # Combine train and val for training (standard practice for fine-tuning)
                train_indices = setid_mat['trnid'].flatten() - 1
                val_indices = setid_mat['valid'].flatten() - 1
                indices = list(train_indices) + list(val_indices)
            else:
                indices = setid_mat['trnid'].flatten() - 1
        elif split == "val":
            indices = setid_mat['valid'].flatten() - 1
        elif split == "test":
            indices = setid_mat['tstid'].flatten() - 1
        else:
            raise ValueError(f"Unknown split: {split}")
        
        self.images = []
        self.labels = []
        for idx in indices:
            img_name = f"image_{idx+1:05d}.jpg"
            self.images.append(self.root / "jpg" / img_name)
            self.labels.append(int(all_labels[idx]))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        img = Image.open(self.images[i]).convert("RGB")
        return self.transform(img), self.labels[i]


class DTDDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transform, fold=1):
        self.root = Path(root)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load class names from directory structure
        images_dir = self.root / "images"
        self.classes = sorted([d.name for d in images_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        # Load split file (train1.txt, val1.txt, test1.txt for fold 1)
        split_file = self.root / "labels" / f"{split}{fold}.txt"
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Format: category/image_name.jpg
                    category = line.split('/')[0]
                    self.images.append(self.root / "images" / line)
                    self.labels.append(self.class_to_idx[category])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        img = Image.open(self.images[i]).convert("RGB")
        return self.transform(img), self.labels[i]


class AircraftDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transform):
        self.root = Path(root)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load variant classes (100 classes)
        variants_file = self.root / "variants.txt"
        with open(variants_file, 'r') as f:
            self.classes = [line.strip() for line in f if line.strip()]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        # Map split names: use trainval for training (standard practice)
        split_map = {"train": "trainval", "val": "val", "test": "test"}
        split_name = split_map.get(split, split)
        
        # Load split file
        split_file = self.root / f"images_variant_{split_name}.txt"
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Format: image_id variant_name (variant_name can have spaces)
                    parts = line.split(' ', 1)
                    image_id = parts[0]
                    variant = parts[1] if len(parts) > 1 else ""
                    self.images.append(self.root / "images" / f"{image_id}.jpg")
                    self.labels.append(self.class_to_idx.get(variant, 0))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        img = Image.open(self.images[i]).convert("RGB")
        return self.transform(img), self.labels[i]


class AIDDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transform, train_ratio=0.8, seed=42):
        self.root = Path(root)
        self.transform = transform
        
        # Get all classes from subdirectories
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        # Collect all images with their labels (sorted for deterministic ordering)
        all_images = []
        all_labels = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            class_images = []
            for ext in ["*.jpg", "*.png", "*.tif", "*.jpeg"]:
                class_images.extend(class_dir.glob(ext))
            # Sort to ensure deterministic ordering across runs/filesystems
            class_images = sorted(class_images)
            for img_path in class_images:
                all_images.append(img_path)
                all_labels.append(self.class_to_idx[class_name])
        
        # Create reproducible train/test split per class (stratified)
        rng = random.Random(seed)
        train_images, train_labels = [], []
        test_images, test_labels = [], []
        
        for class_idx in range(len(self.classes)):
            class_imgs = [img for img, lbl in zip(all_images, all_labels) if lbl == class_idx]
            rng.shuffle(class_imgs)
            n_train = int(len(class_imgs) * train_ratio)
            
            train_images.extend(class_imgs[:n_train])
            train_labels.extend([class_idx] * n_train)
            test_images.extend(class_imgs[n_train:])
            test_labels.extend([class_idx] * (len(class_imgs) - n_train))
        
        if split == "train":
            self.images = train_images
            self.labels = train_labels
        else:  # val or test
            self.images = test_images
            self.labels = test_labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        img = Image.open(self.images[i]).convert("RGB")
        return self.transform(img), self.labels[i]


class MedMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, data_flag, split, transform, download=False, root=None, size=224):
        self.transform = transform
        
        # Map split names
        if split == "test":
            split = "test"
        elif split == "val":
            split = "val"
        else:
            split = "train"
        
        # Get the dataset class dynamically
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])
        
        # MedMNIST supports size=28, 64, 128, 224
        if size not in [28, 64, 128, 224]:
            size = 224
        
        self.dataset = DataClass(
            split=split,
            transform=None,  # We apply our own transform
            download=download,
            root=root,
            size=size,
        )
        
        self.labels = self.dataset.labels.squeeze()
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i):
        img, _ = self.dataset[i]
        # MedMNIST returns PIL Image, convert to RGB if grayscale
        img = img.convert("RGB")
        label = int(self.labels[i])
        return self.transform(img), label


# ============================================================================
# Factory Functions
# ============================================================================

def create_dataset(dataset_name, split, img_size=224, use_deit_aug=False, combine_train_val=False):
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset_name]
    
    # Get appropriate transform
    if split == "train":
        transform = get_train_transform(img_size, use_deit_aug)
    else:
        transform = get_val_transform(img_size)
    
    # Create dataset based on type
    if config["type"] == "huggingface":
        hf_split = config.get("train_split" if split == "train" else "val_split", split)
        ds = hf_load_dataset(
            config["path"],
            config.get("config"),
            cache_dir=config.get("cache"),
            split=hf_split,
            trust_remote_code=True,
        )
        return HuggingFaceDataset(
            ds, 
            transform, 
            label_key=config.get("label_key", "label"),
            multi_label=config.get("multi_label", False),
            num_classes=config.get("num_classes"),
        )
    
    elif config["type"] == "original":
        return _create_original_dataset(dataset_name, config, split, transform, combine_train_val)
    
    elif config["type"] == "medmnist":
        # Use config size if specified (e.g., 28 for faster download)
        medmnist_size = config.get("size", img_size)
        return MedMNISTDataset(
            data_flag=config["data_flag"],
            split=split,
            transform=transform,
            download=True,
            root=config.get("cache", IMAGENET_ZOO_CACHE),
            size=medmnist_size,
        )
    
    else:
        raise ValueError(f"Unknown dataset type: {config['type']}")


def _create_original_dataset(dataset_name, config, split, transform, combine_train_val=False):
    if dataset_name == "flowers":
        return FlowersDataset(config["root"], split, transform, combine_train_val=combine_train_val)
    elif dataset_name == "dtd":
        return DTDDataset(config["root"], split, transform, fold=config.get("fold", 1))
    elif dataset_name == "aircraft":
        return AircraftDataset(config["root"], split, transform)
    elif dataset_name == "aid":
        # Always use seed=42 for reproducible train/test split
        return AIDDataset(config["root"], split, transform, train_ratio=config.get("train_ratio", 0.8), seed=42)
    else:
        raise ValueError(f"Unknown original dataset: {dataset_name}")


# ============================================================================
# Utility Functions
# ============================================================================

def get_num_classes(dataset_name):
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASET_CONFIGS[dataset_name]["num_classes"]


def is_multi_label(dataset_name):
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASET_CONFIGS[dataset_name].get("multi_label", False)


def get_few_shot_indices(dataset, shots, num_classes):
    if shots <= 0:
        return None
    
    indices = []
    label_key = None
    for k in ['label', 'labels', 'fine_label', 'coarse_label', 'variant']:
        if k in dataset.features:
            label_key = k
            break
    
    if label_key is None:
        raise ValueError(f"No label key found in dataset features: {list(dataset.features.keys())}")
    
    labels = np.array(dataset[label_key])
    for c in range(num_classes):
        c_indices = np.where(labels == c)[0]
        if len(c_indices) > 0:
            selected = np.random.choice(c_indices, min(shots, len(c_indices)), replace=False)
            indices.extend(selected.tolist())

    return indices

