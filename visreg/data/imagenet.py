import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from datasets import load_dataset

from pathlib import Path

from .multicrop import DinoMultiCrop
from .multicrop_kornia import DinoMultiCropKornia


# None => use the standard HuggingFace datasets cache (honors HF_HOME / HF_DATASETS_CACHE).
# Override per-call via the `cache_dir` argument.
IMAGENET_CACHE_DIR = None


class ImageNetDataset(Dataset):
    """ImageNet-1k dataset with optional multi-crop augmentation."""
    
    def __init__(
        self,
        split: str,
        *,
        n_global: int | None = None,
        n_local: int | None = None,
        global_img_size: int = 224,
        local_img_size: int = 98,
        cache_dir: str | None = None,
        multicrop_backend: str = "torchvision",
    ):
        self.split = split
        cache = cache_dir or IMAGENET_CACHE_DIR
        self.ds = load_dataset("ILSVRC/imagenet-1k", cache_dir=cache, split=split)
        print(f"Loaded ImageNet-1k: 1000 classes, {len(self.ds)} samples")

        self.multi_crop = None
        if n_global is not None and n_local is not None:
            crop_cls = DinoMultiCropKornia if multicrop_backend == "kornia" else DinoMultiCrop
            self.multi_crop = crop_cls(
                n_global=n_global,
                n_local=n_local,
                global_size=global_img_size,
                local_size=local_img_size,
                global_scale=(0.3, 1.0),
                local_scale=(0.05, 0.3),
            )

        self.test = v2.Compose(
            [
                v2.Resize(global_img_size),
                v2.CenterCrop(global_img_size),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self, i):
        item = self.ds[i]
        img = item["image"].convert("RGB")
        if self.multi_crop is not None:
            return self.multi_crop(img), item["label"]
        return self.test(img).unsqueeze(0), item["label"]

    def __len__(self):
        return len(self.ds)
    
    @property
    def num_classes(self) -> int:
        return 1000

