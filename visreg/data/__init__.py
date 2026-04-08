from .imagenet import ImageNetDataset
from .imagenette import ImagenetteDataset
from .multicrop import DinoMultiCrop, PILGaussianBlur
from .multicrop_kornia import DinoMultiCropKornia, build_kornia_aug_pipeline

DATASET_REGISTRY = {
    "imagenet": ImageNetDataset,
    "imagenette": ImagenetteDataset,
}


def build_dataset(name: str, **kwargs):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name](**kwargs)


__all__ = [
    "ImageNetDataset",
    "ImagenetteDataset",
    "DinoMultiCrop",
    "DinoMultiCropKornia",
    "build_kornia_aug_pipeline",
    "PILGaussianBlur",
    "build_dataset",
]
