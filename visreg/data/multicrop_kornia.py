from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import InterpolationMode

import kornia.augmentation as K


@dataclass
class DinoMultiCropKornia:
    n_global: int = 2
    n_local: int = 6
    global_size: int = 224
    local_size: int = 98
    global_scale: Tuple[float, float] = (0.3, 1.0)
    local_scale: Tuple[float, float] = (0.05, 0.3)
    hflip_p: float = 0.5

    _g_tf: Optional[v2.Transform] = field(default=None, init=False, repr=False)
    _l_tf: Optional[v2.Transform] = field(default=None, init=False, repr=False)
    return_tensor: bool = True

    def __post_init__(self) -> None:
        if self.n_global < 0 or self.n_local < 0:
            raise ValueError(
                f"n_global/n_local must be non-negative, got {self.n_global}/{self.n_local}"
            )
        if self.n_global > 0:
            self._g_tf = self._make_crop_transform(size=self.global_size, scale=self.global_scale)
        if self.n_local > 0:
            self._l_tf = self._make_crop_transform(size=self.local_size, scale=self.local_scale)

    def _make_crop_transform(self, *, size: int, scale: Tuple[float, float]) -> v2.Transform:
        return v2.Compose([
            v2.RandomResizedCrop(size, scale=scale, interpolation=InterpolationMode.BICUBIC),
            v2.RandomHorizontalFlip(p=self.hflip_p),
            v2.ToImage(),
        ])

    def __call__(self, img):
        crops: List[torch.Tensor] = []
        if self.n_global > 0:
            assert self._g_tf is not None
            for _ in range(self.n_global):
                crops.append(self._g_tf(img))
        if self.n_local > 0:
            assert self._l_tf is not None
            for _ in range(self.n_local):
                crops.append(self._l_tf(img))
        if self.return_tensor and len(crops) > 0:
            h0, w0 = int(crops[0].shape[-2]), int(crops[0].shape[-1])
            if all(int(c.shape[-2]) == h0 and int(c.shape[-1]) == w0 for c in crops):
                return torch.stack(crops, dim=0)
        return crops

    @property
    def num_crops(self) -> int:
        return int(self.n_global) + int(self.n_local)


def build_kornia_aug_pipeline(
    cj_brightness: float = 0.4,
    cj_contrast: float = 0.4,
    cj_saturation: float = 0.2,
    cj_hue: float = 0.1,
    color_jitter_p: float = 0.8,
    grayscale_p: float = 0.2,
    blur_kernel_size: int = 9,
    blur_sigma: Tuple[float, float] = (0.1, 2.0),
    blur_p: float = 0.5,
    solarize_p: float = 0.2,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> K.AugmentationSequential:
    return K.AugmentationSequential(
        K.ColorJitter(
            brightness=cj_brightness, contrast=cj_contrast,
            saturation=cj_saturation, hue=cj_hue,
            p=color_jitter_p,
        ),
        K.RandomGrayscale(
            rgb_weights=torch.tensor([0.299, 0.587, 0.114]),
            p=grayscale_p,
        ),
        K.RandomGaussianBlur(
            kernel_size=(blur_kernel_size, blur_kernel_size),
            sigma=blur_sigma,
            p=blur_p,
            separable=False,
        ),
        K.RandomSolarize(thresholds=0.0, additions=0.0, p=solarize_p),
        K.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        same_on_batch=False,
    )
