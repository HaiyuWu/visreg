"""
DINO-style multi-crop augmentation (global + local views).

Key idea:
- For each image, produce N_global "global" crops (e.g. 224, scale 0.3-1.0)
- and N_local "local" crops (e.g. 98, scale 0.05-0.3)
- with the SAME color/geom augments applied to both crop types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PIL import ImageFilter
import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import InterpolationMode


class PILGaussianBlur:
    """PIL-based Gaussian blur like BarlowTwins/DINO."""
    
    def __init__(self, sigma_min: float = 0.1, sigma_max: float = 2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def __call__(self, img):
        sigma = torch.empty(1).uniform_(self.sigma_min, self.sigma_max).item()
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))


@dataclass
class DinoMultiCrop:
    """
    Multi-crop transform producing [global..., local...] tensors.

    Defaults:
    - Global: RandomResizedCrop(224, scale=(0.3, 1.0))
    - Local:  RandomResizedCrop(98,  scale=(0.05, 0.3))
    - RandomHorizontalFlip(p=0.5)
    - ColorJitter(p=0.8): brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
    - RandomGrayscale(p=0.2)
    - GaussianBlur(p=0.5)
    - RandomSolarize(p=0.2, threshold=128)
    - Normalize(ImageNet)
    """

    n_global: int = 2
    n_local: int = 6
    global_size: int = 224
    local_size: int = 98
    global_scale: Tuple[float, float] = (0.3, 1.0)
    local_scale: Tuple[float, float] = (0.05, 0.3)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    hflip_p: float = 0.5
    color_jitter_p: float = 0.8
    cj_brightness: float = 0.4
    cj_contrast: float = 0.4
    cj_saturation: float = 0.2
    cj_hue: float = 0.1
    grayscale_p: float = 0.2
    blur_p: float = 0.5
    # 新增：高斯模糊的具体参数
    blur_kernel_size: int = 9
    blur_sigma: Tuple[float, float] = (0.1, 2.0)
    solarize_p: float = 0.2
    solarize_threshold: int = 0.5

    _g_tf: Optional[v2.Transform] = field(default=None, init=False, repr=False)
    _l_tf: Optional[v2.Transform] = field(default=None, init=False, repr=False)
    return_tensor: bool = True

    def __post_init__(self) -> None:
        if self.n_global < 0 or self.n_local < 0:
            raise ValueError(
                f"n_global/n_local must be non-negative, got {self.n_global}/{self.n_local}"
            )
        if self.n_global > 0:
            self._g_tf = self._make_crop_transform(
                size=self.global_size, scale=self.global_scale
            )
        if self.n_local > 0:
            self._l_tf = self._make_crop_transform(
                size=self.local_size, scale=self.local_scale
            )

    def _shared_aug(self) -> v2.Transform:
        return v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),

                v2.RandomHorizontalFlip(p=self.hflip_p),
                v2.RandomApply(
                    [
                        v2.ColorJitter(
                            brightness=self.cj_brightness,
                            contrast=self.cj_contrast,
                            saturation=self.cj_saturation,
                            hue=self.cj_hue,
                        )
                    ],
                    p=self.color_jitter_p,
                ),
                v2.RandomGrayscale(p=self.grayscale_p),
                
                # 彻底解耦，读取 dataclass 属性
                v2.RandomApply(
                    [v2.GaussianBlur(kernel_size=self.blur_kernel_size, sigma=self.blur_sigma)], 
                    p=self.blur_p
                ),
                
                # 彻底解耦，读取 dataclass 属性
                v2.RandomApply(
                    [v2.RandomSolarize(threshold=self.solarize_threshold)],
                    p=self.solarize_p,
                ),
                
                v2.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def _make_crop_transform(self, *, size: int, scale: Tuple[float, float]) -> v2.Transform:
        return v2.Compose(
            [
                v2.RandomResizedCrop(
                    size, scale=scale, interpolation=InterpolationMode.BICUBIC
                ),
                self._shared_aug(),
            ]
        )

    def __call__(self, img):
        """
        Returns:
        - If crop sizes differ: List[Tensor] length (n_global + n_local)
        - If all crops same size and return_tensor=True: Tensor [V, 3, H, W]
        """
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

