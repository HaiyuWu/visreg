import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from datasets import load_dataset
from .multicrop import DinoMultiCrop
from .multicrop_kornia import DinoMultiCropKornia


class ImagenetteDataset(Dataset):
    def __init__(
        self,
        split,
        *,
        n_global: int | None = None,
        n_local: int | None = None,
        global_img_size: int = 128,
        local_img_size: int = 64,
        multicrop_backend: str = "torchvision",
    ):
        self.split = split
        self.ds = load_dataset("frgfm/imagenette", "160px", split=split, trust_remote_code=True)

        self.multi_crop = None
        if n_global is not None and split == "train":
            if multicrop_backend == "kornia":
                self.multi_crop = DinoMultiCropKornia(
                    n_global=n_global,
                    n_local=n_local or 0,
                    global_size=global_img_size,
                    local_size=local_img_size,
                    global_scale=(0.08, 1.0),
                    local_scale=(0.05, 0.3),
                )
            else:
                self.multi_crop = DinoMultiCrop(
                    n_global=n_global,
                    n_local=n_local or 0,
                    global_size=global_img_size,
                    local_size=local_img_size,
                    global_scale=(0.08, 1.0),
                    local_scale=(0.05, 0.3),
                    cj_brightness=0.8,
                    cj_contrast=0.8,
                    cj_saturation=0.8,
                    cj_hue=0.2,
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
    def num_classes(self):
        return self.ds.features["label"].num_classes

