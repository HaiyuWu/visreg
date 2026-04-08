import torch
import torch.nn as nn
import timm
from torchvision.ops import MLP


VIT_CONFIGS = {
    "vit_s": ("vit_small_patch8_224", 0.1),   # embed_dim=384, patch8
    "vit_b": ("vit_base_patch16_224", 0.1),   # embed_dim=768, patch16
    "vit_l": ("vit_large_patch14_224", 0.1),  # embed_dim=1024, patch14
    "vit_h": ("vit_huge_patch14_224", 0.2),   # embed_dim=1280, patch14
    "vit_g": ("vit_giant_patch14_224", 0.3),  # embed_dim=1408, patch14
}


class ViTEncoder(nn.Module):
    def __init__(self, model_name: str = "vit_l", proj_dim: int | None = None, pretrained: bool = False, proj_gelu: bool = False):
        super().__init__()
        if model_name not in VIT_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(VIT_CONFIGS.keys())}")
        
        timm_name, drop_path = VIT_CONFIGS[model_name]
        self.model_name = model_name
        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=drop_path,
            dynamic_img_size=True,
        )
        self.embed_dim = int(getattr(self.backbone, "embed_dim"))
        # Default proj_dim to 1/3 of embed_dim if not specified
        if proj_dim is None:
            proj_dim = self.embed_dim // 3
        self.proj_dim = int(proj_dim)
        activation_layer = nn.GELU if proj_gelu else nn.ReLU
        self.proj = MLP(self.embed_dim, [2048, 2048, self.proj_dim], norm_layer=nn.BatchNorm1d, activation_layer=activation_layer)

    def _extract_features(self, x: torch.Tensor):
        _, intermediates = self.backbone.forward_intermediates(
            x, 
            indices=[-2, -1],  # Last two layers
            return_prefix_tokens=True,
            norm=True,
        )
        
        features = []
        for layer_out in intermediates:
            _, prefix_tokens = layer_out
            cls_token = prefix_tokens[:, 0, :]
            features.append(cls_token)
        
        last_cls = features[-1]
        concat_cls = torch.cat(features, dim=-1)
        return last_cls, concat_cls

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            V = len(x)
            if V == 0:
                raise ValueError("Received empty crop list")
            B = x[0].shape[0]
            for i, crop in enumerate(x):
                if crop.dim() != 4:
                    raise ValueError(f"Crop {i} must be [B,C,H,W], got shape {tuple(crop.shape)}")
                if crop.shape[0] != B:
                    raise ValueError("All crops must have the same batch size B")

            by_size = {}
            for i, crop in enumerate(x):
                key = (int(crop.shape[-2]), int(crop.shape[-1]))
                by_size.setdefault(key, []).append(i)

            emb_per_crop = [None] * V
            proj_per_crop = [None] * V
            for _, idxs in by_size.items():
                batch = torch.cat([x[i] for i in idxs], dim=0)
                last_cls, concat_cls = self._extract_features(batch)
                p = self.proj(last_cls)
                e_chunks = concat_cls.split(B, dim=0)
                p_chunks = p.split(B, dim=0)
                for j, i in enumerate(idxs):
                    emb_per_crop[i] = e_chunks[j]
                    proj_per_crop[i] = p_chunks[j]

            emb_vb = torch.stack(emb_per_crop, dim=0)
            emb_cat = emb_vb.transpose(0, 1).reshape(B * V, -1)

            proj = torch.stack(proj_per_crop, dim=0)
            return emb_cat, proj
        else:
            N, V = x.shape[:2]
            last_cls, concat_cls = self._extract_features(x.flatten(0, 1))
            proj = self.proj(last_cls).reshape(N, V, -1).transpose(0, 1)
            return concat_cls, proj

