import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VISReg(nn.Module):
    def __init__(self, num_projections: int = 256, scale_weight: float = 1.0, shape_weight: float = 1.0, center_weight: float = 1.0):
        super().__init__()
        self.K = num_projections
        self._cached_B = -1
        self._cached_target = None
        self.scale_weight = scale_weight
        self.shape_weight = shape_weight
        self.center_weight = center_weight

    def _get_target(self, B: int, device) -> torch.Tensor:
        if self._cached_B != B:
            q = torch.linspace(1, B, B, device=device, dtype=torch.float32) / (B + 1)
            self._cached_target = torch.erfinv(2 * q - 1).mul_(math.sqrt(2))
            self._cached_B = B
        return self._cached_target.to(device=device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        _, B, D = z.shape

        mu = z.mean(dim=1, keepdim=True)
        center_loss = mu.pow(2).mean()

        z_centered = z - mu
        std = z_centered.norm(dim=1).div(math.sqrt(B)).clamp_min(1e-6)
        scale_loss = (std - 1.0).pow(2).mean()

        z_norm = z_centered / std.detach().unsqueeze(1)
        W = F.normalize(torch.randn(D, self.K, device=z.device, dtype=z.dtype), dim=0)
        p_sorted = (z_norm @ W).sort(dim=1).values
        target = self._get_target(B, z.device).view(1, B, 1)
        shape_loss = (p_sorted - target).pow(2).mean()

        return self.scale_weight * scale_loss + self.shape_weight * shape_loss + self.center_weight * center_loss