import torch
import torch.nn as nn


class SlicedWasserstein(nn.Module):
    def __init__(self, num_projections=256):
        super().__init__()
        self.K = num_projections

    def forward(self, z):
        V, B, D = z.shape
        W = torch.randn(D, self.K, device=z.device, dtype=z.dtype)
        W = W / (W.norm(dim=0, keepdim=True) + 1e-6)
        p = torch.matmul(z, W)
        p_sorted, _ = p.sort(dim=1)
        quantiles = torch.arange(1, B + 1, device=z.device, dtype=z.dtype) / (B + 1)
        quantiles = quantiles.clamp(min=1e-4, max=1-1e-4)
        target = torch.distributions.Normal(0, 1).icdf(quantiles)
        target = target.view(1, B, 1)
        return (p_sorted - target).pow(2).mean()

