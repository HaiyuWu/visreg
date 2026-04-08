import torch.nn as nn
import torch.nn.functional as F


class VICReg(nn.Module):
    def __init__(self, var_weight=25.0, cov_weight=1.0, gamma=1.0):
        super().__init__()
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.gamma = gamma

    def forward(self, z):
        V, B, D = z.shape
        z_flat = z.reshape(V * B, D)
        std = z_flat.std(dim=0) + 1e-6
        var_loss = F.relu(self.gamma - std).pow(2).mean()
        z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)
        cov = (z_centered.T @ z_centered) / (V * B - 1)
        off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        cov_loss = off_diag / D
        return self.var_weight * var_loss + self.cov_weight * cov_loss

