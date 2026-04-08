import torch
import torch.nn as nn


class BarlowTwins(nn.Module):
    def __init__(self, lambda_off_diag: float = 0.0051):
        super().__init__()
        self.lambda_off = lambda_off_diag
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 2:
            z = z.unsqueeze(0)
        V, B, D = z.shape
        z_norm = (z - z.mean(dim=1, keepdim=True)) / (z.std(dim=1, keepdim=True) + 1e-6)
        
        total = z.new_zeros(())
        num_pairs = 0
        if V >= 2:
            for i in range(V - 1):
                C = (z_norm[i].T @ z_norm[i + 1]) / B
                on_diag = (C.diag() - 1).pow(2).sum()
                off_diag = C.pow(2).sum() - C.diag().pow(2).sum()
                total = total + on_diag + self.lambda_off * off_diag
                num_pairs += 1
        else:
            C = (z_norm[0].T @ z_norm[0]) / B
            on_diag = (C.diag() - 1).pow(2).sum()
            off_diag = C.pow(2).sum() - C.diag().pow(2).sum()
            total = total + on_diag + self.lambda_off * off_diag
            num_pairs = 1
        return total / max(1, num_pairs)

