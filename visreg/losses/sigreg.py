import torch
import torch.nn as nn


class SIGReg(nn.Module):
    def __init__(self, *, knots: int = 17, t_max: float = 3.0, num_projections: int = 256):
        super().__init__()
        self.num_projections = int(num_projections)
        t = torch.linspace(0, float(t_max), int(knots), dtype=torch.float32)
        dt = float(t_max) / (int(knots) - 1)
        weights = torch.full((int(knots),), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        _, B, D = proj.shape

        A = torch.randn(D, self.num_projections, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0, keepdim=True) + 1e-12)

        t = self.t.to(device=proj.device, dtype=proj.dtype)
        phi = self.phi.to(device=proj.device, dtype=proj.dtype)
        weights = self.weights.to(device=proj.device, dtype=proj.dtype)

        x_t = (proj @ A).unsqueeze(-1) * t
        err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ weights) * B
        loss = statistic.mean()

        return loss

