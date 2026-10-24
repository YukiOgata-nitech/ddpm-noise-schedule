"""2 次元データ用の小さな DDPM。"""

import warnings

import torch
import torch.nn as nn

T = 200  # 拡散のステップ数
SAMPLE_LIMIT = 50.0  # 生成した点がこの範囲（絶対値）を超えたら知らせる。データの座標はおおよそ -10〜15


def get_betas(name: str, T: int = T) -> torch.Tensor:
    if name == "linear":
        return torch.linspace(1e-4, 0.02, T)
    raise ValueError(f"unknown schedule: {name}")


class Denoiser(nn.Module):
    """x_t と t から、加えたノイズを予測する。"""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tt = (t.float() / T).unsqueeze(1)
        return self.net(torch.cat([x, tt], dim=1))


def train(betas, data, steps: int = 3000, batch: int = 256, lr: float = 1e-3, seed: int = 0):
    torch.manual_seed(seed)
    alphas_bar = torch.cumprod(1 - betas, dim=0)
    model = Denoiser()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x0_all = torch.as_tensor(data, dtype=torch.float32)
    for _ in range(steps):
        idx = torch.randint(0, len(x0_all), (batch,))
        x0 = x0_all[idx]
        t = torch.randint(0, len(betas), (batch,))
        noise = torch.randn_like(x0)
        ab = alphas_bar[t].unsqueeze(1)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        loss = ((model(xt, t) - noise) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def sample(model, betas, n: int = 2000, seed: int = 0):
    torch.manual_seed(seed)
    alphas = 1 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    x = torch.randn(n, 2)
    for t in reversed(range(len(betas))):
        eps = model(x, torch.full((n,), t))
        x = (x - betas[t] / (1 - alphas_bar[t]).sqrt() * eps) / alphas[t].sqrt()
        if t > 0:
            x = x + betas[t].sqrt() * torch.randn_like(x)
    outside = int((x.abs() > SAMPLE_LIMIT).any(dim=1).sum())
    if outside:
        warnings.warn(f"{outside} / {n} samples are outside [-{SAMPLE_LIMIT:g}, {SAMPLE_LIMIT:g}]")
    return x.numpy()
