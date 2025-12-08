import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ActionConditionedDiffusion(nn.Module):
    """
    Simple DDPM-style network that predicts noise on low-dimensional actions
    conditioned on state features (no images). It uses a small MLP with
    timestep embeddings injected at every block.
    """

    def __init__(
        self,
        action_dim: int = 3,
        cond_dim: int = 9,
        hidden_dim: int = 256,
        time_embed_dim: int = 128,
        num_layers: int = 3,
        num_diffusion_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.cond_dim = cond_dim
        self.time_embed_dim = time_embed_dim
        self.num_diffusion_steps = num_diffusion_steps

        betas = torch.linspace(beta_start, beta_end, num_diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        blocks = []
        in_dim = action_dim + cond_dim + time_embed_dim
        for _ in range(num_layers):
            blocks.append(nn.Linear(in_dim, hidden_dim))
            blocks.append(nn.SiLU())
            in_dim = hidden_dim
        blocks.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*blocks)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Predict noise for actions at timestep t."""
        t_emb = timestep_embedding(timesteps, self.time_embed_dim)
        t_emb = self.time_mlp(t_emb)
        # match batch dims
        if t_emb.dim() == 1:
            t_emb = t_emb.unsqueeze(0)
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        h = torch.cat([x_t, cond, t_emb], dim=-1)
        return self.net(h)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Generate actions by reverse diffusion."""
        device = cond.device
        x = torch.randn(num_samples, self.action_dim, device=device)
        for t in reversed(range(self.num_diffusion_steps)):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            eps_theta = self.forward(x, t_batch, cond)
            beta_t = self.betas[t]
            alpha_t = 1.0 - beta_t
            alpha_bar_t = self.alphas_cumprod[t]
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)
            sqrt_alpha_t = torch.sqrt(alpha_t)

            mean = (1 / torch.sqrt(alpha_t)) * (x - beta_t / sqrt_one_minus_alpha_bar * eps_theta)
            if t > 0:
                noise = torch.randn_like(x)
                var = beta_t * (1.0 - self.alphas_cumprod[t - 1]) / (1.0 - alpha_bar_t)
                x = mean + torch.sqrt(var) * noise
            else:
                x = mean
        return x
