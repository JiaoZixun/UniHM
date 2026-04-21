import torch
import torch.nn as nn
import torch.nn.functional as F


class _TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t_norm: torch.Tensor) -> torch.Tensor:
        return self.net(t_norm.unsqueeze(-1))


class DiffusionDynamics(nn.Module):
    """Conditional DDPM-style one-step transition model.

    Inputs are history features (z_t, o_t). Targets are x_{t+1}=[z_{t+1}, o_{t+1}].
    """

    def __init__(
        self,
        latent_dim: int,
        obj_state_dim: int,
        d_model: int = 256,
        n_layer: int = 2,
        diffusion_steps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()
        self.in_dim = latent_dim + obj_state_dim
        self.latent_dim = latent_dim
        self.obj_state_dim = obj_state_dim
        self.out_dim = latent_dim + obj_state_dim
        self.backend = "diffusion"
        self.diffusion_steps = diffusion_steps

        self.in_proj = nn.Linear(self.in_dim, d_model)
        self.cond_gru = nn.GRU(d_model, d_model, num_layers=n_layer, batch_first=True)
        self.cond_norm = nn.LayerNorm(d_model)

        self.time_mlp = _TimeEmbedding(d_model)
        self.denoiser = nn.Sequential(
            nn.Linear(self.out_dim + d_model + d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, self.out_dim),
        )

        betas = torch.linspace(beta_start, beta_end, diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

    def _condition(self, z_seq: torch.Tensor, o_seq: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_seq, o_seq], dim=-1)
        h = self.in_proj(x)
        h, _ = self.cond_gru(h)
        return self.cond_norm(h)

    def _predict_eps(self, x_t: torch.Tensor, h: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        t_norm = t_idx.to(x_t.dtype) / max(1, self.diffusion_steps - 1)
        t_emb = self.time_mlp(t_norm)
        inp = torch.cat([x_t, h, t_emb], dim=-1)
        return self.denoiser(inp)

    def training_loss(self, z_in: torch.Tensor, o_in: torch.Tensor, z_gt: torch.Tensor, o_gt: torch.Tensor):
        h = self._condition(z_in, o_in)
        x0 = torch.cat([z_gt, o_gt], dim=-1)

        b, t, _ = x0.shape
        t_idx = torch.randint(0, self.diffusion_steps, (b, t), device=x0.device)
        eps = torch.randn_like(x0)

        sqrt_ab = self.sqrt_alpha_bars[t_idx].unsqueeze(-1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bars[t_idx].unsqueeze(-1)
        x_t = sqrt_ab * x0 + sqrt_1mab * eps

        eps_pred = self._predict_eps(x_t, h, t_idx)
        loss = F.mse_loss(eps_pred, eps)

        x0_hat = (x_t - sqrt_1mab * eps_pred) / (sqrt_ab + 1e-8)
        z_hat, o_hat = x0_hat[..., : self.latent_dim], x0_hat[..., self.latent_dim :]
        lz = F.mse_loss(z_hat, z_gt)
        lo = F.mse_loss(o_hat, o_gt)

        return {"loss": loss, "z_loss": lz, "o_loss": lo}

    @torch.no_grad()
    def forward(self, z_seq: torch.Tensor, o_seq: torch.Tensor):
        h = self._condition(z_seq, o_seq)
        x_t = torch.randn(z_seq.size(0), z_seq.size(1), self.out_dim, device=z_seq.device, dtype=z_seq.dtype)

        for step in reversed(range(self.diffusion_steps)):
            t_idx = torch.full((x_t.size(0), x_t.size(1)), step, device=x_t.device, dtype=torch.long)
            eps_theta = self._predict_eps(x_t, h, t_idx)

            alpha_t = self.alphas[step]
            alpha_bar_t = self.alpha_bars[step]
            beta_t = self.betas[step]

            coeff = beta_t / torch.sqrt(1.0 - alpha_bar_t)
            mean = self.sqrt_recip_alphas[step] * (x_t - coeff * eps_theta)

            if step > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(beta_t) * noise
            else:
                x_t = mean

        z_next = x_t[..., : self.latent_dim]
        o_next = x_t[..., self.latent_dim :]
        return {"z_next": z_next, "o_next": o_next}
