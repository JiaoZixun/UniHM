import torch
import torch.nn as nn


class MambaDynamics(nn.Module):
    """Predict next latent and next object state in closed loop.

    If mamba_ssm is unavailable, falls back to GRU with same interface.
    """
    def __init__(self, latent_dim: int, obj_state_dim: int, d_model: int = 256, n_layer: int = 4):
        super().__init__()
        self.in_dim = latent_dim + obj_state_dim
        self.latent_dim = latent_dim
        self.obj_state_dim = obj_state_dim
        self.in_proj = nn.Linear(self.in_dim, d_model)

        self.backend = "gru"
        try:
            from mamba_ssm import Mamba  # type: ignore
            self.blocks = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layer)])
            self.norm = nn.LayerNorm(d_model)
            self.backend = "mamba"
        except Exception:
            self.gru = nn.GRU(d_model, d_model, num_layers=2, batch_first=True)
            self.norm = nn.LayerNorm(d_model)

        self.out_latent = nn.Linear(d_model, latent_dim)
        self.out_obj = nn.Linear(d_model, obj_state_dim)

    def forward(self, z_seq: torch.Tensor, o_seq: torch.Tensor):
        """z_seq/o_seq: (B, T, D), predict next for each step.

        returns dict with predicted z_next/o_next as (B, T, D)
        """
        x = torch.cat([z_seq, o_seq], dim=-1)
        h = self.in_proj(x)

        if self.backend == "mamba":
            for blk in self.blocks:
                h = h + blk(h)
            h = self.norm(h)
        else:
            h, _ = self.gru(h)
            h = self.norm(h)

        z_next = self.out_latent(h)
        o_next = self.out_obj(h)
        return {"z_next": z_next, "o_next": o_next}
