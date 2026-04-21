import torch
import torch.nn as nn


class TransformerDynamics(nn.Module):
    """Autoregressive next-state predictor with a causal Transformer encoder."""

    def __init__(
        self,
        latent_dim: int,
        obj_state_dim: int,
        d_model: int = 256,
        n_layer: int = 4,
        n_head: int = 8,
        max_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = latent_dim + obj_state_dim
        self.latent_dim = latent_dim
        self.obj_state_dim = obj_state_dim
        self.backend = "transformer"

        self.in_proj = nn.Linear(self.in_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.norm = nn.LayerNorm(d_model)
        self.out_latent = nn.Linear(d_model, latent_dim)
        self.out_obj = nn.Linear(d_model, obj_state_dim)

    def forward(self, z_seq: torch.Tensor, o_seq: torch.Tensor):
        x = torch.cat([z_seq, o_seq], dim=-1)
        h = self.in_proj(x)
        t = h.size(1)
        if t > self.pos_emb.size(1):
            raise ValueError(f"Sequence length {t} exceeds max_len={self.pos_emb.size(1)}")
        h = h + self.pos_emb[:, :t]

        causal_mask = torch.triu(
            torch.ones((t, t), device=h.device, dtype=torch.bool),
            diagonal=1,
        )
        h = self.encoder(h, mask=causal_mask)
        h = self.norm(h)

        z_next = self.out_latent(h)
        o_next = self.out_obj(h)
        return {"z_next": z_next, "o_next": o_next}
