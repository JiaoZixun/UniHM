import torch
import torch.nn as nn
from typing import List, Optional, Dict

from UniHM.vqvae.encoder import MLPEncoder
from UniHM.vqvae.decoder import MLPDecoder


class TeacherFusionEncoder(nn.Module):
    """Teacher encoder for mano + object geometry.

    Inputs:
      mano: (B, Dm)
      object_feat: (B, Do)
    Output:
      mu, logvar: (B, Dz)
    """
    def __init__(self, mano_dim: int, object_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.mano_encoder = MLPEncoder(mano_dim, hidden_dim, 2, hidden_dim, embedding_dim=hidden_dim)
        self.object_proj = nn.Sequential(
            nn.Linear(object_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, mano: torch.Tensor, object_feat: torch.Tensor):
        h_m = self.mano_encoder(mano)
        h_o = self.object_proj(object_feat)
        h = self.fuse(torch.cat([h_m, h_o], dim=-1))
        return self.mu(h), self.logvar(h)


class MultiDecoderVAE(nn.Module):
    """Shared-latent VAE with multiple robot decoders.

    Current version uses z_shared only (as requested).
    """
    def __init__(
        self,
        mano_dim: int,
        object_dim: int,
        hidden_dim: int,
        latent_dim: int,
        decoder_out_dims: List[int],
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.teacher_encoder = TeacherFusionEncoder(mano_dim, object_dim, hidden_dim, latent_dim)
        self.decoders = nn.ModuleList([
            MLPDecoder(latent_dim, hidden_dim, 2, hidden_dim, out_channels=d) for d in decoder_out_dims
        ])

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_all(self, z: torch.Tensor) -> List[torch.Tensor]:
        return [dec(z).squeeze(-1) for dec in self.decoders]

    def forward(self, mano: torch.Tensor, object_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.teacher_encoder(mano, object_feat)
        z = self.reparameterize(mu, logvar)
        preds = self.decode_all(z)
        return {"mu": mu, "logvar": logvar, "z": z, "preds": preds}


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
