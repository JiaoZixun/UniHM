from .mamba import MambaDynamics
from .transformer import TransformerDynamics
from .diffusion import DiffusionDynamics


__all__ = ["MambaDynamics", "TransformerDynamics", "DiffusionDynamics", "build_dynamics"]


def build_dynamics(
    backend: str,
    latent_dim: int,
    obj_state_dim: int,
    d_model: int = 256,
    n_layer: int = 4,
    n_head: int = 8,
    max_len: int = 256,
    diffusion_steps: int = 50,
):
    backend = backend.lower()
    if backend == "mamba":
        return MambaDynamics(latent_dim=latent_dim, obj_state_dim=obj_state_dim, d_model=d_model, n_layer=n_layer)
    if backend == "transformer":
        return TransformerDynamics(
            latent_dim=latent_dim,
            obj_state_dim=obj_state_dim,
            d_model=d_model,
            n_layer=n_layer,
            n_head=n_head,
            max_len=max_len,
        )
    if backend == "diffusion":
        return DiffusionDynamics(
            latent_dim=latent_dim,
            obj_state_dim=obj_state_dim,
            d_model=d_model,
            n_layer=max(1, min(4, n_layer // 2)),
            diffusion_steps=diffusion_steps,
        )
    raise ValueError(f"Unsupported dynamics backend: {backend}")
