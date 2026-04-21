import argparse
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from UniHM.SFT.utils import build_seq_dataloaders
from UniHM.vae.multi_vae import MultiDecoderVAE
from UniHM.dynamics import build_dynamics
from UniHM.visualization.training_viz import plot_losses


def object_feature_from_pointcloud(pc: torch.Tensor) -> torch.Tensor:
    mu = pc.mean(dim=1)
    std = pc.std(dim=1)
    return torch.cat([mu, std], dim=-1)


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vae_ckpt = torch.load(args.vae_ckpt, map_location="cpu")
    vae = MultiDecoderVAE(
        mano_dim=vae_ckpt["mano_dim"],
        object_dim=6,
        hidden_dim=vae_ckpt["hidden_dim"],
        latent_dim=vae_ckpt["latent_dim"],
        decoder_out_dims=vae_ckpt["out_dims"],
    ).to(device)
    vae.load_state_dict(vae_ckpt["model"], strict=True)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    train_loader, val_loader = build_seq_dataloaders(args.seq_glob, batch_size=args.batch_size, num_workers=args.num_workers)
    dyn = build_dynamics(
        backend=args.dyn_backend,
        latent_dim=vae_ckpt["latent_dim"],
        obj_state_dim=7,
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        max_len=args.max_len,
        diffusion_steps=args.diffusion_steps,
    ).to(device)
    optim = torch.optim.AdamW(dyn.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    hist = {"train_total": [], "train_lat": [], "train_obj": [], "val_total": []}
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        dyn.train()
        ttot = tlat = tobj = 0.0
        n = 0
        for batch in tqdm(train_loader, desc=f"Dyn train {epoch}", leave=False):
            mano = batch["mano_pose"].to(device)       # (B,T,Dm)
            pc = batch["pointcloud"].to(device)
            obj = batch["object_pose_seq"].to(device)  # (B,T,7)
            B, T, Dm = mano.shape
            with torch.no_grad():
                obj_feat = object_feature_from_pointcloud(pc).unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
                mu, _ = vae.teacher_encoder(mano.reshape(B * T, Dm), obj_feat)
                z = mu.view(B, T, -1)

            z_in, z_gt = z[:, :-1], z[:, 1:]
            o_in, o_gt = obj[:, :-1], obj[:, 1:]
            if dyn.backend == "diffusion":
                losses = dyn.training_loss(z_in, o_in, z_gt, o_gt)
                loss = losses["loss"]
                lz = losses["z_loss"]
                lo = losses["o_loss"]
            else:
                pred = dyn(z_in, o_in)
                lz = F.mse_loss(pred["z_next"], z_gt)
                lo = F.mse_loss(pred["o_next"], o_gt)
                loss = lz + args.obj_weight * lo

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            ttot += float(loss.item())
            tlat += float(lz.item())
            tobj += float(lo.item())
            n += 1

        hist["train_total"].append(ttot / max(1, n))
        hist["train_lat"].append(tlat / max(1, n))
        hist["train_obj"].append(tobj / max(1, n))

        dyn.eval()
        vtot, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                mano = batch["mano_pose"].to(device)
                pc = batch["pointcloud"].to(device)
                obj = batch["object_pose_seq"].to(device)
                B, T, Dm = mano.shape
                obj_feat = object_feature_from_pointcloud(pc).unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
                mu, _ = vae.teacher_encoder(mano.reshape(B * T, Dm), obj_feat)
                z = mu.view(B, T, -1)
                pred = dyn(z[:, :-1], obj[:, :-1])
                loss = F.mse_loss(pred["z_next"], z[:, 1:]) + args.obj_weight * F.mse_loss(pred["o_next"], obj[:, 1:])
                vtot += float(loss.item())
                vn += 1

        v = vtot / max(1, vn)
        hist["val_total"].append(v)
        print(f"Epoch {epoch}: train={hist['train_total'][-1]:.6f}, val={v:.6f}, backend={dyn.backend}")

        if v < best:
            best = v
            os.makedirs(os.path.dirname(args.save_ckpt), exist_ok=True)
            torch.save({
                "model": dyn.state_dict(),
                "latent_dim": vae_ckpt["latent_dim"],
                "backend": dyn.backend,
                "d_model": args.d_model,
                "n_layer": args.n_layer,
                "n_head": args.n_head,
                "max_len": args.max_len,
                "diffusion_steps": args.diffusion_steps,
            }, args.save_ckpt)

        plot_losses(hist, os.path.join(args.log_dir, "dynamics_losses.png"))


if __name__ == "__main__":
    p = argparse.ArgumentParser("Train Mamba latent dynamics")
    p.add_argument("--seq-glob", type=str, default="/data1/jiaozx/UniHM/processed_dexycb/*.npz")
    p.add_argument("--vae-ckpt", type=str, default="/data1/jiaozx/UniHM/checkpoints/vae_shared_best.pth")
    p.add_argument("--save-ckpt", type=str, default="/data1/jiaozx/UniHM/checkpoints/mamba_dynamics_best.pth")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--dyn-backend", type=str, default="mamba", choices=["mamba", "transformer", "diffusion"])
    p.add_argument("--diffusion-steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--obj-weight", type=float, default=1.0)
    p.add_argument("--log-dir", type=str, default="/data1/jiaozx/UniHM/logs")
    args = p.parse_args()
    train(args)
