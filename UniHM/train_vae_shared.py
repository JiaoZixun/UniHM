import argparse
import os
from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from UniHM.SFT.utils import build_seq_dataloaders, DECODER_KEY_ALIASES, ROBOT_KEYS_ORDER
from UniHM.vae.multi_vae import MultiDecoderVAE, kl_loss
from UniHM.visualization.training_viz import plot_losses


def object_feature_from_pointcloud(pc: torch.Tensor) -> torch.Tensor:
    # pc: (B,N,3) -> (B,6): mean/std
    mu = pc.mean(dim=1)
    std = pc.std(dim=1)
    return torch.cat([mu, std], dim=-1)


def resolve_targets(targets: Dict[str, torch.Tensor], present_keys: List[str], device: torch.device):
    ys = []
    for k in present_keys:
        aliases = DECODER_KEY_ALIASES.get(k, [k])
        kk = next((a for a in aliases if a in targets), None)
        if kk is None:
            ys.append(None)
        else:
            ys.append(targets[kk].to(device))
    return ys


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_seq_dataloaders(args.seq_glob, batch_size=args.batch_size, num_workers=args.num_workers)

    first = next(iter(train_loader))
    mano_dim = int(first["mano_pose"].shape[-1])
    sample_targets = first["targets"]
    present_keys = [k for k in ROBOT_KEYS_ORDER if any(a in sample_targets for a in DECODER_KEY_ALIASES.get(k, [k]))]
    out_dims = []
    for k in present_keys:
        aliases = DECODER_KEY_ALIASES.get(k, [k])
        kk = next(a for a in aliases if a in sample_targets)
        out_dims.append(int(sample_targets[kk].shape[-1]))

    model = MultiDecoderVAE(
        mano_dim=mano_dim,
        object_dim=6,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        decoder_out_dims=out_dims,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = {"train_total": [], "train_recon": [], "train_kl": [], "val_total": []}
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_total = tr_recon = tr_kl = 0.0
        nb = 0
        for batch in tqdm(train_loader, desc=f"VAE train {epoch}", leave=False):
            mano = batch["mano_pose"].to(device)      # (B,T,D)
            pc = batch["pointcloud"].to(device)
            targets = batch["targets"]

            B, T, Dm = mano.shape
            mano_bt = mano.reshape(B * T, Dm)
            obj_feat = object_feature_from_pointcloud(pc).unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)

            out = model(mano_bt, obj_feat)
            preds = out["preds"]
            ys = resolve_targets(targets, present_keys, device)

            recon = 0.0
            used = 0
            for i, y in enumerate(ys):
                if y is None:
                    continue
                y_bt = y.reshape(B * T, -1)
                recon = recon + F.l1_loss(preds[i], y_bt)
                used += 1
            if used == 0:
                continue

            kll = kl_loss(out["mu"], out["logvar"])
            loss = recon + args.beta_kl * kll
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            tr_total += float(loss.item())
            tr_recon += float(recon.item())
            tr_kl += float(kll.item())
            nb += 1

        if nb == 0:
            raise RuntimeError("No valid training batch with targets.")

        history["train_total"].append(tr_total / nb)
        history["train_recon"].append(tr_recon / nb)
        history["train_kl"].append(tr_kl / nb)

        model.eval()
        va = 0.0
        vb = 0
        with torch.no_grad():
            for batch in val_loader:
                mano = batch["mano_pose"].to(device)
                pc = batch["pointcloud"].to(device)
                targets = batch["targets"]
                B, T, Dm = mano.shape
                mano_bt = mano.reshape(B * T, Dm)
                obj_feat = object_feature_from_pointcloud(pc).unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
                out = model(mano_bt, obj_feat)
                preds = out["preds"]
                ys = resolve_targets(targets, present_keys, device)
                recon = 0.0
                used = 0
                for i, y in enumerate(ys):
                    if y is None:
                        continue
                    y_bt = y.reshape(B * T, -1)
                    recon = recon + F.l1_loss(preds[i], y_bt)
                    used += 1
                if used == 0:
                    continue
                loss = recon + args.beta_kl * kl_loss(out["mu"], out["logvar"])
                va += float(loss.item())
                vb += 1

        val_loss = va / max(1, vb)
        history["val_total"].append(val_loss)
        print(f"Epoch {epoch}: train={history['train_total'][-1]:.6f}, val={val_loss:.6f}")

        if val_loss < best:
            best = val_loss
            os.makedirs(os.path.dirname(args.save_ckpt), exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "present_keys": present_keys,
                "out_dims": out_dims,
                "mano_dim": mano_dim,
                "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim,
            }, args.save_ckpt)

        plot_losses(history, os.path.join(args.log_dir, "vae_losses.png"))


if __name__ == "__main__":
    p = argparse.ArgumentParser("Train shared-latent VAE on DexYCB sequences")
    p.add_argument("--seq-glob", type=str, default="/public/home/jiaozixun/UniHM/processed_dexycb/*.npz")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--beta-kl", type=float, default=0.01)
    p.add_argument("--save-ckpt", type=str, default="/public/home/jiaozixun/UniHM/checkpoints/vae_shared_best.pth")
    p.add_argument("--log-dir", type=str, default="/public/home/jiaozixun/UniHM/logs")
    args = p.parse_args()
    train(args)
