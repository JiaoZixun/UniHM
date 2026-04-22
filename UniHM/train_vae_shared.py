import argparse
import json
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


def balanced_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, max_pos_weight: float = 50.0) -> torch.Tensor:
    """Class-balanced BCE to handle long non-contact segments.

    contact-positive is usually sparse; this raises positive weight so the
    network does not collapse to always predicting non-contact.
    """
    target = target.to(dtype=logits.dtype)
    pos = target.sum()
    total = torch.tensor(target.numel(), dtype=logits.dtype, device=logits.device)
    neg = total - pos
    pos_weight = (neg / (pos + 1e-6)).clamp(min=1.0, max=max_pos_weight)
    return F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight
    )


def freeze_ref_and_encoder(model: MultiDecoderVAE):
    for p in model.teacher_encoder.parameters():
        p.requires_grad = False
    for p in model.ref_mano_decoder.parameters():
        p.requires_grad = False
    if model.ref_contact_obj_decoder is not None:
        for p in model.ref_contact_obj_decoder.parameters():
            p.requires_grad = False
    if model.ref_contact_hand_decoder is not None:
        for p in model.ref_contact_hand_decoder.parameters():
            p.requires_grad = False
    for dec in model.decoders:
        for p in dec.parameters():
            p.requires_grad = True


def run_ref_epoch(
    model, loader, device, beta_kl, w_mano, w_obj_contact, w_hand_contact,
    max_contact_pos_weight, enable_obj_contact, enable_hand_contact, train=True, optim=None
):
    if train:
        model.train()
    else:
        model.eval()
    total = 0.0
    nb = 0
    for batch in loader:
        mano = batch["mano_pose"].to(device)
        pc = batch["pointcloud"].to(device)
        contact_obj = batch.get("contact_obj_map", None)
        contact_hand = batch.get("contact_hand_map", None)
        if contact_obj is None:
            continue
        if contact_hand is None:
            contact_hand = torch.zeros((mano.size(0), mano.size(1), 21), dtype=torch.float32)
        contact_obj = contact_obj.to(device)
        contact_hand = contact_hand.to(device)

        B, T, Dm = mano.shape
        mano_bt = mano.reshape(B * T, Dm)
        contact_obj_bt = contact_obj.reshape(B * T, -1)
        contact_hand_bt = contact_hand.reshape(B * T, -1)
        obj_feat = object_feature_from_pointcloud(pc).unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)

        with torch.set_grad_enabled(train):
            out = model.forward_ref(mano_bt, obj_feat)
            loss_mano = F.l1_loss(out["mano_rec"], mano_bt)
            loss_obj = (
                balanced_bce_with_logits(out["contact_obj_logits"], contact_obj_bt, max_pos_weight=max_contact_pos_weight)
                if (enable_obj_contact and out["contact_obj_logits"] is not None)
                else torch.tensor(0.0, device=device)
            )
            loss_hand = (
                balanced_bce_with_logits(out["contact_hand_logits"], contact_hand_bt, max_pos_weight=max_contact_pos_weight)
                if (enable_hand_contact and out["contact_hand_logits"] is not None)
                else torch.tensor(0.0, device=device)
            )
            kll = kl_loss(out["mu"], out["logvar"])
            loss = w_mano * loss_mano + w_obj_contact * loss_obj + w_hand_contact * loss_hand + beta_kl * kll

        if train and optim is not None:
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
        total += float(loss.item())
        nb += 1
    return total / max(1, nb)


def run_robot_epoch(model, loader, device, present_keys, train=True, optim=None):
    if train:
        model.train()
    else:
        model.eval()
    total = 0.0
    nb = 0
    for batch in loader:
        mano = batch["mano_pose"].to(device)
        pc = batch["pointcloud"].to(device)
        targets = batch["targets"]
        B, T, Dm = mano.shape
        mano_bt = mano.reshape(B * T, Dm)
        obj_feat = object_feature_from_pointcloud(pc).unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
        ys = resolve_targets(targets, present_keys, device)

        with torch.set_grad_enabled(train):
            out = model(mano_bt, obj_feat)
            preds = out["preds"]
            recon = 0.0
            used = 0
            for i, y in enumerate(ys):
                if y is None:
                    continue
                recon = recon + F.l1_loss(preds[i], y.reshape(B * T, -1))
                used += 1
            if used == 0:
                continue

        if train and optim is not None:
            optim.zero_grad(set_to_none=True)
            recon.backward()
            optim.step()
        total += float(recon.item())
        nb += 1
    return total / max(1, nb)


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_seq_dataloaders(args.seq_glob, batch_size=args.batch_size, num_workers=args.num_workers)
    os.makedirs(args.log_dir, exist_ok=True)

    first = next(iter(train_loader))
    mano_dim = int(first["mano_pose"].shape[-1])
    sample_targets = first["targets"]
    present_keys = [k for k in ROBOT_KEYS_ORDER if any(a in sample_targets for a in DECODER_KEY_ALIASES.get(k, [k]))]
    out_dims = []
    for k in present_keys:
        aliases = DECODER_KEY_ALIASES.get(k, [k])
        kk = next(a for a in aliases if a in sample_targets)
        out_dims.append(int(sample_targets[kk].shape[-1]))

    use_obj_contact = args.contact_target in ["object", "both"]
    use_hand_contact = args.contact_target in ["hand", "both"]

    model = MultiDecoderVAE(
        mano_dim=mano_dim,
        object_dim=6,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        decoder_out_dims=out_dims,
        contact_obj_dim=args.contact_num_points if use_obj_contact else 0,
        contact_hand_dim=21 if use_hand_contact else 0,
    ).to(device)

    # Stage 1: train ref (mano + contact) to inject contact-aware latent.
    ref_optim = torch.optim.AdamW(model.parameters(), lr=args.lr_ref, weight_decay=args.weight_decay)
    ref_hist = {"train_total": [], "val_total": []}
    best_ref = float("inf")
    save_ref_ckpt = args.save_ref_ckpt
    if save_ref_ckpt == "":
        root, ext = os.path.splitext(args.save_ckpt)
        save_ref_ckpt = f"{root}_ref{ext or '.pth'}"
    for epoch in range(1, args.ref_epochs + 1):
        tr = run_ref_epoch(
            model, tqdm(train_loader, desc=f"Ref train {epoch}", leave=False), device, args.beta_kl,
            args.w_mano, args.w_contact_obj,
            args.w_contact_hand if use_hand_contact else 0.0,
            args.max_contact_pos_weight,
            use_obj_contact,
            use_hand_contact,
            train=True, optim=ref_optim,
        )
        val_loss = run_ref_epoch(
            model, val_loader, device, args.beta_kl,
            args.w_mano, args.w_contact_obj,
            args.w_contact_hand if use_hand_contact else 0.0,
            args.max_contact_pos_weight,
            use_obj_contact,
            use_hand_contact,
            train=False,
        )
        ref_hist["train_total"].append(tr)
        ref_hist["val_total"].append(val_loss)
        print(f"[Ref] Epoch {epoch}: train={tr:.6f}, val={val_loss:.6f}")
        if val_loss < best_ref:
            best_ref = val_loss
            os.makedirs(os.path.dirname(save_ref_ckpt), exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "present_keys": present_keys,
                "out_dims": out_dims,
                "mano_dim": mano_dim,
                "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim,
                "contact_obj_dim": args.contact_num_points if use_obj_contact else 0,
                "contact_hand_dim": 21 if use_hand_contact else 0,
                "training_stage": "ref",
                "best_ref_val": best_ref,
            }, save_ref_ckpt)

    # Stage 2: freeze ref and train robot decoders only.
    freeze_ref_and_encoder(model)
    robot_optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr_robot, weight_decay=args.weight_decay
    )
    robot_hist = {"train_total": [], "val_total": []}
    best_robot = float("inf")
    for epoch in range(1, args.robot_epochs + 1):
        tr = run_robot_epoch(
            model, tqdm(train_loader, desc=f"Robot train {epoch}", leave=False), device, present_keys,
            train=True, optim=robot_optim
        )
        val_loss = run_robot_epoch(model, val_loader, device, present_keys, train=False)
        robot_hist["train_total"].append(tr)
        robot_hist["val_total"].append(val_loss)
        print(f"[Robot] Epoch {epoch}: train={tr:.6f}, val={val_loss:.6f}")
        if val_loss < best_robot:
            best_robot = val_loss
            os.makedirs(os.path.dirname(args.save_ckpt), exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "present_keys": present_keys,
                "out_dims": out_dims,
                "mano_dim": mano_dim,
                "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim,
                "contact_obj_dim": args.contact_num_points if use_obj_contact else 0,
                "contact_hand_dim": 21 if use_hand_contact else 0,
                "training_stage": "robot_decoder",
            }, args.save_ckpt)
    with open(os.path.join(args.log_dir, "vae_ref_history.json"), "w", encoding="utf-8") as f:
        json.dump(ref_hist, f, indent=2)
    with open(os.path.join(args.log_dir, "vae_robot_history.json"), "w", encoding="utf-8") as f:
        json.dump(robot_hist, f, indent=2)
    plot_losses({"train_total": ref_hist["train_total"], "val_total": ref_hist["val_total"]},
                os.path.join(args.log_dir, "vae_ref_losses.png"))
    plot_losses({"train_total": robot_hist["train_total"], "val_total": robot_hist["val_total"]},
                os.path.join(args.log_dir, "vae_robot_losses.png"))


if __name__ == "__main__":
    p = argparse.ArgumentParser("Train shared-latent VAE on DexYCB sequences")
    p.add_argument("--seq-glob", type=str, default="/public/home/jiaozixun/UniHM/processed_dexycb/*.npz")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ref-epochs", type=int, default=80)
    p.add_argument("--robot-epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--lr-ref", type=float, default=1e-4)
    p.add_argument("--lr-robot", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--beta-kl", type=float, default=0.01)
    p.add_argument("--contact-num-points", type=int, default=1024)
    p.add_argument("--contact-target", type=str, default="both", choices=["object", "hand", "both"])
    p.add_argument("--w-mano", type=float, default=1.0)
    p.add_argument("--w-contact-obj", type=float, default=1.0)
    p.add_argument("--w-contact-hand", type=float, default=0.5)
    p.add_argument("--max-contact-pos-weight", type=float, default=50.0)
    p.add_argument("--save-ref-ckpt", type=str, default="", help="Optional path to save best ref-stage checkpoint.")
    p.add_argument("--save-ckpt", type=str, default="/public/home/jiaozixun/UniHM/checkpoints/vae_shared_best.pth")
    p.add_argument("--log-dir", type=str, default="/public/home/jiaozixun/UniHM/logs")
    args = p.parse_args()
    train(args)
