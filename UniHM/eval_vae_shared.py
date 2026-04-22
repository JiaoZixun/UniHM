import argparse
import os
import numpy as np
import torch

from UniHM.SFT.utils import build_seq_dataloaders, DECODER_KEY_ALIASES, ROBOT_KEYS_ORDER
from UniHM.vae.multi_vae import MultiDecoderVAE
from UniHM.metrics.common_metrics import mpjpe, fhlt, fhlr, fid, smoothness_l2, rollout_drift
from UniHM.visualization.training_viz import render_hand_object_sequence


def object_feature_from_pointcloud(pc: torch.Tensor) -> torch.Tensor:
    mu = pc.mean(dim=1)
    std = pc.std(dim=1)
    return torch.cat([mu, std], dim=-1)


def resolve_targets(targets, present_keys, device):
    out = []
    for k in present_keys:
        aliases = DECODER_KEY_ALIASES.get(k, [k])
        kk = next((a for a in aliases if a in targets), None)
        out.append(targets[kk].to(device) if kk else None)
    return out


def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = MultiDecoderVAE(
        mano_dim=ckpt["mano_dim"],
        object_dim=6,
        hidden_dim=ckpt["hidden_dim"],
        latent_dim=ckpt["latent_dim"],
        decoder_out_dims=ckpt["out_dims"],
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    present_keys = ckpt["present_keys"]

    _, val_loader = build_seq_dataloaders(args.seq_glob, batch_size=1, num_workers=max(1, args.num_workers // 2))

    metrics = {"mpjpe": [], "fhlt": [], "fhlr": [], "fid": [], "smooth": [], "drift": []}
    os.makedirs(args.render_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            mano = batch["mano_pose"].to(device)
            pc = batch["pointcloud"].to(device)
            obj_pose = batch["object_pose_seq"].cpu().numpy()[0]
            targets = batch["targets"]
            B, T, Dm = mano.shape

            mano_bt = mano.reshape(B * T, Dm)
            obj_feat = object_feature_from_pointcloud(pc).unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
            out = model(mano_bt, obj_feat)
            preds = out["preds"]
            ys = resolve_targets(targets, present_keys, device)

            # use first available hand to align with UniHM-style core metrics
            pred_seq = gt_seq = None
            for j, y in enumerate(ys):
                if y is None:
                    continue
                pred_seq = preds[j].view(B, T, -1)[0].cpu().numpy()
                gt_seq = y[0].cpu().numpy()
                break
            if pred_seq is None:
                continue

            metrics["mpjpe"].append(mpjpe(pred_seq, gt_seq))
            metrics["fhlt"].append(fhlt(pred_seq, gt_seq))
            metrics["fhlr"].append(fhlr(pred_seq, gt_seq))
            metrics["fid"].append(fid(pred_seq, gt_seq))
            metrics["smooth"].append(smoothness_l2(pred_seq))
            metrics["drift"].append(rollout_drift(pred_seq, gt_seq))

            if i % args.render_every == 0:
                obj_local = pc[0].cpu().numpy()
                render_hand_object_sequence(
                    hand_seq_gt=gt_seq,
                    hand_seq_pred=pred_seq,
                    object_pose_seq=obj_pose,
                    object_points_local=obj_local,
                    save_path=os.path.join(args.render_dir, f"vae_seq_{i:05d}.png"),
                    stride=args.render_stride,
                )

    print("===== VAE Eval (UniHM-compatible + extra) =====")
    for k, v in metrics.items():
        print(f"{k}: {float(np.mean(v)) if v else float('nan'):.6f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser("Evaluate shared VAE")
    p.add_argument("--seq-glob", type=str, default="/public/home/jiaozixun/UniHM/processed_dexycb/*.npz")
    p.add_argument("--ckpt", type=str, default="/public/home/jiaozixun/UniHM/checkpoints/vae_shared_best.pth")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--render-dir", type=str, default="/public/home/jiaozixun/UniHM/renders/vae")
    p.add_argument("--render-every", type=int, default=100)
    p.add_argument("--render-stride", type=int, default=5)
    args = p.parse_args()
    evaluate(args)
