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
    training_stage = ckpt.get("training_stage", "robot_decoder")
    model = MultiDecoderVAE(
        mano_dim=ckpt["mano_dim"],
        object_dim=6,
        hidden_dim=ckpt["hidden_dim"],
        latent_dim=ckpt["latent_dim"],
        decoder_out_dims=ckpt["out_dims"],
        contact_obj_dim=int(ckpt.get("contact_obj_dim", 0)),
        contact_hand_dim=int(ckpt.get("contact_hand_dim", 0)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    present_keys = ckpt["present_keys"]

    _, val_loader = build_seq_dataloaders(args.seq_glob, batch_size=1, num_workers=max(1, args.num_workers // 2))

    eval_stage = args.eval_stage
    if eval_stage == "auto":
        eval_stage = "ref" if training_stage == "ref" else "robot"

    if eval_stage == "ref":
        metrics = {"mano_l1": [], "contact_obj_bce": [], "contact_obj_f1": [], "contact_hand_bce": [], "contact_hand_f1": []}
        with torch.no_grad():
            for batch in val_loader:
                mano = batch["mano_pose"].to(device)
                pc = batch["pointcloud"].to(device)
                contact_obj = batch.get("contact_obj_map", None)
                contact_hand = batch.get("contact_hand_map", None)
                B, T, Dm = mano.shape
                mano_bt = mano.reshape(B * T, Dm)
                obj_feat = object_feature_from_pointcloud(pc).unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
                out = model.forward_ref(mano_bt, obj_feat)
                metrics["mano_l1"].append(float(torch.nn.functional.l1_loss(out["mano_rec"], mano_bt).item()))
                if out["contact_obj_logits"] is not None and contact_obj is not None:
                    gt_obj = contact_obj.to(device).reshape(B * T, -1)
                    pred_obj = torch.sigmoid(out["contact_obj_logits"])
                    bce_obj = torch.nn.functional.binary_cross_entropy(pred_obj, gt_obj)
                    pred_bin = (pred_obj > 0.5).float()
                    tp = (pred_bin * gt_obj).sum()
                    fp = (pred_bin * (1 - gt_obj)).sum()
                    fn = ((1 - pred_bin) * gt_obj).sum()
                    f1 = (2 * tp / (2 * tp + fp + fn + 1e-6)).item()
                    metrics["contact_obj_bce"].append(float(bce_obj.item()))
                    metrics["contact_obj_f1"].append(float(f1))
                if out["contact_hand_logits"] is not None and contact_hand is not None:
                    gt_hand = contact_hand.to(device).reshape(B * T, -1)
                    pred_hand = torch.sigmoid(out["contact_hand_logits"])
                    bce_hand = torch.nn.functional.binary_cross_entropy(pred_hand, gt_hand)
                    pred_bin = (pred_hand > 0.5).float()
                    tp = (pred_bin * gt_hand).sum()
                    fp = (pred_bin * (1 - gt_hand)).sum()
                    fn = ((1 - pred_bin) * gt_hand).sum()
                    f1 = (2 * tp / (2 * tp + fp + fn + 1e-6)).item()
                    metrics["contact_hand_bce"].append(float(bce_hand.item()))
                    metrics["contact_hand_f1"].append(float(f1))
        print("===== VAE Ref Eval =====")
        for k, v in metrics.items():
            if len(v) > 0:
                print(f"{k}: {float(np.mean(v)):.6f}")
        return

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
    p.add_argument("--eval-stage", type=str, default="auto", choices=["auto", "ref", "robot"],
                   help="auto: infer from ckpt training_stage; ref: eval mano/contact; robot: eval robot decoder outputs.")
    args = p.parse_args()
    evaluate(args)
