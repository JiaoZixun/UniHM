import argparse
import os
import numpy as np
import torch

from UniHM.SFT.utils import build_seq_dataloaders, DECODER_KEY_ALIASES
from UniHM.vae.multi_vae import MultiDecoderVAE
from UniHM.dynamics.mamba import MambaDynamics
from UniHM.metrics.common_metrics import mpjpe, fhlt, fhlr, fid, smoothness_l2, rollout_drift
from UniHM.visualization.training_viz import render_hand_object_sequence


def object_feature_from_pointcloud(pc: torch.Tensor) -> torch.Tensor:
    mu = pc.mean(dim=1)
    std = pc.std(dim=1)
    return torch.cat([mu, std], dim=-1)


def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    vae_ckpt = torch.load(args.vae_ckpt, map_location="cpu")
    vae = MultiDecoderVAE(
        mano_dim=vae_ckpt["mano_dim"], object_dim=6, hidden_dim=vae_ckpt["hidden_dim"],
        latent_dim=vae_ckpt["latent_dim"], decoder_out_dims=vae_ckpt["out_dims"]
    ).to(device)
    vae.load_state_dict(vae_ckpt["model"], strict=True)
    vae.eval()

    dyn_ckpt = torch.load(args.dyn_ckpt, map_location="cpu")
    dyn = MambaDynamics(latent_dim=dyn_ckpt["latent_dim"], obj_state_dim=7, d_model=args.d_model, n_layer=args.n_layer).to(device)
    dyn.load_state_dict(dyn_ckpt["model"], strict=False)
    dyn.eval()

    present_keys = vae_ckpt["present_keys"]
    _, val_loader = build_seq_dataloaders(args.seq_glob, batch_size=1, num_workers=max(1, args.num_workers // 2))
    os.makedirs(args.render_dir, exist_ok=True)

    metrics = {"mpjpe": [], "fhlt": [], "fhlr": [], "fid": [], "smooth": [], "drift": []}

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            mano = batch["mano_pose"].to(device)
            obj = batch["object_pose_seq"].to(device)
            pc = batch["pointcloud"].to(device)
            targets = batch["targets"]
            B, T, Dm = mano.shape

            obj_feat = object_feature_from_pointcloud(pc).unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
            mu, _ = vae.teacher_encoder(mano.reshape(B * T, Dm), obj_feat)
            z_gt = mu.view(B, T, -1)

            # rollout: warm frames from GT, then self-predict
            warm = min(args.warm_frames, T - 1)
            z_roll = [z_gt[:, t:t+1] for t in range(warm)]
            o_roll = [obj[:, t:t+1] for t in range(warm)]
            for t in range(warm, T):
                z_in = torch.cat(z_roll, dim=1)
                o_in = torch.cat(o_roll, dim=1)
                pred = dyn(z_in, o_in)
                z_roll.append(pred["z_next"][:, -1:])
                o_roll.append(pred["o_next"][:, -1:])
            z_pred = torch.cat(z_roll, dim=1)[:, :T]

            # decode first key for metric parity with UniHM single-hand reporting
            dec_pred = vae.decode_all(z_pred.reshape(B * T, -1))[0].view(B, T, -1)[0].cpu().numpy()
            aliases = DECODER_KEY_ALIASES.get(present_keys[0], [present_keys[0]])
            tgt_key = next((a for a in aliases if a in targets), None)
            if tgt_key is None:
                continue
            gt = targets[tgt_key][0].cpu().numpy()

            metrics["mpjpe"].append(mpjpe(dec_pred, gt))
            metrics["fhlt"].append(fhlt(dec_pred, gt))
            metrics["fhlr"].append(fhlr(dec_pred, gt))
            metrics["fid"].append(fid(dec_pred, gt))
            metrics["smooth"].append(smoothness_l2(dec_pred))
            metrics["drift"].append(rollout_drift(dec_pred, gt))

            if i % args.render_every == 0:
                render_hand_object_sequence(
                    hand_seq_gt=gt,
                    hand_seq_pred=dec_pred,
                    object_pose_seq=obj[0].cpu().numpy(),
                    object_points_local=pc[0].cpu().numpy(),
                    save_path=os.path.join(args.render_dir, f"dyn_seq_{i:05d}.png"),
                    stride=args.render_stride,
                )

    print("===== Dynamics Eval (UniHM-compatible + extra) =====")
    for k, v in metrics.items():
        print(f"{k}: {float(np.mean(v)) if v else float('nan'):.6f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser("Evaluate mamba dynamics with closed-loop rollout")
    p.add_argument("--seq-glob", type=str, default="/data1/jiaozx/UniHM/processed_dexycb/*.npz")
    p.add_argument("--vae-ckpt", type=str, default="/data1/jiaozx/UniHM/checkpoints/vae_shared_best.pth")
    p.add_argument("--dyn-ckpt", type=str, default="/data1/jiaozx/UniHM/checkpoints/mamba_dynamics_best.pth")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--warm-frames", type=int, default=5)
    p.add_argument("--render-dir", type=str, default="/data1/jiaozx/UniHM/renders/dynamics")
    p.add_argument("--render-every", type=int, default=100)
    p.add_argument("--render-stride", type=int, default=5)
    args = p.parse_args()
    evaluate(args)
