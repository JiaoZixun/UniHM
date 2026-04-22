import argparse
import os
from typing import Dict
import numpy as np
import torch

from UniHM.SFT.utils import build_seq_dataloaders, DECODER_KEY_ALIASES, ROBOT_KEYS_ORDER
from UniHM.vae.multi_vae import MultiDecoderVAE
from UniHM.metrics.common_metrics import mpjpe, fhlt, fhlr, fid, smoothness_l2, rollout_drift
from UniHM.visualization.training_viz import render_hand_object_sequence, render_fullbody_gt_pred_video
from UniHM.utils.mano_layer import MANOLayer
from dex_retargeting.constants import HandType, RobotName
from UniHM.utils.retargeting_processor import RetargetingProcessor


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


_ROBOT_KEY_TO_NAME = {
    "allegro_hand_qpos": RobotName.allegro,
    "shadow_hand_qpos": RobotName.shadow,
    "svh_hand_qpos": RobotName.svh,
    "schunk_svh_hand_qpos": RobotName.svh,
    "leap_hand_qpos": RobotName.leap,
    "ability_hand_qpos": RobotName.ability,
    "panda_hand_qpos": RobotName.panda,
    "panda_gripper_qpos": RobotName.panda,
    "inspire_hand_qpos": RobotName.inspire,
}

def decode_mano_joints_world(
    mano_seq: np.ndarray,
    hand_shape: np.ndarray,
    extrinsics: np.ndarray,
    mano_model_dir: str | None = None,
) -> np.ndarray:
    """Decode MANO pose sequence to 21 joints in world frame."""
    if mano_seq.ndim != 2 or mano_seq.shape[1] < 51:
        raise ValueError(f"Unexpected mano_seq shape: {mano_seq.shape}")
    layer = MANOLayer("right", hand_shape.astype(np.float32), mano_root=mano_model_dir)
    p = torch.from_numpy(mano_seq[:, :48].astype(np.float32))
    t = torch.from_numpy(mano_seq[:, 48:51].astype(np.float32))
    _, joint = layer(p, t)
    joint = joint.detach().cpu().numpy()
    camera_pose = np.linalg.inv(extrinsics)
    joint = joint @ camera_pose[:3, :3].T + camera_pose[:3, 3]
    return np.ascontiguousarray(joint, dtype=np.float32)


def _decode_ee_from_qpos(
    processor: RetargetingProcessor,
    ridx: int,
    qpos_seq: np.ndarray,
) -> np.ndarray:
    """Decode EE keypoints from qpos using loaded robot kinematics."""
    robot = processor.robots[ridx]
    retargeting = processor.retargetings[ridx]
    links = {l.name: l for l in robot.get_links()}
    target_names = list(getattr(retargeting.optimizer, "target_link_names", []) or [])
    if len(target_names) == 0:
        target_names = [n for n in links.keys() if "tip" in n.lower()]
    if len(target_names) == 0:
        raise RuntimeError("Cannot resolve target EE links from retargeting optimizer.")

    out = []
    for t in range(qpos_seq.shape[0]):
        q = qpos_seq[t]
        robot.set_qpos(q.astype(np.float32))
        if processor.scene is not None:
            processor.scene.step()
        pts = []
        for n in target_names:
            if n in links:
                pts.append(links[n].get_pose().p)
        if len(pts) == 0:
            raise RuntimeError(f"No valid EE links found for names: {target_names}")
        out.append(np.stack(pts, axis=0))
    return np.asarray(out, dtype=np.float32)


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

    metric_names = ["mpjpe_legacy", "qpos_mae", "trans_mae", "rot_mae", "fid", "smooth", "drift"]
    per_robot = {k: {m: [] for m in metric_names} for k in present_keys}
    os.makedirs(args.render_dir, exist_ok=True)
    rendered_videos = 0
    need_fk_keys = [k for k in present_keys if k in _ROBOT_KEY_TO_NAME]
    uniq_robot_names = []
    for k in need_fk_keys:
        rn = _ROBOT_KEY_TO_NAME[k]
        if rn not in uniq_robot_names:
            uniq_robot_names.append(rn)
    fk_processor = None
    robot_name_to_index: Dict[RobotName, int] = {}
    if len(uniq_robot_names) > 0:
        fk_processor = RetargetingProcessor(
            robot_names=uniq_robot_names,
            hand_type=HandType.right,
            urdf_dir=args.urdf_dir if args.urdf_dir else None,
            mano_model_dir=args.mano_model_dir if args.mano_model_dir else None,
        )
        robot_name_to_index = {rn: i for i, rn in enumerate(uniq_robot_names)}

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

            # Evaluate every available robot decoder separately.
            first_pred_seq = first_gt_seq = None
            gt_by_robot = {}
            pred_by_robot = {}
            for j, y in enumerate(ys):
                if y is None:
                    continue
                pred_seq = preds[j].view(B, T, -1)[0].cpu().numpy()
                gt_seq = y[0].cpu().numpy()
                rkey = present_keys[j]
                gt_by_robot[rkey] = gt_seq
                pred_by_robot[rkey] = pred_seq
                per_robot[rkey]["mpjpe_legacy"].append(mpjpe(pred_seq, gt_seq))
                per_robot[rkey]["qpos_mae"].append(float(np.abs(pred_seq[:, 6:] - gt_seq[:, 6:]).mean()))
                per_robot[rkey]["trans_mae"].append(fhlt(pred_seq, gt_seq))
                per_robot[rkey]["rot_mae"].append(fhlr(pred_seq, gt_seq))
                per_robot[rkey]["fid"].append(fid(pred_seq, gt_seq))
                per_robot[rkey]["smooth"].append(smoothness_l2(pred_seq))
                per_robot[rkey]["drift"].append(rollout_drift(pred_seq, gt_seq))
                if first_pred_seq is None:
                    first_pred_seq, first_gt_seq = pred_seq, gt_seq

            if i % args.render_every == 0:
                obj_local = pc[0].cpu().numpy()
                render_hand_object_sequence(
                    hand_seq_gt=first_gt_seq if first_gt_seq is not None else np.zeros((1, 7), dtype=np.float32),
                    hand_seq_pred=first_pred_seq if first_pred_seq is not None else np.zeros((1, 7), dtype=np.float32),
                    object_pose_seq=obj_pose,
                    object_points_local=obj_local,
                    save_path=os.path.join(args.render_dir, f"vae_seq_{i:05d}.png"),
                    stride=args.render_stride,
                )
                if rendered_videos < args.render_max_videos and len(gt_by_robot) > 0 and fk_processor is not None:
                    gt_ee_by_robot = {}
                    pred_ee_by_robot = {}
                    for rkey, gt_seq in gt_by_robot.items():
                        if rkey not in _ROBOT_KEY_TO_NAME:
                            continue
                        ridx = robot_name_to_index[_ROBOT_KEY_TO_NAME[rkey]]
                        gt_ee_by_robot[rkey] = _decode_ee_from_qpos(fk_processor, ridx, gt_seq)
                        pred_ee_by_robot[rkey] = _decode_ee_from_qpos(fk_processor, ridx, pred_by_robot[rkey])

                    if "mano_joint_3d_world" in batch:
                        mano_joints_world = batch["mano_joint_3d_world"][0].cpu().numpy()
                    else:
                        mano_joints_world = decode_mano_joints_world(
                            mano_seq=mano[0].cpu().numpy(),
                            hand_shape=batch["hand_shape"][0].cpu().numpy(),
                            extrinsics=batch["extrinsics"][0].cpu().numpy(),
                            mano_model_dir=args.mano_model_dir if args.mano_model_dir else None,
                        )

                    if len(gt_ee_by_robot) > 0:
                        render_fullbody_gt_pred_video(
                            save_path=os.path.join(args.render_dir, f"vae_compare_{i:05d}.mp4"),
                            mano_joints_world=mano_joints_world,
                            gt_ee_by_robot=gt_ee_by_robot,
                            pred_ee_by_robot=pred_ee_by_robot,
                            object_pose_seq=obj_pose,
                            object_points_local=obj_local,
                            fps=args.render_fps,
                            stride=args.render_stride,
                            max_frames=args.render_max_frames,
                        )
                        rendered_videos += 1

    print("===== VAE Eval by Robot (UniHM-compatible + extra) =====")
    print("[note] qpos_mae is in native robot joint units (typically radians).")
    print("[note] trans_mae is typically meters; rot_mae is typically radians.")
    print("[note] mpjpe_legacy is not true 3D-joint MPJPE (kept only for backward comparison).")
    macro = {m: [] for m in metric_names}
    for rkey in present_keys:
        vals = per_robot[rkey]
        print(f"\n[{rkey}]")
        for m in metric_names:
            mv = float(np.mean(vals[m])) if len(vals[m]) > 0 else float("nan")
            print(f"{m}: {mv:.6f}")
            if np.isfinite(mv):
                macro[m].append(mv)

    print("\n[macro_avg]")
    for m in metric_names:
        print(f"{m}: {float(np.mean(macro[m])) if len(macro[m]) > 0 else float('nan'):.6f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser("Evaluate shared VAE")
    p.add_argument("--seq-glob", type=str, default="/public/home/jiaozixun/UniHM/processed_dexycb/*.npz")
    p.add_argument("--ckpt", type=str, default="/public/home/jiaozixun/UniHM/checkpoints/vae_shared_best.pth")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--render-dir", type=str, default="/public/home/jiaozixun/UniHM/renders/vae")
    p.add_argument("--render-every", type=int, default=100)
    p.add_argument("--render-stride", type=int, default=5)
    p.add_argument("--render-max-videos", type=int, default=5, help="Maximum number of comparison videos to render.")
    p.add_argument("--render-fps", type=int, default=20, help="FPS for rendered comparison videos.")
    p.add_argument("--render-max-frames", type=int, default=0, help="Maximum frames per video; 0 means all.")
    p.add_argument("--mano-model-dir", type=str, default="", help="Optional MANO model dir for decoding 21 joints.")
    p.add_argument("--urdf-dir", type=str, default="", help="Optional URDF root for robot FK decoding.")
    p.add_argument("--eval-stage", type=str, default="auto", choices=["auto", "ref", "robot"],
                   help="auto: infer from ckpt training_stage; ref: eval mano/contact; robot: eval robot decoder outputs.")
    args = p.parse_args()
    evaluate(args)
