import argparse
import os
from glob import glob

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


ROBOT_FIELDS = [
    ("allegro_hand", "allegro"),
    ("shadow_hand", "shadow"),
    ("schunk_svh_hand", "svh"),
    ("leap_hand", "leap"),
    ("ability_hand", "ability"),
    ("panda_gripper", "panda"),
    ("inspire_hand", "inspire"),
]


def _safe_robot_qpos(data, key):
    if key not in data:
        return None
    try:
        return dict(data[key].tolist())["robot_qpos"]
    except Exception:
        return None


def _pick_object_traj(data):
    object_pose = data["object_pose"]
    if object_pose.ndim == 2:
        return object_pose
    if object_pose.ndim == 3:
        ycb_ids = data.get("ycb_ids", None)
        grasped_ycb_id = data.get("grasped_ycb_id", None)
        if ycb_ids is not None and grasped_ycb_id is not None:
            try:
                ids = ycb_ids.tolist()
                g = int(grasped_ycb_id)
                idx = ids.index(g)
                return object_pose[:, idx, :]
            except Exception:
                pass
        return object_pose[:, 0, :]
    raise ValueError(f"Unexpected object_pose shape: {object_pose.shape}")


def visualize_one(npz_path: str, out_dir: str):
    data = np.load(npz_path, allow_pickle=True)
    hand_pose = data["hand_pose"]  # (T, 51)
    obj_pose = _pick_object_traj(data)  # (T, 7) assumed pos+quat
    wrist_xyz = hand_pose[:, 48:51] if hand_pose.shape[-1] >= 51 else hand_pose[:, :3]
    obj_xyz = obj_pose[:, 4:7] if obj_pose.shape[-1] >= 7 else obj_pose[:, :3]
    t = np.arange(min(len(wrist_xyz), len(obj_xyz)))

    stem = os.path.splitext(os.path.basename(npz_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    # Figure 1: 3D trajectory (hand wrist + object)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot(wrist_xyz[t, 0], wrist_xyz[t, 1], wrist_xyz[t, 2], label="hand_wrist_xyz", linewidth=1.5)
    ax.plot(obj_xyz[t, 0], obj_xyz[t, 1], obj_xyz[t, 2], label="object_xyz", linewidth=1.5)
    ax.scatter(wrist_xyz[0, 0], wrist_xyz[0, 1], wrist_xyz[0, 2], s=18, label="hand_start")
    ax.scatter(obj_xyz[0, 0], obj_xyz[0, 1], obj_xyz[0, 2], s=18, label="obj_start")
    ax.set_title(f"{stem} - GT 3D trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{stem}_gt_hand_object_3d.png"), dpi=160)
    plt.close(fig)

    # Figure 2: time-series for xyz
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    labels = ["x", "y", "z"]
    for i, name in enumerate(labels):
        axes[0, i].plot(t, wrist_xyz[t, i], linewidth=1.2)
        axes[0, i].set_title(f"hand {name}")
        axes[1, i].plot(t, obj_xyz[t, i], linewidth=1.2)
        axes[1, i].set_title(f"object {name}")
    axes[0, 0].set_ylabel("hand")
    axes[1, 0].set_ylabel("object")
    for i in range(3):
        axes[1, i].set_xlabel("frame")
    fig.suptitle(f"{stem} - GT xyz over time")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{stem}_gt_xyz_timeseries.png"), dpi=160)
    plt.close(fig)

    # Figure 3: multi-robot qpos norms
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    has_robot = False
    for field, alias in ROBOT_FIELDS:
        qpos = _safe_robot_qpos(data, field)
        if qpos is None or qpos.ndim != 2 or qpos.shape[0] < 1:
            continue
        qnorm = np.linalg.norm(qpos, axis=1)
        ax.plot(np.arange(len(qnorm)), qnorm, linewidth=1.1, label=f"{alias}_||q||")
        has_robot = True
    if has_robot:
        ax.set_title(f"{stem} - robot qpos norm over time (GT)")
        ax.set_xlabel("frame")
        ax.set_ylabel("L2 norm")
        ax.legend(loc="best", ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{stem}_gt_robot_qpos_norm.png"), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser("Visualize GT hand/robot/object trajectories from processed npz")
    parser.add_argument("--npz", type=str, default="", help="Single npz file path")
    parser.add_argument("--glob", type=str, default="", help="Glob for npz files, e.g. /path/*.npz")
    parser.add_argument("--out-dir", type=str, default="./gt_viz", help="Directory for rendered PNG files")
    parser.add_argument("--max-files", type=int, default=10, help="Max files to visualize when --glob is set")
    args = parser.parse_args()

    files = []
    if args.npz:
        files = [args.npz]
    elif args.glob:
        files = sorted(glob(args.glob))[: max(1, args.max_files)]
    else:
        raise ValueError("Please provide either --npz or --glob")

    os.makedirs(args.out_dir, exist_ok=True)
    for fp in files:
        try:
            visualize_one(fp, args.out_dir)
            print(f"[OK] {fp}")
        except Exception as err:
            print(f"[FAIL] {fp}: {err}")

    print(f"Saved visualizations to: {args.out_dir}")


if __name__ == "__main__":
    main()
