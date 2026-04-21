import argparse
import os
from glob import glob

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

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


def _quat_to_rotmat_xyzw(q):
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    x, y, z, w = q
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform_object_points(local_points, obj_pose_7):
    # DexYCB object pose uses [quat(4), trans(3)], quaternion in xyzw convention.
    q = obj_pose_7[:4]
    t = obj_pose_7[4:7]
    r = _quat_to_rotmat_xyzw(q)
    return local_points @ r.T + t[None, :]


def _load_object_pointcloud(data, grasp_idx):
    if "object_mesh_file" not in data:
        return None
    mesh_files = data["object_mesh_file"]
    if hasattr(mesh_files, "tolist"):
        mesh_files = mesh_files.tolist()
    if isinstance(mesh_files, str):
        mesh_files = [mesh_files]
    if not mesh_files or grasp_idx >= len(mesh_files):
        return None
    xyz_path = os.path.join(os.path.dirname(mesh_files[grasp_idx]), "points.xyz")
    if not os.path.exists(xyz_path):
        return None
    try:
        points = np.loadtxt(xyz_path)
        if points.ndim == 2 and points.shape[1] >= 3:
            return points[:, :3]
    except Exception:
        return None
    return None


def visualize_one(npz_path: str, out_dir: str):
    data = np.load(npz_path, allow_pickle=True)
    hand_pose = data["hand_pose"]  # (T, 51)
    hand_joints = data["mano_joint_3d"] if "mano_joint_3d" in data else None
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

    # Figure 1b: sampled 3D frames with 21 hand keypoints + ee targets + object point cloud.
    fig = plt.figure(figsize=(14, 4))
    sample_frames = np.linspace(0, len(t) - 1, num=min(3, len(t)), dtype=int)
    grasp_idx = int(data["grasped_obj_idx"]) if "grasped_obj_idx" in data else 0
    object_points_local = _load_object_pointcloud(data, grasp_idx)
    ee_colors = {
        "allegro_hand": "tab:blue",
        "shadow_hand": "tab:orange",
        "schunk_svh_hand": "tab:green",
        "leap_hand": "tab:red",
        "ability_hand": "tab:purple",
        "panda_gripper": "tab:brown",
        "inspire_hand": "tab:pink",
    }
    for pi, fi in enumerate(sample_frames):
        ax = fig.add_subplot(1, len(sample_frames), pi + 1, projection="3d")
        ax.set_title(f"{stem} frame={fi}")
        if hand_joints is not None and fi < hand_joints.shape[0]:
            joints = hand_joints[fi]
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=12, c="k", label="hand_21kps")
            for a, b in HAND_EDGES:
                ax.plot(
                    [joints[a, 0], joints[b, 0]],
                    [joints[a, 1], joints[b, 1]],
                    [joints[a, 2], joints[b, 2]],
                    linewidth=1.0,
                    c="k",
                    alpha=0.8,
                )
        if object_points_local is not None and fi < obj_pose.shape[0]:
            obj_w = _transform_object_points(object_points_local, obj_pose[fi])
            obj_sub = obj_w[:: max(1, len(obj_w) // 1500)]
            ax.scatter(obj_sub[:, 0], obj_sub[:, 1], obj_sub[:, 2], s=1, c="gray", alpha=0.35, label="object_pc")

        for field, _ in ROBOT_FIELDS:
            q = _safe_robot_qpos(data, field)
            if q is None:
                continue
            robot_item = dict(data[field].tolist())
            if "ee_target" not in robot_item:
                continue
            ee = robot_item["ee_target"]  # (T, K, 3)
            if fi >= ee.shape[0]:
                continue
            pts = ee[fi]
            color = ee_colors.get(field, None)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=15, alpha=0.85, c=color, label=f"{field}_ee")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    handles, labels = fig.axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{stem}_gt_kps_ee_objectpc.png"), dpi=180)
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

    # Figure 4: contact heuristic timeline if available
    if "contact_min_dist" in data:
        d = data["contact_min_dist"]
        c = data["contact_flag"] if "contact_flag" in data else (d < 0.12).astype(np.uint8)
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 3.5))
        ax1.plot(np.arange(len(d)), d, linewidth=1.3, label="min_dist(hand_21kp, object_center)")
        ax1.set_xlabel("frame")
        ax1.set_ylabel("distance (m)")
        ax2 = ax1.twinx()
        ax2.plot(np.arange(len(c)), c, linewidth=1.1, c="tab:red", label="contact_flag")
        ax2.set_ylabel("flag")
        ax1.set_title(f"{stem} - heuristic contact")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{stem}_gt_contact_timeline.png"), dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser("Visualize GT hand/robot/object trajectories from processed npz")
    parser.add_argument("--npz", type=str, default="", help="Single npz file path")
    parser.add_argument("--glob", type=str, default="/public/home/jiaozixun/UniHM/processed_dexycb/*.npz", help="Glob for npz files, e.g. /path/*.npz")
    parser.add_argument("--out-dir", type=str, default="/public/home/jiaozixun/UniHM/gt_viz", help="Directory for rendered PNG files")
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
