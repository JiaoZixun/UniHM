import argparse
import os
from glob import glob

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from UniHM.dataset import load_dataset_squential


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
    q = obj_pose_7[:4]
    t = obj_pose_7[4:7]
    r = _quat_to_rotmat_xyzw(q)
    return local_points @ r.T + t[None, :]


def _safe_robot_ee(data, field):
    if field not in data:
        return None
    try:
        item = dict(data[field].tolist())
    except Exception:
        return None
    return item.get("ee_target", None)


def render_video(npz_path: str, out_dir: str, fps: int = 20, stride: int = 1, max_frames: int = 0):
    raw = np.load(npz_path, allow_pickle=True)
    ds = load_dataset_squential(npz_path)

    hand_joints = raw["mano_joint_3d"] if "mano_joint_3d" in raw else None
    if hand_joints is None:
        raise ValueError("NPZ missing `mano_joint_3d`. Please re-run preprocess_dexycb.py with updated code.")

    obj_pose_seq = ds["grasped_obj_pose"]
    if hasattr(obj_pose_seq, "cpu"):
        obj_pose_seq = obj_pose_seq.cpu().numpy()
    object_points_local = ds["grasped_obj_point3d"]
    if hasattr(object_points_local, "cpu"):
        object_points_local = object_points_local.cpu().numpy()

    ee_data = {}
    for field, alias in ROBOT_FIELDS:
        ee = _safe_robot_ee(raw, field)
        if ee is not None:
            ee_data[alias] = ee

    t_total = min(hand_joints.shape[0], obj_pose_seq.shape[0])
    frame_ids = np.arange(0, t_total, max(1, stride))
    if max_frames > 0:
        frame_ids = frame_ids[:max_frames]
    if len(frame_ids) == 0:
        raise ValueError("No frames to render after stride/max_frames filtering.")

    stem = os.path.splitext(os.path.basename(npz_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    out_video = os.path.join(out_dir, f"{stem}_gt_alignment.mp4")

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_title("GT alignment: hand(21kps) + robot ee + object pointcloud")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    colors = {
        "allegro": "tab:blue",
        "shadow": "tab:orange",
        "svh": "tab:green",
        "leap": "tab:red",
        "ability": "tab:purple",
        "panda": "tab:brown",
        "inspire": "tab:pink",
    }

    def update(k):
        fi = frame_ids[k]
        ax.cla()
        ax.set_title(f"{stem} | frame={fi}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        joints = hand_joints[fi]
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=12, c="k", label="hand_21kps")
        for a, b in HAND_EDGES:
            ax.plot(
                [joints[a, 0], joints[b, 0]],
                [joints[a, 1], joints[b, 1]],
                [joints[a, 2], joints[b, 2]],
                linewidth=1.0,
                c="k",
                alpha=0.85,
            )

        obj_w = _transform_object_points(object_points_local, obj_pose_seq[fi])
        obj_sub = obj_w[:: max(1, len(obj_w) // 2000)]
        ax.scatter(obj_sub[:, 0], obj_sub[:, 1], obj_sub[:, 2], s=1, c="gray", alpha=0.35, label="object_pc")

        for alias, ee in ee_data.items():
            if fi >= ee.shape[0]:
                continue
            pts = ee[fi]
            ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                s=16, c=colors.get(alias, None), alpha=0.9, label=f"{alias}_ee"
            )

        if k == 0:
            ax.legend(loc="upper right", fontsize=8)
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(frame_ids), interval=1000.0 / fps, blit=False)
    writer = animation.FFMpegWriter(fps=fps, codec="libx264")
    anim.save(out_video, writer=writer, dpi=140)
    plt.close(fig)
    return out_video


def main():
    parser = argparse.ArgumentParser("Render GT alignment videos (hand 21kps + robot ee + object pointcloud).")
    parser.add_argument("--npz", type=str, default="", help="Single NPZ file.")
    parser.add_argument("--glob", type=str, default="", help="Glob pattern for NPZ files.")
    parser.add_argument("--out-dir", type=str, default="./gt_videos", help="Output directory for MP4 videos.")
    parser.add_argument("--max-files", type=int, default=10, help="Maximum files when --glob is used.")
    parser.add_argument("--fps", type=int, default=20, help="Output video FPS.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride for rendering.")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames per video. 0 means all.")
    args = parser.parse_args()

    if not args.npz and not args.glob:
        raise ValueError("Please provide either --npz or --glob.")

    files = [args.npz] if args.npz else sorted(glob(args.glob))[: max(1, args.max_files)]
    os.makedirs(args.out_dir, exist_ok=True)

    for fp in files:
        try:
            video = render_video(fp, args.out_dir, fps=args.fps, stride=args.stride, max_frames=args.max_frames)
            print(f"[OK] {fp} -> {video}")
        except Exception as err:
            print(f"[FAIL] {fp}: {err}")


if __name__ == "__main__":
    main()
