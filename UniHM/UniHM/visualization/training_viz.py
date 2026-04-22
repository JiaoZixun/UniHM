import os
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from UniHM.optimizer.utils import posquat_to_T, transform_points


def plot_losses(history: Dict[str, List[float]], save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    for k, v in history.items():
        if len(v) > 0:
            plt.plot(v, label=k)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def render_hand_object_sequence(
    hand_seq_gt: np.ndarray,
    hand_seq_pred: np.ndarray,
    object_pose_seq: np.ndarray,
    object_points_local: np.ndarray,
    save_path: str,
    stride: int = 5,
):
    """Light-weight 3D rendering with matplotlib for GT vs Pred sanity-check.

    hand_seq_* are rendered using first 3 dims as proxy wrist translation.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    t = min(hand_seq_gt.shape[0], hand_seq_pred.shape[0], object_pose_seq.shape[0])
    frames = list(range(0, t, max(1, stride)))

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax1.set_title("GT hand-object")
    ax2.set_title("Pred hand-object")

    for fi in frames:
        T = posquat_to_T(object_pose_seq[fi])
        obj_w = transform_points(T, object_points_local)
        gt_wrist = hand_seq_gt[fi, :3]
        pr_wrist = hand_seq_pred[fi, :3]
        ax1.scatter(obj_w[:, 0], obj_w[:, 1], obj_w[:, 2], s=1, alpha=0.05, c="gray")
        ax2.scatter(obj_w[:, 0], obj_w[:, 1], obj_w[:, 2], s=1, alpha=0.05, c="gray")
        ax1.scatter(gt_wrist[0], gt_wrist[1], gt_wrist[2], c="green", s=6)
        ax2.scatter(pr_wrist[0], pr_wrist[1], pr_wrist[2], c="red", s=6)

    for ax in (ax1, ax2):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def render_multi_robot_comparison_video(
    gt_by_robot: Dict[str, np.ndarray],
    pred_by_robot: Dict[str, np.ndarray],
    object_pose_seq: np.ndarray,
    object_points_local: np.ndarray,
    save_path: str,
    fps: int = 20,
    stride: int = 2,
    max_frames: int = 0,
):
    """Render side-by-side GT vs Pred video for multiple robot morphologies.

    Each robot sequence is visualized with first 3 dims as wrist/EE proxy.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    common_keys = [k for k in gt_by_robot.keys() if k in pred_by_robot]
    if len(common_keys) == 0:
        raise ValueError("No common robot keys between gt_by_robot and pred_by_robot.")

    t = object_pose_seq.shape[0]
    for k in common_keys:
        t = min(t, gt_by_robot[k].shape[0], pred_by_robot[k].shape[0])
    if t <= 0:
        raise ValueError("No valid frames to render.")

    frame_ids = np.arange(0, t, max(1, stride))
    if max_frames > 0:
        frame_ids = frame_ids[:max_frames]
    if frame_ids.size == 0:
        raise ValueError("No frames after stride/max_frames filtering.")

    colors = [
        "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
        "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
    ]
    color_map = {k: colors[i % len(colors)] for i, k in enumerate(common_keys)}

    # World bounds from object and all robot trajectories (GT + Pred)
    min_xyz = np.min(object_pose_seq[:t, 4:7], axis=0)
    max_xyz = np.max(object_pose_seq[:t, 4:7], axis=0)
    for k in common_keys:
        gt_xyz = gt_by_robot[k][:t, :3]
        pr_xyz = pred_by_robot[k][:t, :3]
        min_xyz = np.minimum(min_xyz, np.minimum(gt_xyz.min(axis=0), pr_xyz.min(axis=0)))
        max_xyz = np.maximum(max_xyz, np.maximum(gt_xyz.max(axis=0), pr_xyz.max(axis=0)))
    margin = 0.05
    min_xyz -= margin
    max_xyz += margin
    tail_len = min(20, len(frame_ids))

    fig = plt.figure(figsize=(12, 6))
    ax_gt = fig.add_subplot(1, 2, 1, projection="3d")
    ax_pr = fig.add_subplot(1, 2, 2, projection="3d")

    def _setup_axis(ax, title: str):
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim(min_xyz[0], max_xyz[0])
        ax.set_ylim(min_xyz[1], max_xyz[1])
        ax.set_zlim(min_xyz[2], max_xyz[2])

    def update(frame_idx: int):
        fi = int(frame_ids[frame_idx])
        ax_gt.cla()
        ax_pr.cla()
        _setup_axis(ax_gt, f"GT hand-object | frame={fi}")
        _setup_axis(ax_pr, f"Pred hand-object | frame={fi}")

        t_obj = posquat_to_T(object_pose_seq[fi])
        obj_w = transform_points(t_obj, object_points_local)
        obj_sub = obj_w[:: max(1, len(obj_w) // 2000)]

        ax_gt.scatter(obj_sub[:, 0], obj_sub[:, 1], obj_sub[:, 2], s=1, c="gray", alpha=0.3)
        ax_pr.scatter(obj_sub[:, 0], obj_sub[:, 1], obj_sub[:, 2], s=1, c="gray", alpha=0.3)

        st = max(0, fi - tail_len)
        for key in common_keys:
            c = color_map[key]
            gt_wrist = gt_by_robot[key][fi, :3]
            pr_wrist = pred_by_robot[key][fi, :3]
            gt_traj = gt_by_robot[key][st : fi + 1, :3]
            pr_traj = pred_by_robot[key][st : fi + 1, :3]

            ax_gt.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2], c=c, linewidth=1.0, alpha=0.7)
            ax_pr.plot(pr_traj[:, 0], pr_traj[:, 1], pr_traj[:, 2], c=c, linewidth=1.0, alpha=0.7)
            ax_gt.scatter(gt_wrist[0], gt_wrist[1], gt_wrist[2], c=c, s=18, label=key)
            ax_pr.scatter(pr_wrist[0], pr_wrist[1], pr_wrist[2], c=c, s=18, label=key)

        if frame_idx == 0:
            ax_gt.legend(loc="upper right", fontsize=8)
            ax_pr.legend(loc="upper right", fontsize=8)

        return []

    ani = animation.FuncAnimation(fig, update, frames=len(frame_ids), interval=1000.0 / fps, blit=False)
    writer = animation.FFMpegWriter(fps=fps, codec="libx264")
    ani.save(save_path, writer=writer, dpi=140)
    plt.close(fig)
