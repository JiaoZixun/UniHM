import os
from typing import Dict, List

import matplotlib.pyplot as plt
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
