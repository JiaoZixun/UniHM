from pathlib import Path
from typing import Dict, List, Optional

import sys
import numpy as np
import sapien
import torch
from pytransform3d import rotations

from .mano_layer import MANOLayer

_ROOT = Path(__file__).resolve().parents[1]
_POS_RETARGET_DIR = _ROOT / "dex-retargeting" / "example" / "position_retargeting"
if str(_POS_RETARGET_DIR) not in sys.path:
    sys.path.append(str(_POS_RETARGET_DIR))

from dex_retargeting.constants import (
    HandType,
    RetargetingType,
    RobotName,
    ROBOT_NAME_MAP,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting


class RetargetingProcessor:
    """Pure retargeting processor — no visualization, no rendering."""

    def __init__(
        self,
        robot_names: List[RobotName],
        hand_type: HandType,
        urdf_dir: Optional[str] = None,
        mano_model_dir: Optional[str] = None,
    ):
        sapien.render.set_log_level("error")
        self.scene: Optional[sapien.Scene] = None
        self.headless_mode = False
        try:
            self.scene = sapien.Scene()
        except RuntimeError as err:
            if "rendering device" not in str(err).lower():
                raise
            self.headless_mode = True
            print(
                "[RetargetingProcessor] Rendering device unavailable, "
                "falling back to headless mode."
            )

        if urdf_dir:
            urdf_root = Path(urdf_dir).expanduser()
            if not urdf_root.exists():
                raise FileNotFoundError(f"Retargeting URDF directory does not exist: {urdf_root}")
            RetargetingConfig.set_default_urdf_dir(str(urdf_root))

        self.mano_model_dir = mano_model_dir

        self.robot_names = robot_names
        self.retargetings: List[SeqRetargeting] = []
        self.retarget2sapien: List[np.ndarray] = []
        self.hand_type = hand_type
        self.sapien_joint_names: List[List[str]] = []
        self.robot_file_names: List[str] = []
        self.robots: List[Optional[sapien.Articulation]] = []
        loader = None
        if self.scene is not None:
            loader = self.scene.create_urdf_loader()
            loader.fix_root_link = True
            loader.load_multiple_collisions_from_file = True

        for robot_name in robot_names:
            config_path = get_default_config_path(
                robot_name, RetargetingType.position, hand_type
            )
            override = dict(add_dummy_free_joint=True)
            config = RetargetingConfig.load_from_file(config_path, override=override)
            retargeting = config.build()
            self.robot_file_names.append(Path(config.urdf_path).stem)
            self.retargetings.append(retargeting)

            if loader is None:
                self.robots.append(None)
                sapien_joint_names = list(retargeting.joint_names)
                retarget2sapien = np.arange(len(sapien_joint_names), dtype=int)
            else:
                try:
                    robot = loader.load(config.urdf_path)
                    if robot is None:
                        raise RuntimeError(f"Failed to load URDF: {config.urdf_path}")
                    self.robots.append(robot)
                    sapien_joint_names = [joint.name for joint in robot.get_active_joints()]
                    retarget2sapien = np.array(
                        [retargeting.joint_names.index(n) for n in sapien_joint_names]
                    ).astype(int)
                except (OSError, RuntimeError) as err:
                    if "libqt5core" not in str(err).lower():
                        raise
                    self.headless_mode = True
                    self.robots.append(None)
                    print(
                        f"[RetargetingProcessor] Qt runtime unavailable while loading {config.urdf_path}; "
                        "falling back to headless joint mapping for this robot."
                    )
                    sapien_joint_names = list(retargeting.joint_names)
                    retarget2sapien = np.arange(len(sapien_joint_names), dtype=int)
            self.sapien_joint_names.append(sapien_joint_names)
            self.retarget2sapien.append(retarget2sapien)

        self.mano_layer: Optional[MANOLayer] = None
        self.camera_mat: Optional[np.ndarray] = None
        self.objects: List[sapien.Entity] = []

    def export_joint_mapping_visualization(self, output_dir: str):
        """Export visualizations for retargeting->output joint mapping.

        The saved figures help verify the mapping used in normal and headless modes.
        """
        output_root = Path(output_dir).expanduser()
        output_root.mkdir(parents=True, exist_ok=True)
        summary_lines = [
            f"headless_mode={self.headless_mode}",
            "Columns: output_joint_name, output_index, source_retarget_index, source_retarget_name",
        ]

        for robot_name, retargeting, output_joint_names, mapping in zip(
            self.robot_names, self.retargetings, self.sapien_joint_names, self.retarget2sapien
        ):
            robot_label = ROBOT_NAME_MAP[robot_name]
            mapping = np.asarray(mapping, dtype=int)
            if len(mapping) != len(output_joint_names):
                raise ValueError(
                    f"Mapping length mismatch for {robot_label}: "
                    f"{len(mapping)} vs {len(output_joint_names)}"
                )

            csv_path = output_root / f"{robot_label}_joint_mapping.csv"
            with csv_path.open("w", encoding="utf-8") as f:
                f.write("output_joint_name,output_index,source_retarget_index,source_retarget_name\n")
                for i, (joint_name, src_idx) in enumerate(zip(output_joint_names, mapping)):
                    f.write(f"{joint_name},{i},{src_idx},{retargeting.joint_names[src_idx]}\n")

            summary_lines.append(f"[{robot_label}] rows={len(mapping)} csv={csv_path.name}")

            try:
                import matplotlib.pyplot as plt
            except Exception as err:  # pragma: no cover - best effort visualization
                summary_lines.append(f"[{robot_label}] plot_skipped={err}")
                continue

            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            axes[0].plot(np.arange(len(mapping)), mapping, marker="o", markersize=2, linewidth=1)
            axes[0].set_title(f"{robot_label}: output index -> retarget index")
            axes[0].set_xlabel("output joint index")
            axes[0].set_ylabel("retarget joint index")
            axes[0].grid(True, alpha=0.3)

            perm = np.zeros((len(mapping), len(mapping)), dtype=np.float32)
            perm[np.arange(len(mapping)), mapping] = 1.0
            axes[1].imshow(perm, cmap="viridis", aspect="auto")
            axes[1].set_title(f"{robot_label}: permutation matrix")
            axes[1].set_xlabel("retarget joint index")
            axes[1].set_ylabel("output joint index")

            fig.tight_layout()
            png_path = output_root / f"{robot_label}_joint_mapping.png"
            fig.savefig(png_path, dpi=160)
            plt.close(fig)
            summary_lines.append(f"[{robot_label}] figure={png_path.name}")

        summary_path = output_root / "joint_mapping_summary.txt"
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    def load_object_hand(self, data: Dict):
        self.setup_hand(data)
        if self.scene is None:
            return

        for actor in self.objects:
            self.scene.remove_actor(actor)
        self.objects = []

        mesh_files = data.get("object_mesh_file", [])
        if not isinstance(mesh_files, list):
            mesh_files = [mesh_files]
        
        for mesh_path in mesh_files:
            builder = self.scene.create_actor_builder()
            if mesh_path:
                 builder.add_visual_from_file(str(mesh_path))
                 builder.add_multiple_convex_collisions_from_file(str(mesh_path))
            # Create a separate kinematic actor for each object mesh
            actor = builder.build_kinematic(name="object")
            self.objects.append(actor)

    def setup_hand(self, data: Dict):
        """Initialize MANO layer and camera transform from data.

        Required keys: ``hand_shape``, ``extrinsics``.
        """
        hand_shape = data["hand_shape"]
        extrinsic_mat = data["extrinsics"]
        self.mano_layer = MANOLayer("right", hand_shape.astype(np.float32), mano_root=self.mano_model_dir)
        # In HandViewer, camera_pose is defined as Camera -> World.
        # extrinsic_mat is World -> Camera, so invert it directly for headless compatibility.
        self.camera_mat = np.linalg.inv(extrinsic_mat)

    def _compute_joint_positions(self, hand_pose_frame: np.ndarray) -> Optional[np.ndarray]:
        """Compute MANO joint positions in world frame.

        Returns ``None`` when the hand pose is invalid (near-zero).
        """
        if np.abs(hand_pose_frame).sum() < 1e-5:
            return None
        p = torch.from_numpy(hand_pose_frame[:, :48].astype(np.float32))
        t = torch.from_numpy(hand_pose_frame[:, 48:51].astype(np.float32))
        _, joint = self.mano_layer(p, t)
        joint = joint.cpu().numpy()[0]
        joint = joint @ self.camera_mat[:3, :3].T + self.camera_mat[:3, 3]
        return np.ascontiguousarray(joint)

    def retarget(self, data: Dict) -> Dict:
        """Retarget a hand-pose trajectory to all loaded robots.

        Required keys: ``hand_pose``, ``capture_name``, ``object_pose``,
        ``extrinsics``, ``ycb_ids``, ``hand_shape``.
        Call :meth:`setup_hand` before this method.
        """
        hand_pose = data["hand_pose"]
        num_frame = hand_pose.shape[0]

        # Find first valid frame
        start_frame = 0
        for i in range(num_frame):
            joint = self._compute_joint_positions(hand_pose[i])
            if joint is not None:
                start_frame = i
                break

        # Warm-start retargeting optimizers
        hand_pose_start = hand_pose[start_frame]
        wrist_quat = rotations.quaternion_from_compact_axis_angle(
            hand_pose_start[0, 0:3]
        )
        joint = self._compute_joint_positions(hand_pose_start)
        for retargeting in self.retargetings:
            retargeting.warm_start(
                joint[0, :],
                wrist_quat,
                hand_type=self.hand_type,
                is_mano_convention=True,
            )

        result = {
            "capture_name": data["capture_name"],
            "object_pose": data["object_pose"][start_frame:num_frame],
            "extrinsics": data["extrinsics"],
            "ycb_ids": data["ycb_ids"],
            "hand_shape": data["hand_shape"],
            "hand_pose": hand_pose[start_frame:num_frame, 0, :],
            "start_frame": start_frame,
        }
        if "object_mesh_file" in data:
            result["object_mesh_file"] = data["object_mesh_file"]

        # Retarget each frame
        robot_qpos_trajectories = [[] for _ in range(len(self.robot_names))]
        robot_ee_target_trajectories = [[] for _ in range(len(self.robot_names))]
        mano_joint_trajectory = []
        robot_index = {
            ROBOT_NAME_MAP[name]: i for i, name in enumerate(self.robot_names)
        }

        for i in range(start_frame, num_frame):
            joint = self._compute_joint_positions(hand_pose[i])
            mano_joint_trajectory.append(joint)
            for robotname, retargeting, retarget2sapien in zip(
                self.robot_names, self.retargetings, self.retarget2sapien
            ):
                indices = retargeting.optimizer.target_link_human_indices
                ref_value = joint[indices, :]
                qpos = retargeting.retarget(ref_value)[retarget2sapien]
                ridx = robot_index[ROBOT_NAME_MAP[robotname]]
                robot_qpos_trajectories[ridx].append(qpos)
                robot_ee_target_trajectories[ridx].append(ref_value)

        result["mano_joint_3d"] = np.array(mano_joint_trajectory)
        for robotname, qpos in zip(self.robot_names, robot_qpos_trajectories):
            ridx = robot_index[ROBOT_NAME_MAP[robotname]]
            item = {
                "robot_name": ROBOT_NAME_MAP[robotname],
                "robot_qpos": np.array(qpos),
                "ee_target": np.array(robot_ee_target_trajectories[ridx]),
            }
            result[ROBOT_NAME_MAP[robotname]] = item

        return result
