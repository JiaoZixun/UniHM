import argparse
import os
import traceback

# Force headless-friendly backends to avoid Qt runtime dependency in server environments.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYGLET_HEADLESS", "true")

import numpy as np

for name, value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if not hasattr(np, name):
        setattr(np, name, value)
from tqdm import tqdm

from utils.dataset import DexYCBVideoDataset, YCB_CLASSES
from utils.retargeting_processor import RetargetingProcessor
from dex_retargeting.constants import RobotName, HandType


def _pose7_to_matrix(pose7: np.ndarray) -> np.ndarray:
    """DexYCB [qx,qy,qz,qw,x,y,z] -> 4x4 transform."""
    qx, qy, qz, qw, x, y, z = pose7.astype(np.float64)
    # pytransform3d uses [w, x, y, z].
    q_wxyz = np.array([qw, qx, qy, qz], dtype=np.float64)
    from pytransform3d import rotations as pr

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = pr.matrix_from_quaternion(q_wxyz)
    T[:3, 3] = np.array([x, y, z], dtype=np.float64)
    return T


def _matrix_to_pose7(T: np.ndarray) -> np.ndarray:
    """4x4 transform -> DexYCB [qx,qy,qz,qw,x,y,z]."""
    from pytransform3d import rotations as pr

    pos = T[:3, 3]
    qw, qx, qy, qz = pr.quaternion_from_matrix(T[:3, :3])
    return np.array([qx, qy, qz, qw, pos[0], pos[1], pos[2]], dtype=np.float64)


def _transform_pose_sequence_camera_to_world(object_pose: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    """Convert pose_y sequence from camera frame to world frame."""
    world_from_camera = np.linalg.inv(extrinsics)
    object_pose = np.asarray(object_pose)
    transformed = np.empty_like(object_pose, dtype=np.float64)
    for t in range(object_pose.shape[0]):
        for k in range(object_pose.shape[1]):
            camera_from_object = _pose7_to_matrix(object_pose[t, k])
            world_from_object = world_from_camera @ camera_from_object
            transformed[t, k] = _matrix_to_pose7(world_from_object)
    return transformed


def _min_contact_distance(mano_joint_3d: np.ndarray, object_pose: np.ndarray, grasp_idx: int) -> np.ndarray:
    """Per-frame min distance from hand joints to grasped object center."""
    obj_center = object_pose[:, grasp_idx, 4:7]
    hand_to_center = np.linalg.norm(mano_joint_3d - obj_center[:, None, :], axis=-1)
    return hand_to_center.min(axis=1)


def _transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to (..., 3) points."""
    R = T[:3, :3]
    t = T[:3, 3]
    return points @ R.T + t


def _align_world_data_to_grasped_object_frame(
    object_pose_world: np.ndarray,
    mano_joint_world: np.ndarray,
    grasp_idx: int,
    result: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Align all spatial trajectories to grasped-object local frame per frame."""
    T, K = object_pose_world.shape[:2]
    object_pose_obj = np.empty_like(object_pose_world, dtype=np.float64)
    mano_joint_obj = np.empty_like(mano_joint_world, dtype=np.float64)

    for t in range(T):
        world_from_grasped = _pose7_to_matrix(object_pose_world[t, grasp_idx])
        grasped_from_world = np.linalg.inv(world_from_grasped)

        mano_joint_obj[t] = _transform_points(grasped_from_world, mano_joint_world[t])
        for k in range(K):
            world_from_objk = _pose7_to_matrix(object_pose_world[t, k])
            grasped_from_objk = grasped_from_world @ world_from_objk
            object_pose_obj[t, k] = _matrix_to_pose7(grasped_from_objk)

        # Align retargeted ee targets as well (if present) to keep all modalities consistent.
        for key, value in result.items():
            if not isinstance(value, dict):
                continue
            ee = value.get("ee_target", None)
            if ee is None or t >= ee.shape[0]:
                continue
            if "ee_target_world" not in value:
                value["ee_target_world"] = ee.copy()
            ee[t] = _transform_points(grasped_from_world, ee[t])

    return object_pose_obj, mano_joint_obj


def process(args):
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = DexYCBVideoDataset(args.dexycb_dir, hand_type=args.hand_type, filter_objects=[])
    processor = RetargetingProcessor(
        robot_names=[
            RobotName.allegro,
            RobotName.shadow,
            RobotName.svh,
            RobotName.leap,
            RobotName.ability,
            RobotName.panda,
            RobotName.inspire,
        ],
        hand_type=HandType.right if args.hand_type == "right" else HandType.left,
        urdf_dir=args.retargeting_urdf_dir,
        mano_model_dir=args.mano_model_dir,
    )
    if args.joint_mapping_viz_dir:
        processor.export_joint_mapping_visualization(args.joint_mapping_viz_dir)

    failed = []
    for i in tqdm(range(len(dataset)), desc="DexYCB preprocessing"):
        capture_name = f"sample_{i}"
        try:
            data = dataset[i]
            capture_name = data["capture_name"]
            processor.setup_hand(data)
            result = processor.retarget(data)
            ycb_ids_names = [" ".join(YCB_CLASSES[ycb_id].split("_")[1:]) for ycb_id in data["ycb_ids"]]
            result["ycb_ids_names"] = ycb_ids_names
            move_score = np.sum(data["object_pose"][0] - data["object_pose"][-1] > 0, axis=1)
            grasp_idx = int(np.argmax(move_score))
            result["grasped_ycb_id"] = data["ycb_ids"][grasp_idx]
            result["grasped_ycb_name"] = ycb_ids_names[grasp_idx]
            if "mano_joint_3d" in result:
                # Coordinate-system alignment:
                # mano_joint_3d has already been transformed to world frame in RetargetingProcessor,
                # while DexYCB pose_y is camera-frame. Convert object_pose to world frame before saving.
                object_pose_camera = result["object_pose"]
                object_pose_world = _transform_pose_sequence_camera_to_world(
                    object_pose_camera, result["extrinsics"]
                )
                hand_joints_world = result["mano_joint_3d"]  # (T, 21, 3), world frame
                object_pose_obj, hand_joints_obj = _align_world_data_to_grasped_object_frame(
                    object_pose_world, hand_joints_world, grasp_idx, result
                )

                min_dist_camera = _min_contact_distance(hand_joints_world, object_pose_camera, grasp_idx)
                min_dist_world = _min_contact_distance(hand_joints_world, object_pose_world, grasp_idx)
                min_dist_obj = _min_contact_distance(hand_joints_obj, object_pose_obj, grasp_idx)
                median_camera = float(np.median(min_dist_camera))
                median_world = float(np.median(min_dist_world))
                median_obj = float(np.median(min_dist_obj))

                result["object_pose_camera"] = object_pose_camera
                result["object_pose_world"] = object_pose_world
                result["object_pose"] = object_pose_obj
                result["object_pose_frame"] = "grasped_object_local"
                result["camera_pose_world"] = np.linalg.inv(result["extrinsics"])
                result["mano_joint_3d_world"] = hand_joints_world
                result["mano_joint_3d"] = hand_joints_obj
                result["contact_min_dist_camera"] = min_dist_camera
                result["contact_min_dist_world"] = min_dist_world
                result["contact_min_dist_object"] = min_dist_obj
                result["contact_min_dist"] = min_dist_obj
                result["contact_flag"] = (min_dist_obj < args.contact_threshold).astype(np.uint8)
                result["contact_median_camera"] = median_camera
                result["contact_median_world"] = median_world
                result["contact_median_object"] = median_obj
            np.savez_compressed(os.path.join(args.output_dir, f"{capture_name}.npz"), **result)
        except Exception as e:
            failed.append((capture_name, str(e)))
            if args.print_fail_traceback and len(failed) <= args.max_traceback_print:
                print(f"\n[preprocess] Failed sample: {capture_name}")
                traceback.print_exc()

    print(f"Processed={len(dataset) - len(failed)}, failed={len(failed)}")
    if failed:
        unique_errors = {}
        for _, err in failed:
            unique_errors[err] = unique_errors.get(err, 0) + 1
        print("Top failure reasons:")
        for err, cnt in sorted(unique_errors.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  ({cnt}) {err}")
        with open(os.path.join(args.output_dir, "failed.txt"), "w") as f:
            for n, e in failed:
                f.write(f"{n}\t{e}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser("Preprocess raw DexYCB into UniHM npz format")
    p.add_argument("--dexycb-dir", type=str, default="/public/home/jiaozixun/DexYCB_data")
    p.add_argument("--output-dir", type=str, default="/public/home/jiaozixun/UniHM/processed_dexycb")
    p.add_argument("--hand-type", type=str, default="right", choices=["right", "left"])
    p.add_argument("--retargeting-urdf-dir", type=str,
                   default="/public/home/jiaozixun/dex-retargeting/assets/robots/hands",
                   help="Root directory containing hand URDF subfolders, e.g. .../robots/hands")
    p.add_argument("--mano-model-dir", type=str,
                   default="/public/home/jiaozixun/UniHuman2Rob/manopth/mano/models",
                   help="Directory containing MANO_LEFT.pkl and MANO_RIGHT.pkl")
    p.add_argument(
        "--joint-mapping-viz-dir",
        type=str,
        default=None,
        help="Optional output directory for retargeting->output joint mapping CSV/plots.",
    )
    p.add_argument(
        "--print-fail-traceback",
        action="store_true",
        help="Print traceback for first few failed samples to diagnose environment issues.",
    )
    p.add_argument(
        "--max-traceback-print",
        type=int,
        default=3,
        help="Max number of sample tracebacks printed when --print-fail-traceback is set.",
    )
    p.add_argument(
        "--contact-threshold",
        type=float,
        default=0.008,
        help="Distance threshold (meters) for heuristic hand-object contact flag.",
    )
    args = p.parse_args()
    process(args)
