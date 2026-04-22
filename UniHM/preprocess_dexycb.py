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
            # Lightweight contact heuristic based on distance to grasped object center.
            # object_pose format in DexYCB is [quat(4), trans(3)] so center is [:, 4:7].
            if "mano_joint_3d" in result:
                hand_joints = result["mano_joint_3d"]  # (T, 21, 3)
                obj_center = result["object_pose"][:, grasp_idx, 4:7]  # (T, 3)
                hand_to_center = np.linalg.norm(
                    hand_joints - obj_center[:, None, :], axis=-1
                )  # (T, 21)
                min_dist = hand_to_center.min(axis=1)  # (T,)
                result["contact_min_dist"] = min_dist
                result["contact_flag"] = (min_dist < args.contact_threshold).astype(np.uint8)
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
