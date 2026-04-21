import argparse
import os
import numpy as np
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
            np.savez_compressed(os.path.join(args.output_dir, f"{capture_name}.npz"), **result)
        except Exception as e:
            failed.append((capture_name, str(e)))

    print(f"Processed={len(dataset) - len(failed)}, failed={len(failed)}")
    if failed:
        with open(os.path.join(args.output_dir, "failed.txt"), "w") as f:
            for n, e in failed:
                f.write(f"{n}\t{e}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser("Preprocess raw DexYCB into UniHM npz format")
    p.add_argument("--dexycb-dir", type=str, default="/data1/jiaozx/data/DexYCB_data")
    p.add_argument("--output-dir", type=str, default="/data1/jiaozx/UniHM/processed_dexycb")
    p.add_argument("--hand-type", type=str, default="right", choices=["right", "left"])
    p.add_argument("--retargeting-urdf-dir", type=str,
                   default="/data1/jiaozx/dex-retargeting/assets/robots/hands",
                   help="Root directory containing hand URDF subfolders, e.g. .../robots/hands")
    p.add_argument("--mano-model-dir", type=str,
                   default="/data1/jiaozx/manopth/mano/models",
                   help="Directory containing MANO_LEFT.pkl and MANO_RIGHT.pkl")
    args = p.parse_args()
    process(args)
