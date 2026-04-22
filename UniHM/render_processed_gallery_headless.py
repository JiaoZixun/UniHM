#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1] if len(_THIS_FILE.parents) > 1 else Path.cwd()
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

try:
    from dexycb_canonical_pipeline.pipeline_config import load_pipeline_config, cfg_get
except Exception:
    def load_pipeline_config(path: str):
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def cfg_get(cfg: Dict[str, Any], *keys, default=None):
        cur = cfg
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

legacy_aliases = {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "str": str,
}
for n, v in legacy_aliases.items():
    if n not in np.__dict__:
        setattr(np, n, v)
if "unicode" not in np.__dict__:
    setattr(np, "unicode", getattr(np, "str_", str))

ROBOT_TO_CONFIG = {
    "allegro": "allegro_hand_right.yml",
    "shadow": "shadow_hand_right.yml",
    "svh": "schunk_svh_hand_right.yml",
    "leap": "leap_hand_right.yml",
    "ability": "ability_hand_right.yml",
    "panda": "panda_gripper.yml",
}
ROBOT_BLOCK_ALIASES = {
    "allegro": ["allegro_hand", "allegro", "allegro_hand_qpos"],
    "shadow": ["shadow_hand", "shadow", "shadow_hand_qpos"],
    "svh": ["schunk_svh_hand", "svh", "svh_hand_qpos"],
    "leap": ["leap_hand", "leap", "leap_hand_qpos"],
    "ability": ["ability_hand", "ability", "ability_hand_qpos"],
    "panda": ["panda_gripper", "gripper", "panda_hand_qpos"],
}

COLOR_BLUE = np.array([0.10, 0.65, 1.00, 1.0], dtype=np.float32)
COLOR_RED = np.array([1.00, 0.15, 0.15, 1.0], dtype=np.float32)
COLOR_HAND = np.array([0.96, 0.75, 0.69, 1.0], dtype=np.float32)
COLOR_TABLE = np.array([0.80, 0.80, 0.80, 1.0], dtype=np.float32)
COLOR_GROUND = np.array([0.90, 0.90, 0.90, 1.0], dtype=np.float32)


def add_repo_to_syspath(repo_root: Path):
    for p in [
        repo_root,
        repo_root / "src",
        repo_root / "example" / "position_retargeting",
        repo_root / "example" / "position_retargeting" / "manopth",
    ]:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def append_stem_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(path.stem + suffix + path.suffix)


def npz_to_dict(path: str | Path) -> Dict[str, Any]:
    out = {}
    with np.load(path, allow_pickle=True) as data:
        for k in data.files:
            v = data[k]
            if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ():
                out[k] = v.item()
            else:
                out[k] = v
    return out


def maybe_object_dict(v):
    if isinstance(v, dict):
        return v
    if isinstance(v, np.ndarray) and v.dtype == object:
        try:
            return dict(v.tolist())
        except Exception:
            if v.shape == ():
                vv = v.item()
                if isinstance(vv, dict):
                    return vv
    return None


def pick_robot_block(seq: Dict[str, Any], key: str):
    for alias in ROBOT_BLOCK_ALIASES.get(key, [key]):
        if alias in seq:
            v = seq[alias]
            block = maybe_object_dict(v)
            if isinstance(block, dict):
                return block
            if isinstance(v, dict):
                return v
    return None


def infer_layout(arr: np.ndarray) -> str:
    a = np.asarray(arr)
    if a.ndim != 3:
        raise ValueError(f"expected 3D array, got {a.shape}")
    n0, n1 = a.shape[:2]
    return "NTC" if n0 <= 128 and n1 >= 8 else "TNC"


def to_tnc(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 3:
        raise ValueError(f"expected 3D array, got {a.shape}")
    return np.transpose(a, (1, 0, 2)) if infer_layout(a) == "NTC" else a


def quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    q = q / np.clip(np.linalg.norm(q), 1e-8, None)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def pose7_qt_first_to_matrix(pose7: np.ndarray) -> np.ndarray:
    p = np.asarray(pose7, dtype=np.float32).reshape(7)
    q_xyzw = p[:4]
    t = p[4:7]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = quat_xyzw_to_rotmat(q_xyzw)
    T[:3, 3] = t
    return T


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    return pts @ T[:3, :3].T + T[:3, 3]


def trimesh_load_meshes(mesh_file: str) -> List[Any]:
    import trimesh

    loaded = trimesh.load(mesh_file, process=False, force="scene", skip_materials=False)
    return scene_like_to_mesh_list(loaded)


def scene_like_to_mesh_list(scene_like: Any) -> List[Any]:
    import trimesh

    if scene_like is None:
        return []
    if isinstance(scene_like, trimesh.Trimesh):
        return [scene_like.copy()]
    if isinstance(scene_like, trimesh.Scene):
        dumped = scene_like.dump(concatenate=False)
        if isinstance(dumped, trimesh.Trimesh):
            return [dumped.copy()]
        if isinstance(dumped, (list, tuple)):
            out = []
            for g in dumped:
                out.extend(scene_like_to_mesh_list(g))
            return out
        return []
    if isinstance(scene_like, (list, tuple)):
        out = []
        for x in scene_like:
            out.extend(scene_like_to_mesh_list(x))
        return out
    return []


def apply_transform_to_meshes(meshes: Iterable[Any], T: np.ndarray) -> List[Any]:
    out = []
    for m in meshes:
        mm = m.copy()
        mm.apply_transform(T)
        out.append(mm)
    return out


def concat_points(point_groups: Iterable[np.ndarray]) -> np.ndarray:
    pts = [np.asarray(p, dtype=np.float32).reshape(-1, 3) for p in point_groups if p is not None and np.asarray(p).size > 0]
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(pts, axis=0)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float32)) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float32).reshape(3)
    target = np.asarray(target, dtype=np.float32).reshape(3)
    up = np.asarray(up, dtype=np.float32).reshape(3)

    z = eye - target
    z = z / np.clip(np.linalg.norm(z), 1e-8, None)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-8:
        alt_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        x = np.cross(alt_up, z)
    x = x / np.clip(np.linalg.norm(x), 1e-8, None)
    y = np.cross(z, x)
    y = y / np.clip(np.linalg.norm(y), 1e-8, None)

    T = np.eye(4, dtype=np.float32)
    T[:3, 0] = x
    T[:3, 1] = y
    T[:3, 2] = z
    T[:3, 3] = eye
    return T


def compute_camera_pose(points: np.ndarray) -> Tuple[np.ndarray, float]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        radius = 0.35
    else:
        pmin = pts.min(axis=0)
        pmax = pts.max(axis=0)
        center = 0.5 * (pmin + pmax)
        radius = float(np.linalg.norm(pmax - pmin) * 0.5)
        radius = max(radius, 0.20)
    target = center + np.array([0.0, 0.0, 0.03], dtype=np.float32)
    direction = np.array([2.2, -0.9, 1.2], dtype=np.float32)
    direction = direction / np.clip(np.linalg.norm(direction), 1e-8, None)
    distance = max(0.55, 3.3 * radius)
    eye = target + direction * distance
    return look_at(eye, target), distance


class MANOExactDecoder:
    def __init__(self, mano_root: str, side: str = "right"):
        import torch
        from manopth.manolayer import ManoLayer

        self.torch = torch
        self.layer = ManoLayer(flat_hand_mean=False, ncomps=45, side=side, mano_root=mano_root, use_pca=True)
        self.faces = self.layer.th_faces.detach().cpu().numpy().astype(np.int32).copy()

    def decode_raw_pose_m(self, pose_m51: np.ndarray, hand_shape10: np.ndarray):
        pose_m51 = np.asarray(pose_m51, dtype=np.float32)
        if pose_m51.ndim == 3 and pose_m51.shape[1] == 1 and pose_m51.shape[2] == 51:
            pose_m51 = pose_m51[:, 0, :]
        hand_shape10 = np.asarray(hand_shape10, dtype=np.float32).reshape(-1)[:10].copy()
        T = pose_m51.shape[0]
        p = self.torch.from_numpy(pose_m51[:, :48].copy()).float()
        t = self.torch.from_numpy(pose_m51[:, 48:51].copy()).float()
        b = self.torch.from_numpy(hand_shape10).float().unsqueeze(0).expand(T, -1)
        with self.torch.no_grad():
            verts, joints = self.layer(p, b, t)
        verts = (verts / 1000.0).cpu().numpy().astype(np.float32)
        joints = (joints / 1000.0).cpu().numpy().astype(np.float32)
        return verts, joints


class RobotMeshProvider:
    def __init__(self, robot_name: str, repo_root: Path, robot_asset_root: Path):
        add_repo_to_syspath(repo_root)
        from dex_retargeting import yourdfpy as urdf
        from dex_retargeting.retargeting_config import RetargetingConfig

        self.robot_name = robot_name
        self.urdf = urdf
        self.RetargetingConfig = RetargetingConfig
        self.RetargetingConfig.set_default_urdf_dir(str(robot_asset_root))

        config_path = repo_root / "src" / "dex_retargeting" / "configs" / "offline" / ROBOT_TO_CONFIG[robot_name]
        config = self.RetargetingConfig.load_from_file(str(config_path), override={"add_dummy_free_joint": True})
        urdf_path = Path(config.urdf_path)
        if "glb" not in urdf_path.stem:
            urdf_path = append_stem_suffix(urdf_path, "_glb")

        try:
            self.model = self.urdf.URDF.load(str(urdf_path), add_dummy_free_joints=True, build_scene_graph=True)
        except TypeError:
            self.model = self.urdf.URDF.load(str(urdf_path), build_scene_graph=True)

        self.actuated_joint_names = list(getattr(self.model, "actuated_joint_names", []))
        if len(self.actuated_joint_names) == 0:
            joints = getattr(self.model, "actuated_joints", [])
            self.actuated_joint_names = [j.name for j in joints]

    def reorder_qpos_if_needed(self, block: Dict[str, Any], qpos_t: np.ndarray) -> np.ndarray:
        qpos_t = np.asarray(qpos_t, dtype=np.float32).reshape(-1)
        saved_names = [str(x) for x in np.asarray(block["robot_joint_names"]).reshape(-1).tolist()] if "robot_joint_names" in block else []

        if len(saved_names) == qpos_t.shape[0] and len(self.actuated_joint_names) > 0:
            name_to_idx = {n: i for i, n in enumerate(saved_names)}
            filtered = [name_to_idx[n] for n in self.actuated_joint_names if n in name_to_idx]
            if len(filtered) == len(self.actuated_joint_names):
                return np.asarray([qpos_t[i] for i in filtered], dtype=np.float32)

        if len(self.actuated_joint_names) == 0 or qpos_t.shape[0] == len(self.actuated_joint_names):
            return qpos_t.astype(np.float32)

        raise ValueError(f"{self.robot_name} qpos dim mismatch: saved={qpos_t.shape[0]}, active={len(self.actuated_joint_names)}")

    def _update_cfg(self, qpos_t: np.ndarray):
        qpos_t = np.asarray(qpos_t, dtype=np.float32).reshape(-1)
        if len(self.actuated_joint_names) == 0:
            return
        cfg_dict = {n: float(v) for n, v in zip(self.actuated_joint_names, qpos_t.tolist())}
        for fn in [
            lambda: self.model.update_cfg(cfg_dict),
            lambda: self.model.update_cfg(configuration=cfg_dict),
            lambda: self.model.update_cfg(qpos_t),
            lambda: self.model.update_cfg(configuration=qpos_t),
        ]:
            try:
                fn()
                return
            except Exception:
                continue
        raise RuntimeError("update_cfg failed for yourdfpy model")

    def get_meshes(self, block: Dict[str, Any], qpos_t: np.ndarray) -> List[Any]:
        q = self.reorder_qpos_if_needed(block, qpos_t)
        self._update_cfg(q)
        return scene_like_to_mesh_list(getattr(self.model, "scene", None))


class PyrenderPanelRenderer:
    def __init__(self, width: int, height: int):
        os.environ.setdefault("PYOPENGL_PLATFORM", os.environ.get("HEADLESS_GL_PLATFORM", "egl"))
        import pyrender
        import trimesh

        self.pyrender = pyrender
        self.trimesh = trimesh
        self.width = int(width)
        self.height = int(height)
        self.renderer = pyrender.OffscreenRenderer(viewport_width=self.width, viewport_height=self.height)
        self.camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(55.0), znear=0.01, zfar=20.0)
        self.hand_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=COLOR_HAND.tolist(), metallicFactor=0.0, roughnessFactor=0.75)
        self.table_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=COLOR_TABLE.tolist(), metallicFactor=0.0, roughnessFactor=0.90)
        self.blue_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=COLOR_BLUE.tolist(), metallicFactor=0.0, roughnessFactor=0.35)
        self.red_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=COLOR_RED.tolist(), metallicFactor=0.0, roughnessFactor=0.35)
        self.sphere_cache: Dict[float, Any] = {}

    def _get_sphere(self, radius: float):
        key = round(float(radius), 6)
        if key not in self.sphere_cache:
            self.sphere_cache[key] = self.trimesh.creation.icosphere(subdivisions=2, radius=float(radius))
        return self.sphere_cache[key]

    def _add_meshes(self, scene, meshes: Iterable[Any], material=None, smooth=False):
        for mesh in meshes:
            if mesh is None:
                continue
            try:
                m = self.pyrender.Mesh.from_trimesh(mesh, material=material, smooth=smooth)
            except TypeError:
                m = self.pyrender.Mesh.from_trimesh(mesh, smooth=smooth)
            scene.add(m)

    def render(self, subject_meshes, object_meshes, markers_blue, markers_red, camera_pose, marker_radius, show_markers=True, subject_material=None):
        scene = self.pyrender.Scene(bg_color=[0.05, 0.05, 0.05, 1.0], ambient_light=[0.22, 0.22, 0.22])
        scene.add(self.camera, pose=camera_pose)
        self._add_meshes(scene, object_meshes, material=None, smooth=False)
        self._add_meshes(scene, subject_meshes, material=subject_material, smooth=False)
        if show_markers:
            sphere = self._get_sphere(marker_radius)
            blue_meshes = [sphere.copy() for _ in range(len(markers_blue))]
            red_meshes = [sphere.copy() for _ in range(len(markers_red))]
            for m, p in zip(blue_meshes, np.asarray(markers_blue, dtype=np.float32).reshape(-1, 3)):
                m.apply_translation(p)
            for m, p in zip(red_meshes, np.asarray(markers_red, dtype=np.float32).reshape(-1, 3)):
                m.apply_translation(p)
            self._add_meshes(scene, blue_meshes, material=self.blue_material, smooth=True)
            self._add_meshes(scene, red_meshes, material=self.red_material, smooth=True)
        color, _ = self.renderer.render(scene, flags=self.pyrender.constants.RenderFlags.RGBA)
        return np.asarray(color[..., :3], dtype=np.uint8)


class HeadlessProcessedMultiRobotViewer:
    def __init__(self, robot_names, mano_root, repo_root, robot_asset_root, width=1920, height=1080, show_markers=True):
        self.robot_names_requested = list(robot_names)
        self.show_markers = bool(show_markers)
        self.panel_w = int(width) // 4
        self.panel_h = int(height) // 2
        self.height = int(height)
        self.width = int(width)
        self.repo_root = repo_root
        self.robot_asset_root = robot_asset_root
        self.mano = MANOExactDecoder(mano_root, side="right")
        self.renderer = PyrenderPanelRenderer(width=self.panel_w, height=self.panel_h)
        self.object_meshes_local = []
        self.robot_providers = {}
        self.panel_camera_poses = {}

    def _split_contact_points(self, xyzc: np.ndarray):
        arr = np.asarray(xyzc, dtype=np.float32)
        xyz = arr[:, :3]
        contact = arr[:, 3] > 0.5
        return xyz[~contact], xyz[contact]

    def _compose_grid(self, panel_images: Dict[str, np.ndarray], t: int):
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        slots = {"human": (0, 0), "allegro": (1, 0), "shadow": (2, 0), "svh": (3, 0), "leap": (0, 1), "ability": (1, 1), "panda": (2, 1)}
        for name, img in panel_images.items():
            if name not in slots:
                continue
            col, row = slots[name]
            x0 = col * self.panel_w
            y0 = row * self.panel_h
            canvas[y0 : y0 + self.panel_h, x0 : x0 + self.panel_w] = cv2.resize(img, (self.panel_w, self.panel_h), interpolation=cv2.INTER_AREA)
            cv2.putText(canvas, name, (x0 + 18, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"t={t}", (self.panel_w * 3 + 22, self.panel_h + 142), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas

    def load_sequence_assets(self, seq: Dict[str, Any], requested_robot_names: List[str]):
        active_robot_names = []
        for rk in requested_robot_names:
            if pick_robot_block(seq, rk) is not None:
                active_robot_names.append(rk)
        self.object_meshes_local = trimesh_load_meshes(str(np.asarray(seq["object_mesh_file"]).reshape(-1)[0]))
        for rk in active_robot_names:
            self.robot_providers[rk] = RobotMeshProvider(rk, self.repo_root, self.robot_asset_root)
        return active_robot_names

    def render_sequence(self, seq: Dict[str, Any], active_robot_names: List[str], save_dir: str, fps: int = 10):
        import trimesh

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        hand_pose = np.asarray(seq["hand_pose"], dtype=np.float32)
        hand_shape = np.asarray(seq["hand_shape"], dtype=np.float32)
        extrinsics = np.asarray(seq["extrinsics"], dtype=np.float32)
        world_T_camera = np.linalg.inv(extrinsics).astype(np.float32)
        verts_cam_tnc, _ = self.mano.decode_raw_pose_m(hand_pose, hand_shape)
        verts_world_tnc = np.stack([verts_cam_tnc[t] @ world_T_camera[:3, :3].T + world_T_camera[:3, 3] for t in range(verts_cam_tnc.shape[0])], axis=0)
        human_xyzc_tnc = to_tnc(seq["human_xyzc_binary"])
        object_pose = np.asarray(seq["grasped_obj_pose"], dtype=np.float32)
        robot_xyzc_cache = {}
        robot_qpos_cache = {}
        for rk in active_robot_names:
            block = pick_robot_block(seq, rk)
            robot_xyzc_cache[rk] = to_tnc(block["robot_xyzc_binary"])
            robot_qpos_cache[rk] = np.asarray(block["robot_qpos"], dtype=np.float32)

        object_points = concat_points([m.vertices for m in self.object_meshes_local])
        self.panel_camera_poses["human"] = compute_camera_pose(concat_points([verts_world_tnc.reshape(-1, 3), object_points]))[0]
        for rk in active_robot_names:
            self.panel_camera_poses[rk] = compute_camera_pose(concat_points([robot_xyzc_cache[rk][..., :3].reshape(-1, 3), object_points]))[0]

        frames = []
        T = hand_pose.shape[0]
        for t in range(T):
            object_meshes_world = apply_transform_to_meshes(self.object_meshes_local, pose7_qt_first_to_matrix(object_pose[t]))
            panel_images = {}
            human_mesh = trimesh.Trimesh(vertices=verts_world_tnc[t], faces=self.mano.faces, process=False)
            hblue, hred = self._split_contact_points(human_xyzc_tnc[t])
            panel_images["human"] = self.renderer.render([human_mesh], object_meshes_world, hblue, hred, self.panel_camera_poses["human"], 0.013, self.show_markers, self.renderer.hand_material)
            for rk in active_robot_names:
                block = pick_robot_block(seq, rk)
                robot_meshes = self.robot_providers[rk].get_meshes(block, robot_qpos_cache[rk][t])
                rblue, rred = self._split_contact_points(robot_xyzc_cache[rk][t])
                panel_images[rk] = self.renderer.render(robot_meshes, object_meshes_world, rblue, rred, self.panel_camera_poses[rk], 0.015, self.show_markers, None)
            frames.append(self._compose_grid(panel_images, t))

        video_path = save_dir / "processed_gallery_headless.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (frames[0].shape[1], frames[0].shape[0]))
        for f in frames:
            writer.write(f[..., ::-1])
        writer.release()
        imageio.imwrite(save_dir / "first_frame.png", frames[0])
        imageio.imwrite(save_dir / "last_frame.png", frames[-1])
        with (save_dir / "sequence_qc.json").open("w", encoding="utf-8") as f:
            json.dump({"renderer": "pyrender_offscreen", "num_frames": len(frames)}, f, ensure_ascii=False, indent=2)
        return {"video_path": str(video_path)}


def main():
    default_config = _PROJECT_ROOT / "configs" / "dexycb_pipeline_config.yaml"
    parser = argparse.ArgumentParser(description="Headless processed multi-robot gallery renderer based on pyrender")
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--input", required=True)
    parser.add_argument("--save-dir", required=False, default=None)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--robot-asset-root", default=None)
    parser.add_argument("--mano-root", default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--robots", nargs="+", default=None)
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    seq_path = Path(args.input).expanduser().resolve()
    seq_name = seq_path.stem
    args.save_dir = args.save_dir or str(Path(cfg_get(cfg, "paths", "viz_root", default="./viz")) / seq_name)
    args.repo_root = args.repo_root or cfg_get(cfg, "paths", "repo_root")
    args.robot_asset_root = args.robot_asset_root or cfg_get(cfg, "paths", "robot_asset_root")
    args.mano_root = args.mano_root or cfg_get(cfg, "paths", "mano_root")
    args.fps = args.fps if args.fps is not None else int(cfg_get(cfg, "render", "fps", default=10))
    args.robots = args.robots or cfg_get(cfg, "robots", "keys", default=["allegro", "shadow", "svh", "leap", "ability", "panda"])

    viewer = HeadlessProcessedMultiRobotViewer(
        robot_names=args.robots,
        mano_root=str(args.mano_root),
        repo_root=Path(args.repo_root).expanduser().resolve(),
        robot_asset_root=Path(args.robot_asset_root).expanduser().resolve(),
    )
    seq = npz_to_dict(seq_path)
    active_robot_names = viewer.load_sequence_assets(seq, args.robots)
    out = viewer.render_sequence(seq, active_robot_names, args.save_dir, fps=args.fps)
    print("[OK] outputs:")
    for k, v in out.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
