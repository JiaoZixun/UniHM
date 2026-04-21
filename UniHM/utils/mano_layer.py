# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""Wrapper layer for manopth ManoLayer."""

import os
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import torch

from torch.nn import Module


def _rodrigues_numpy(src):
    src = np.asarray(src, dtype=np.float64)
    if src.shape == (3, 3):
        trace = np.trace(src)
        cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        if theta < 1e-12:
            return np.zeros((3, 1), dtype=np.float64), None
        denom = 2.0 * np.sin(theta)
        rx = (src[2, 1] - src[1, 2]) / denom
        ry = (src[0, 2] - src[2, 0]) / denom
        rz = (src[1, 0] - src[0, 1]) / denom
        rvec = theta * np.array([[rx], [ry], [rz]], dtype=np.float64)
        return rvec, None

    rvec = src.reshape(3, 1)
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64), None
    k = (rvec[:, 0] / theta).astype(np.float64)
    kx, ky, kz = k
    k_mat = np.array(
        [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]],
        dtype=np.float64,
    )
    r = (
        np.eye(3, dtype=np.float64)
        + np.sin(theta) * k_mat
        + (1.0 - np.cos(theta)) * (k_mat @ k_mat)
    )
    return r, None


def _ensure_cv2_or_stub():
    try:
        importlib.import_module("cv2")
        return
    except ImportError as err:
        if "libqt5core" not in str(err).lower():
            raise
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.Rodrigues = _rodrigues_numpy
    sys.modules["cv2"] = cv2_stub


class MANOLayer(Module):
    """Wrapper layer for manopth ManoLayer."""

    def __init__(self, side, betas, mano_root=None):
        """Constructor.
        Args:
          side: MANO hand type. 'right' or 'left'.
          betas: A numpy array of shape [10] containing the betas.
        """
        super(MANOLayer, self).__init__()

        self._side = side
        self._betas = betas
        if mano_root is None:
            mano_root = os.environ.get("UNIHM_MANO_MODEL_DIR", "../../dex-ycb-toolkit/manopth/mano/models")
        mano_root = str(Path(mano_root).expanduser())
        if not Path(mano_root).exists():
            raise FileNotFoundError(f"MANO model directory not found: {mano_root}")

        _ensure_cv2_or_stub()
        mano_layer_cls = importlib.import_module("manopth.manolayer").ManoLayer
        self._mano_layer = mano_layer_cls(
            flat_hand_mean=False,
            ncomps=45,
            side=self._side,
            mano_root=mano_root,
            use_pca=True,
        )

        b = torch.from_numpy(self._betas).unsqueeze(0)
        f = self._mano_layer.th_faces
        self.register_buffer("b", b)
        self.register_buffer("f", f)

        v = (
            torch.matmul(self._mano_layer.th_shapedirs, self.b.transpose(0, 1)).permute(
                2, 0, 1
            )
            + self._mano_layer.th_v_template
        )
        r = torch.matmul(self._mano_layer.th_J_regressor[0], v)
        self.register_buffer("root_trans", r)

    def forward(self, p, t):
        """Forward function.
        Args:
          p: A tensor of shape [B, 48] containing the pose.
          t: A tensor of shape [B, 3] containing the trans.
        Returns:
          v: A tensor of shape [B, 778, 3] containing the vertices.
          j: A tensor of shape [B, 21, 3] containing the joints.
        """
        v, j = self._mano_layer(p, self.b.expand(p.size(0), -1), t)
        v /= 1000
        j /= 1000
        return v, j
