import numpy as np
from scipy import linalg


def _to_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3:
        return x.mean(0)
    return x


def truncate_pair(a: np.ndarray, b: np.ndarray, max_t: int = 72):
    a2, b2 = _to_2d(a), _to_2d(b)
    t = min(max_t, a2.shape[0], b2.shape[0])
    return a2[:t], b2[:t]


def mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = truncate_pair(pred, gt)
    if p.shape[0] == 0:
        return 0.0
    return float(np.abs(p[:, 6:] - g[:, 6:]).sum() / p.shape[0])


def fhlt(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = truncate_pair(pred, gt)
    if p.shape[0] == 0:
        return 0.0
    return float(np.abs(p[:, :3] - g[:, :3]).sum() / p.shape[0])


def fhlr(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = truncate_pair(pred, gt)
    if p.shape[0] == 0:
        return 0.0
    return float(np.abs(p[:, 3:6] - g[:, 3:6]).sum() / p.shape[0])


def fid(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    p = pred.reshape(-1, pred.shape[-1]) if pred.ndim == 3 else pred
    g = gt.reshape(-1, gt.shape[-1]) if gt.ndim == 3 else gt
    mu1, mu2 = p.mean(0), g.mean(0)
    sig1, sig2 = np.cov(p, rowvar=False), np.cov(g, rowvar=False)
    sig1 += np.eye(sig1.shape[0]) * eps
    sig2 += np.eye(sig2.shape[0]) * eps
    covmean, _ = linalg.sqrtm(sig1.dot(sig2), disp=False)
    if not np.isfinite(covmean).all():
        covmean = np.eye(sig1.shape[0])
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(sig1 + sig2 - 2 * covmean))


def smoothness_l2(seq: np.ndarray) -> float:
    s = _to_2d(seq)
    if s.shape[0] < 3:
        return 0.0
    d2 = s[2:] - 2 * s[1:-1] + s[:-2]
    return float(np.linalg.norm(d2, axis=-1).mean())


def rollout_drift(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = truncate_pair(pred, gt)
    if p.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(p[-1] - g[-1]))
