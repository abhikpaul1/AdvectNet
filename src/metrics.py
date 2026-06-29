"""Evaluation metrics for normalized satellite frames."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity as ssim_fn


def mse(pred: np.ndarray, target: np.ndarray, axis=None) -> np.ndarray:
    pred = np.clip(pred, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)
    return np.mean((pred - target) ** 2, axis=axis)


def psnr_from_mse(err: np.ndarray | float, eps: float = 1e-12) -> np.ndarray:
    return 10.0 * np.log10(1.0 / np.clip(err, eps, None))


def batch_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Return MSE, PSNR, and SSIM for arrays shaped [N,H,W]."""
    pred = np.clip(pred, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)
    err = mse(pred, target, axis=(1, 2))
    ssim = np.array([ssim_fn(p, t, data_range=1.0) for p, t in zip(pred, target)])
    return {
        "mse": float(err.mean()),
        "psnr": float(psnr_from_mse(err).mean()),
        "ssim": float(ssim.mean()),
    }


def valid_mse(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    valid = valid & np.isfinite(pred) & np.isfinite(target)
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean((np.clip(pred[valid], 0, 1) - np.clip(target[valid], 0, 1)) ** 2))
