"""Patchwise AdvectNet inference helpers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def starts(n: int, patch: int = 256, stride: int = 256) -> list[int]:
    if n <= patch:
        return [0]
    vals = list(range(0, n - patch + 1, stride))
    if vals[-1] != n - patch:
        vals.append(n - patch)
    return vals


def backwarp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    b, _, h, w = img.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=img.device),
        torch.arange(w, device=img.device),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), 0).float()[None].repeat(b, 1, 1, 1) + flow
    gx = 2 * grid[:, 0] / max(w - 1, 1) - 1
    gy = 2 * grid[:, 1] / max(h - 1, 1) - 1
    return F.grid_sample(
        img,
        torch.stack((gx, gy), -1),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def inter_flows(f03: torch.Tensor, f30: torch.Tensor, alpha: torch.Tensor):
    ft0 = -(1 - alpha) * alpha * f03 + alpha * alpha * f30
    ft3 = (1 - alpha) ** 2 * f03 - alpha * (1 - alpha) * f30
    return ft0, ft3


@torch.no_grad()
def tiled_predict(predict_patch, x0, x3, m0, m3, alpha: float, patch: int = 256, stride: int = 256):
    """Stitch full-frame predictions from a callable operating on patch batches."""
    h, w = x0.shape
    out = np.zeros((h, w), np.float32)
    wgt = np.zeros((h, w), np.float32)
    win1 = np.hanning(patch).astype(np.float32)
    win = np.maximum(np.outer(win1, win1), 0.05).astype(np.float32)

    for y in starts(h, patch, stride):
        for x in starts(w, patch, stride):
            sl = (slice(y, y + patch), slice(x, x + patch))
            if (m0[sl] & m3[sl]).mean() < 0.25:
                continue
            pred = predict_patch(x0[sl][None], x3[sl][None], alpha)[0]
            out[sl] += pred * win
            wgt[sl] += win

    base = (1 - alpha) * x0 + alpha * x3
    return np.where(wgt > 0, out / np.maximum(wgt, 1e-6), base).astype(np.float32)
