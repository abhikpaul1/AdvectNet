"""Training utilities for AdvectNet notebooks."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((pred - target) ** 2 + eps**2).mean()


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    gx = ((pred[..., 1:] - pred[..., :-1]).abs() - (target[..., 1:] - target[..., :-1]).abs()).abs().mean()
    gy = (
        (pred[..., 1:, :] - pred[..., :-1, :]).abs()
        - (target[..., 1:, :] - target[..., :-1, :]).abs()
    ).abs().mean()
    return gx + gy


def spatial_gradient(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = F.pad(x[..., 1:] - x[..., :-1], (0, 1))
    dy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
    return dx, dy


def advection_consistency_loss(
    pred: torch.Tensor,
    x0: torch.Tensor,
    ft0: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Motion-consistency prior using intermediate backward flow Ft->0.

    This is a practical scalar-advection regularizer, not a true supervised wind loss.
    """
    dpx, dpy = spatial_gradient(pred)
    residual = (pred - x0) + ft0[:, 0:1] * dpx + ft0[:, 1:2] * dpy
    return torch.sqrt(residual**2 + eps**2).mean()


def sparse_source_loss(residual: torch.Tensor) -> torch.Tensor:
    return residual.abs().mean()


def physics_guided_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    x0: torch.Tensor,
    ft0: torch.Tensor,
    residual: torch.Tensor,
    lambda_rec: float = 1.0,
    lambda_grad: float = 0.5,
    lambda_adv: float = 0.15,
    lambda_src: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    rec = charbonnier_loss(pred, target)
    grad = gradient_loss(pred, target)
    adv = advection_consistency_loss(pred, x0, ft0)
    src = sparse_source_loss(residual)
    total = lambda_rec * rec + lambda_grad * grad + lambda_adv * adv + lambda_src * src
    return total, {"rec": rec, "grad": grad, "adv": adv, "src": src}


def save_checkpoint(path: str | Path, raft, unet, *, optimizer=None, metrics=None, config=None):
    obj = {
        "unet": unet.state_dict(),
        "metrics": metrics or {},
        "config": config or {},
    }
    if raft is not None:
        obj["raft"] = raft.state_dict()
    if optimizer is not None:
        obj["optimizer"] = optimizer.state_dict()
    torch.save(obj, path)


def load_checkpoint(path: str | Path, raft, unet, device="cpu", strict_unet=True):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "raft" in ckpt and ckpt["raft"] is not None and raft is not None:
        raft.load_state_dict(ckpt["raft"], strict=False)
    if isinstance(ckpt, dict) and "unet" in ckpt:
        missing, unexpected = unet.load_state_dict(ckpt["unet"], strict=strict_unet)
    elif isinstance(ckpt, dict) and "model" in ckpt:
        missing, unexpected = unet.load_state_dict(ckpt["model"], strict=strict_unet)
    else:
        missing, unexpected = unet.load_state_dict(ckpt, strict=strict_unet)
    return ckpt, missing, unexpected
