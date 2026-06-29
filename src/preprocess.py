"""Preprocessing helpers for Himawari-9 and INSAT-3DR thermal imagery."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import h5py
import numpy as np


BT_MIN = 180.0
BT_MAX = 310.0


def normalize_bt(bt: np.ndarray, bt_min: float = BT_MIN, bt_max: float = BT_MAX) -> np.ndarray:
    norm = (np.clip(bt, bt_min, bt_max) - bt_min) / (bt_max - bt_min)
    return norm.astype(np.float32)


def denormalize_bt(norm: np.ndarray, bt_min: float = BT_MIN, bt_max: float = BT_MAX) -> np.ndarray:
    return norm.astype(np.float32) * (bt_max - bt_min) + bt_min


def parse_insat_time(path: str | Path) -> dt.datetime:
    import re

    name = Path(path).name
    match = re.search(r"3RIMG_(\d{2}[A-Z]{3}\d{4})_(\d{4})_", name)
    if not match:
        raise ValueError(f"cannot parse INSAT timestamp from {name}")
    return dt.datetime.strptime(match.group(1) + match.group(2), "%d%b%Y%H%M")


def read_insat_tir1(path: str | Path, bt_min: float = BT_MIN, bt_max: float = BT_MAX):
    """Read INSAT-3DR L1C `IMG_TIR1` and convert counts to normalized brightness temperature."""
    with h5py.File(path, "r") as f:
        raw = np.asarray(f["IMG_TIR1"])
        if raw.ndim == 3:
            raw = raw[0]
        raw = raw.astype(np.int64)

        lut = np.asarray(f["IMG_TIR1_TEMP"]).astype(np.float32).reshape(-1)
        fill = f["IMG_TIR1"].attrs.get("_FillValue", None)
        if hasattr(fill, "tolist"):
            fill = fill.tolist()
        if isinstance(fill, (list, tuple, np.ndarray)):
            fill = fill[0]

        valid = (raw >= 0) & (raw < len(lut))
        if fill is not None:
            valid &= raw != int(fill)

        bt = lut[np.clip(raw, 0, len(lut) - 1)].astype(np.float32)
        valid &= np.isfinite(bt) & (bt > 100.0) & (bt < 400.0)

    norm = normalize_bt(bt, bt_min, bt_max)
    return np.where(valid, norm, 0.0).astype(np.float32), valid.astype(bool), bt
