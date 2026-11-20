"""Feature extraction for every kept patch of a grid (resumable) and for arbitrary images.

``backbone`` is any callable ``img(HxWx3 uint8) -> 1-D vector``; if it also has a ``batch``
method taking a list of images, that is used.  The same function is used for the baseline
colour/texture features and for learned backbones, so they are compared like for like.
"""
from __future__ import annotations

import os
from typing import Callable, Optional, Sequence

import numpy as np


def _apply(backbone, imgs: Sequence[np.ndarray]) -> np.ndarray:
    if hasattr(backbone, "batch"):
        return np.asarray(backbone.batch(list(imgs)), dtype=np.float32)
    return np.stack([np.asarray(backbone(i), dtype=np.float32) for i in imgs])


def extract_from_images(images: Sequence[np.ndarray], backbone, batch_size: int = 32) -> np.ndarray:
    out = [_apply(backbone, images[i:i + batch_size]) for i in range(0, len(images), batch_size)]
    return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)


def extract_grid_features(grid, reader, backbone, normaliser: Optional[Callable] = None,
                          batch_size: int = 32, chunk_size: int = 512, chunk_dir: Optional[str] = None,
                          progress: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
    """Features for all kept patches, in grid order.  With ``chunk_dir`` the work resumes after a crash."""
    n = grid.n
    if chunk_dir:
        os.makedirs(chunk_dir, exist_ok=True)
    parts = []
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        path = os.path.join(chunk_dir, f"chunk_{start:08d}.npy") if chunk_dir else None
        if path and os.path.exists(path):
            parts.append(np.load(path))
        else:
            imgs = []
            for r in range(start, stop):
                win = reader.read(int(grid.x0[r]), int(grid.y0[r]), grid.patch_px, grid.patch_px)
                rgb = normaliser(win) if normaliser is not None else np.asarray(win[..., :3])
                imgs.append(np.ascontiguousarray(rgb, dtype=np.uint8))
            feats = extract_from_images(imgs, backbone, batch_size)
            if path:
                tmp = path + ".tmp.npy"
                np.save(tmp, feats)
                os.replace(tmp, path)
            parts.append(feats)
        if progress:
            progress(stop, n)
    return np.concatenate(parts) if parts else np.zeros((0, 0), dtype=np.float32)
