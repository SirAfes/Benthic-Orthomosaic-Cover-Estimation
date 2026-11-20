"""Per-mosaic colour normalisation (percentile stretch + gray-world balance).

Underwater mosaics differ in colour cast (depth, water, strobe).  The normaliser is fitted
on valid pixels of ONE mosaic and applied to that mosaic only, so no label information is used.
"""
from __future__ import annotations

import json

import numpy as np


class ColourNormaliser:
    def __init__(self, lo: np.ndarray, hi: np.ndarray, gains: np.ndarray):
        self.lo, self.hi, self.gains = np.asarray(lo, float), np.asarray(hi, float), np.asarray(gains, float)

    @classmethod
    def fit(cls, pixels: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> "ColourNormaliser":
        px = np.asarray(pixels, dtype=float).reshape(-1, 3)
        if len(px) == 0:
            raise ValueError("No pixels to fit the colour normaliser.")
        lo = np.percentile(px, p_lo, axis=0)
        hi = np.percentile(px, p_hi, axis=0)
        hi = np.maximum(hi, lo + 1.0)
        stretched = np.clip((px - lo) / (hi - lo), 0.0, 1.0)
        means = stretched.mean(0)
        gains = means.mean() / np.maximum(means, 1e-6)       # gray-world: equalise channel means
        return cls(lo, hi, gains)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        x = np.asarray(img[..., :3], dtype=float)
        x = np.clip((x - self.lo) / (self.hi - self.lo), 0.0, 1.0) * self.gains
        return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    def to_json(self) -> str:
        return json.dumps({"lo": self.lo.tolist(), "hi": self.hi.tolist(), "gains": self.gains.tolist()})

    @classmethod
    def from_json(cls, s: str) -> "ColourNormaliser":
        d = json.loads(s)
        return cls(d["lo"], d["hi"], d["gains"])


def sample_valid_pixels(reader, grid, n_patches: int = 300, per_patch: int = 400, seed: int = 0) -> np.ndarray:
    """Random pixels from random kept patches (alpha > 0), for fitting the normaliser."""
    rng = np.random.default_rng(seed)
    rows = rng.choice(grid.n, size=min(n_patches, grid.n), replace=False)
    out = []
    for r in rows:
        win = reader.read(int(grid.x0[r]), int(grid.y0[r]), grid.patch_px, grid.patch_px)
        flat = win.reshape(-1, win.shape[-1])
        if flat.shape[1] >= 4:
            flat = flat[flat[:, 3] > 0]
        if len(flat) == 0:
            continue
        pick = rng.choice(len(flat), size=min(per_patch, len(flat)), replace=False)
        out.append(flat[pick, :3])
    return np.concatenate(out) if out else np.zeros((0, 3))
