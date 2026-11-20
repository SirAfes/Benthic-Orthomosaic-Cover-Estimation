"""Fixed-physical-size patch grid over a mosaic, plus window readers.

Patches are non-overlapping squares of ``patch_px`` pixels (stride = patch size), cut at a
fixed physical size (``patch_cm``) so that mosaics with different GSD (e.g. 0.41 and
0.66 mm/px) give patches covering the same real-world area.  Backbones resize the patch
to their own input size, which is what normalises scale.

A patch is kept when at least ``min_valid_frac`` of it is valid (alpha > 0), judged on a
decimated mask.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def patch_px_for(patch_cm: float, mm_per_px: float) -> int:
    return max(1, int(round(patch_cm * 10.0 / mm_per_px)))


@dataclass(frozen=True)
class Grid:
    patch_px: int
    width: int
    height: int
    n_rows: int
    n_cols: int
    x0: np.ndarray            # (n,) top-left x of kept patches, px
    y0: np.ndarray
    valid_frac: np.ndarray
    lookup: np.ndarray        # (n_rows, n_cols) -> row in the arrays above, or -1

    @property
    def n(self) -> int:
        return int(self.x0.size)

    @property
    def centers(self) -> np.ndarray:
        h = self.patch_px / 2.0
        return np.column_stack([self.x0 + h, self.y0 + h])

    def save(self, path: str) -> None:
        np.savez(path, x0=self.x0, y0=self.y0, valid_frac=self.valid_frac, patch_px=self.patch_px,
                 width=self.width, height=self.height, n_rows=self.n_rows, n_cols=self.n_cols)

    @classmethod
    def load(cls, path: str) -> "Grid":
        d = np.load(path)
        n_rows, n_cols, p = int(d["n_rows"]), int(d["n_cols"]), int(d["patch_px"])
        x0, y0 = d["x0"].astype(int), d["y0"].astype(int)
        lookup = np.full((n_rows, n_cols), -1, dtype=int)
        lookup[y0 // p, x0 // p] = np.arange(len(x0))
        return cls(p, int(d["width"]), int(d["height"]), n_rows, n_cols, x0, y0, d["valid_frac"], lookup)

    def in_mask(self, mask_low: np.ndarray, scale_xy: tuple[float, float]) -> np.ndarray:
        """Boolean per kept patch: is its centre inside ``mask_low`` (e.g. the eroded population)?"""
        c = self.centers
        H, W = mask_low.shape
        cx = np.clip((c[:, 0] / scale_xy[0]).astype(int), 0, W - 1)
        cy = np.clip((c[:, 1] / scale_xy[1]).astype(int), 0, H - 1)
        return mask_low[cy, cx]

    def index_of(self, xy: np.ndarray) -> np.ndarray:
        """Row of the kept patch containing each (x_px, y_px); -1 if outside the grid or not kept."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        ix = np.floor(xy[:, 0] / self.patch_px).astype(int)
        iy = np.floor(xy[:, 1] / self.patch_px).astype(int)
        ok = (ix >= 0) & (ix < self.n_cols) & (iy >= 0) & (iy < self.n_rows)
        out = np.full(len(xy), -1, dtype=int)
        out[ok] = self.lookup[iy[ok], ix[ok]]
        return out


def make_grid(mask_low: np.ndarray, scale_xy: tuple[float, float], width: int, height: int,
              patch_px: int, min_valid_frac: float = 0.98) -> Grid:
    mask_low = np.asarray(mask_low, dtype=bool)
    sx, sy = scale_xy
    H, W = mask_low.shape
    n_cols, n_rows = width // patch_px, height // patch_px
    if n_cols < 1 or n_rows < 1:
        raise ValueError("Patch larger than the mosaic.")
    sat = np.zeros((H + 1, W + 1), dtype=np.int64)
    sat[1:, 1:] = np.cumsum(np.cumsum(mask_low.astype(np.int64), 0), 1)
    xs = np.arange(n_cols) * patch_px
    ys = np.arange(n_rows) * patch_px
    c0 = np.clip(np.floor(xs / sx).astype(int), 0, W - 1)
    c1 = np.clip(np.maximum(c0 + 1, np.ceil((xs + patch_px) / sx).astype(int)), 1, W)
    r0 = np.clip(np.floor(ys / sy).astype(int), 0, H - 1)
    r1 = np.clip(np.maximum(r0 + 1, np.ceil((ys + patch_px) / sy).astype(int)), 1, H)
    tot = (sat[np.ix_(r1, c1)] - sat[np.ix_(r0, c1)] - sat[np.ix_(r1, c0)] + sat[np.ix_(r0, c0)])
    area = (r1 - r0)[:, None] * (c1 - c0)[None, :]
    frac = tot / np.maximum(area, 1)
    keep = frac >= min_valid_frac
    iy, ix = np.nonzero(keep)
    lookup = np.full((n_rows, n_cols), -1, dtype=int)
    lookup[iy, ix] = np.arange(len(iy))
    return Grid(patch_px, width, height, n_rows, n_cols, (ix * patch_px).astype(int),
                (iy * patch_px).astype(int), frac[iy, ix], lookup)


class ArrayReader:
    """Window reader over an in-memory HxWxC array (tests, small mosaics)."""

    def __init__(self, arr: np.ndarray):
        self.arr = arr
        self.height, self.width = arr.shape[:2]

    def read(self, x0: int, y0: int, w: int, h: int) -> np.ndarray:
        return self.arr[y0:y0 + h, x0:x0 + w]


class RasterioReader:
    """Windowed reads from a GeoTIFF (needs rasterio).  Returns HxWxbands uint8."""

    def __init__(self, path: str, bands: tuple[int, ...] = (1, 2, 3, 4)):
        import rasterio  # lazy

        self._src = rasterio.open(path)
        self.width, self.height = self._src.width, self._src.height
        self.bands = tuple(b for b in bands if b <= self._src.count)
        self._window = rasterio.windows.Window

    def read(self, x0: int, y0: int, w: int, h: int) -> np.ndarray:
        arr = self._src.read(list(self.bands), window=self._window(x0, y0, w, h))
        return np.moveaxis(arr, 0, -1)

    def close(self) -> None:
        self._src.close()
