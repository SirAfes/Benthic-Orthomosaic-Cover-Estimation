"""Percent-cover density grids and object components from a patch-label map.

The patch map has the resolution of the patch size (e.g. 5 cm).  Density cells aggregate many
patches (e.g. 10 x 10 patches = 50 cm).  Raw cell fractions are shown for visualisation only:
the mosaic-level cover numbers come from ``estimation`` (design-based, with intervals).
"""
from __future__ import annotations

import csv

import numpy as np
from scipy import ndimage


def label_grid(grid, labels: np.ndarray, fill: str = "") -> np.ndarray:
    """Patch labels as a (n_rows, n_cols) object array (``fill`` where no patch)."""
    out = np.full((grid.n_rows, grid.n_cols), fill, dtype=object)
    iy = (grid.y0 // grid.patch_px).astype(int)
    ix = (grid.x0 // grid.patch_px).astype(int)
    out[iy, ix] = np.asarray(labels, dtype=object)
    return out


def density_cells(lgrid: np.ndarray, classes: list[str], cell_patches: int = 10, min_patches: int = 20):
    """Fraction of each class per cell of ``cell_patches`` x ``cell_patches`` patches.

    Returns (fractions[n_cy, n_cx, n_classes], counts[n_cy, n_cx]); cells with fewer than
    ``min_patches`` valid patches are NaN so partial edge cells are not over-interpreted.
    """
    n_rows, n_cols = lgrid.shape
    ncy, ncx = -(-n_rows // cell_patches), -(-n_cols // cell_patches)
    fr = np.full((ncy, ncx, len(classes)), np.nan)
    cnt = np.zeros((ncy, ncx), dtype=int)
    for cy in range(ncy):
        for cx in range(ncx):
            blk = lgrid[cy * cell_patches:(cy + 1) * cell_patches, cx * cell_patches:(cx + 1) * cell_patches]
            valid = blk != ""
            n = int(valid.sum())
            cnt[cy, cx] = n
            if n >= min_patches:
                vals = blk[valid]
                for j, c in enumerate(classes):
                    fr[cy, cx, j] = float((vals == c).sum()) / n
    return fr, cnt


def write_density_csv(path: str, fr: np.ndarray, cnt: np.ndarray, classes: list[str], cell_px: int,
                      transform=None) -> None:
    """Long-format CSV (cell centre in px and, if an affine is given, in world units)."""
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["cell_row", "cell_col", "x_px", "y_px", "x_m", "y_m", "n_patches", "class", "fraction"])
        for cy in range(fr.shape[0]):
            for cx in range(fr.shape[1]):
                if np.isnan(fr[cy, cx]).all():
                    continue
                x, y = (cx + 0.5) * cell_px, (cy + 0.5) * cell_px
                xm, ym = ((transform.a * x + transform.c), (transform.e * y + transform.f)) if transform else ("", "")
                for j, c in enumerate(classes):
                    wr.writerow([cy, cx, f"{x:.1f}", f"{y:.1f}", xm, ym, cnt[cy, cx], c, f"{fr[cy, cx, j]:.4f}"])


def components(lgrid: np.ndarray, cls: str, min_patches: int = 1, patch_m: float = 0.05):
    """Connected components (8-neighbour) of one class on the patch grid, with area and bbox.

    Resolution is the patch size; use for counts/rough extents of discrete organisms, not for
    precise outlines.  Returns a list of dicts sorted by area.
    """
    mask = lgrid == cls
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        area = int((lab[sl] == i).sum())
        if area < min_patches:
            continue
        ys, xs = sl
        out.append({"id": i, "n_patches": area, "area_m2": area * patch_m ** 2,
                    "row0": ys.start, "row1": ys.stop, "col0": xs.start, "col1": xs.stop})
    return sorted(out, key=lambda d: -d["n_patches"])
