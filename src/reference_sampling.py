"""Reference-sample design for benthic orthomosaics.

Draws a *simple random* sample of points (and optional dense "gold" windows)
over the valid area of a mosaic, BEFORE any model exists.  The sample is the
probability sample that later supports design-based accuracy and cover
estimates (Olofsson et al. 2013/2014; prediction-powered inference).

Design notes
------------
* Population = valid (alpha > 0) area, eroded by ``edge_buffer_px`` so that
  seams, holes and mosaic edges are excluded.  Points near those edges have
  inclusion probability zero; state this in any report.
* Valid area is discretised on a decimated mask whose cells have equal area,
  so choosing a valid cell uniformly and jittering inside it is uniform over
  the valid area.
* An optional full-resolution check rejects points that fall on invalid
  pixels (rejection sampling keeps the sample uniform over the valid set).
* Everything except ``read_mask_rasterio`` is numpy-only and unit-tested.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Callable, Optional

import numpy as np


def erode(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    """Binary erosion with a (2r+1)^2 square element; border counts as invalid."""
    m = np.asarray(mask, dtype=bool)
    r = int(radius_cells)
    if r <= 0:
        return m.copy()
    padded = np.pad(m, r, constant_values=False)
    out = np.ones_like(m)
    h, w = m.shape
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            out &= padded[dy:dy + h, dx:dx + w]
    return out


def eroded_mask(mask_low: np.ndarray, scale_xy: tuple[float, float], mm_per_px: float,
                edge_buffer_mm: float) -> np.ndarray:
    """The sampling population: valid area minus an edge band.  Used by BOTH the sampler and
    the estimator so that they refer to exactly the same population."""
    cells = int(np.ceil(edge_buffer_mm / mm_per_px / max(scale_xy)))
    return erode(mask_low, cells)


def draw_points(
    mask_low: np.ndarray,
    scale_xy: tuple[float, float],
    n: int,
    seed: int,
    full_res_valid: Optional[Callable[[float, float], bool]] = None,
    max_rounds: int = 200,
) -> np.ndarray:
    """Return (n, 2) array of (x_px, y_px) in full-resolution pixel units."""
    cells = np.flatnonzero(mask_low.ravel())
    if cells.size == 0:
        raise ValueError("No valid cells left in the mask; check edge buffer / mask.")
    rng = np.random.default_rng(seed)
    h, w = mask_low.shape
    sx, sy = scale_xy
    accepted: list[tuple[float, float]] = []
    for _ in range(max_rounds):
        need = n - len(accepted)
        if need <= 0:
            break
        m = max(need * 2, 16)
        pick = rng.choice(cells, size=m, replace=True)
        rows, cols = np.divmod(pick, w)
        xs = (cols + rng.random(m)) * sx
        ys = (rows + rng.random(m)) * sy
        for x, y in zip(xs, ys):
            if full_res_valid is None or full_res_valid(float(x), float(y)):
                accepted.append((float(x), float(y)))
                if len(accepted) == n:
                    break
    if len(accepted) < n:
        raise RuntimeError(
            f"Only {len(accepted)}/{n} points passed the full-resolution validity check; "
            "the decimated mask may disagree strongly with the full-resolution alpha."
        )
    return np.asarray(accepted, dtype=float)


def draw_windows(
    mask_low: np.ndarray,
    scale_xy: tuple[float, float],
    win_px: int,
    n: int,
    seed: int,
    max_tries: int = 20000,
) -> list[tuple[int, int, int, int]]:
    """Non-overlapping square windows lying fully inside the valid mask.

    Returns (x0, y0, w, h) in full-resolution pixels.
    """
    sx, sy = scale_xy
    wc = int(np.ceil(win_px / sx))   # window size in low-res cells (x)
    hc = int(np.ceil(win_px / sy))
    H, W = mask_low.shape
    if wc > W or hc > H:
        raise ValueError("Window larger than the mosaic.")
    # summed-area table -> fully valid windows
    sat = np.zeros((H + 1, W + 1), dtype=np.int64)
    sat[1:, 1:] = np.cumsum(np.cumsum(mask_low.astype(np.int64), axis=0), axis=1)
    full = (sat[hc:, wc:] - sat[:-hc, wc:] - sat[hc:, :-wc] + sat[:-hc, :-wc]) == wc * hc
    cand = np.argwhere(full)  # (row, col) of window origins in low-res cells
    if len(cand) == 0:
        raise RuntimeError("No fully valid window of the requested size exists.")
    rng = np.random.default_rng(seed)
    chosen: list[tuple[int, int, int, int]] = []
    order = rng.permutation(len(cand))
    for k in order[:max_tries]:
        r, c = cand[k]
        x0, y0 = int(c * sx), int(r * sy)
        w, h = int(win_px), int(win_px)
        if all(x0 + w <= cx or cx + cw <= x0 or y0 + h <= cy or cy + ch <= y0
               for cx, cy, cw, ch in chosen):
            chosen.append((x0, y0, w, h))
            if len(chosen) == n:
                break
    if len(chosen) < n:
        raise RuntimeError(f"Placed only {len(chosen)}/{n} non-overlapping windows.")
    return chosen


def read_mask_rasterio(path: str, max_dim: int = 4000, alpha_band: int = 4):
    """Decimated validity mask from a raster's alpha band (needs rasterio)."""
    import rasterio  # lazy: only the I/O wrapper needs it

    with rasterio.open(path) as src:
        scale = max(src.width, src.height) / float(max_dim)
        scale = max(scale, 1.0)
        out_w = int(np.ceil(src.width / scale))
        out_h = int(np.ceil(src.height / scale))
        if src.count >= alpha_band:
            a = src.read(alpha_band, out_shape=(out_h, out_w))
            mask = a > 0
        else:
            raise ValueError(f"{path} has no alpha band {alpha_band}; pass a mask instead.")
        scale_xy = (src.width / out_w, src.height / out_h)
        return mask, scale_xy, src.transform, (src.width, src.height)


def pixel_to_world(transform, x_px: float, y_px: float) -> tuple[float, float]:
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    return a * x_px + b * y_px + c, d * x_px + e * y_px + f


class AffineXY:
    """Minimal north-up affine (a, e = pixel size, c, f = origin)."""

    def __init__(self, x0: float, y0: float, px_m: float):
        self.a, self.b, self.c = px_m, 0.0, x0
        self.d, self.e, self.f = 0.0, -px_m, y0


def read_mask_png(path: str):
    """Validity mask from a PNG (non-zero = valid), e.g. exported from an overview."""
    from PIL import Image

    return np.array(Image.open(path)) > 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raster", required=True, help="Raster name (or path when no --mask-png).")
    ap.add_argument("--mask-png", help="Use this validity mask instead of reading the raster.")
    ap.add_argument("--full-size", type=int, nargs=2, metavar=("W", "H"),
                    help="Full-resolution size in px (required with --mask-png).")
    ap.add_argument("--origin", type=float, nargs=2, metavar=("X0", "Y0"),
                    help="World coordinate of the top-left corner (with --mask-png).")
    ap.add_argument("--n-points", type=int, default=400)
    ap.add_argument("--seed", type=int, required=True, help="Record this in your methods.")
    ap.add_argument("--edge-buffer-mm", type=float, default=20.0)
    ap.add_argument("--mm-per-px", type=float, required=True)
    ap.add_argument("--n-windows", type=int, default=5)
    ap.add_argument("--window-cm", type=float, default=50.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.mask_png:
        if not (args.full_size and args.origin):
            ap.error("--mask-png needs --full-size and --origin")
        mask = read_mask_png(args.mask_png)
        W, H = args.full_size
        sx, sy = W / mask.shape[1], H / mask.shape[0]
        transform = AffineXY(args.origin[0], args.origin[1], args.mm_per_px / 1000.0)
    else:
        mask, (sx, sy), transform, (W, H) = read_mask_rasterio(args.raster)
    eroded = eroded_mask(mask, (sx, sy), args.mm_per_px, args.edge_buffer_mm)
    pts = draw_points(eroded, (sx, sy), args.n_points, args.seed)
    win_px = int(round(args.window_cm * 10.0 / args.mm_per_px))
    wins = draw_windows(eroded, (sx, sy), win_px, args.n_windows, args.seed + 1)

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed + 2)
    order = rng.permutation(len(pts))  # blind, shuffled labelling order
    with open(os.path.join(args.out, "reference_points.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["id", "x_px", "y_px", "x_m", "y_m", "label"])
        for rank, i in enumerate(order, start=1):
            x, y = pts[i]
            xm, ym = pixel_to_world(transform, x, y)
            wr.writerow([f"P{rank:04d}", f"{x:.2f}", f"{y:.2f}", f"{xm:.6f}", f"{ym:.6f}", ""])
    with open(os.path.join(args.out, "gold_windows.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["id", "x0_px", "y0_px", "w_px", "h_px"])
        for k, (x0, y0, w, h) in enumerate(wins, start=1):
            wr.writerow([f"W{k:02d}", x0, y0, w, h])
    cell_area_m2 = (sx * sy) * (args.mm_per_px / 1000.0) ** 2
    manifest = {
        "raster": os.path.basename(args.raster), "seed": args.seed,
        "n_points": args.n_points, "n_windows": args.n_windows,
        "window_cm": args.window_cm, "mm_per_px": args.mm_per_px,
        "edge_buffer_mm": args.edge_buffer_mm, "mask_scale_xy": [sx, sy],
        "valid_area_m2_after_buffer": float(eroded.sum() * cell_area_m2),
        "design": "simple random over eroded valid area; excluded edge band has inclusion probability 0",
    }
    with open(os.path.join(args.out, "sampling_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
