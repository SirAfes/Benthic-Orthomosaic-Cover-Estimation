"""Instance detection of target organisms (YOLO-seg, Grounding DINO + SAM2, ...) feeding the same estimators.

A detector is any callable ``img_rgb -> [{"segmentation": bool HxW, "class": str, "score": float}]``.
Everything that is not a detected target is ``other``, so the map has the strata (targets + other)
that ``estimate`` needs; matrix classes (sand, rock, seaweed) need no model of their own.  Cover is
still estimated from the probability reference sample, never from the detections themselves.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from objects import finalize_candidates


def coverage_by_patch(mask: np.ndarray, x_off: int, y_off: int, grid, stride: int = 4):
    """(rows, fractions): share of each kept patch covered by ``mask`` (sampled every ``stride`` px)."""
    sub = np.asarray(mask, dtype=bool)[::stride, ::stride]
    ys, xs = np.nonzero(sub)
    if len(ys) == 0:
        return np.zeros(0, dtype=int), np.zeros(0)
    rows = grid.index_of(np.column_stack([xs * stride + x_off + 0.5, ys * stride + y_off + 0.5]))
    rows = rows[rows >= 0]
    counts = np.bincount(rows, minlength=grid.n)
    per_patch = float(int(np.ceil(grid.patch_px / stride)) ** 2)
    hit = np.nonzero(counts)[0]
    return hit, np.minimum(counts[hit] / per_patch, 1.0)


def detect_tile(img_rgb: np.ndarray, x_off: int, y_off: int, core: tuple, detector: Callable, grid,
                mmu_px: int, min_score: float = 0.25, classes_keep: Optional[set] = None,
                max_overlap: float = 0.5, simplify_tol: float = 1.5, stride: int = 4) -> list:
    """Detections in one tile window, each with outline, class, score and per-patch ``coverage``."""
    cands = []
    for d in detector(img_rgb):
        seg = np.asarray(d["segmentation"], dtype=bool)
        if int(seg.sum()) < mmu_px or float(d["score"]) < min_score:
            continue
        if classes_keep is not None and d["class"] not in classes_keep:
            continue
        cands.append((float(d["score"]), seg, str(d["class"]), float(d["score"])))
    out = finalize_candidates(cands, img_rgb.shape[:2], x_off, y_off, core, max_overlap, simplify_tol,
                              keep_mask=True)
    for rec in out:
        rec["coverage"] = coverage_by_patch(rec.pop("mask"), x_off, y_off, grid, stride)
    return out


def detect_mosaic(reader, grid, detector: Callable, tile_px: int = 1024, margin_px: int = 128,
                  normaliser: Optional[Callable] = None, mmu_px: int = 400, min_score: float = 0.25,
                  classes_keep: Optional[set] = None, progress: Optional[Callable] = None, **kw) -> list:
    """Run ``detect_tile`` over every tile that holds at least one valid patch."""
    ctr = grid.centers
    tiles = []
    for ty in range(0, grid.height, tile_px):
        for tx in range(0, grid.width, tile_px):
            x1, y1 = min(tx + tile_px, grid.width), min(ty + tile_px, grid.height)
            inside = (ctr[:, 0] >= tx) & (ctr[:, 0] < x1) & (ctr[:, 1] >= ty) & (ctr[:, 1] < y1)
            if inside.any():
                tiles.append((tx, ty, x1, y1))
    out: list = []
    for i, (tx, ty, x1, y1) in enumerate(tiles):
        wx0, wy0 = max(0, tx - margin_px), max(0, ty - margin_px)
        wx1, wy1 = min(grid.width, x1 + margin_px), min(grid.height, y1 + margin_px)
        win = reader.read(wx0, wy0, wx1 - wx0, wy1 - wy0)
        rgb = normaliser(win) if normaliser is not None else np.asarray(win[..., :3])
        out += detect_tile(np.ascontiguousarray(rgb, dtype=np.uint8), wx0, wy0, (tx, ty, x1, y1), detector,
                           grid, mmu_px, min_score, classes_keep, **kw)
        if progress:
            progress(i + 1, len(tiles))
    return out


def patch_labels(grid, instances: list, other: str = "other", min_cover: float = 0.3):
    """Per-patch label and confidence from instance coverage.

    A patch gets the class with the largest covered share if that share is at least ``min_cover``,
    otherwise ``other``.  Confidence is the covered share (``other``: the uncovered share).
    """
    classes = sorted({i["class"] for i in instances})
    cover = np.zeros((len(classes), grid.n))
    for inst in instances:
        rows, frac = inst["coverage"]
        cover[classes.index(inst["class"]), rows] += frac
    cover = np.minimum(cover, 1.0)
    labels = np.full(grid.n, other, dtype=object)
    conf = np.ones(grid.n)
    if classes:
        best = cover.argmax(0)
        top = cover[best, np.arange(grid.n)]
        take = top >= min_cover
        labels[take] = np.array(classes, dtype=object)[best[take]]
        conf = np.where(take, top, 1.0 - np.minimum(cover.sum(0), 1.0))
    return labels, conf


def polygons_to_instances(polys, class_ids, names, scores, shape) -> list:
    """Polygon outlines (x, y) to detector instances with boolean masks of size ``shape`` = (h, w)."""
    from PIL import Image, ImageDraw

    out = []
    for poly, cid, score in zip(polys, class_ids, scores):
        pts = np.asarray(poly, dtype=float).reshape(-1, 2)
        if len(pts) < 3:
            continue
        im = Image.new("L", (shape[1], shape[0]), 0)
        ImageDraw.Draw(im).polygon([tuple(p) for p in pts], fill=1)
        out.append({"segmentation": np.array(im, dtype=bool), "class": str(names[int(cid)]),
                    "score": float(score)})
    return out
