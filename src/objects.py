"""Object outlines for discrete organisms: candidate masks (SAM2) + class vote + polygons.

A patch-resolution class map (5 cm) cannot give outlines, and SAM2 masks carry no class.  Each
mask is therefore classified by the area-weighted mean of the patch probabilities it covers:
outline from SAM2, identity from the classifier that was validated against the reference sample.

Per tile: generator masks -> minimum mapping unit -> class vote (object classes above
``min_conf``) -> greedy non-maximum suppression -> keep only masks whose centroid lies in the
tile core (so objects on tile borders are reported once) -> outer boundary, simplified.

Matrix classes (turf, crusts, sand) are not outlined; they are reported as cover.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy import ndimage

# clockwise neighbour offsets (dy, dx) in image coordinates, starting from west
_DIRS = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]


def trace_outer_boundary(mask: np.ndarray) -> np.ndarray:
    """Moore-neighbour trace of the outer boundary of the first component in raster order.

    Returns (n, 2) pixel centres (x, y), clockwise, closed implicitly.  Holes are ignored.
    """
    m = np.pad(np.asarray(mask, dtype=bool), 1)
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return np.zeros((0, 2))
    start = (int(ys[0]), int(xs[0]))                # west and north neighbours are background
    cur, back = start, 0
    pts = [cur]
    seen = {(cur, back)}
    while True:
        for i in range(1, 9):
            k = (back + i) % 8
            y, x = cur[0] + _DIRS[k][0], cur[1] + _DIRS[k][1]
            if m[y, x]:
                p = _DIRS[(k - 1) % 8]
                bg = (cur[0] + p[0], cur[1] + p[1])
                back = _DIRS.index((bg[0] - y, bg[1] - x))
                cur = (y, x)
                break
        else:                                       # isolated pixel
            break
        if (cur, back) in seen:
            break
        seen.add((cur, back))
        pts.append(cur)
    while len(pts) > 1 and pts[-1] == pts[0]:
        pts.pop()
    arr = np.array(pts, dtype=float)
    return np.column_stack([arr[:, 1] - 1, arr[:, 0] - 1])


def _rdp(pts: np.ndarray, tol: float) -> np.ndarray:
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    norm = float(np.hypot(*ab))
    if norm == 0:
        dist = np.hypot(*(pts - a).T)
    else:
        dist = np.abs(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])) / norm
    k = int(np.argmax(dist))
    if dist[k] > tol:
        return np.vstack([_rdp(pts[:k + 1], tol)[:-1], _rdp(pts[k:], tol)])
    return np.vstack([a, b])


def simplify(ring: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker on a closed ring (n, 2)."""
    if len(ring) <= 4 or tol <= 0:
        return ring
    j = int(np.argmax(np.hypot(*(ring - ring[0]).T)))        # split at the point farthest from vertex 0
    first = _rdp(ring[:j + 1], tol)
    second = _rdp(np.vstack([ring[j:], ring[:1]]), tol)
    return np.vstack([first[:-1], second[:-1]])


def polygon_area(ring: np.ndarray) -> float:
    if len(ring) < 3:
        return 0.0
    x, y = ring[:, 0], ring[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def vote_class(mask: np.ndarray, x_off: int, y_off: int, grid, proba: np.ndarray, classes,
               max_samples: int = 2000, seed: int = 0, min_inside: float = 0.5):
    """(class, confidence, mean probabilities) from the patches under ``mask``; None if mostly off-grid."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    if len(ys) > max_samples:
        pick = np.random.default_rng(seed).choice(len(ys), max_samples, replace=False)
        ys, xs = ys[pick], xs[pick]
    rows = grid.index_of(np.column_stack([xs + x_off + 0.5, ys + y_off + 0.5]))
    ok = rows >= 0
    if ok.mean() < min_inside:
        return None
    p = proba[rows[ok]].mean(0)
    j = int(np.argmax(p))
    return str(classes[j]), float(p[j]), p


def delineate_tile(img_rgb: np.ndarray, x_off: int, y_off: int, core: tuple, generator: Callable,
                   grid, proba: np.ndarray, classes, object_classes, mmu_px: int,
                   min_conf: float = 0.5, max_overlap: float = 0.5, simplify_tol: float = 1.5) -> list:
    """Objects in one tile window.  ``core`` = (x0, y0, x1, y1) in full-resolution px."""
    cands = []
    for m in generator(img_rgb):
        seg = np.asarray(m["segmentation"], dtype=bool)
        if int(seg.sum()) < mmu_px:
            continue
        v = vote_class(seg, x_off, y_off, grid, proba, classes)
        if v is None or v[0] not in object_classes or v[1] < min_conf:
            continue
        score = v[1] * float(m.get("stability_score", 1.0)) * float(m.get("predicted_iou", 1.0))
        cands.append((score, seg, v))
    cands.sort(key=lambda t: -t[0])

    occupied = np.zeros(img_rgb.shape[:2], dtype=bool)
    cx0, cy0, cx1, cy1 = core
    out = []
    for _, seg, (cls, conf, _) in cands:
        if (seg & occupied).sum() > max_overlap * int(seg.sum()):
            continue
        occupied |= seg                              # also blocks duplicates of objects owned by a neighbour tile
        ys, xs = np.nonzero(seg)
        cx, cy = xs.mean() + x_off, ys.mean() + y_off
        if not (cx0 <= cx < cx1 and cy0 <= cy < cy1):
            continue
        lab, n = ndimage.label(seg)
        if n > 1:
            seg = lab == (1 + int(np.argmax(ndimage.sum(seg, lab, range(1, n + 1)))))
        ring = simplify(trace_outer_boundary(seg), simplify_tol) + np.array([x_off, y_off], dtype=float)
        out.append({"class": cls, "confidence": conf, "n_pixels": int(seg.sum()),
                    "ring_px": ring, "centroid_px": (float(cx), float(cy))})
    return out


def delineate_mosaic(reader, grid, proba, classes, object_classes, generator, tile_px: int = 1024,
                     margin_px: int = 128, normaliser: Optional[Callable] = None, mmu_px: int = 400,
                     min_conf: float = 0.5, **kw) -> list:
    """Run ``delineate_tile`` over tiles whose core holds at least one patch predicted as an object class."""
    classes = list(classes)
    object_classes = set(object_classes)
    obj_idx = [i for i, c in enumerate(classes) if c in object_classes]
    pred_obj = np.isin(proba.argmax(1), obj_idx)
    ctr = grid.centers
    out: list = []
    for ty in range(0, grid.height, tile_px):
        for tx in range(0, grid.width, tile_px):
            x1, y1 = min(tx + tile_px, grid.width), min(ty + tile_px, grid.height)
            inside = (ctr[:, 0] >= tx) & (ctr[:, 0] < x1) & (ctr[:, 1] >= ty) & (ctr[:, 1] < y1)
            if not (pred_obj & inside).any():
                continue
            wx0, wy0 = max(0, tx - margin_px), max(0, ty - margin_px)
            wx1, wy1 = min(grid.width, x1 + margin_px), min(grid.height, y1 + margin_px)
            win = reader.read(wx0, wy0, wx1 - wx0, wy1 - wy0)
            rgb = normaliser(win) if normaliser is not None else np.asarray(win[..., :3])
            out += delineate_tile(np.ascontiguousarray(rgb, dtype=np.uint8), wx0, wy0, (tx, ty, x1, y1),
                                  generator, grid, proba, classes, object_classes, mmu_px, min_conf, **kw)
    return out


def to_geojson(objects: list, px_size_m: float, transform=None) -> dict:
    """FeatureCollection of polygons.  ``transform`` is north-up affine-like (a, c, e, f); without it, pixels."""
    feats = []
    for k, o in enumerate(objects, start=1):
        ring = o["ring_px"]
        if len(ring) < 3:
            continue
        if transform is not None:
            ring = np.column_stack([transform.a * ring[:, 0] + transform.c,
                                    transform.e * ring[:, 1] + transform.f])
        feats.append({"type": "Feature",
                      "geometry": {"type": "Polygon", "coordinates": [ring.tolist() + [ring[0].tolist()]]},
                      "properties": {"id": k, "class": o["class"], "confidence": round(o["confidence"], 4),
                                     "area_m2": round(o["n_pixels"] * px_size_m ** 2, 6)}})
    return {"type": "FeatureCollection", "features": feats}


def load_sam2_generator(config: str, checkpoint: str, device: str = "cpu", **kw) -> Callable:
    """SAM2 automatic mask generator as ``img -> list of dicts`` (needs torch + the ``sam2`` package).

    Not exercised by the tests: it needs a checkpoint and a GPU/CPU torch install.
    """
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # lazy
    from sam2.build_sam import build_sam2

    return SAM2AutomaticMaskGenerator(build_sam2(config, checkpoint, device=device), **kw).generate
