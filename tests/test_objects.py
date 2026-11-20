import os
import sys
from types import SimpleNamespace

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import objects  # noqa: E402
import tiling  # noqa: E402

CLASSES = ["turf", "coral"]
OBJECT_CLASSES = {"coral"}
SIZE, PATCH = 240, 20


def rect(h, w, x0, y0, x1, y1):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def make_scene(boxes):
    """RGB image, grid and probabilities; each box (x0, y0, x1, y1) is a red 'coral' block on grey turf."""
    img = np.full((SIZE, SIZE, 3), 90, dtype=np.uint8)
    for x0, y0, x1, y1 in boxes:
        img[y0:y1, x0:x1] = (220, 40, 40)
    grid = tiling.make_grid(np.ones((SIZE // 4, SIZE // 4), bool), (4.0, 4.0), SIZE, SIZE, PATCH)
    c = grid.centers
    proba = np.tile([0.9, 0.1], (grid.n, 1))
    for x0, y0, x1, y1 in boxes:
        inside = (c[:, 0] >= x0) & (c[:, 0] < x1) & (c[:, 1] >= y0) & (c[:, 1] < y1)
        proba[inside] = [0.1, 0.9]
    return img, grid, proba


def red_generator(img):
    """Stand-in for SAM2: one mask per connected red component, in the coordinates of ``img``."""
    lab, n = ndimage.label(img[..., 0] > 150)
    return [{"segmentation": lab == k, "stability_score": 0.9, "predicted_iou": 0.9} for k in range(1, n + 1)]


def run_tile(boxes, generator=red_generator, x_off=0, y_off=0, size=SIZE, core=(0, 0, SIZE, SIZE), **kw):
    img, grid, proba = make_scene(boxes)
    win = np.ascontiguousarray(img[y_off:y_off + size, x_off:x_off + size])
    args = dict(mmu_px=400)
    args.update(kw)
    return objects.delineate_tile(win, x_off, y_off, core, generator, grid, proba, CLASSES, OBJECT_CLASSES, **args)


# --- geometry -------------------------------------------------------------

def test_trace_outer_boundary_encloses_rectangle():
    ring = objects.trace_outer_boundary(rect(20, 30, 5, 4, 15, 10))        # 10 wide, 6 high
    assert objects.polygon_area(ring) == (10 - 1) * (6 - 1)
    assert ring[:, 0].min() == 5 and ring[:, 0].max() == 14
    assert ring[:, 1].min() == 4 and ring[:, 1].max() == 9


def test_trace_outer_boundary_ignores_holes():
    m = rect(30, 30, 5, 5, 25, 25)
    m[12:18, 12:18] = False
    ring = objects.trace_outer_boundary(m)
    assert objects.polygon_area(ring) == 19 * 19


def test_trace_outer_boundary_single_pixel_and_empty():
    assert len(objects.trace_outer_boundary(rect(5, 5, 2, 2, 3, 3))) == 1
    assert len(objects.trace_outer_boundary(np.zeros((5, 5), bool))) == 0


def test_simplify_reduces_rectangle_to_corners_and_keeps_area():
    ring = objects.trace_outer_boundary(rect(40, 60, 5, 5, 45, 25))
    s = objects.simplify(ring, 1.0)
    assert len(ring) > 8 and len(s) == 4
    assert objects.polygon_area(s) == objects.polygon_area(ring)


# --- classification by vote -----------------------------------------------

def test_vote_class_picks_patch_class_under_mask():
    _, grid, proba = make_scene([(60, 60, 120, 120)])
    cls, conf, _ = objects.vote_class(rect(SIZE, SIZE, 70, 70, 110, 110), 0, 0, grid, proba, CLASSES)
    assert cls == "coral" and conf > 0.85


def test_vote_class_respects_window_offset():
    _, grid, proba = make_scene([(140, 140, 200, 200)])
    local = rect(100, 100, 50, 50, 90, 90)                                   # global 150..190
    cls, _, _ = objects.vote_class(local, 100, 100, grid, proba, CLASSES)
    assert cls == "coral"


def test_vote_class_none_when_mask_is_off_grid():
    _, grid, proba = make_scene([])
    big = np.zeros((SIZE, SIZE), bool)
    big[:10, :10] = True
    assert objects.vote_class(big, SIZE + 50, SIZE + 50, grid, proba, CLASSES) is None


# --- one tile ---------------------------------------------------------------

def test_tile_returns_object_with_class_and_global_ring():
    out = run_tile([(60, 60, 112, 112)])
    assert len(out) == 1
    o = out[0]
    assert o["class"] == "coral" and o["n_pixels"] == 52 * 52
    assert o["ring_px"][:, 0].min() == 60 and o["ring_px"][:, 1].max() == 111


def test_tile_ring_uses_global_coordinates_with_offset():
    out = run_tile([(140, 140, 200, 200)], x_off=100, y_off=100, size=140, core=(100, 100, 240, 240))
    assert len(out) == 1
    assert out[0]["ring_px"][:, 0].min() == 140 and out[0]["ring_px"][:, 1].min() == 140


def test_tile_does_not_outline_matrix_class():
    img, grid, proba = make_scene([])

    def grey_generator(im):
        return [{"segmentation": rect(*im.shape[:2], 60, 60, 120, 120)}]

    out = objects.delineate_tile(img, 0, 0, (0, 0, SIZE, SIZE), grey_generator, grid, proba, CLASSES,
                                 OBJECT_CLASSES, mmu_px=400)
    assert out == []


def test_tile_drops_masks_below_mapping_unit():
    assert run_tile([(60, 60, 80, 80)], mmu_px=401) == []                    # 20x20 = 400 px
    assert len(run_tile([(60, 60, 80, 80)], mmu_px=400)) == 1


def test_tile_applies_confidence_threshold():
    assert run_tile([(60, 60, 112, 112)], min_conf=0.95) == []


def test_tile_suppresses_nested_duplicate_masks():
    def two(im):
        a = {"segmentation": rect(*im.shape[:2], 60, 60, 112, 112), "stability_score": 0.9}
        b = {"segmentation": rect(*im.shape[:2], 66, 66, 106, 106), "stability_score": 0.8}
        return [b, a]

    out = run_tile([(60, 60, 112, 112)], generator=two)
    assert len(out) == 1 and out[0]["n_pixels"] == 52 * 52


def test_tile_keeps_only_objects_whose_centroid_is_in_core():
    assert run_tile([(60, 60, 112, 112)], core=(120, 0, SIZE, SIZE)) == []


# --- whole mosaic -----------------------------------------------------------

class CountingReader(tiling.ArrayReader):
    def __init__(self, arr):
        super().__init__(arr)
        self.reads = 0

    def read(self, x0, y0, w, h):
        self.reads += 1
        return super().read(x0, y0, w, h)


def test_mosaic_reports_object_on_tile_border_exactly_once_and_skips_empty_tiles():
    img, grid, proba = make_scene([(100, 60, 140, 100)])                     # straddles x = 120
    reader = CountingReader(img)
    out = objects.delineate_mosaic(reader, grid, proba, CLASSES, OBJECT_CLASSES, red_generator,
                                   tile_px=120, margin_px=40, mmu_px=400)
    assert len(out) == 1 and out[0]["n_pixels"] == 40 * 40
    assert reader.reads == 2                                                 # bottom tiles hold no object patches


# --- export -------------------------------------------------------------------

def test_geojson_polygon_is_closed_and_carries_area():
    ring = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    obj = {"class": "coral", "confidence": 0.8123456, "n_pixels": 10000, "ring_px": ring, "centroid_px": (50, 50)}
    fc = objects.to_geojson([obj], px_size_m=0.001)
    f = fc["features"][0]
    coords = f["geometry"]["coordinates"][0]
    assert coords[0] == coords[-1] and len(coords) == 5
    assert f["properties"]["class"] == "coral" and abs(f["properties"]["area_m2"] - 0.01) < 1e-9


def test_geojson_applies_affine_like_transform():
    ring = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    obj = {"class": "coral", "confidence": 0.9, "n_pixels": 50, "ring_px": ring, "centroid_px": (5, 5)}
    t = SimpleNamespace(a=0.5, c=100.0, e=-0.5, f=200.0)
    coords = objects.to_geojson([obj], 0.5, transform=t)["features"][0]["geometry"]["coordinates"][0]
    assert coords[1] == [105.0, 200.0] and coords[2] == [105.0, 195.0]
