import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import detection  # noqa: E402
import tiling  # noqa: E402
from test_objects import PATCH, SIZE, make_scene, rect  # noqa: E402


def coral_detector(img):
    """Stand-in for a YOLO-seg / Grounding-DINO model: red blobs are 'coral', score 0.9."""
    lab, n = ndimage.label(img[..., 0] > 150)
    return [{"segmentation": lab == k, "class": "coral", "score": 0.9} for k in range(1, n + 1)]


def run_tile(boxes, detector=coral_detector, core=(0, 0, SIZE, SIZE), **kw):
    img, grid, _ = make_scene(boxes)
    args = dict(mmu_px=400, min_score=0.25)
    args.update(kw)
    return detection.detect_tile(img, 0, 0, core, detector, grid, **args), grid


def test_coverage_by_patch_gives_covered_fraction_per_patch():
    _, grid, _ = make_scene([])
    rows, frac = detection.coverage_by_patch(rect(SIZE, SIZE, 0, 0, 40, 30), 0, 0, grid, stride=2)
    got = dict(zip(rows.tolist(), np.round(frac, 2).tolist()))
    full = grid.index_of(np.array([[10.0, 10.0], [30.0, 10.0]]))
    half = grid.index_of(np.array([[10.0, 30.0], [30.0, 30.0]]))
    assert all(got[r] == 1.0 for r in full) and all(got[r] == 0.5 for r in half) and len(got) == 4


def test_coverage_by_patch_uses_global_coordinates():
    _, grid, _ = make_scene([])
    rows, _ = detection.coverage_by_patch(rect(60, 60, 0, 0, 20, 20), 100, 100, grid, stride=1)
    assert rows.tolist() == [int(grid.index_of(np.array([[110.0, 110.0]]))[0])]


def test_tile_takes_class_and_score_from_the_detector_and_attaches_coverage():
    out, grid = run_tile([(60, 60, 120, 120)])
    assert len(out) == 1
    o = out[0]
    assert o["class"] == "coral" and o["confidence"] == 0.9 and o["n_pixels"] == 60 * 60
    rows, frac = o["coverage"]
    assert len(rows) == 9 and np.allclose(frac, 1.0) and "mask" not in o


def test_tile_filters_by_score_class_and_size():
    assert run_tile([(60, 60, 120, 120)], min_score=0.95)[0] == []
    assert run_tile([(60, 60, 120, 120)], classes_keep={"sponge"})[0] == []
    assert run_tile([(60, 60, 80, 80)], mmu_px=401)[0] == []


def test_tile_suppresses_duplicate_detections_of_one_organism():
    def twice(img):
        m = rect(*img.shape[:2], 60, 60, 120, 120)
        return [{"segmentation": m, "class": "coral", "score": 0.9},
                {"segmentation": rect(*img.shape[:2], 64, 64, 116, 116), "class": "coral", "score": 0.6}]

    assert len(run_tile([(60, 60, 120, 120)], detector=twice)[0]) == 1


class CountingDetector:
    def __init__(self):
        self.calls = 0

    def __call__(self, img):
        self.calls += 1
        return coral_detector(img)


def test_mosaic_reports_border_object_once_and_skips_tiles_without_valid_patches():
    img, _, _ = make_scene([(100, 60, 140, 100)])
    low = np.ones((SIZE // 4, SIZE // 4), bool)
    low[:, SIZE // 8:] = False                                              # right half of the mosaic is empty
    grid = tiling.make_grid(low, (4.0, 4.0), SIZE, SIZE, PATCH, min_valid_frac=0.5)
    det = CountingDetector()
    out = detection.detect_mosaic(tiling.ArrayReader(img), grid, det, tile_px=120, margin_px=40, mmu_px=400)
    assert len(out) == 1 and det.calls == 2                                  # only the two left tiles hold patches


def test_patch_labels_assign_target_class_or_other():
    out, grid = run_tile([(60, 60, 120, 120)])
    labels, conf = detection.patch_labels(grid, out, other="other", min_cover=0.3)
    inside = grid.index_of(np.array([[70.0, 70.0]]))[0]
    outside = grid.index_of(np.array([[10.0, 10.0]]))[0]
    assert labels[inside] == "coral" and labels[outside] == "other"
    assert (labels == "coral").sum() == 9 and conf[inside] == 1.0 and conf[outside] == 1.0


def test_patch_labels_need_min_cover_and_larger_cover_wins():
    _, grid, _ = make_scene([])
    r = int(grid.index_of(np.array([[10.0, 10.0]]))[0])
    insts = [{"class": "a", "coverage": (np.array([r]), np.array([0.2]))},
             {"class": "b", "coverage": (np.array([r]), np.array([0.5]))}]
    labels, _ = detection.patch_labels(grid, insts, min_cover=0.3)
    assert labels[r] == "b"
    labels, _ = detection.patch_labels(grid, insts[:1], min_cover=0.3)
    assert labels[r] == "other"


def test_polygons_to_instances_rasterises_and_names_classes():
    polys = [np.array([[10.0, 10.0], [50.0, 10.0], [50.0, 40.0], [10.0, 40.0]])]
    out = detection.polygons_to_instances(polys, class_ids=[1], names={0: "a", 1: "sponge"}, scores=[0.7],
                                          shape=(60, 80))
    assert out[0]["class"] == "sponge" and out[0]["score"] == 0.7
    assert abs(int(out[0]["segmentation"].sum()) - 40 * 30) < 140            # polygon edge pixels only


def test_patch_labels_add_up_coverage_of_several_instances_of_one_class():
    _, grid, _ = make_scene([])
    r = int(grid.index_of(np.array([[10.0, 10.0]]))[0])
    insts = [{"class": "a", "coverage": (np.array([r]), np.array([0.2]))},
             {"class": "a", "coverage": (np.array([r]), np.array([0.2]))}]
    labels, conf = detection.patch_labels(grid, insts, min_cover=0.3)
    assert labels[r] == "a" and abs(conf[r] - 0.4) < 1e-9
