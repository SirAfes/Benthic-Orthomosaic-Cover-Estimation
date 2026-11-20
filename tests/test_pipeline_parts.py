import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import classifier as clf  # noqa: E402
import colour  # noqa: E402
import patch_features as pf  # noqa: E402
import tiling  # noqa: E402


def make_mosaic(h=240, w=360, seed=0):
    """Synthetic RGBA mosaic: left third red-ish, middle green-ish, right blue-ish, with a hole."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 4), dtype=np.uint8)
    base = np.array([[200, 60, 60], [60, 200, 60], [60, 60, 200]], dtype=float)
    for i in range(3):
        sl = slice(i * w // 3, (i + 1) * w // 3)
        img[:, sl, :3] = np.clip(base[i] + rng.normal(0, 12, (h, w // 3, 3)), 0, 255)
    img[..., 3] = 255
    img[100:140, 150:200, 3] = 0   # hole
    return img


def low_mask(img, f=4):
    a = img[..., 3] > 0
    return a[::f, ::f]


def test_patch_px_for_same_physical_size():
    assert tiling.patch_px_for(5, 0.412284) == 121
    assert tiling.patch_px_for(5, 0.664) == 75


def test_grid_drops_patches_on_hole_and_indexes_points():
    img = make_mosaic()
    g = tiling.make_grid(low_mask(img), (4.0, 4.0), img.shape[1], img.shape[0], patch_px=20, min_valid_frac=0.98)
    assert g.n_rows == 12 and g.n_cols == 18
    assert g.n < g.n_rows * g.n_cols                       # hole patches removed
    # a point inside the hole has no patch; a point elsewhere maps to its own patch
    idx = g.index_of(np.array([[170.0, 120.0], [5.0, 5.0], [10_000.0, 5.0]]))
    assert idx[0] == -1 and idx[2] == -1 and idx[1] >= 0
    r = idx[1]
    assert (g.x0[r], g.y0[r]) == (0, 0)
    assert np.allclose(g.centers[r], [10.0, 10.0])
    assert (g.valid_frac >= 0.98).all()


def test_colour_normaliser_balances_cast_and_roundtrips():
    rng = np.random.default_rng(1)
    base = rng.uniform(0.2, 0.8, (5000, 3))
    cast = base * np.array([1.0, 0.7, 0.35]) * 255          # strong blue/green loss
    norm = colour.ColourNormaliser.fit(cast)
    out = norm(cast.reshape(-1, 1, 3)).reshape(-1, 3).astype(float)
    means = out.mean(0)
    assert means.max() - means.min() < 6
    n2 = colour.ColourNormaliser.from_json(norm.to_json())
    assert np.array_equal(n2(cast.reshape(-1, 1, 3)), norm(cast.reshape(-1, 1, 3)))


def test_extract_grid_features_order_and_resume(tmp_path):
    img = make_mosaic()
    reader = tiling.ArrayReader(img)
    g = tiling.make_grid(low_mask(img), (4.0, 4.0), img.shape[1], img.shape[0], 20)
    calls = {"n": 0}

    def backbone(x):
        calls["n"] += 1
        return x.reshape(-1, 3).mean(0)

    direct = pf.extract_grid_features(g, reader, backbone, chunk_size=50)
    assert direct.shape == (g.n, 3) and calls["n"] == g.n
    cd = str(tmp_path / "chunks")
    calls["n"] = 0
    a = pf.extract_grid_features(g, reader, backbone, chunk_size=50, chunk_dir=cd)
    assert calls["n"] == g.n and np.allclose(a, direct)
    calls["n"] = 0
    b = pf.extract_grid_features(g, reader, backbone, chunk_size=50, chunk_dir=cd)   # resume: nothing recomputed
    assert calls["n"] == 0 and np.allclose(b, direct)
    # first patch is red-ish
    assert direct[0, 0] > direct[0, 1]


def test_batch_backbone_is_used():
    seen = []

    class B:
        def __call__(self, x):
            raise AssertionError("per-image path should not be used")

        def batch(self, xs):
            seen.append(len(xs))
            return np.zeros((len(xs), 2))

    out = pf.extract_from_images([np.zeros((4, 4, 3), np.uint8)] * 5, B(), batch_size=2)
    assert out.shape == (5, 2) and seen == [2, 2, 1]


def test_classifier_reject_and_blocked_cv():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (120, 4)), rng.normal(4, 1, (120, 4))])
    y = np.array(["a"] * 120 + ["b"] * 120)
    xy = np.column_stack([rng.uniform(0, 1000, 240), rng.uniform(0, 1000, 240)])
    blocks = clf.spatial_block_ids(xy, 250.0)
    assert len(np.unique(blocks)) <= 16
    m = clf.PatchClassifier().fit(X, y)
    assert (m.predict(X) == y).mean() > 0.97
    ambiguous = np.full((1, 4), 2.0)
    assert m.predict(ambiguous, min_conf=0.99)[0] == clf.REJECT
    oof = clf.blocked_cv_predict(X, y, blocks, n_splits=4)
    assert (oof == y).mean() > 0.95
    with pytest.raises(ValueError):
        clf.PatchClassifier().fit(X, np.array(["a"] * 240))
