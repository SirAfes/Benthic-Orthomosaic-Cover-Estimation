import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import active_learning as al  # noqa: E402
import classifier as clf  # noqa: E402
import colour  # noqa: E402
import cover_maps as cm  # noqa: E402
import estimation as est  # noqa: E402
import patch_features as pf  # noqa: E402
import reference_sampling as rs  # noqa: E402
import tiling  # noqa: E402
from features import ColorTextureFeatures  # noqa: E402

NAMES = ["rock", "algae", "sponge"]
COLOURS = np.array([[150, 120, 90], [70, 45, 95], [205, 125, 45]], dtype=float)


def synthetic_survey(h=480, w=720, seed=0):
    """RGBA mosaic with a known per-pixel truth map, texture differences and a hole/margin."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    truth = np.zeros((h, w), dtype=int)
    truth[(yy > 0.55 * h + 35 * np.sin(xx / 70.0))] = 1
    truth[(xx > 0.62 * w) & (yy < 0.5 * h)] = 2
    truth[(xx < 0.2 * w) & (yy < 0.4 * h)] = 2
    noise_sd = np.array([14.0, 30.0, 18.0])[truth][..., None]
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[..., :3] = np.clip(COLOURS[truth] + rng.normal(0, 1, (h, w, 3)) * noise_sd, 0, 255)
    img[..., 3] = 255
    img[200:250, 300:380, 3] = 0           # hole
    img[:, :6, 3] = 0                      # ragged margin
    return img, truth


def test_end_to_end_cover_estimate_on_synthetic_survey():
    img, truth = synthetic_survey()
    h, w = truth.shape
    f = 4
    valid_px = img[..., 3] > 0
    mask_low = valid_px[::f, ::f]
    eroded_low = rs.erode(mask_low, 3)

    # ground truth: class shares over the sampled population (eroded valid area)
    pop_px = np.kron(eroded_low, np.ones((f, f), dtype=bool))[:h, :w]
    true_share = {c: float((truth[pop_px] == i).mean()) for i, c in enumerate(NAMES)}

    # 1. reference sample FIRST, labelled from truth (stands in for the expert)
    pts = rs.draw_points(eroded_low, (float(f), float(f)), 400, seed=11)
    y_ref = np.array(NAMES)[truth[pts[:, 1].astype(int), pts[:, 0].astype(int)]]

    # 2. grid, colour normalisation, features (baseline backbone)
    reader = tiling.ArrayReader(img)
    grid = tiling.make_grid(mask_low, (float(f), float(f)), w, h, patch_px=16)
    norm = colour.ColourNormaliser.fit(colour.sample_valid_pixels(reader, grid, seed=1))
    X = pf.extract_grid_features(grid, reader, ColorTextureFeatures(), normaliser=norm)
    assert X.shape[0] == grid.n

    # 3. training patches: eligible = away from reference points, labels from truth at patch centre
    ctr = grid.centers
    eligible = al.eligible_mask(ctr, pts, exclude_radius_px=16.0)
    rng = np.random.default_rng(3)
    train = rng.choice(np.flatnonzero(eligible), size=250, replace=False)
    yt = np.array(NAMES)[truth[ctr[train, 1].astype(int), ctr[train, 0].astype(int)]]
    model = clf.PatchClassifier().fit(X[train], yt)
    proba = model.predict_proba(X)
    map_labels = clf.labels_from_proba(proba, model.classes_)

    # 4. population = patches whose centre lies in the eroded valid area
    cx = (ctr[:, 0] / f).astype(int).clip(0, eroded_low.shape[1] - 1)
    cy = (ctr[:, 1] / f).astype(int).clip(0, eroded_low.shape[0] - 1)
    in_pop = eroded_low[cy, cx]
    # reference point -> prediction of the grid patch that contains it
    idx = grid.index_of(pts)
    usable = idx >= 0
    assert usable.mean() > 0.97
    yh_ref = map_labels[idx[usable]]
    res = est.post_stratified_estimate(y_ref[usable], yh_ref, map_labels[in_pop], NAMES)
    direct = est.direct_estimate(y_ref[usable], NAMES)

    for c in NAMES:
        e = res["estimates"][c]
        assert abs(e["p"] - true_share[c]) < 3.5 * max(e["se"], 1e-3), (c, e, true_share[c])
        assert res["estimates"][c]["se"] <= direct[c]["se"] * 1.05
    # the classifier is genuinely informative on held-out patches
    held = np.setdiff1d(np.flatnonzero(eligible), train)
    yh_true = np.array(NAMES)[truth[ctr[held, 1].astype(int), ctr[held, 0].astype(int)]]
    assert (map_labels[held] == yh_true).mean() > 0.85

    # 5. density grid + components run on the real patch map
    lg = cm.label_grid(grid, map_labels)
    fr, cnt = cm.density_cells(lg, NAMES, cell_patches=6, min_patches=10)
    assert np.nanmax(fr) <= 1.0 and np.nansum(fr[0, 0]) == pytest.approx(1.0)
    comps = cm.components(lg, "sponge", min_patches=4)
    assert comps and comps[0]["n_patches"] >= comps[-1]["n_patches"]


def test_suggest_respects_exclusions_spacing_and_rare_preference():
    rng = np.random.default_rng(0)
    n, k_cls = 3000, 3
    xy = rng.uniform(0, 2000, (n, 2))
    feats = rng.normal(size=(n, 6))
    proba = rng.dirichlet(np.ones(k_cls) * 3, size=n)
    ref = rng.uniform(0, 2000, (60, 2))
    wins = [(500, 500, 300, 300)]
    ok = al.eligible_mask(xy, ref, exclude_radius_px=40.0, exclude_windows=wins)
    idx, why = al.suggest(proba, feats, xy, k=50, eligible=ok, min_sep_px=30.0, explore_frac=0.3, seed=1)
    assert len(idx) == 50 and len(set(idx.tolist())) == 50
    assert ok[idx].all()                                      # never from excluded zones
    d = np.hypot(*(xy[idx][:, None, :] - xy[idx][None, :, :]).transpose(2, 0, 1))
    assert d[np.triu_indices(50, 1)].min() >= 30.0
    assert (why == "explore").sum() == 15 and (why == "uncertain").sum() == 35
    again, _ = al.suggest(proba, feats, xy, k=50, eligible=ok, min_sep_px=30.0, explore_frac=0.3, seed=1)
    assert np.array_equal(idx, again)                         # reproducible


def test_rare_predicted_class_is_favoured_at_equal_margin():
    n = 2000
    proba = np.tile([0.5, 0.3, 0.2], (n, 1)).astype(float)    # same margin everywhere
    proba[:100] = [0.2, 0.3, 0.5]                             # a rare predicted class (5 %)
    xy = np.column_stack([np.arange(n, dtype=float) * 10, np.zeros(n)])
    feats = np.random.default_rng(0).normal(size=(n, 4))
    idx, _ = al.suggest(proba, feats, xy, k=20, eligible=np.ones(n, bool), explore_frac=0.0, seed=0)
    assert (idx < 100).mean() > 0.5


def test_density_and_components_small_case():
    lg = np.full((4, 6), "", dtype=object)
    lg[:, :2] = "rock"
    lg[0:2, 2:4] = "sponge"
    lg[2:4, 2:4] = "rock"
    lg[3, 5] = "sponge"
    fr, cnt = cm.density_cells(lg, ["rock", "sponge"], cell_patches=2, min_patches=4)
    assert fr.shape == (2, 3, 2) and cnt.tolist() == [[4, 4, 0], [4, 4, 1]]
    assert fr[0, 0].tolist() == [1.0, 0.0] and fr[0, 1].tolist() == [0.0, 1.0]
    assert np.isnan(fr[0, 2]).all() and np.isnan(fr[1, 2]).all()   # too few valid patches
    comps = cm.components(lg, "sponge", min_patches=1, patch_m=0.05)
    assert [c["n_patches"] for c in comps] == [4, 1]
    assert comps[0]["area_m2"] == pytest.approx(4 * 0.0025)
