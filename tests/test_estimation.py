import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import estimation as est  # noqa: E402

CLASSES = ["rock", "algae", "sponge", "sand"]
TRUE_P = np.array([0.55, 0.30, 0.10, 0.05])
# rows: true class, columns: predicted class (a deliberately biased classifier)
CONF = np.array([
    [0.80, 0.10, 0.08, 0.02],
    [0.25, 0.65, 0.07, 0.03],
    [0.30, 0.20, 0.45, 0.05],
    [0.20, 0.05, 0.05, 0.70],
])


def make_population(n=30000, seed=0):
    rng = np.random.default_rng(seed)
    truth = rng.choice(4, size=n, p=TRUE_P)
    pred = np.array([rng.choice(4, p=CONF[t]) for t in truth])
    return np.array(CLASSES)[truth], np.array(CLASSES)[pred]


def test_wilson_basic():
    lo, hi = est.wilson(50, 100)
    assert 0.40 < lo < 0.5 < hi < 0.60
    assert est.wilson(0, 0) == (0.0, 1.0)
    assert est.wilson(0, 400)[0] == pytest.approx(0.0, abs=1e-12)


def test_coverage_and_bias_by_simulation():
    truth, pred = make_population()
    true_share = {c: float((truth == c).mean()) for c in CLASSES}
    naive = est.naive_map_estimate(pred, CLASSES)
    assert max(abs(naive[c]["p"] - true_share[c]) for c in CLASSES) > 0.02   # raw map counts are clearly biased

    rng = np.random.default_rng(1)
    n_ref, reps = 400, 600
    cover = {m: {c: 0 for c in CLASSES} for m in ("direct", "post", "ppi")}
    bias = {m: {c: [] for c in CLASSES} for m in ("direct", "post", "ppi")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(reps):
            idx = rng.choice(len(truth), size=n_ref, replace=False)
            y, yh = truth[idx], pred[idx]
            res = {"direct": est.direct_estimate(y, CLASSES),
                   "post": est.post_stratified_estimate(y, yh, pred, CLASSES)["estimates"],
                   "ppi": est.ppi_estimate(y, yh, pred, CLASSES)["estimates"]}
            for m, r in res.items():
                for c in CLASSES:
                    cover[m][c] += r[c]["lo"] <= true_share[c] <= r[c]["hi"]
                    bias[m][c].append(r[c]["p"] - true_share[c])
    for m in cover:
        for c in CLASSES:
            cov = cover[m][c] / reps
            floor = 0.90 if c == "sand" else 0.92         # the 5 % class is the hardest for normal-theory CIs
            assert floor <= cov <= 0.995, (m, c, cov)
            assert abs(np.mean(bias[m][c])) < 0.006, (m, c, np.mean(bias[m][c]))


def test_map_assistance_tightens_intervals_for_a_good_classifier():
    rng = np.random.default_rng(2)
    n = 30000
    truth = rng.choice(["a", "b", "c"], size=n, p=[0.6, 0.3, 0.1])
    flip = rng.random(n) < 0.05
    pred = np.where(flip, rng.choice(["a", "b", "c"], size=n), truth)
    idx = rng.choice(n, size=300, replace=False)
    d = est.direct_estimate(truth[idx], ["a", "b", "c"])
    p = est.post_stratified_estimate(truth[idx], pred[idx], pred, ["a", "b", "c"])["estimates"]
    assert all(p[c]["se"] < d[c]["se"] for c in "abc")


def test_unsampled_stratum_is_flagged_not_hidden():
    map_labels = np.array(["a"] * 90 + ["b"] * 9 + ["c"] * 1)
    y_ref = np.array(["a"] * 20 + ["b"] * 5)
    yh_ref = np.array(["a"] * 20 + ["b"] * 5)
    with pytest.warns(UserWarning, match="no reference points"):
        out = est.post_stratified_estimate(y_ref, yh_ref, map_labels, ["a", "b", "c"])
    assert out["diagnostics"]["unsampled_strata"] == ["c"]
    assert out["diagnostics"]["unsampled_area_share"] == pytest.approx(0.01)


def test_single_point_stratum_gets_nonzero_variance():
    map_labels = np.array(["a"] * 50 + ["b"] * 50)
    out = est.post_stratified_estimate(["a", "a", "a", "b"], ["a", "a", "a", "b"], map_labels, ["a", "b"])
    assert out["diagnostics"]["single_point_strata"] == ["b"]
    assert out["estimates"]["b"]["se"] > 0


def test_reference_policy_drops_unscorable():
    keep, n_drop = est.apply_reference_policy(["a", "UNSCORABLE", "b", "UNSCORABLE", "UNSURE"])
    assert keep.tolist() == [True, False, True, False, True] and n_drop == 2
