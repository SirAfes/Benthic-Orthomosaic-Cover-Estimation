import csv
import os
import sys

import numpy as np
import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC)
import crop_eval as ce  # noqa: E402
import crop_library as cl  # noqa: E402
from features import ColorTextureFeatures  # noqa: E402


def test_features_shape_determinism_and_constant_image():
    fx = ColorTextureFeatures()
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (40, 30, 3), dtype=np.uint8)
    a, b = fx(img), fx(img)
    assert a.shape == (len(fx.names),) and np.array_equal(a, b) and np.isfinite(a).all()
    flat = fx(np.full((16, 16, 3), 120, dtype=np.uint8))
    assert np.isfinite(flat).all()
    with pytest.raises(ValueError):
        fx(np.zeros((5, 5), dtype=np.uint8))


def test_features_separate_red_from_green():
    fx = ColorTextureFeatures()
    red = np.zeros((20, 20, 3), np.uint8)
    red[..., 0] = 200
    green = np.zeros((20, 20, 3), np.uint8)
    green[..., 1] = 200
    d = np.abs(fx(red) - fx(green))
    assert d[fx.names.index("mean_r")] > 0.5 and d[fx.names.index("mean_g")] > 0.5


def test_source_id_and_schema(tmp_path):
    assert cl.source_id("0126_10_ADS_0926_3.png") == "0126_10_ADS_0926"
    p = tmp_path / "s.csv"
    with open(p, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["raw_folder", "n_crops", "n_source_photos", "canonical_label", "morphotype_group", "layer", "note"])
        wr.writerow(["A1", 1, 1, "A", "g", "B", ""])
        wr.writerow(["A2", 1, 1, "A", "g", "B", ""])
        wr.writerow(["F", 1, 1, "Fish", "f", "exclude", ""])
        wr.writerow(["(none)", 0, 0, "Sand", "s", "A", ""])
    sch = cl.load_schema(str(p))
    assert set(sch) == {"A1", "A2", "F"} and sch["A2"]["label"] == "A"
    rows = [
        {"folder": "A1", "file": "p_1_0.png", "w": 100, "h": 100, "source_photo": "p_1"},
        {"folder": "A2", "file": "p_2_0.png", "w": 100, "h": 100, "source_photo": "p_2"},
        {"folder": "A1", "file": "p_3_0.png", "w": 3, "h": 3, "source_photo": "p_3"},
        {"folder": "F", "file": "p_4_0.png", "w": 100, "h": 100, "source_photo": "p_4"},
        {"folder": "ZZ", "file": "p_5_0.png", "w": 100, "h": 100, "source_photo": "p_5"},
    ]
    kept, dropped = cl.build_table(rows, sch)
    assert [r["label"] for r in kept] == ["A", "A"]          # merged folders share a label
    assert sorted(d["reason"] for d in dropped) == ["folder not in schema", "layer exclude", "min side < 32px"]
    assert cl.groups_per_label(kept) == {"A": 2}


def _toy(n_groups=6, per=8, seed=0):
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for gi in range(n_groups):
        for lab, c in (("a", 0.0), ("b", 3.0), ("c", -3.0)):
            for _ in range(per):
                X.append(rng.normal(c, 1.0, 5))
                y.append(lab)
                g.append(f"g{gi}")
    # a label that exists in a single photo only
    for _ in range(5):
        X.append(rng.normal(8, 1, 5))
        y.append("solo")
        g.append("g0")
    return np.array(X), np.array(y), np.array(g)


def test_logo_never_trains_on_test_group_and_skips_single_group_labels():
    X, y, g = _toy()
    report, (ys, gs, preds) = ce.evaluate(X, y, g, model_names=("logreg",), n_boot=50)
    assert "solo" in report["labels_skipped_lt_min_groups"]
    assert "solo" not in report["labels_evaluated"] and set(ys) == {"a", "b", "c"}
    assert report["results"]["logreg"]["macro_f1"] > 0.9
    assert report["results"]["majority"]["macro_f1"] < 0.4
    lo, hi = report["results"]["logreg"]["macro_f1_ci95"]
    assert lo <= report["results"]["logreg"]["macro_f1"] <= hi + 1e-9


def test_no_group_leakage_in_logo(monkeypatch):
    X, y, g = _toy(n_groups=4)
    seen = []
    orig = ce.make_model

    def spy(name, seed=0):
        m = orig(name, seed)
        fit = m.fit

        def fit_spy(Xt, yt):
            seen.append(len(Xt))
            return fit(Xt, yt)
        m.fit = fit_spy
        return m
    monkeypatch.setattr(ce, "make_model", spy)
    sel = y != "solo"
    ce.logo_predict(X[sel], y[sel], g[sel], "logreg")
    n = sel.sum()
    per_group = n // 4
    assert seen == [n - per_group] * 4            # every fold trained without its test photo
