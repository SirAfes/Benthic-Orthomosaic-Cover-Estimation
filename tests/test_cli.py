import csv
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pipeline  # noqa: E402
import reference_sampling as rs  # noqa: E402
import tiling  # noqa: E402
from test_workflow import NAMES, synthetic_survey  # noqa: E402

GEO = ["--mm-per-px", "1.0"]
MASKGEO = ["--full-size", "720", "480", "--origin", "0", "0"]


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_cli_end_to_end(tmp_path, monkeypatch):
    img, truth = synthetic_survey()
    monkeypatch.setattr(pipeline, "open_reader", lambda path: tiling.ArrayReader(img))
    f = 4
    mask_png = str(tmp_path / "mask.png")
    Image.fromarray(((img[..., 3] > 0)[::f, ::f] * 255).astype("uint8")).save(mask_png)
    mg = ["--mask-png", mask_png] + MASKGEO

    # reference sample first (edge buffer 3 mm at 1 mm/px)
    ref_dir = str(tmp_path / "ref")
    assert rs.main(["--raster", "x.tif", "--mask-png", mask_png, "--full-size", "720", "480", "--origin", "0", "0",
                    "--mm-per-px", "1.0", "--seed", "5", "--n-points", "400", "--n-windows", "2",
                    "--window-cm", "6", "--edge-buffer-mm", "12", "--out", ref_dir]) == 0
    rows = read_csv(os.path.join(ref_dir, "reference_points.csv"))
    lab_path = os.path.join(ref_dir, "reference_labelled.csv")
    with open(lab_path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["id", "x_px", "y_px", "label"])
        for r in rows:
            x, y = float(r["x_px"]), float(r["y_px"])
            wr.writerow([r["id"], r["x_px"], r["y_px"], NAMES[truth[int(y), int(x)]]])
    # one unscorable point and one unlabelled point must be handled, not crash
    with open(lab_path, "a", newline="") as fh:
        csv.writer(fh).writerows([["PX1", "100", "100", "UNSCORABLE"], ["PX2", "120", "120", ""]])

    grid_path = str(tmp_path / "grid.npz")
    pipeline.main(["grid", *mg, *GEO, "--patch-cm", "1.6", "--out", grid_path])
    g = tiling.Grid.load(grid_path)
    assert g.patch_px == 16 and g.n > 1000

    feats = str(tmp_path / "feats.npy")
    pipeline.main(["features", "--raster", "x.tif", "--grid", grid_path, "--backbone", "color",
                   "--colour-normalise", "--out", feats])
    assert np.load(feats).shape[0] == g.n and os.path.exists(feats + ".colour.json")

    # training labels: random patches, truth at the centre
    rng = np.random.default_rng(0)
    pick = rng.choice(g.n, 400, replace=False)
    c = g.centers[pick]
    train_csv = str(tmp_path / "train.csv")
    with open(train_csv, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["id", "x_px", "y_px", "label"])
        for i, (x, y) in enumerate(c):
            wr.writerow([i, x, y, NAMES[truth[int(y), int(x)]]])
    model_dir = str(tmp_path / "model")
    pipeline.main(["train", "--grid", grid_path, "--features", feats, "--labels", train_csv,
                   "--exclude", lab_path, "--exclusion-radius-cm", "0.8", *GEO, "--origin", "0", "0",
                   "--out", model_dir])
    rep = json.load(open(os.path.join(model_dir, "train_report.json")))
    assert rep["n_train_patches"] > 100 and rep["dropped_near_reference_sample"] > 0 and sorted(rep["classes"]) == sorted(NAMES)
    preds = os.path.join(model_dir, "patch_predictions.csv")
    assert len(read_csv(preds)) == g.n

    est_dir = str(tmp_path / "est")
    pipeline.main(["estimate", *mg, *GEO, "--grid", grid_path, "--predictions", preds,
                   "--reference", lab_path, "--edge-buffer-mm", "12", "--out", est_dir])
    er = json.load(open(os.path.join(est_dir, "estimate_report.json")))
    assert er["unscorable_points_dropped"] == 1 and er["unlabelled_points_ignored"] == 1
    assert er["primary_estimator"] == "post_stratified"
    table = {r["class"]: r for r in read_csv(os.path.join(est_dir, "cover_estimates.csv"))}
    assert set(table) == set(NAMES)
    assert abs(sum(float(r["post_p"]) for r in table.values()) - 1.0) < 1e-3
    for r in table.values():
        assert float(r["post_lo95"]) <= float(r["post_p"]) <= float(r["post_hi95"])
        assert float(r["area_m2"]) >= 0

    sug = str(tmp_path / "sug" / "suggestions.csv")
    pipeline.main(["suggest", "--grid", grid_path, "--features", feats,
                   "--proba", os.path.join(model_dir, "proba.npy"),
                   "--classes", os.path.join(model_dir, "classes.json"),
                   "--exclude", lab_path, "--gold-windows", os.path.join(ref_dir, "gold_windows.csv"),
                   "--exclusion-radius-cm", "0.8", "--min-sep-cm", "3", *GEO, "-k", "40", "--out", sug])
    srows = read_csv(sug)
    assert len(srows) == 40 and all(r["label"] == "" for r in srows)
    ref_xy = np.array([[float(r["x_px"]), float(r["y_px"])] for r in rows])
    sxy = np.array([[float(r["x_px"]), float(r["y_px"])] for r in srows])
    d = np.hypot(*(sxy[:, None, :] - ref_xy[None, :, :]).transpose(2, 0, 1)).min()
    assert d > 8.0                                           # never next to the reference sample

    maps_dir = str(tmp_path / "maps")
    pipeline.main(["maps", "--grid", grid_path, "--predictions", preds, "--cell-patches", "6",
                   "--min-patches", "10", "--object-classes", "sponge", *GEO, "--out", maps_dir])
    assert read_csv(os.path.join(maps_dir, "density_cells.csv")) and read_csv(os.path.join(maps_dir, "components.csv"))
