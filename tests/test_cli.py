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


def test_cli_objects_outlines_only_object_classes_and_georeferences(tmp_path, monkeypatch):
    from scipy import ndimage

    img, truth = synthetic_survey()
    monkeypatch.setattr(pipeline, "open_reader", lambda path: tiling.ArrayReader(img))

    def orange_generator(win):                                  # stand-in for SAM2: smoothed orange blobs
        rgb = win[..., :3].astype(float)
        score = ndimage.uniform_filter(rgb[..., 0] - rgb[..., 2], 9)
        lab, n = ndimage.label(score > 110)
        return [{"segmentation": lab == k, "stability_score": 0.9, "predicted_iou": 0.9} for k in range(1, n + 1)]

    monkeypatch.setattr(pipeline, "open_generator", lambda a: orange_generator)

    g = tiling.make_grid((img[..., 3] > 0)[::4, ::4], (4.0, 4.0), 720, 480, 16)
    c = g.centers
    t = truth[c[:, 1].astype(int), c[:, 0].astype(int)]
    proba = np.full((g.n, 3), 0.025)
    proba[np.arange(g.n), t] = 0.95
    grid_path, proba_path, classes_path = (str(tmp_path / n) for n in ("grid.npz", "proba.npy", "classes.json"))
    g.save(grid_path)
    np.save(proba_path, proba)
    json.dump(NAMES, open(classes_path, "w"))

    out = str(tmp_path / "objects.geojson")
    pipeline.main(["objects", "--raster", "x.tif", "--grid", grid_path, "--proba", proba_path,
                   "--classes", classes_path, "--object-classes", "sponge", "--mm-per-px", "1.0",
                   "--origin", "100", "200", "--min-area-cm2", "20", "--out", out])
    fc = json.load(open(out))
    feats = fc["features"]
    assert len(feats) == 2 and {f["properties"]["class"] for f in feats} == {"sponge"}
    true_m2 = float(((truth == 2) & (img[..., 3] > 0)).sum()) * 1e-6
    got_m2 = sum(f["properties"]["area_m2"] for f in feats)
    assert abs(got_m2 - true_m2) / true_m2 < 0.10
    xy = np.array([p for f in feats for p in f["geometry"]["coordinates"][0]])
    assert xy[:, 0].min() >= 100 and xy[:, 0].max() <= 100.72 and xy[:, 1].max() <= 200 and xy[:, 1].min() >= 199.52


def test_cli_detect_feeds_estimate_and_recovers_true_cover(tmp_path, monkeypatch):
    from scipy import ndimage

    img, truth = synthetic_survey()
    monkeypatch.setattr(pipeline, "open_reader", lambda path: tiling.ArrayReader(img))

    def sponge_detector(win):                                   # stand-in for YOLO-seg / Grounding DINO
        rgb = win[..., :3].astype(float)
        lab, n = ndimage.label(ndimage.uniform_filter(rgb[..., 0] - rgb[..., 2], 9) > 110)
        return [{"segmentation": lab == k, "class": "sponge", "score": 0.8} for k in range(1, n + 1)]

    monkeypatch.setattr(pipeline, "open_detector", lambda a: sponge_detector)
    f = 4
    low = (img[..., 3] > 0)[::f, ::f]
    mask_png = str(tmp_path / "mask.png")
    Image.fromarray((low * 255).astype("uint8")).save(mask_png)
    mg = ["--mask-png", mask_png] + MASKGEO

    ref_dir = str(tmp_path / "ref")
    assert rs.main(["--raster", "x.tif", "--mask-png", mask_png, "--full-size", "720", "480", "--origin", "0", "0",
                    "--mm-per-px", "1.0", "--seed", "5", "--n-points", "400", "--n-windows", "2",
                    "--window-cm", "6", "--edge-buffer-mm", "12", "--out", ref_dir]) == 0
    lab_path = str(tmp_path / "ref_labelled.csv")
    with open(lab_path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["id", "x_px", "y_px", "label"])
        for r in read_csv(os.path.join(ref_dir, "reference_points.csv")):
            wr.writerow([r["id"], r["x_px"], r["y_px"],
                         "sponge" if truth[int(float(r["y_px"])), int(float(r["x_px"]))] == 2 else "other"])

    grid_path = str(tmp_path / "grid.npz")
    pipeline.main(["grid", *mg, *GEO, "--patch-cm", "1.6", "--out", grid_path])
    g = tiling.Grid.load(grid_path)
    det_dir = str(tmp_path / "det")
    pipeline.main(["detect", "--raster", "x.tif", "--grid", grid_path, *GEO, "--origin", "0", "0",
                   "--min-area-cm2", "20", "--out", det_dir])
    assert len(json.load(open(os.path.join(det_dir, "instances.geojson")))["features"]) == 2
    preds = os.path.join(det_dir, "patch_predictions.csv")
    labels = [r["label"] for r in read_csv(preds)]
    assert len(labels) == g.n and set(labels) == {"sponge", "other"}
    rep = json.load(open(os.path.join(det_dir, "detect_report.json")))
    assert rep["instances_per_class"] == {"sponge": 2}

    est_dir = str(tmp_path / "est")
    pipeline.main(["estimate", *mg, *GEO, "--grid", grid_path, "--predictions", preds, "--reference", lab_path,
                   "--edge-buffer-mm", "12", "--out", est_dir])
    table = {r["class"]: r for r in read_csv(os.path.join(est_dir, "cover_estimates.csv"))}
    pop_low = rs.eroded_mask(low, (float(f), float(f)), 1.0, 12)
    pop_px = np.kron(pop_low, np.ones((f, f), dtype=bool))[:480, :720]
    true_share = float((truth[pop_px] == 2).mean())
    assert abs(float(table["sponge"]["post_p"]) - true_share) < 0.07
    assert float(table["sponge"]["post_lo95"]) <= true_share <= float(table["sponge"]["post_hi95"]) + 0.02
