"""Command-line pipeline: grid -> features -> train -> estimate / suggest / maps.

    python src/pipeline.py grid     ...   fixed-physical-size patch grid over the valid area
    python src/pipeline.py features ...   (resumable) features for every patch
    python src/pipeline.py train    ...   fit on labelled patches, predict every patch
    python src/pipeline.py estimate ...   design-based cover estimates with intervals
    python src/pipeline.py suggest  ...   next patches to label (training only)
    python src/pipeline.py maps     ...   density grid and object components
    python src/pipeline.py objects  ...   SAM2 outlines of discrete organisms, classified by patch vote
    python src/pipeline.py detect   ...   YOLO-seg / Grounding DINO + SAM2 instances -> patch map for `estimate`

Geometry options are shared: either ``--raster`` (needs rasterio) or ``--mask-png`` with
``--full-size`` and ``--origin`` (see reference_sampling.py), plus ``--mm-per-px``.

The reference sample (reference_points.csv) is used ONLY by ``estimate`` and as an exclusion
zone in ``train`` / ``suggest``; its labels never reach the classifier.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import active_learning as al  # noqa: E402
import classifier as clf  # noqa: E402
import colour  # noqa: E402
import cover_maps as cm  # noqa: E402
import detection as det  # noqa: E402
import estimation as est  # noqa: E402
import features as ft  # noqa: E402
import objects as ob  # noqa: E402
import patch_features as pf  # noqa: E402
import reference_sampling as rs  # noqa: E402
import tiling  # noqa: E402


def open_reader(path: str):
    return tiling.RasterioReader(path)


def open_generator(a):
    """Mask generator for ``objects`` (SAM2).  Replaced in tests; needs torch + sam2 + a checkpoint."""
    if not (a.sam2_config and a.sam2_checkpoint):
        raise SystemExit("objects needs --sam2-config and --sam2-checkpoint")
    return ob.load_sam2_generator(a.sam2_config, a.sam2_checkpoint, a.device)


def open_detector(a):
    """Detector for ``detect``.  Replaced in tests; the backends need torch and model weights."""
    import detectors

    if a.backend == "yolo":
        if not a.weights:
            raise SystemExit("--backend yolo needs --weights")
        return detectors.load_yolo_seg(a.weights, a.device, a.min_score, a.imgsz)
    prompts = dict(p.split("=", 1) for p in a.prompt or [])
    if not (prompts and a.sam2_config and a.sam2_checkpoint):
        raise SystemExit("--backend grounded needs --prompt CLASS=phrase, --sam2-config and --sam2-checkpoint")
    return detectors.load_grounded_sam(prompts, a.sam2_config, a.sam2_checkpoint, a.device)


def add_geometry(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--raster", help="Mosaic GeoTIFF (needed for features; for the mask if no --mask-png)")
    ap.add_argument("--mask-png", help="Validity mask PNG (non-zero = valid) instead of reading the alpha band")
    ap.add_argument("--full-size", type=int, nargs=2, metavar=("W", "H"))
    ap.add_argument("--origin", type=float, nargs=2, metavar=("X0", "Y0"))
    ap.add_argument("--mm-per-px", type=float, required=True)


def load_mask(a):
    if a.mask_png:
        if not (a.full_size and a.origin):
            raise SystemExit("--mask-png needs --full-size and --origin")
        mask = rs.read_mask_png(a.mask_png)
        W, H = a.full_size
        scale = (W / mask.shape[1], H / mask.shape[0])
        return mask, scale, rs.AffineXY(a.origin[0], a.origin[1], a.mm_per_px / 1000.0), (W, H)
    if not a.raster:
        raise SystemExit("Give --raster or --mask-png")
    return rs.read_mask_rasterio(a.raster)


def read_points(path: str, label_col: str | None = "label"):
    """(ids, xy_px, labels).  Rows with an empty label are returned with label ''."""
    ids, xy, lab = [], [], []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            ids.append(r.get("id", ""))
            xy.append((float(r["x_px"]), float(r["y_px"])))
            lab.append((r.get(label_col) or "").strip() if label_col else "")
    return ids, np.array(xy, dtype=float).reshape(-1, 2), np.array(lab, dtype=object)


def world(transform, xy):
    if transform is None:
        return [("", "")] * len(xy)
    return [rs.pixel_to_world(transform, x, y) for x, y in xy]


# ---------------------------------------------------------------- commands
def cmd_grid(a):
    mask, scale, _, (W, H) = load_mask(a)
    p = tiling.patch_px_for(a.patch_cm, a.mm_per_px)
    g = tiling.make_grid(mask, scale, W, H, p, a.min_valid_frac)
    g.save(a.out)
    print(f"patch {a.patch_cm} cm = {p} px; {g.n} kept of {g.n_rows * g.n_cols} patches -> {a.out}")


def cmd_features(a):
    g = tiling.Grid.load(a.grid)
    reader = open_reader(a.raster)
    norm = None
    if a.colour_normalise:
        norm = colour.ColourNormaliser.fit(colour.sample_valid_pixels(reader, g, seed=a.seed))
        with open(a.out + ".colour.json", "w") as fh:
            fh.write(norm.to_json())
    backbone = ft.load_backbone(a.backbone, a.device)

    def progress(done, total):
        print(f"\r{done}/{total} patches", end="", flush=True)

    X = pf.extract_grid_features(g, reader, backbone, normaliser=norm, batch_size=a.batch_size,
                                 chunk_dir=a.out + ".chunks", progress=progress)
    print()
    np.save(a.out, X.astype(np.float32))
    print(f"features {X.shape} -> {a.out}")


def _exclusion_xy(paths):
    pts = [read_points(p, None)[1] for p in paths or []]
    return np.concatenate(pts) if pts else np.zeros((0, 2))


def cmd_train(a):
    import joblib

    g = tiling.Grid.load(a.grid)
    X = np.load(a.features)
    _, xy, lab = read_points(a.labels)
    idx = g.index_of(xy)
    ok = (idx >= 0) & (lab != "")
    n_outside = int(((idx < 0) & (lab != "")).sum())
    excl_xy = _exclusion_xy(a.exclude)
    if len(excl_xy):
        r_px = a.exclusion_radius_cm * 10.0 / a.mm_per_px
        near = ~al.eligible_mask(g.centers[idx.clip(0)], excl_xy, r_px)
        n_leak = int((ok & near).sum())
        ok &= ~near
    else:
        n_leak = 0
    rows, labels = idx[ok], lab[ok].astype(str)
    # one label per patch; patches with conflicting labels are dropped and reported
    keep_rows, keep_lab, conflicts = [], [], 0
    for r in np.unique(rows):
        ls = set(labels[rows == r])
        if len(ls) == 1:
            keep_rows.append(r)
            keep_lab.append(ls.pop())
        else:
            conflicts += 1
    keep_rows = np.array(keep_rows)
    model = clf.PatchClassifier(C=a.C).fit(X[keep_rows], keep_lab)
    proba = model.predict_proba(X)
    labels_all = clf.labels_from_proba(proba, model.classes_, a.min_conf)
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "proba.npy"), proba.astype(np.float32))
    with open(os.path.join(a.out, "classes.json"), "w") as fh:
        json.dump(list(map(str, model.classes_)), fh)
    joblib.dump(model, os.path.join(a.out, "model.joblib"))
    ctr = g.centers
    tr = rs.AffineXY(a.origin[0], a.origin[1], a.mm_per_px / 1000.0) if a.origin else None
    w = world(tr, ctr)
    with open(os.path.join(a.out, "patch_predictions.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["patch", "x0", "y0", "x_px", "y_px", "x_m", "y_m", "label", "confidence"])
        for i in range(g.n):
            wr.writerow([i, g.x0[i], g.y0[i], f"{ctr[i, 0]:.1f}", f"{ctr[i, 1]:.1f}", w[i][0], w[i][1],
                         labels_all[i], f"{proba[i].max():.4f}"])
    rep = {"n_train_patches": int(len(keep_rows)), "classes": list(map(str, model.classes_)),
           "labelled_outside_grid": n_outside, "dropped_near_reference_sample": n_leak,
           "dropped_conflicting_patches": conflicts, "min_conf": a.min_conf}
    with open(os.path.join(a.out, "train_report.json"), "w") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps(rep, indent=2))


def cmd_estimate(a):
    g = tiling.Grid.load(a.grid)
    mask, scale, _, _ = load_mask(a)
    pop_low = rs.eroded_mask(mask, scale, a.mm_per_px, a.edge_buffer_mm)
    in_pop = g.in_mask(pop_low, scale)
    with open(a.predictions, newline="") as fh:
        map_labels = np.array([r["label"] for r in csv.DictReader(fh)], dtype=object)
    if len(map_labels) != g.n:
        raise SystemExit("predictions do not match the grid (different number of patches)")
    _, xy, lab = read_points(a.reference)
    n_unlabelled = int((lab == "").sum())
    lab_ok = lab != ""
    keep, n_unscorable = est.apply_reference_policy(lab[lab_ok])
    xy, lab = xy[lab_ok][keep], lab[lab_ok][keep].astype(str)
    idx = g.index_of(xy)
    inside = idx >= 0
    n_off = int((~inside).sum())
    y_ref, yh_ref = lab[inside], map_labels[idx[inside]].astype(str)
    pop_labels = map_labels[in_pop].astype(str)
    classes = sorted(set(y_ref) | set(pop_labels) | set(yh_ref))
    post = est.post_stratified_estimate(y_ref, yh_ref, pop_labels, classes)
    direct = est.direct_estimate(y_ref, classes)
    ppi = est.ppi_estimate(y_ref, yh_ref, pop_labels, classes)
    naive = est.naive_map_estimate(pop_labels, classes)
    patch_m2 = (g.patch_px * a.mm_per_px / 1000.0) ** 2
    pop_area = float(in_pop.sum() * patch_m2)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "cover_estimates.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["class", "n_ref", "post_p", "post_se", "post_lo95", "post_hi95", "area_m2",
                     "direct_p", "direct_lo95", "direct_hi95", "ppi_p", "ppi_lo95", "ppi_hi95", "naive_map_p"])
        for c in classes:
            e, d, q = post["estimates"][c], direct[c], ppi["estimates"][c]
            wr.writerow([c, int((y_ref == c).sum()), f"{e['p']:.4f}", f"{e['se']:.4f}", f"{e['lo']:.4f}",
                         f"{e['hi']:.4f}", f"{e['p'] * pop_area:.3f}", f"{d['p']:.4f}", f"{d['lo']:.4f}",
                         f"{d['hi']:.4f}", f"{q['p']:.4f}", f"{q['lo']:.4f}", f"{q['hi']:.4f}",
                         f"{naive[c]['p']:.4f}"])
    rep = {"n_reference_used": int(len(y_ref)), "unlabelled_points_ignored": n_unlabelled,
           "unscorable_points_dropped": n_unscorable, "points_outside_grid_dropped": n_off,
           "population_patches": int(in_pop.sum()), "population_area_m2": pop_area,
           "primary_estimator": "post_stratified", "diagnostics": post["diagnostics"],
           "note": "Cover is of the scorable area inside the eroded valid region."}
    with open(os.path.join(a.out, "estimate_report.json"), "w") as fh:
        json.dump(rep, fh, indent=2, default=float)
    print(open(os.path.join(a.out, "cover_estimates.csv")).read())


def cmd_suggest(a):
    g = tiling.Grid.load(a.grid)
    X = np.load(a.features)
    proba = np.load(a.proba)
    with open(a.classes) as fh:
        classes = json.load(fh)
    excl = _exclusion_xy(a.exclude)
    wins = []
    for p in a.gold_windows or []:
        with open(p, newline="") as fh:
            wins += [(int(r["x0_px"]), int(r["y0_px"]), int(r["w_px"]), int(r["h_px"])) for r in csv.DictReader(fh)]
    r_px = a.exclusion_radius_cm * 10.0 / a.mm_per_px
    ctr = g.centers
    ok = al.eligible_mask(ctr, excl, r_px, wins, margin_px=r_px)
    sep = a.min_sep_cm * 10.0 / a.mm_per_px
    idx, why = al.suggest(proba, X, ctr, a.k, ok, min_sep_px=sep, explore_frac=a.explore_frac, seed=a.seed)
    tr = rs.AffineXY(a.origin[0], a.origin[1], a.mm_per_px / 1000.0) if a.origin else None
    w = world(tr, ctr[idx])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["id", "x_px", "y_px", "x_m", "y_m", "reason", "predicted", "confidence", "label"])
        for j, i in enumerate(idx, start=1):
            wr.writerow([f"S{j:04d}", f"{ctr[i, 0]:.1f}", f"{ctr[i, 1]:.1f}", w[j - 1][0], w[j - 1][1], why[j - 1],
                         classes[int(proba[i].argmax())], f"{proba[i].max():.3f}", ""])
    print(f"{len(idx)} suggestions ({(why == 'explore').sum()} exploratory) -> {a.out}")


def cmd_maps(a):
    g = tiling.Grid.load(a.grid)
    with open(a.predictions, newline="") as fh:
        labels = np.array([r["label"] for r in csv.DictReader(fh)], dtype=object)
    lg = cm.label_grid(g, labels)
    classes = sorted(set(labels))
    fr, cnt = cm.density_cells(lg, classes, a.cell_patches, a.min_patches)
    tr = rs.AffineXY(a.origin[0], a.origin[1], a.mm_per_px / 1000.0) if a.origin else None
    os.makedirs(a.out, exist_ok=True)
    cm.write_density_csv(os.path.join(a.out, "density_cells.csv"), fr, cnt, classes, a.cell_patches * g.patch_px, tr)
    patch_m = g.patch_px * a.mm_per_px / 1000.0
    with open(os.path.join(a.out, "components.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["class", "n_patches", "area_m2", "row0", "row1", "col0", "col1"])
        for c in a.object_classes or []:
            for comp in cm.components(lg, c, a.min_component_patches, patch_m):
                wr.writerow([c, comp["n_patches"], f"{comp['area_m2']:.5f}", comp["row0"], comp["row1"],
                             comp["col0"], comp["col1"]])
    print(f"density cells -> {a.out}/density_cells.csv; components for {a.object_classes or []}")


def cmd_objects(a):
    g = tiling.Grid.load(a.grid)
    proba = np.load(a.proba)
    with open(a.classes) as fh:
        classes = json.load(fh)
    unknown = sorted(set(a.object_classes) - set(classes))
    if unknown:
        raise SystemExit(f"object classes not in the model: {unknown}")
    norm = None
    if a.colour_json:
        with open(a.colour_json) as fh:
            norm = colour.ColourNormaliser.from_json(fh.read())
    reader = open_reader(a.raster)
    mmu_px = int(round(a.min_area_cm2 * 100.0 / a.mm_per_px ** 2))
    objs = ob.delineate_mosaic(reader, g, proba, classes, a.object_classes, open_generator(a),
                               tile_px=a.tile_px, margin_px=a.margin_px, normaliser=norm,
                               mmu_px=mmu_px, min_conf=a.min_conf)
    tr = rs.AffineXY(a.origin[0], a.origin[1], a.mm_per_px / 1000.0) if a.origin else None
    fc = ob.to_geojson(objs, a.mm_per_px / 1000.0, tr)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(fc, fh)
    counts = {c: sum(o["class"] == c for o in objs) for c in a.object_classes}
    print(f"{len(objs)} objects {counts} -> {a.out} (coordinates: {'world' if tr else 'pixels'})")


def cmd_detect(a):
    g = tiling.Grid.load(a.grid)
    norm = None
    if a.colour_json:
        with open(a.colour_json) as fh:
            norm = colour.ColourNormaliser.from_json(fh.read())
    reader = open_reader(a.raster)
    mmu_px = int(round(a.min_area_cm2 * 100.0 / a.mm_per_px ** 2))

    def progress(done, total):
        print(f"\rtile {done}/{total}", end="", flush=True)

    insts = det.detect_mosaic(reader, g, open_detector(a), tile_px=a.tile_px, margin_px=a.margin_px,
                              normaliser=norm, mmu_px=mmu_px, min_score=a.min_score,
                              classes_keep=set(a.classes) if a.classes else None, progress=progress)
    print()
    labels, conf = det.patch_labels(g, insts, other=a.other_label, min_cover=a.min_cover)
    tr = rs.AffineXY(a.origin[0], a.origin[1], a.mm_per_px / 1000.0) if a.origin else None
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "instances.geojson"), "w") as fh:
        json.dump(ob.to_geojson(insts, a.mm_per_px / 1000.0, tr), fh)
    ctr = g.centers
    w = world(tr, ctr)
    with open(os.path.join(a.out, "patch_predictions.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["patch", "x0", "y0", "x_px", "y_px", "x_m", "y_m", "label", "confidence"])
        for i in range(g.n):
            wr.writerow([i, g.x0[i], g.y0[i], f"{ctr[i, 0]:.1f}", f"{ctr[i, 1]:.1f}", w[i][0], w[i][1],
                         labels[i], f"{conf[i]:.4f}"])
    per_class = {c: sum(i["class"] == c for i in insts) for c in sorted({i["class"] for i in insts})}
    rep = {"backend": a.backend, "instances_per_class": per_class,
           "patches_per_label": {str(k): int((labels == k).sum()) for k in sorted(set(labels))},
           "min_score": a.min_score, "min_cover": a.min_cover, "min_area_cm2": a.min_area_cm2,
           "note": "Labels come from detections only; cover is estimated by `estimate` from the reference sample."}
    with open(os.path.join(a.out, "detect_report.json"), "w") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps(rep, indent=2))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("grid")
    add_geometry(g)
    g.add_argument("--patch-cm", type=float, default=5.0)
    g.add_argument("--min-valid-frac", type=float, default=0.98)
    g.add_argument("--out", required=True)
    g.set_defaults(fn=cmd_grid)

    f = sub.add_parser("features")
    f.add_argument("--raster", required=True)
    f.add_argument("--grid", required=True)
    f.add_argument("--backbone", default="color", help="color | dinov2_vits14 | dinov2_vitb14 ...")
    f.add_argument("--device", default="cpu")
    f.add_argument("--batch-size", type=int, default=32)
    f.add_argument("--colour-normalise", action="store_true")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--out", required=True, help="features .npy")
    f.set_defaults(fn=cmd_features)

    t = sub.add_parser("train")
    t.add_argument("--grid", required=True)
    t.add_argument("--features", required=True)
    t.add_argument("--labels", required=True, help="CSV x_px,y_px,label (training labels)")
    t.add_argument("--exclude", nargs="*", help="CSVs with x_px,y_px kept away from training (reference sample)")
    t.add_argument("--exclusion-radius-cm", type=float, default=10.0)
    t.add_argument("--mm-per-px", type=float, required=True)
    t.add_argument("--origin", type=float, nargs=2)
    t.add_argument("--min-conf", type=float, default=0.0)
    t.add_argument("--C", type=float, default=1.0)
    t.add_argument("--out", required=True)
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("estimate")
    add_geometry(e)
    e.add_argument("--grid", required=True)
    e.add_argument("--predictions", required=True)
    e.add_argument("--reference", required=True, help="labelled reference_points.csv")
    e.add_argument("--edge-buffer-mm", type=float, default=20.0, help="must equal the value used for sampling")
    e.add_argument("--out", required=True)
    e.set_defaults(fn=cmd_estimate)

    s = sub.add_parser("suggest")
    s.add_argument("--grid", required=True)
    s.add_argument("--features", required=True)
    s.add_argument("--proba", required=True)
    s.add_argument("--classes", required=True)
    s.add_argument("--exclude", nargs="*", help="reference points and already labelled points")
    s.add_argument("--gold-windows", nargs="*")
    s.add_argument("--exclusion-radius-cm", type=float, default=10.0)
    s.add_argument("--min-sep-cm", type=float, default=20.0)
    s.add_argument("--mm-per-px", type=float, required=True)
    s.add_argument("--origin", type=float, nargs=2)
    s.add_argument("-k", type=int, default=100)
    s.add_argument("--explore-frac", type=float, default=0.3)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_suggest)

    m = sub.add_parser("maps")
    m.add_argument("--grid", required=True)
    m.add_argument("--predictions", required=True)
    m.add_argument("--cell-patches", type=int, default=10)
    m.add_argument("--min-patches", type=int, default=20)
    m.add_argument("--object-classes", nargs="*")
    m.add_argument("--min-component-patches", type=int, default=2)
    m.add_argument("--mm-per-px", type=float, required=True)
    m.add_argument("--origin", type=float, nargs=2)
    m.add_argument("--out", required=True)
    m.set_defaults(fn=cmd_maps)

    o = sub.add_parser("objects")
    o.add_argument("--raster", required=True)
    o.add_argument("--grid", required=True)
    o.add_argument("--proba", required=True)
    o.add_argument("--classes", required=True)
    o.add_argument("--object-classes", nargs="+", required=True, help="classes that are discrete organisms")
    o.add_argument("--mm-per-px", type=float, required=True)
    o.add_argument("--origin", type=float, nargs=2)
    o.add_argument("--min-area-cm2", type=float, default=10.0, help="minimum mapping unit")
    o.add_argument("--min-conf", type=float, default=0.5)
    o.add_argument("--tile-px", type=int, default=1024)
    o.add_argument("--margin-px", type=int, default=128, help="must exceed the largest organism radius")
    o.add_argument("--colour-json", help="features .npy.colour.json, to give SAM2 the same normalised colours")
    o.add_argument("--sam2-config")
    o.add_argument("--sam2-checkpoint")
    o.add_argument("--device", default="cpu")
    o.add_argument("--out", required=True, help="GeoJSON")
    o.set_defaults(fn=cmd_objects)

    d = sub.add_parser("detect")
    d.add_argument("--raster", required=True)
    d.add_argument("--grid", required=True)
    d.add_argument("--backend", choices=["yolo", "grounded"], default="yolo")
    d.add_argument("--weights", help="YOLO-seg weights (.pt)")
    d.add_argument("--prompt", action="append", metavar="CLASS=PHRASE", help="grounded backend, repeatable")
    d.add_argument("--sam2-config")
    d.add_argument("--sam2-checkpoint")
    d.add_argument("--classes", nargs="*", help="keep only these detected classes (default: all)")
    d.add_argument("--other-label", default="other")
    d.add_argument("--mm-per-px", type=float, required=True)
    d.add_argument("--origin", type=float, nargs=2)
    d.add_argument("--min-area-cm2", type=float, default=10.0)
    d.add_argument("--min-score", type=float, default=0.25)
    d.add_argument("--min-cover", type=float, default=0.3, help="share of a patch an instance must cover")
    d.add_argument("--tile-px", type=int, default=1024)
    d.add_argument("--margin-px", type=int, default=128)
    d.add_argument("--imgsz", type=int, default=1024)
    d.add_argument("--colour-json")
    d.add_argument("--device", default="cpu")
    d.add_argument("--out", required=True, help="output directory")
    d.set_defaults(fn=cmd_detect)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
