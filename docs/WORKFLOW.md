# Workflow (Tuesday mosaic; Wednesday after its TIFF is fixed)

All commands run from the repository root. `X0 Y0` are the top-left world coordinates
(`tuesday_geo.json`), `0.412284` is the Tuesday mm/px. Pre-compute `tuesday_alpha_f3.png`
(an overview alpha mask) or give `--raster` if rasterio is installed.

## 0. Reference sample (done once, before any model; labelled blind)
```bash
python src/reference_sampling.py --raster Tuesday.tif --mask-png tuesday_alpha_f3.png \
  --full-size 62369 21529 --origin X0 Y0 --mm-per-px 0.412284 --seed 20261119 \
  --n-points 400 --n-windows 5 --window-cm 50 --edge-buffer-mm 20 --out out/tuesday
```
Label `reference_points.csv` (see `LABELLING_PROTOCOL.md`). These labels are used ONLY by `estimate`.

## 1. Patch grid and features
```bash
G="--mask-png tuesday_alpha_f3.png --full-size 62369 21529 --origin X0 Y0 --mm-per-px 0.412284"
python src/pipeline.py grid $G --patch-cm 5 --out out/tuesday/grid.npz
python src/pipeline.py features --raster Tuesday.tif --grid out/tuesday/grid.npz \
  --backbone color --colour-normalise --out out/tuesday/feats_color.npy        # baseline
python src/pipeline.py features --raster Tuesday.tif --grid out/tuesday/grid.npz \
  --backbone dinov2_vits14 --device cuda --colour-normalise --out out/tuesday/feats_dino.npy
```
`features` is resumable (chunks in `<out>.chunks/`). DINOv2 needs `torch` and network access to
fetch the weights (torch.hub) on the machine that runs it.

## 2. Training labels (separate from the reference sample)
Make `train_labels.csv` with columns `x_px,y_px,label` (QGIS export of the points you label for training).
```bash
python src/pipeline.py train --grid out/tuesday/grid.npz --features out/tuesday/feats_dino.npy \
  --labels train_labels.csv --exclude out/tuesday/reference_points.csv \
  --mm-per-px 0.412284 --origin X0 Y0 --out out/tuesday/model_r0
```
Training patches within `--exclusion-radius-cm` of any reference point are dropped and counted
in `train_report.json`.

## 3. Cover estimate with intervals (uses the labelled reference sample)
```bash
python src/pipeline.py estimate $G --grid out/tuesday/grid.npz \
  --predictions out/tuesday/model_r0/patch_predictions.csv \
  --reference out/tuesday/reference_points.csv --edge-buffer-mm 20 --out out/tuesday/est_r0
```
`cover_estimates.csv`: primary = `post_*` (post-stratified, design-based). `direct_*` ignores the map,
`ppi_*` is the prediction-powered alternative, `naive_map_p` is shown only to expose the map's bias.
`estimate_report.json` lists unscorable/unlabelled points and any map class that has no reference point.

## 4. Next labels (active learning) and repeat
```bash
python src/pipeline.py suggest --grid out/tuesday/grid.npz --features out/tuesday/feats_dino.npy \
  --proba out/tuesday/model_r0/proba.npy --classes out/tuesday/model_r0/classes.json \
  --exclude out/tuesday/reference_points.csv train_labels.csv \
  --gold-windows out/tuesday/gold_windows.csv --mm-per-px 0.412284 --origin X0 Y0 \
  -k 100 --out out/tuesday/suggest_r1/suggestions.csv
```
Label `suggestions.csv`, append to `train_labels.csv`, rerun step 2. Suggested patches are never
within the exclusion radius of the reference sample or the gold windows.

## 5. Maps
```bash
python src/pipeline.py maps --grid out/tuesday/grid.npz --predictions out/tuesday/model_r0/patch_predictions.csv \
  --cell-patches 10 --object-classes "Corynactis viridis" --mm-per-px 0.412284 --origin X0 Y0 --out out/tuesday/maps_r0
```
Density cells are for visualisation; report mosaic-level cover from step 3.

## Status
Unit and synthetic end-to-end tests pass (including a coverage simulation of the estimators).
Not yet run on the real mosaics, with rasterio reading, or with the DINOv2 backbone.
