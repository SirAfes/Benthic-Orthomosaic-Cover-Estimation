# Benthic Orthomosaic Cover Estimation

Tools and protocol for estimating **unbiased percent cover** of benthic communities from
underwater orthomosaics, starting from a small number of expert labels.

Status: pipeline implemented and tested on synthetic data; not yet run on the real mosaics.

## What is here

| Path | Purpose |
|---|---|
| `src/reference_sampling.py` | Draws a seeded simple-random reference sample (points + dense "gold" windows) over the valid area of a mosaic, before any model exists. |
| `docs/label_schema_v1_draft.csv` | Draft mapping from raw crop folders to a merged, two-layer label schema (A = matrix/cover classes, B = discrete organisms). |
| `docs/LABELLING_PROTOCOL.md` | Blind labelling protocol for the reference sample. |
| `src/features.py` | Baseline colour/texture features (numpy only) and the interface for learned backbone features. |
| `src/crop_library.py`, `src/crop_eval.py` | Crop library to canonical labels; leave-one-source-photo-out evaluation with cluster-bootstrap CIs. |
| `scripts/extract_crop_features.py` | Feature extraction for a crop folder tree (numpy + Pillow). |
| `docs/BASELINE_CROPS.md` | First baseline result on the crop library. |
| `src/tiling.py`, `src/colour.py`, `src/patch_features.py` | Fixed-physical-size patch grid, per-mosaic colour normalisation, resumable feature extraction. |
| `src/classifier.py`, `src/estimation.py` | Patch classifier with reject option; design-based cover estimators (post-stratified primary, PPI, direct) with intervals. |
| `src/active_learning.py`, `src/cover_maps.py` | Next-label suggestions that respect the reference sample; density cells and object components. |
| `src/objects.py` | Outlines of discrete organisms: SAM2 candidate masks, class by patch-probability vote, minimum mapping unit, non-maximum suppression, GeoJSON polygons. |
| `src/pipeline.py` | CLI: `grid`, `features`, `train`, `estimate`, `suggest`, `maps`, `objects`. |
| `docs/WORKFLOW.md` | Step-by-step commands. |
| `tests/` | Unit, simulation (estimator coverage) and synthetic end-to-end tests. |

## Design principles

1. The reference sample is a probability sample, drawn first, labelled blind, and never used for training.
2. Cover estimates are corrected with design-based estimators and reported with confidence intervals.
3. Actively selected labels are used for training only, never for accuracy or cover estimation.
4. Validation of the model is spatially blocked; validation of the map is the probability sample.

## Usage

```bash
pip install -r requirements.txt
python src/reference_sampling.py --raster Tuesday.tif \
    --mask-png tuesday_alpha.png --full-size 62369 21529 --origin X0 Y0 \
    --mm-per-px 0.412284 --seed 20261119 --out out/tuesday
```

With rasterio installed, omit `--mask-png` and the alpha band is read directly (decimated).

## Related

* [Spectral_Benchmark](https://github.com/SirAfes/Spectral_Benchmark): spatially grouped nested cross-validation benchmark for pixel classifiers.

## License

MIT, see `LICENSE`.
