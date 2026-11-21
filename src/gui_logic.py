"""Form specifications and the runner behind the GUI (no Tk imports, so it is testable headless).

Each ``Step`` describes the form of one pipeline command.  ``build_argv`` turns form values into the
argv of the real command, and ``tests/test_gui_logic.py`` checks every form against the real argparse
parsers, so the GUI cannot drift away from the CLI.
"""
from __future__ import annotations

import contextlib
import os
import re
import shlex
from dataclasses import dataclass, field
from typing import Callable

import pipeline
import reference_sampling as rs

KINDS = ("file", "save", "dir", "text", "float", "int", "choice", "flag", "xy", "multi", "repeat")


@dataclass(frozen=True)
class Field:
    key: str
    flag: str
    label: str
    kind: str = "text"
    default: str = ""
    required: bool = False
    choices: tuple = ()
    help: str = ""


@dataclass(frozen=True)
class Step:
    name: str
    title: str
    entry: str                       # "pipeline" (sub-command ``cmd``) or "reference"
    cmd: str
    help: str
    fields: tuple = field(default_factory=tuple)


F = Field
MM = F("mm_per_px", "--mm-per-px", "Pixel size (mm/px)", "float", "0.412284", True)
ORIGIN = F("origin", "--origin", "Origin X0 Y0 (world coords of top-left)", "xy",
           help="Only needed with a mask PNG, or to write world coordinates.")
GEOM = (F("raster", "--raster", "Mosaic (GeoTIFF)", "file"),
        F("mask_png", "--mask-png", "Validity mask PNG (optional)", "file"),
        F("full_size", "--full-size", "Full size W H (with mask PNG)", "xy"), ORIGIN, MM)

STEPS = (
    Step("reference", "1. Reference sample", "reference", "",
         "Draw the random reference points and gold windows BEFORE any model exists.",
         (F("raster", "--raster", "Mosaic (GeoTIFF)", "file", required=True),
          F("mask_png", "--mask-png", "Validity mask PNG (optional)", "file"),
          F("full_size", "--full-size", "Full size W H (with mask PNG)", "xy"), ORIGIN, MM,
          F("seed", "--seed", "Random seed (write it in your methods)", "int", "20261119", True),
          F("n_points", "--n-points", "Number of points", "int", "400"),
          F("n_windows", "--n-windows", "Gold windows", "int", "5"),
          F("window_cm", "--window-cm", "Window size (cm)", "float", "50"),
          F("edge_buffer_mm", "--edge-buffer-mm", "Edge buffer (mm)", "float", "20"),
          F("out", "--out", "Output folder", "dir", required=True))),
    Step("grid", "2. Patch grid", "pipeline", "grid", "Cut the valid area into fixed-size patches.",
         GEOM + (F("patch_cm", "--patch-cm", "Patch size (cm)", "float", "5"),
                 F("min_valid_frac", "--min-valid-frac", "Min. valid fraction", "float", "0.98"),
                 F("out", "--out", "Output grid file", "save", required=True))),
    Step("features", "3. Features", "pipeline", "features", "Compute features for every patch (resumable).",
         (F("raster", "--raster", "Mosaic (GeoTIFF)", "file", required=True),
          F("grid", "--grid", "Grid file", "file", required=True),
          F("backbone", "--backbone", "Backbone", "choice", "color", choices=("color", "dinov2_vits14", "dinov2_vitb14")),
          F("device", "--device", "Device", "choice", "cpu", choices=("cpu", "cuda")),
          F("batch_size", "--batch-size", "Batch size", "int", "32"),
          F("colour_normalise", "--colour-normalise", "Per-mosaic colour normalisation", "flag"),
          F("out", "--out", "Output features file (.npy)", "save", required=True))),
    Step("train", "4. Train", "pipeline", "train", "Fit the patch classifier on training labels and predict every patch.",
         (F("grid", "--grid", "Grid file", "file", required=True),
          F("features", "--features", "Features file", "file", required=True),
          F("labels", "--labels", "Training labels CSV (x_px,y_px,label)", "file", required=True),
          F("exclude", "--exclude", "Reference CSVs to keep out of training", "multi"),
          F("exclusion_radius_cm", "--exclusion-radius-cm", "Exclusion radius (cm)", "float", "10"),
          MM, ORIGIN,
          F("min_conf", "--min-conf", "Reject below confidence", "float", "0"),
          F("C", "--C", "Regularisation C", "float", "1"),
          F("out", "--out", "Output folder", "dir", required=True))),
    Step("detect", "4b. Detect (YOLO-seg / Grounding DINO)", "pipeline", "detect",
         "Instance detections of the target organisms become the patch map (everything else = other).",
         (F("raster", "--raster", "Mosaic (GeoTIFF)", "file", required=True),
          F("grid", "--grid", "Grid file", "file", required=True),
          F("backend", "--backend", "Backend", "choice", "yolo", choices=("yolo", "grounded")),
          F("weights", "--weights", "YOLO-seg weights (.pt)", "file"),
          F("prompt", "--prompt", "Grounded prompts: Class=phrase; Class2=phrase2", "repeat"),
          F("sam2_config", "--sam2-config", "SAM2 config", "text"),
          F("sam2_checkpoint", "--sam2-checkpoint", "SAM2 checkpoint", "file"),
          F("classes", "--classes", "Keep only these classes (optional)", "multi"),
          MM, ORIGIN,
          F("min_area_cm2", "--min-area-cm2", "Min. object area (cm2)", "float", "10"),
          F("min_score", "--min-score", "Min. detector score", "float", "0.25"),
          F("min_cover", "--min-cover", "Min. patch cover to assign", "float", "0.3"),
          F("device", "--device", "Device", "choice", "cpu", choices=("cpu", "cuda")),
          F("out", "--out", "Output folder", "dir", required=True))),
    Step("estimate", "5. Estimate cover", "pipeline", "estimate",
         "Design-based cover with 95% intervals from the labelled reference sample.",
         GEOM + (F("grid", "--grid", "Grid file", "file", required=True),
                 F("predictions", "--predictions", "Patch predictions CSV", "file", required=True),
                 F("reference", "--reference", "Labelled reference points CSV", "file", required=True),
                 F("edge_buffer_mm", "--edge-buffer-mm", "Edge buffer (mm) - same as step 1", "float", "20"),
                 F("out", "--out", "Output folder", "dir", required=True))),
    Step("suggest", "6. Suggest labels", "pipeline", "suggest",
         "Pick the most useful patches to label next (training only; never near the reference sample).",
         (F("grid", "--grid", "Grid file", "file", required=True),
          F("features", "--features", "Features file", "file", required=True),
          F("proba", "--proba", "proba.npy", "file", required=True),
          F("classes", "--classes", "classes.json", "file", required=True),
          F("exclude", "--exclude", "Reference + already labelled CSVs", "multi"),
          F("gold_windows", "--gold-windows", "Gold windows CSV", "multi"),
          F("exclusion_radius_cm", "--exclusion-radius-cm", "Exclusion radius (cm)", "float", "10"),
          F("min_sep_cm", "--min-sep-cm", "Min. separation (cm)", "float", "20"),
          MM, ORIGIN,
          F("k", "-k", "How many", "int", "100"),
          F("explore_frac", "--explore-frac", "Random exploration share", "float", "0.3"),
          F("out", "--out", "Output suggestions CSV", "save", required=True))),
)


# form keys that mean the same thing in every step, so one value is shared across forms
SHARED = frozenset({"raster", "mask_png", "full_size", "origin", "mm_per_px", "grid", "features",
                    "predictions", "proba", "classes", "edge_buffer_mm", "gold_windows", "exclude",
                    "reference", "points", "device"})


def propagate(step: Step, values: dict) -> dict:
    """Shared-field values to pre-fill in later steps once ``step`` has produced its outputs."""
    out = str(values.get("out") or "").strip()
    if not out:
        return {}
    j = os.path.join
    if step.name == "reference":
        return {"points": j(out, "reference_points.csv"), "gold_windows": j(out, "gold_windows.csv")}
    if step.name == "grid":
        return {"grid": out}
    if step.name == "features":
        return {"features": out}
    if step.name == "train":
        return {"predictions": j(out, "patch_predictions.csv"), "proba": j(out, "proba.npy"),
                "classes": j(out, "classes.json")}
    if step.name == "detect":
        return {"predictions": j(out, "patch_predictions.csv")}
    return {}


def step_by_name(name: str) -> Step:
    return next(s for s in STEPS if s.name == name)


def _split_paths(text: str) -> list:
    return [t.strip('"').strip("'") for t in shlex.split(text, posix=False)]


def build_argv(step: Step, values: dict) -> list:
    """argv for the real command; empty fields are left out so the CLI defaults apply."""
    argv = [step.cmd] if step.entry == "pipeline" else []
    for f in step.fields:
        v = values.get(f.key)
        if f.kind == "flag":
            if v in (True, 1, "1", "true", "True"):
                argv.append(f.flag)
            continue
        if v is None or str(v).strip() == "":
            continue
        v = str(v).strip()
        if f.kind == "xy":
            argv += [f.flag] + [p for p in re.split(r"[\s,]+", v) if p]
        elif f.kind == "multi":
            argv += [f.flag] + _split_paths(v)
        elif f.kind == "repeat":
            for item in v.split(";"):
                if item.strip():
                    argv += [f.flag, item.strip()]
        else:
            argv += [f.flag, v]
    return argv


def missing_fields(step: Step, values: dict) -> list:
    return [f.label for f in step.fields if f.required and not str(values.get(f.key) or "").strip()]


class _Writer:
    def __init__(self, log: Callable[[str], None]):
        self._log = log

    def write(self, s: str) -> int:
        if s:
            self._log(s)
        return len(s)

    def flush(self) -> None:
        pass


def run_step(step: Step, values: dict, log: Callable[[str], None]) -> tuple:
    """Run one step in-process, streaming output to ``log``.  Returns (ok, message); never raises."""
    missing = missing_fields(step, values)
    if missing:
        return False, "Missing: " + ", ".join(missing)
    argv = build_argv(step, values)
    main = rs.main if step.entry == "reference" else pipeline.main
    w = _Writer(log)
    try:
        with contextlib.redirect_stdout(w), contextlib.redirect_stderr(w):
            main(argv)
    except SystemExit as e:
        if e.code in (None, 0):
            return True, "Done."
        return False, str(e.code) if isinstance(e.code, str) else f"Exited with code {e.code} (see log)."
    except Exception as e:  # noqa: BLE001 - the GUI must report, not crash
        return False, f"{type(e).__name__}: {e}"
    return True, "Done."
