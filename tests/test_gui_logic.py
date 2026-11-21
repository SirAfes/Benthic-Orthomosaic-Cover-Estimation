import argparse
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import gui_logic as gl  # noqa: E402
import pipeline  # noqa: E402
import reference_sampling as rs  # noqa: E402


def parser_for(step):
    if step.entry == "reference":
        return rs.build_parser()
    sub = next(a for a in pipeline.build_parser()._actions if isinstance(a, argparse._SubParsersAction))
    return sub.choices[step.cmd]


def sample_values(step):
    vals = {}
    for f in step.fields:
        if not f.required:
            continue
        vals[f.key] = {"float": "1.5", "int": "3", "xy": "1 2", "choice": f.choices[0] if f.choices else ""}.get(f.kind, "x")
    return vals


@pytest.mark.parametrize("step", gl.STEPS, ids=lambda s: s.name)
def test_every_step_builds_an_argv_the_real_parser_accepts(step):
    argv = gl.build_argv(step, sample_values(step))
    parser_for(step).parse_args(argv[1:] if step.entry == "pipeline" else argv)


@pytest.mark.parametrize("step", gl.STEPS, ids=lambda s: s.name)
def test_form_covers_all_required_options_and_names_no_unknown_flag(step):
    actions = parser_for(step)._actions
    required = {a.option_strings[0] for a in actions if a.option_strings and a.required}
    known = {o for a in actions for o in a.option_strings}
    flags = {f.flag for f in step.fields}
    assert required <= flags, f"form misses required {required - flags}"
    assert flags <= known, f"form uses unknown {flags - known}"


def test_build_argv_skips_empty_and_handles_flags_pairs_and_multi_values():
    step = next(s for s in gl.STEPS if s.name == "train")
    argv = gl.build_argv(step, {"grid": "g.npz", "features": "f.npy", "labels": "t.csv",
                                "exclude": '"C:\\my dir\\ref.csv" other.csv', "mm_per_px": "0.4",
                                "origin": "10, 20", "out": "m", "C": ""})
    assert argv[0] == "train"
    assert argv[argv.index("--exclude") + 1:argv.index("--exclude") + 3] == ["C:\\my dir\\ref.csv", "other.csv"]
    assert argv[argv.index("--origin") + 1:argv.index("--origin") + 3] == ["10", "20"]
    assert "--C" not in argv
    feat = next(s for s in gl.STEPS if s.name == "features")
    assert "--colour-normalise" in gl.build_argv(feat, {"raster": "r", "grid": "g", "out": "o", "colour_normalise": True})
    assert "--colour-normalise" not in gl.build_argv(feat, {"raster": "r", "grid": "g", "out": "o", "colour_normalise": False})


def test_missing_required_fields_are_listed_by_label():
    step = next(s for s in gl.STEPS if s.name == "grid")
    missing = gl.missing_fields(step, {"mm_per_px": "0.4"})
    assert "Output grid file" in missing and "Pixel size (mm/px)" not in missing


def test_run_step_streams_output_and_reports_success(tmp_path):
    low = np.ones((20, 30), dtype=bool)
    Image.fromarray((low * 255).astype("uint8")).save(tmp_path / "mask.png")
    step = next(s for s in gl.STEPS if s.name == "grid")
    lines = []
    ok, msg = gl.run_step(step, {"mask_png": str(tmp_path / "mask.png"), "full_size": "300 200", "origin": "0 0",
                                 "mm_per_px": "1.0", "patch_cm": "2", "out": str(tmp_path / "g.npz")}, lines.append)
    assert ok and os.path.exists(tmp_path / "g.npz") and "patch" in "".join(lines)


def test_run_step_reports_errors_instead_of_raising(tmp_path):
    step = next(s for s in gl.STEPS if s.name == "estimate")
    ok, msg = gl.run_step(step, {"mask_png": "m.png", "full_size": "10 10", "origin": "0 0", "mm_per_px": "1",
                                 "grid": str(tmp_path / "nope.npz"), "predictions": "p.csv", "reference": "r.csv",
                                 "out": str(tmp_path / "o")}, lambda s: None)
    assert not ok and msg


def test_run_step_reports_missing_input_files():
    step = next(s for s in gl.STEPS if s.name == "detect")
    ok, msg = gl.run_step(step, {"raster": "r", "grid": "g", "mm_per_px": "1", "out": "o", "backend": "yolo"},
                          lambda s: None)
    assert not ok and msg


def test_run_step_refuses_to_start_with_missing_required_fields():
    step = next(s for s in gl.STEPS if s.name == "grid")
    lines = []
    ok, msg = gl.run_step(step, {"mm_per_px": "1"}, lines.append)
    assert not ok and "Output grid file" in msg and lines == []


def test_propagate_prefills_inputs_of_later_steps_from_outputs():
    j = os.path.join
    assert gl.propagate(gl.step_by_name("grid"), {"out": "g.npz"}) == {"grid": "g.npz"}
    assert gl.propagate(gl.step_by_name("features"), {"out": "f.npy"}) == {"features": "f.npy"}
    assert gl.propagate(gl.step_by_name("train"), {"out": "m"}) == {
        "predictions": j("m", "patch_predictions.csv"), "proba": j("m", "proba.npy"), "classes": j("m", "classes.json")}
    assert gl.propagate(gl.step_by_name("detect"), {"out": "d"}) == {"predictions": j("d", "patch_predictions.csv")}
    assert gl.propagate(gl.step_by_name("reference"), {"out": "r"}) == {
        "points": j("r", "reference_points.csv"), "gold_windows": j("r", "gold_windows.csv")}
    assert gl.propagate(gl.step_by_name("estimate"), {"out": "e"}) == {}
    assert gl.propagate(gl.step_by_name("grid"), {"out": ""}) == {}


def test_every_propagated_key_is_a_shared_form_field():
    keys = {f.key for s in gl.STEPS for f in s.fields}
    for name, out in (("grid", "g"), ("features", "f"), ("train", "m"), ("detect", "d")):
        for k in gl.propagate(gl.step_by_name(name), {"out": out}):
            assert k in keys and k in gl.SHARED
