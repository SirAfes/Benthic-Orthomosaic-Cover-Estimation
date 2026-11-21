import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import labeltool as lt  # noqa: E402
import tiling  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def make_points(tmp_path, n=5):
    p = str(tmp_path / "points.csv")
    with open(p, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["id", "x_px", "y_px"])
        for i in range(n):
            wr.writerow([f"P{i}", 50 + 10 * i, 60 + 5 * i])
    return p


def read_out(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_session_labels_advance_and_persist_each_answer(tmp_path):
    pts, out = make_points(tmp_path), str(tmp_path / "labelled.csv")
    s = lt.LabelSession(pts, out)
    assert s.current()["id"] == "P0" and s.progress() == (0, 5)
    s.set_label("sand")
    assert s.current()["id"] == "P1" and s.progress() == (1, 5)
    rows = read_out(out)
    assert len(rows) == 5 and rows[0]["label"] == "sand" and rows[1]["label"] == ""


def test_session_resumes_at_first_unlabelled_point(tmp_path):
    pts, out = make_points(tmp_path), str(tmp_path / "labelled.csv")
    s = lt.LabelSession(pts, out)
    s.set_label("sand")
    s.set_label("rock")
    s2 = lt.LabelSession(pts, out)
    assert s2.current()["id"] == "P2" and s2.progress() == (2, 5)


def test_undo_restores_previous_point_and_clears_its_label(tmp_path):
    pts, out = make_points(tmp_path), str(tmp_path / "labelled.csv")
    s = lt.LabelSession(pts, out)
    s.set_label("sand")
    s.set_label("rock")
    s.undo()
    assert s.current()["id"] == "P1" and s.progress() == (1, 5)
    assert read_out(out)[1]["label"] == ""


def test_skip_leaves_blank_and_unscorable_counts_as_answered(tmp_path):
    pts, out = make_points(tmp_path), str(tmp_path / "labelled.csv")
    s = lt.LabelSession(pts, out)
    s.skip()
    s.unscorable()
    assert s.progress() == (1, 5) and read_out(out)[1]["label"] == "UNSCORABLE"
    assert s.current()["id"] == "P2"


def test_finished_session_reports_done_and_ignores_blank_labels(tmp_path):
    pts, out = make_points(tmp_path, n=2), str(tmp_path / "labelled.csv")
    s = lt.LabelSession(pts, out)
    s.set_label("a")
    s.set_label("b")
    assert s.done() and s.current() is None
    s.set_label("   ")                                       # no-op once done / blank
    assert s.progress() == (2, 2)


def test_labels_with_commas_and_unicode_roundtrip(tmp_path):
    pts, out = make_points(tmp_path), str(tmp_path / "labelled.csv")
    s = lt.LabelSession(pts, out)
    s.set_label("Anemone, orange (Corynactis) é")
    assert lt.LabelSession(pts, out).labels()["P0"] == "Anemone, orange (Corynactis) é"


def test_existing_labels_are_matched_by_id_not_row_order(tmp_path):
    pts, out = make_points(tmp_path), str(tmp_path / "labelled.csv")
    with open(out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["id", "x_px", "y_px", "label"])
        wr.writerow(["P3", 80, 75, "rock"])
    s = lt.LabelSession(pts, out)
    assert s.labels()["P3"] == "rock" and s.progress() == (1, 5) and s.current()["id"] == "P0"


def test_crop_is_centred_on_the_point_and_padded_at_the_border():
    yy, xx = np.mgrid[0:100, 0:100]
    img = np.dstack([xx, yy, xx + yy]).astype(np.uint8)
    reader = tiling.ArrayReader(img)
    c = lt.crop_around(reader, 50, 40, 21)
    assert c.shape == (21, 21, 3) and tuple(c[10, 10]) == (50, 40, 90)
    edge = lt.crop_around(reader, 2, 2, 21)
    assert edge.shape == (21, 21, 3) and tuple(edge[10, 10]) == (2, 2, 4) and edge[0, 0].sum() == 0


def test_classes_from_schema_reads_canonical_labels():
    classes = lt.classes_from_schema(os.path.join(ROOT, "docs", "label_schema_v1_draft.csv"))
    assert "Actinothoe sphyrodeta" in classes and len(classes) == len(set(classes)) and "" not in classes
