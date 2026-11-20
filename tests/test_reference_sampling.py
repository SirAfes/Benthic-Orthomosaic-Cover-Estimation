import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import reference_sampling as rs  # noqa: E402


def make_mask():
    m = np.zeros((100, 200), dtype=bool)
    m[10:90, 10:190] = True
    m[40:60, 90:110] = False  # hole
    return m


def test_erode_removes_border_and_hole_margin():
    m = make_mask()
    e = rs.erode(m, 3)
    assert not e[10, 50] and e[14, 50]           # outer edge band removed
    assert not e[38, 100] and e[34, 100]         # margin around the hole
    assert not rs.erode(np.ones((5, 5), bool), 1)[0, 0]  # array border invalid


def test_points_are_valid_and_reproducible():
    m = rs.erode(make_mask(), 2)
    a = rs.draw_points(m, (4.0, 4.0), 300, seed=1)
    b = rs.draw_points(m, (4.0, 4.0), 300, seed=1)
    c = rs.draw_points(m, (4.0, 4.0), 300, seed=2)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    cols = (a[:, 0] / 4.0).astype(int)
    rows = (a[:, 1] / 4.0).astype(int)
    assert m[rows, cols].all()


def test_points_are_uniform_over_area():
    m = np.zeros((100, 100), bool)
    m[:, :40] = True      # full-height strip on the left
    m[50:, 40:80] = True  # half-height block on the right (unequal areas)
    pts = rs.draw_points(m, (1.0, 1.0), 20000, seed=3)
    left_share = (pts[:, 0] < 40).mean()
    expected = m[:, :40].sum() / m.sum()
    assert abs(left_share - expected) < 0.02


def test_full_res_rejection_keeps_points_on_valid_pixels():
    m = np.ones((50, 50), bool)
    # full-res truth: right half is actually invalid
    pts = rs.draw_points(m, (2.0, 2.0), 200, seed=4, full_res_valid=lambda x, y: x < 50)
    assert (pts[:, 0] < 50).all()
    with pytest.raises(RuntimeError):
        rs.draw_points(m, (2.0, 2.0), 10, seed=4, full_res_valid=lambda x, y: False, max_rounds=3)


def test_windows_inside_valid_and_disjoint():
    m = rs.erode(make_mask(), 1)
    wins = rs.draw_windows(m, (4.0, 4.0), win_px=80, n=4, seed=5)
    assert len(wins) == 4
    for x0, y0, w, h in wins:
        c0, c1 = x0 // 4, (x0 + w + 3) // 4
        r0, r1 = y0 // 4, (y0 + h + 3) // 4
        assert m[r0:r1, c0:c1].all()
    for i, (ax, ay, aw, ah) in enumerate(wins):
        for bx, by, bw, bh in wins[i + 1:]:
            assert ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay


def test_empty_mask_raises():
    with pytest.raises(ValueError):
        rs.draw_points(np.zeros((5, 5), bool), (1, 1), 3, seed=0)


def test_cli_with_mask_png(tmp_path):
    from PIL import Image

    m = np.zeros((60, 120), dtype=np.uint8)
    m[5:55, 5:115] = 255
    png = tmp_path / "m.png"
    Image.fromarray(m).save(png)
    out = tmp_path / "out"
    rc = rs.main(["--raster", "x.tif", "--mask-png", str(png), "--full-size", "1200", "600",
                  "--origin", "100.0", "50.0", "--mm-per-px", "1.0", "--seed", "7",
                  "--n-points", "50", "--n-windows", "2", "--window-cm", "5",
                  "--edge-buffer-mm", "20", "--out", str(out)])
    assert rc == 0
    import csv
    rows = list(csv.DictReader(open(out / "reference_points.csv")))
    assert len(rows) == 50 and len({r["id"] for r in rows}) == 50
    for r in rows:
        assert 100.0 <= float(r["x_m"]) <= 101.2 and 49.4 <= float(r["y_m"]) <= 50.0
        assert 50 <= float(r["x_px"]) <= 1150 and 50 <= float(r["y_px"]) <= 550
