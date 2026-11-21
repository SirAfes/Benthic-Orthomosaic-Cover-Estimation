"""Logic of the blind point-labelling tool (no GUI code, so it is testable without a display).

The tool shows the mosaic around each reference point and nothing else: no model output, no
predictions.  Every answer is written to disk immediately, so a crash or a closed window loses nothing.
"""
from __future__ import annotations

import csv
import os

import numpy as np

UNSCORABLE = "UNSCORABLE"


def classes_from_schema(path: str) -> list:
    """Unique ``canonical_label`` values of a label-schema CSV, in file order."""
    seen, out = set(), []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            c = (r.get("canonical_label") or "").strip()
            if c and c not in seen:
                seen.add(c)
                out.append(c)
    return out


def crop_around(reader, x: float, y: float, size: int) -> np.ndarray:
    """size x size x 3 uint8 window centred on (x, y) in px; zero-padded where it leaves the mosaic."""
    half = size // 2
    x0, y0 = int(round(x)) - half, int(round(y)) - half
    out = np.zeros((size, size, 3), dtype=np.uint8)
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(reader.width, x0 + size), min(reader.height, y0 + size)
    if cx1 <= cx0 or cy1 <= cy0:
        return out
    win = np.asarray(reader.read(cx0, cy0, cx1 - cx0, cy1 - cy0))
    if win.ndim == 2:
        win = np.dstack([win] * 3)
    out[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = win[..., :3]
    return out


class LabelSession:
    """Sequential labelling of the points of a CSV (id, x_px, y_px), saved after every answer."""

    def __init__(self, points_csv: str, out_csv: str):
        self.out_csv = out_csv
        self._pts = []
        with open(points_csv, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                self._pts.append({"id": r["id"], "x_px": r["x_px"], "y_px": r["y_px"]})
        self._labels = {p["id"]: "" for p in self._pts}
        if os.path.exists(out_csv):
            with open(out_csv, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    if r.get("id") in self._labels:
                        self._labels[r["id"]] = (r.get("label") or "").strip()
        self._history: list = []
        self._i = self._first_unlabelled(0)

    # ---------------------------------------------------------------- queries
    def labels(self) -> dict:
        return dict(self._labels)

    def progress(self) -> tuple:
        return sum(1 for v in self._labels.values() if v), len(self._pts)

    def done(self) -> bool:
        return self._i is None

    def current(self):
        if self._i is None:
            return None
        p = self._pts[self._i]
        return {"id": p["id"], "x_px": float(p["x_px"]), "y_px": float(p["y_px"]), "index": self._i,
                "label": self._labels[p["id"]]}

    # ---------------------------------------------------------------- moves
    def _first_unlabelled(self, start: int):
        n = len(self._pts)
        for k in range(n):
            j = (start + k) % n
            if not self._labels[self._pts[j]["id"]]:
                return j
        return None

    def set_label(self, label: str) -> None:
        label = (label or "").strip()
        if self._i is None or not label:
            return
        self._labels[self._pts[self._i]["id"]] = label
        self._history.append(self._i)
        self.save()
        self._i = self._first_unlabelled(self._i + 1)

    def unscorable(self) -> None:
        self.set_label(UNSCORABLE)

    def skip(self) -> None:
        if self._i is None:
            return
        nxt = self._first_unlabelled(self._i + 1)
        self._i = nxt if nxt is not None else self._i

    def undo(self) -> None:
        if not self._history:
            return
        self._i = self._history.pop()
        self._labels[self._pts[self._i]["id"]] = ""
        self.save()

    # ---------------------------------------------------------------- disk
    def save(self) -> None:
        tmp = self.out_csv + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["id", "x_px", "y_px", "label"])
            for p in self._pts:
                wr.writerow([p["id"], p["x_px"], p["y_px"], self._labels[p["id"]]])
        os.replace(tmp, self.out_csv)
