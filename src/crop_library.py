"""Crop library: raw folders -> canonical labels, with source-photo groups.

Crops cut from the same photograph are not independent (same light, water, camera,
neighbouring organisms).  Every split in this project is therefore made by source photo.
"""
from __future__ import annotations

import csv
from pathlib import Path


def source_id(filename: str) -> str:
    """``0126_10_ADS_0926_3.png`` -> ``0126_10_ADS_0926`` (drop the per-crop index)."""
    return "_".join(Path(filename).stem.split("_")[:-1])


def load_schema(path: str) -> dict[str, dict]:
    """raw_folder -> {label, group, layer}.  Rows without a raw folder are mosaic-only classes."""
    out: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            raw = r["raw_folder"]
            if raw in ("", "(none)"):
                continue
            out[raw] = {"label": r["canonical_label"], "group": r["morphotype_group"], "layer": r["layer"]}
    return out


def build_table(rows: list[dict], schema: dict[str, dict], min_side: int = 32,
                drop_layers: tuple[str, ...] = ("exclude",)) -> tuple[list[dict], list[dict]]:
    """Attach labels/groups.  Returns (kept, dropped-with-reason); nothing is dropped silently."""
    kept, dropped = [], []
    for r in rows:
        folder = r["folder"]
        if folder not in schema:
            dropped.append({**r, "reason": "folder not in schema"})
            continue
        sc = schema[folder]
        w, h = int(r["w"]), int(r["h"])
        if min(w, h) < min_side:
            dropped.append({**r, "reason": f"min side < {min_side}px"})
            continue
        if sc["layer"] in drop_layers:
            dropped.append({**r, "reason": f"layer {sc['layer']}"})
            continue
        kept.append({**r, "label": sc["label"], "morphotype": sc["group"], "layer": sc["layer"],
                     "source": r.get("source_photo") or source_id(r["file"])})
    return kept, dropped


def groups_per_label(table: list[dict]) -> dict[str, int]:
    seen: dict[str, set] = {}
    for r in table:
        seen.setdefault(r["label"], set()).add(r["source"])
    return {k: len(v) for k, v in seen.items()}
