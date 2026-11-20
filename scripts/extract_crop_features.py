"""Extract baseline features for every PNG crop under ROOT (read-only on the crops).

Needs only numpy + Pillow.  Usage:
    python extract_crop_features.py ROOT OUT.csv
Output columns: folder,file,source_photo,w,h,feat_*
"""
import csv
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import ColorTextureFeatures  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
MAX_SIDE = 128


def source_id(fname):
    return "_".join(os.path.splitext(fname)[0].split("_")[:-1])


def main(root, out_csv):
    fx = ColorTextureFeatures()
    rows, skipped = [], []
    for folder in sorted(os.listdir(root)):
        fp = os.path.join(root, folder)
        if not os.path.isdir(fp):
            continue
        for f in sorted(os.listdir(fp)):
            if not f.lower().endswith(".png"):
                continue
            with Image.open(os.path.join(fp, f)) as im:
                w, h = im.size
                if min(w, h) < 8:
                    skipped.append((folder, f, w, h))
                    rows.append([folder, f, source_id(f), w, h] + [""] * len(fx.names))
                    continue
                im = im.convert("RGB")
                im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
                feats = fx(np.asarray(im))
            rows.append([folder, f, source_id(f), w, h] + [f"{v:.6g}" for v in feats])
    with open(out_csv, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["folder", "file", "source_photo", "w", "h"] + [f"feat_{n}" for n in fx.names])
        wr.writerows(r for r in rows if r[5] != "")
    print(len(rows) - len(skipped), "crops written;", len(skipped), "skipped (<8 px):", skipped)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
