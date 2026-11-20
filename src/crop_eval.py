"""Leave-one-source-photo-out evaluation of crop classifiers.

Only labels seen in >= ``min_groups`` source photos are evaluated (otherwise a held-out
photo would contain a class the model has never seen, or the class could never be tested).
Uncertainty is a cluster bootstrap over source photos (resampling photos, not crops).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
import crop_library as cl  # noqa: E402


def make_model(name: str, seed: int = 0):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                      random_state=seed, n_jobs=-1)
    raise ValueError(name)


def logo_predict(X, y, groups, model_name: str, seed: int = 0):
    """Leave-one-group-out predictions.  Returns y_pred aligned with y (None where untestable)."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    pred = np.full(len(y), None, dtype=object)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        m = make_model(model_name, seed).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def majority_baseline(y, groups):
    """Leave-one-group-out majority class (the floor any feature set must beat)."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    pred = np.empty(len(y), dtype=object)
    for g in np.unique(groups):
        te = groups == g
        vals, cnt = np.unique(y[~te], return_counts=True)
        pred[te] = vals[np.argmax(cnt)]
    return pred


def macro_f1(y, p, labels):
    return f1_score(y, p, labels=labels, average="macro", zero_division=0)


def group_bootstrap(y, p, groups, labels, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in ug}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([idx_by[g] for g in pick])
        present = [lab for lab in labels if (y[idx] == lab).any()]
        vals.append(macro_f1(y[idx], p[idx], present))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def evaluate(X, y, groups, model_names=("logreg", "rf"), min_groups=2, seed=0, n_boot=1000):
    y = np.asarray(y)
    groups = np.asarray(groups)
    per_label_groups = {lab: len(set(groups[y == lab])) for lab in np.unique(y)}
    keep_labels = sorted(k for k, v in per_label_groups.items() if v >= min_groups)
    skipped = sorted(k for k, v in per_label_groups.items() if v < min_groups)
    sel = np.isin(y, keep_labels)
    Xs, ys, gs = X[sel], y[sel], groups[sel]
    results = {}
    base = majority_baseline(ys, gs)
    results["majority"] = {"macro_f1": macro_f1(ys, base, keep_labels),
                           "accuracy": accuracy_score(ys, base)}
    preds = {}
    for name in model_names:
        p = logo_predict(Xs, ys, gs, name, seed)
        ok = np.array([v is not None for v in p])
        lo, hi = group_bootstrap(ys[ok], p[ok].astype(str), gs[ok], keep_labels, n_boot, seed)
        results[name] = {"macro_f1": macro_f1(ys[ok], p[ok].astype(str), keep_labels),
                         "macro_f1_ci95": [lo, hi], "accuracy": accuracy_score(ys[ok], p[ok].astype(str)),
                         "n_tested": int(ok.sum())}
        preds[name] = p
    return {"labels_evaluated": keep_labels, "labels_skipped_lt_min_groups": skipped,
            "n_crops_evaluated": int(sel.sum()), "n_groups": int(len(np.unique(gs))),
            "results": results}, (ys, gs, preds)


def load_features(path: str):
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        fcols = [c for c in rd.fieldnames if c.startswith("feat_")]
        rows = list(rd)
    X = np.array([[float(r[c]) for c in fcols] for r in rows], dtype=float)
    return rows, X


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", required=True, help="CSV with folder,file,w,h,source_photo,feat_* columns")
    ap.add_argument("--schema", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-groups", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)

    rows, X = load_features(a.features)
    schema = cl.load_schema(a.schema)
    table, dropped = cl.build_table(rows, schema)
    keep = {(r["folder"], r["file"]) for r in table}
    sel = np.array([(r["folder"], r["file"]) in keep for r in rows])
    order = {(r["folder"], r["file"]): r for r in table}
    t = [order[(r["folder"], r["file"])] for r, k in zip(rows, sel) if k]
    y = np.array([r["label"] for r in t])
    g = np.array([r["source"] for r in t])
    report, (ys, gs, preds) = evaluate(X[sel], y, g, min_groups=a.min_groups, seed=a.seed, n_boot=a.n_boot)
    report["dropped"] = [{"folder": d["folder"], "file": d["file"], "reason": d["reason"]} for d in dropped]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "crop_baseline_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    # per-class recall for the best model
    best = max(preds, key=lambda k: report["results"][k]["macro_f1"])
    with open(os.path.join(a.out, f"per_class_recall_{best}.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["label", "n_crops", "n_source_photos", "recall"])
        p = preds[best]
        for lab in report["labels_evaluated"]:
            m = (ys == lab) & np.array([v is not None for v in p])
            wr.writerow([lab, int((ys == lab).sum()), len(set(gs[ys == lab])),
                         f"{(p[m] == lab).mean():.3f}" if m.any() else ""])
    print(json.dumps(report["results"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
