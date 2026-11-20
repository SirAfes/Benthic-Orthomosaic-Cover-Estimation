"""Design-based percent-cover estimation from a probability reference sample plus a map.

Population: the set of grid patches whose centre lies in the (eroded) valid area, each patch
having equal area.  The reference points are a simple random sample of that area.

Estimators for the cover proportion p_k of class k
--------------------------------------------------
direct          n_k / n from the reference sample alone (Wilson interval).  Ignores the map.
post_stratified sum_h W_h * n_hk / n_h with strata h = MAP class, W_h = map area share.  The
                standard stratified-estimator variance of Olofsson et al. (2013) is used,
                conditional on the realised stratum sizes n_h.  This is the PRIMARY estimator.
ppi             prediction-powered estimate: map share q_k plus the mean point-wise residual
                1[y=k] - 1[yhat=k]; variance from the residuals (Angelopoulos et al. 2023).

Both map-assisted estimators are unbiased whatever the classifier does, provided the reference
sample is a probability sample taken independently of the model and the reference labels were
never used to train it.  A better classifier only makes them tighter.
"""
from __future__ import annotations

import warnings
from typing import Iterable, Sequence

import numpy as np

Z95 = 1.959963984540054


def wilson(k: float, n: int, z: float = Z95) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def _row(p, se, z=Z95):
    p = float(np.clip(p, 0.0, 1.0))
    return {"p": p, "se": float(se), "lo": float(max(0.0, p - z * se)), "hi": float(min(1.0, p + z * se))}


def map_shares(map_labels: Sequence, strata: Iterable | None = None) -> dict:
    labels = np.asarray(map_labels).astype(str)
    names = sorted(set(labels)) if strata is None else list(strata)
    n = len(labels)
    return {h: float((labels == h).sum()) / n for h in names}


def direct_estimate(y_ref: Sequence, classes: Sequence) -> dict:
    y = np.asarray(y_ref).astype(str)
    n = len(y)
    out = {}
    for k in classes:
        c = int((y == k).sum())
        lo, hi = wilson(c, n)
        out[k] = {"p": c / n if n else 0.0, "se": float(np.sqrt((c / n) * (1 - c / n) / n)) if n else 0.0,
                  "lo": lo, "hi": hi}
    return out


def post_stratified_estimate(y_ref: Sequence, yhat_ref: Sequence, map_labels: Sequence,
                             classes: Sequence | None = None) -> dict:
    """Primary estimator.  Returns {'estimates': {class: {...}}, 'diagnostics': {...}}."""
    y = np.asarray(y_ref).astype(str)
    yh = np.asarray(yhat_ref).astype(str)
    if len(y) != len(yh):
        raise ValueError("y_ref and yhat_ref must have equal length.")
    W = map_shares(map_labels)
    classes = list(classes) if classes is not None else sorted(set(y) | set(W))
    unsampled = [h for h, w in W.items() if w > 0 and not (yh == h).any()]
    small = [h for h, w in W.items() if w > 0 and 0 < (yh == h).sum() < 2]
    sampled_mass = sum(w for h, w in W.items() if h not in unsampled)
    if unsampled:
        warnings.warn(f"Map classes with no reference points ({unsampled}) cover "
                      f"{1 - sampled_mass:.1%} of the area; estimates are renormalised over the sampled strata "
                      "and are biased if those strata differ. Add reference points there.", stacklevel=2)
    est = {}
    for k in classes:
        p = var = 0.0
        for h, w in W.items():
            if h in unsampled or w == 0:
                continue
            in_h = yh == h
            nh = int(in_h.sum())
            nhk = int((in_h & (y == k)).sum())
            w_adj = w / sampled_mass
            p += w_adj * nhk / nh
            if nh >= 2:
                phk = nhk / nh
                var += w_adj ** 2 * phk * (1 - phk) / (nh - 1)
            else:                                   # single point: smoothed, conservative
                pt = (nhk + 0.5) / (nh + 1.0)
                var += w_adj ** 2 * pt * (1 - pt) / (nh + 1.0)
        est[k] = _row(p, np.sqrt(var))
    diag = {"n_reference": int(len(y)), "map_shares": W, "unsampled_strata": unsampled,
            "single_point_strata": small, "unsampled_area_share": float(1 - sampled_mass)}
    return {"estimates": est, "diagnostics": diag}


def ppi_estimate(y_ref: Sequence, yhat_ref: Sequence, map_labels: Sequence,
                 classes: Sequence | None = None) -> dict:
    y = np.asarray(y_ref).astype(str)
    yh = np.asarray(yhat_ref).astype(str)
    q = map_shares(map_labels)
    classes = list(classes) if classes is not None else sorted(set(y) | set(q))
    n = len(y)
    est = {}
    for k in classes:
        d = (y == k).astype(float) - (yh == k).astype(float)
        est[k] = _row(q.get(k, 0.0) + d.mean(), d.std(ddof=1) / np.sqrt(n) if n > 1 else 1.0)
    return {"estimates": est, "diagnostics": {"n_reference": n, "map_shares": q}}


def naive_map_estimate(map_labels: Sequence, classes: Sequence | None = None) -> dict:
    """Raw share of each class in the map; shown only to expose its bias, never as a result."""
    q = map_shares(map_labels)
    classes = list(classes) if classes is not None else sorted(q)
    return {k: {"p": q.get(k, 0.0)} for k in classes}


def apply_reference_policy(labels: Sequence, unscorable: str = "UNSCORABLE") -> tuple[np.ndarray, int]:
    """Drop points that could not be scored.  Cover is then 'of the scorable area'; report the count."""
    y = np.asarray(labels).astype(str)
    keep = y != unscorable
    return keep, int((~keep).sum())
