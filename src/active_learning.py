"""Suggest which patches to label next (training labels only, never for estimation).

Selection mixes three things, because uncertainty alone misses confident errors and
over-samples the majority class (iSAGE, DCAU-AL):

* uncertainty (small margin between the two best classes), re-weighted so that patches
  predicted as rare classes are favoured;
* diversity: at most one pick per feature-space cluster;
* exploration: a fixed fraction drawn uniformly at random from the eligible pool.

Patches near the frozen reference sample, the gold windows, or already labelled patches are
excluded, so training labels never leak into the validation/estimation sample.  Picks are also
kept at least ``min_sep_px`` apart.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans


def margin_uncertainty(proba: np.ndarray) -> np.ndarray:
    part = np.sort(proba, axis=1)
    return 1.0 - (part[:, -1] - part[:, -2])


def class_weights_from_predictions(proba: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Weight each patch by (1 / predicted-class share)^alpha, normalised to mean 1."""
    pred = proba.argmax(1)
    share = np.bincount(pred, minlength=proba.shape[1]) / len(pred)
    w = (1.0 / np.maximum(share[pred], 1e-6)) ** alpha
    return w / w.mean()


def eligible_mask(xy_px: np.ndarray, exclude_xy_px: np.ndarray | None, exclude_radius_px: float,
                  exclude_windows: list[tuple[int, int, int, int]] | None = None,
                  margin_px: float = 0.0) -> np.ndarray:
    ok = np.ones(len(xy_px), dtype=bool)
    if exclude_xy_px is not None and len(exclude_xy_px):
        d, _ = cKDTree(np.asarray(exclude_xy_px, float)).query(xy_px, k=1)
        ok &= d > exclude_radius_px
    for x0, y0, w, h in exclude_windows or []:
        inside = ((xy_px[:, 0] >= x0 - margin_px) & (xy_px[:, 0] < x0 + w + margin_px)
                  & (xy_px[:, 1] >= y0 - margin_px) & (xy_px[:, 1] < y0 + h + margin_px))
        ok &= ~inside
    return ok


def suggest(proba: np.ndarray, feats: np.ndarray, xy_px: np.ndarray, k: int, eligible: np.ndarray,
            min_sep_px: float = 0.0, explore_frac: float = 0.3, rare_alpha: float = 0.5,
            n_clusters: int | None = None, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, reason) with reason in {'uncertain', 'explore'}."""
    rng = np.random.default_rng(seed)
    elig = np.flatnonzero(eligible)
    if len(elig) == 0:
        return np.array([], dtype=int), np.array([], dtype=object)
    k = min(k, len(elig))
    n_explore = int(round(k * explore_frac))
    n_uncert = k - n_explore

    score = margin_uncertainty(proba) * class_weights_from_predictions(proba, rare_alpha)
    chosen: list[int] = []
    reasons: list[str] = []
    placed: list[np.ndarray] = []

    def far_enough(i):
        return min_sep_px <= 0 or not placed or np.min(np.hypot(*(np.array(placed) - xy_px[i]).T)) >= min_sep_px

    # 1) uncertain + diverse: best-scoring patch of each feature cluster, highest scores first
    if n_uncert > 0:
        nc = n_clusters or max(n_uncert, 2)
        nc = min(nc, len(elig))
        km = MiniBatchKMeans(n_clusters=nc, random_state=seed, n_init=3, batch_size=1024)
        lab = km.fit_predict(feats[elig])
        best_per_cluster = []
        for c in range(nc):
            members = elig[lab == c]
            if len(members):
                best_per_cluster.append(members[np.argmax(score[members])])
        for i in sorted(best_per_cluster, key=lambda j: -score[j]):
            if len(chosen) >= n_uncert:
                break
            if far_enough(i):
                chosen.append(int(i))
                reasons.append("uncertain")
                placed.append(xy_px[i])
        if len(chosen) < n_uncert:                      # top up from the global ranking
            for i in elig[np.argsort(-score[elig])]:
                if len(chosen) >= n_uncert:
                    break
                if int(i) not in chosen and far_enough(i):
                    chosen.append(int(i))
                    reasons.append("uncertain")
                    placed.append(xy_px[i])
    # 2) exploration: uniform random from what is left
    pool = rng.permutation(elig)
    for i in pool:
        if len(chosen) >= k:
            break
        if int(i) not in chosen and far_enough(i):
            chosen.append(int(i))
            reasons.append("explore")
            placed.append(xy_px[i])
    return np.array(chosen, dtype=int), np.array(reasons, dtype=object)
