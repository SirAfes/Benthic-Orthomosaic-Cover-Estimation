"""Feature-space classifier for patches, with spatial-block CV and a reject option."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REJECT = "Unclassified"


class PatchClassifier:
    def __init__(self, C: float = 1.0, class_weight="balanced", max_iter: int = 3000):
        self.model = make_pipeline(StandardScaler(),
                                   LogisticRegression(C=C, class_weight=class_weight, max_iter=max_iter))
        self.classes_: np.ndarray | None = None

    def fit(self, X, y) -> "PatchClassifier":
        y = np.asarray(y).astype(str)
        if len(np.unique(y)) < 2:
            raise ValueError("Need at least two classes to train.")
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X, min_conf: float = 0.0) -> np.ndarray:
        return labels_from_proba(self.predict_proba(X), self.classes_, min_conf)


def labels_from_proba(proba: np.ndarray, classes, min_conf: float = 0.0) -> np.ndarray:
    classes = np.asarray(classes).astype(str)
    out = classes[np.argmax(proba, axis=1)].astype(object)
    if min_conf > 0:
        out[proba.max(axis=1) < min_conf] = REJECT
    return out.astype(str)


def spatial_block_ids(xy_px: np.ndarray, block_px: float) -> np.ndarray:
    """Integer id of the square block containing each point (for grouped CV)."""
    xy = np.asarray(xy_px, dtype=float)
    ix = np.floor(xy[:, 0] / block_px).astype(np.int64)
    iy = np.floor(xy[:, 1] / block_px).astype(np.int64)
    return ix * 1_000_003 + iy


def blocked_cv_predict(X, y, blocks, n_splits: int = 5, **kw) -> np.ndarray:
    """Out-of-fold labels with whole spatial blocks held out.  Labels unseen in a training fold
    cannot be predicted and come back as REJECT."""
    X = np.asarray(X)
    y = np.asarray(y).astype(str)
    blocks = np.asarray(blocks)
    n_splits = min(n_splits, len(np.unique(blocks)))
    pred = np.full(len(y), REJECT, dtype=object)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, blocks):
        if len(np.unique(y[tr])) < 2:
            continue
        m = PatchClassifier(**kw).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred.astype(str)
