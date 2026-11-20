"""Hand-crafted colour/texture features (numpy only).

This is the *baseline* feature set that every learned feature set has to beat
(hypothesis H1).  It is deliberately simple, deterministic and dependency-free so
it can run anywhere, including machines without PyTorch.

``TorchBackbone`` is the interface for the learned features of method M.  It is
imported lazily and is not exercised by the unit tests in environments without torch.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-9


def _hsv(rgb01: np.ndarray):
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    mx = rgb01.max(-1)
    d = mx - rgb01.min(-1)
    nz = d > _EPS
    dd = np.where(nz, d, 1.0)
    is_r = nz & (mx == r)
    is_g = nz & ~is_r & (mx == g)
    is_b = nz & ~is_r & ~is_g
    h6 = np.where(is_r, ((g - b) / dd) % 6.0, 0.0)
    h6 = np.where(is_g, (b - r) / dd + 2.0, h6)
    h6 = np.where(is_b, (r - g) / dd + 4.0, h6)
    h = (h6 / 6.0) % 1.0
    s = np.where(mx > _EPS, d / np.where(mx > _EPS, mx, 1.0), 0.0)
    return h, s, mx


def _norm_hist(values, bins, rng, weights=None):
    hist, _ = np.histogram(values, bins=bins, range=rng, weights=weights)
    hist = hist.astype(float)
    return hist / (hist.sum() + _EPS)


def _lbp_ones_hist(gray: np.ndarray) -> np.ndarray:
    """Histogram of how many of the 8 neighbours are >= the centre pixel."""
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return np.zeros(9)
    c = gray[1:-1, 1:-1]
    h, w = c.shape
    ones = np.zeros_like(c, dtype=np.int64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ones += (gray[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] >= c)
    return _norm_hist(ones.ravel(), 9, (0, 9))


class ColorTextureFeatures:
    """51-dimensional colour + gradient + local-binary-pattern descriptor."""

    names: list[str] = (
        [f"mean_{c}" for c in "rgb"] + [f"std_{c}" for c in "rgb"]
        + ["chroma_r_mean", "chroma_g_mean", "chroma_r_std", "chroma_g_std"]
        + [f"hue_{i}" for i in range(12)] + [f"sat_{i}" for i in range(4)]
        + [f"val_{i}" for i in range(4)]
        + ["grad_mean", "grad_std", "grad_p50", "grad_p90"]
        + [f"gori_{i}" for i in range(8)] + [f"lbp_{i}" for i in range(9)]
    )

    def __call__(self, img: np.ndarray) -> np.ndarray:
        x = np.asarray(img, dtype=np.float64)
        if x.ndim != 3 or x.shape[2] != 3:
            raise ValueError("Expected an HxWx3 RGB image.")
        x = x / 255.0
        flat = x.reshape(-1, 3)
        mean, std = flat.mean(0), flat.std(0)
        tot = flat.sum(1) + _EPS
        chroma = flat[:, :2] / tot[:, None]
        h, s, v = _hsv(x)
        hue = _norm_hist(h.ravel(), 12, (0, 1), weights=s.ravel())
        sat = _norm_hist(s.ravel(), 4, (0, 1))
        val = _norm_hist(v.ravel(), 4, (0, 1))
        gray = x @ np.array([0.299, 0.587, 0.114])
        gy, gx = np.gradient(gray) if min(gray.shape) > 1 else (np.zeros_like(gray),) * 2
        mag = np.hypot(gx, gy).ravel()
        ang = np.arctan2(gy, gx).ravel()
        gori = _norm_hist(ang, 8, (-np.pi, np.pi), weights=mag)
        gstats = [mag.mean(), mag.std(), np.percentile(mag, 50), np.percentile(mag, 90)]
        feats = np.concatenate([mean, std, chroma.mean(0), chroma.std(0), hue, sat, val,
                                gstats, gori, _lbp_ones_hist(gray)])
        assert feats.size == len(self.names)
        return feats.astype(np.float32)


class TorchBackbone:
    """Frozen foundation-model features (DINOv2 family) for method M.

    Not used by the unit tests.  Requires ``torch``; the model is loaded from the
    path/hub name given by the caller so no download is attempted implicitly.
    """

    def __init__(self, model, device: str = "cpu", size: int = 224):
        import torch  # lazy

        self._torch = torch
        self.model = model.eval().to(device)
        self.device = device
        self.size = size
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def batch(self, imgs) -> np.ndarray:
        torch = self._torch
        ts = []
        for img in imgs:
            t = torch.from_numpy(np.ascontiguousarray(img)).to(self.device).permute(2, 0, 1)[None].float() / 255.0
            ts.append(torch.nn.functional.interpolate(t, size=(self.size, self.size), mode="bilinear",
                                                      align_corners=False))
        x = (torch.cat(ts) - self._mean) / self._std
        with torch.no_grad():
            out = self.model(x)
        return out.reshape(len(imgs), -1).cpu().numpy().astype(np.float32)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.batch([img])[0]


def load_backbone(name: str, device: str = "cpu"):
    """``color`` (baseline) or a DINOv2 hub name such as ``dinov2_vits14``.

    DINOv2 is loaded with torch.hub (``facebookresearch/dinov2``), which needs network access
    and ``torch`` on the machine that runs it.  Outputs the CLS embedding.
    """
    if name == "color":
        return ColorTextureFeatures()
    if name.startswith("dinov2"):
        import torch  # lazy

        model = torch.hub.load("facebookresearch/dinov2", name)
        return TorchBackbone(model, device=device, size=224)
    raise ValueError(f"Unknown backbone {name!r}")
