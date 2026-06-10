"""Target-dominant support selection utilities."""

from __future__ import annotations

import numpy as np


def threshold_support(
    reliability: np.ndarray,
    threshold: float = 0.5,
    min_bins: int = 1,
) -> np.ndarray:
    """Convert a continuous reliability map into a boolean TF support."""
    score = np.asarray(reliability, dtype=float)
    if score.ndim != 2:
        raise ValueError("reliability must be a 2-D array")
    support = score >= float(threshold)
    if int(support.sum()) >= int(min_bins):
        return support

    flat = score.ravel()
    keep = max(1, int(min_bins))
    keep = min(keep, flat.size)
    idx = np.argpartition(flat, flat.size - keep)[flat.size - keep :]
    out = np.zeros_like(flat, dtype=bool)
    out[idx] = True
    return out.reshape(score.shape)


def energy_support(
    magnitude_tf: np.ndarray,
    percentile: float = 80.0,
    min_bins: int = 1,
) -> np.ndarray:
    """Simple energy-selected support used for ablation checks."""
    mag = np.asarray(magnitude_tf, dtype=float)
    if mag.ndim != 2:
        raise ValueError("magnitude_tf must be a 2-D array")
    thr = float(np.percentile(mag[np.isfinite(mag)], percentile))
    return threshold_support(mag, threshold=thr, min_bins=min_bins)


def support_coverage(support: np.ndarray) -> float:
    mask = np.asarray(support).astype(bool)
    return float(mask.mean()) if mask.size else float("nan")
