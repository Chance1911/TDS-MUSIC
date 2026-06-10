"""Support-selected covariance construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrequencyCovariance:
    """Spatial covariance estimated at one frequency bin."""

    frequency_hz: float
    covariance: np.ndarray
    n_snapshots: int


def as_complex_tf(x_tf: np.ndarray) -> np.ndarray:
    """Normalize TF observations to complex ``[T, F, M]`` layout."""
    x = np.asarray(x_tf)
    if np.iscomplexobj(x):
        if x.ndim != 3:
            raise ValueError("complex x_tf must have shape [T, F, M]")
        return x.astype(np.complex128, copy=False)
    if x.ndim == 4 and x.shape[-1] == 2:
        return (x[..., 0] + 1j * x[..., 1]).astype(np.complex128, copy=False)
    raise ValueError("x_tf must be complex [T, F, M] or real-imag [T, F, M, 2]")


def _regularize_covariance(cov: np.ndarray, diagonal_loading: float) -> np.ndarray:
    if diagonal_loading <= 0:
        return cov
    m = cov.shape[0]
    scale = float(np.real(np.trace(cov)) / max(m, 1))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return cov + diagonal_loading * scale * np.eye(m, dtype=cov.dtype)


def selected_covariances(
    x_tf: np.ndarray,
    freqs_hz: np.ndarray,
    support: np.ndarray,
    min_snapshots_per_freq: int = 8,
    diagonal_loading: float = 1e-6,
    trace_normalize: bool = True,
) -> list[FrequencyCovariance]:
    """Estimate one spatial covariance per selected frequency bin."""
    x = as_complex_tf(x_tf)
    support_bool = np.asarray(support).astype(bool)
    if support_bool.shape != x.shape[:2]:
        raise ValueError("support must have shape [T, F]")
    freqs = np.asarray(freqs_hz, dtype=float)
    if freqs.ndim != 1 or freqs.shape[0] != x.shape[1]:
        raise ValueError("freqs_hz must have shape [F]")

    out: list[FrequencyCovariance] = []
    for f_idx, freq in enumerate(freqs):
        idx = support_bool[:, f_idx]
        n = int(idx.sum())
        if n < min_snapshots_per_freq:
            continue
        xf = x[idx, f_idx, :]
        cov = (xf.conj().T @ xf) / float(n)
        if trace_normalize:
            tr = float(np.real(np.trace(cov)))
            if np.isfinite(tr) and tr > 0:
                cov = cov / tr
        cov = _regularize_covariance(cov, diagonal_loading)
        out.append(FrequencyCovariance(float(freq), cov, n))
    if not out:
        raise ValueError("no frequency bin has enough selected snapshots")
    return out


def pooled_covariance(
    x_tf: np.ndarray,
    freqs_hz: np.ndarray,
    support: np.ndarray,
    diagonal_loading: float = 1e-6,
) -> FrequencyCovariance:
    """Estimate a single pooled covariance for the fast variant."""
    x = as_complex_tf(x_tf)
    support_bool = np.asarray(support).astype(bool)
    if support_bool.shape != x.shape[:2]:
        raise ValueError("support must have shape [T, F]")
    if not support_bool.any():
        raise ValueError("support is empty")

    snapshots = x[support_bool, :]
    cov = (snapshots.conj().T @ snapshots) / float(snapshots.shape[0])
    tr = float(np.real(np.trace(cov)))
    if np.isfinite(tr) and tr > 0:
        cov = cov / tr
    cov = _regularize_covariance(cov, diagonal_loading)

    freqs = np.asarray(freqs_hz, dtype=float)
    f_grid = np.broadcast_to(freqs[None, :], support_bool.shape)
    center_freq = float(np.mean(f_grid[support_bool]))
    return FrequencyCovariance(center_freq, cov, int(snapshots.shape[0]))
