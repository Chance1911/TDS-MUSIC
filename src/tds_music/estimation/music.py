"""Wideband MUSIC spectrum evaluation."""

from __future__ import annotations

import numpy as np

from .covariance import FrequencyCovariance
from ..arrays.geometry import C_LIGHT, steering_vector


def noise_subspace(covariance: np.ndarray, n_sources: int = 1) -> np.ndarray:
    """Return the noise-subspace eigenvectors of a Hermitian covariance."""
    cov = np.asarray(covariance)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be square")
    if not (0 < n_sources < cov.shape[0]):
        raise ValueError("n_sources must be between 1 and M-1")
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)
    return vecs[:, order[:-n_sources]]


def music_spectrum(
    covariances: list[FrequencyCovariance],
    positions_m: np.ndarray,
    az_grid_deg: np.ndarray,
    el_grid_deg: np.ndarray,
    n_sources: int = 1,
    propagation_speed: float = C_LIGHT,
    eps: float = 1e-12,
) -> np.ndarray:
    """Evaluate the incoherent wideband MUSIC pseudospectrum."""
    az = np.asarray(az_grid_deg, dtype=float)
    el = np.asarray(el_grid_deg, dtype=float)
    az_mesh, el_mesh = np.meshgrid(az, el)
    az_flat = az_mesh.ravel()
    el_flat = el_mesh.ravel()

    spec = np.zeros(az_flat.shape[0], dtype=float)
    for item in covariances:
        en = noise_subspace(item.covariance, n_sources=n_sources)
        steering = steering_vector(
            positions_m,
            az_flat,
            el_flat,
            item.frequency_hz,
            propagation_speed=propagation_speed,
        )
        proj = steering.conj() @ en
        denom = np.sum(np.abs(proj) ** 2, axis=1)
        spec += 1.0 / np.maximum(denom, eps)

    spec = spec.reshape(el.shape[0], az.shape[0])
    peak = float(np.nanmax(spec))
    if np.isfinite(peak) and peak > 0:
        spec = spec / peak
    return spec


def peak_az_el(
    spectrum: np.ndarray,
    az_grid_deg: np.ndarray,
    el_grid_deg: np.ndarray,
) -> tuple[float, float, float]:
    """Return ``(az, el, value)`` at the maximum of a 2-D spectrum.

    The azimuth is shifted by 180 degrees to match the TF512 steering convention
    used by the packaged experiments.
    """
    spec = np.asarray(spectrum, dtype=float)
    if spec.ndim != 2:
        raise ValueError("spectrum must be [El, Az]")
    idx = int(np.nanargmax(spec))
    el_idx, az_idx = np.unravel_index(idx, spec.shape)
    az = (float(np.asarray(az_grid_deg, dtype=float)[az_idx]) + 180.0) % 360.0
    el = float(np.asarray(el_grid_deg, dtype=float)[el_idx])
    return az, el, float(spec[el_idx, az_idx])
