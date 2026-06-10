"""Convenience estimators for TDS-MUSIC and TDS-MUSIC-Fast."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .covariance import pooled_covariance, selected_covariances
from .music import music_spectrum, peak_az_el


@dataclass(frozen=True)
class DOAEstimate:
    az_deg: float
    el_deg: float
    peak_value: float
    spectrum: np.ndarray
    n_frequency_bins: int
    n_snapshots: int


def estimate_tds_music(
    x_tf: np.ndarray,
    freqs_hz: np.ndarray,
    support: np.ndarray,
    positions_m: np.ndarray,
    az_grid_deg: np.ndarray,
    el_grid_deg: np.ndarray,
    n_sources: int = 1,
    min_snapshots_per_freq: int = 8,
) -> DOAEstimate:
    """Estimate DOA with support-selected incoherent wideband MUSIC."""
    covs = selected_covariances(
        x_tf,
        freqs_hz,
        support,
        min_snapshots_per_freq=min_snapshots_per_freq,
    )
    spec = music_spectrum(covs, positions_m, az_grid_deg, el_grid_deg, n_sources=n_sources)
    az, el, val = peak_az_el(spec, az_grid_deg, el_grid_deg)
    return DOAEstimate(
        az_deg=az,
        el_deg=el,
        peak_value=val,
        spectrum=spec,
        n_frequency_bins=len(covs),
        n_snapshots=sum(item.n_snapshots for item in covs),
    )


def estimate_tds_music_fast(
    x_tf: np.ndarray,
    freqs_hz: np.ndarray,
    support: np.ndarray,
    positions_m: np.ndarray,
    az_grid_deg: np.ndarray,
    el_grid_deg: np.ndarray,
    n_sources: int = 1,
) -> DOAEstimate:
    """Estimate DOA with a pooled covariance used by the fast variant."""
    cov = pooled_covariance(x_tf, freqs_hz, support)
    spec = music_spectrum([cov], positions_m, az_grid_deg, el_grid_deg, n_sources=n_sources)
    az, el, val = peak_az_el(spec, az_grid_deg, el_grid_deg)
    return DOAEstimate(
        az_deg=az,
        el_deg=el,
        peak_value=val,
        spectrum=spec,
        n_frequency_bins=1,
        n_snapshots=cov.n_snapshots,
    )
