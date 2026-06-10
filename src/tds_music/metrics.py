"""Metrics used by the TDS-MUSIC experiments."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def circular_az_error_deg(pred: float | np.ndarray, gt: float | np.ndarray) -> float | np.ndarray:
    """Circular azimuth absolute error in degrees."""
    return np.abs((np.asarray(pred) - np.asarray(gt) + 180.0) % 360.0 - 180.0)


def angular_distance_deg(
    az1: float | np.ndarray,
    el1: float | np.ndarray,
    az2: float | np.ndarray,
    el2: float | np.ndarray,
) -> float | np.ndarray:
    """Great-circle angular distance for azimuth/elevation pairs in degrees."""
    az1r = np.deg2rad(az1)
    el1r = np.deg2rad(el1)
    az2r = np.deg2rad(az2)
    el2r = np.deg2rad(el2)
    u1 = np.stack(
        [np.cos(el1r) * np.cos(az1r), np.cos(el1r) * np.sin(az1r), np.sin(el1r)],
        axis=0,
    )
    u2 = np.stack(
        [np.cos(el2r) * np.cos(az2r), np.cos(el2r) * np.sin(az2r), np.sin(el2r)],
        axis=0,
    )
    dot = np.sum(u1 * u2, axis=0)
    return np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))


def rmse(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(math.sqrt(np.mean(arr * arr))) if arr.size else float("nan")


def success_at_delta(errors: Iterable[float], delta: float = 10.0) -> float:
    arr = np.asarray(list(errors), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr <= delta)) if arr.size else float("nan")


def association_success_rate(
    pred_az: Iterable[float],
    pred_el: Iterable[float],
    target_az: Iterable[float],
    target_el: Iterable[float],
    jammer_az: Iterable[float],
    jammer_el: Iterable[float],
) -> float:
    """Threshold-free ASR: prediction closer to target than to the jammer."""
    pt = angular_distance_deg(pred_az, pred_el, target_az, target_el)
    pj = angular_distance_deg(pred_az, pred_el, jammer_az, jammer_el)
    return float(np.mean(np.asarray(pt) < np.asarray(pj)))


def lock_partition(
    pred_az: Iterable[float],
    pred_el: Iterable[float],
    target_az: Iterable[float],
    target_el: Iterable[float],
    jammer_az: Iterable[float],
    jammer_el: Iterable[float],
    delta: float = 10.0,
) -> dict[str, float]:
    """Return target-lock, jammer-lock, and neither fractions."""
    pt = np.asarray(angular_distance_deg(pred_az, pred_el, target_az, target_el))
    pj = np.asarray(angular_distance_deg(pred_az, pred_el, jammer_az, jammer_el))
    finite = np.isfinite(pt) & np.isfinite(pj)
    if not finite.any():
        return {"target_lock": float("nan"), "jammer_lock": float("nan"), "neither": float("nan")}
    target_lock = finite & (pt < pj) & (pt <= delta)
    jammer_lock = finite & (pj <= pt) & (pj <= delta)
    neither = finite & ~(target_lock | jammer_lock)
    den = float(finite.sum())
    return {
        "target_lock": float(target_lock.sum() / den),
        "jammer_lock": float(jammer_lock.sum() / den),
        "neither": float(neither.sum() / den),
    }
