"""Array geometry and steering-vector utilities."""

from __future__ import annotations

import numpy as np


C_LIGHT = 299_792_458.0


def make_uca_positions(num_sensors: int = 8, radius_m: float = 0.14) -> np.ndarray:
    """Return uniform circular array coordinates with shape ``[M, 3]``."""
    if num_sensors < 2:
        raise ValueError("num_sensors must be at least 2")
    angles = 2.0 * np.pi * np.arange(num_sensors, dtype=float) / float(num_sensors)
    return np.column_stack(
        [
            radius_m * np.cos(angles),
            radius_m * np.sin(angles),
            np.zeros_like(angles),
        ]
    )


def unit_vector(az_deg: float | np.ndarray, el_deg: float | np.ndarray) -> np.ndarray:
    """Convert azimuth/elevation in degrees to Cartesian unit vectors."""
    az = np.deg2rad(np.asarray(az_deg, dtype=float))
    el = np.deg2rad(np.asarray(el_deg, dtype=float))
    return np.stack(
        [
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ],
        axis=-1,
    )


def steering_vector(
    positions_m: np.ndarray,
    az_deg: float | np.ndarray,
    el_deg: float | np.ndarray,
    frequency_hz: float,
    propagation_speed: float = C_LIGHT,
) -> np.ndarray:
    """Return normalized narrowband steering vectors.

    The output shape is ``[..., M]`` where ``M`` is the number of array
    elements. Scalar angles therefore return a one-dimensional vector.
    """
    positions = np.asarray(positions_m, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_m must have shape [M, 3]")
    direction = unit_vector(az_deg, el_deg)
    delay_m = np.einsum("...d,md->...m", direction, positions)
    phase = -2j * np.pi * float(frequency_hz) * delay_m / float(propagation_speed)
    return np.exp(phase) / np.sqrt(float(positions.shape[0]))
