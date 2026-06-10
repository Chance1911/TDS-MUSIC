"""TDS-MUSIC core package."""

from .arrays.geometry import make_uca_positions, steering_vector, unit_vector
from .estimation.pipeline import DOAEstimate, estimate_tds_music
from .metrics import (
    angular_distance_deg,
    association_success_rate,
    circular_az_error_deg,
    lock_partition,
    rmse,
    success_at_delta,
)

__all__ = [
    "DOAEstimate",
    "angular_distance_deg",
    "association_success_rate",
    "circular_az_error_deg",
    "estimate_tds_music",
    "lock_partition",
    "make_uca_positions",
    "rmse",
    "steering_vector",
    "success_at_delta",
    "unit_vector",
]
