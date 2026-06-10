"""Small I/O helpers for TF observations."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def load_tf_h5(path: str | Path) -> dict[str, np.ndarray]:
    """Load the common TF512 H5 fields used by the estimator."""
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as h5:
        for key in ("X_tf", "freqs_hz", "az_gt", "el_gt", "target_mask", "interference_mask"):
            if key in h5:
                out[key] = h5[key][()]
    if "X_tf" not in out or "freqs_hz" not in out:
        raise KeyError("H5 file must contain X_tf and freqs_hz")
    return out
