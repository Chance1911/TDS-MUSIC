from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tds_music.arrays.geometry import make_uca_positions, steering_vector
from tds_music.estimation.pipeline import estimate_tds_music
from tds_music.metrics import angular_distance_deg


class MusicSmokeTests(unittest.TestCase):
    def test_single_source_peak(self) -> None:
        rng = np.random.default_rng(7)
        positions = make_uca_positions(num_sensors=8, radius_m=0.14)
        freqs = np.array([2.397e9, 2.407e9, 2.417e9])
        az_true = 35.0
        el_true = 15.0
        t = 160
        x = np.zeros((t, len(freqs), positions.shape[0]), dtype=np.complex128)

        for f_idx, freq in enumerate(freqs):
            a = steering_vector(positions, az_true, el_true, freq)
            signal = (
                rng.standard_normal(t) + 1j * rng.standard_normal(t)
            ) / np.sqrt(2.0)
            noise = 0.05 * (
                rng.standard_normal((t, positions.shape[0]))
                + 1j * rng.standard_normal((t, positions.shape[0]))
            ) / np.sqrt(2.0)
            x[:, f_idx, :] = signal[:, None] * a[None, :] + noise

        support = np.ones((t, len(freqs)), dtype=bool)
        az_grid = np.arange(0.0, 360.0, 5.0)
        el_grid = np.arange(-30.0, 31.0, 5.0)
        est = estimate_tds_music(
            x,
            freqs,
            support,
            positions,
            az_grid,
            el_grid,
            min_snapshots_per_freq=16,
        )
        self.assertLessEqual(float(angular_distance_deg(est.az_deg, est.el_deg, az_true, el_true)), 5.0)
        self.assertEqual(est.n_frequency_bins, len(freqs))


if __name__ == "__main__":
    unittest.main()
