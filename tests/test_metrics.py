from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tds_music.metrics import angular_distance_deg, circular_az_error_deg, success_at_delta


class MetricTests(unittest.TestCase):
    def test_circular_azimuth_error_wraps(self) -> None:
        self.assertAlmostEqual(float(circular_az_error_deg(359.0, 1.0)), 2.0)
        self.assertAlmostEqual(float(circular_az_error_deg(1.0, 359.0)), 2.0)

    def test_angular_distance_zero_for_same_direction(self) -> None:
        self.assertAlmostEqual(float(angular_distance_deg(40.0, 12.0, 40.0, 12.0)), 0.0, places=5)

    def test_success_rate(self) -> None:
        self.assertAlmostEqual(success_at_delta([1.0, 9.0, 11.0], delta=10.0), 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
