#!/usr/bin/env python3
"""Generate TF512 signal-plus-noise simulation data."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tds_music.simulation.awgn_tf512 import main  # noqa: E402


if __name__ == "__main__":
    main()
