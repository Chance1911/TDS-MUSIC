#!/usr/bin/env python3
"""Generate TF512 simulation data with controlled interference."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tds_music.simulation.interference_tf512 import main  # noqa: E402


if __name__ == "__main__":
    main()
