#!/usr/bin/env python3
"""Render TF512 H5 observations as an ImageFolder TFI dataset."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tds_music.simulation.build_tfi_imagefolder import main  # noqa: E402


if __name__ == "__main__":
    main()
