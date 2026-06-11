#!/usr/bin/env python3
"""Regression checks for externally supplied summary tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def close(name: str, actual: float, expected: float, tol: float) -> tuple[bool, str]:
    ok = math.isfinite(actual) and abs(actual - expected) <= tol
    return ok, f"{name}: actual={actual:.4f}, expected={expected:.4f}, tol={tol}"


def check_runtime(data_root: Path) -> list[tuple[bool, str]]:
    rows = read_csv(data_root / "summaries/exp10/stage_summary.csv")
    total = next((r for r in rows if r.get("stage") == "Total"), None)
    if total is None:
        return [(False, "runtime Total row missing")]
    speed_path = data_root / "summaries/exp10/speedups.json"
    if not speed_path.exists():
        raise FileNotFoundError(speed_path)
    with speed_path.open(encoding="utf-8") as fh:
        speed = json.load(fh)
    return [
        close("runtime IMUSIC median ms", float(total["imusic_median_ms"]), 6333.30, 1.0),
        close("runtime TDS-MUSIC median ms", float(total["tds_music_median_ms"]), 905.45, 1.0),
        close("runtime TDS-MUSIC-Fast median ms", float(total["tds_music_fast_median_ms"]), 85.24, 1.0),
        close("speedup IMUSIC/Fast", float(speed["imusic_over_tds_music_fast"]), 74.30, 0.2),
    ]


def check_exp5(data_root: Path) -> list[tuple[bool, str]]:
    rows = read_csv(data_root / "summaries/exp3_exp5/exp5_asr_summary_by_sir_overlap_method.csv")
    target = [
        r
        for r in rows
        if r["sir_db"] in {"-10", "-10.0"}
        and r["overlap"] == "0.600-0.800"
        and r["method"] == "tfmask"
    ]
    if not target:
        return [(False, "exp5 TDS-MUSIC SIR=-10 overlap=0.600-0.800 row missing")]
    row = target[0]
    return [
        close("exp5 TDS-MUSIC ASR pct", float(row["asr_pct"]), 57.5, 1.0),
        close("exp5 TDS-MUSIC az success pct", float(row["az_success_at_delta_pct"]), 61.0, 1.5),
    ]


def check_field(data_root: Path) -> list[tuple[bool, str]]:
    rows = read_csv(data_root / "field/field_summary_tables.csv")
    by_method = {r["method"]: r for r in rows}
    return [
        close("field IMUSIC Az RMSE", float(by_method["IMUSIC"]["az_rmse"]), 41.33, 0.1),
        close("field TDS-MUSIC Az RMSE", float(by_method["TDS-MUSIC"]["az_rmse"]), 12.27, 0.1),
        close("field TDS-MUSIC El RMSE", float(by_method["TDS-MUSIC"]["el_rmse"]), 4.67, 0.1),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data",
        help="Directory containing summaries/ and field/ result tables.",
    )
    args = parser.parse_args()

    checks: list[tuple[bool, str]] = []
    try:
        checks.extend(check_runtime(args.data_root))
        checks.extend(check_exp5(args.data_root))
        checks.extend(check_field(args.data_root))
    except FileNotFoundError as exc:
        print(
            f"Missing result table: {exc.filename or exc}. "
            "Generate the experiment outputs or provide --data-root before running this check.",
            file=sys.stderr,
        )
        return 2
    for ok, msg in checks:
        print(("OK  " if ok else "BAD ") + msg)
    return 1 if any(not ok for ok, _msg in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
