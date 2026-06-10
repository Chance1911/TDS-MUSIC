#!/usr/bin/env python3
"""Recompute field summary metrics from the anonymized per-frame table."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tds_music.metrics import rmse, success_at_delta  # noqa: E402


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def percentile(values: list[float], q: float) -> float:
    vals = sorted(v for v in values if v == v)
    if not vals:
        return float("nan")
    pos = (len(vals) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def summarize(rows: list[dict[str, str]]) -> list[dict[str, float | int | str]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["method_name"]].append(row)

    out = []
    for method in sorted(groups):
        items = groups[method]
        az = [float(r["az_err"]) for r in items]
        el = [float(r["el_err"]) for r in items]
        out.append(
            {
                "method": method,
                "n": len(items),
                "az_rmse": rmse(az),
                "az_p50": percentile(az, 50),
                "az_p90": percentile(az, 90),
                "success_at_10_pct": 100.0 * success_at_delta(az, 10.0),
                "el_rmse": rmse(el),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data/field/per_frame_pred_gt.csv")
    args = parser.parse_args()
    if not args.input.exists():
        parser.error(f"missing input table: {args.input}")

    summary = summarize(read_rows(args.input))
    print(f"{'method':<16} {'n':>7} {'AzRMSE':>8} {'AzP50':>8} {'AzP90':>8} {'succ@10':>8} {'ElRMSE':>8}")
    for row in summary:
        print(
            f"{str(row['method']):<16} {int(row['n']):7d} "
            f"{float(row['az_rmse']):8.2f} {float(row['az_p50']):8.2f} "
            f"{float(row['az_p90']):8.2f} {float(row['success_at_10_pct']):7.1f}% "
            f"{float(row['el_rmse']):8.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
