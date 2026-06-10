#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_tfi_imagefolder.py

从统一的 TF512 h5 数据集中构建 TFI 分类数据集。
新接口约定：
- 优先读取 X_tf: [512,512,8,2]
- 若不存在，兼容回退到 X_stft
- 输出 PNG 分类数据集 + manifest.csv

示例：
python scripts/build_tfi_imagefolder.py \
  --h5_root outputs/tf512_h5 \
  --out_dir outputs/tfi_imagefolder \
  --mode mean_mag

可选 mode:
- sensor0      : 仅用 0 号阵元幅度图
- mean_mag     : 8 阵元幅度取平均（推荐）
- mean_complex : 8 阵元复数先平均，再取幅度
"""

import os
import re
import csv
import json
import argparse
import tempfile
from pathlib import Path

import h5py
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "tds_music_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VALID_SPLITS = ("train", "val", "test")


def ri_to_complex(x_ri: np.ndarray) -> np.ndarray:
    return x_ri[..., 0] + 1j * x_ri[..., 1]


def proxy_type_to_label(proxy_type: str) -> int:
    s = proxy_type.strip()
    m = re.match(r"U(\d+)(?:$|_)", s)
    if not m:
        raise ValueError(f"Unexpected proxy_type: {proxy_type!r} (expected startswith 'U1'..'U9')")
    idx = int(m.group(1))
    if not (1 <= idx <= 9):
        raise ValueError(f"proxy_type out of range: {proxy_type!r}")
    return idx - 1


def discover_presplit_h5s(h5_root: Path, limit_per_split: int = 0):
    split_items = {k: [] for k in VALID_SPLITS}
    for split_name in VALID_SPLITS:
        split_dir = h5_root / split_name
        if not split_dir.exists():
            continue
        files = sorted(split_dir.rglob("*.h5"))
        if limit_per_split and limit_per_split > 0:
            files = files[:limit_per_split]
        split_items[split_name].extend(files)
    return split_items


def infer_split_from_h5_path_or_attr(h5_path: Path, hf: h5py.File) -> str:
    attr = hf.attrs.get("split", None)
    if isinstance(attr, bytes):
        attr = attr.decode("utf-8")
    if isinstance(attr, str) and attr in VALID_SPLITS:
        return attr
    for part in h5_path.parts:
        if part in VALID_SPLITS:
            return part
    raise ValueError(f"Unable to infer split for {h5_path}")


def load_xtf_ri(hf: h5py.File) -> np.ndarray:
    if "X_tf" in hf:
        arr = hf["X_tf"][:]
        key = "X_tf"
    elif "X_stft" in hf:
        arr = hf["X_stft"][:]
        key = "X_stft"
    else:
        raise ValueError("Neither 'X_tf' nor 'X_stft' exists in h5")

    if arr.ndim != 4 or arr.shape[-1] != 2:
        raise ValueError(f"{key} expected [T,F,M,2], got shape {arr.shape}")
    return arr


def build_tfi_db(X_tf_ri: np.ndarray, mode: str = "mean_mag", eps: float = 1e-12) -> np.ndarray:
    X = ri_to_complex(X_tf_ri)
    if X.ndim != 3:
        raise ValueError(f"X expected [T,F,M], got shape {X.shape}")

    if mode == "sensor0":
        X_tf = X[:, :, 0]
        dB = 20.0 * np.log10(np.abs(X_tf) + eps)
    elif mode == "mean_mag":
        X_tf = np.mean(np.abs(X), axis=2)
        dB = 20.0 * np.log10(X_tf + eps)
    elif mode == "mean_complex":
        X_tf = np.mean(X, axis=2)
        dB = 20.0 * np.log10(np.abs(X_tf) + eps)
    else:
        raise ValueError("mode must be one of: sensor0, mean_mag, mean_complex")

    return dB.astype(np.float32)


def save_db_image(dB: np.ndarray, out_png: Path, dpi: int = 120):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 6), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(dB.T, origin="lower", aspect="auto")
    ax.set_axis_off()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5_root", type=str, required=True, help="Root containing pre-split train/val/test h5 folders")
    ap.add_argument("--out_dir", type=str, required=True, help="Output dataset folder")
    ap.add_argument(
        "--mode",
        type=str,
        default="mean_mag",
        choices=["sensor0", "mean_mag", "mean_complex"],
        help="How to derive a single TFI image from multi-sensor X_tf",
    )
    ap.add_argument("--dpi", type=int, default=120, help="PNG render DPI")
    ap.add_argument("--limit_per_split", type=int, default=0, help="Optional cap per split (0 = no limit)")
    args = ap.parse_args()

    h5_root = Path(args.h5_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_items = discover_presplit_h5s(h5_root, limit_per_split=args.limit_per_split)
    total_h5 = sum(len(v) for v in split_items.values())
    if total_h5 == 0:
        raise SystemExit(f"No .h5 found under split folders in {h5_root}")

    manifest_path = out_dir / "manifest.csv"
    summary = {split: {i: 0 for i in range(9)} for split in VALID_SPLITS}

    with open(manifest_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow([
            "split", "label", "proxy_type", "snr_db",
            "h5_path", "png_path", "source_key", "mode"
        ])

        for split_name in VALID_SPLITS:
            for h5_path in split_items[split_name]:
                with h5py.File(h5_path, "r") as hf:
                    actual_split = infer_split_from_h5_path_or_attr(h5_path, hf)

                    proxy_type = hf.attrs.get("proxy_type", None)
                    if proxy_type is None:
                        raise ValueError(f"{h5_path} missing attrs['proxy_type']")
                    if isinstance(proxy_type, bytes):
                        proxy_type = proxy_type.decode("utf-8")
                    proxy_type = str(proxy_type)

                    label = proxy_type_to_label(proxy_type)
                    snr_db = hf.attrs.get("snr_db", None)
                    if snr_db is None:
                        meta_json = hf.attrs.get("meta_json", "")
                        if isinstance(meta_json, bytes):
                            meta_json = meta_json.decode("utf-8")
                        snr_db = "" if not meta_json else json.loads(meta_json).get("snr_db", "")

                    source_key = "X_tf" if "X_tf" in hf else ("X_stft" if "X_stft" in hf else "none")
                    X_tf_ri = load_xtf_ri(hf)

                dB = build_tfi_db(X_tf_ri, mode=args.mode)
                png_name = h5_path.stem + ".png"
                png_path = out_dir / actual_split / str(label) / png_name
                save_db_image(dB, png_path, dpi=args.dpi)

                w.writerow([
                    actual_split, label, proxy_type, snr_db,
                    str(h5_path), str(png_path), source_key, args.mode
                ])
                summary[actual_split][label] += 1

    print("Done.")
    for split_name in VALID_SPLITS:
        total = sum(summary[split_name].values())
        print(f"{split_name}: total={total}, per_class={summary[split_name]}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
