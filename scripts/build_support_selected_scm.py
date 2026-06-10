#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_support_selected_scm.py

用途
----
基于统一原始仿真观测模型 X_tf，使用 TF 掩码重构 pooled TF-SCM。

新的统一接口约定
----------------
输入 h5 优先包含：
- X_tf            : [T,F,8,2]，复数时频域阵列观测（real/imag）
- time_axis_s     : [T]
- freqs_hz        : [F]
- labels/az_gt
- labels/el_gt

兼容旧接口（仅作为回退）：
- X_stft
- stft

输出新 h5：
- 保留原始关键坐标与标签：
    /time_axis_s
    /freqs_hz
    /labels
- 新增：
    /<out_dataset>        : [1,8,8,2] 或 [N,8,8,2]
    /<out_dataset>_valid  : [1] 或 [N]
- 复制常用 attrs

核心方法
--------
对 mask_tf 选中的所有 (t,f) snapshot 做池化，构造一个 pooled TF-SCM：

    R = (1/K) * sum_{(t,f) in Omega} x(t,f) x(t,f)^H

注意：
这不是标准逐频宽带 SCM，而是 pooled TF-SCM。
"""

import os
import glob
import argparse
from typing import Dict, Optional, Tuple

import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm


def build_mask_stem_map(mask_root: str, suffix: str = "_mask_tf.npy") -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for root, _, files in os.walk(mask_root):
        for fname in files:
            if not fname.endswith(suffix):
                continue
            stem = os.path.splitext(fname)[0]
            mapping[stem] = os.path.join(root, fname)
    print(f"[INFO] 已建立 TF 掩码索引，共 {len(mapping)} 个 mask。")
    return mapping


def pick_input_key(hf: h5py.File) -> str:
    for key in ("X_tf", "X_stft", "stft"):
        if key in hf:
            return key
    raise KeyError("输入 h5 中不存在 'X_tf' / 'X_stft' / 'stft' 任一数据集。")


def load_observation_shape(ds) -> Tuple[int, int, int, int, int]:
    if ds.ndim == 4:
        T, F, C, R2 = ds.shape
        return 1, T, F, C, R2
    if ds.ndim == 5:
        N, T, F, C, R2 = ds.shape
        return N, T, F, C, R2
    raise ValueError(f"输入数据维度异常，期望 4D 或 5D，实际 shape={ds.shape}")


def get_one_clip_observation(ds, idx: int) -> np.ndarray:
    if ds.ndim == 4:
        return ds[:]
    return ds[idx]


def resize_mask_to_tf(mask_tf: np.ndarray, T: int, F: int) -> np.ndarray:
    m = np.asarray(mask_tf)
    if m.ndim != 2:
        raise ValueError(f"mask_tf 期望是 2D，实际 shape={m.shape}")

    if m.shape == (T, F):
        pass
    elif m.shape == (F, T):
        m = m.T
    elif m.shape[0] == F:
        m_img = Image.fromarray(m.astype(np.float32))
        m_img = m_img.resize((T, F), resample=Image.NEAREST)
        m = np.array(m_img, dtype=np.float32).T
    else:
        m_img = Image.fromarray(m.astype(np.float32))
        m_img = m_img.resize((F, T), resample=Image.NEAREST)
        m = np.array(m_img, dtype=np.float32)

    m = m.astype(np.float32)
    if m.max() > 1.0:
        m = (m > 0.5).astype(np.float32)

    return (m > 0.5).astype(np.float32)


def reconstruct_pooled_tf_scm(
    x_tf_ri: np.ndarray,
    mask_tf: np.ndarray,
    min_tf_samples: int = 30,
) -> Optional[np.ndarray]:
    T, F, C, R2 = x_tf_ri.shape
    if not (C == 8 and R2 == 2):
        raise ValueError(f"x_tf_ri 形状异常，期望 [T,F,8,2]，实际 {x_tf_ri.shape}")

    mask_bin = resize_mask_to_tf(mask_tf, T, F) > 0.5

    X = x_tf_ri[..., 0].astype(np.float32) + 1j * x_tf_ri[..., 1].astype(np.float32)
    X_flat = X.reshape(-1, C)
    mask_flat = mask_bin.reshape(-1)

    idx = np.where(mask_flat)[0]
    if idx.size < min_tf_samples:
        return None

    X_sel = X_flat[idx, :]
    R = (X_sel.conj().T @ X_sel) / X_sel.shape[0]
    return R.astype(np.complex64)


def copy_if_exists(src: h5py.File, dst: h5py.File, key: str):
    if key in src:
        src.copy(src[key], dst, name=key)


def copy_core_attrs(src: h5py.File, dst: h5py.File):
    keep_keys = [
        "proxy_type",
        "split",
        "snr_db",
        "dataset_mode",
        "meta_json",
        "clip_duration_ms",
        "observation_duration_ms",
        "T",
        "F",
        "df_hz",
    ]
    for k in keep_keys:
        if k in src.attrs:
            dst.attrs[k] = src.attrs[k]


def parse_args():
    ap = argparse.ArgumentParser(
        description="基于 TF 掩码和 X_tf 重构 pooled TF-SCM，输出到新的 h5 根目录。"
    )
    ap.add_argument("--h5_root", required=True, help="原始 h5 根目录")
    ap.add_argument("--mask_root", required=True, help="TF 掩码根目录")
    ap.add_argument("--out_root", required=True, help="输出 h5 根目录")
    ap.add_argument("--min_tf_samples", type=int, default=30, help="最少 TF snapshot 数")
    ap.add_argument("--out_dataset", type=str, default="cc_scm_tf", help="输出 pooled TF-SCM 数据集名")
    ap.add_argument("--overwrite", action="store_true", help="若目标 h5 已存在，允许覆盖")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_root, exist_ok=True)

    mask_map = build_mask_stem_map(args.mask_root, suffix="_mask_tf.npy")
    h5_paths = sorted(glob.glob(os.path.join(args.h5_root, "**", "*.h5"), recursive=True))
    if not h5_paths:
        print(f"[WARN] 在 {args.h5_root} 下未找到任何 .h5")
        return

    print(f"[INFO] 共找到 {len(h5_paths)} 个 h5，开始重构 pooled TF-SCM ...")

    out_name = args.out_dataset
    valid_name = out_name + "_valid"

    for h5_path in h5_paths:
        rel = os.path.relpath(h5_path, args.h5_root)
        out_h5_path = os.path.join(args.out_root, rel)
        os.makedirs(os.path.dirname(out_h5_path), exist_ok=True)

        if os.path.exists(out_h5_path) and not args.overwrite:
            print(f"[SKIP] 已存在: {out_h5_path}")
            continue
        if os.path.exists(out_h5_path) and args.overwrite:
            os.remove(out_h5_path)

        print(f"\n===== 处理 {h5_path} =====")
        try:
            with h5py.File(h5_path, "r") as fr, h5py.File(out_h5_path, "w") as fw:
                in_key = pick_input_key(fr)
                x_ds = fr[in_key]
                N, T, F, C, R2 = load_observation_shape(x_ds)
                if not (C == 8 and R2 == 2):
                    raise ValueError(f"{in_key} 形状异常，实际 {x_ds.shape}，期望最后两维为 [8,2]")

                print(f"[INFO] 使用输入键: {in_key}, shape={x_ds.shape}")

                copy_if_exists(fr, fw, "labels")
                copy_if_exists(fr, fw, "time_axis_s")
                copy_if_exists(fr, fw, "freqs_hz")
                copy_core_attrs(fr, fw)

                scm_ds = fw.create_dataset(out_name, shape=(N, 8, 8, 2), dtype="float32")
                valid_ds = fw.create_dataset(valid_name, shape=(N,), dtype="uint8")

                h5_base = os.path.basename(h5_path)
                if h5_base.endswith("_rf.h5"):
                    clip_stem = h5_base[:-6]
                else:
                    clip_stem = os.path.splitext(h5_base)[0]

                num_valid = 0
                for i in tqdm(range(N), desc="samples", ncols=100):
                    if N == 1:
                        mask_stem = f"{clip_stem}_mask_tf"
                    else:
                        mask_stem = f"{clip_stem}_F{i:03d}_mask_tf"

                    if mask_stem not in mask_map:
                        scm_ds[i, :, :, :] = 0.0
                        valid_ds[i] = 0
                        continue

                    mask_path = mask_map[mask_stem]
                    mask_tf = np.load(mask_path)
                    x_one = get_one_clip_observation(x_ds, i)

                    R = reconstruct_pooled_tf_scm(
                        x_tf_ri=x_one,
                        mask_tf=mask_tf,
                        min_tf_samples=args.min_tf_samples,
                    )

                    if R is None:
                        scm_ds[i, :, :, :] = 0.0
                        valid_ds[i] = 0
                        continue

                    scm_ds[i, :, :, 0] = R.real.astype(np.float32)
                    scm_ds[i, :, :, 1] = R.imag.astype(np.float32)
                    valid_ds[i] = 1
                    num_valid += 1

                print(f"[OK] 输出完成: {out_h5_path}")
                print(f"[OK] 有效 pooled TF-SCM 数: {num_valid} / {N}")

        except Exception as e:
            print(f"[ERROR] 处理失败: {h5_path}")
            print(f"        {type(e).__name__}: {e}")

    print("\n[DONE] 全部完成。")


if __name__ == "__main__":
    main()
