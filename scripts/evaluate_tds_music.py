#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
evaluate_tds_music.py

全面评估：TF-SCM + 2D MUSIC vs GT，并与 baseline (fw_baseline/fy_baseline) 对比。

输入：
- data_root: 你的“重构后的 TF-SCM h5 根目录”，内部递归搜索 *.h5 或按 split 列表匹配
- split_txt: 每行一个 clip 文件名，与训练用 split 格式一致

输出（out_dir/split_name/）：
- per_clip_metrics.csv      : 每段(clip) 的 MAE / valid 帧数 / baseline 对比 / 提升量
- summary.json              : 全局 frame-weighted & clip-unweighted 统计
- plots_best / plots_worst / plots_random : 若干段的 (GT vs Baseline vs MUSIC) 曲线+误差图

说明：
- MUSIC 只在 scm_*_valid==1 的帧上做估计。
- baseline 的误差默认也在“同一批 valid 帧”上统计（更公平）；同时也会给出 baseline_allframes 的统计（如果你需要）。

运行示例：
python evaluate_tds_music.py \
  --data_root data/tfscm_out_root \
  --split_txt splits/split_test.txt \
  --out_dir outputs/music_eval_full \
  --split_name test \
  --plot_best_k 10 --plot_worst_k 10 --plot_random_k 10
"""

import os
import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt


# ================== 阵列 / 物理参数（与 check_tf_scm_music_oneclip.py 保持一致） ==================
fc = 2.407e9       # Hz
c = 3e8
wavelength = c / fc

SensorNum = 8
rr = 0.14          # 阵元半径（单位与波长一致比例即可）

theta_1 = 0.0
Theta_Loc = theta_1 + np.arange(SensorNum) * 360.0 / SensorNum
dix = rr * np.cos(np.deg2rad(Theta_Loc))
diy = rr * np.sin(np.deg2rad(Theta_Loc))
diz = np.zeros(SensorNum, dtype=np.float32)


# ================== 基础工具 ==================

def circular_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """360 度环形误差：min(|a-b|, 360-|a-b|)"""
    diff = np.abs(a - b)
    return np.minimum(diff, 360.0 - diff)


def circular_mean_deg(angles_deg: np.ndarray) -> float:
    """Circular mean of finite angle entries (degrees)."""
    arr = np.asarray(angles_deg, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    rad = np.deg2rad(arr)
    s = float(np.mean(np.sin(rad)))
    c = float(np.mean(np.cos(rad)))
    return float(np.rad2deg(np.arctan2(s, c)) % 360.0)

def safe_makedirs(p: str):
    os.makedirs(p, exist_ok=True)

def load_split_list(path_txt: str) -> List[str]:
    clips = []
    with open(path_txt, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            clips.append(s)
    return clips

def build_h5_basename_map(h5_root: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    dup = 0
    for root, _, files in os.walk(h5_root):
        for fname in files:
            if not fname.endswith(".h5"):
                continue
            full_path = os.path.join(root, fname)
            if fname in mapping and mapping[fname] != full_path:
                # 同名冲突：保留第一条，但给出提示，避免评估到旧文件
                dup += 1
                if dup <= 20:
                    print(f"[WARN] duplicate basename '{fname}': keep '{mapping[fname]}', ignore '{full_path}'")
                continue
            mapping[fname] = full_path
    if dup > 20:
        print(f"[WARN] duplicate basename count={dup} (only first 20 shown)")
    return mapping

def match_h5_paths(data_root: str, split_txt: str) -> List[str]:
    name_map = build_h5_basename_map(data_root)
    want = load_split_list(split_txt)
    paths = []
    missing = 0
    for item in want:
        if os.path.isabs(item) and os.path.exists(item):
            paths.append(item); continue
        p2 = os.path.join(data_root, item)
        if os.path.exists(p2):
            paths.append(p2); continue
        base = os.path.basename(item)
        if base in name_map:
            paths.append(name_map[base]); continue
        missing += 1
    print(f"[split] {os.path.basename(split_txt)} 中 {len(want)} 个名字，匹配到 {len(paths)} 个 h5（缺失 {missing} 个）。")
    return sorted(paths)

def _find_vec_in_labels(grp: h5py.Group, candidates: List[str]) -> np.ndarray:
    # 精确匹配优先
    for name in candidates:
        if name in grp:
            arr = np.asarray(grp[name][()]).squeeze()
            if arr.ndim == 1:
                return arr.astype(np.float32)
    # 模糊匹配
    for k in grp.keys():
        low = k.lower()
        for token in candidates:
            if token in low:
                arr = np.asarray(grp[k][()]).squeeze()
                if arr.ndim == 1:
                    return arr.astype(np.float32)
    raise KeyError(f"在 labels 里找不到候选 {candidates}，keys={list(grp.keys())}")

def infer_gt_from_labels(h5f: h5py.File, ds_name: str = "labels") -> Tuple[np.ndarray, np.ndarray]:
    if ds_name not in h5f:
        raise KeyError(f"h5 中没有 '{ds_name}'，顶层 keys={list(h5f.keys())}")
    obj = h5f[ds_name]
    if isinstance(obj, h5py.Dataset):
        arr = np.asarray(obj[()]).squeeze()
        if arr.ndim == 1:
            az = arr.astype(np.float32)
            el = np.zeros_like(az, dtype=np.float32)
            return az, el
        if arr.ndim == 2 and arr.shape[1] >= 2:
            az = arr[:, 0].astype(np.float32)
            el = arr[:, 1].astype(np.float32)
            return az, el
        raise ValueError(f"[labels] Dataset 形状无法解析: {arr.shape}")

    if isinstance(obj, h5py.Group):
        grp = obj
        az = _find_vec_in_labels(grp, ["az_gt", "az", "az_deg", "azimuth", "azi"])
        el = _find_vec_in_labels(grp, ["el_gt", "el", "el_deg", "elevation", "ele"])
        L = min(len(az), len(el))
        return az[:L], el[:L]

    raise TypeError(f"[labels] '{ds_name}' 既不是 Dataset 也不是 Group: {type(obj)}")

def load_baseline_from_labels(h5f: h5py.File, ds_name: str = "labels") -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if ds_name not in h5f:
        return None, None
    obj = h5f[ds_name]
    if not isinstance(obj, h5py.Group):
        return None, None
    grp = obj
    if "fw_baseline" in grp and "fy_baseline" in grp:
        fw = np.asarray(grp["fw_baseline"][()]).squeeze().astype(np.float32)
        fy = np.asarray(grp["fy_baseline"][()]).squeeze().astype(np.float32)
        L = min(len(fw), len(fy))
        return fw[:L], fy[:L]
    return None, None

def find_scm_tf_keys(h5f: h5py.File) -> Tuple[str, str]:
    """
    返回 (scm_key, valid_key)
    - 优先: scm_tf + scm_tf_valid
    - 否则：在顶层找形状 [N,8,8,2] 的 dataset，并要求同时存在 *_valid
    """
    if "scm_tf" in h5f and "scm_tf_valid" in h5f:
        return "scm_tf", "scm_tf_valid"

    candidates = []
    for k in h5f.keys():
        obj = h5f[k]
        if isinstance(obj, h5py.Dataset) and obj.ndim == 4 and obj.shape[1:] == (8, 8, 2):
            candidates.append(k)
    for k in candidates:
        vk = k + "_valid"
        if vk in h5f and isinstance(h5f[vk], h5py.Dataset):
            return k, vk

    raise KeyError(f"找不到 TF-SCM 数据集：需要 scm_tf/scm_tf_valid 或任意 (N,8,8,2)+*_valid。keys={list(h5f.keys())}")


# ================== MUSIC 加速实现（预计算 steering 向量） ==================

@dataclass
class MusicGrid:
    az_grid: np.ndarray      # [A]
    el_grid: np.ndarray      # [E]
    A: np.ndarray            # [M,8] complex64, M=E*A
    M: int
    A_count: int
    E_count: int

def build_music_grid(az_step_deg: float = 1.0, el_step_deg: float = 1.0) -> MusicGrid:
    az = np.arange(0.0, 360.0, az_step_deg, dtype=np.float32)
    el = np.arange(0.0, 90.0, el_step_deg, dtype=np.float32)

    AZ, EL = np.meshgrid(az, el, indexing="xy")  # [E,A]
    azr = np.deg2rad(AZ).astype(np.float32)
    elr = np.deg2rad(EL).astype(np.float32)

    # Match simulation/preview geometry:
    # u = [cos(el)*cos(az), cos(el)*sin(az), sin(el)]
    ca = np.cos(azr)
    sa = np.sin(azr)
    ce = np.cos(elr)
    se = np.sin(elr)

    ux = ce * ca
    uy = ce * sa
    uz = se

    phase = (dix[None, None, :] * ux[:, :, None] +
             diy[None, None, :] * uy[:, :, None] +
             diz[None, None, :] * uz[:, :, None])

    # Match simulation sign: exp(-j * 2π/λ * phase)
    Asteer = np.exp(-1j * 2.0 * np.pi / wavelength * phase).reshape(-1, SensorNum).astype(np.complex64)

    return MusicGrid(az_grid=az, el_grid=el, A=Asteer, M=Asteer.shape[0], A_count=az.size, E_count=el.size)

def preprocess_scm(R: np.ndarray,
                   hermitianize: bool = True,
                   diag_load: float = 0.0) -> np.ndarray:
    """让 SCM 更数值稳定：Hermitian 化 + 对角加载（可选）"""
    if hermitianize:
        R = 0.5 * (R + R.conj().T)
    if diag_load and diag_load > 0:
        tr = np.real(np.trace(R))
        if np.isfinite(tr) and tr > 0:
            R = R + (diag_load * tr / R.shape[0]) * np.eye(R.shape[0], dtype=R.dtype)
    return R


def estimate_music_single(R: np.ndarray,
                         grid: MusicGrid,
                         n_src: int = 1) -> Tuple[float, float]:
    """单帧 MUSIC（n_src 信源）。R 需为 (8,8) 复数 SCM。"""
    eigvals, eigvecs = np.linalg.eigh(R)
    idx_sort = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx_sort]
    n_src = int(max(1, min(n_src, SensorNum - 1)))
    Un = eigvecs[:, n_src:]       # noise subspace
    Un_proj = Un @ Un.conj().T    # [8,8]

    A = grid.A                    # [M,8]
    denom = np.einsum("bi,ij,bj->b", A.conj(), Un_proj, A, optimize=True)
    P = 1.0 / (np.abs(denom) + 1e-12)

    idx = int(np.argmax(P))
    ie = idx // grid.A_count
    ia = idx % grid.A_count
    el_est = float(grid.el_grid[ie])
    az_est = float(grid.az_grid[ia])
    az_est = (az_est + 180.0) % 360.0

    return az_est, el_est


def _steering_vectors(az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
    """为给定 az/el 网格生成 steering 向量，返回 [M,8] 复数（与仿真/preview一致）。"""
    AZ, EL = np.meshgrid(az_deg, el_deg, indexing="xy")  # [E,A]
    azr = np.deg2rad(AZ).astype(np.float32)
    elr = np.deg2rad(EL).astype(np.float32)

    ca = np.cos(azr)
    sa = np.sin(azr)
    ce = np.cos(elr)
    se = np.sin(elr)

    ux = ce * ca
    uy = ce * sa
    uz = se

    phase = (dix[None, None, :] * ux[:, :, None] +
             diy[None, None, :] * uy[:, :, None] +
             diz[None, None, :] * uz[:, :, None])

    return np.exp(-1j * 2.0 * np.pi / wavelength * phase).reshape(-1, SensorNum).astype(np.complex64)


def estimate_music_coarse_to_fine(R: np.ndarray,
                                 coarse_grid: MusicGrid,
                                 fine_az_step: float,
                                 fine_el_step: float,
                                 fine_window: float,
                                 n_src: int = 1) -> Tuple[float, float]:
    """
    先 coarse 网格找峰，再在峰附近 +/- fine_window 做 fine 搜索。
    - fine_step: 细网格步长（例如 0.25）
    - fine_window: 细搜窗口半径（度）
    """
    az0, el0 = estimate_music_single(R, coarse_grid, n_src=n_src)
    # 反推 coarse 的未修正 az（estimate_music_single 做了 +180 修正）。
    # 细搜时直接在修正后的 az0 附近找即可（仍然应用 +180 修正规则）。

    # az 环绕处理
    az_list = np.arange(az0 - fine_window, az0 + fine_window + 1e-6, fine_az_step, dtype=np.float32)
    az_list = np.mod(az_list, 360.0)
    # 为避免重复（接近 0/360）导致的网格重复，做 unique
    az_list = np.unique(np.round(az_list / fine_az_step).astype(np.int64)) * fine_az_step
    az_list = np.mod(az_list.astype(np.float32), 360.0)
    az_list.sort()

    el_lo = max(0.0, el0 - fine_window)
    el_hi = min(90.0 - 1e-6, el0 + fine_window)
    el_list = np.arange(el_lo, el_hi + 1e-6, fine_el_step, dtype=np.float32)
    if el_list.size == 0:
        el_list = np.array([float(np.clip(el0, 0.0, 90.0 - 1e-6))], dtype=np.float32)

    A_local = _steering_vectors(az_list, el_list)  # [M,8]

    eigvals, eigvecs = np.linalg.eigh(R)
    idx_sort = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx_sort]
    n_src = int(max(1, min(n_src, SensorNum - 1)))
    Un = eigvecs[:, n_src:]
    Un_proj = Un @ Un.conj().T

    denom = np.einsum("bi,ij,bj->b", A_local.conj(), Un_proj, A_local, optimize=True)
    P = 1.0 / (np.abs(denom) + 1e-12)
    idx = int(np.argmax(P))
    # A_local 的排序是 el-major（meshgrid reshape），与 build_music_grid 一致
    ia = idx % az_list.size
    ie = idx // az_list.size
    el_est = float(el_list[ie])
    az_est = float(az_list[ia])
    # 保持与你原始实现一致的 +180 修正
    return az_est, el_est


# ================== 评估 / 绘图 ==================

def stat_err(x: np.ndarray) -> Dict[str, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def temporal_delta_metrics(angle_or_value: np.ndarray,
                           valid_mask: np.ndarray,
                           is_azimuth: bool,
                           jump_thr_deg: Tuple[float, float] = (10.0, 20.0)) -> Dict[str, float]:
    """
    计算时间稳定性指标（只在 valid_mask==1 且数值有限的序列上）：
      - delta_std: 相邻有效帧差分的标准差
      - jump_rate_T: |delta|>T 的比例（T=10/20 默认）

    az 使用环形差分，el/其它使用普通差分。
    """
    idx = np.where((valid_mask == 1) & np.isfinite(angle_or_value))[0]
    if idx.size < 2:
        out = {"delta_std": float("nan")}
        for t in jump_thr_deg:
            out[f"jump_rate_{int(t)}"] = float("nan")
        return out

    seq = angle_or_value[idx].astype(np.float32)
    if is_azimuth:
        d = circular_diff_deg(seq[1:], seq[:-1])
    else:
        d = np.abs(seq[1:] - seq[:-1])

    out = {"delta_std": float(np.std(d))}
    for t in jump_thr_deg:
        out[f"jump_rate_{int(t)}"] = float(np.mean(d > float(t)))
    return out


def extract_valid_deltas(angle_or_value: np.ndarray,
                         valid_mask: np.ndarray,
                         is_azimuth: bool) -> np.ndarray:
    """提取相邻有效帧的差分序列（用于全局统计）。"""
    idx = np.where((valid_mask == 1) & np.isfinite(angle_or_value))[0]
    if idx.size < 2:
        return np.array([], dtype=np.float32)
    x = angle_or_value[idx].astype(np.float32)
    if is_azimuth:
        d = circular_diff_deg(x[1:], x[:-1]).astype(np.float32)
    else:
        d = (x[1:] - x[:-1]).astype(np.float32)
        d = np.abs(d)
    return d

def plot_clip_curves(out_png: str,
                     frames: np.ndarray,
                     az_gt: np.ndarray, el_gt: np.ndarray,
                     az_music: np.ndarray, el_music: np.ndarray,
                     fw: Optional[np.ndarray], fy: Optional[np.ndarray],
                     valid_mask: np.ndarray):
    idx = np.where((valid_mask == 1) & np.isfinite(az_music) & np.isfinite(el_music))[0]

    az_err_m = np.full_like(az_music, np.nan, dtype=np.float32)
    el_err_m = np.full_like(el_music, np.nan, dtype=np.float32)
    az_err_b = el_err_b = None

    if idx.size > 0:
        az_err_m[idx] = circular_diff_deg(az_music[idx], az_gt[idx])
        el_err_m[idx] = np.abs(el_music[idx] - el_gt[idx])

    if fw is not None and fy is not None:
        az_err_b = np.full_like(az_gt, np.nan, dtype=np.float32)
        el_err_b = np.full_like(el_gt, np.nan, dtype=np.float32)
        az_err_b[idx] = circular_diff_deg(fw[idx], az_gt[idx])
        el_err_b[idx] = np.abs(fy[idx] - el_gt[idx])

    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(frames, az_gt, label="GT Az")
    if fw is not None:
        axs[0].plot(frames, fw, "--", label="Baseline fw")
    axs[0].plot(frames, az_music, "-.", label="TF-SCM+MUSIC Az")
    axs[0].set_ylabel("Az (deg)")
    axs[0].grid(True)
    axs[0].legend(loc="best")

    axs[1].plot(frames, el_gt, label="GT El")
    if fy is not None:
        axs[1].plot(frames, fy, "--", label="Baseline fy")
    axs[1].plot(frames, el_music, "-.", label="TF-SCM+MUSIC El")
    axs[1].set_ylabel("El (deg)")
    axs[1].grid(True)
    axs[1].legend(loc="best")

    axs[2].plot(frames, az_err_m, label="MUSIC Az err (valid)")
    if az_err_b is not None:
        axs[2].plot(frames, az_err_b, "--", label="Baseline Az err (valid)")
    axs[2].set_ylabel("Az err (deg)")
    axs[2].grid(True)
    axs[2].legend(loc="best")

    axs[3].plot(frames, el_err_m, label="MUSIC El err (valid)")
    if el_err_b is not None:
        axs[3].plot(frames, el_err_b, "--", label="Baseline El err (valid)")
    axs[3].set_ylabel("El err (deg)")
    axs[3].set_xlabel("Frame idx")
    axs[3].grid(True)
    axs[3].legend(loc="best")

    fig.suptitle(os.path.basename(out_png).replace(".png", ""), y=0.995)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

def parse_args():
    p = argparse.ArgumentParser(description="Full eval: TF-SCM + MUSIC vs GT, compare baseline")
    p.add_argument("--data_root", type=str, required=True, help="重构 TF-SCM h5 根目录")
    p.add_argument("--split_txt", type=str, required=True, help="split_test/val/train txt")
    p.add_argument("--split_name", type=str, default="test", help="输出子目录名")
    p.add_argument("--out_dir", type=str, required=True, help="输出目录根")
    p.add_argument("--az_step", type=float, default=1.0, help="MUSIC az 搜索步长(度)")
    p.add_argument("--el_step", type=float, default=1.0, help="MUSIC el 搜索步长(度)")
    # --- 速度/稳定性增强 ---
    p.add_argument("--coarse_to_fine", action="store_true",
                   help="启用 coarse-to-fine MUSIC：先粗网格找峰，再在峰附近细搜（大幅加速）")
    p.add_argument("--coarse_az_step", type=float, default=2.0, help="coarse 网格 az 步长(度)")
    p.add_argument("--coarse_el_step", type=float, default=2.0, help="coarse 网格 el 步长(度)")
    p.add_argument("--fine_window", type=float, default=2.0, help="细搜窗口半径(度)，例如 2 表示 +/-2 度")
    p.add_argument("--hermitianize", action="store_true", default=True,
                   help="对每帧 SCM 做 Hermitian 化（默认开启）")
    p.add_argument("--no_hermitianize", action="store_true",
                   help="关闭 Hermitian 化（调试用）")
    p.add_argument("--diag_load", type=float, default=0.0,
                   help="对角加载强度（例如 0.002）。实际加载为 diag_load*trace(R)/M")
    p.add_argument("--n_src", type=int, default=1, help="MUSIC 假设信源数（默认 1）")
    p.add_argument("--plot_worst_k", type=int, default=10)
    p.add_argument("--plot_best_k", type=int, default=10)
    p.add_argument("--plot_random_k", type=int, default=10)
    p.add_argument("--plot_seed", type=int, default=0)
    p.add_argument("--max_frames_per_clip_plot", type=int, default=800, help="画图最多画多少帧(防止太大)")
    return p.parse_args()

def main():
    args = parse_args()
    safe_makedirs(args.out_dir)
    out_split_dir = os.path.join(args.out_dir, args.split_name)
    safe_makedirs(out_split_dir)

    h5_paths = match_h5_paths(args.data_root, args.split_txt)
    if not h5_paths:
        raise RuntimeError("没有找到任何 h5，请检查 data_root / split_txt 是否匹配。")

    hermitianize = args.hermitianize and (not args.no_hermitianize)

    if args.coarse_to_fine:
        print(f"[MUSIC] coarse-to-fine enabled. coarse_step=(az={args.coarse_az_step}, el={args.coarse_el_step}), fine_step=(az={args.az_step}, el={args.el_step}), window=+/-{args.fine_window}deg")
        coarse_grid = build_music_grid(args.coarse_az_step, args.coarse_el_step)
        print(f"[MUSIC] coarse grid size: E={coarse_grid.E_count}, A={coarse_grid.A_count}, total={coarse_grid.M}")
        grid = None
    else:
        print(f"[MUSIC] building grid: az_step={args.az_step}, el_step={args.el_step}")
        grid = build_music_grid(args.az_step, args.el_step)
        print(f"[MUSIC] grid size: E={grid.E_count}, A={grid.A_count}, total={grid.M}")
        coarse_grid = None
    if args.diag_load and args.diag_load > 0:
        print(f"[MUSIC] diag_load={args.diag_load} (scaled by trace(R)/M)")
    print(f"[MUSIC] hermitianize={'ON' if hermitianize else 'OFF'}, n_src={args.n_src}")

    rows = []
    all_az_err_music = []
    all_el_err_music = []
    all_az_err_base_valid = []
    all_el_err_base_valid = []
    all_az_err_base_all = []
    all_el_err_base_all = []
    all_az_delta_music = []
    all_el_delta_music = []
    all_az_delta_base_valid = []
    all_el_delta_base_valid = []
    total_valid_frames = 0

    for h5_path in h5_paths:
        clip = os.path.basename(h5_path)
        try:
            with h5py.File(h5_path, "r") as f:
                scm_key, valid_key = find_scm_tf_keys(f)
                scm = np.asarray(f[scm_key][()]).astype(np.float32)   # [N,8,8,2]
                valid = np.asarray(f[valid_key][()]).astype(np.uint8) # [N]
                az_gt, el_gt = infer_gt_from_labels(f, "labels")
                fw, fy = load_baseline_from_labels(f, "labels")
        except Exception as e:
            print(f"Warning 跳过 {h5_path}（读取失败）：{e}")
            continue

        N = scm.shape[0]
        L = min(N, len(valid), len(az_gt), len(el_gt))
        if fw is not None:
            L = min(L, len(fw), len(fy))

        scm = scm[:L]
        valid = valid[:L]
        az_gt = az_gt[:L]
        el_gt = el_gt[:L]
        if fw is not None:
            fw = fw[:L]
            fy = fy[:L]

        R = (scm[..., 0] + 1j * scm[..., 1]).astype(np.complex64)  # [L,8,8]
        idx_valid = np.where(valid == 1)[0]
        total_valid_frames += int(idx_valid.size)

        az_music = np.full(L, np.nan, dtype=np.float32)
        el_music = np.full(L, np.nan, dtype=np.float32)

        for i in idx_valid:
            try:
                Ri = preprocess_scm(R[i], hermitianize=hermitianize, diag_load=args.diag_load)
                if args.coarse_to_fine:
                    a, e = estimate_music_coarse_to_fine(
                        Ri,
                        coarse_grid,
                        fine_az_step=args.az_step,
                        fine_el_step=args.el_step,
                        fine_window=args.fine_window,
                        n_src=args.n_src,
                    )
                else:
                    a, e = estimate_music_single(Ri, grid, n_src=args.n_src)
            except Exception:
                continue
            az_music[i] = a
            el_music[i] = e

        idx_m = np.where((valid == 1) & np.isfinite(az_music) & np.isfinite(el_music))[0]
        if idx_m.size > 0:
            az_err_m = circular_diff_deg(az_music[idx_m], az_gt[idx_m])
            el_err_m = np.abs(el_music[idx_m] - el_gt[idx_m])
            all_az_err_music.append(az_err_m)
            all_el_err_music.append(el_err_m)
            music_az_mae = float(np.mean(az_err_m))
            music_el_mae = float(np.mean(el_err_m))
            # Clip-level single-point prediction, so downstream association
            # analysis can compare against target/jammer DOA the same way it
            # does for the incoherent evaluators. Circular mean for azimuth,
            # plain mean for elevation, both over valid frames only.
            pred_az_clip = circular_mean_deg(az_music[idx_m])
            pred_el_clip = float(np.mean(el_music[idx_m]))
        else:
            music_az_mae = float("nan")
            music_el_mae = float("nan")
            pred_az_clip = float("nan")
            pred_el_clip = float("nan")

        base_az_mae_valid = base_el_mae_valid = float("nan")
        # --- temporal stability (valid frames) ---
        m_az_tmp = temporal_delta_metrics(az_music, valid, is_azimuth=True)
        m_el_tmp = temporal_delta_metrics(el_music, valid, is_azimuth=False)
        # 用于全局统计的 delta 序列
        all_az_delta_music.append(extract_valid_deltas(az_music, valid, is_azimuth=True))
        all_el_delta_music.append(extract_valid_deltas(el_music, valid, is_azimuth=False))

        b_az_tmp = {"delta_std": float("nan"), "jump_rate_10": float("nan"), "jump_rate_20": float("nan")}
        b_el_tmp = {"delta_std": float("nan"), "jump_rate_10": float("nan"), "jump_rate_20": float("nan")}
        if fw is not None and fy is not None and idx_valid.size > 0:
            az_err_bv = circular_diff_deg(fw[idx_valid], az_gt[idx_valid])
            el_err_bv = np.abs(fy[idx_valid] - el_gt[idx_valid])
            all_az_err_base_valid.append(az_err_bv)
            all_el_err_base_valid.append(el_err_bv)
            base_az_mae_valid = float(np.mean(az_err_bv))
            base_el_mae_valid = float(np.mean(el_err_bv))

            az_err_ba = circular_diff_deg(fw, az_gt)
            el_err_ba = np.abs(fy - el_gt)
            all_az_err_base_all.append(az_err_ba)
            all_el_err_base_all.append(el_err_ba)

            b_az_tmp = temporal_delta_metrics(fw, valid, is_azimuth=True)
            b_el_tmp = temporal_delta_metrics(fy, valid, is_azimuth=False)
            all_az_delta_base_valid.append(extract_valid_deltas(fw, valid, is_azimuth=True))
            all_el_delta_base_valid.append(extract_valid_deltas(fy, valid, is_azimuth=False))

        rows.append({
            "clip": clip,
            "h5_path": h5_path,
            "scm_key": scm_key,
            "valid_key": valid_key,
            "frames_total": int(L),
            "valid_frames": int(idx_valid.size),
            "pred_az": pred_az_clip,
            "pred_el": pred_el_clip,
            "music_az_mae": music_az_mae,
            "music_el_mae": music_el_mae,
            "music_az_p90": stat_err(az_err_m)["p90"] if idx_m.size > 0 else float("nan"),
            "music_az_p95": stat_err(az_err_m)["p95"] if idx_m.size > 0 else float("nan"),
            "music_az_max": stat_err(az_err_m)["max"] if idx_m.size > 0 else float("nan"),
            "music_el_p90": stat_err(el_err_m)["p90"] if idx_m.size > 0 else float("nan"),
            "music_el_p95": stat_err(el_err_m)["p95"] if idx_m.size > 0 else float("nan"),
            "music_el_max": stat_err(el_err_m)["max"] if idx_m.size > 0 else float("nan"),
            "music_az_delta_std": m_az_tmp.get("delta_std", float("nan")),
            "music_az_jump_rate_10": m_az_tmp.get("jump_rate_10", float("nan")),
            "music_az_jump_rate_20": m_az_tmp.get("jump_rate_20", float("nan")),
            "music_el_delta_std": m_el_tmp.get("delta_std", float("nan")),
            "music_el_jump_rate_10": m_el_tmp.get("jump_rate_10", float("nan")),
            "music_el_jump_rate_20": m_el_tmp.get("jump_rate_20", float("nan")),
            "baseline_az_mae_valid": base_az_mae_valid,
            "baseline_el_mae_valid": base_el_mae_valid,
            "baseline_az_delta_std": b_az_tmp.get("delta_std", float("nan")),
            "baseline_az_jump_rate_10": b_az_tmp.get("jump_rate_10", float("nan")),
            "baseline_az_jump_rate_20": b_az_tmp.get("jump_rate_20", float("nan")),
            "baseline_el_delta_std": b_el_tmp.get("delta_std", float("nan")),
            "baseline_el_jump_rate_10": b_el_tmp.get("jump_rate_10", float("nan")),
            "baseline_el_jump_rate_20": b_el_tmp.get("jump_rate_20", float("nan")),
            "az_improve_valid": (base_az_mae_valid - music_az_mae) if np.isfinite(base_az_mae_valid) and np.isfinite(music_az_mae) else float("nan"),
            "el_improve_valid": (base_el_mae_valid - music_el_mae) if np.isfinite(base_el_mae_valid) and np.isfinite(music_el_mae) else float("nan"),
        })

    if not rows:
        raise RuntimeError("没有任何可用 clip 被评估（可能全部读取失败/无valid）。")

    df = pd.DataFrame(rows)
    df_path = os.path.join(out_split_dir, "per_clip_metrics.csv")
    df.to_csv(df_path, index=False, encoding="utf-8")
    print(f"[CSV] {df_path}")

    def _cat(xs: List[np.ndarray]) -> np.ndarray:
        return np.concatenate(xs) if xs else np.array([], dtype=np.float32)

    az_m_all = _cat(all_az_err_music)
    el_m_all = _cat(all_el_err_music)
    az_bv_all = _cat(all_az_err_base_valid)
    el_bv_all = _cat(all_el_err_base_valid)
    az_ba_all = _cat(all_az_err_base_all)
    el_ba_all = _cat(all_el_err_base_all)
    az_dm_all = _cat(all_az_delta_music)
    el_dm_all = _cat(all_el_delta_music)
    az_db_all = _cat(all_az_delta_base_valid)
    el_db_all = _cat(all_el_delta_base_valid)

    def _jump_rate(d: np.ndarray, thr: float) -> float:
        d = d[np.isfinite(d)]
        if d.size == 0:
            return float("nan")
        return float(np.mean(d > thr))

    summary = {
        "split_name": args.split_name,
        "num_clips_input": len(h5_paths),
        "num_clips_eval": int(df.shape[0]),
        "total_valid_frames": int(total_valid_frames),

        "music_vs_gt_frame_weighted": {"az": stat_err(az_m_all), "el": stat_err(el_m_all)},
        "baseline_vs_gt_frame_weighted_on_valid": {"az": stat_err(az_bv_all), "el": stat_err(el_bv_all)},
        "baseline_vs_gt_frame_weighted_all_frames": {"az": stat_err(az_ba_all), "el": stat_err(el_ba_all)},

        "music_temporal_on_valid": {
            "az_delta": stat_err(az_dm_all),
            "el_delta": stat_err(el_dm_all),
            "az_jump_rate_10": _jump_rate(az_dm_all, 10.0),
            "az_jump_rate_20": _jump_rate(az_dm_all, 20.0),
            "el_jump_rate_10": _jump_rate(el_dm_all, 10.0),
            "el_jump_rate_20": _jump_rate(el_dm_all, 20.0),
        },
        "baseline_temporal_on_valid": {
            "az_delta": stat_err(az_db_all),
            "el_delta": stat_err(el_db_all),
            "az_jump_rate_10": _jump_rate(az_db_all, 10.0),
            "az_jump_rate_20": _jump_rate(az_db_all, 20.0),
            "el_jump_rate_10": _jump_rate(el_db_all, 10.0),
            "el_jump_rate_20": _jump_rate(el_db_all, 20.0),
        },

        "music_vs_gt_clip_unweighted": {
            "az_mean_of_clip_mae": float(np.nanmean(df["music_az_mae"].values)),
            "el_mean_of_clip_mae": float(np.nanmean(df["music_el_mae"].values)),
        },
        "baseline_vs_gt_clip_unweighted_on_valid": {
            "az_mean_of_clip_mae": float(np.nanmean(df["baseline_az_mae_valid"].values)),
            "el_mean_of_clip_mae": float(np.nanmean(df["baseline_el_mae_valid"].values)),
        },
        "improve_clip_unweighted_on_valid": {
            "az_mean": float(np.nanmean(df["az_improve_valid"].values)),
            "el_mean": float(np.nanmean(df["el_improve_valid"].values)),
        },
    }

    summary_path = os.path.join(out_split_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[JSON] {summary_path}")

    # ===== 画 best / worst / random =====
    rng = np.random.default_rng(args.plot_seed)
    score = df["music_az_mae"].values + df["music_el_mae"].values
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=1e9)
    order = np.argsort(score)

    best_idx = order[: max(args.plot_best_k, 0)]
    worst_idx = order[-max(args.plot_worst_k, 0):] if args.plot_worst_k > 0 else np.array([], dtype=int)

    rand_pool = np.arange(df.shape[0])
    rng.shuffle(rand_pool)
    random_idx = rand_pool[: max(args.plot_random_k, 0)]

    for tag, idxs in [("plots_best", best_idx), ("plots_worst", worst_idx), ("plots_random", random_idx)]:
        if idxs.size == 0:
            continue
        outp = os.path.join(out_split_dir, tag)
        safe_makedirs(outp)

        for j, ridx in enumerate(idxs):
            row = df.iloc[int(ridx)]
            h5_path = row["h5_path"]
            clip = row["clip"]

            with h5py.File(h5_path, "r") as f:
                scm_key, valid_key = find_scm_tf_keys(f)
                scm = np.asarray(f[scm_key][()]).astype(np.float32)
                valid = np.asarray(f[valid_key][()]).astype(np.uint8)
                az_gt, el_gt = infer_gt_from_labels(f, "labels")
                fw, fy = load_baseline_from_labels(f, "labels")

            N = scm.shape[0]
            L = min(N, len(valid), len(az_gt), len(el_gt))
            if fw is not None:
                L = min(L, len(fw), len(fy))

            Lp = min(L, args.max_frames_per_clip_plot)

            scm = scm[:Lp]
            valid = valid[:Lp]
            az_gt = az_gt[:Lp]
            el_gt = el_gt[:Lp]
            if fw is not None:
                fw = fw[:Lp]
                fy = fy[:Lp]

            R = (scm[..., 0] + 1j * scm[..., 1]).astype(np.complex64)

            idx_valid = np.where(valid == 1)[0]
            az_music = np.full(Lp, np.nan, dtype=np.float32)
            el_music = np.full(Lp, np.nan, dtype=np.float32)
            for i in idx_valid:
                Ri = preprocess_scm(R[i], hermitianize=hermitianize, diag_load=args.diag_load)
                if args.coarse_to_fine:
                    a, e = estimate_music_coarse_to_fine(
                        Ri,
                        coarse_grid,
                        fine_az_step=args.az_step,
                        fine_el_step=args.el_step,
                        fine_window=args.fine_window,
                        n_src=args.n_src,
                    )
                else:
                    a, e = estimate_music_single(Ri, grid, n_src=args.n_src)
                az_music[i] = a
                el_music[i] = e

            frames = np.arange(Lp, dtype=np.int32)
            out_png = os.path.join(outp, f"{j:02d}_{clip.replace('.h5','')}_music_vs_gt.png")
            plot_clip_curves(out_png, frames, az_gt, el_gt, az_music, el_music, fw, fy, valid)

    print("\nOK MUSIC 全面评估完成。")

if __name__ == "__main__":
    main()
