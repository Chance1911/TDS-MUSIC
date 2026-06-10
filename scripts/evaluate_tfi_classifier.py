
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced TFI MobileNetV3-Small test / evaluation script.

Features
--------
- Loads ImageFolder test split: data_root/test/<class>/*.png
- Supports optional manifest.csv to recover metadata (especially SNR)
- Computes:
  * loss, top-1 acc, top-k acc
  * accuracy, balanced accuracy
  * precision / recall / F1 (macro, micro, weighted)
  * MCC, Cohen's kappa
  * per-class precision / recall / F1 / specificity / NPV / FPR / FNR / AUC
  * ROC-AUC and PR-AUC (overall + per-class when possible)
  * calibration statistics (ECE / MCE / Brier-like multiclass score)
  * per-SNR metrics when SNR metadata is available
- Saves:
  * confusion matrices (count / row-normalized / column-normalized)
  * per-class metric bar charts
  * per-SNR metric bar charts
  * per-SNR class-recall heatmap
  * per-SNR confusion matrices (optional, default on)
  * reliability diagram
  * confidence histograms
  * top confusion pairs bar chart
  * CSV / JSON / Markdown summaries
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


# -------------------------------
# Transforms / dataset helpers
# -------------------------------
class KeepMiddleFreqBand(object):
    """Keep only the middle band on the frequency axis (H dimension)."""

    def __init__(self, keep_ratio: float = 0.25):
        assert 0 < keep_ratio <= 1.0
        self.keep_ratio = keep_ratio

    def __call__(self, x: torch.Tensor):
        # x: [C, H, W]
        _, H, _ = x.shape
        band_h = max(1, int(H * self.keep_ratio))
        start = (H - band_h) // 2
        end = start + band_h
        mask = torch.zeros_like(x)
        mask[:, start:end, :] = 1.0
        return x * mask


class PerImageStandardize(object):
    def __call__(self, x: torch.Tensor):
        mean = x.mean()
        std = x.std()
        return (x - mean) / (std + 1e-6)


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return img, target, path


def build_test_loader(tfi_root: str, img_size: int = 512, batch_size: int = 64, num_workers: int = 8):
    test_dir = os.path.join(tfi_root, "test")
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        PerImageStandardize(),
    ])

    test_set = ImageFolderWithPaths(root=test_dir, transform=test_transform)

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print("=== Test 数据集信息 ===")
    print(f"test: {len(test_set)} 样本, 类别: {test_set.classes}")
    print(f"class_to_idx 映射: {test_set.class_to_idx}")
    return test_loader, test_set.classes, test_set.class_to_idx


def build_test_loader(tfi_root: str, img_size: int = 512, batch_size: int = 64, num_workers: int = 8):
    test_dir = os.path.join(tfi_root, "test")
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        PerImageStandardize(),
    ])

    test_set = ImageFolderWithPaths(root=test_dir, transform=test_transform)

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print("=== Test 数据集信息 ===")
    print(f"test: {len(test_set)} 样本, 类别: {test_set.classes}")
    print(f"class_to_idx 映射: {test_set.class_to_idx}")
    return test_loader, test_set.classes, test_set.class_to_idx


# -------------------------------
# Publication-style plot settings
# -------------------------------
def set_paper_style():
    plt.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def entropy_from_probs(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)

def confidence_margin(probs: np.ndarray) -> np.ndarray:
    if probs.shape[1] < 2:
        return np.zeros((probs.shape[0],), dtype=np.float32)
    sorted_probs = np.sort(probs, axis=1)
    return sorted_probs[:, -1] - sorted_probs[:, -2]

def save_confusion_triptych(cm: np.ndarray, class_names: Sequence[str], save_path: Path, title: str = "Confusion Matrices"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    modes = [("Count", None), ("Row-normalized", "true"), ("Col-normalized", "pred")]
    for ax, (subtitle, mode) in zip(axes, modes):
        mat = cm.astype(np.float64) if mode is None else normalize_cm(cm, mode)
        im = ax.imshow(mat, cmap="Blues", aspect="auto")
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(subtitle)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                txt = f"{int(val)}" if mode is None else f"{val:.2f}"
                color = "white" if ((mode is None and val > mat.max() * 0.55) or (mode is not None and val > 0.5)) else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)
    fig.suptitle(title, y=1.02, fontsize=13)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def save_metric_dashboard(per_snr_df: pd.DataFrame, save_path: Path):
    # 2x3 dashboard for publication
    metrics = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("f1_macro", "Macro-F1"),
        ("precision_macro", "Macro Precision"),
        ("recall_macro", "Macro Recall"),
        ("ece_15", "ECE"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), dpi=300)
    axes = axes.ravel()
    x = per_snr_df["snr_db"].astype(float).to_numpy()
    for ax, (col, title) in zip(axes, metrics):
        if col not in per_snr_df.columns:
            ax.axis("off")
            continue
        y = per_snr_df[col].astype(float).to_numpy()
        ax.plot(x, y, marker="o", linewidth=2.2)
        ax.set_title(title)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(title)
        ax.grid(True, linestyle="--", alpha=0.3)
    fig.suptitle("Performance vs SNR", y=1.02, fontsize=14)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def save_confidence_by_snr(pred_df: pd.DataFrame, save_path: Path):
    df = pred_df.dropna(subset=["snr_db"]).copy()
    if len(df) == 0:
        return
    sns_snr = sorted(df["snr_db"].astype(float).unique().tolist())
    data = [df[np.isclose(df["snr_db"].astype(float), snr)]["confidence"].astype(float).to_numpy() for snr in sns_snr]
    fig = plt.figure(figsize=(max(10, len(sns_snr) * 0.45), 5.5), dpi=300)
    ax = fig.add_axes([0.08, 0.14, 0.9, 0.76])
    ax.boxplot(data, labels=[f"{s:g}" for s in sns_snr], showfliers=False)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Max softmax confidence")
    ax.set_title("Confidence Distribution by SNR")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def save_snr_metric_heatmap(per_snr_class_df: pd.DataFrame, class_names: Sequence[str], save_path: Path, value_cols: Sequence[str] = ("recall", "f1")):
    # expects long-format table with columns: snr_db, class, recall/f1...
    snr_vals = sorted(per_snr_class_df["snr_db"].astype(float).unique().tolist())
    nrows = len(value_cols)
    fig, axes = plt.subplots(nrows, 1, figsize=(max(9, len(class_names) * 0.9), max(4.5*nrows, len(snr_vals) * 0.42 * nrows)), dpi=300)
    if nrows == 1:
        axes = [axes]
    for ax, col in zip(axes, value_cols):
        pivot = per_snr_class_df.pivot_table(index="snr_db", columns="class", values=col, aggfunc="mean").reindex(index=snr_vals, columns=class_names)
        mat = pivot.to_numpy(dtype=float)
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(snr_vals)))
        ax.set_yticklabels([f"{s:g} dB" for s in snr_vals])
        ax.set_xlabel("Class")
        ax.set_ylabel("SNR")
        ax.set_title(f"{col.capitalize()} by SNR and Class")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white" if val > 0.5 else "black")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def save_error_analysis_tables(pred_df: pd.DataFrame, tables_dir: Path, top_n: int = 50):
    err_df = pred_df[pred_df["correct"] == 0].copy().sort_values(["confidence", "snr_db"], ascending=[False, True])
    if len(err_df) > 0:
        err_df.head(top_n).to_csv(tables_dir / "top_high_confidence_errors.csv", index=False)
        err_df.sort_values(["confidence", "snr_db"], ascending=[True, True]).head(top_n).to_csv(
            tables_dir / "top_low_confidence_errors.csv", index=False
        )
    correct_df = pred_df[pred_df["correct"] == 1].copy().sort_values(["confidence", "snr_db"], ascending=[True, True])
    if len(correct_df) > 0:
        correct_df.head(top_n).to_csv(tables_dir / "low_confidence_corrects.csv", index=False)

def save_confusion_pair_table(cm: np.ndarray, class_names: Sequence[str], tables_dir: Path, top_n: int = 20):
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                continue
            cnt = int(cm[i, j])
            if cnt > 0:
                pairs.append({"true": class_names[i], "pred": class_names[j], "count": cnt})
    pair_df = pd.DataFrame(sorted(pairs, key=lambda x: x["count"], reverse=True))
    if len(pair_df) > 0:
        pair_df.head(top_n).to_csv(tables_dir / "top_confusion_pairs.csv", index=False)

# -------------------------------
# Model helpers
# -------------------------------
def build_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    if pretrained:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
    else:
        model = models.mobilenet_v3_small(weights=None)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", None)
    if state is None:
        state = ckpt.get("state_dict", None)
    if state is None:
        raise KeyError("Checkpoint must contain 'model_state' or 'state_dict'.")

    # Strip accidental 'module.' prefixes
    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v

    model.load_state_dict(new_state, strict=True)
    return ckpt


# -------------------------------
# Metadata / SNR helpers
# -------------------------------
def _parse_snr_from_segment(segment: str) -> Optional[float]:
    """Parse SNR from one filename/path segment.

    Supported examples:
      snr-20, snr_-20, snr20, snr+10, snr_m20, snr_neg20, snr_minus20,
      20dB, -20dB, 20db, -20db.

    For tokens like ``snr-20`` the hyphen immediately after ``snr`` is treated
    as the sign delimiter, so the result is -20 (not +20).
    """
    if not segment:
        return None

    s = str(segment).strip()
    if not s:
        return None
    low = s.lower()

    def _parse_number(rest: str, sign: int = 1) -> Optional[float]:
        m = re.match(r"^\s*(\d+(?:\.\d+)?)", rest)
        if not m:
            return None
        try:
            return float(m.group(1)) * sign
        except Exception:
            return None

    # 1) Prefer explicit SNR tokens in the segment.
    for token in ("snr_db", "snrdb", "snr"):
        idx = low.find(token)
        if idx == -1:
            continue
        rest = s[idx + len(token):]
        rest = rest.lstrip(" 	:=_[]{}()<>|,;")
        if not rest:
            continue

        # handle optional textual negatives
        if rest.lower().startswith("minus"):
            num = _parse_number(rest[5:].lstrip(" 	:=_[]{}()<>|,;"), sign=-1)
            if num is not None:
                return num
        if rest.lower().startswith("neg"):
            num = _parse_number(rest[3:].lstrip(" 	:=_[]{}()<>|,;"), sign=-1)
            if num is not None:
                return num

        # handle compact sign encodings: snr-20, snr+10, snr_m20.
        sign = 1
        if rest[0] in "+-":
            sign = -1 if rest[0] == "-" else 1
            rest = rest[1:]
        elif rest[0].lower() == "m" and len(rest) > 1 and rest[1].isdigit():
            sign = -1
            rest = rest[1:]

        num = _parse_number(rest, sign=sign)
        if num is not None:
            return num

    # 2) Fallback: standalone numeric-with-db patterns within the segment.
    for pat in (
        re.compile(r"(?i)([+-]?\d+(?:\.\d+)?)\s*d\s*b"),
        re.compile(r"(?i)([+-]?\d+(?:\.\d+)?)\s*db"),
    ):
        m = pat.search(s)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass

    return None



def parse_snr_from_string(text: str) -> Optional[float]:
    """Extract SNR from a path or filename.

    The basename/stem are checked first so directory names do not steal the
    match. If not found, parent path segments are checked from near to far.
    """
    if text is None:
        return None
    p = Path(str(text))
    candidates: List[str] = []
    candidates.append(p.name)
    candidates.append(p.stem)
    for part in reversed(p.parts):
        if part not in candidates:
            candidates.append(part)

    for cand in candidates:
        snr = _parse_snr_from_segment(cand)
        if snr is not None:
            return snr
    return None


def normalize_path(p: str | Path) -> str:
    return str(Path(p).resolve(strict=False))


def find_manifest_csv(data_root: str, explicit: Optional[str] = None) -> Optional[Path]:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    root = Path(data_root)
    candidates.extend([
        root / "manifest.csv",
        root.parent / "manifest.csv",
        root.parent / "data_cls" / "manifest.csv",
        root.parent / "dataset" / "manifest.csv",
    ])
    for c in candidates:
        if c.exists():
            return c
    return None


def load_metadata_from_manifest(manifest_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)
    if "png_path" not in df.columns:
        raise ValueError(f"manifest {manifest_csv} missing 'png_path' column")

    # Normalize/augment columns
    for col in ["split", "label", "proxy_type", "snr_db", "h5_path", "png_path"]:
        if col not in df.columns:
            df[col] = np.nan

    df["png_path"] = df["png_path"].astype(str).map(normalize_path)

    if "h5_path" in df.columns:
        df["h5_path"] = df["h5_path"].astype(str)
    if "proxy_type" in df.columns:
        df["proxy_type"] = df["proxy_type"].astype(str)

    # File name / PNG path is authoritative for SNR if it contains the token.
    snr_vals = []
    proxy_vals = []
    split_vals = []
    for _, row in df.iterrows():
        snr = parse_snr_from_string(str(row["png_path"]))
        proxy = row.get("proxy_type", None)
        split = row.get("split", None)

        # Fallback to manifest value or H5 attrs only when the filename does not encode SNR.
        if snr is None:
            snr = row.get("snr_db", np.nan)

        h5_path = row.get("h5_path", None)
        if isinstance(h5_path, str) and h5_path and Path(h5_path).exists():
            try:
                with h5py.File(h5_path, "r") as hf:
                    if pd.isna(snr) and "snr_db" in hf.attrs:
                        snr = float(hf.attrs["snr_db"])
                    elif pd.isna(snr) and "meta_json" in hf.attrs:
                        try:
                            meta = json.loads(hf.attrs["meta_json"])
                            if "snr_db" in meta and meta["snr_db"] is not None:
                                snr = float(meta["snr_db"])
                        except Exception:
                            pass
                    if (proxy is None or proxy == "nan" or proxy == "") and "proxy_type" in hf.attrs:
                        proxy = hf.attrs["proxy_type"]
                        if isinstance(proxy, bytes):
                            proxy = proxy.decode("utf-8")
                    if (split is None or split == "nan" or split == "") and "split" in hf.attrs:
                        split = hf.attrs["split"]
                        if isinstance(split, bytes):
                            split = split.decode("utf-8")
            except Exception:
                pass

        snr_vals.append(snr)
        proxy_vals.append(proxy)
        split_vals.append(split)

    df["snr_db_resolved"] = snr_vals
    df["proxy_type_resolved"] = proxy_vals
    df["split_resolved"] = split_vals
    return df


def index_manifest_by_png(manifest_df: Optional[pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    if manifest_df is None:
        return {}
    idx = {}
    for _, row in manifest_df.iterrows():
        idx[normalize_path(row["png_path"])] = row.to_dict()
    return idx


def resolve_sample_meta(sample_path: str, manifest_idx: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    path_norm = normalize_path(sample_path)
    meta = manifest_idx.get(path_norm, {}).copy() if path_norm in manifest_idx else {}

    # File name is authoritative for SNR. Use it first.
    snr = parse_snr_from_string(path_norm)
    if snr is None:
        snr = meta.get("snr_db_resolved", meta.get("snr_db", np.nan))
        if pd.isna(snr):
            snr = None
    meta["snr_db_resolved"] = snr

    if "proxy_type_resolved" not in meta:
        meta["proxy_type_resolved"] = meta.get("proxy_type", None)
    if "split_resolved" not in meta:
        meta["split_resolved"] = meta.get("split", None)
    return meta


# -------------------------------
# Metrics
# -------------------------------
# -------------------------------
# Metrics
# -------------------------------
def compute_ece_mce(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> Tuple[float, float]:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y_true).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)
        if not np.any(mask):
            continue
        acc_bin = correct[mask].mean()
        conf_bin = confidences[mask].mean()
        gap = abs(acc_bin - conf_bin)
        weight = mask.mean()
        ece += weight * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def multiclass_brier_score(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    y_onehot = np.zeros_like(probs, dtype=np.float64)
    y_onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def safe_auc_ovr(y_true: np.ndarray, probs: np.ndarray, average: str = "macro") -> float:
    try:
        return float(roc_auc_score(y_true, probs, multi_class="ovr", average=average))
    except Exception:
        return float("nan")


def safe_ap_macro(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    y_bin = label_binarize(y_true, classes=np.arange(num_classes))
    try:
        return float(average_precision_score(y_bin, probs, average="macro"))
    except Exception:
        return float("nan")


def safe_topk_accuracy(y_true: np.ndarray, probs: np.ndarray, k: int) -> float:
    k = int(min(k, probs.shape[1]))
    try:
        return float(top_k_accuracy_score(y_true, probs, k=k, labels=np.arange(probs.shape[1])))
    except Exception:
        # fallback manual
        topk = np.argsort(-probs, axis=1)[:, :k]
        return float(np.mean([yt in row for yt, row in zip(y_true, topk)]))


def compute_overall_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, class_names: Sequence[str], topk_list: Sequence[int]):
    num_classes = len(class_names)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    metrics = {
        "num_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "kappa": float(cohen_kappa_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "roc_auc_macro_ovr": safe_auc_ovr(y_true, probs, average="macro"),
        "roc_auc_weighted_ovr": safe_auc_ovr(y_true, probs, average="weighted"),
        "pr_auc_macro_ovr": safe_ap_macro(y_true, probs, num_classes),
        "ece_15": compute_ece_mce(probs, y_true, n_bins=15)[0],
        "mce_15": compute_ece_mce(probs, y_true, n_bins=15)[1],
        "brier_multiclass": multiclass_brier_score(y_true, probs, num_classes),
    }

    for k in topk_list:
        metrics[f"top{k}_acc"] = safe_topk_accuracy(y_true, probs, k)

    # flatten report
    for key, val in report.items():
        if isinstance(val, dict):
            for subk, subv in val.items():
                metrics[f"report/{key}/{subk}"] = float(subv) if isinstance(subv, (int, float, np.number)) else subv
        else:
            metrics[f"report/{key}"] = float(val) if isinstance(val, (int, float, np.number)) else val

    return metrics, report


def per_class_dataframe(confmat: np.ndarray, class_names: Sequence[str], y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    total = confmat.sum()
    rows = []
    num_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=np.arange(num_classes))

    for i, cname in enumerate(class_names):
        tp = float(confmat[i, i])
        fn = float(confmat[i, :].sum() - tp)
        fp = float(confmat[:, i].sum() - tp)
        tn = float(total - tp - fp - fn)

        support = int(confmat[i, :].sum())
        pred_support = int(confmat[:, i].sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan

        cls_true = y_true == i
        if cls_true.sum() > 0 and (cls_true.sum() < len(y_true)):
            try:
                roc_auc = float(roc_auc_score(y_bin[:, i], probs[:, i]))
            except Exception:
                roc_auc = float("nan")
            try:
                pr_auc = float(average_precision_score(y_bin[:, i], probs[:, i]))
            except Exception:
                pr_auc = float("nan")
        else:
            roc_auc = float("nan")
            pr_auc = float("nan")

        rows.append({
            "class": cname,
            "class_idx": i,
            "support": support,
            "pred_support": pred_support,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "npv": npv,
            "fpr": fpr,
            "fnr": fnr,
            "roc_auc_ovr_binary": roc_auc,
            "pr_auc_binary": pr_auc,
            "accuracy_on_class": recall,  # same as recall for one-vs-rest
        })

    return pd.DataFrame(rows)


def evaluate_subset(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, class_names: Sequence[str], topk_list: Sequence[int]) -> Tuple[Dict[str, Any], pd.DataFrame, np.ndarray]:
    num_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    metrics, _ = compute_overall_metrics(y_true, y_pred, probs, class_names, topk_list)
    class_df = per_class_dataframe(cm, class_names, y_true, y_pred, probs)
    return metrics, class_df, cm


def normalize_cm(cm: np.ndarray, mode: str) -> np.ndarray:
    cm = cm.astype(np.float64)
    if mode == "true":
        denom = cm.sum(axis=1, keepdims=True)
    elif mode == "pred":
        denom = cm.sum(axis=0, keepdims=True)
    elif mode == "all":
        denom = np.array([[cm.sum()]])
    else:
        raise ValueError("mode must be one of: true, pred, all")
    denom = np.where(denom == 0, 1.0, denom)
    return cm / denom


# -------------------------------
# Plot helpers
# -------------------------------
def _setup_bar_axes(ax, title: str, ylabel: str, xticks: Sequence[str], rotation: int = 45):
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(xticks)))
    ax.set_xticklabels(list(xticks), rotation=rotation, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)


def save_confusion_heatmap(cm: np.ndarray, class_names: Sequence[str], save_path: Path, title: str, normalize_mode: Optional[str] = None):
    if normalize_mode is not None:
        cm_show = normalize_cm(cm, normalize_mode)
    else:
        cm_show = cm.astype(np.float64)

    fig = plt.figure(figsize=(8, 6), dpi=180)
    ax = fig.add_axes([0.12, 0.12, 0.72, 0.76])
    im = ax.imshow(cm_show, cmap="Blues")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(cm_show.shape[0]):
        for j in range(cm_show.shape[1]):
            val = cm_show[i, j]
            if normalize_mode is None:
                txt = f"{int(val)}"
            else:
                txt = f"{val:.2f}"
            color = "white" if (normalize_mode is None and val > cm_show.max() * 0.5) or (normalize_mode is not None and val > 0.5) else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=8)

    cax = fig.add_axes([0.86, 0.12, 0.03, 0.76])
    fig.colorbar(im, cax=cax)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_grouped_bar_metrics(df: pd.DataFrame, cols: Sequence[str], save_path: Path, title: str):
    x = np.arange(len(df))
    width = 0.8 / max(1, len(cols))
    fig = plt.figure(figsize=(max(10, len(df) * 0.8), 6), dpi=180)
    ax = fig.add_axes([0.08, 0.14, 0.86, 0.76])

    for i, col in enumerate(cols):
        vals = df[col].astype(float).to_numpy()
        ax.bar(x + (i - (len(cols)-1)/2) * width, vals, width=width, label=col)

    ax.set_xticks(x)
    ax.set_xticklabels(df["class"].tolist(), rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_single_bar(df: pd.DataFrame, x_col: str, y_col: str, save_path: Path, title: str, ylabel: str, rotation: int = 45, figsize=(12, 5)):
    x = np.arange(len(df))
    fig = plt.figure(figsize=figsize, dpi=180)
    ax = fig.add_axes([0.08, 0.18, 0.88, 0.74])
    ax.bar(x, df[y_col].astype(float).to_numpy())
    ax.set_xticks(x)
    ax.set_xticklabels(df[x_col].astype(str).tolist(), rotation=rotation, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_line_plot(df: pd.DataFrame, x_col: str, y_cols: Sequence[str], save_path: Path, title: str, ylabel: str):
    fig = plt.figure(figsize=(12, 5), dpi=180)
    ax = fig.add_axes([0.08, 0.18, 0.88, 0.74])
    x = np.arange(len(df))
    for col in y_cols:
        ax.plot(x, df[col].astype(float).to_numpy(), marker="o", linewidth=2, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(df[x_col].astype(str).tolist(), rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_reliability_diagram(y_true: np.ndarray, probs: np.ndarray, save_path: Path, n_bins: int = 15):
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == y_true).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    accs = []
    confs = []
    weights = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        if i < n_bins - 1:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)
        if not np.any(mask):
            continue
        bin_centers.append((lo + hi) / 2)
        accs.append(correct[mask].mean())
        confs.append(confidences[mask].mean())
        weights.append(mask.mean())

    fig = plt.figure(figsize=(7, 6), dpi=180)
    ax = fig.add_axes([0.14, 0.12, 0.78, 0.8])
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=2, label="Perfect calibration")
    ax.plot(confs, accs, marker="o", linewidth=2, label="Model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_confidence_histograms(confidences: np.ndarray, correct_mask: np.ndarray, save_path: Path):
    fig = plt.figure(figsize=(8, 5), dpi=180)
    ax = fig.add_axes([0.1, 0.15, 0.85, 0.78])
    ax.hist(confidences[correct_mask], bins=20, alpha=0.65, label="Correct")
    ax.hist(confidences[~correct_mask], bins=20, alpha=0.65, label="Incorrect")
    ax.set_xlabel("Max softmax confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_top_confusions(cm: np.ndarray, class_names: Sequence[str], save_path: Path, top_n: int = 10):
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                continue
            c = int(cm[i, j])
            if c > 0:
                pairs.append((c, class_names[i], class_names[j]))
    pairs.sort(reverse=True, key=lambda x: x[0])
    pairs = pairs[:top_n]
    if not pairs:
        return
    labels = [f"{t}->{p}" for _, t, p in pairs]
    counts = [c for c, _, _ in pairs]

    fig = plt.figure(figsize=(10, 5), dpi=180)
    ax = fig.add_axes([0.08, 0.18, 0.88, 0.74])
    ax.bar(np.arange(len(pairs)), counts)
    ax.set_xticks(np.arange(len(pairs)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Top Misclassification Pairs")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_snr_class_heatmap(per_snr_class_df: pd.DataFrame, class_names: Sequence[str], save_path: Path, value_col_prefix: str = "recall"):
    snr_vals = per_snr_class_df["snr_db"].astype(float).tolist()
    mat = []
    for _, row in per_snr_class_df.iterrows():
        mat.append([float(row.get(c, np.nan)) for c in class_names])
    mat = np.array(mat, dtype=float)

    fig = plt.figure(figsize=(max(8, len(class_names) * 1.1), max(6, len(snr_vals) * 0.35)), dpi=180)
    ax = fig.add_axes([0.16, 0.12, 0.7, 0.78])
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(snr_vals)))
    ax.set_yticklabels([f"{s:g} dB" for s in snr_vals])
    ax.set_xlabel("Class")
    ax.set_ylabel("SNR")
    ax.set_title(f"{value_col_prefix.capitalize()} by SNR and Class")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white" if val > 0.5 else "black")

    cax = fig.add_axes([0.88, 0.12, 0.03, 0.78])
    fig.colorbar(im, cax=cax)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# -------------------------------
# Evaluation core
# -------------------------------
@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    running_loss = 0.0
    total = 0

    all_true: List[int] = []
    all_pred: List[int] = []
    all_probs: List[np.ndarray] = []
    all_paths: List[str] = []

    pbar = tqdm(loader, desc="Test", ncols=100)
    for imgs, labels, paths in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        running_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)

        all_true.extend(labels.cpu().numpy().tolist())
        all_pred.extend(preds.cpu().numpy().tolist())
        all_probs.append(probs.cpu().numpy())
        all_paths.extend(list(paths))

    y_true = np.asarray(all_true, dtype=np.int64)
    y_pred = np.asarray(all_pred, dtype=np.int64)
    probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 0), dtype=np.float32)
    avg_loss = running_loss / max(1, total)
    return avg_loss, y_true, y_pred, probs, all_paths



def evaluate_and_export(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    sample_paths: Sequence[str],
    class_names: Sequence[str],
    output_dir: Path,
    manifest_idx: Dict[str, Dict[str, Any]],
    topk_list: Sequence[int],
    save_per_snr_confmat: bool = True,
    save_roc_pr: bool = True,
    save_calibration: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"
    paper_plots_dir = plots_dir / "paper"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    paper_plots_dir.mkdir(parents=True, exist_ok=True)

    # sample-level dataframe
    records = []
    confidences = probs.max(axis=1)
    margins = confidence_margin(probs)
    entropies = entropy_from_probs(probs)
    for i, path in enumerate(sample_paths):
        meta = resolve_sample_meta(path, manifest_idx)
        snr = meta.get("snr_db_resolved", None)
        try:
            snr = float(snr) if snr is not None and not pd.isna(snr) else None
        except Exception:
            snr = None

        topk = {}
        for k in topk_list:
            kk = min(int(k), probs.shape[1])
            topk_idx = np.argsort(-probs[i])[:kk]
            topk[f"top{kk}_hit"] = int(int(y_true[i]) in topk_idx)

        rec = {
            "index": i,
            "path": path,
            "true_idx": int(y_true[i]),
            "true_label": class_names[int(y_true[i])],
            "pred_idx": int(y_pred[i]),
            "pred_label": class_names[int(y_pred[i])],
            "correct": int(int(y_true[i]) == int(y_pred[i])),
            "confidence": float(confidences[i]),
            "margin": float(margins[i]),
            "entropy": float(entropies[i]),
            "snr_db": snr,
            "proxy_type": meta.get("proxy_type_resolved", None),
            "split": meta.get("split_resolved", None),
        }
        for k, v in topk.items():
            rec[k] = int(v)
        for j, cname in enumerate(class_names):
            rec[f"prob_{cname}"] = float(probs[i, j])
        records.append(rec)

    pred_df = pd.DataFrame(records)
    pred_df.to_csv(tables_dir / "predictions.csv", index=False)
    pred_df.to_csv(paper_plots_dir / "predictions_snapshot.csv", index=False)

    # overall metrics
    overall_metrics, overall_report = compute_overall_metrics(y_true, y_pred, probs, class_names, topk_list)
    overall_metrics["mean_confidence"] = float(np.mean(confidences)) if len(confidences) else float("nan")
    overall_metrics["median_confidence"] = float(np.median(confidences)) if len(confidences) else float("nan")
    overall_metrics["mean_margin"] = float(np.mean(margins)) if len(margins) else float("nan")
    overall_metrics["mean_entropy"] = float(np.mean(entropies)) if len(entropies) else float("nan")
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, ensure_ascii=False, indent=2)

    # confusion matrices
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    np.savetxt(tables_dir / "confusion_matrix_count.csv", cm, fmt="%d", delimiter=",")
    pd.DataFrame(normalize_cm(cm, "true"), index=class_names, columns=class_names).to_csv(tables_dir / "confusion_matrix_row_norm.csv")
    pd.DataFrame(normalize_cm(cm, "pred"), index=class_names, columns=class_names).to_csv(tables_dir / "confusion_matrix_col_norm.csv")

    save_confusion_heatmap(cm, class_names, plots_dir / "confusion_matrix_count.png", "Confusion Matrix (Count)")
    save_confusion_heatmap(cm, class_names, plots_dir / "confusion_matrix_row_norm.png", "Confusion Matrix (Row-normalized)", normalize_mode="true")
    save_confusion_heatmap(cm, class_names, plots_dir / "confusion_matrix_col_norm.png", "Confusion Matrix (Col-normalized)", normalize_mode="pred")
    save_confusion_triptych(cm, class_names, paper_plots_dir / "confusion_matrices_triptych.png", "Confusion Matrix Summary")
    save_top_confusions(cm, class_names, plots_dir / "top_misclassification_pairs.png", top_n=10)
    save_confusion_pair_table(cm, class_names, tables_dir, top_n=20)

    # per-class metrics
    per_class_df = per_class_dataframe(cm, class_names, y_true, y_pred, probs)
    per_class_df.to_csv(tables_dir / "per_class_metrics.csv", index=False)
    per_class_df.sort_values("f1", ascending=False).to_csv(tables_dir / "per_class_metrics_sorted_by_f1.csv", index=False)
    per_class_df.sort_values("recall", ascending=False).to_csv(tables_dir / "per_class_metrics_sorted_by_recall.csv", index=False)

    save_grouped_bar_metrics(per_class_df, ["precision", "recall", "f1"], plots_dir / "per_class_precision_recall_f1.png", "Per-class Precision / Recall / F1")
    save_grouped_bar_metrics(per_class_df, ["specificity", "npv", "fpr", "fnr"], plots_dir / "per_class_specificity_npv_fpr_fnr.png", "Per-class Specificity / NPV / FPR / FNR")
    save_single_bar(per_class_df, "class", "support", plots_dir / "per_class_support.png", "Per-class Support", "Count", rotation=45, figsize=(10, 5))
    save_single_bar(per_class_df, "class", "f1", paper_plots_dir / "per_class_f1_bar.png", "Per-class F1", "F1", rotation=45, figsize=(10, 5))
    save_single_bar(per_class_df, "class", "recall", paper_plots_dir / "per_class_recall_bar.png", "Per-class Recall", "Recall", rotation=45, figsize=(10, 5))

    # high-confidence errors / low-confidence corrects
    save_error_analysis_tables(pred_df, tables_dir, top_n=50)

    # ROC / PR curves
    if save_roc_pr:
        try:
            from sklearn.metrics import roc_curve, auc, precision_recall_curve
            y_bin = label_binarize(y_true, classes=np.arange(len(class_names)))

            fig = plt.figure(figsize=(7.5, 6.2), dpi=300)
            ax = fig.add_axes([0.13, 0.12, 0.82, 0.8])
            for i, cname in enumerate(class_names):
                if y_bin[:, i].sum() == 0 or y_bin[:, i].sum() == len(y_true):
                    continue
                fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, linewidth=1.8, label=f"{cname} (AUC={roc_auc:.3f})")
            ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="gray")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curves (OvR)")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(fontsize=8, ncols=1)
            fig.savefig(plots_dir / "roc_curves_ovr.png", bbox_inches="tight")
            plt.close(fig)

            fig = plt.figure(figsize=(7.5, 6.2), dpi=300)
            ax = fig.add_axes([0.13, 0.12, 0.82, 0.8])
            for i, cname in enumerate(class_names):
                if y_bin[:, i].sum() == 0:
                    continue
                prec, rec, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
                ap = average_precision_score(y_bin[:, i], probs[:, i])
                ax.plot(rec, prec, linewidth=1.8, label=f"{cname} (AP={ap:.3f})")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall Curves (OvR)")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(fontsize=8, ncols=1)
            fig.savefig(plots_dir / "pr_curves_ovr.png", bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"Warning ROC/PR plot skipped: {e}")

    # calibration / confidence
    correct_mask = (y_true == y_pred)
    save_confidence_histograms(confidences, correct_mask, plots_dir / "confidence_histogram.png")
    save_confidence_by_snr(pred_df, paper_plots_dir / "confidence_box_by_snr.png")
    if save_calibration:
        save_reliability_diagram(y_true, probs, plots_dir / "reliability_diagram.png", n_bins=15)

    # per-snr metrics if available
    known_snr_mask = pred_df["snr_db"].notna()
    snr_df = pred_df.loc[known_snr_mask].copy()
    per_snr_metrics_rows = []
    per_snr_class_rows = []
    per_snr_cm_dir = plots_dir / "per_snr_confmats"
    if save_per_snr_confmat:
        per_snr_cm_dir.mkdir(parents=True, exist_ok=True)

    if len(snr_df) > 0:
        snr_values = sorted(snr_df["snr_db"].astype(float).unique().tolist())
        for snr in snr_values:
            sub = snr_df[np.isclose(snr_df["snr_db"].astype(float), float(snr))]
            idx = sub.index.to_numpy()
            yt = y_true[idx]
            yp = y_pred[idx]
            pr = probs[idx]

            metrics_s, class_df_s, cm_s = evaluate_subset(yt, yp, pr, class_names, topk_list)
            metrics_s["snr_db"] = float(snr)
            metrics_s["num_samples"] = int(len(sub))
            metrics_s["error_rate"] = float(1.0 - np.mean(yt == yp))
            metrics_s["mean_confidence"] = float(sub["confidence"].mean())
            metrics_s["median_confidence"] = float(sub["confidence"].median())
            metrics_s["mean_margin"] = float(sub["margin"].mean())
            metrics_s["mean_entropy"] = float(sub["entropy"].mean())
            per_snr_metrics_rows.append(metrics_s)

            # long-format per-class rows
            for _, r in class_df_s.iterrows():
                rr = r.to_dict()
                rr["snr_db"] = float(snr)
                rr["num_samples"] = int(len(sub))
                per_snr_class_rows.append(rr)

            if save_per_snr_confmat:
                save_confusion_heatmap(
                    cm_s,
                    class_names,
                    per_snr_cm_dir / f"confmat_snr_{snr:g}dB_row_norm.png",
                    title=f"Confusion Matrix @ {snr:g} dB (Row-normalized)",
                    normalize_mode="true",
                )
                pd.DataFrame(cm_s, index=class_names, columns=class_names).to_csv(per_snr_cm_dir / f"confmat_snr_{snr:g}dB_count.csv")
                pd.DataFrame(normalize_cm(cm_s, "true"), index=class_names, columns=class_names).to_csv(per_snr_cm_dir / f"confmat_snr_{snr:g}dB_row_norm.csv")

        per_snr_df = pd.DataFrame(per_snr_metrics_rows).sort_values("snr_db")
        per_snr_df.to_csv(tables_dir / "per_snr_metrics.csv", index=False)

        per_snr_class_long_df = pd.DataFrame(per_snr_class_rows)
        per_snr_class_long_df.to_csv(tables_dir / "per_snr_class_metrics_long.csv", index=False)

        # Wide tables for direct plotting / paper
        per_snr_class_recall_df = per_snr_class_long_df.pivot_table(index="snr_db", columns="class", values="recall", aggfunc="mean").reset_index()
        per_snr_class_f1_df = per_snr_class_long_df.pivot_table(index="snr_db", columns="class", values="f1", aggfunc="mean").reset_index()
        per_snr_class_recall_df.to_csv(tables_dir / "per_snr_class_recall.csv", index=False)
        per_snr_class_f1_df.to_csv(tables_dir / "per_snr_class_f1.csv", index=False)

        save_metric_dashboard(per_snr_df, paper_plots_dir / "metric_dashboard_by_snr.png")
        save_single_bar(per_snr_df, "snr_db", "accuracy", plots_dir / "accuracy_by_snr.png", "Accuracy by SNR", "Accuracy", rotation=0, figsize=(14, 5))
        save_single_bar(per_snr_df, "snr_db", "f1_macro", plots_dir / "macro_f1_by_snr.png", "Macro-F1 by SNR", "Macro-F1", rotation=0, figsize=(14, 5))
        save_single_bar(per_snr_df, "snr_db", "balanced_accuracy", plots_dir / "balanced_accuracy_by_snr.png", "Balanced Accuracy by SNR", "Balanced Accuracy", rotation=0, figsize=(14, 5))
        if "top1_acc" in per_snr_df.columns:
            save_single_bar(per_snr_df, "snr_db", "top1_acc", plots_dir / "top1_accuracy_by_snr.png", "Top-1 Accuracy by SNR", "Top-1 Accuracy", rotation=0, figsize=(14, 5))
        save_line_plot(per_snr_df, "snr_db", ["accuracy", "f1_macro", "balanced_accuracy"], paper_plots_dir / "accuracy_f1_balacc_by_snr.png", "Core Metrics vs SNR", "Score")
        if "ece_15" in per_snr_df.columns:
            save_line_plot(per_snr_df, "snr_db", ["ece_15", "mce_15"], paper_plots_dir / "calibration_by_snr.png", "Calibration vs SNR", "Calibration Error")
        save_snr_metric_heatmap(per_snr_class_long_df, class_names, plots_dir / "recall_heatmap_snr_vs_class.png", value_cols=("recall", "f1"))
        save_snr_metric_heatmap(per_snr_class_long_df, class_names, paper_plots_dir / "recall_f1_heatmap_snr_vs_class.png", value_cols=("recall", "f1"))
    else:
        per_snr_df = pd.DataFrame(columns=["snr_db"])
        per_snr_class_long_df = pd.DataFrame(columns=["snr_db"])
        print("Warning 没有可用的 SNR 元数据，跳过 per-SNR 统计。")
        per_snr_df.to_csv(tables_dir / "per_snr_metrics.csv", index=False)
        per_snr_class_long_df.to_csv(tables_dir / "per_snr_class_metrics_long.csv", index=False)

    # group by predicted / true distribution
    pred_dist = pd.DataFrame({
        "class": list(class_names),
        "true_count": [int((y_true == i).sum()) for i in range(len(class_names))],
        "pred_count": [int((y_pred == i).sum()) for i in range(len(class_names))],
    })
    pred_dist.to_csv(tables_dir / "label_distribution.csv", index=False)
    save_grouped_bar_metrics(pred_dist, ["true_count", "pred_count"], plots_dir / "true_vs_pred_distribution.png", "True vs Predicted Distribution")
    save_single_bar(pred_dist, "class", "true_count", plots_dir / "true_distribution.png", "True Label Distribution", "Count", rotation=45, figsize=(10, 5))
    save_single_bar(pred_dist, "class", "pred_count", plots_dir / "pred_distribution.png", "Predicted Label Distribution", "Count", rotation=45, figsize=(10, 5))

    # sample-level CSV derived summary per SNR
    if len(snr_df) > 0:
        snr_group_summary = snr_df.groupby("snr_db").agg(
            num_samples=("correct", "size"),
            acc=("correct", "mean"),
            error_rate=("correct", lambda x: 1.0 - float(np.mean(x))),
            mean_confidence=("confidence", "mean"),
            median_confidence=("confidence", "median"),
            mean_margin=("margin", "mean"),
            mean_entropy=("entropy", "mean"),
        ).reset_index()
        snr_group_summary.to_csv(tables_dir / "per_snr_sample_summary.csv", index=False)
        # high-level compact plot
        save_line_plot(snr_group_summary, "snr_db", ["acc"], paper_plots_dir / "accuracy_by_snr_sample_summary.png", "Accuracy by SNR (sample summary)", "Accuracy")
    else:
        pd.DataFrame().to_csv(tables_dir / "per_snr_sample_summary.csv", index=False)

    # write markdown report
    md = []
    md.append("# TFI Evaluation Report")
    md.append("")
    md.append("## Overall Metrics")
    for k in sorted([k for k in overall_metrics.keys() if not k.startswith("report/")]):
        v = overall_metrics[k]
        md.append(f"- **{k}**: {v}")
    md.append("")
    md.append("## Per-class Metrics")
    md.append(per_class_df.to_markdown(index=False))
    md.append("")
    if len(snr_df) > 0:
        md.append("## Per-SNR Metrics")
        md.append(per_snr_df.to_markdown(index=False))
        md.append("")
        md.append("## Per-SNR Class Metrics (Long Format)")
        md.append(per_snr_class_long_df.to_markdown(index=False))
        md.append("")
    md.append("## Classification Report (sklearn)")
    md.append(pd.DataFrame(overall_report).T.to_markdown())
    md.append("")
    with open(output_dir / "evaluation_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    return {
        "overall_metrics": overall_metrics,
        "per_class_df": per_class_df,
        "per_snr_df": per_snr_df,
        "per_snr_class_df": per_snr_class_long_df,
        "confusion_matrix": cm,
        "predictions_df": pred_df,
    }

# -------------------------------
# CLI

# -------------------------------
def parse_topk(s: str) -> List[int]:
    vals = []
    for part in re.split(r"[,\s]+", s.strip()):
        if not part:
            continue
        vals.append(int(part))
    return sorted({v for v in vals if v > 0})


def parse_args():
    parser = argparse.ArgumentParser(description="TFI MobileNet UAV Type Classification - Enhanced Test")

    parser.add_argument("--data_root", type=str, required=True, help="TFI dataset root containing test/")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--manifest_csv", type=str, default=None, help="Optional manifest.csv for png->h5 metadata mapping")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--topk", type=str, default="1,3,5", help="Top-k accuracy values, comma-separated")
    parser.add_argument("--save_per_snr_confmat", action="store_true", help="Save per-SNR confusion matrices (on by default if metadata exists)")
    parser.add_argument("--no_save_per_snr_confmat", action="store_true", help="Disable per-SNR confusion matrices")
    parser.add_argument("--no_roc_pr", action="store_true", help="Disable ROC/PR plots")
    parser.add_argument("--no_calibration", action="store_true", help="Disable calibration plot")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights as initialization before loading checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_paper_style()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    test_loader, class_names, test_class_to_idx = build_test_loader(
        tfi_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    num_classes = len(class_names)

    print(f"\n从 checkpoint 加载模型: {args.checkpoint}")
    model = build_model(num_classes, pretrained=args.pretrained).to(device)
    ckpt = load_checkpoint(model, args.checkpoint, device)
    print("OK 模型权重加载完成。")

    # Sanity check class mapping
    train_map = ckpt.get("meta", {}).get("class_to_idx", None)
    if train_map is not None:
        train_items = sorted(train_map.items(), key=lambda x: x[1])
        test_items = sorted(test_class_to_idx.items(), key=lambda x: x[1])
        if train_items != test_items:
            print("Warning 警告：test 集的 class_to_idx 与训练时不一致，注意检查数据/类别映射")
            print(f"  训练时: {train_map}")
            print(f"  测试时: {test_class_to_idx}")
        else:
            print("OK 训练/测试的 class_to_idx 一致。")

    criterion = nn.CrossEntropyLoss()
    test_loss, y_true, y_pred, probs, sample_paths = collect_predictions(model, test_loader, criterion, device)
    print(f"\n📊 Test loss = {test_loss:.6f}")

    topk_list = parse_topk(args.topk)

    manifest_csv = find_manifest_csv(args.data_root, args.manifest_csv)
    manifest_df = load_metadata_from_manifest(manifest_csv) if manifest_csv is not None else None
    manifest_idx = index_manifest_by_png(manifest_df)

    results = evaluate_and_export(
        y_true=y_true,
        y_pred=y_pred,
        probs=probs,
        sample_paths=sample_paths,
        class_names=class_names,
        output_dir=Path(args.output_dir),
        manifest_idx=manifest_idx,
        topk_list=topk_list,
        save_per_snr_confmat=(not args.no_save_per_snr_confmat),
        save_roc_pr=(not args.no_roc_pr),
        save_calibration=(not args.no_calibration),
    )

    # Save concise console summary
    overall = results["overall_metrics"]
    print("\n=== Overall Metrics ===")
    for k in [
        "accuracy", "balanced_accuracy",
        "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted",
        "mcc", "kappa",
        "roc_auc_macro_ovr", "pr_auc_macro_ovr",
        "ece_15", "mce_15", "brier_multiclass",
    ]:
        if k in overall:
            print(f"{k:>24s}: {overall[k]:.6f}")
    for k in topk_list:
        kk = f"top{k}_acc"
        if kk in overall:
            print(f"{kk:>24s}: {overall[kk]:.6f}")

    # announce if SNR stats exist
    if len(results["per_snr_df"]) > 0:
        print("\nOK 已生成不同 SNR 的统计结果与图表。")
    else:
        print("\nWarning 没有检测到 SNR 元数据；如需 per-SNR 统计，请提供 manifest.csv 或确保 png 路径/ H5 metadata 可解析 SNR。")

    print(f"\n结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()