#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
build_tf_support.py

功能：
- 使用已训练好的 MobileNetV3-Small 识别模型，
  对 data_root 下的 train/val/test 全部 TFI 图像：
    1) 计算 TF evidence map（不落盘）
    2) 基于 support + TFI 做时间/频率分析 + 能量门控 + 谷值切分
    3) 只保存 STFT 坐标系下的 TF 掩码：
         <stem>_mask_tf.npy : [T,F] (time,freq)

目录假设：
  data_root/
    train/T01/*.png
    train/T02/*.png
    val/T01/*.png
    test/Txx/*.png

生成：
  mask_root/
    train/T01/<stem>_mask_tf.npy
    val/Txx/<stem>_mask_tf.npy
    test/Txx/<stem>_mask_tf.npy

其中：
  - TFI 生成时约定：mag = sqrt(real^2+imag^2); img = flipud(mag.T)
  - 此处 mask_tf = mask_img[::-1, :].T，对应 STFT 的 [T,F]
"""

import os
import glob
import argparse
import json

from typing import Optional
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


# ---------------- 通用预处理 ----------------

class PerImageStandardize(object):
    def __call__(self, x: torch.Tensor):
        mean = x.mean()
        std = x.std()
        return (x - mean) / (std + 1e-6)


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


class FolderDatasetWithFixedMapping(Dataset):
    def __init__(self, split_dir: str, class_to_idx: dict, transform=None):
        self.split_dir = split_dir
        self.class_to_idx = {str(k): int(v) for k, v in class_to_idx.items()}
        self.transform = transform
        self.samples = []

        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"split dir 不存在: {split_dir}")

        classes_present = []
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            class_name = str(class_name)
            if class_name not in self.class_to_idx:
                raise KeyError(f"checkpoint class_to_idx 不包含类别目录: {class_name}")
            classes_present.append(class_name)
            for root, _, files in os.walk(class_dir):
                for fname in sorted(files):
                    if not fname.lower().endswith(IMG_EXTS):
                        continue
                    img_path = os.path.join(root, fname)
                    self.samples.append((img_path, self.class_to_idx[class_name]))

        if not self.samples:
            raise RuntimeError(f"{split_dir} 下没有找到任何图像样本")

        self.classes = sorted(classes_present, key=lambda x: self.class_to_idx[x])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_loader_for_split(data_root, split, img_size=512, batch_size=64, num_workers=8, class_to_idx: Optional[dict] = None):
    split_dir = os.path.join(data_root, split)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        PerImageStandardize(),
    ])
    if class_to_idx is None:
        dataset = datasets.ImageFolder(root=split_dir, transform=transform)
    else:
        dataset = FolderDatasetWithFixedMapping(split_dir=split_dir, class_to_idx=class_to_idx, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    print(f"=== {split} 集信息 ===")
    print(f"{split}: {len(dataset)} 样本, 类别: {dataset.classes}")
    if hasattr(dataset, "class_to_idx"):
        print(f"class_to_idx: {dataset.class_to_idx}")
    return loader, dataset


def load_checkpoint_class_to_idx(ckpt: dict):
    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
    class_to_idx = meta.get("class_to_idx", None)
    if isinstance(class_to_idx, dict) and class_to_idx:
        out = {}
        for k, v in class_to_idx.items():
            out[str(k)] = int(v)
        return out
    return None


def infer_num_classes(data_root: str, requested_splits, ckpt: dict) -> int:
    class_to_idx = load_checkpoint_class_to_idx(ckpt)
    if class_to_idx:
        return len(class_to_idx)

    discovered_class_names = set()
    for split in requested_splits:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            continue
        for name in os.listdir(split_dir):
            full = os.path.join(split_dir, name)
            if os.path.isdir(full):
                discovered_class_names.add(str(name))
    if discovered_class_names:
        return len(discovered_class_names)

    raise RuntimeError(
        "无法确定类别数：数据目录中没有可用类别子目录，checkpoint 里也没有 meta.class_to_idx"
    )


# ---------------- MobileNetV3-Small 模型 ----------------

def build_model(num_classes, pretrained=False):
    if pretrained:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
    else:
        model = models.mobilenet_v3_small(weights=None)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model


# ---------------- TF evidence map ----------------

class TFEvidenceOperator:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def __call__(self, x: torch.Tensor, class_idx: torch.Tensor):
        """
        x: [B,3,H,W]
        class_idx: [B] 需要可视化的类别
        返回:
          cams: [B,Hf,Wf]，归一化到 [0,1]
        """
        B = x.size(0)
        self.model.zero_grad()
        logits = self.model(x)  # [B,num_classes]

        one_hot = torch.zeros_like(logits)
        one_hot[torch.arange(B), class_idx] = 1.0
        loss = (logits * one_hot).sum()

        loss.backward()

        A = self.activations  # [B,C,Hf,Wf]
        G = self.gradients    # [B,C,Hf,Wf]

        weights = G.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]
        cam = (weights * A).sum(dim=1)              # [B,Hf,Wf]
        cam = torch.relu(cam)

        cams = []
        for b in range(B):
            c = cam[b]
            c_min = c.min()
            c_max = c.max()
            if (c_max - c_min) < 1e-6:
                c_norm = torch.zeros_like(c)
            else:
                c_norm = (c - c_min) / (c_max - c_min)
            cams.append(c_norm)
        cams = torch.stack(cams, dim=0)
        return cams


# ---------------- TF-MASK 生成工具 ----------------

def _normalize_01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max > x_min:
        return (x - x_min) / (x_max - x_min)
    else:
        return np.zeros_like(x, dtype=np.float32)


def find_freq_segments_by_threshold(
    score: np.ndarray,
    thr: float,
    min_len_frac: float = 0.01,
):
    H = score.shape[0]
    s_norm = _normalize_01(score)
    above = s_norm >= thr
    min_len = max(1, int(H * min_len_frac))

    segments = []
    i = 0
    while i < H:
        if not above[i]:
            i += 1
            continue
        j = i + 1
        while j < H and above[j]:
            j += 1
        if j - i >= min_len:
            segments.append((i, j - 1))
        i = j

    return segments, s_norm


def find_segments_1d(
    score: np.ndarray,
    thr_main: float,
    thr_valley: Optional[float] = None,
    min_len: int = 3,
    min_gap: int = 3,
):
    L = score.shape[0]
    segments = []

    above_main = score >= thr_main
    i = 0
    while i < L:
        if not above_main[i]:
            i += 1
            continue
        j = i + 1
        while j < L and above_main[j]:
            j += 1

        if j - i < min_len:
            i = j
            continue

        if thr_valley is None:
            segments.append((i, j - 1))
        else:
            seg = score[i:j]
            low_mask = seg < thr_valley

            k = 0
            seg_start = i
            while k < seg.shape[0]:
                if not low_mask[k]:
                    k += 1
                    continue
                l = k + 1
                while l < seg.shape[0] and low_mask[l]:
                    l += 1

                if l - k >= min_gap:
                    part_start = seg_start
                    part_end = i + k - 1
                    if part_end >= part_start and (part_end - part_start + 1) >= min_len:
                        segments.append((part_start, part_end))
                    seg_start = i + l

                k = l

            if seg_start <= j - 1 and (j - seg_start) >= min_len:
                segments.append((seg_start, j - 1))

        i = j

    return segments


def analyze_cam_basic(cam: np.ndarray, freq_thresh: float, freq_min_len_frac: float):
    H, W = cam.shape
    time_score = cam.mean(axis=0)  # [W]
    freq_score = cam.mean(axis=1)  # [H]

    freq_segments_idx, freq_score_norm = find_freq_segments_by_threshold(
        freq_score,
        thr=freq_thresh,
        min_len_frac=freq_min_len_frac,
    )
    time_score_norm = _normalize_01(time_score)

    return time_score_norm, freq_score_norm, freq_segments_idx


def refine_time_segments_with_energy(
    time_score_norm: np.ndarray,
    img_gray: np.ndarray,
    freq_segments_idx,
    energy_percentile: float = 10.0,
    time_thr_main: float = 0.3,
    time_thr_valley: float = 0.15,
    time_min_len_frac: float = 0.01,
    time_min_gap_frac: float = 0.02,
):
    H, W = img_gray.shape

    freq_mask = np.zeros(H, dtype=bool)
    for fs, fe in freq_segments_idx:
        fs0 = max(0, min(H - 1, fs))
        fe0 = max(0, min(H - 1, fe))
        if fe0 >= fs0:
            freq_mask[fs0:fe0 + 1] = True
    if not freq_mask.any():
        freq_mask[:] = True

    band_slice = img_gray[freq_mask, :]    # [H_band, W]
    energy_time = band_slice.mean(axis=0)  # [W]
    energy_norm = _normalize_01(energy_time)

    e_thr = np.percentile(energy_norm, energy_percentile)
    energy_mask = energy_norm > e_thr

    time_score_gated = time_score_norm.copy()
    time_score_gated[~energy_mask] = 0.0
    time_score_gated_norm = _normalize_01(time_score_gated)

    min_len = max(1, int(W * time_min_len_frac))
    min_gap = max(1, int(W * time_min_gap_frac))

    candidate_segments = find_segments_1d(
        time_score_gated_norm,
        thr_main=time_thr_main,
        thr_valley=time_thr_valley,
        min_len=min_len,
        min_gap=min_gap,
    )

    final_segments = []
    for (s, e) in candidate_segments:
        i = s
        while i <= e:
            if not energy_mask[i]:
                i += 1
                continue

            j = i + 1
            while j <= e and energy_mask[j]:
                j += 1

            if (j - i) >= min_len:
                final_segments.append((i, j - 1))

            i = j

    return time_score_gated_norm, final_segments, energy_norm, energy_mask


def build_mask_from_cam_array(
    img_path: str,
    cam_resized: np.ndarray,
    time_thresh=0.3,
    time_valley_thresh=0.15,
    time_min_len_frac=0.01,
    time_min_gap_frac=0.02,
    energy_percentile=10.0,
    freq_thresh=0.3,
    freq_min_len_frac=0.01,
):
    """
    输入：
      img_path   : 原始 TFI PNG 路径
      cam_resized: [H,W]，与模型输入同尺寸的 CAM（0~1）

    输出：
      mask_tf: [T,F] (time,freq) 的 TF 掩码（float32，0/1）
    """
    cam = cam_resized.astype(np.float32)
    Hc, Wc = cam.shape

    # 打开原图并 resize 到 CAM 大小，保证对齐
    img = Image.open(img_path).convert("RGB")
    img = img.resize((Wc, Hc), resample=Image.BILINEAR)
    img_np = np.array(img)  # [H,W,3]
    gray = img_np.mean(axis=2).astype(np.float32)  # [H,W]

    H, W = gray.shape
    assert (H, W) == (Hc, Wc), f"CAM 与图像尺寸不一致: cam={cam.shape}, img={gray.shape}"

    # 1) CAM 基础统计
    time_score_norm0, freq_score_norm, freq_segments_idx = analyze_cam_basic(
        cam,
        freq_thresh=freq_thresh,
        freq_min_len_frac=freq_min_len_frac,
    )

    # 2) 时间方向：能量门控 + 谷值切分
    _, time_segments_idx, _, _ = refine_time_segments_with_energy(
        time_score_norm=time_score_norm0,
        img_gray=gray,
        freq_segments_idx=freq_segments_idx,
        energy_percentile=energy_percentile,
        time_thr_main=time_thresh,
        time_thr_valley=time_valley_thresh,
        time_min_len_frac=time_min_len_frac,
        time_min_gap_frac=time_min_gap_frac,
    )

    # 3) 生成 TFI 坐标掩码 [H,W]
    freq_mask = np.zeros(H, dtype=np.float32)
    for fs_idx, fe_idx in freq_segments_idx:
        fs0 = max(0, min(H - 1, fs_idx))
        fe0 = max(0, min(H - 1, fe_idx))
        if fe0 >= fs0:
            freq_mask[fs0:fe0 + 1] = 1.0
    if not freq_mask.any():
        freq_mask[:] = 1.0

    time_mask = np.zeros(W, dtype=np.float32)
    for ts, te in time_segments_idx:
        ts0 = max(0, min(W - 1, ts))
        te0 = max(0, min(W - 1, te))
        if te0 >= ts0:
            time_mask[ts0:te0 + 1] = 1.0

    mask_img = freq_mask[:, None] * time_mask[None, :]   # [H,W] = [freq,time]

    # 4) 映射到 STFT 坐标 [T,F]：
    #    STFT(T,F) -> mag -> mag.T(F,T) -> flipud(F,T)=img
    #    => mask_tf = (mask_img 再 flipud^-1).T
    mask_tf = mask_img[::-1, :].T                        # [T=W,F=H]

    return mask_tf.astype(np.float32)


# ---------------- 主流程 ----------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 MobileNetV3-Small 生成 train/val/test 的 TF 掩码（仅保存 _mask_tf.npy）"
    )
    parser.add_argument("--data_root", type=str, required=True,
                        help="TFI 数据集根目录（包含 train/val/test）")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="训练好的 .pth 模型路径（mobilenetv3_tfi_*）")
    parser.add_argument("--mask_root", type=str, required=True,
                        help="TF 掩码输出根目录（结构与 data_root 对齐）")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--use_pred_label", action="store_true",
                        help="若设置，则 TF evidence 用预测类别；默认用真标签")
    parser.add_argument("--splits", type=str, default="train,val,test",
                        help="Comma-separated splits to process, e.g. test or train,val,test")
    # TF-MASK标准1 参数
    ### alternate dataset parameters
    # parser.add_argument("--time_thresh", type=float, default=0.7)
    # parser.add_argument("--time_valley_thresh", type=float, default=0.15)
    # parser.add_argument("--time_min_len_frac", type=float, default=0.01)
    # parser.add_argument("--time_min_gap_frac", type=float, default=0.02)
    # parser.add_argument("--energy_percentile", type=float, default=40.0)
    # parser.add_argument("--freq_thresh", type=float, default=0.7)
    # parser.add_argument("--freq_min_len_frac", type=float, default=0.01)

    ### TF-MASK标准2 数据集的参数
    parser.add_argument("--time_thresh", type=float, default=0.3)
    parser.add_argument("--time_valley_thresh", type=float, default=0.15)
    parser.add_argument("--time_min_len_frac", type=float, default=0.01)
    parser.add_argument("--time_min_gap_frac", type=float, default=0.02)
    parser.add_argument("--energy_percentile", type=float, default=10.0)
    parser.add_argument("--freq_thresh", type=float, default=0.3)
    parser.add_argument("--freq_min_len_frac", type=float, default=0.01)





    return parser.parse_args()




def main():
    args = parse_args()
    os.makedirs(args.mask_root, exist_ok=True)
    requested_splits = [x.strip() for x in str(args.splits).split(",") if x.strip()]
    if not requested_splits:
        raise RuntimeError("至少需要一个 split")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    # 加载模型
    print(f"\n从 checkpoint 加载模型: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_class_to_idx = load_checkpoint_class_to_idx(ckpt)
    if ckpt_class_to_idx:
        print(f"checkpoint class_to_idx: {json.dumps(ckpt_class_to_idx, ensure_ascii=False)}")

    num_classes = infer_num_classes(args.data_root, requested_splits, ckpt)
    model = build_model(num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    print("OK 模型权重加载完成。")

    target_layer = model.features
    evidence_op = TFEvidenceOperator(model, target_layer)

    for split in requested_splits:
        split_dir = os.path.join(args.data_root, split)
        if not os.path.isdir(split_dir):
            print(f"Warning 跳过 {split}（目录不存在: {split_dir}）")
            continue

        loader, dataset = build_loader_for_split(
            data_root=args.data_root,
            split=split,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            class_to_idx=ckpt_class_to_idx,
        )

        index_loader = DataLoader(
            list(range(len(dataset))),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        pbar = tqdm(zip(index_loader, loader),
                    total=len(loader),
                    desc=f"Gen TF-MASK {split}",
                    ncols=100)

        for batch_indices, (imgs, labels) in pbar:
            batch_indices = batch_indices.tolist()
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                logits = model(imgs)
                preds = logits.argmax(dim=1)

            if args.use_pred_label:
                target_cls = preds
            else:
                target_cls = labels

            cams = evidence_op(imgs, target_cls)  # [B,Hf,Wf]

            for i_in_batch, ds_idx in enumerate(batch_indices):
                img_path, cls_idx = dataset.samples[ds_idx]
                cam_tensor = cams[i_in_batch]

                # CAM resize 到 img_size×img_size（与预处理保持一致）
                cam_np = cam_tensor.cpu().numpy()
                cam_img = Image.fromarray((cam_np * 255).astype(np.uint8))
                cam_img = cam_img.resize((args.img_size, args.img_size),
                                         resample=Image.BILINEAR)
                cam_resized = np.array(cam_img).astype(np.float32) / 255.0  # [H,W]

                # 构建 mask_tf（不保存 cam、不保存 mask_img）
                try:
                    mask_tf = build_mask_from_cam_array(
                        img_path=img_path,
                        cam_resized=cam_resized,
                        time_thresh=args.time_thresh,
                        time_valley_thresh=args.time_valley_thresh,
                        time_min_len_frac=args.time_min_len_frac,
                        time_min_gap_frac=args.time_min_gap_frac,
                        energy_percentile=args.energy_percentile,
                        freq_thresh=args.freq_thresh,
                        freq_min_len_frac=args.freq_min_len_frac,
                    )

                except Exception as e:
                    print(f"Warning 处理 {img_path} 时出错: {e}")
                    continue

                # 保存路径：mask_root/相对路径 + _mask_tf.npy
                rel_path = os.path.relpath(img_path, args.data_root)   # e.g. train/T01/xxx.png
                rel_dir = os.path.dirname(rel_path)                    # train/T01
                stem = os.path.splitext(os.path.basename(rel_path))[0]

                out_dir = os.path.join(args.mask_root, rel_dir)
                os.makedirs(out_dir, exist_ok=True)
                mask_tf_path = os.path.join(out_dir, f"{stem}_mask_tf.npy")

                mask_tf_u8 = (mask_tf > 0.5).astype(np.uint8)
                np.save(mask_tf_path, mask_tf_u8)

    print("\nOK 全部 TF 掩码(_mask_tf.npy) 已生成，保存在 mask_root 对应目录中。")


if __name__ == "__main__":
    main()
