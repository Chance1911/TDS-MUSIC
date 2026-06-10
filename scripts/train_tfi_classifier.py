# train_tfi_classifier.py
"""
TFI 时频图无人机类型识别 - 训练脚本（轻量版 backbone）
- 数据结构:
    data_root/
      train/T01/*.png
      train/T02/*.png
      ...
      val/T01/*.png
      val/T02/*.png
"""

import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

class KeepMiddleFreqBand(object):
    """
    在图像的频率轴上（H 方向）仅保留中间 keep_ratio 的频带，其余置 0。
    假设纵轴 H 是频率轴。
    """
    def __init__(self, keep_ratio: float = 0.25):
        assert 0 < keep_ratio <= 1.0
        self.keep_ratio = keep_ratio

    def __call__(self, x: torch.Tensor):
        # x: [C, H, W]
        C, H, W = x.shape
        band_h = max(1, int(H * self.keep_ratio))
        start = (H - band_h) // 2
        end = start + band_h

        # 构造 mask，纵向中间一条为 1，其余为 0
        mask = torch.zeros_like(x)
        mask[:, start:end, :] = 1.0
        return x * mask

class PerImageStandardize(object):
    def __call__(self, x: torch.Tensor):
        # x: [C,H,W]
        mean = x.mean()
        std  = x.std()
        return (x - mean) / (std + 1e-6)


def build_dataloaders(tfi_root, img_size=512, batch_size=32, num_workers=4):
    train_dir = os.path.join(tfi_root, "train")
    val_dir   = os.path.join(tfi_root, "val")

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # [0,1]
        # KeepMiddleFreqBand(keep_ratio=0.25),  #  只保留中间 1/4 频带
        PerImageStandardize(),  #  再做每图标准化
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # KeepMiddleFreqBand(keep_ratio=0.25),
        PerImageStandardize(),
    ])

    train_set = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_set   = datasets.ImageFolder(root=val_dir,   transform=eval_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    print("=== 数据集信息 ===")
    print(f"train: {len(train_set)} 样本, 类别: {train_set.classes}")
    print(f"val:   {len(val_set)} 样本, 类别: {val_set.classes}")
    print(f"class_to_idx 映射: {train_set.class_to_idx}")

    return train_loader, val_loader, train_set.class_to_idx


def build_model(num_classes, pretrained=False):
    """
    换成更轻量的 MobileNetV3-Small 作为 backbone
    """
    if pretrained:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
    else:
        model = models.mobilenet_v3_small(weights=None)

    # 替换最后一层分类头
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    # 打印一下参数量，方便你对比
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params/1e6:.2f} M (MobileNetV3-Small)")

    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Train", ncols=100)
    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

        pbar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, desc="Val"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=desc, ncols=100)
    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, labels)

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


def save_checkpoint(state, ckpt_path):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(state, ckpt_path)
    print(f"OK 已保存模型到: {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="TFI MobileNet UAV Type Classification - Train")

    parser.add_argument("--data_root", type=str,
                        default="data/tfi_imagefolder",
                        help="TFI 数据集根目录（包含 train/val）")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/checkpoints_tfi_mobilenet",
                        help="保存模型的目录")

    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_pretrain", action="store_true",
                        help="如果设置，将不使用ImageNet预训练权重")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    train_loader, val_loader, class_to_idx = build_dataloaders(
        tfi_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    num_classes = len(class_to_idx)
    model = build_model(num_classes, pretrained=not args.no_pretrain).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_val_acc = 0.0
    last_val_acc = None  # 记录最后一个 epoch 的 val_acc
    meta = {
        "class_to_idx": class_to_idx,
        "img_size": args.img_size,
    }

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device, desc="Val")

        scheduler.step()

        last_val_acc = val_acc  # 更新最后一轮的 val_acc

        print(f"Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")

        # 保存“当前为止最好的”模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
                "meta": meta,
            }
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_name = f"mobilenetv3_tfi_best_{timestamp}.pth"
            ckpt_path = os.path.join(args.output_dir, ckpt_name)
            save_checkpoint(ckpt, ckpt_path)

    # 训练结束后，额外保存“最后一个 epoch”的模型
    if last_val_acc is not None:
        final_ckpt = {
            "epoch": args.epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_acc": last_val_acc,
            "meta": meta,
        }
        final_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_name = f"mobilenetv3_tfi_last_{final_ts}.pth"
        final_path = os.path.join(args.output_dir, final_name)
        save_checkpoint(final_ckpt, final_path)

    print(f"\n🎯 训练结束，最佳验证集 acc = {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
