import os
import json
import math
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def read_h5(path):
    with h5py.File(path, "r") as f:
        if "img" in f:
            return f["img"][:]
        if "mask" in f:
            return f["mask"][:]
        key = list(f.keys())[0]
        return f[key][:]


class L4SDataset(Dataset):
    def __init__(self, data_root, split="TrainData", mean=None, std=None):
        self.data_root = Path(data_root)
        self.split = split

        self.img_dir = self.data_root / split / "img"

        if split == "TestData":
            self.mask_dir = self.data_root / split / "test"
        else:
            self.mask_dir = self.data_root / split / "mask"

        self.img_paths = sorted(self.img_dir.glob("*.h5"))

        if len(self.img_paths) == 0:
            raise RuntimeError(f"No h5 images found in {self.img_dir}")

        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.img_paths)

    def _find_mask(self, img_path):
        stem = img_path.stem
        idx = stem.split("_")[-1]

        candidates = [
            self.mask_dir / f"mask_{idx}.h5",
            self.mask_dir / f"{stem}.h5",
            self.mask_dir / img_path.name,
        ]

        for p in candidates:
            if p.exists():
                return p

        raise FileNotFoundError(f"Mask not found for {img_path}, tried: {candidates}")

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self._find_mask(img_path)

        img = read_h5(img_path).astype(np.float32)
        mask = read_h5(mask_path).astype(np.int64)

        if img.ndim != 3:
            raise RuntimeError(f"Image shape should be H,W,C, got {img.shape} from {img_path}")

        if img.shape[-1] != 14:
            raise RuntimeError(f"L4S image should have 14 channels, got {img.shape[-1]} from {img_path}")

        if self.mean is not None and self.std is not None:
            img = (img - self.mean.reshape(1, 1, -1)) / (self.std.reshape(1, 1, -1) + 1e-6)

        img = np.transpose(img, (2, 0, 1))

        return torch.from_numpy(img).float(), torch.from_numpy(mask).long()


def compute_stats(data_root, out_json):
    out_json = Path(out_json)

    if out_json.exists():
        with open(out_json, "r") as f:
            obj = json.load(f)
        return np.array(obj["mean"], dtype=np.float32), np.array(obj["std"], dtype=np.float32)

    img_dir = Path(data_root) / "TrainData" / "img"
    img_paths = sorted(img_dir.glob("*.h5"))

    s = np.zeros(14, dtype=np.float64)
    ss = np.zeros(14, dtype=np.float64)
    n = 0

    for p in img_paths:
        img = read_h5(p).astype(np.float64)
        flat = img.reshape(-1, img.shape[-1])
        s += flat.sum(axis=0)
        ss += (flat ** 2).sum(axis=0)
        n += flat.shape[0]

    mean = s / n
    var = ss / n - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)

    return mean.astype(np.float32), std.astype(np.float32)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet14(nn.Module):
    def __init__(self, in_ch=14, num_classes=2, base=32):
        super().__init__()

        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.head(d1)


def build_deeplabv3(in_ch=14, num_classes=2):
    try:
        from torchvision.models.segmentation import deeplabv3_resnet50
        model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=num_classes, aux_loss=False)
    except TypeError:
        from torchvision.models.segmentation import deeplabv3_resnet50
        model = deeplabv3_resnet50(pretrained=False, progress=True, num_classes=num_classes, aux_loss=False)

    old_conv = model.backbone.conv1
    new_conv = nn.Conv2d(
        in_ch,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
    model.backbone.conv1 = new_conv

    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            out = self.m(x)["out"]
            if out.shape[-2:] != x.shape[-2:]:
                out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return out

    return Wrapper(model)


def build_model(name):
    name = name.lower()
    if name == "unet":
        return UNet14(in_ch=14, num_classes=2, base=32)
    if name == "deeplabv3":
        return build_deeplabv3(in_ch=14, num_classes=2)
    raise ValueError(f"Unknown model: {name}")


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        inter = torch.sum(probs * target_onehot, dims)
        union = torch.sum(probs + target_onehot, dims)

        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


def compute_confusion(pred, gt, num_classes=2):
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)
    mask = (gt >= 0) & (gt < num_classes)
    hist = torch.bincount(
        num_classes * gt[mask].long() + pred[mask].long(),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return hist


def metrics_from_cm(cm):
    cm = cm.float()

    tp = torch.diag(cm)
    row_sum = cm.sum(dim=1)
    col_sum = cm.sum(dim=0)
    total = cm.sum()

    iou = tp / torch.clamp(row_sum + col_sum - tp, min=1.0)
    f1 = 2 * tp / torch.clamp(row_sum + col_sum, min=1.0)
    recall = tp / torch.clamp(row_sum, min=1.0)
    precision = tp / torch.clamp(col_sum, min=1.0)

    oa = tp.sum() / torch.clamp(total, min=1.0)

    return {
        "mIoU": iou.mean().item(),
        "OA": oa.item(),
        "NonLandslide_IoU": iou[0].item(),
        "Landslide_IoU": iou[1].item(),
        "NonLandslide_F1": f1[0].item(),
        "Landslide_F1": f1[1].item(),
        "Landslide_Precision": precision[1].item(),
        "Landslide_Recall": recall[1].item(),
    }


def train_one_epoch(model, loader, optimizer, scaler, ce_loss, dice_loss, device, amp=True, grad_clip=1.0):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = ce_loss(logits, y) + dice_loss(logits, y)

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, ce_loss, dice_loss, device, amp=True):
    model.eval()
    total_loss = 0.0
    cm = torch.zeros(2, 2, dtype=torch.long)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = ce_loss(logits, y) + dice_loss(logits, y)

        pred = torch.argmax(logits, dim=1)

        total_loss += loss.item() * x.size(0)
        cm += compute_confusion(pred.cpu(), y.cpu(), num_classes=2)

    m = metrics_from_cm(cm)
    m["loss"] = total_loss / len(loader.dataset)
    m["confusion_matrix"] = cm.tolist()
    return m


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True, choices=["unet", "deeplabv3"])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--ce_w0", type=float, default=0.5)
    parser.add_argument("--ce_w1", type=float, default=1.73)

    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    log_dir = Path(args.log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    stats_json = log_dir / "stats_14ch.json"
    mean, std = compute_stats(args.data_root, stats_json)

    train_set = L4SDataset(args.data_root, split="TrainData", mean=mean, std=std)
    val_set = L4SDataset(args.data_root, split="ValidData", mean=mean, std=std)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = build_model(args.model).to(device)

    ce_weight = torch.tensor([args.ce_w0, args.ce_w1], dtype=torch.float32, device=device)
    ce_loss = nn.CrossEntropyLoss(weight=ce_weight)
    dice_loss = DiceLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        if args.epochs <= 1:
            return 1.0
        cos = 0.5 * (1 + math.cos(math.pi * epoch / args.epochs))
        min_ratio = args.min_lr / args.lr
        return min_ratio + (1 - min_ratio) * cos

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp))

    rows = []
    best_iou = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, ce_loss, dice_loss, device,
            amp=bool(args.amp),
            grad_clip=args.grad_clip,
        )

        val_m = evaluate(model, val_loader, ce_loss, dice_loss, device, amp=bool(args.amp))
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_m["loss"],
            "mIoU": val_m["mIoU"],
            "OA": val_m["OA"],
            "NonLandslide_IoU": val_m["NonLandslide_IoU"],
            "Landslide_IoU": val_m["Landslide_IoU"],
            "NonLandslide_F1": val_m["NonLandslide_F1"],
            "Landslide_F1": val_m["Landslide_F1"],
            "Landslide_Precision": val_m["Landslide_Precision"],
            "Landslide_Recall": val_m["Landslide_Recall"],
        }
        rows.append(row)

        pd.DataFrame(rows).to_csv(log_dir / "metrics.csv", index=False)

        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "mean": mean.tolist(),
            "std": std.tolist(),
        }

        torch.save(ckpt, save_dir / "last.pth")

        if val_m["Landslide_IoU"] > best_iou:
            best_iou = val_m["Landslide_IoU"]
            torch.save(ckpt, save_dir / "best.pth")

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_m['loss']:.4f} | "
            f"mIoU={val_m['mIoU']:.4f} | "
            f"Landslide_IoU={val_m['Landslide_IoU']:.4f} | "
            f"best={best_iou:.4f}"
        )

    print("=" * 100)
    print(f"Training finished. Best Val Landslide IoU: {best_iou:.6f}")
    print(f"CSV saved to: {log_dir / 'metrics.csv'}")
    print("=" * 100)


if __name__ == "__main__":
    main()
