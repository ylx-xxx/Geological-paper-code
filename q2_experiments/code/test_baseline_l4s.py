import json
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
    def __init__(self, data_root, split="TestData", mean=None, std=None):
        self.data_root = Path(data_root)
        self.split = split

        self.img_dir = self.data_root / split / "img"

        if split == "TestData":
            self.mask_dir = self.data_root / split / "test"
        else:
            self.mask_dir = self.data_root / split / "mask"

        self.img_paths = sorted(self.img_dir.glob("*.h5"))
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
        raise FileNotFoundError(f"Mask not found for {img_path}")

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self._find_mask(img_path)

        img = read_h5(img_path).astype(np.float32)
        mask = read_h5(mask_path).astype(np.int64)

        if self.mean is not None and self.std is not None:
            img = (img - self.mean.reshape(1, 1, -1)) / (self.std.reshape(1, 1, -1) + 1e-6)

        img = np.transpose(img, (2, 0, 1))

        return torch.from_numpy(img).float(), torch.from_numpy(mask).long(), img_path.name


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
    if name == "unet":
        return UNet14(in_ch=14, num_classes=2, base=32)
    if name == "deeplabv3":
        return build_deeplabv3(in_ch=14, num_classes=2)
    raise ValueError(name)


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
        "Test_mIoU": iou.mean().item(),
        "Test_OA": oa.item(),
        "NonLandslide_IoU": iou[0].item(),
        "Landslide_IoU": iou[1].item(),
        "NonLandslide_F1": f1[0].item(),
        "Landslide_F1": f1[1].item(),
        "Landslide_Precision": precision[1].item(),
        "Landslide_Recall": recall[1].item(),
        "TN": int(cm[0, 0].item()),
        "FP": int(cm[0, 1].item()),
        "FN": int(cm[1, 0].item()),
        "TP": int(cm[1, 1].item()),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["unet", "deeplabv3"])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--method_name", type=str, default="")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    mean = np.array(ckpt.get("mean", [0.0] * 14), dtype=np.float32)
    std = np.array(ckpt.get("std", [1.0] * 14), dtype=np.float32)

    model = build_model(args.model)
    msg = model.load_state_dict(state, strict=False)
    print("Missing keys:", len(msg.missing_keys))
    print("Unexpected keys:", len(msg.unexpected_keys))

    model = model.to(device)
    model.eval()

    test_set = L4SDataset(args.data_root, split="TestData", mean=mean, std=std)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    cm = torch.zeros(2, 2, dtype=torch.long)

    for x, y, _ in test_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=bool(args.amp)):
            logits = model(x)

        pred = torch.argmax(logits, dim=1)
        cm += compute_confusion(pred.cpu(), y.cpu(), 2)

    m = metrics_from_cm(cm)
    m["Method"] = args.method_name if args.method_name else args.model
    m["Checkpoint"] = args.ckpt

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([m]).to_csv(out_csv, index=False)

    print("=" * 100)
    for k, v in m.items():
        print(f"{k}: {v}")
    print(f"Saved test metrics to: {out_csv}")
    print("=" * 100)


if __name__ == "__main__":
    main()
