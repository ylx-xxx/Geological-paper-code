import os
import csv
import argparse
import random
from glob import glob

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from model import SwinUPerNetL4S


class LoveDADataset(Dataset):
    def __init__(self, root, split="train", img_size=512, train=True):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.train = train

        if split == "train":
            base = os.path.join(root, "Train", "Train")
        else:
            base = os.path.join(root, "Val", "Val")

        self.samples = []

        for domain in ["Rural", "Urban"]:
            img_dir = os.path.join(base, domain, "images_png")
            mask_dir = os.path.join(base, domain, "masks_png")

            imgs = sorted(glob(os.path.join(img_dir, "*.png")))

            for img_path in imgs:
                name = os.path.basename(img_path)
                mask_path = os.path.join(mask_dir, name)

                if os.path.exists(mask_path):
                    self.samples.append((img_path, mask_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"No LoveDA samples found in {root}, split={split}")

        print(f"[Info] LoveDA {split} samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _map_mask(self, mask):
        """
        LoveDA original:
        0 = ignore
        1~7 = valid classes

        Train target:
        255 = ignore
        0~6 = classes
        """
        out = np.full_like(mask, 255, dtype=np.uint8)

        valid = (mask >= 1) & (mask <= 7)
        out[valid] = mask[valid] - 1

        return out

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        img = np.asarray(img).astype(np.float32) / 255.0
        mask = np.asarray(mask).astype(np.uint8)
        mask = self._map_mask(mask)

        if self.train:
            if random.random() < 0.5:
                img = np.flip(img, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()

            if random.random() < 0.5:
                img = np.flip(img, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()

            if random.random() < 0.5:
                k = random.randint(0, 3)
                img = np.rot90(img, k, axes=(0, 1)).copy()
                mask = np.rot90(mask, k, axes=(0, 1)).copy()

            if random.random() < 0.5:
                scale = random.uniform(0.85, 1.15)
                bias = random.uniform(-0.08, 0.08)
                img = np.clip(img * scale + bias, 0.0, 1.0)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        img = (img - mean) / std

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()

        return img, mask


class DiceLoss(nn.Module):
    def __init__(self, num_classes=7, ignore_index=255, eps=1e-6, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.class_weights = None

    def forward(self, logits, target):
        valid = target != self.ignore_index

        target_safe = target.clone()
        target_safe[~valid] = 0

        probs = torch.softmax(logits, dim=1)

        onehot = F.one_hot(
            target_safe,
            num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()

        valid = valid.unsqueeze(1).float()

        probs = probs * valid
        onehot = onehot * valid

        dims = (0, 2, 3)

        inter = torch.sum(probs * onehot, dims)
        union = torch.sum(probs + onehot, dims)

        dice = (2.0 * inter + self.eps) / (union + self.eps)
        loss = 1.0 - dice

        if self.class_weights is not None:
            w = self.class_weights.to(logits.device)
            w = w / (w.mean() + self.eps)
            loss = loss * w

        return loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, num_classes=7, ignore_index=255):
        super().__init__()

        # LoveDA 类别不均衡，给小类略高权重，不要过强。
        class_weights = [0.85, 1.05, 1.20, 1.00, 1.80, 1.15, 0.90]

        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32)
        )

        self.dice = DiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            class_weights=class_weights,
        )

        self.ignore_index = ignore_index

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights.to(logits.device),
            ignore_index=self.ignore_index,
            label_smoothing=0.05,
        )

        dice = self.dice(logits, target)

        return ce + 0.8 * dice


def update_hist(logits, target, hist, num_classes=7, ignore_index=255):
    pred = torch.argmax(logits, dim=1)

    pred = pred.detach().cpu()
    target = target.detach().cpu()

    mask = target != ignore_index
    pred = pred[mask].long()
    target = target[mask].long()

    inds = num_classes * target + pred
    h = torch.bincount(
        inds,
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes).double()

    hist += h

    return hist


def compute_scores(hist):
    diag = torch.diag(hist)

    iou = diag / (hist.sum(1) + hist.sum(0) - diag + 1e-10)
    miou = torch.mean(iou).item()

    oa = diag.sum() / (hist.sum() + 1e-10)

    return miou, oa.item(), iou.numpy().tolist()


@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()

    hist = torch.zeros((args.num_classes, args.num_classes), dtype=torch.float64)
    total_loss = 0.0
    count = 0

    criterion = CombinedLoss(num_classes=args.num_classes).to(device)

    for imgs, masks in tqdm(loader, desc="Val", ncols=120):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if args.channels_last:
            imgs = imgs.contiguous(memory_format=torch.channels_last)

        with autocast(device_type="cuda", enabled=args.amp):
            if args.tta:
                logits = model(imgs)

                imgs_h = torch.flip(imgs, dims=[3])
                logits_h = torch.flip(model(imgs_h), dims=[3])

                imgs_v = torch.flip(imgs, dims=[2])
                logits_v = torch.flip(model(imgs_v), dims=[2])

                logits = (logits + torch.flip(logits_h, dims=[3]) + torch.flip(logits_v, dims=[2])) / 3.0
            else:
                logits = model(imgs)

            loss = criterion(logits, masks)

        total_loss += loss.item() * imgs.size(0)
        count += imgs.size(0)

        hist = update_hist(logits, masks, hist, num_classes=args.num_classes)

    miou, oa, iou_list = compute_scores(hist)

    return total_loss / max(count, 1), miou, oa, iou_list


def make_loader(dataset, batch_size, shuffle, num_workers):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4

    return DataLoader(**kwargs)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/datasetss/loveda")
    parser.add_argument("--save_dir", type=str, default="/root/autodl-tmp/checkpoints/train_la_qz_v7_b1")
    parser.add_argument("--log_dir", type=str, default="/root/autodl-tmp/logs/train_la_qz_v7_b1")

    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--backbone", type=str, default="segformer_b1")

    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--channels_last", type=int, default=1)
    parser.add_argument("--tta", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    args = parser.parse_args()

    args.amp = bool(args.amp)
    args.channels_last = bool(args.channels_last)
    args.tta = bool(args.tta)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = LoveDADataset(args.data_root, split="train", img_size=args.img_size, train=True)
    val_set = LoveDADataset(args.data_root, split="val", img_size=args.img_size, train=False)

    train_loader = make_loader(train_set, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_set, args.batch_size, False, args.num_workers)

    model = SwinUPerNetL4S(
        num_classes=args.num_classes,
        in_chans=3,
        backbone=args.backbone,
        img_size=args.img_size,
        fpn_dim=256,
    )

    model = model.to(device)

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    criterion = CombinedLoss(num_classes=args.num_classes).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    scaler = GradScaler("cuda", enabled=args.amp)

    csv_path = os.path.join(args.log_dir, "metrics.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "val_mIoU",
            "val_OA",
            "IoU_0",
            "IoU_1",
            "IoU_2",
            "IoU_3",
            "IoU_4",
            "IoU_5",
            "IoU_6",
            "lr",
        ])

    best_miou = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=120)

        for imgs, masks in pbar:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if args.channels_last:
                imgs = imgs.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=args.amp):
                logits = model(imgs)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * imgs.size(0)
            count += imgs.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        train_loss = total_loss / max(count, 1)
        val_loss, val_miou, val_oa, iou_list = validate(model, val_loader, device, args)

        lr_now = optimizer.param_groups[0]["lr"]

        print("=" * 100)
        print(f"Epoch [{epoch}/{args.epochs}]")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val mIoU:   {val_miou:.4f}")
        print(f"Val OA:     {val_oa:.4f}")
        print(f"IoU list:   {[round(x, 4) for x in iou_list]}")
        print(f"Best mIoU:  {max(best_miou, val_miou):.4f}")
        print(f"LR:         {lr_now:.8f}")
        print("=" * 100)

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                round(train_loss, 6),
                round(val_loss, 6),
                round(val_miou, 6),
                round(val_oa, 6),
                *[round(x, 6) for x in iou_list],
                lr_now,
            ])

        if val_miou > best_miou:
            best_miou = val_miou

            save_path = os.path.join(args.save_dir, "loveda_segformer_best.pth")
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_miou": best_miou,
                    "args": vars(args),
                },
                save_path,
            )
            print(f"[Saved] {save_path}")

        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "best_miou": best_miou,
                "args": vars(args),
            },
            os.path.join(args.save_dir, "last.pth"),
        )

    print("Training finished.")
    print(f"Best LoveDA mIoU: {best_miou:.4f}")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
