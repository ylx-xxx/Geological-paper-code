import os
import csv
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from dataset import Landslide4SenseDataset
from model import SwinUPerNetL4S, load_loveda_pretrained
from losses import CombinedLoss
from metrics import BinarySegMetric
from utils import seed_everything, save_checkpoint


def init_csv(csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if not os.path.exists(csv_path):
        header = [
            "epoch",
            "train_loss",
            "val_loss",
            "mIoU",
            "OA",
            "NonLandslide_IoU",
            "Landslide_IoU",
            "NonLandslide_F1",
            "Landslide_F1",
            "NonLandslide_Precision",
            "Landslide_Precision",
            "NonLandslide_Recall",
            "Landslide_Recall",
            "lr",
            "best_mIoU",
            "best_Landslide_IoU",
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def append_csv(csv_path, epoch, train_loss, val_loss, scores, lr, best_miou, best_landslide_iou):
    row = [
        epoch,
        round(train_loss, 6),
        round(val_loss, 6),
        round(scores["mIoU"], 6),
        round(scores["OA"], 6),
        round(scores["NonLandslide_IoU"], 6),
        round(scores["Landslide_IoU"], 6),
        round(scores["NonLandslide_F1"], 6),
        round(scores["Landslide_F1"], 6),
        round(scores["NonLandslide_Precision"], 6),
        round(scores["Landslide_Precision"], 6),
        round(scores["NonLandslide_Recall"], 6),
        round(scores["Landslide_Recall"], 6),
        lr,
        round(best_miou, 6),
        round(best_landslide_iou, 6),
    ]

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def compute_auto_class_weights(dataset):
    counts = torch.zeros(2, dtype=torch.float64)

    print("[Info] Computing class weights from training masks...")
    for _, mask in tqdm(dataset, desc="class weights"):
        counts[0] += torch.sum(mask == 0).item()
        counts[1] += torch.sum(mask == 1).item()

    freq = counts / counts.sum()

    # sqrt inverse frequency, more stable than raw inverse.
    weights = torch.sqrt(1.0 / (freq + 1e-8))
    weights = weights / weights.mean()
    weights = torch.clamp(weights, min=0.5, max=5.0)

    print(f"[Info] pixel counts: non-landslide={int(counts[0])}, landslide={int(counts[1])}")
    print(f"[Info] class weights: {weights.tolist()}")

    return weights.tolist()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, args):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", ncols=120)

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

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, args):
    model.eval()
    total_loss = 0.0
    metric = BinarySegMetric(num_classes=args.num_classes, ignore_index=255)

    pbar = tqdm(loader, desc=f"Val Epoch {epoch}", ncols=120)

    for imgs, masks in pbar:
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

                logits = (logits + logits_h + logits_v) / 3.0
            else:
                logits = model(imgs)

            loss = criterion(logits, masks)

        total_loss += loss.item()
        metric.update(logits, masks)

    scores = metric.compute()
    return total_loss / len(loader), scores


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/datasetss/landslide4Sense")
    parser.add_argument("--save_dir", type=str, default="/root/autodl-tmp/checkpoints/train_l4s_qz")
    parser.add_argument("--log_dir", type=str, default="/root/autodl-tmp/logs/train_l4s_qz")

    parser.add_argument("--loveda_ckpt", type=str, default="/root/autodl-tmp/checkpoints/train_la_qz/loveda_best.pth")
    parser.add_argument("--resume_ckpt", type=str, default="")

    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--in_chans", type=int, default=14)
    parser.add_argument("--num_classes", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--channels_last", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--tta", type=int, default=1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_loveda_pretrain", type=int, default=1)
    parser.add_argument("--auto_class_weights", type=int, default=1)

    parser.add_argument("--backbone", type=str, default="swin_tiny_patch4_window7_224.ms_in1k")

    args = parser.parse_args()

    args.amp = bool(args.amp)
    args.channels_last = bool(args.channels_last)
    args.tta = bool(args.tta)
    args.use_loveda_pretrain = bool(args.use_loveda_pretrain)
    args.auto_class_weights = bool(args.auto_class_weights)

    seed_everything(args.seed)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    csv_path = os.path.join(args.log_dir, "metrics.csv")
    init_csv(csv_path)

    train_set = Landslide4SenseDataset(
        args.data_root,
        split="train",
        img_size=args.img_size,
        train=True,
        use_augmentation=True,
    )

    val_set = Landslide4SenseDataset(
        args.data_root,
        split="val",
        img_size=args.img_size,
        train=False,
        use_augmentation=False,
    )

    print(f"Train samples: {len(train_set)}")
    print(f"Val samples: {len(val_set)}")
    print(f"CSV log path: {csv_path}")

    if args.auto_class_weights:
        class_weights = compute_auto_class_weights(train_set)
    else:
        class_weights = [0.7, 1.3]

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    model = SwinUPerNetL4S(
        num_classes=args.num_classes,
        in_chans=args.in_chans,
        backbone=args.backbone,
        img_size=args.img_size,
    )

    if args.use_loveda_pretrain:
        load_loveda_pretrained(model, args.loveda_ckpt)

    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        print(f"[Info] Loading full L4S checkpoint for terrain fine-tuning: {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        msg = model.load_state_dict(state, strict=False)
        print(f"[Info] Resume missing keys: {len(msg.missing_keys)}")
        print(f"[Info] Resume unexpected keys: {len(msg.unexpected_keys)}")

    model = model.to(device)

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    criterion = CombinedLoss(
        num_classes=args.num_classes,
        ignore_index=255,
        ce_weight=1.0,
        dice_weight=1.0,
        class_weights=class_weights,
        label_smoothing=0.02,
    ).to(device)

    print(f"Using class weights: {class_weights}")

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    scaler = GradScaler("cuda", enabled=args.amp)
    writer = SummaryWriter(args.log_dir)

    best_miou = 0.0
    best_landslide_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, args)
        val_loss, scores = validate(model, val_loader, criterion, device, epoch, args)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        miou = scores["mIoU"]
        landslide_iou = scores["Landslide_IoU"]

        improved = miou > best_miou
        if improved:
            best_miou = miou
            best_landslide_iou = max(best_landslide_iou, landslide_iou)

            save_checkpoint(
                os.path.join(args.save_dir, "l4s_best.pth"),
                model,
                optimizer,
                scheduler,
                epoch,
                best_miou,
                best_landslide_iou,
            )

            print(f"Saved best checkpoint: mIoU={best_miou:.4f}, Landslide_IoU={landslide_iou:.4f}")

        if landslide_iou > best_landslide_iou:
            best_landslide_iou = landslide_iou
            save_checkpoint(
                os.path.join(args.save_dir, "l4s_best_landslide_iou.pth"),
                model,
                optimizer,
                scheduler,
                epoch,
                best_miou,
                best_landslide_iou,
            )

        save_checkpoint(
            os.path.join(args.save_dir, "last.pth"),
            model,
            optimizer,
            scheduler,
            epoch,
            best_miou,
            best_landslide_iou,
        )

        append_csv(csv_path, epoch, train_loss, val_loss, scores, current_lr, best_miou, best_landslide_iou)

        print("=" * 100)
        print(f"Epoch [{epoch}/{args.epochs}]")
        print(f"Train Loss:          {train_loss:.4f}")
        print(f"Val Loss:            {val_loss:.4f}")
        print(f"Val mIoU:            {scores['mIoU']:.4f}")
        print(f"Val OA:              {scores['OA']:.4f}")
        print(f"NonLandslide IoU:    {scores['NonLandslide_IoU']:.4f}")
        print(f"Landslide IoU:       {scores['Landslide_IoU']:.4f}")
        print(f"Landslide F1/Dice:   {scores['Landslide_F1']:.4f}")
        print(f"Landslide Precision: {scores['Landslide_Precision']:.4f}")
        print(f"Landslide Recall:    {scores['Landslide_Recall']:.4f}")
        print(f"Best mIoU:           {best_miou:.4f}")
        print(f"Best Landslide IoU:  {best_landslide_iou:.4f}")
        print(f"LR:                  {current_lr:.8f}")
        print(f"Confusion Matrix:    {scores['hist']}")
        print("=" * 100)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Metric/mIoU", scores["mIoU"], epoch)
        writer.add_scalar("Metric/OA", scores["OA"], epoch)
        writer.add_scalar("Metric/Landslide_IoU", scores["Landslide_IoU"], epoch)
        writer.add_scalar("Metric/Landslide_F1", scores["Landslide_F1"], epoch)
        writer.add_scalar("Metric/Landslide_Precision", scores["Landslide_Precision"], epoch)
        writer.add_scalar("Metric/Landslide_Recall", scores["Landslide_Recall"], epoch)
        writer.add_scalar("LR", current_lr, epoch)

    writer.close()

    print("Training finished.")
    print(f"Best mIoU: {best_miou:.4f}")
    print(f"Best Landslide IoU: {best_landslide_iou:.4f}")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
