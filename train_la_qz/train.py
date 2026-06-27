import os
import csv
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from dataset import LoveDADataset
from model import SwinUPerNet
from losses import CombinedLoss
from metrics import SegMetric
from utils import seed_everything, save_checkpoint


CLASS_NAMES = [
    "Building",
    "Road",
    "Water",
    "Barren",
    "Forest",
    "Agricultural",
    "Background",
]


def init_csv(csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if not os.path.exists(csv_path):
        header = [
            "epoch",
            "train_loss",
            "val_loss",
            "mIoU",
            "OA",
            "mF1",
            "lr",
            "best_mIoU",
        ]

        header += [f"{name}_IoU" for name in CLASS_NAMES]
        header += [f"{name}_F1" for name in CLASS_NAMES]

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def append_csv(csv_path, epoch, train_loss, val_loss, scores, lr, best_miou):
    row = [
        epoch,
        round(train_loss, 6),
        round(val_loss, 6),
        round(scores["mIoU"], 6),
        round(scores["OA"], 6),
        round(scores["mF1"], 6),
        lr,
        round(best_miou, 6),
    ]

    row += [round(x, 6) for x in scores["class_IoU"]]
    row += [round(x, 6) for x in scores["class_F1"]]

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


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
    metric = SegMetric(num_classes=args.num_classes, ignore_index=255)

    pbar = tqdm(loader, desc=f"Val Epoch {epoch}", ncols=120)

    for imgs, masks in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if args.channels_last:
            imgs = imgs.contiguous(memory_format=torch.channels_last)

        with autocast(device_type="cuda", enabled=args.amp):
            if getattr(args, "tta", 0):
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

    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/datasetss/loveda")
    parser.add_argument("--save_dir", type=str, default="/root/autodl-tmp/checkpoints/train_la_qz")
    parser.add_argument("--log_dir", type=str, default="/root/autodl-tmp/logs/train_la_qz")

    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=7)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=16)

    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--channels_last", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--tta", type=int, default=1)
    parser.add_argument("--pretrained", type=int, default=1)

    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="/root/autodl-tmp/pre_model/model.safetensors",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="swin_tiny_patch4_window7_224.ms_in1k",
    )

    args = parser.parse_args()

    args.amp = bool(args.amp)
    args.channels_last = bool(args.channels_last)
    args.pretrained = bool(args.pretrained)

    seed_everything(args.seed)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    csv_path = os.path.join(args.log_dir, "metrics.csv")
    init_csv(csv_path)

    train_set = LoveDADataset(args.data_root, split="train", img_size=args.img_size, train=True)
    val_set = LoveDADataset(args.data_root, split="val", img_size=args.img_size, train=False)

    print(f"Train samples: {len(train_set)}")
    print(f"Val samples: {len(val_set)}")
    print(f"CSV log path: {csv_path}")

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

    model = SwinUPerNet(
        num_classes=args.num_classes,
        backbone=args.backbone,
        pretrained=args.pretrained,
        pretrained_path=args.pretrained_path,
        img_size=args.img_size,
    ).to(device)

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # LoveDA label mapping:
    # original mask label 1~7 -> class index 0~6
    #
    # Pixel distribution shows:
    # class_index=4 is low-frequency and hard to learn.
    # Therefore we increase class 4 weight moderately.
    #
    # class_index:  0     1     2     3     4     5     6
    class_weights = [0.85, 1.05, 1.20, 1.00, 1.80, 1.15, 0.90]

    criterion = CombinedLoss(
        num_classes=args.num_classes,
        ignore_index=255,
        dice_weight=1.0,
        ce_weight=1.0,
        class_weights=class_weights,
    )

    criterion = criterion.to(device)

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

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            args=args,
        )

        val_loss, scores = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            args=args,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        miou = scores["mIoU"]
        oa = scores["OA"]
        mf1 = scores["mF1"]

        if miou > best_miou:
            best_miou = miou
            save_checkpoint(
                os.path.join(args.save_dir, "loveda_best.pth"),
                model,
                optimizer,
                scheduler,
                epoch,
                best_miou,
            )
            print(f"Saved best checkpoint: mIoU={best_miou:.4f}")

        save_checkpoint(
            os.path.join(args.save_dir, "last.pth"),
            model,
            optimizer,
            scheduler,
            epoch,
            best_miou,
        )

        append_csv(
            csv_path=csv_path,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            scores=scores,
            lr=current_lr,
            best_miou=best_miou,
        )

        print("=" * 100)
        print(f"Epoch [{epoch}/{args.epochs}]")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val mIoU:   {miou:.4f}")
        print(f"Val OA:     {oa:.4f}")
        print(f"Val mF1:    {mf1:.4f}")
        print(f"Best mIoU:  {best_miou:.4f}")
        print(f"LR:         {current_lr:.8f}")
        print(f"Class IoU:  {[round(x, 4) for x in scores['class_IoU']]}")
        print(f"Class F1:   {[round(x, 4) for x in scores['class_F1']]}")
        print("=" * 100)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Metric/mIoU", miou, epoch)
        writer.add_scalar("Metric/OA", oa, epoch)
        writer.add_scalar("Metric/mF1", mf1, epoch)
        writer.add_scalar("Metric/best_mIoU", best_miou, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        for i, name in enumerate(CLASS_NAMES):
            writer.add_scalar(f"Class_IoU/{name}", scores["class_IoU"][i], epoch)
            writer.add_scalar(f"Class_F1/{name}", scores["class_F1"][i], epoch)

    writer.close()

    print(f"Training finished.")
    print(f"Best mIoU: {best_miou:.4f}")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
