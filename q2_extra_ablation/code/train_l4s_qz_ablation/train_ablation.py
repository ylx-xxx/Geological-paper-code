import os
import csv
import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_ablation import Landslide4SenseAblationDataset, get_in_chans, read_h5_first
from model import SwinUPerNetL4S
from losses import CombinedLoss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def confusion_from_pred(pred, target):
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    cm = np.zeros((2, 2), dtype=np.float64)
    for t in [0, 1]:
        for p in [0, 1]:
            cm[t, p] = np.logical_and(target == t, pred == p).sum()
    return cm


def scores_from_cm(cm):
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]

    non_iou = tn / max(tn + fp + fn, 1.0)
    land_iou = tp / max(tp + fp + fn, 1.0)
    miou = (non_iou + land_iou) / 2.0
    oa = (tn + tp) / max(cm.sum(), 1.0)

    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1.0)

    return {
        "miou": float(miou),
        "oa": float(oa),
        "non_iou": float(non_iou),
        "landslide_iou": float(land_iou),
        "landslide_f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def compute_auto_class_weights(dataset):
    non_count = 0
    land_count = 0

    print("[Info] Computing class weights from training masks.")
    for _, mask_path in tqdm(dataset.samples, desc="class weights"):
        mask = read_h5_first(mask_path).astype(np.int64)
        if mask.ndim == 3:
            mask = np.squeeze(mask)
        mask = np.where(mask > 0, 1, 0)
        non_count += int((mask == 0).sum())
        land_count += int((mask == 1).sum())

    counts = np.array([non_count, land_count], dtype=np.float64)
    freq = counts / max(counts.sum(), 1.0)

    weights = 1.0 / np.sqrt(freq + 1e-12)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.5, 2.5)

    print(f"[Info] pixel counts: non-landslide={non_count}, landslide={land_count}")
    print(f"[Info] class weights: {weights.tolist()}")

    return weights.tolist()


def adapt_patch_embed_weight_ablation(old_w, new_w, patch_init="mean"):
    if old_w.ndim != 4 or new_w.ndim != 4:
        return None

    if old_w.shape[0] != new_w.shape[0] or old_w.shape[2:] != new_w.shape[2:]:
        return None

    old_in = old_w.shape[1]
    new_in = new_w.shape[1]

    out = new_w.clone()

    copy_ch = min(old_in, new_in)
    out[:, :copy_ch] = old_w[:, :copy_ch]

    if new_in > old_in:
        if patch_init == "mean":
            mean_w = old_w.mean(dim=1, keepdim=True)
            out[:, old_in:new_in] = mean_w.repeat(1, new_in - old_in, 1, 1)
        elif patch_init == "random":
            # Keep the target model's random initialization for extra channels.
            pass
        elif patch_init == "zero":
            out[:, old_in:new_in] = 0.0
        else:
            raise ValueError(f"Unsupported patch_init={patch_init}")

    return out


def load_loveda_pretrained_ablation(model, ckpt_path, patch_init="mean"):
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print(f"[Warning] LoveDA checkpoint not found: {ckpt_path}")
        return

    print(f"[Info] Loading LoveDA checkpoint: {ckpt_path}")
    print(f"[Info] Patch embedding init mode: {patch_init}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    model_state = model.state_dict()
    converted = {}

    loaded = 0
    adapted = 0
    skipped = 0

    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]

        if nk not in model_state:
            skipped += 1
            continue

        if model_state[nk].shape == v.shape:
            converted[nk] = v
            loaded += 1
            continue

        if "patch_embed.proj.weight" in nk:
            new_v = adapt_patch_embed_weight_ablation(
                v,
                model_state[nk],
                patch_init=patch_init,
            )
            if new_v is not None and new_v.shape == model_state[nk].shape:
                converted[nk] = new_v
                adapted += 1
                continue

        skipped += 1

    msg = model.load_state_dict(converted, strict=False)

    print("[Info] LoveDA pretrained transfer finished.")
    print(f"[Info] Direct loaded keys : {loaded}")
    print(f"[Info] Adapted keys       : {adapted}")
    print(f"[Info] Skipped keys       : {skipped}")
    print(f"[Info] Missing keys       : {len(msg.missing_keys)}")
    print(f"[Info] Unexpected keys    : {len(msg.unexpected_keys)}")


def build_criterion(args, class_weights):
    if args.loss_mode == "ce":
        return CombinedLoss(
            num_classes=args.num_classes,
            ce_weight=1.0,
            dice_weight=0.0,
            class_weights=None,
            label_smoothing=args.label_smoothing,
        )

    if args.loss_mode == "ce_dice":
        return CombinedLoss(
            num_classes=args.num_classes,
            ce_weight=1.0,
            dice_weight=1.0,
            class_weights=None,
            label_smoothing=args.label_smoothing,
        )

    if args.loss_mode == "ce_dice_cw":
        return CombinedLoss(
            num_classes=args.num_classes,
            ce_weight=1.0,
            dice_weight=1.0,
            class_weights=class_weights,
            label_smoothing=args.label_smoothing,
        )

    raise ValueError(f"Unsupported loss_mode={args.loss_mode}")


def forward_tta(model, x, use_tta):
    logits = model(x)

    if not use_tta:
        return logits

    logits_sum = logits

    x_h = torch.flip(x, dims=[3])
    logits_h = model(x_h)
    logits_h = torch.flip(logits_h, dims=[3])
    logits_sum = logits_sum + logits_h

    x_v = torch.flip(x, dims=[2])
    logits_v = model(x_v)
    logits_v = torch.flip(logits_v, dims=[2])
    logits_sum = logits_sum + logits_v

    return logits_sum / 3.0


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, args):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
    for imgs, masks in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if args.channels_last:
            imgs = imgs.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            logits = model(imgs)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, args):
    model.eval()
    total_loss = 0.0
    cm_total = np.zeros((2, 2), dtype=np.float64)

    pbar = tqdm(loader, desc=f"Val Epoch {epoch}")
    for imgs, masks in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if args.channels_last:
            imgs = imgs.contiguous(memory_format=torch.channels_last)

        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            logits = forward_tta(model, imgs, args.val_tta)
            loss = criterion(logits, masks)

        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        gts = masks.detach().cpu().numpy()

        for p, t in zip(preds, gts):
            cm_total += confusion_from_pred(p, t)

        total_loss += float(loss.item())

    scores = scores_from_cm(cm_total)
    return total_loss / max(len(loader), 1), scores


def append_metrics(csv_path, epoch, train_loss, val_loss, scores, lr, best_land_iou):
    is_new = not Path(csv_path).exists()

    fields = [
        "epoch",
        "train_loss",
        "val_loss",
        "lr",
        "miou",
        "oa",
        "non_iou",
        "landslide_iou",
        "landslide_f1",
        "precision",
        "recall",
        "tn",
        "fp",
        "fn",
        "tp",
        "best_landslide_iou",
    ]

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            writer.writeheader()

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "lr": lr,
            "best_landslide_iou": best_land_iou,
        }
        row.update({k: round(v, 6) for k, v in scores.items()})
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/datasetss/landslide4Sense")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--loveda_ckpt", type=str, default="/root/autodl-tmp/checkpoints/train_la_qz/loveda_best.pth")

    parser.add_argument("--channel_mode", type=str, default="full14",
                        choices=["rgb", "ms12", "rgb_topo", "full14"])
    parser.add_argument("--patch_init", type=str, default="mean",
                        choices=["mean", "random", "zero"])
    parser.add_argument("--loss_mode", type=str, default="ce_dice_cw",
                        choices=["ce", "ce_dice", "ce_dice_cw"])

    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.02)

    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--channels_last", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_tta", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_loveda_pretrain", type=int, default=1)
    parser.add_argument("--backbone", type=str, default="swin_tiny_patch4_window7_224.ms_in1k")

    args = parser.parse_args()

    args.amp = bool(args.amp)
    args.channels_last = bool(args.channels_last)
    args.val_tta = bool(args.val_tta)
    args.use_loveda_pretrain = bool(args.use_loveda_pretrain)

    args.in_chans = get_in_chans(args.channel_mode)

    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    log_dir = Path(args.log_dir)
    ensure_dir(save_dir)
    ensure_dir(log_dir)

    config = vars(args).copy()
    with open(log_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    train_set = Landslide4SenseAblationDataset(
        args.data_root,
        split="train",
        img_size=args.img_size,
        train=True,
        use_augmentation=True,
        channel_mode=args.channel_mode,
    )

    val_set = Landslide4SenseAblationDataset(
        args.data_root,
        split="val",
        img_size=args.img_size,
        train=False,
        use_augmentation=False,
        channel_mode=args.channel_mode,
    )

    print(f"Train samples: {len(train_set)}")
    print(f"Val samples:   {len(val_set)}")
    print(f"Channel mode:  {args.channel_mode}")
    print(f"In channels:   {args.in_chans}")
    print(f"Patch init:    {args.patch_init}")
    print(f"Loss mode:     {args.loss_mode}")

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

    if args.loss_mode == "ce_dice_cw":
        class_weights = compute_auto_class_weights(train_set)
    else:
        class_weights = None

    print(f"Using class weights: {class_weights}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SwinUPerNetL4S(
        num_classes=args.num_classes,
        in_chans=args.in_chans,
        backbone=args.backbone,
        img_size=args.img_size,
    )

    if args.use_loveda_pretrain:
        load_loveda_pretrained_ablation(
            model,
            args.loveda_ckpt,
            patch_init=args.patch_init,
        )

    model = model.to(device)

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    criterion = build_criterion(args, class_weights)
    criterion = criterion.to(device)

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

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    csv_path = log_dir / "metrics.csv"

    best_landslide_iou = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, args)
        val_loss, scores = validate(model, val_loader, criterion, device, epoch, args)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        if scores["landslide_iou"] > best_landslide_iou:
            best_landslide_iou = scores["landslide_iou"]
            best_epoch = epoch

            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_landslide_iou": best_landslide_iou,
                    "args": vars(args),
                },
                save_dir / "best.pth",
            )

            print(
                f"Saved best checkpoint: "
                f"epoch={epoch}, Landslide_IoU={best_landslide_iou:.4f}"
            )

        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "best_landslide_iou": best_landslide_iou,
                "args": vars(args),
            },
            save_dir / "last.pth",
        )

        append_metrics(csv_path, epoch, train_loss, val_loss, scores, lr, best_landslide_iou)

        print("=" * 100)
        print(f"Epoch [{epoch}/{args.epochs}]")
        print(f"Train Loss:          {train_loss:.4f}")
        print(f"Val Loss:            {val_loss:.4f}")
        print(f"Val mIoU:            {scores['miou']:.4f}")
        print(f"Val OA:              {scores['oa']:.4f}")
        print(f"NonLandslide IoU:    {scores['non_iou']:.4f}")
        print(f"Landslide IoU:       {scores['landslide_iou']:.4f}")
        print(f"Landslide F1/Dice:   {scores['landslide_f1']:.4f}")
        print(f"Landslide Precision: {scores['precision']:.4f}")
        print(f"Landslide Recall:    {scores['recall']:.4f}")
        print(f"Best Landslide IoU:  {best_landslide_iou:.4f} @ epoch {best_epoch}")
        print(f"LR:                  {lr:.8f}")
        print(f"Confusion Matrix:    {np.array([[scores['tn'], scores['fp']], [scores['fn'], scores['tp']]]).tolist()}")
        print("=" * 100)

    with open(log_dir / "best_summary.json", "w") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_landslide_iou": best_landslide_iou,
                "channel_mode": args.channel_mode,
                "patch_init": args.patch_init,
                "loss_mode": args.loss_mode,
            },
            f,
            indent=2,
        )

    print("Training finished.")
    print(f"Best Landslide IoU: {best_landslide_iou:.4f}")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
