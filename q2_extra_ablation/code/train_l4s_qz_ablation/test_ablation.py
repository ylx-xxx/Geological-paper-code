import csv
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_ablation import Landslide4SenseAblationDataset, get_in_chans
from model import SwinUPerNetL4S


def ensure_dir(p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)


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
        "test_miou": float(miou),
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


def load_ckpt(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    msg = model.load_state_dict(state, strict=False)
    print(f"[Info] Loaded checkpoint: {ckpt_path}")
    print(f"[Info] Missing keys: {len(msg.missing_keys)}")
    print(f"[Info] Unexpected keys: {len(msg.unexpected_keys)}")


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    cm_total = np.zeros((2, 2), dtype=np.float64)

    for imgs, masks in tqdm(loader, desc="Testing"):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if args.channels_last:
            imgs = imgs.contiguous(memory_format=torch.channels_last)

        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            logits = forward_tta(model, imgs, args.tta)

        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        gts = masks.detach().cpu().numpy()

        for p, t in zip(preds, gts):
            cm_total += confusion_from_pred(p, t)

    return scores_from_cm(cm_total)


def write_csv(path, row):
    ensure_dir(path)
    fields = list(row.keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/datasetss/landslide4Sense")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)

    parser.add_argument("--channel_mode", type=str, default="full14",
                        choices=["rgb", "ms12", "rgb_topo", "full14"])
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--backbone", type=str, default="swin_tiny_patch4_window7_224.ms_in1k")
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--channels_last", type=int, default=1)
    parser.add_argument("--tta", type=int, default=1)

    args = parser.parse_args()

    args.amp = bool(args.amp)
    args.channels_last = bool(args.channels_last)
    args.tta = bool(args.tta)
    args.in_chans = get_in_chans(args.channel_mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_set = Landslide4SenseAblationDataset(
        args.data_root,
        split="test",
        img_size=args.img_size,
        train=False,
        use_augmentation=False,
        channel_mode=args.channel_mode,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Test samples:  {len(test_set)}")
    print(f"Channel mode:  {args.channel_mode}")
    print(f"In channels:   {args.in_chans}")
    print(f"TTA:           {args.tta}")

    model = SwinUPerNetL4S(
        num_classes=args.num_classes,
        in_chans=args.in_chans,
        backbone=args.backbone,
        img_size=args.img_size,
    )

    load_ckpt(model, args.ckpt)

    model = model.to(device)

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    scores = evaluate(model, test_loader, device, args)

    row = {
        "ckpt": args.ckpt,
        "channel_mode": args.channel_mode,
        "in_chans": args.in_chans,
        "tta": int(args.tta),
    }
    row.update({k: round(v, 6) for k, v in scores.items()})

    write_csv(args.out_csv, row)

    print("=" * 100)
    print("Test finished.")
    for k, v in row.items():
        print(f"{k}: {v}")
    print(f"CSV saved to: {args.out_csv}")
    print("=" * 100)


if __name__ == "__main__":
    main()
