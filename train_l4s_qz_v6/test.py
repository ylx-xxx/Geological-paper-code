import os
import csv
import argparse
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm

from dataset import Landslide4SenseDataset
from model import SwinUPerNetL4S
from metrics import BinarySegMetric


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    metric = BinarySegMetric(num_classes=args.num_classes, ignore_index=255)

    pbar = tqdm(loader, desc="Test", ncols=120)

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

        metric.update(logits, masks)

    return metric.compute()


def save_result_csv(path, scores):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    header = [
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
    ]

    row = [
        scores["mIoU"],
        scores["OA"],
        scores["NonLandslide_IoU"],
        scores["Landslide_IoU"],
        scores["NonLandslide_F1"],
        scores["Landslide_F1"],
        scores["NonLandslide_Precision"],
        scores["Landslide_Precision"],
        scores["NonLandslide_Recall"],
        scores["Landslide_Recall"],
    ]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow([round(x, 6) for x in row])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/datasetss/landslide4Sense")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="/root/autodl-tmp/logs/train_l4s_qz/test_metrics.csv")

    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--in_chans", type=int, default=14)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--channels_last", type=int, default=1)
    parser.add_argument("--tta", type=int, default=1)
    parser.add_argument("--backbone", type=str, default="swin_tiny_patch4_window7_224.ms_in1k")

    args = parser.parse_args()

    args.amp = bool(args.amp)
    args.channels_last = bool(args.channels_last)
    args.tta = bool(args.tta)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_set = Landslide4SenseDataset(
        args.data_root,
        split="test",
        img_size=args.img_size,
        train=False,
        use_augmentation=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    print(f"Test samples: {len(test_set)}")

    model = SwinUPerNetL4S(
        num_classes=args.num_classes,
        in_chans=args.in_chans,
        backbone=args.backbone,
        img_size=args.img_size,
    )

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    msg = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {args.ckpt}")
    print(f"Missing keys: {len(msg.missing_keys)}")
    print(f"Unexpected keys: {len(msg.unexpected_keys)}")

    model = model.to(device)

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    scores = evaluate(model, test_loader, device, args)

    print("=" * 100)
    print("Final Test Results")
    print(f"Test mIoU:              {scores['mIoU']:.4f}")
    print(f"Test OA:                {scores['OA']:.4f}")
    print(f"NonLandslide IoU:       {scores['NonLandslide_IoU']:.4f}")
    print(f"Landslide IoU:          {scores['Landslide_IoU']:.4f}")
    print(f"NonLandslide F1:        {scores['NonLandslide_F1']:.4f}")
    print(f"Landslide F1/Dice:      {scores['Landslide_F1']:.4f}")
    print(f"Landslide Precision:    {scores['Landslide_Precision']:.4f}")
    print(f"Landslide Recall:       {scores['Landslide_Recall']:.4f}")
    print(f"Confusion Matrix:       {scores['hist']}")
    print("=" * 100)

    save_result_csv(args.out_csv, scores)
    print(f"Saved test metrics to: {args.out_csv}")


if __name__ == "__main__":
    main()
