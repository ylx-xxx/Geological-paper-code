import os
import csv
import argparse
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm

from dataset import Landslide4SenseDataset
from model import SwinUPerNetL4S


def compute_scores_from_hist(hist):
    hist = torch.tensor(hist, dtype=torch.float64)
    diag = torch.diag(hist)

    total = hist.sum()
    oa = diag.sum() / (total + 1e-10)

    iou = diag / (hist.sum(1) + hist.sum(0) - diag + 1e-10)
    precision = diag / (hist.sum(0) + 1e-10)
    recall = diag / (hist.sum(1) + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return {
        "mIoU": torch.mean(iou).item(),
        "OA": oa.item(),
        "NonLandslide_IoU": iou[0].item(),
        "Landslide_IoU": iou[1].item(),
        "NonLandslide_F1": f1[0].item(),
        "Landslide_F1": f1[1].item(),
        "NonLandslide_Precision": precision[0].item(),
        "Landslide_Precision": precision[1].item(),
        "NonLandslide_Recall": recall[0].item(),
        "Landslide_Recall": recall[1].item(),
        "hist": hist.numpy().tolist(),
    }


def update_hist(pred, target, hist):
    pred = torch.from_numpy(pred).long()
    target = torch.from_numpy(target).long()

    mask = target != 255
    pred = pred[mask]
    target = target[mask]

    inds = 2 * target + pred
    h = torch.bincount(inds, minlength=4).reshape(2, 2).numpy()
    hist += h
    return hist


def postprocess_mask(prob, threshold=0.5, min_area=0, morph_ksize=0):
    pred = (prob >= threshold).astype(np.uint8)

    if morph_ksize > 0:
        kernel = np.ones((morph_ksize, morph_ksize), np.uint8)
        pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)

    if min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)
        cleaned = np.zeros_like(pred, dtype=np.uint8)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned[labels == i] = 1

        pred = cleaned

    return pred


@torch.no_grad()
def collect_probs(model, loader, device, args):
    model.eval()

    probs_all = []
    masks_all = []

    for imgs, masks in tqdm(loader, desc="Collect probs", ncols=120):
        imgs = imgs.to(device, non_blocking=True)

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

            probs = torch.softmax(logits, dim=1)[:, 1]

        probs_all.append(probs.cpu())
        masks_all.append(masks.cpu())

    probs_all = torch.cat(probs_all, dim=0).numpy()
    masks_all = torch.cat(masks_all, dim=0).numpy()

    return probs_all, masks_all


def evaluate_setting(probs_all, masks_all, threshold, min_area, morph_ksize):
    hist = np.zeros((2, 2), dtype=np.float64)

    for prob, mask in zip(probs_all, masks_all):
        pred = postprocess_mask(
            prob,
            threshold=threshold,
            min_area=min_area,
            morph_ksize=morph_ksize,
        )
        hist = update_hist(pred, mask, hist)

    return compute_scores_from_hist(hist)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/datasetss/landslide4Sense")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--out_csv", type=str, required=True)

    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--in_chans", type=int, default=14)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--channels_last", type=int, default=1)
    parser.add_argument("--tta", type=int, default=1)

    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--min_area", type=int, default=-1)
    parser.add_argument("--morph_ksize", type=int, default=-1)

    parser.add_argument("--backbone", type=str, default="swin_tiny_patch4_window7_224.ms_in1k")

    args = parser.parse_args()

    args.amp = bool(args.amp)
    args.channels_last = bool(args.channels_last)
    args.tta = bool(args.tta)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Landslide4SenseDataset(
        args.data_root,
        split=args.split,
        img_size=args.img_size,
        train=False,
        use_augmentation=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    print(f"Split: {args.split}")
    print(f"Samples: {len(dataset)}")

    model = SwinUPerNetL4S(
        num_classes=args.num_classes,
        in_chans=args.in_chans,
        backbone=args.backbone,
        img_size=args.img_size,
    )

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    msg = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {args.ckpt}")
    print(f"Missing keys: {len(msg.missing_keys)}")
    print(f"Unexpected keys: {len(msg.unexpected_keys)}")

    model = model.to(device)

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    probs_all, masks_all = collect_probs(model, loader, device, args)

    if args.threshold >= 0:
        thresholds = [args.threshold]
    else:
        thresholds = [round(x / 100, 2) for x in range(38, 67, 2)]

    if args.min_area >= 0:
        min_areas = [args.min_area]
    else:
        min_areas = [0, 4, 8, 16, 24, 32, 48, 64]

    if args.morph_ksize >= 0:
        morph_ksizes = [args.morph_ksize]
    else:
        morph_ksizes = [0, 3]

    rows = []
    best = None

    for th in thresholds:
        for ma in min_areas:
            for mk in morph_ksizes:
                scores = evaluate_setting(
                    probs_all,
                    masks_all,
                    threshold=th,
                    min_area=ma,
                    morph_ksize=mk,
                )

                row = {
                    "threshold": th,
                    "min_area": ma,
                    "morph_ksize": mk,
                    **scores,
                }

                rows.append(row)

                if best is None or scores["Landslide_IoU"] > best["Landslide_IoU"]:
                    best = row

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "threshold",
            "min_area",
            "morph_ksize",
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
        ])

        for r in rows:
            writer.writerow([
                r["threshold"],
                r["min_area"],
                r["morph_ksize"],
                round(r["mIoU"], 6),
                round(r["OA"], 6),
                round(r["NonLandslide_IoU"], 6),
                round(r["Landslide_IoU"], 6),
                round(r["NonLandslide_F1"], 6),
                round(r["Landslide_F1"], 6),
                round(r["NonLandslide_Precision"], 6),
                round(r["Landslide_Precision"], 6),
                round(r["NonLandslide_Recall"], 6),
                round(r["Landslide_Recall"], 6),
            ])

    print("=" * 100)
    print("Best setting by Landslide IoU")
    print(f"threshold:           {best['threshold']}")
    print(f"min_area:            {best['min_area']}")
    print(f"morph_ksize:         {best['morph_ksize']}")
    print(f"mIoU:                {best['mIoU']:.4f}")
    print(f"OA:                  {best['OA']:.4f}")
    print(f"Landslide IoU:       {best['Landslide_IoU']:.4f}")
    print(f"Landslide F1/Dice:   {best['Landslide_F1']:.4f}")
    print(f"Landslide Precision: {best['Landslide_Precision']:.4f}")
    print(f"Landslide Recall:    {best['Landslide_Recall']:.4f}")
    print(f"Confusion Matrix:    {best['hist']}")
    print("=" * 100)
    print(f"Saved CSV to: {args.out_csv}")


if __name__ == "__main__":
    main()
