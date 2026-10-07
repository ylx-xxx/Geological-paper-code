import os
import csv
import argparse
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm

from dataset import Landslide4SenseDataset
from model import SwinUPerNetL4S


def update_hist_from_pred(pred, target, hist):
    pred = pred.detach().cpu()
    target = target.detach().cpu()

    mask = target != 255
    pred = pred[mask]
    target = target[mask]

    inds = 2 * target.long() + pred.long()
    h = torch.bincount(inds, minlength=4).reshape(2, 2).double()
    hist += h
    return hist


def compute_scores(hist):
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


@torch.no_grad()
def collect_probs(model, loader, device, args):
    model.eval()

    all_probs = []
    all_masks = []

    pbar = tqdm(loader, desc="Collect probs", ncols=120)

    for imgs, masks in pbar:
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

            prob = torch.softmax(logits, dim=1)[:, 1]

        all_probs.append(prob.detach().cpu())
        all_masks.append(masks.detach().cpu())

    return torch.cat(all_probs, dim=0), torch.cat(all_masks, dim=0)


def evaluate_threshold(probs, masks, threshold):
    pred = (probs >= threshold).long()
    hist = torch.zeros((2, 2), dtype=torch.float64)
    hist = update_hist_from_pred(pred, masks, hist)
    return compute_scores(hist)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/datasetss/landslide4Sense")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--out_csv", type=str, default="/root/autodl-tmp/logs/train_l4s_qz/threshold_search.csv")

    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--in_chans", type=int, default=14)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--channels_last", type=int, default=1)
    parser.add_argument("--tta", type=int, default=1)

    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--backbone", type=str, default="swin_tiny_patch4_window7_224.ms_in1k")

    args = parser.parse_args()

    args.amp = bool(args.amp)
    args.channels_last = bool(args.channels_last)
    args.tta = bool(args.tta)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

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

    probs, masks = collect_probs(model, loader, device, args)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    if args.threshold >= 0:
        thresholds = [args.threshold]
    else:
        thresholds = [round(x / 100, 2) for x in range(20, 81, 2)]

    rows = []
    best = None

    for th in thresholds:
        scores = evaluate_threshold(probs, masks, th)
        row = {
            "threshold": th,
            **scores,
        }
        rows.append(row)

        if best is None or scores["Landslide_F1"] > best["Landslide_F1"]:
            best = row

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "threshold",
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
    print("Best threshold by Landslide F1")
    print(f"Threshold:           {best['threshold']}")
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
