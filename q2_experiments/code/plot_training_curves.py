import argparse
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pick_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


def plot_loss(df, epoch_col, out_path, title):
    train_loss = pick_col(df, ["train_loss", "Train_Loss", "loss_train"])
    val_loss = pick_col(df, ["val_loss", "Val_Loss", "loss", "Loss"])

    plt.figure(figsize=(7, 5))

    plotted = False

    if train_loss is not None:
        plt.plot(df[epoch_col], df[train_loss], label="Train Loss")
        plotted = True

    if val_loss is not None:
        plt.plot(df[epoch_col], df[val_loss], label="Val Loss")
        plotted = True

    if not plotted:
        print(f"[Warning] No loss columns found for {title}")
        plt.close()
        return

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_metrics(df, epoch_col, out_path, title):
    metric_cols = [
        "mIoU",
        "OA",
        "NonLandslide_IoU",
        "Landslide_IoU",
        "Landslide_F1",
        "Landslide_F1/Dice",
        "Landslide_Precision",
        "Landslide_Recall"
    ]

    plt.figure(figsize=(8, 5))

    plotted = False

    for c in metric_cols:
        if c in df.columns:
            plt.plot(df[epoch_col], df[c], label=c)
            plotted = True

    if not plotted:
        print(f"[Warning] No metric columns found for {title}")
        plt.close()
        return

    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    epoch_col = pick_col(df, ["epoch", "Epoch"])

    if epoch_col is None:
        df["epoch"] = range(1, len(df) + 1)
        epoch_col = "epoch"

    plot_loss(
        df,
        epoch_col,
        out_dir / f"{args.name}_loss_curve.png",
        f"{args.name} Loss Curve"
    )

    plot_metrics(
        df,
        epoch_col,
        out_dir / f"{args.name}_metric_curve.png",
        f"{args.name} Metric Curve"
    )

    print("=" * 100)
    print(f"Saved curves for {args.name} to: {out_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
