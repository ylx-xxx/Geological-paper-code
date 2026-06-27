import os
import sys
import json
import time
import math
import traceback
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/root/autodl-tmp/q2_experiments")
DATA_ROOT = Path("/root/autodl-tmp/datasetss/landslide4Sense")
DATA_ROOT_TRAINVAL = Path("/root/autodl-tmp/datasetss/landslide4Sense_trainval")
FINAL_CKPT = Path("/root/autodl-tmp/final_tune_lab/best_pool/current_best.pth")

FIG_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "tables"
CODE_DIR = ROOT / "code"

for d in [
    FIG_DIR,
    TABLE_DIR,
    FIG_DIR / "framework",
    FIG_DIR / "comparison",
    FIG_DIR / "curves",
    FIG_DIR / "confusion_matrix",
    FIG_DIR / "dataset",
    FIG_DIR / "visualizations",
    FIG_DIR / "visualizations" / "final_v8",
    FIG_DIR / "area_group",
    FIG_DIR / "sample_distribution",
    FIG_DIR / "complexity",
]:
    d.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Basic IO
# -----------------------------

def read_h5(path):
    with h5py.File(path, "r") as f:
        if "img" in f:
            return f["img"][:]
        if "mask" in f:
            return f["mask"][:]
        if "test" in f:
            return f["test"][:]
        key = list(f.keys())[0]
        return f[key][:]


def safe_read_csv(path):
    path = Path(path)
    if not path.exists():
        print(f"[Skip] Missing CSV: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[Skip] Failed to read CSV: {path} | {e}")
        return None


def save_df_markdown(df, path):
    path = Path(path)
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            vals.append("" if pd.isna(v) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def normalize_metric_name(name):
    return str(name).strip().replace("/", "_").replace(" ", "_")


# -----------------------------
# 1. Framework diagram
# -----------------------------

def draw_box(ax, xy, w, h, text, fontsize=9):
    x, y = xy
    rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)
    return rect


def arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=1.5)
    )


def plot_framework():
    out_dir = FIG_DIR / "framework"
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(1.8, 5.6, "Stage I: Source-domain pretraining", ha="center", fontsize=12, fontweight="bold")
    ax.text(7.0, 5.6, "Stage II: Target-domain transfer training", ha="center", fontsize=12, fontweight="bold")
    ax.text(12.2, 5.6, "Stage III: Final testing", ha="center", fontsize=12, fontweight="bold")

    draw_box(ax, (0.4, 4.2), 2.8, 0.8, "LoveDA RGB images", 9)
    draw_box(ax, (0.4, 3.0), 2.8, 0.8, "7-class semantic masks", 9)
    draw_box(ax, (0.4, 1.6), 2.8, 0.9, "Swin-UPerNet\nsource-domain pretraining", 9)

    arrow(ax, (1.8, 4.2), (1.8, 3.8))
    arrow(ax, (1.8, 3.0), (1.8, 2.5))

    draw_box(ax, (4.2, 3.5), 2.4, 0.9, "Transferred encoder weights", 9)
    arrow(ax, (3.2, 2.05), (4.2, 3.95))

    draw_box(ax, (7.0, 4.2), 2.8, 0.8, "Landslide4Sense\n14-channel patch", 9)
    draw_box(ax, (7.0, 3.0), 2.8, 0.8, "12 MS bands + DEM + slope", 9)
    draw_box(ax, (7.0, 1.6), 2.8, 0.9, "Swin-UPerNet\n2-class segmentation", 9)

    arrow(ax, (8.4, 4.2), (8.4, 3.8))
    arrow(ax, (8.4, 3.0), (8.4, 2.5))
    arrow(ax, (6.6, 3.95), (7.0, 2.05))

    draw_box(ax, (11.0, 4.2), 2.4, 0.8, "Prediction mask", 9)
    draw_box(ax, (11.0, 3.0), 2.4, 0.8, "Quantitative metrics", 9)
    draw_box(ax, (11.0, 1.6), 2.4, 0.9, "Error maps and\nvisual analysis", 9)

    arrow(ax, (9.8, 2.05), (11.0, 4.6))
    arrow(ax, (9.8, 2.05), (11.0, 3.4))
    arrow(ax, (9.8, 2.05), (11.0, 2.05))

    ax.text(
        7.0,
        0.45,
        "Final performance: Test mIoU = 0.7328, Landslide IoU = 0.4793, F1 = 0.6480",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(out_dir / "overall_framework.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / "overall_framework.pdf", bbox_inches="tight")
    plt.close()

    print(f"[OK] Framework figure saved to: {out_dir}")


# -----------------------------
# 2. Main comparison figures
# -----------------------------

def ensure_main_table():
    table_path = TABLE_DIR / "main_comparison_table.csv"
    if table_path.exists():
        return pd.read_csv(table_path)

    rows = [
        ["U-Net", "TrainData only", "None", 0.7154, 0.4460, 0.6169, 0.5995, 0.6352],
        ["DeepLabV3", "TrainData only", "None", 0.6992, 0.4151, 0.5867, 0.5582, 0.6182],
        ["Swin-UPerNet", "TrainData only", "No LoveDA", 0.6877, 0.3916, 0.5628, 0.5789, 0.5476],
        ["Swin-UPerNet", "TrainData only", "LoveDA", 0.7136, 0.4426, 0.6136, 0.5912, 0.6378],
        ["Ours Final", "TrainData + ValidData", "LoveDA", 0.7328, 0.4793, 0.6480, 0.6357, 0.6609],
    ]

    df = pd.DataFrame(rows, columns=[
        "Method", "Train_Setting", "Pretraining",
        "Test_mIoU", "Landslide_IoU", "Landslide_F1",
        "Landslide_Precision", "Landslide_Recall"
    ])
    table_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(table_path, index=False)
    save_df_markdown(df, TABLE_DIR / "main_comparison_table.md")
    return df


def method_labels(df):
    labels = []
    for _, r in df.iterrows():
        m = str(r["Method"])
        p = str(r["Pretraining"])
        if m == "Swin-UPerNet" and p == "No LoveDA":
            labels.append("Swin\nw/o LoveDA")
        elif m == "Swin-UPerNet" and p == "LoveDA":
            labels.append("Swin\n+ LoveDA")
        elif m == "Ours Final":
            labels.append("Ours\nFinal")
        else:
            labels.append(m)
    return labels


def plot_single_bar(df, metric, ylabel, filename):
    out_dir = FIG_DIR / "comparison"
    labels = method_labels(df)
    values = df[metric].astype(float).values
    x = np.arange(len(values))

    plt.figure(figsize=(8, 5))
    bars = plt.bar(x, values)
    plt.xticks(x, labels, fontsize=10)
    plt.ylabel(ylabel)
    plt.ylim(0, max(values) * 1.18)
    plt.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_grouped_main_metrics(df):
    out_dir = FIG_DIR / "comparison"
    labels = method_labels(df)
    metrics = ["Test_mIoU", "Landslide_IoU", "Landslide_F1"]
    metric_labels = ["mIoU", "Landslide IoU", "F1"]

    x = np.arange(len(df))
    width = 0.24

    plt.figure(figsize=(10, 5))

    for i, metric in enumerate(metrics):
        plt.bar(x + (i - 1) * width, df[metric].astype(float), width, label=metric_labels[i])

    plt.xticks(x, labels)
    plt.ylabel("Metric")
    plt.ylim(0, 0.85)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "grouped_main_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_precision_recall(df):
    out_dir = FIG_DIR / "comparison"
    labels = method_labels(df)
    x = np.arange(len(df))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, df["Landslide_Precision"].astype(float), width, label="Precision")
    plt.bar(x + width / 2, df["Landslide_Recall"].astype(float), width, label="Recall")
    plt.xticks(x, labels)
    plt.ylabel("Metric")
    plt.ylim(0, 0.75)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "bar_precision_recall.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_main_comparison():
    df = ensure_main_table()
    plot_single_bar(df, "Landslide_IoU", "Landslide IoU", "bar_landslide_iou.png")
    plot_single_bar(df, "Landslide_F1", "Landslide F1", "bar_landslide_f1.png")
    plot_single_bar(df, "Test_mIoU", "Test mIoU", "bar_test_miou.png")
    plot_grouped_main_metrics(df)
    plot_precision_recall(df)
    print(f"[OK] Main comparison figures saved to: {FIG_DIR / 'comparison'}")


# -----------------------------
# 3. Ablation and improvement figures
# -----------------------------

def ensure_training_strategy_table():
    path = TABLE_DIR / "training_strategy_ablation.csv"
    if path.exists():
        return pd.read_csv(path)

    rows = [
        ["Swin-UPerNet without LoveDA", "TrainData only", 0.6877, 0.3916, 0.5628, 0.5789, 0.5476],
        ["Swin-UPerNet + LoveDA", "TrainData only", 0.7136, 0.4426, 0.6136, 0.5912, 0.6378],
        ["Final V8 TrainVal", "TrainData + ValidData", 0.7328, 0.4793, 0.6480, 0.6357, 0.6609],
        ["Continue fine-tuning", "TrainVal + further fine-tuning", 0.7292, 0.4724, 0.6417, 0.6257, 0.6585],
        ["Model Soup top4", "Checkpoint averaging", 0.7315, 0.4766, 0.6455, 0.6426, 0.6485],
        ["Weighted Soup alpha=0.95", "Weighted checkpoint averaging", 0.7327, 0.4792, 0.6479, 0.6356, 0.6607],
    ]

    df = pd.DataFrame(rows, columns=[
        "Variant", "Training_Setting", "Test_mIoU", "Landslide_IoU",
        "Landslide_F1", "Precision", "Recall"
    ])
    df.to_csv(path, index=False)
    save_df_markdown(df, TABLE_DIR / "training_strategy_ablation.md")
    return df


def ensure_improvement_table():
    path = TABLE_DIR / "improvement_summary.csv"
    if path.exists():
        return pd.read_csv(path)

    rows = [
        ["Swin + LoveDA vs Swin without LoveDA", 0.3916, 0.4426, 0.4426 - 0.3916, (0.4426 - 0.3916) / 0.3916 * 100],
        ["Final V8 vs Swin + LoveDA", 0.4426, 0.4793, 0.4793 - 0.4426, (0.4793 - 0.4426) / 0.4426 * 100],
        ["Final V8 vs U-Net", 0.4460, 0.4793, 0.4793 - 0.4460, (0.4793 - 0.4460) / 0.4460 * 100],
        ["Final V8 vs DeepLabV3", 0.4151, 0.4793, 0.4793 - 0.4151, (0.4793 - 0.4151) / 0.4151 * 100],
    ]
    df = pd.DataFrame(rows, columns=[
        "Comparison", "Baseline_Landslide_IoU", "Improved_Landslide_IoU",
        "Absolute_Gain", "Relative_Gain_%"
    ])
    for c in ["Baseline_Landslide_IoU", "Improved_Landslide_IoU", "Absolute_Gain", "Relative_Gain_%"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(4)
    df.to_csv(path, index=False)
    save_df_markdown(df, TABLE_DIR / "improvement_summary.md")
    return df


def plot_ablation_and_improvement():
    out_dir = FIG_DIR / "comparison"
    strategy = ensure_training_strategy_table()
    imp = ensure_improvement_table()

    # Strategy Landslide IoU
    labels = [
        "No\nLoveDA",
        "+LoveDA",
        "Final\nV8",
        "Further\nFT",
        "Soup\nTop4",
        "Weighted\nSoup"
    ][:len(strategy)]

    plt.figure(figsize=(9, 5))
    x = np.arange(len(strategy))
    vals = strategy["Landslide_IoU"].astype(float).values
    bars = plt.bar(x, vals)
    plt.xticks(x, labels, fontsize=9)
    plt.ylabel("Landslide IoU")
    plt.ylim(0, max(vals) * 1.18)
    plt.grid(axis="y", alpha=0.3)

    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "training_strategy_landslide_iou.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Improvement absolute gain
    plt.figure(figsize=(9, 5))
    x = np.arange(len(imp))
    vals = imp["Absolute_Gain"].astype(float).values
    short_labels = ["LoveDA\npretrain", "TrainVal\nfinal", "vs\nU-Net", "vs\nDeepLabV3"]
    bars = plt.bar(x, vals)
    plt.xticks(x, short_labels, fontsize=10)
    plt.ylabel("Absolute gain in Landslide IoU")
    plt.grid(axis="y", alpha=0.3)

    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"+{v:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "improvement_absolute_gain.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Improvement relative gain
    plt.figure(figsize=(9, 5))
    vals = imp["Relative_Gain_%"].astype(float).values
    bars = plt.bar(x, vals)
    plt.xticks(x, short_labels, fontsize=10)
    plt.ylabel("Relative gain (%)")
    plt.grid(axis="y", alpha=0.3)

    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "improvement_relative_gain.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Ablation and improvement figures saved to: {out_dir}")


# -----------------------------
# 4. Training curves
# -----------------------------

def pick_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


def plot_training_curve(csv_path, name):
    out_dir = FIG_DIR / "curves"
    csv_path = Path(csv_path)
    df = safe_read_csv(csv_path)

    if df is None:
        return

    epoch_col = pick_col(df, ["epoch", "Epoch"])

    if epoch_col is None:
        df["epoch"] = range(1, len(df) + 1)
        epoch_col = "epoch"

    train_loss = pick_col(df, ["train_loss", "Train_Loss", "loss_train"])
    val_loss = pick_col(df, ["val_loss", "Val_Loss", "loss", "Loss"])

    plt.figure(figsize=(7, 5))
    plotted = False

    if train_loss:
        plt.plot(df[epoch_col], df[train_loss], label="Train Loss")
        plotted = True

    if val_loss:
        plt.plot(df[epoch_col], df[val_loss], label="Val Loss")
        plotted = True

    if plotted:
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{name} Loss Curve")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_loss_curve.png", dpi=300, bbox_inches="tight")

    plt.close()

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

    if plotted:
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title(f"{name} Metric Curve")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_metric_curve.png", dpi=300, bbox_inches="tight")

    plt.close()


def plot_all_training_curves():
    items = [
        (ROOT / "runs/unet_14ch/logs/metrics.csv", "unet_14ch"),
        (ROOT / "runs/deeplabv3_14ch/logs/metrics.csv", "deeplabv3_14ch"),
        (ROOT / "runs/swin_no_loveda/logs/metrics.csv", "swin_no_loveda"),
        (Path("/root/autodl-tmp/logs/train_l4s_qz_v8_trainval/metrics.csv"), "final_v8_trainval"),
        (Path("/root/autodl-tmp/logs/train_l4s_qz/metrics.csv"), "swin_loveda_train_only"),
    ]

    for csv_path, name in items:
        plot_training_curve(csv_path, name)

    print(f"[OK] Training curves saved to: {FIG_DIR / 'curves'}")


# -----------------------------
# 5. Confusion matrix
# -----------------------------

def get_final_cm():
    csv_path = ROOT / "runs/final_v8_trainval/logs/test_metrics.csv"
    df = safe_read_csv(csv_path)

    if df is not None:
        row = df.iloc[-1].to_dict()
        keys = {str(k).strip(): v for k, v in row.items()}
        if all(k in keys for k in ["TN", "FP", "FN", "TP"]):
            return np.array([[keys["TN"], keys["FP"]], [keys["FN"], keys["TP"]]], dtype=np.float64)

    # fallback from your final test log
    return np.array([
        [12765924, 93745],
        [83950, 163581]
    ], dtype=np.float64)


def plot_cm(cm, normalize, out_path, title):
    class_names = ["Non-landslide", "Landslide"]

    if normalize:
        cm_plot = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-12)
        vmax = 1.0
    else:
        cm_plot = cm.copy()
        vmax = cm_plot.max()

    fig, ax = plt.subplots(figsize=(6.4, 5.6))

    im = ax.imshow(
        cm_plot,
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=vmax,
        aspect="equal",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Prediction", fontsize=11)
    ax.set_ylabel("Ground truth", fontsize=11)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)

    ax.set_xticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    threshold = cm_plot.max() * 0.55

    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            value = cm_plot[i, j]

            if normalize:
                text = f"{value * 100:.2f}%"
            else:
                text = f"{int(round(cm[i, j])):,}"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white" if value > threshold else "#1f2937",
            )

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_final_confusion_matrix():
    out_dir = FIG_DIR / "confusion_matrix"
    cm = get_final_cm()

    plot_cm(
        cm,
        False,
        out_dir / "final_v8_confusion_matrix_raw.png",
        "Final V8 Confusion Matrix"
    )

    plot_cm(
        cm,
        True,
        out_dir / "final_v8_confusion_matrix_normalized.png",
        "Final V8 Normalized Confusion Matrix"
    )

    cm_df = pd.DataFrame(cm, index=["GT_NonLandslide", "GT_Landslide"], columns=["Pred_NonLandslide", "Pred_Landslide"])
    cm_df.to_csv(out_dir / "final_v8_confusion_matrix.csv")

    print(f"[OK] Confusion matrix figures saved to: {out_dir}")


# -----------------------------
# 6. Dataset distribution
# -----------------------------

def mask_dir_for_split(data_root, split):
    if split == "TestData":
        return Path(data_root) / split / "test"
    return Path(data_root) / split / "mask"


def find_masks(data_root, split):
    mdir = mask_dir_for_split(data_root, split)
    return sorted(mdir.glob("*.h5"))


def plot_dataset_distribution():
    out_dir = FIG_DIR / "dataset"
    rows = []
    sample_rows = []

    for split in ["TrainData", "ValidData", "TestData"]:
        masks = find_masks(DATA_ROOT, split)
        non_count = 0
        land_count = 0

        for p in masks:
            m = read_h5(p).astype(np.uint8)
            non = int((m == 0).sum())
            land = int((m == 1).sum())

            non_count += non
            land_count += land

            sample_rows.append({
                "Split": split,
                "Mask": p.name,
                "NonLandslide_Pixels": non,
                "Landslide_Pixels": land,
                "Positive_Ratio_%": land / max(non + land, 1) * 100,
            })

        total = non_count + land_count
        rows.append({
            "Split": split,
            "Num_Masks": len(masks),
            "NonLandslide_Pixels": non_count,
            "Landslide_Pixels": land_count,
            "Total_Pixels": total,
            "Landslide_Ratio_%": land_count / max(total, 1) * 100,
        })

    df = pd.DataFrame(rows)
    sdf = pd.DataFrame(sample_rows)

    df.to_csv(out_dir / "dataset_pixel_distribution.csv", index=False)
    sdf.to_csv(out_dir / "sample_positive_ratio_distribution.csv", index=False)

    # Split ratio bar
    plt.figure(figsize=(7, 5))
    x = np.arange(len(df))
    vals = df["Landslide_Ratio_%"].astype(float).values
    bars = plt.bar(x, vals)
    plt.xticks(x, df["Split"].values)
    plt.ylabel("Landslide pixel ratio (%)")
    plt.grid(axis="y", alpha=0.3)

    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "landslide_pixel_ratio_by_split.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Overall pie
    total_non = df["NonLandslide_Pixels"].sum()
    total_land = df["Landslide_Pixels"].sum()

    plt.figure(figsize=(5.5, 5.5))
    plt.pie(
        [total_non, total_land],
        labels=["Non-landslide", "Landslide"],
        autopct="%1.2f%%",
        startangle=90
    )
    plt.title("Overall pixel-level class distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "overall_class_distribution_pie.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Sample positive ratio histogram
    plt.figure(figsize=(7, 5))
    for split in ["TrainData", "ValidData", "TestData"]:
        tmp = sdf[sdf["Split"] == split]
        plt.hist(tmp["Positive_Ratio_%"], bins=30, alpha=0.45, label=split)

    plt.xlabel("Landslide pixel ratio per sample (%)")
    plt.ylabel("Number of samples")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "sample_positive_ratio_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(df.to_string(index=False))
    print(f"[OK] Dataset distribution figures saved to: {out_dir}")


# -----------------------------
# 7. Final model visualization
# -----------------------------

def load_swin_model(img_size=128, device="cuda"):
    sys.path.append("/root/autodl-tmp/train_l4s_qz")

    try:
        from model import SwinUPerNet
    except Exception as e:
        raise RuntimeError(f"Cannot import SwinUPerNet from /root/autodl-tmp/train_l4s_qz/model.py: {e}")

    candidates = [
        dict(num_classes=2, img_size=img_size),
        dict(num_classes=2),
        dict(in_chans=14, num_classes=2, img_size=img_size),
        dict(in_channels=14, num_classes=2, img_size=img_size),
        dict(),
    ]

    last_error = None
    model = None

    for kwargs in candidates:
        try:
            model = SwinUPerNet(**kwargs)
            print(f"[Info] Built SwinUPerNet with args: {kwargs}")
            break
        except Exception as e:
            last_error = e
            model = None

    if model is None:
        raise RuntimeError(f"Failed to build SwinUPerNet. Last error: {last_error}")

    ckpt = torch.load(FINAL_CKPT, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    try:
        msg = model.load_state_dict(state, strict=False)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to load state dict. Model constructor may not match training model. Error: {e}")

    print("[Info] Missing keys:", len(msg.missing_keys))
    print("[Info] Unexpected keys:", len(msg.unexpected_keys))

    model = model.to(device)
    model.eval()

    return model


def percentile_stretch(x):
    x = x.astype(np.float32)
    out = np.zeros_like(x, dtype=np.float32)
    for c in range(x.shape[-1]):
        p2, p98 = np.percentile(x[..., c], 2), np.percentile(x[..., c], 98)
        out[..., c] = np.clip((x[..., c] - p2) / (p98 - p2 + 1e-6), 0, 1)
    return out


def make_rgb(img14):
    if img14.shape[-1] >= 4:
        rgb = np.stack([img14[..., 3], img14[..., 2], img14[..., 1]], axis=-1)
    else:
        rgb = np.repeat(img14[..., :1], 3, axis=-1)
    return percentile_stretch(rgb)


def make_false_color(img14):
    if img14.shape[-1] >= 8:
        fc = np.stack([img14[..., 7], img14[..., 3], img14[..., 2]], axis=-1)
    else:
        fc = make_rgb(img14)
    return percentile_stretch(fc)


def overlay_mask(rgb, mask, color=(1, 0, 0), alpha=0.35):
    out = rgb.copy()
    color_arr = np.zeros_like(rgb)
    color_arr[..., 0] = color[0]
    color_arr[..., 1] = color[1]
    color_arr[..., 2] = color[2]

    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * color_arr[m]
    return np.clip(out, 0, 1)


def make_error_map(gt, pred):
    err = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.float32)

    tp = (gt == 1) & (pred == 1)
    fp = (gt == 0) & (pred == 1)
    fn = (gt == 1) & (pred == 0)

    err[tp] = [0.0, 1.0, 0.0]    # green
    err[fp] = [1.0, 0.0, 0.0]    # red
    err[fn] = [0.0, 0.2, 1.0]    # blue

    return err


def sample_metrics(gt, pred):
    gt = gt.astype(np.uint8)
    pred = pred.astype(np.uint8)

    tp = int(((gt == 1) & (pred == 1)).sum())
    fp = int(((gt == 0) & (pred == 1)).sum())
    fn = int(((gt == 1) & (pred == 0)).sum())
    tn = int(((gt == 0) & (pred == 0)).sum())

    iou = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)

    pos_ratio = float((gt == 1).mean() * 100)

    if pos_ratio < 1:
        area_group = "0-1%"
    elif pos_ratio < 3:
        area_group = "1-3%"
    elif pos_ratio < 5:
        area_group = "3-5%"
    else:
        area_group = ">5%"

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Positive_Ratio_%": pos_ratio,
        "Area_Group": area_group,
    }


@torch.no_grad()
def predict(model, img14, device="cuda", tta=True):
    x = torch.from_numpy(img14.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    logits = model(x)

    if tta:
        xf = torch.flip(x, dims=[3])
        logit_f = model(xf)
        logit_f = torch.flip(logit_f, dims=[3])
        logits = 0.5 * (logits + logit_f)

    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred


def find_mask_for_img(mask_dir, img_path):
    idx = img_path.stem.split("_")[-1]
    candidates = [
        mask_dir / f"mask_{idx}.h5",
        mask_dir / img_path.name,
        mask_dir / f"{img_path.stem}.h5",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(f"Mask not found for image {img_path}")


def plot_single_visual(name, img, gt, pred, metrics, out_path):
    rgb = make_rgb(img)
    false_color = make_false_color(img)
    gt_overlay = overlay_mask(rgb, gt, color=(0, 1, 0), alpha=0.35)
    pred_overlay = overlay_mask(rgb, pred, color=(1, 0, 0), alpha=0.35)
    err = make_error_map(gt, pred)

    fig, axes = plt.subplots(1, 6, figsize=(21, 4))

    axes[0].imshow(rgb)
    axes[0].set_title("RGB")

    axes[1].imshow(false_color)
    axes[1].set_title("False color")

    axes[2].imshow(gt_overlay)
    axes[2].set_title("Ground truth")

    axes[3].imshow(pred_overlay)
    axes[3].set_title("Prediction")

    axes[4].imshow(err)
    axes[4].set_title("Error map\nGreen=TP Red=FP Blue=FN")

    axes[5].imshow(rgb)
    try:
        axes[5].contour(gt, levels=[0.5], linewidths=0.8)
        axes[5].contour(pred, levels=[0.5], linewidths=0.8)
    except Exception:
        pass
    axes[5].set_title(f"Boundary\nIoU={metrics['IoU']:.3f}")

    for ax in axes:
        ax.axis("off")

    plt.suptitle(f"{name} | IoU={metrics['IoU']:.3f}, F1={metrics['F1']:.3f}, Pos={metrics['Positive_Ratio_%']:.2f}%")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_montage(selected, out_path, title):
    n = len(selected)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 4, figsize=(14, 3.2 * n))

    if n == 1:
        axes = np.expand_dims(axes, 0)

    for i, item in enumerate(selected):
        name, img, gt, pred, metrics = item
        rgb = make_rgb(img)
        pred_overlay = overlay_mask(rgb, pred, color=(1, 0, 0), alpha=0.35)
        err = make_error_map(gt, pred)

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title("Input")

        axes[i, 1].imshow(gt, cmap="gray")
        axes[i, 1].set_title("GT")

        axes[i, 2].imshow(pred_overlay)
        axes[i, 2].set_title("Prediction")

        axes[i, 3].imshow(err)
        axes[i, 3].set_title(f"Error | IoU={metrics['IoU']:.3f}")

        for j in range(4):
            axes[i, j].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def visualize_final_model(num_samples=16, device="cuda"):
    out_dir = FIG_DIR / "visualizations" / "final_v8"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not FINAL_CKPT.exists():
        print(f"[Skip] Final checkpoint missing: {FINAL_CKPT}")
        return

    if not DATA_ROOT_TRAINVAL.exists():
        print(f"[Skip] TrainVal data root missing: {DATA_ROOT_TRAINVAL}")
        return

    img_dir = DATA_ROOT_TRAINVAL / "TestData" / "img"
    mask_dir = DATA_ROOT_TRAINVAL / "TestData" / "test"

    img_paths = sorted(img_dir.glob("*.h5"))

    if len(img_paths) == 0:
        print(f"[Skip] No test images found in {img_dir}")
        return

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    try:
        model = load_swin_model(img_size=128, device=device)
    except Exception:
        print("[Warning] Visualization failed while loading model.")
        traceback.print_exc()
        return

    rows = []
    cache = []

    for img_path in img_paths:
        try:
            mask_path = find_mask_for_img(mask_dir, img_path)
            img = read_h5(img_path).astype(np.float32)
            gt = read_h5(mask_path).astype(np.uint8)
            pred = predict(model, img, device=device, tta=True)

            m = sample_metrics(gt, pred)
            m["Sample"] = img_path.name
            rows.append(m)
            cache.append((img_path.name, img, gt, pred, m))
        except Exception as e:
            print(f"[Warning] Failed sample {img_path.name}: {e}")

    if len(cache) == 0:
        print("[Skip] No valid visualization samples.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_sample_metrics.csv", index=False)

    order = ["0-1%", "1-3%", "3-5%", ">5%"]
    group = df.groupby("Area_Group").agg({
        "IoU": "mean",
        "Precision": "mean",
        "Recall": "mean",
        "F1": "mean",
        "Sample": "count",
        "Positive_Ratio_%": "mean",
    }).reset_index().rename(columns={"Sample": "Num_Samples"})

    group["Area_Group"] = pd.Categorical(group["Area_Group"], categories=order, ordered=True)
    group = group.sort_values("Area_Group")
    group.to_csv(out_dir / "area_group_metrics.csv", index=False)

    # Select mixed cases: worst + best
    cache_sorted = sorted(cache, key=lambda x: x[4]["IoU"])
    k = max(num_samples // 2, 1)
    selected = cache_sorted[:k] + cache_sorted[-k:]
    selected = selected[:num_samples]

    indiv_dir = out_dir / "individual_cases"
    indiv_dir.mkdir(parents=True, exist_ok=True)

    for name, img, gt, pred, m in selected:
        safe = name.replace(".h5", "")
        plot_single_visual(name, img, gt, pred, m, indiv_dir / f"vis_{safe}.png")

    worst = cache_sorted[:min(6, len(cache_sorted))]
    best = cache_sorted[-min(6, len(cache_sorted)):]

    plot_montage(worst, out_dir / "montage_failure_cases.png", "Failure cases with low sample-level IoU")
    plot_montage(best, out_dir / "montage_success_cases.png", "Successful cases with high sample-level IoU")
    plot_montage(selected[:min(8, len(selected))], out_dir / "montage_mixed_cases.png", "Mixed qualitative prediction and error maps")

    # Per-sample IoU histogram
    plt.figure(figsize=(7, 5))
    plt.hist(df["IoU"], bins=30)
    plt.xlabel("Sample-level Landslide IoU")
    plt.ylabel("Number of samples")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "sample_distribution" / "per_sample_iou_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Positive ratio vs IoU scatter
    plt.figure(figsize=(7, 5))
    plt.scatter(df["Positive_Ratio_%"], df["IoU"], s=15, alpha=0.7)
    plt.xlabel("Landslide pixel ratio per sample (%)")
    plt.ylabel("Sample-level Landslide IoU")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "sample_distribution" / "positive_ratio_vs_iou_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Final model visualizations saved to: {out_dir}")
    print(group.to_string(index=False))


# -----------------------------
# 8. Area-group figures
# -----------------------------

def plot_area_group_metrics():
    csv_path = FIG_DIR / "visualizations" / "final_v8" / "area_group_metrics.csv"
    out_dir = FIG_DIR / "area_group"
    df = safe_read_csv(csv_path)

    if df is None:
        print("[Skip] Area group figure requires final_v8/area_group_metrics.csv")
        return

    order = ["0-1%", "1-3%", "3-5%", ">5%"]

    if "Area_Group" in df.columns:
        df["Area_Group"] = pd.Categorical(df["Area_Group"], categories=order, ordered=True)
        df = df.sort_values("Area_Group")

    metrics = ["IoU", "Precision", "Recall", "F1"]
    available = [m for m in metrics if m in df.columns]

    x = np.arange(len(df))
    width = 0.18

    plt.figure(figsize=(9, 5))

    for i, m in enumerate(available):
        plt.bar(
            x + (i - len(available) / 2) * width + width / 2,
            df[m].astype(float),
            width,
            label=m
        )

    plt.xticks(x, df["Area_Group"].astype(str))
    plt.xlabel("Landslide pixel ratio group")
    plt.ylabel("Metric")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "area_group_metrics_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Num samples
    if "Num_Samples" in df.columns:
        plt.figure(figsize=(7, 5))
        bars = plt.bar(np.arange(len(df)), df["Num_Samples"].astype(float))
        plt.xticks(np.arange(len(df)), df["Area_Group"].astype(str))
        plt.xlabel("Landslide pixel ratio group")
        plt.ylabel("Number of samples")
        plt.grid(axis="y", alpha=0.3)

        for b, v in zip(bars, df["Num_Samples"].astype(float)):
            plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(v)}", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(out_dir / "area_group_sample_counts.png", dpi=300, bbox_inches="tight")
        plt.close()

    print(f"[OK] Area-group figures saved to: {out_dir}")


# -----------------------------
# 9. Complexity
# -----------------------------

def count_state_params(ckpt_path):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    total = 0
    for k, v in state.items():
        if torch.is_tensor(v):
            total += v.numel()
    return total


def plot_complexity():
    out_dir = FIG_DIR / "complexity"

    items = [
        ("U-Net", ROOT / "runs/unet_14ch/checkpoints/best.pth", 0.4460),
        ("DeepLabV3", ROOT / "runs/deeplabv3_14ch/checkpoints/best.pth", 0.4151),
        ("Swin-UPerNet\nw/o LoveDA", ROOT / "runs/swin_no_loveda/checkpoints/l4s_best.pth", 0.3916),
        ("Ours Final", FINAL_CKPT, 0.4793),
    ]

    rows = []

    for name, ckpt, iou in items:
        params = count_state_params(ckpt)
        if params is None:
            print(f"[Skip] Missing ckpt for complexity: {ckpt}")
            continue

        rows.append({
            "Model": name.replace("\n", " "),
            "Checkpoint": str(ckpt),
            "Params_M": round(params / 1e6, 3),
            "Input_Size": "14x128x128",
            "Landslide_IoU": iou,
        })

    if len(rows) == 0:
        print("[Skip] No complexity rows.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "model_complexity.csv", index=False)

    # Params bar
    plt.figure(figsize=(8, 5))
    labels = df["Model"].values
    vals = df["Params_M"].astype(float).values
    bars = plt.bar(np.arange(len(df)), vals)
    plt.xticks(np.arange(len(df)), labels, rotation=15)
    plt.ylabel("Parameters (M)")
    plt.grid(axis="y", alpha=0.3)

    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}M", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "model_params_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Params vs IoU scatter
    plt.figure(figsize=(7, 5))
    plt.scatter(df["Params_M"], df["Landslide_IoU"], s=80)

    for _, r in df.iterrows():
        plt.text(r["Params_M"], r["Landslide_IoU"], r["Model"], fontsize=8)

    plt.xlabel("Parameters (M)")
    plt.ylabel("Landslide IoU")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "params_vs_landslide_iou.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(df.to_string(index=False))
    print(f"[OK] Complexity figures saved to: {out_dir}")


# -----------------------------
# Main
# -----------------------------

def main():
    print("=" * 120)
    print("Generating all Q2 paper figures")
    print("=" * 120)

    steps = [
        ("Framework", plot_framework),
        ("Main comparison", plot_main_comparison),
        ("Ablation and improvement", plot_ablation_and_improvement),
        ("Training curves", plot_all_training_curves),
        ("Confusion matrix", plot_final_confusion_matrix),
        ("Dataset distribution", plot_dataset_distribution),
        ("Final prediction visualization", lambda: visualize_final_model(num_samples=16, device="cuda")),
        ("Area group figures", plot_area_group_metrics),
        ("Complexity", plot_complexity),
    ]

    failed = []

    for name, fn in steps:
        print("\n" + "=" * 120)
        print(f"[Running] {name}")
        print("=" * 120)

        try:
            fn()
        except Exception as e:
            print(f"[Failed] {name}: {e}")
            traceback.print_exc()
            failed.append(name)

    # Result index
    index_path = ROOT / "FIGURE_INDEX.md"
    index_path.write_text(
        """# Q2 Experiment Figure Index

## Framework
- figures/framework/overall_framework.png
- figures/framework/overall_framework.pdf

## Main comparison
- figures/comparison/bar_landslide_iou.png
- figures/comparison/bar_landslide_f1.png
- figures/comparison/bar_test_miou.png
- figures/comparison/grouped_main_metrics.png
- figures/comparison/bar_precision_recall.png

## Ablation and improvement
- figures/comparison/training_strategy_landslide_iou.png
- figures/comparison/improvement_absolute_gain.png
- figures/comparison/improvement_relative_gain.png

## Training curves
- figures/curves/

## Confusion matrix
- figures/confusion_matrix/final_v8_confusion_matrix_raw.png
- figures/confusion_matrix/final_v8_confusion_matrix_normalized.png

## Dataset distribution
- figures/dataset/landslide_pixel_ratio_by_split.png
- figures/dataset/overall_class_distribution_pie.png
- figures/dataset/sample_positive_ratio_histogram.png

## Final prediction and error maps
- figures/visualizations/final_v8/individual_cases/
- figures/visualizations/final_v8/montage_failure_cases.png
- figures/visualizations/final_v8/montage_success_cases.png
- figures/visualizations/final_v8/montage_mixed_cases.png

## Sample-level analysis
- figures/sample_distribution/per_sample_iou_histogram.png
- figures/sample_distribution/positive_ratio_vs_iou_scatter.png

## Area-group analysis
- figures/area_group/area_group_metrics_bar.png
- figures/area_group/area_group_sample_counts.png

## Complexity
- figures/complexity/model_params_bar.png
- figures/complexity/params_vs_landslide_iou.png
- tables/model_complexity.csv
""",
        encoding="utf-8"
    )

    print("\n" + "=" * 120)
    print("All figure generation finished.")
    print(f"Figure index saved to: {index_path}")

    if failed:
        print("Failed steps:")
        for x in failed:
            print(" -", x)
    else:
        print("No failed steps.")

    print("=" * 120)


if __name__ == "__main__":
    main()
