from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(r"C:\Users\yulmf\Desktop\地质all")
FIG_DIR = ROOT / "paper_draft" / "figures"
EXT_DIR = ROOT / "external_sen12_rgb"

FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 160


def add_value_labels(ax, bars, fmt="{:.4f}", fontsize=8):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.006,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def clean_bar(
    labels,
    values,
    title,
    ylabel,
    out_path,
    ylim=None,
    figsize=(5.8, 3.4),
):
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    bars = ax.bar(x, values, width=0.58)

    ax.set_title(title, fontsize=11, pad=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(0, max(values) * 1.22)

    ax.grid(axis="y", alpha=0.25)
    add_value_labels(ax, bars)

    fig.tight_layout()
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)


def plot_individual_ablation_figures():
    clean_bar(
        labels=["RGB", "12MS", "RGB+Topo", "Full14"],
        values=[0.3983, 0.4203, 0.4215, 0.4231],
        title="Input modality ablation",
        ylabel="Landslide IoU",
        out_path=FIG_DIR / "input_modality_landslide_iou.png",
        ylim=(0, 0.50),
    )

    clean_bar(
        labels=["Random extra", "RGB-mean extra"],
        values=[0.4470, 0.4231],
        title="Patch embedding adaptation ablation",
        ylabel="Landslide IoU",
        out_path=FIG_DIR / "patch_embedding_landslide_iou.png",
        ylim=(0, 0.52),
    )

    clean_bar(
        labels=["CE", "CE+Dice", "CE+Dice+CW"],
        values=[0.3812, 0.4321, 0.4231],
        title="Loss and class imbalance ablation",
        ylabel="Landslide IoU",
        out_path=FIG_DIR / "loss_ablation_landslide_iou.png",
        ylim=(0, 0.52),
    )

    clean_bar(
        labels=["w/o TTA", "w/ TTA"],
        values=[0.4739, 0.4793],
        title="Test-time augmentation ablation",
        ylabel="Landslide IoU",
        out_path=FIG_DIR / "tta_ablation_landslide_iou.png",
        ylim=(0, 0.56),
    )


def plot_ablation_overview_panel():
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0))
    axes = axes.ravel()

    panels = [
        (
            ["RGB", "12MS", "RGB+Topo", "Full14"],
            [0.3983, 0.4203, 0.4215, 0.4231],
            "(a) Input modality",
        ),
        (
            ["Random extra", "RGB-mean extra"],
            [0.4470, 0.4231],
            "(b) Patch embedding",
        ),
        (
            ["CE", "CE+Dice", "CE+Dice+CW"],
            [0.3812, 0.4321, 0.4231],
            "(c) Loss design",
        ),
        (
            ["w/o TTA", "w/ TTA"],
            [0.4739, 0.4793],
            "(d) Test-time augmentation",
        ),
    ]

    for ax, (labels, values, title) in zip(axes, panels):
        x = np.arange(len(labels))
        bars = ax.bar(x, values, width=0.58)
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_ylabel("Landslide IoU", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
        ax.set_ylim(0, 0.56)
        ax.grid(axis="y", alpha=0.25)

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.007,
                f"{h:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout(w_pad=2.0, h_pad=2.2)
    out_path = FIG_DIR / "ablation_overview_clean.png"
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)


def plot_seed_stability_clean():
    seeds = [2026, 2027, 2028]
    values = [0.4248, 0.4322, 0.4351]

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.plot(seeds, values, marker="o")
    ax.set_title("Seed stability", fontsize=11, pad=8)
    ax.set_xlabel("Random seed", fontsize=10)
    ax.set_ylabel("Landslide IoU", fontsize=10)
    ax.set_xticks(seeds)
    ax.set_xticklabels([str(s) for s in seeds], rotation=0)
    ax.grid(alpha=0.25)

    for x, y in zip(seeds, values):
        ax.text(x, y + 0.0008, f"{y:.4f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_path = FIG_DIR / "seed_stability_landslide_iou.png"
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)


def read_fewshot_history():
    path_candidates = [
        EXT_DIR / "fewshot_run" / "tables" / "fewshot_train_history.csv",
        ROOT / "paper_draft" / "tables" / "fewshot_train_history.csv",
    ]

    for path in path_candidates:
        if path.exists():
            rows = []
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
            return rows

    return []


def plot_external_metric_comparison_clean():
    metrics = ["L-IoU", "F1", "Precision", "Recall", "mIoU"]
    zero = [0.0256, 0.0499, 0.3036, 0.0272, 0.4982]
    few = [0.3143, 0.4782, 0.4895, 0.4675, 0.6426]

    x = np.arange(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    b1 = ax.bar(x - width / 2, zero, width, label="Zero-shot")
    b2 = ax.bar(x + width / 2, few, width, label="Few-shot 50")

    ax.set_title("External validation on Sen12Landslides RGB subset", fontsize=10.5, pad=8)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=0, ha="center", fontsize=8.5)
    ax.set_ylim(0, 0.72)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8.5)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.012,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.tight_layout()
    out_path = FIG_DIR / "sen12_rgb_external_metric_comparison.png"
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)


def plot_fewshot_training_curve_clean():
    rows = read_fewshot_history()
    if not rows:
        print("Warning: fewshot_train_history.csv not found. Skip training curve.")
        return

    epochs = [int(float(r["epoch"])) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    land_iou = [float(r["landslide_iou"]) for r in rows]
    f1 = [float(r["landslide_f1"]) for r in rows]

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(epochs, train_loss, label="Train loss")
    ax.plot(epochs, land_iou, label="Val Landslide IoU")
    ax.plot(epochs, f1, label="Val F1")

    ax.set_title("Few-shot fine-tuning curve", fontsize=11, pad=8)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Value", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5)

    fig.tight_layout()
    out_path = FIG_DIR / "sen12_rgb_fewshot_training_curve.png"
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)


def plot_external_summary_panel():
    rows = read_fewshot_history()

    metrics = ["L-IoU", "F1", "Precision", "Recall", "mIoU"]
    zero = [0.0256, 0.0499, 0.3036, 0.0272, 0.4982]
    few = [0.3143, 0.4782, 0.4895, 0.4675, 0.6426]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))

    ax = axes[0]
    x = np.arange(len(metrics))
    width = 0.36
    b1 = ax.bar(x - width / 2, zero, width, label="Zero-shot")
    b2 = ax.bar(x + width / 2, few, width, label="Few-shot 50")
    ax.set_title("(a) Metric comparison", fontsize=11, pad=8)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=0, ha="center", fontsize=8.5)
    ax.set_ylim(0, 0.72)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8.5)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.012,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax = axes[1]
    if rows:
        epochs = [int(float(r["epoch"])) for r in rows]
        train_loss = [float(r["train_loss"]) for r in rows]
        land_iou = [float(r["landslide_iou"]) for r in rows]
        f1 = [float(r["landslide_f1"]) for r in rows]

        ax.plot(epochs, train_loss, label="Train loss")
        ax.plot(epochs, land_iou, label="Val Landslide IoU")
        ax.plot(epochs, f1, label="Val F1")
    ax.set_title("(b) Few-shot fine-tuning curve", fontsize=11, pad=8)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Value", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5)

    fig.tight_layout(w_pad=2.0)
    out_path = FIG_DIR / "sen12_rgb_external_summary_clean.png"
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)


def main():
    plot_individual_ablation_figures()
    plot_ablation_overview_panel()
    plot_seed_stability_clean()
    plot_external_metric_comparison_clean()
    plot_fewshot_training_curve_clean()
    plot_external_summary_panel()

    print("\nAll clean figures saved to:")
    print(FIG_DIR)


if __name__ == "__main__":
    main()