from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = Path("/root/autodl-tmp/q2_experiments/figures/confusion_matrix")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = Path("/root/autodl-tmp/q2_experiments/runs/final_v8_trainval/logs/test_metrics.csv")


def load_confusion_matrix():
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        row = df.iloc[-1].to_dict()
        row = {str(k).strip(): v for k, v in row.items()}

        if all(k in row for k in ["TN", "FP", "FN", "TP"]):
            return np.array(
                [
                    [float(row["TN"]), float(row["FP"])],
                    [float(row["FN"]), float(row["TP"])],
                ],
                dtype=np.float64,
            )

    # fallback: final V8 result from test log
    return np.array(
        [
            [12765924, 93745],
            [83950, 163581],
        ],
        dtype=np.float64,
    )


def format_raw_number(x):
    return f"{int(round(x)):,}"


def plot_confusion_matrix(cm, normalize, out_path, title):
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

    # Cell borders for publication clarity
    ax.set_xticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Write values
    threshold = cm_plot.max() * 0.55

    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            value = cm_plot[i, j]

            if normalize:
                text = f"{value * 100:.2f}%"
            else:
                text = format_raw_number(cm[i, j])

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

    # Clean academic style
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    cm = load_confusion_matrix()

    cm_df = pd.DataFrame(
        cm,
        index=["GT_NonLandslide", "GT_Landslide"],
        columns=["Pred_NonLandslide", "Pred_Landslide"],
    )
    cm_df.to_csv(OUT_DIR / "final_v8_confusion_matrix.csv")

    plot_confusion_matrix(
        cm,
        normalize=False,
        out_path=OUT_DIR / "final_v8_confusion_matrix_raw.png",
        title="Final V8 Confusion Matrix",
    )

    plot_confusion_matrix(
        cm,
        normalize=True,
        out_path=OUT_DIR / "final_v8_confusion_matrix_normalized.png",
        title="Final V8 Normalized Confusion Matrix",
    )

    print("=" * 100)
    print(f"Saved blue-white confusion matrices to: {OUT_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    main()
