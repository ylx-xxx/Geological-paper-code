from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Users\yulmf\Desktop\地质all")
FIG_DIR = ROOT / "paper_draft" / "figures"

history_candidates = [
    ROOT / "q2_extra_ablation" / "external_sen12_rgb" / "fewshot_run" / "tables" / "fewshot_train_history.csv",
    ROOT / "external_sen12_rgb" / "fewshot_run" / "tables" / "fewshot_train_history.csv",
    ROOT / "paper_draft" / "tables" / "fewshot_train_history.csv",
]

history_path = None
for p in history_candidates:
    if p.exists():
        history_path = p
        break

if history_path is None:
    raise FileNotFoundError(
        "没有找到 fewshot_train_history.csv。请确认以下路径至少存在一个：\n"
        + "\n".join(str(p) for p in history_candidates)
    )

print("Using history:", history_path)

rows = []
with open(history_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

if len(rows) == 0:
    raise RuntimeError("fewshot_train_history.csv 是空的。")

epochs = [int(float(r["epoch"])) for r in rows]
train_loss = [float(r["train_loss"]) for r in rows]
land_iou = [float(r["landslide_iou"]) for r in rows]
f1 = [float(r["landslide_f1"]) for r in rows]

metrics = ["L-IoU", "F1", "Precision", "Recall", "mIoU"]
zero = [0.0256, 0.0499, 0.3036, 0.0272, 0.4982]
few = [0.3143, 0.4782, 0.4895, 0.4675, 0.6426]

FIG_DIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.7))

ax = axes[0]
x = np.arange(len(metrics))
w = 0.36

bars1 = ax.bar(x - w / 2, zero, width=w, label="Zero-shot")
bars2 = ax.bar(x + w / 2, few, width=w, label="Few-shot 50")

ax.set_title("(a) Metric comparison", fontsize=11)
ax.set_ylabel("Score", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=0, fontsize=9)
ax.set_ylim(0, 0.72)
ax.grid(axis="y", alpha=0.25)
ax.legend(fontsize=8)

for bars in [bars1, bars2]:
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + 0.012,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

ax = axes[1]
ax.plot(epochs, train_loss, label="Train loss")
ax.plot(epochs, land_iou, label="Val Landslide IoU")
ax.plot(epochs, f1, label="Val F1")

ax.set_title("(b) Few-shot fine-tuning curve", fontsize=11)
ax.set_xlabel("Epoch", fontsize=10)
ax.set_ylabel("Value", fontsize=10)
ax.set_xlim(min(epochs), max(epochs))
ax.set_ylim(0, max(max(train_loss), max(land_iou), max(f1)) * 1.08)
ax.grid(alpha=0.25)
ax.legend(fontsize=8)

fig.tight_layout(w_pad=2.0)

out_path = FIG_DIR / "sen12_rgb_external_summary_clean.png"
fig.savefig(out_path, dpi=400, bbox_inches="tight")
plt.close(fig)

print("Saved:", out_path)
