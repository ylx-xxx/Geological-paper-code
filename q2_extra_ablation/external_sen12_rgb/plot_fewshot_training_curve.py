from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

root = Path("/root/autodl-tmp/q2_extra_ablation/external_sen12_rgb")
csv_path = root / "fewshot_run" / "tables" / "fewshot_train_history.csv"
out_path = root / "figures" / "sen12_rgb_fewshot_training_curve.png"

df = pd.read_csv(csv_path)

plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train_loss"], label="Train loss")
plt.plot(df["epoch"], df["landslide_iou"], label="Val Landslide IoU")
plt.plot(df["epoch"], df["landslide_f1"], label="Val F1")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Few-shot fine-tuning curve on Sen12Landslides RGB subset")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", out_path)
