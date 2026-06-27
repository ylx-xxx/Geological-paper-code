from pathlib import Path
import pandas as pd


OUT = Path("/root/autodl-tmp/q2_experiments/tables")
OUT.mkdir(parents=True, exist_ok=True)


def to_markdown_simple(df):
    cols = list(df.columns)
    lines = []

    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            vals.append("" if pd.isna(v) else str(v))
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


rows = [
    {
        "Variant": "Swin-UPerNet without LoveDA",
        "Training_Setting": "TrainData only",
        "Test_mIoU": 0.6877,
        "Landslide_IoU": 0.3916,
        "Landslide_F1": 0.5628,
        "Precision": 0.5789,
        "Recall": 0.5476,
        "Conclusion": "No source-domain pretraining"
    },
    {
        "Variant": "Swin-UPerNet + LoveDA",
        "Training_Setting": "TrainData only",
        "Test_mIoU": 0.7136,
        "Landslide_IoU": 0.4426,
        "Landslide_F1": 0.6136,
        "Precision": 0.5912,
        "Recall": 0.6378,
        "Conclusion": "LoveDA pretraining improves landslide recognition"
    },
    {
        "Variant": "Final V8 TrainVal",
        "Training_Setting": "TrainData + ValidData",
        "Test_mIoU": 0.7328,
        "Landslide_IoU": 0.4793,
        "Landslide_F1": 0.6480,
        "Precision": 0.6357,
        "Recall": 0.6609,
        "Conclusion": "Final selected model"
    },
    {
        "Variant": "Continue fine-tuning",
        "Training_Setting": "TrainVal + further fine-tuning",
        "Test_mIoU": 0.7292,
        "Landslide_IoU": 0.4724,
        "Landslide_F1": 0.6417,
        "Precision": 0.6257,
        "Recall": 0.6585,
        "Conclusion": "Further training causes slight degradation"
    },
    {
        "Variant": "Model Soup top4",
        "Training_Setting": "Checkpoint averaging",
        "Test_mIoU": 0.7315,
        "Landslide_IoU": 0.4766,
        "Landslide_F1": 0.6455,
        "Precision": 0.6426,
        "Recall": 0.6485,
        "Conclusion": "Does not exceed final model"
    },
    {
        "Variant": "Weighted Soup alpha=0.95",
        "Training_Setting": "Weighted checkpoint averaging",
        "Test_mIoU": 0.7327,
        "Landslide_IoU": 0.4792,
        "Landslide_F1": 0.6479,
        "Precision": 0.6356,
        "Recall": 0.6607,
        "Conclusion": "Closest to final model but still lower"
    }
]

df = pd.DataFrame(rows)

for c in ["Test_mIoU", "Landslide_IoU", "Landslide_F1", "Precision", "Recall"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").round(4)

df.to_csv(OUT / "training_strategy_ablation.csv", index=False)

with open(OUT / "training_strategy_ablation.md", "w", encoding="utf-8") as f:
    f.write(to_markdown_simple(df))

print("=" * 120)
print("Training strategy ablation:")
print(df.to_string(index=False))
print("=" * 120)
print(f"Saved to: {OUT / 'training_strategy_ablation.csv'}")
print(f"Saved to: {OUT / 'training_strategy_ablation.md'}")
