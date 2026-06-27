from pathlib import Path
import pandas as pd

out_dir = Path("/root/autodl-tmp/q2_experiments/tables")
out_dir.mkdir(parents=True, exist_ok=True)


def to_markdown_simple(df):
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

    return "\n".join(lines)


rows = [
    {
        "Comparison": "Swin + LoveDA vs Swin without LoveDA",
        "Baseline_Landslide_IoU": 0.3916,
        "Improved_Landslide_IoU": 0.4426,
        "Absolute_Gain": 0.4426 - 0.3916,
        "Relative_Gain_%": (0.4426 - 0.3916) / 0.3916 * 100,
        "Interpretation": "Effect of LoveDA source-domain pretraining"
    },
    {
        "Comparison": "Final V8 vs Swin + LoveDA",
        "Baseline_Landslide_IoU": 0.4426,
        "Improved_Landslide_IoU": 0.4793,
        "Absolute_Gain": 0.4793 - 0.4426,
        "Relative_Gain_%": (0.4793 - 0.4426) / 0.4426 * 100,
        "Interpretation": "Effect of final TrainVal training"
    },
    {
        "Comparison": "Final V8 vs U-Net",
        "Baseline_Landslide_IoU": 0.4460,
        "Improved_Landslide_IoU": 0.4793,
        "Absolute_Gain": 0.4793 - 0.4460,
        "Relative_Gain_%": (0.4793 - 0.4460) / 0.4460 * 100,
        "Interpretation": "Improvement over classical CNN baseline"
    },
    {
        "Comparison": "Final V8 vs DeepLabV3",
        "Baseline_Landslide_IoU": 0.4151,
        "Improved_Landslide_IoU": 0.4793,
        "Absolute_Gain": 0.4793 - 0.4151,
        "Relative_Gain_%": (0.4793 - 0.4151) / 0.4151 * 100,
        "Interpretation": "Improvement over dilated CNN baseline"
    }
]

df = pd.DataFrame(rows)

for c in [
    "Baseline_Landslide_IoU",
    "Improved_Landslide_IoU",
    "Absolute_Gain",
    "Relative_Gain_%"
]:
    df[c] = pd.to_numeric(df[c], errors="coerce").round(4)

df.to_csv(out_dir / "improvement_summary.csv", index=False)

with open(out_dir / "improvement_summary.md", "w", encoding="utf-8") as f:
    f.write(to_markdown_simple(df))

print("=" * 120)
print("Improvement summary:")
print(df.to_string(index=False))
print("=" * 120)
print(f"Saved to: {out_dir / 'improvement_summary.csv'}")
print(f"Saved to: {out_dir / 'improvement_summary.md'}")
