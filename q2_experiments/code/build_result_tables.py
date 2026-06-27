from pathlib import Path
import pandas as pd


ROOT = Path("/root/autodl-tmp/q2_experiments")
OUT = ROOT / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def normalize_row(row):
    """
    兼容不同 test.py 保存出来的字段名：
    - Test_mIoU / mIoU
    - Test_OA / OA
    - Landslide_F1 / Landslide_F1/Dice
    """
    clean = {}
    for k, v in row.items():
        nk = str(k).strip()
        clean[nk] = v
    return clean


def pick(row, *keys, default=None):
    for k in keys:
        if k in row:
            return row[k]
    return default


def read_one(path, method, train_setting, pretraining, note):
    path = Path(path)

    if not path.exists():
        print(f"[Warning] Missing CSV: {path}")
        return None

    df = pd.read_csv(path)
    if len(df) == 0:
        print(f"[Warning] Empty CSV: {path}")
        return None

    row = normalize_row(df.iloc[-1].to_dict())

    return {
        "Method": method,
        "Train_Setting": train_setting,
        "Pretraining": pretraining,

        "Test_mIoU": pick(row, "Test_mIoU", "mIoU"),
        "Test_OA": pick(row, "Test_OA", "OA"),
        "NonLandslide_IoU": pick(row, "NonLandslide_IoU"),
        "Landslide_IoU": pick(row, "Landslide_IoU"),
        "NonLandslide_F1": pick(row, "NonLandslide_F1"),
        "Landslide_F1": pick(row, "Landslide_F1", "Landslide_F1/Dice"),
        "Landslide_Precision": pick(row, "Landslide_Precision"),
        "Landslide_Recall": pick(row, "Landslide_Recall"),

        "TN": pick(row, "TN"),
        "FP": pick(row, "FP"),
        "FN": pick(row, "FN"),
        "TP": pick(row, "TP"),

        "Note": note,
        "Source_CSV": str(path),
    }


def to_markdown_simple(df):
    cols = list(df.columns)
    lines = []

    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if pd.isna(v):
                vals.append("")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def main():
    items = [
        (
            ROOT / "runs/unet_14ch/logs/test_metrics.csv",
            "U-Net",
            "TrainData only",
            "None",
            "Classical CNN segmentation baseline"
        ),
        (
            ROOT / "runs/deeplabv3_14ch/logs/test_metrics.csv",
            "DeepLabV3",
            "TrainData only",
            "None",
            "Dilated CNN segmentation baseline"
        ),
        (
            ROOT / "runs/swin_no_loveda/logs/test_metrics.csv",
            "Swin-UPerNet",
            "TrainData only",
            "No LoveDA",
            "Ablation for source-domain pretraining"
        ),
        (
            ROOT / "runs/swin_loveda/logs/test_metrics.csv",
            "Swin-UPerNet",
            "TrainData only",
            "LoveDA",
            "Source-domain pretrained Swin-UPerNet"
        ),
        (
            ROOT / "runs/final_v8_trainval/logs/test_metrics.csv",
            "Ours Final",
            "TrainData + ValidData",
            "LoveDA",
            "Final selected model"
        ),
    ]

    rows = []

    for item in items:
        r = read_one(*item)
        if r is not None:
            rows.append(r)

    if len(rows) == 0:
        raise RuntimeError("No valid result CSV files were found.")

    df = pd.DataFrame(rows)

    metric_cols = [
        "Test_mIoU",
        "Test_OA",
        "NonLandslide_IoU",
        "Landslide_IoU",
        "NonLandslide_F1",
        "Landslide_F1",
        "Landslide_Precision",
        "Landslide_Recall",
    ]

    for c in metric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(4)

    # 论文主表：只保留核心字段
    paper_cols = [
        "Method",
        "Train_Setting",
        "Pretraining",
        "Test_mIoU",
        "Landslide_IoU",
        "Landslide_F1",
        "Landslide_Precision",
        "Landslide_Recall",
        "Note",
    ]

    paper_df = df[paper_cols].copy()

    # 完整表：包含混淆矩阵和 CSV 来源
    df.to_csv(OUT / "main_comparison_table_full.csv", index=False)
    paper_df.to_csv(OUT / "main_comparison_table.csv", index=False)

    with open(OUT / "main_comparison_table.md", "w", encoding="utf-8") as f:
        f.write(to_markdown_simple(paper_df))

    print("=" * 120)
    print("Main comparison table:")
    print(paper_df.to_string(index=False))
    print("=" * 120)
    print(f"Saved paper table to: {OUT / 'main_comparison_table.csv'}")
    print(f"Saved full table to:  {OUT / 'main_comparison_table_full.csv'}")
    print(f"Saved markdown to:    {OUT / 'main_comparison_table.md'}")


if __name__ == "__main__":
    main()
