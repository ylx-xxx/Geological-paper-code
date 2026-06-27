from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("/root/autodl-tmp/q2_extra_ablation")
RUNS = BASE / "runs"
TABLES = BASE / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def read_metric(run_name):
    p = RUNS / run_name / "logs" / "test_metrics.csv"
    if not p.exists():
        print(f"[Warning] missing: {p}")
        return None

    df = pd.read_csv(p)
    if len(df) == 0:
        return None
    return df.iloc[-1].to_dict()


def metric_row(label, run_name, extra=None):
    m = read_metric(run_name)
    if m is None:
        return None

    row = {
        "Variant": label,
        "Run": run_name,
        "Test_mIoU": m.get("test_miou", np.nan),
        "Landslide_IoU": m.get("landslide_iou", np.nan),
        "Landslide_F1": m.get("landslide_f1", np.nan),
        "Precision": m.get("precision", np.nan),
        "Recall": m.get("recall", np.nan),
    }

    if extra:
        row.update(extra)

    return row


def save_table(rows, name):
    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows)

    csv_path = TABLES / f"{name}.csv"
    md_path = TABLES / f"{name}.md"

    df.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        cols = list(df.columns)
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for _, row in df.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if isinstance(v, float):
                    vals.append(f"{v:.6f}")
                else:
                    vals.append(str(v))
            f.write("| " + " | ".join(vals) + " |\n")

    print("=" * 100)
    print(name)
    print(df.to_string(index=False))
    print(f"Saved: {csv_path}")
    print("=" * 100)

    return df


def main():
    input_rows = [
        metric_row("RGB-only", "input_rgb", {"Input": "First 3 channels", "Channels": 3}),
        metric_row("12 MS only", "input_ms12", {"Input": "12 Sentinel-2 bands", "Channels": 12}),
        metric_row("RGB + topo", "input_rgb_topo", {"Input": "First 3 channels + slope + DEM", "Channels": 5}),
        metric_row("Full 14-channel", "input_full14", {"Input": "12 MS + slope + DEM", "Channels": 14}),
    ]
    input_df = save_table(input_rows, "input_modality_ablation")

    patch_rows = [
        metric_row("Random extra-channel init", "patch_random_extra", {"Extra_Channel_Init": "Random"}),
        metric_row("RGB-mean extra-channel init", "input_full14", {"Extra_Channel_Init": "Mean RGB kernel"}),
    ]
    patch_df = save_table(patch_rows, "patch_embedding_ablation")

    loss_rows = [
        metric_row("CE only", "loss_ce", {"Dice": "No", "Class_Weight": "No"}),
        metric_row("CE + Dice", "loss_ce_dice", {"Dice": "Yes", "Class_Weight": "No"}),
        metric_row("CE + Dice + class weight", "input_full14", {"Dice": "Yes", "Class_Weight": "Yes"}),
    ]
    loss_df = save_table(loss_rows, "loss_ablation")

    tta_rows = [
        metric_row("Without TTA", "tta_off", {"TTA": "No"}),
        metric_row("With TTA", "tta_on", {"TTA": "Yes"}),
    ]
    tta_df = save_table(tta_rows, "tta_ablation")

    seed_rows = [
        metric_row("Seed 2026", "seed_2026", {"Seed": 2026}),
        metric_row("Seed 2027", "seed_2027", {"Seed": 2027}),
        metric_row("Seed 2028", "seed_2028", {"Seed": 2028}),
    ]
    seed_df = save_table(seed_rows, "seed_stability_raw")

    if len(seed_df) > 0:
        metrics = ["Test_mIoU", "Landslide_IoU", "Landslide_F1", "Precision", "Recall"]
        mean_row = {"Variant": "Mean", "Run": "-", "Seed": "Mean"}
        std_row = {"Variant": "Std", "Run": "-", "Seed": "Std"}

        for m in metrics:
            mean_row[m] = seed_df[m].astype(float).mean()
            std_row[m] = seed_df[m].astype(float).std(ddof=1)

        seed_summary = pd.DataFrame([mean_row, std_row])
        seed_summary.to_csv(TABLES / "seed_stability_summary.csv", index=False)

        with open(TABLES / "seed_stability_summary.md", "w", encoding="utf-8") as f:
            cols = list(seed_summary.columns)
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
            for _, row in seed_summary.iterrows():
                vals = []
                for c in cols:
                    v = row[c]
                    if isinstance(v, float):
                        vals.append(f"{v:.6f}")
                    else:
                        vals.append(str(v))
                f.write("| " + " | ".join(vals) + " |\n")

        print("=" * 100)
        print("seed_stability_summary")
        print(seed_summary.to_string(index=False))
        print("=" * 100)

    all_rows = []
    for df in [input_df, patch_df, loss_df, tta_df, seed_df]:
        if df is not None and len(df) > 0:
            all_rows.append(df)

    if all_rows:
        all_df = pd.concat(all_rows, ignore_index=True)
        all_df.to_csv(TABLES / "all_extra_ablation_summary.csv", index=False)

    print("All extra ablation tables generated.")


if __name__ == "__main__":
    main()
