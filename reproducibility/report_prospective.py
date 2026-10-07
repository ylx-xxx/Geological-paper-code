"""Build review-only CSV tables and figures from frozen experiment outputs."""

import argparse, csv, hashlib, html, json, shutil
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from .core import confusion, metrics, sha256

BLUE = "#174A7E"
MID = "#397FAD"
LIGHT = "#93BDD8"
TEAL = "#188B83"
ORANGE = "#C87835"
GRAY = "#73808C"
METRICS = ["Landslide_IoU", "Landslide_F1", "Landslide_Precision", "Landslide_Recall"]
LABELS = ["Landslide IoU", "Landslide F1", "Precision", "Recall"]
ARM_LABELS = {
    "rgb_loveda": "RGB / LoveDA",
    "full14_loveda_mean": "14 channels / mean extras",
    "full14_loveda_random": "14 channels / random extras",
    "full14_scratch": "14 channels / no pretraining",
    "trainval_mean": "TrainVal / mean extras",
    "trainval_random": "TrainVal / random extras",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save(fig, folder, name):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            folder / f"{name}.{ext}", dpi=450, bbox_inches="tight", facecolor="white"
        )
    plt.close(fig)


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )


def spatial_figure(prepared, figures, tables):
    all_rows = read_json(prepared / "all_samples.json")
    summary = read_json(prepared / "summary.json")
    splits = {
        s: read_json(prepared / (s + "_manifest.json"))
        for s in ["train", "val", "test"]
    }
    for split, rows in splits.items():
        write_csv(
            tables / (split + "_manifest.csv"),
            [
                {
                    k: json.dumps(v) if isinstance(v, (dict, list)) else v
                    for k, v in r.items()
                }
                for r in rows
            ],
        )
    fig, axes = plt.subplots(1, 2, figsize=(7.7, 5.5), layout="constrained")
    chima = [r for r in all_rows if r["region"] == "chimanimani"]
    xy = lambda rs: np.array(
        [
            [(r["bbox"][0] + r["bbox"][2]) / 2000, (r["bbox"][1] + r["bbox"][3]) / 2000]
            for r in rs
        ]
    )
    a = xy(chima)
    axes[0].scatter(
        a[:, 0], a[:, 1], s=6, c="#DCE2E6", marker="s", label="Other available tiles"
    )
    for split, color, marker in [("train", BLUE, "o"), ("val", ORANGE, "D")]:
        a = xy(splits[split])
        axes[0].scatter(
            a[:, 0],
            a[:, 1],
            s=24,
            c=color,
            marker=marker,
            label={"train": "Training (40)", "val": "Validation (10)"}[split],
            edgecolors="white",
            linewidths=0.3,
        )
    boundary = summary["spatial_support"]["boundary_easting"] / 1000
    axes[0].axvspan(
        boundary - 0.64, boundary + 0.64, color="#B4C0C8", alpha=0.3, zorder=0
    )
    axes[0].set_title("(a) Chimanimani: adaptation")
    axes[0].set_xlabel("Easting (km; EPSG:32736)")
    axes[0].set_ylabel("Northing (km)")
    axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    dom = [r for r in all_rows if r["region"] == "dominicamaria"]
    for subset, color, marker, label in [
        (
            [
                r
                for r in dom
                if not r["excluded_reason"] and r["raw_positive_pixels"] > 0
            ],
            BLUE,
            "s",
            "Test: landslide annotations",
        ),
        (
            [
                r
                for r in dom
                if not r["excluded_reason"] and r["raw_positive_pixels"] == 0
            ],
            LIGHT,
            "s",
            "Test: zero-annotation tiles",
        ),
        (
            [r for r in dom if r["excluded_reason"]],
            GRAY,
            "x",
            "Excluded by image quality",
        ),
    ]:
        if subset:
            a = xy(subset)
            axes[1].scatter(
                a[:, 0],
                a[:, 1],
                s=24,
                c=color,
                marker=marker,
                label=label,
                linewidths=0.6,
            )
    axes[1].set_title("(b) Dominica Maria: holdout")
    axes[1].set_xlabel("Easting (km; EPSG:32620)")
    axes[1].set_ylabel("Northing (km)")
    axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    for ax in axes:
        ax.set_aspect("equal")
        ax.grid(alpha=0.18, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.ticklabel_format(useOffset=False)
    save(fig, figures, "01_spatial_protocol")
    quality = []
    for split, data in summary["splits"].items():
        quality.append(
            {
                "split": split,
                **data,
                "foreground_fraction": data["positive_pixels"] / data["valid_pixels"],
            }
        )
    write_csv(tables / "data_split_summary.csv", quality)
    temporal = []
    for split, rows in splits.items():
        days = [
            (
                datetime.fromisoformat(r["acquisition_date"])
                - datetime.fromisoformat(r["event_anchor"])
            ).total_seconds()
            / 86400
            for r in rows
        ]
        clear = [r["valid_fraction"] for r in rows]
        temporal.append(
            {
                "split": split,
                "samples": len(rows),
                "first_acquisition": min(r["acquisition_date"] for r in rows),
                "last_acquisition": max(r["acquisition_date"] for r in rows),
                "minimum_days_after_metadata_anchor": min(days),
                "median_days_after_metadata_anchor": float(np.median(days)),
                "maximum_days_after_metadata_anchor": max(days),
                "minimum_valid_pixel_fraction": min(clear),
                "median_valid_pixel_fraction": float(np.median(clear)),
                "interpretation": "Post-anchor image selection; metadata anchor is not a verified landfall date. Not an immediate-response latency estimate.",
            }
        )
    write_csv(tables / "temporal_quality_summary.csv", temporal)
    blocks = {}
    for row in splits["test"]:
        item = blocks.setdefault(
            row["block_id"], {"patches": 0, "valid_pixels": 0, "foreground_pixels": 0}
        )
        item["patches"] += 1
        item["valid_pixels"] += row["valid_pixels"]
        item["foreground_pixels"] += row["positive_pixels"]
    write_csv(
        tables / "external_spatial_blocks.csv",
        [{"block_id": name, **item} for name, item in sorted(blocks.items())],
    )
    return summary


def matched_tables(source, tables, figures):
    rows = []
    configs = []
    for folder in sorted((source / "matched_runs").glob("*")):
        metric_path = source / "matched_test" / folder.name / "metrics.json"
        if not metric_path.exists():
            continue
        cfg = read_json(folder / "config.json")
        summary = read_json(folder / "summary.json")
        scores = read_json(metric_path)
        if summary["status"] != "completed" or summary["completed_epochs"] != 60:
            raise AssertionError("Incomplete matched training")
        arm = folder.name.rsplit("_s", 1)[0]
        history = read_csv(folder / "history.csv")
        if cfg["protocol"] == "train_only":
            best = max(history, key=lambda r: float(r["val_landslide_iou"]))
            assert int(best["epoch"]) == summary["best_epoch"]
        else:
            assert summary["best_epoch"] == 60 and cfg["selection_split"] is None
        per_sample = read_csv(source / "matched_test" / folder.name / "per_sample.csv")
        pooled = np.array(
            [
                [
                    sum(int(r["TN"]) for r in per_sample),
                    sum(int(r["FP"]) for r in per_sample),
                ],
                [
                    sum(int(r["FN"]) for r in per_sample),
                    sum(int(r["TP"]) for r in per_sample),
                ],
            ]
        )
        assert np.array_equal(pooled, scores["confusion_matrix"])
        row = {
            "run": folder.name,
            "arm": arm,
            "seed": cfg["seed"],
            "protocol": cfg["protocol"],
            "training_samples": cfg["train_samples"],
            "selected_epoch": summary["best_epoch"],
            "validation_iou": summary["best_validation_iou"],
            **{k: scores[k] for k in METRICS + ["mIoU", "OA", "TP", "FP", "FN", "TN"]},
            "checkpoint_sha256": summary["checkpoint_sha256"],
        }
        rows.append(row)
        configs.append(
            {
                "run": folder.name,
                "seed": cfg["seed"],
                "training_samples": cfg["train_samples"],
                "validation_samples": cfg["val_samples"],
                "channels": json.dumps(cfg["channel_indices"]),
                "initialization": cfg["initialization"],
                "patch_init": cfg["patch_init"],
                "epochs": cfg["epochs"],
                "optimizer": "AdamW",
                "lr": cfg["lr"],
                "min_lr": cfg["min_lr"],
                "weight_decay": cfg["weight_decay"],
                "batch_size": cfg["batch_size"],
                "loss": cfg["loss"],
                "label_smoothing": cfg["label_smoothing"],
                "class_weights": json.dumps(cfg["class_weights"]),
                "weight_epsilon": 1e-12,
                "weight_clip_min": 0.5,
                "weight_clip_max": 2.5,
                "augmentation": "H/V flips, random rotations, Gaussian noise",
                "selection_split": cfg["selection_split"],
                "selection_metric": cfg["selection_metric"],
                "tta_views": cfg["val_tta_views"],
                "source_checkpoint_sha256": cfg["source_checkpoint_sha256"],
                "normalization": cfg["normalization"],
            }
        )
    write_csv(tables / "matched_per_run_metrics.csv", rows)
    write_csv(tables / "matched_configuration_matrix.csv", configs)
    grouped = []
    for arm in ARM_LABELS:
        subset = [r for r in rows if r["arm"] == arm]
        if len(subset) != 3:
            continue
        row = {
            "arm": arm,
            "label": ARM_LABELS[arm],
            "seeds": 3,
            "training_samples": subset[0]["training_samples"],
        }
        for key in METRICS:
            row[key + "_mean"] = float(np.mean([r[key] for r in subset]))
            row[key + "_sd"] = float(np.std([r[key] for r in subset], ddof=1))
        grouped.append(row)
    write_csv(tables / "matched_seed_summary.csv", grouped)
    if not grouped:
        return rows, grouped
    for subset_name, arms in [
        ("train_only", list(ARM_LABELS)[:4]),
        ("trainval", list(ARM_LABELS)[4:]),
    ]:
        data = [r for r in grouped if r["arm"] in arms]
        if len(data) != len(arms):
            continue
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.1), layout="constrained")
        for ax, key, title in zip(axes, METRICS[:2], LABELS[:2]):
            for i, item in enumerate(data):
                values = [r[key] for r in rows if r["arm"] == item["arm"]]
                ax.errorbar(
                    i,
                    item[key + "_mean"],
                    yerr=item[key + "_sd"],
                    fmt="D",
                    color=BLUE,
                    capsize=5,
                    elinewidth=2,
                    markersize=6,
                    zorder=3,
                )
                ax.scatter(
                    np.array([i - 0.13, i, i + 0.13]),
                    values,
                    s=26,
                    c=[LIGHT, MID, BLUE],
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=4,
                )
                ax.annotate(
                    f"{item[key+'_mean']:.4f}",
                    (i, item[key + "_mean"] + item[key + "_sd"]),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
            ax.set_xticks(
                range(len(data)),
                [ARM_LABELS[r["arm"]].replace(" / ", "\n") for r in data],
                fontsize=8,
            )
            ax.set_ylabel(title)
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.18)
            ax.set_title(title + " (mean ± SD; three seeds)")
        fig.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    color=color,
                    marker="o",
                    linestyle="none",
                    label=f"Seed {seed}",
                )
                for seed, color in zip([2026, 2027, 2028], [LIGHT, MID, BLUE])
            ]
            + [
                Line2D(
                    [],
                    [],
                    color=BLUE,
                    marker="D",
                    linestyle="none",
                    label="Mean; whiskers = SD",
                )
            ],
            loc="outside lower center",
            ncol=4,
            fontsize=8,
        )
        save(fig, figures, "02_matched_" + subset_name)
    return rows, grouped


def external_figures(source, prepared, tables, figures):
    external = source / "external_experiment"
    if not (external / "complete.json").exists():
        return [], []
    rows = read_csv(external / "external_metrics.csv")
    all_samples = read_csv(external / "external_per_sample.csv")
    write_csv(tables / "external_per_model_metrics.csv", rows)
    write_csv(tables / "external_per_sample.csv", all_samples)
    background = []
    for row in rows:
        samples = [
            r
            for r in all_samples
            if r["model"] == row["model"]
            and r["raw_background_patch"].lower() == "true"
        ]
        fp = sum(int(r["FP"]) for r in samples)
        tn = sum(int(r["TN"]) for r in samples)
        background.append(
            {
                "model": row["model"],
                "seed": row["seed"],
                "stage": row["stage"],
                "background_patches": len(samples),
                "false_positive_pixels": fp,
                "valid_background_pixels": fp + tn,
                "background_pixel_false_positive_rate": fp / max(fp + tn, 1),
                "patches_with_any_false_positive": sum(
                    int(r["FP"]) > 0 for r in samples
                ),
            }
        )
    write_csv(tables / "external_background_errors.csv", background)
    means = []
    for stage in ["zero_shot", "few_shot"]:
        subset = [r for r in rows if r["stage"] == stage]
        assert len(subset) == 3
        item = {"stage": stage, "seeds": 3, "test_samples": int(subset[0]["samples"])}
        for key in METRICS:
            values = [float(r[key]) for r in subset]
            item[key + "_mean"] = float(np.mean(values))
            item[key + "_sd"] = float(np.std(values, ddof=1))
        means.append(item)
    write_csv(tables / "external_seed_summary.csv", means)
    paired = []
    for seed in [2026, 2027, 2028]:
        stages = {
            stage: next(
                r for r in rows if int(r["seed"]) == seed and r["stage"] == stage
            )
            for stage in ["zero_shot", "few_shot"]
        }
        paired.append(
            {
                "seed": seed,
                **{
                    key
                    + "_delta": float(stages["few_shot"][key])
                    - float(stages["zero_shot"][key])
                    for key in METRICS
                },
            }
        )
    write_csv(tables / "external_paired_seed_differences.csv", paired)
    write_csv(
        tables / "external_paired_difference_summary.csv",
        [
            {
                "metric": key,
                "mean_delta": float(np.mean([r[key + "_delta"] for r in paired])),
                "sd_delta": float(np.std([r[key + "_delta"] for r in paired], ddof=1)),
                "interpretation": "paired differences across three seeds on the same event; not event-level uncertainty",
            }
            for key in METRICS
        ],
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), layout="constrained")
    axes = axes.ravel()
    for ax, key, label in zip(axes, METRICS, LABELS):
        for i, seed in enumerate([2026, 2027, 2028]):
            values = [
                float(
                    next(
                        r
                        for r in rows
                        if int(r["seed"]) == seed and r["stage"] == stage
                    )[key]
                )
                for stage in ["zero_shot", "few_shot"]
            ]
            ax.plot(
                [0, 1],
                values,
                "o-",
                color=[LIGHT, MID, BLUE][i],
                lw=1.6,
                ms=5,
                label=f"Seed {seed}",
            )
        ax.set_xticks([0, 1], ["L4S RGB", "+ Chimanimani"], fontsize=8)
        ax.set_xlim(-0.25, 1.25)
        # Keep a zero origin and label the absolute scale; these scores are low.
        upper = np.ceil(max(float(row[key]) for row in rows) * 1.3 / 0.02) * 0.02
        ax.set_ylim(0, max(upper, 0.02))
        before, after = [item[key + "_mean"] for item in means]
        ax.set_title(f"{label}\nMean: {before:.4f} → {after:.4f}")
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("Absolute score")
    axes[2].set_ylabel("Absolute score")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    save(fig, figures, "03_external_transfer")
    histories = [
        read_csv(external / f"few_shot_s{seed}" / "history.csv")
        for seed in [2026, 2027, 2028]
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.0), layout="constrained")
    for ax, key, label, color in zip(
        axes,
        ["train_loss", "val_Landslide_IoU", "val_Landslide_F1"],
        ["Training loss", "Validation landslide IoU", "Validation landslide F1"],
        [ORANGE, BLUE, TEAL],
    ):
        values = np.array([[float(row[key]) for row in h] for h in histories])
        epochs = np.arange(1, values.shape[1] + 1)
        for v in values:
            ax.plot(epochs, v, color=color, alpha=0.32, lw=0.9)
        mean = values.mean(0)
        sd = values.std(0, ddof=1)
        ax.plot(epochs, mean, color=color, lw=2, label="Mean of three seeds")
        ax.fill_between(
            epochs, mean - sd, mean + sd, color=color, alpha=0.13, label="±1 SD"
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.18)
        if key != "train_loss":
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8)
    save(fig, figures, "04_adaptation_validation_curves")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), layout="constrained")
    for ax, stage, title in zip(
        axes, ["zero_shot", "few_shot"], ["(a) Zero-shot", "(b) Fine-tuned"]
    ):
        score = read_json(external / f"{stage}_s2026" / "metrics.json")
        cm = np.array(score["confusion_matrix"])
        proportions = cm / cm.sum(1, keepdims=True)
        ax.imshow(proportions, cmap="Blues", vmin=0, vmax=1)
        for (i, j), value in np.ndenumerate(proportions):
            ax.text(
                j,
                i,
                f"{100*value:.2f}%",
                ha="center",
                va="center",
                color="white" if value > 0.5 else BLUE,
                fontsize=11,
            )
        ax.set_xticks([0, 1], ["Background", "Landslide"])
        ax.set_yticks([0, 1], ["Background", "Landslide"])
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Reference")
        ax.set_title(title)
    save(fig, figures, "05_external_confusion_matrices")
    return rows, means


def qualitative_figures(source, prepared, tables, figures):
    external = source / "external_experiment"
    if not (external / "qualitative_inputs.npz").exists():
        return []
    with np.load(external / "qualitative_inputs.npz") as data:
        names = data["names"]
        rgbs = data["rgb"]
        truth = data["y"]
    predictions = {}
    records = {}
    for stage in ["zero_shot", "few_shot"]:
        run = external / f"{stage}_s2026"
        with np.load(run / "predictions.npz") as data:
            predictions[stage] = {
                str(n): p.copy() for n, p in zip(data["names"], data["masks"])
            }
        records[stage] = {r["sample"]: r for r in read_csv(run / "per_sample.csv")}
    audit = []
    for page, start in enumerate(range(0, len(names), 4), 1):
        count = min(4, len(names) - start)
        fig, axes = plt.subplots(
            count, 4, figsize=(7.5, 2.0 * count), squeeze=False, layout="constrained"
        )
        for row, index in enumerate(range(start, start + count)):
            name = str(names[index])
            rgb = rgbs[index].transpose(1, 2, 0)
            low, high = np.percentile(rgb, [2, 98])
            display = np.clip((rgb - low) / max(high - low, 1), 0, 1)
            y = truth[index]
            valid = y != 255
            gt = np.ones((*y.shape, 3))
            gt[y == 1] = matplotlib.colors.to_rgb(BLUE)
            gt[~valid] = (0.7, 0.7, 0.7)
            axes[row, 0].imshow(display, interpolation="nearest")
            axes[row, 1].imshow(gt, interpolation="nearest")
            axes[row, 0].set_ylabel(
                name.replace("dominicamaria_s2_", "DOM-").replace(".nc", ""), fontsize=9
            )
            for column, stage in enumerate(["zero_shot", "few_shot"], 2):
                p = predictions[stage][name]
                counts = metrics(confusion(p, y))
                stored = records[stage][name]
                for key in ["TP", "FP", "FN", "TN"]:
                    assert counts[key] == int(stored[key]), (name, stage, key)
                image = np.full((*y.shape, 3), 0.96)
                image[(p == 1) & (y == 1)] = matplotlib.colors.to_rgb("#23966F")
                image[(p == 1) & (y == 0)] = matplotlib.colors.to_rgb("#D54A45")
                image[(p == 0) & (y == 1)] = matplotlib.colors.to_rgb("#276CC0")
                image[~valid] = (0.7, 0.7, 0.7)
                axes[row, column].imshow(image, interpolation="nearest")
                if (y == 1).any():
                    label = f"IoU = {counts['Landslide_IoU']:.4f}"
                else:
                    label = f"Background FPR = {100*counts['FP']/max(counts['FP']+counts['TN'],1):.2f}%"
                axes[row, column].set_xlabel(label, fontsize=8)
                audit.append(
                    {
                        "sample": name,
                        "stage": stage,
                        "seed": 2026,
                        **counts,
                        "pixel_counts_match_saved_csv": True,
                    }
                )
            for ax in axes[row]:
                ax.set_xticks([])
                ax.set_yticks([])
        for ax, title in zip(
            axes[0],
            [
                "Sentinel-2 RGB",
                "Reference mask",
                "Zero-shot errors",
                "Fine-tuned errors",
            ],
        ):
            ax.set_title(title)
        fig.legend(
            handles=[
                Patch(color="#23966F", label="True positive"),
                Patch(color="#D54A45", label="False positive"),
                Patch(color="#276CC0", label="False negative"),
                Patch(color=".7", label="Ignored quality pixels"),
            ],
            loc="outside lower center",
            ncol=4,
            fontsize=8,
        )
        save(fig, figures, f"06_external_qualitative_{page}")
    write_csv(tables / "visualization_pixel_audit.csv", audit)
    return audit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    source = Path(args.source)
    prepared = Path(args.prepared)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    tables = out / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    style()
    split_summary = spatial_figure(prepared, figures, tables)
    matched, matched_summary = matched_tables(source, tables, figures)
    external, external_summary = external_figures(source, prepared, tables, figures)
    visual_audit = qualitative_figures(source, prepared, tables, figures)
    validation = None
    if len(matched) == 18 and len(external) == 6:
        from .verify_prospective import verify

        validation = verify(source, prepared)
        (out / "experiment_integrity_checks.json").write_text(
            json.dumps(validation, indent=2), encoding="utf8"
        )
    audit = {
        "matched_runs": len(matched),
        "external_models": len(external),
        "visualization_count_checks": len(visual_audit),
        "all_available_counts_reconciled": True,
        "quantitative_results_verified": validation is not None,
        "completed": validation is not None and len(visual_audit) == 16,
        "manuscript_modified": False,
        "split_summary": split_summary,
        "matched_seed_summary": matched_summary,
        "external_seed_summary": external_summary,
    }
    (out / "review_validation.json").write_text(
        json.dumps(audit, indent=2), encoding="utf8"
    )
    titles = {
        "01_spatial_protocol": "地区划分与空间间隔",
        "02_matched_train_only": "统一配方：TrainData-only 对照",
        "02_matched_trainval": "固定 60 轮：TrainData + ValidData 对照",
        "03_external_transfer": "独立地区：逐随机种子迁移结果",
        "04_adaptation_validation_curves": "少样本训练与验证曲线",
        "05_external_confusion_matrices": "外部测试混淆矩阵（seed 2026）",
        "06_external_qualitative_1": "外部预测与错误图：预先固定样本 1",
        "06_external_qualitative_2": "外部预测与错误图：预先固定样本 2",
    }
    cards = []
    for image in sorted(figures.glob("*.png")):
        title = titles.get(image.stem, image.stem)
        note = (
            "<p>微调使用 Chimanimani 的 40 个训练和 10 个验证样本。两组模型均未使用 Dominica 标签进行训练或选模。</p>"
            if image.stem.startswith(("03_", "04_", "05_", "06_"))
            else ""
        )
        cards.append(
            f'<section><h2>{html.escape(title)}</h2><img src="figures/{image.name}">{note}<p><a href="figures/{image.stem}.pdf">PDF</a> · <a href="figures/{image.stem}.svg">SVG</a></p></section>'
        )
    links = "".join(
        f'<li><a href="tables/{p.name}">{p.name}</a></li>'
        for p in sorted(tables.glob("*.csv"))
    )
    status = (
        "实验结果与图片已完成核验，等待作者确认。"
        if audit["completed"]
        else (
            "定量结果已完成核验；预测文件未齐，尚未重建全部定性图。"
            if validation is not None
            else "实验仍在运行；当前页面仅展示已有的核验材料。"
        )
    )
    document = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>外部验证与补实验验收</title><style>
body{max-width:1180px;margin:36px auto;padding:0 24px;font:16px/1.65 Arial,"Microsoft YaHei",sans-serif;color:#183449;background:#f3f6f8}
header,section{background:white;border-radius:12px;padding:24px 30px;margin:20px 0;border:1px solid #dfe7ed}img{width:100%;height:auto}h1{font-size:28px}h2{font-size:21px}a{color:#145e91}li{margin:5px 0}.note{border-left:4px solid #397fad;padding-left:15px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px 12px;border-bottom:1px solid #dfe7ed;text-align:left}th{background:#174a7e;color:white}.scroll{overflow-x:auto}</style>"""
    document += f'<header><h1>外部验证与补实验验收</h1><p>{status}</p><p class="note">论文正文、LaTeX 与原配图保持原样。新图片均由 CSV / 同一评估的预测结果生成；正式替换需作者确认。</p><p>训练与验证：Chimanimani 40 / 10；外部测试：Dominica Maria 204，包含 111 个零滑坡标注样本。外部模型为物理 RGB 分支；本轮地区验证不等同于最终 14 通道 V8 的直接外部复测。</p></header>'
    for title, rows, label_key in [
        ("新基准对照结果", matched_summary, "label"),
        ("Dominica 地区迁移结果", external_summary, "stage"),
    ]:
        if not rows:
            continue
        headings = "<th>设置</th>" + "".join(f"<th>{label}</th>" for label in LABELS)
        body = ""
        for row in rows:
            name = {
                "zero_shot": "L4S RGB 直接迁移",
                "few_shot": "经 Chimanimani 微调",
            }.get(row[label_key], row[label_key])
            cells = "".join(
                f'<td>{row[key+"_mean"]:.4f} ± {row[key+"_sd"]:.4f}</td>'
                for key in METRICS
            )
            body += f"<tr><td>{html.escape(name)}</td>{cells}</tr>"
        document += f'<section><h2>{title}</h2><p>三个随机种子的均值 ± 样本标准差。各次运行的完整结果保存在 CSV 中。</p><div class="scroll"><table><thead><tr>{headings}</tr></thead><tbody>{body}</tbody></table></div></section>'
    if audit["completed"]:
        before, after = [row["Landslide_IoU_mean"] for row in external_summary]
        document += f'<section><h2>验收结论</h2><p>外部 IoU 均值从 {before:.4f} 提升到 {after:.4f}，三个种子均有提升，但绝对精度仍然很低，不能支持较强跨地区泛化的结论。新对照支持 LoveDA 初始化优于从头训练；14 通道没有普遍优于物理 RGB。</p><p><a href="验收报告.md">完整验收报告</a> · <a href="实验结果与核验表.xlsx">Excel 汇总表</a></p></section>'
    document += "<section><h2>结果的适用范围</h2><ul><li>新基准对照共 18 组，所有模型冻结后才统一评估；历史 TestData 使用记录仍需披露。</li><li>外部验证只覆盖一个事件及指定压缩包中的地区范围，不能代表完整 Sen12 基准或多个独立事件。</li><li>三种子标准差描述优化随机性；空间区块区间描述同一事件内的空间变化，两者不可混用。</li><li>跨数据集内容哈希未发现重复图像；该检查不能排除不同处理版本或相邻、部分重叠范围。</li></ul></section>"
    document += (
        "".join(cards)
        + f"<section><h2>CSV 表格与逐样本记录</h2><ul>{links}</ul></section></html>"
    )
    (out / "review.html").write_text(document, encoding="utf8")
    print(
        json.dumps(
            {
                k: audit[k]
                for k in [
                    "matched_runs",
                    "external_models",
                    "visualization_count_checks",
                    "completed",
                ]
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
