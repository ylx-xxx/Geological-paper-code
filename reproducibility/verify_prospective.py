"""Check saved experiment identities, selection records, and pixel counts."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .core import confusion, metrics, sha256, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_counts(rows, summary):
    """Recompute global scores from per-sample integer counts."""
    if len({row["sample"] for row in rows}) != len(rows):
        raise ValueError("Duplicate sample IDs in evaluation")
    counts = {
        key: sum(int(row[key]) for row in rows) for key in ["TN", "FP", "FN", "TP"]
    }
    matrix = np.array([[counts["TN"], counts["FP"]], [counts["FN"], counts["TP"]]])
    if not np.array_equal(matrix, summary["confusion_matrix"]):
        raise ValueError("Per-sample and aggregate counts disagree")
    for key, value in metrics(matrix).items():
        if key in summary and not np.isclose(
            value, float(summary[key]), rtol=1e-10, atol=1e-10
        ):
            raise ValueError(f"Recomputed metric differs: {key}")
    return counts


def verify_prediction_files(source, prepared):
    """Read saved masks back from disk and compare every count with the CSV."""
    source, prepared = Path(source), Path(prepared)
    external = source / "external_experiment"
    manifest = read_json(prepared / "test_manifest.json")
    truth = {}
    for row in manifest:
        path = prepared / "samples" / row["prepared_file"]
        if sha256(path) != row["prepared_sha256"]:
            raise ValueError("Prepared sample identity changed before prediction audit")
        with np.load(path, allow_pickle=False) as data:
            truth[row["sample"]] = data["y"].copy()
    specs = read_json(external / "checkpoints_frozen_before_test.json")["models"]
    checked = []
    for spec in specs:
        run = external / spec["name"]
        records = {row["sample"]: row for row in read_csv(run / "per_sample.csv")}
        path = run / "predictions.npz"
        with np.load(path, allow_pickle=False) as data:
            names = data["names"].astype(str).tolist()
            masks = data["masks"]
            if (
                len(names) != len(set(names))
                or set(names) != set(truth)
                or len(masks) != len(names)
            ):
                raise ValueError("Prediction archive IDs differ from the test manifest")
            for name, prediction in zip(names, masks):
                counts = metrics(confusion(prediction, truth[name]))
                for key in ["TN", "FP", "FN", "TP"]:
                    if counts[key] != int(records[name][key]):
                        raise ValueError(
                            f"Saved mask disagrees with CSV: {spec['name']}, {name}, {key}"
                        )
        checked.append(
            {
                "model": spec["name"],
                "maps": len(names),
                "predictions_sha256": sha256(path),
            }
        )
    return {
        "status": "passed",
        "checked_prediction_maps": sum(row["maps"] for row in checked),
        "models": checked,
    }


def verify(source, prepared):
    source, prepared = Path(source), Path(prepared)
    complete = read_json(source / "matched_complete.json")
    if complete["status"] != "complete" or complete["jobs"] != 18:
        raise ValueError("Expected 18 completed matched runs")
    plan = read_json(source / "matched_plan.json")
    frozen = read_json(source / "matched_checkpoints_frozen.json")
    if {r["name"] for r in frozen["models"]} != {r["name"] for r in plan["jobs"]}:
        raise ValueError("Evaluated arms differ from frozen plan")
    checked = []
    for spec in frozen["models"]:
        run = source / "matched_runs" / spec["name"]
        test = source / "matched_test" / spec["name"]
        cfg, summary = read_json(run / "config.json"), read_json(run / "summary.json")
        protocol, result = read_json(test / "protocol.json"), read_json(
            test / "metrics.json"
        )
        history = read_csv(run / "history.csv")
        if [int(row["epoch"]) for row in history] != list(range(1, 61)):
            raise ValueError("Training history has missing or duplicated epochs")
        if cfg["seed"] != spec["seed"] or cfg["protocol"] != spec["protocol"]:
            raise ValueError("Training configuration differs from plan")
        train_manifest = read_json(run / "train_manifest.json")
        expected_training_samples = 3799 if cfg["protocol"] == "train_only" else 4044
        if (
            len(train_manifest) != expected_training_samples
            or cfg["train_samples"] != expected_training_samples
        ):
            raise ValueError("Training manifest does not match the declared protocol")
        if len({row["source_image"] for row in train_manifest}) != len(train_manifest):
            raise ValueError("Duplicate source paths in the training manifest")
        if cfg["protocol"] == "train_only":
            val_manifest = read_json(run / "val_manifest.json")
            if len(val_manifest) != 245 or cfg["val_samples"] != 245:
                raise ValueError(
                    "Validation sample count differs from the native split"
                )
            if {row["source_image"] for row in train_manifest} & {
                row["source_image"] for row in val_manifest
            }:
                raise ValueError(
                    "A source file appears in both training and validation"
                )
        if summary["checkpoint_sha256"] != spec["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint differs from frozen identity")
        if protocol["checkpoint_sha256"] != spec["checkpoint_sha256"]:
            raise ValueError("Evaluation used a different checkpoint")
        if cfg["protocol"] == "train_only":
            selected = max(history, key=lambda row: float(row["val_landslide_iou"]))
            if (
                int(selected["epoch"]) != summary["best_epoch"]
                or cfg["selection_split"] != "val"
            ):
                raise ValueError(
                    "Checkpoint does not match earliest validation maximum"
                )
        elif summary["best_epoch"] != 60 or cfg["selection_split"] is not None:
            raise ValueError("TrainVal refit did not use the fixed final epoch")
        rows = read_csv(test / "per_sample.csv")
        if len(rows) != 800:
            raise ValueError("Unexpected benchmark sample count")
        checked.append(
            {"run": spec["name"], "samples": len(rows), **verify_counts(rows, result)}
        )

    external = source / "external_experiment"
    completion = read_json(external / "complete.json")
    data_summary = read_json(prepared / "summary.json")
    training_plan = read_json(external / "training_plan.json")
    if sha256(prepared / "protocol.json") != training_plan["data_protocol_sha256"]:
        raise ValueError("External data protocol identity differs")
    manifests = {}
    for split in ["train", "val", "test"]:
        path = prepared / f"{split}_manifest.json"
        expected = training_plan["manifest_sha256"][split]
        if (
            sha256(path) != expected
            or expected != data_summary["splits"][split]["manifest_sha256"]
        ):
            raise ValueError("External manifest identity differs")
        manifests[split] = {row["sample"]: row for row in read_json(path)}
    if any(
        row["region"] != "chimanimani"
        for split in ["train", "val"]
        for row in manifests[split].values()
    ):
        raise ValueError("Held-out region appears in adaptation data")
    if any(row["region"] != "dominicamaria" for row in manifests["test"].values()):
        raise ValueError("Unexpected test region")
    identities = read_json(external / "checkpoints_frozen_before_test.json")
    opened = datetime.fromisoformat(
        (external / "TEST_EVALUATION_STARTED").read_text().strip()
    )
    if datetime.fromisoformat(identities["frozen_utc"]) > opened:
        raise ValueError("Test opened before checkpoint freeze")
    for spec in identities["models"]:
        run = external / spec["name"]
        result, rows = read_json(run / "metrics.json"), read_csv(run / "per_sample.csv")
        if result["checkpoint_sha256"] != spec["sha256"]:
            raise ValueError("External checkpoint identity differs")
        if {row["sample"] for row in rows} != set(manifests["test"]):
            raise ValueError("External evaluation does not cover the frozen manifest")
        for row in rows:
            meta = manifests["test"][row["sample"]]
            if (
                sum(int(row[key]) for key in ["TN", "FP", "FN", "TP"])
                != meta["valid_pixels"]
            ):
                raise ValueError("Sample valid-pixel count differs from prepared data")
            if int(row["TP"]) + int(row["FN"]) != meta["positive_pixels"]:
                raise ValueError("Sample foreground count differs from prepared data")
        if spec["stage"] == "few_shot":
            history, summary = read_csv(run / "history.csv"), read_json(
                run / "summary.json"
            )
            if [int(row["epoch"]) for row in history] != list(range(1, 41)):
                raise ValueError("Incomplete external fine-tuning history")
            selected = max(history, key=lambda row: float(row["val_Landslide_IoU"]))
            if int(selected["epoch"]) != summary["best_epoch"]:
                raise ValueError(
                    "Fine-tuning selection differs from validation maximum"
                )
        checked.append(
            {"run": spec["name"], "samples": len(rows), **verify_counts(rows, result)}
        )
    if (
        len(identities["models"]) != 6
        or completion["cumulative_training_seconds"] > 14400
    ):
        raise ValueError("Unexpected model count or exceeded training budget")
    return {
        "status": "passed",
        "checked_evaluations": len(checked),
        "checks": checked,
        "cumulative_training_seconds": completion["cumulative_training_seconds"],
        "scope": "Saved protocol, identity, checkpoint selection, and integer-count consistency. Does not establish unobserved historical use or geographic nonoverlap.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--check-predictions",
        action="store_true",
        help="Also requires all prepared sample arrays and saved prediction archives",
    )
    args = parser.parse_args()
    result = verify(args.source, args.prepared)
    if args.check_predictions:
        result["saved_prediction_check"] = verify_prediction_files(
            args.source, args.prepared
        )
    write_json(args.out, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in [
                    "status",
                    "checked_evaluations",
                    "cumulative_training_seconds",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
