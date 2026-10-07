"""Train on 40 Chimanimani patches, select on 10, then open a frozen regional test once."""

import argparse, csv, json, os, random, time, importlib.metadata
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from train_l4s_qz.model import SwinUPerNetL4S
from train_l4s_qz.losses import CombinedLoss
from .core import sha256, write_json, confusion, metrics
from .evaluate import predict
from .sen12_protocol import assert_training_regions, bootstrap_blocks


class PreparedDataset(Dataset):
    def __init__(self, root, split, train=False, allow_test=False):
        if split == "test" and not allow_test:
            raise ValueError("External test access is sealed during development")
        if train and split != "train":
            raise ValueError("Only the training manifest can provide gradient data")
        self.rows = json.loads((Path(root) / (split + "_manifest.json")).read_text())
        if split in {"train", "val"}:
            assert_training_regions(self.rows)
        self.augment = train
        self.cache = []
        for row in self.rows:
            path = Path(root) / "samples" / row["prepared_file"]
            if sha256(path) != row["prepared_sha256"]:
                raise ValueError("Prepared input hash changed")
            with np.load(path, allow_pickle=False) as z:
                x, y = z["x"], z["y"].astype(np.int64)
                if (
                    x.shape != (3, 128, 128)
                    or y.shape != (128, 128)
                    or not np.isin(y, [0, 1, 255]).all()
                ):
                    raise ValueError("Invalid prepared tensor")
                self.cache.append((x, y))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        x, y = self.cache[index]
        if self.augment:
            if random.random() < 0.5:
                x = x[:, :, ::-1]
                y = y[:, ::-1]
            if random.random() < 0.5:
                x = x[:, ::-1, :]
                y = y[::-1, :]
            if random.random() < 0.5:
                k = random.choice([1, 2, 3])
                x = np.rot90(x, k, (1, 2))
                y = np.rot90(y, k)
            if random.random() < 0.3:
                x = x + np.random.normal(0, 0.02, x.shape).astype(np.float32)
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy()), index


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(path, expected, device):
    if sha256(path) != expected:
        raise ValueError("Frozen checkpoint identity changed")
    model = SwinUPerNetL4S(in_chans=3, img_size=128)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device, memory_format=torch.channels_last)


def csv_write(path, rows):
    with Path(path).open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--matched-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = Path(args.data_root)
    matched = Path(args.matched_root)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("Use a new experiment directory")
    complete = json.loads((matched / "matched_complete.json").read_text())
    if complete["status"] != "complete":
        raise ValueError("Source-model training is incomplete")
    budget = 14400 - float(complete["training_seconds_including_prior"])
    if budget < 300:
        raise RuntimeError("Insufficient authorized cumulative training time")
    prepared = json.loads((data / "summary.json").read_text())
    if (
        prepared["status"] != "complete"
        or sha256(data / "protocol.json") != prepared["protocol_sha256"]
    ):
        raise ValueError("Data protocol changed")
    protocol = json.loads((data / "protocol.json").read_text())
    out.mkdir(parents=True)
    models = []
    for seed in protocol["seeds"]:
        source = matched / "matched_runs" / f"rgb_loveda_s{seed}"
        source_config = json.loads((source / "config.json").read_text())
        if (
            source_config["channel_indices"] != [3, 2, 1]
            or source_config["selection_split"] != "val"
        ):
            raise ValueError("Incompatible source model")
        source_summary = json.loads((source / "summary.json").read_text())
        models.append(
            {
                "name": f"zero_shot_s{seed}",
                "seed": seed,
                "stage": "zero_shot",
                "checkpoint": str(source / "best.pt"),
                "sha256": source_summary["checkpoint_sha256"],
            }
        )
    frozen = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_protocol_sha256": sha256(data / "protocol.json"),
        "manifest_sha256": {
            s: sha256(data / (s + "_manifest.json")) for s in ["train", "val", "test"]
        },
        "source_models": models.copy(),
        "remaining_training_budget_seconds": budget,
        "code_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        "epochs": protocol["fine_tune_epochs"],
        "batch_size": protocol["batch_size"],
        "seeds": protocol["seeds"],
        "learning_rate": protocol["lr"],
        "selection_metric": "validation Landslide_IoU",
        "test_region": "dominicamaria",
        "versions": {
            n: importlib.metadata.version(n)
            for n in ["torch", "torchvision", "timm", "numpy", "h5py"]
        },
        "model_loss_sha256": {
            name: sha256(Path(__file__).resolve().parents[1] / "train_l4s_qz" / name)
            for name in ["model.py", "losses.py"]
        },
    }
    write_json(out / "training_plan.json", frozen)
    train = PreparedDataset(data, "train", train=True)
    val = PreparedDataset(data, "val")
    if (len(train), len(val)) != (40, 10):
        raise ValueError("Label allocation must remain 40 training plus 10 validation")
    counts = np.array(
        [sum(int((y == c).sum()) for _, y in train.cache) for c in [0, 1]],
        dtype=np.float64,
    )
    if (counts == 0).any():
        raise ValueError("Both classes must be represented in training")
    weights = 1 / np.sqrt(counts / counts.sum() + 1e-12)
    weights = np.clip(weights / weights.mean(), 0.5, 2.5)
    write_json(
        out / "training_class_weights.json",
        {
            "class_counts": counts.tolist(),
            "weights": weights.tolist(),
            "source": "40 training masks only",
        },
    )
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    started = time.monotonic()
    val_loader = DataLoader(val, batch_size=8, shuffle=False, num_workers=0)
    for source in models.copy():
        seed = source["seed"]
        seed_all(seed)
        run = out / f"few_shot_s{seed}"
        run.mkdir()
        model = load_model(source["checkpoint"], source["sha256"], device)
        loader = DataLoader(
            train,
            batch_size=8,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(seed),
        )
        loss_fn = CombinedLoss(class_weights=weights.tolist(), label_smoothing=0.02).to(
            device
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=protocol["lr"], weight_decay=protocol["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, protocol["fine_tune_epochs"], eta_min=protocol["min_lr"]
        )
        scaler = torch.amp.GradScaler("cuda")
        best = -1.0
        history = []
        best_epoch = 0
        for epoch in range(1, protocol["fine_tune_epochs"] + 1):
            write_json(
                out / "status.json",
                {
                    "phase": "training",
                    "seed": seed,
                    "epoch": epoch,
                    "elapsed_training_seconds": time.monotonic() - started,
                },
            )
            model.train()
            losses = []
            for x, y, _ in loader:
                if time.monotonic() - started > budget - 45:
                    raise TimeoutError("Cumulative four-hour training budget reached")
                x = x.to(device, memory_format=torch.channels_last)
                y = y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda"):
                    loss = loss_fn(model(x), y)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())
            model.eval()
            cm = np.zeros((2, 2), dtype=np.int64)
            with torch.inference_mode():
                for x, y, _ in val_loader:
                    with torch.autocast("cuda"):
                        p = (
                            predict(
                                model,
                                x.to(device, memory_format=torch.channels_last),
                                True,
                            )
                            .argmax(1)
                            .cpu()
                            .numpy()
                        )
                    cm += confusion(p, y.numpy())
            scores = metrics(cm)
            scheduler.step()
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    **{"val_" + k: v for k, v in scores.items()},
                    "seconds": time.monotonic() - started,
                }
            )
            csv_write(run / "history.csv", history)
            if scores["Landslide_IoU"] > best:
                best = scores["Landslide_IoU"]
                best_epoch = epoch
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "source_sha256": source["sha256"],
                        "data_protocol_sha256": frozen["data_protocol_sha256"],
                        "selection_split": "val",
                        "validation_metrics": scores,
                    },
                    run / "best.pt",
                )
            print(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "validation_iou": scores["Landslide_IoU"],
                },
                flush=True,
            )
        digest = sha256(run / "best.pt")
        write_json(
            run / "summary.json",
            {
                "status": "complete",
                "epochs": 40,
                "best_epoch": best_epoch,
                "best_validation_iou": best,
                "checkpoint_sha256": digest,
                "source_sha256": source["sha256"],
                "test_evaluated_during_training": False,
            },
        )
        models.append(
            {
                "name": f"few_shot_s{seed}",
                "seed": seed,
                "stage": "few_shot",
                "checkpoint": str(run / "best.pt"),
                "sha256": digest,
            }
        )
        del model, optimizer, loss_fn
        torch.cuda.empty_cache()
    training_seconds = time.monotonic() - started
    write_json(
        out / "checkpoints_frozen_before_test.json",
        {
            "frozen_utc": datetime.now(timezone.utc).isoformat(),
            "models": models,
            "training_seconds": training_seconds,
        },
    )
    gate = out / "TEST_EVALUATION_STARTED"
    with gate.open("x") as f:
        f.write(datetime.now(timezone.utc).isoformat() + "\n")
    if sha256(data / "test_manifest.json") != frozen["manifest_sha256"]["test"]:
        raise ValueError("Test manifest changed")
    test = PreparedDataset(data, "test", allow_test=True)
    test_loader = DataLoader(test, batch_size=32, shuffle=False, num_workers=0)
    summaries = []
    all_rows = []
    for spec in models:
        run = out / spec["name"]
        run.mkdir(exist_ok=True)
        write_json(out / "status.json", {"phase": "evaluation", "model": spec["name"]})
        model = load_model(spec["checkpoint"], spec["sha256"], device).eval()
        cm = np.zeros((2, 2), dtype=np.int64)
        rows = []
        predictions = []
        with torch.inference_mode():
            for x, y, indices in test_loader:
                with torch.autocast("cuda"):
                    pred = (
                        predict(
                            model, x.to(device, memory_format=torch.channels_last), True
                        )
                        .argmax(1)
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                    )
                for p, g, index in zip(pred, y.numpy(), indices.tolist()):
                    meta = test.rows[index]
                    sample_cm = confusion(p, g)
                    cm += sample_cm
                    predictions.append(p)
                    rows.append(
                        {
                            "model": spec["name"],
                            "seed": spec["seed"],
                            "stage": spec["stage"],
                            "sample": meta["sample"],
                            "region": meta["region"],
                            "block_id": meta["block_id"],
                            "acquisition_date": meta["acquisition_date"],
                            "valid_pixels": meta["valid_pixels"],
                            "positive_pixels": meta["positive_pixels"],
                            "raw_background_patch": meta["raw_positive_pixels"] == 0,
                            **metrics(sample_cm),
                            "foreground_patch_iou": (
                                metrics(sample_cm)["Landslide_IoU"]
                                if meta["positive_pixels"]
                                else ""
                            ),
                        }
                    )
        pooled = np.array(
            [
                [sum(r["TN"] for r in rows), sum(r["FP"] for r in rows)],
                [sum(r["FN"] for r in rows), sum(r["TP"] for r in rows)],
            ]
        )
        assert np.array_equal(cm, pooled) and cm.sum() == sum(
            r["valid_pixels"] for r in rows
        )
        intervals = bootstrap_blocks(rows)
        summary = {
            "model": spec["name"],
            "seed": spec["seed"],
            "stage": spec["stage"],
            "region": "dominicamaria",
            "samples": len(rows),
            **metrics(cm),
            "checkpoint_sha256": spec["sha256"],
            "confusion_matrix": cm.tolist(),
            "sample_global_counts_match": True,
            "spatial_uncertainty": intervals,
        }
        write_json(run / "metrics.json", summary)
        csv_write(run / "per_sample.csv", rows)
        np.savez_compressed(
            run / "predictions.npz",
            names=np.array([r["sample"] for r in rows]),
            masks=np.stack(predictions),
        )
        summaries.append(
            {
                k: v
                for k, v in summary.items()
                if k not in ["confusion_matrix", "spatial_uncertainty"]
            }
            | {
                "spatial_blocks": intervals["block_count"],
                "block_iou_ci_low": (
                    intervals["interval"][0] if intervals["interval"] else ""
                ),
                "block_iou_ci_high": (
                    intervals["interval"][1] if intervals["interval"] else ""
                ),
            }
        )
        all_rows.extend(rows)
        del model
        torch.cuda.empty_cache()
    csv_write(out / "external_metrics.csv", summaries)
    csv_write(out / "external_per_sample.csv", all_rows)
    display = json.loads((data / "visualization_manifest.json").read_text())
    display_rows = [r for r in test.rows if r["sample"] in display["samples"]]
    arrays = {"names": np.array([r["sample"] for r in display_rows])}
    for key in ["rgb", "y"]:
        values = []
        for row in display_rows:
            with np.load(
                data / "samples" / row["prepared_file"], allow_pickle=False
            ) as z:
                values.append(z[key])
        arrays[key] = np.stack(values)
    np.savez_compressed(out / "qualitative_inputs.npz", **arrays)
    write_json(
        out / "complete.json",
        {
            "status": "complete",
            "training_seconds": training_seconds,
            "cumulative_training_seconds": complete["training_seconds_including_prior"]
            + training_seconds,
            "test_events": 1,
            "test_models": 6,
            "test_samples": len(test),
            "test_metrics_used_for_selection": False,
            "sample_global_counts_all_match": True,
        },
    )
    write_json(
        out / "status.json",
        {
            "phase": "complete",
            "cumulative_training_seconds": complete["training_seconds_including_prior"]
            + training_seconds,
        },
    )


if __name__ == "__main__":
    main()
