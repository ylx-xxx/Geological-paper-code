"""Create a frozen 40/10 support split and region-held-out evaluation manifest."""

import argparse, csv, hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
import numpy as np
from .core import sha256, write_json
from .sen12_protocol import (
    read_case,
    select_support,
    assert_training_regions,
    RGB,
    EVENT_ANCHORS,
    GOOD_SCL,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chimanimani-root", required=True)
    ap.add_argument("--dominica-root", required=True)
    ap.add_argument("--dominica-manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("Never overwrite a frozen dataset")
    out.mkdir(parents=True)
    (out / "samples").mkdir()
    base = Path(args.chimanimani_root)
    with (base / "audit_v2/sample_manifest.csv").open(
        encoding="utf8", newline=""
    ) as stream:
        chima = list(csv.DictReader(stream))
    dominica = json.loads(Path(args.dominica_manifest).read_text())["samples"]
    specs = [(r, base / "raw" / r["sample"], "chimanimani") for r in chima]
    specs += [
        (r, Path(args.dominica_root) / r["sample"], "dominicamaria")
        for r in dominica
        if r["region"] == "dominicamaria"
    ]
    plan = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "rgb": RGB,
        "axes": "stored time,x,y -> C,y,x",
        "event_anchors_from_dataset_metadata": EVENT_ANCHORS,
        "date_rule": "strictly after anchor, at most 180 days; highest clear fraction, earliest tie",
        "clear_scl_codes": GOOD_SCL,
        "minimum_clear_fraction": 0.8,
        "invalid_pixels": "MASK=255 for loss and metrics; retained in raw files",
        "normalization": "finite_zero_p1_p99_patch_band_zscore_v1",
        "support": "40 training + 10 validation, label-blind draw, >=1280 m footprint gap",
        "evaluation_region": "dominicamaria",
        "evaluation_sampling": "all 208 Dominica files in pinned s2_part03 before image-quality filtering",
        "training_region": "chimanimani",
        "source_revision": "311f426d0e2fa9b532772fbee0641e2f9db6da00",
        "excluded_region": "china: multi-year mixed-event annotations; incompatible with this single-event image protocol",
        "author_history_confirmation": "Author confirms no prior training in Sen12 regions outside Chimanimani; new test regions excluded from this round of development",
        "seeds": [2026, 2027, 2028],
        "fine_tune_epochs": 40,
        "batch_size": 8,
        "lr": 3e-5,
        "min_lr": 1e-6,
        "weight_decay": 0.05,
        "loss": "weighted CE plus weighted Dice; label smoothing 0.02; training-only inverse-square-root weights",
        "selection": "validation landslide IoU; earliest tie",
        "tta_views": 3,
        "prediction": "argmax of mean logits; no threshold search",
        "evaluation_opening": "after all three fine-tuned checkpoints and all three zero-shot source checkpoints are frozen",
        "uncertainty": "2560 m spatial block bootstrap, 2000 replicates, conditional on the single test event",
        "code_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
    }
    write_json(out / "protocol.json", plan)
    rows = []
    seen = {}
    for original, path, region in specs:
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != original["sha256"]:
            raise ValueError("Raw file differs from verified source: " + path.name)
        sample, meta = read_case(payload, region)
        row = meta | {
            "sample": path.name,
            "raw_sha256": original["sha256"],
            "raw_path": str(path),
        }
        if sample is not None:
            if meta["input_sha256"] in seen:
                raise ValueError("Duplicate selected RGB array: " + path.name)
            seen[meta["input_sha256"]] = path.name
            target = out / "samples" / (path.stem + ".npz")
            np.savez_compressed(target, **sample)
            row["prepared_sha256"] = sha256(target)
            row["prepared_file"] = target.name
        rows.append(row)
        if len(rows) % 100 == 0:
            print(f"Prepared {len(rows)}/{len(specs)}", flush=True)
    train, val, spatial = select_support(rows)
    assert_training_regions(train)
    assert_training_regions(val)
    test = [
        r for r in rows if r["region"] == "dominicamaria" and not r["excluded_reason"]
    ]
    if (
        not test
        or not any(r["positive_pixels"] > 0 for r in test)
        or not any(r["raw_positive_pixels"] == 0 for r in test)
    ):
        raise ValueError(
            "Evaluation must include landslide and natural-background patches"
        )
    splits = {"train": train, "val": val, "test": test}
    ids = [{r["sample"] for r in split} for split in splits.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    for name, values in splits.items():
        write_json(out / (name + "_manifest.json"), values)
        flat = [
            {
                k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                for k, v in r.items()
            }
            for r in values
        ]
        with (out / (name + "_manifest.csv")).open(
            "w", newline="", encoding="utf8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    # Display IDs are fixed before any prediction, so figures cannot select by model score.
    background = sorted(
        [r for r in test if r["raw_positive_pixels"] == 0], key=lambda r: r["sample"]
    )
    positive = sorted(
        [r for r in test if r["positive_pixels"] > 0],
        key=lambda r: (r["positive_pixels"], r["sample"]),
    )
    display = [
        background[i]
        for i in np.linspace(0, len(background) - 1, min(2, len(background)), dtype=int)
    ]
    display += [
        positive[i]
        for i in np.linspace(0, len(positive) - 1, min(6, len(positive)), dtype=int)
    ]
    write_json(
        out / "visualization_manifest.json",
        {
            "seed_to_display": 2026,
            "selection": "2 background and 6 foreground-coverage strata, before model inference",
            "samples": [r["sample"] for r in display],
        },
    )
    write_json(out / "all_samples.json", rows)
    summary = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "spatial_support": spatial,
        "regions": {},
        "splits": {},
        "protocol_sha256": sha256(out / "protocol.json"),
        "external_metrics_inspected": False,
        "primary_test_events": 1,
        "raw_test_samples_in_archive": 208,
    }
    for region in EVENT_ANCHORS:
        subset = [r for r in rows if r["region"] == region]
        summary["regions"][region] = {
            "raw_samples": len(subset),
            "eligible_samples": sum(not r["excluded_reason"] for r in subset),
            "exclusions": dict(
                Counter(r["excluded_reason"] for r in subset if r["excluded_reason"])
            ),
        }
    for name, values in splits.items():
        summary["splits"][name] = {
            "samples": len(values),
            "positive_patches": sum(r["positive_pixels"] > 0 for r in values),
            "raw_background_patches": sum(
                r["raw_positive_pixels"] == 0 for r in values
            ),
            "positive_pixels": sum(r["positive_pixels"] for r in values),
            "valid_pixels": sum(r["valid_pixels"] for r in values),
            "manifest_sha256": sha256(out / (name + "_manifest.json")),
        }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
