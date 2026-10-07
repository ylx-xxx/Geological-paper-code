"""Check exact RGB content across datasets; do not infer geospatial independence from hashes."""

import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np
from .core import pairs, read_h5, normalize_channels, write_json


def signatures(x):
    return {
        hashlib.sha256(np.ascontiguousarray(view).tobytes()).hexdigest()
        for k in range(4)
        for view in [np.rot90(x, k, (1, 2)), np.rot90(x, k, (1, 2))[:, :, ::-1]]
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--l4s-root", required=True)
    ap.add_argument("--external-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("Use a new audit output")
    out.mkdir(parents=True)
    started = time.monotonic()
    external = Path(args.external_root)
    rows = json.loads((external / "test_manifest.json").read_text())
    raw_index = {}
    normalized_index = {}
    degenerate = []
    for row in rows:
        with np.load(external / "samples" / row["prepared_file"]) as data:
            raw = data["rgb"].astype(np.float32)
            normalized = data["x"].astype(np.float32)
        if float(raw.std()) < 1e-6:
            degenerate.append(row["sample"])
            continue
        for digest in signatures(raw):
            raw_index.setdefault(digest, []).append(row["sample"])
        for digest in signatures(normalized):
            normalized_index.setdefault(digest, []).append(row["sample"])
    matches = []
    counts = {}
    for split in ["train", "val", "test"]:
        source = pairs(args.l4s_root, split)
        counts[split] = len(source)
        for image, _ in source:
            array = read_h5(image, "img")
            x = (array.transpose(2, 0, 1) if array.shape[-1] == 14 else array)[
                [3, 2, 1]
            ].astype(np.float32)
            if float(x.std()) < 1e-6:
                continue
            raw_hash = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
            normalized_hash = hashlib.sha256(
                normalize_channels(x).tobytes()
            ).hexdigest()
            for kind, digest, index in [
                ("raw_rgb", raw_hash, raw_index),
                ("normalized_rgb", normalized_hash, normalized_index),
            ]:
                for name in index.get(digest, []):
                    matches.append(
                        {
                            "split": split,
                            "l4s_sample": image.name,
                            "external_sample": name,
                            "match": kind,
                        }
                    )
        print(f"Checked {split}: {len(source)}", flush=True)
    report = {
        "l4s_samples": counts,
        "external_samples": len(rows),
        "exact_content_matches": matches,
        "degenerate_external_samples": degenerate,
        "orientations_checked": 8,
        "seconds": time.monotonic() - started,
        "scope": "exact raw or patch-normalized physical RGB under rotations/reflections",
        "limitation": "Different radiometric processing, resampling, neighboring tiles, or partial overlaps are not excluded by this check. L4S releases do not provide full georeferencing and event IDs.",
    }
    write_json(out / "audit.json", report)
    with (out / "matches.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["split", "l4s_sample", "external_sample", "match"]
        )
        writer.writeheader()
        writer.writerows(matches)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
