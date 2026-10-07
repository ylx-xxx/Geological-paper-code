"""Extract and hash-check a manifest-defined Sen12 region from a pinned archive."""

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from .core import sha256, write_json
from .prepare_sen12_subset import safe_member_name


def extract(
    archive_path,
    inventory_path,
    region,
    expected_archive_sha256,
    output,
    min_free_gib=2,
):
    archive_path, output = Path(archive_path), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new extraction directory")
    inventory = json.loads(Path(inventory_path).read_text(encoding="utf8"))
    rows = [row for row in inventory["samples"] if row["region"] == region]
    expected = {row["sample"]: row for row in rows}
    if not rows or len(expected) != len(rows):
        raise ValueError("Region manifest is empty or has duplicate filenames")
    if any(Path(name).name != name or "\\" in name or ":" in name for name in expected):
        raise ValueError("Manifest contains unsafe filenames")
    actual = sha256(archive_path)
    if actual != expected_archive_sha256:
        raise ValueError("Archive differs from the pinned publisher SHA256")
    output.mkdir(parents=True)
    raw = output / "raw"
    raw.mkdir()
    retained = set()
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            name = safe_member_name(member)
            if not member.isfile() or name not in expected:
                continue
            if name in retained:
                raise ValueError("Repeated selected filename in archive")
            row = expected[name]
            if not 0 < member.size < 128 * 1024**2 or member.size != int(row["bytes"]):
                raise ValueError("Unexpected selected-file size")
            if shutil.disk_usage(raw).free < member.size + min_free_gib * 1024**3:
                raise RuntimeError("Free-space threshold reached")
            payload = archive.extractfile(member).read(member.size + 1)
            if (
                len(payload) != member.size
                or hashlib.sha256(payload).hexdigest() != row["sha256"]
            ):
                raise ValueError("Selected file differs from frozen inventory")
            destination = raw / name
            if destination.resolve().parent != raw:
                raise ValueError("Destination escaped the regional directory")
            with destination.open("xb") as stream:
                stream.write(payload)
            retained.add(name)
    missing = sorted(set(expected) - retained)
    if missing:
        raise ValueError(f"Archive is missing {len(missing)} frozen regional samples")
    receipt = {
        "status": "complete",
        "region": region,
        "files": len(retained),
        "archive_sha256": actual,
        "inventory_sha256": sha256(inventory_path),
        "all_file_sha256_match": True,
        "raw_values_modified": False,
    }
    write_json(output / "extraction_receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--region", default="dominicamaria")
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-free-gib", type=float, default=2)
    args = parser.parse_args()
    if args.min_free_gib < 0:
        raise ValueError("Negative free-space threshold")
    print(
        json.dumps(
            extract(
                args.archive,
                args.inventory,
                args.region,
                args.expected_archive_sha256,
                args.out,
                args.min_free_gib,
            )
        )
    )


if __name__ == "__main__":
    main()
