"""Frozen single-image RGB protocol for regional Sen12 transfer experiments."""

import hashlib
from datetime import datetime, timedelta
from io import BytesIO
import h5py
import numpy as np
from .core import normalize_channels
from .prepare_sen12_subset import attribute_text, dates_from_cf

RGB = ["B04", "B03", "B02"]
EVENT_ANCHORS = {"chimanimani": "2019-03-15", "dominicamaria": "2017-09-23"}
# Dates above are dataset metadata anchors, not independently verified storm dates.
GOOD_SCL = [2, 4, 5, 6, 7]
QUALITY_WINDOW_DAYS = 180
MIN_VALID_FRACTION = 0.8


def choose_time(dates, scl, event_date):
    """Choose the clearest strictly post-anchor frame; never inspect MASK."""
    anchor = datetime.fromisoformat(event_date)
    upper = anchor + timedelta(days=QUALITY_WINDOW_DAYS)
    candidates = []
    for i, date in enumerate(dates):
        stamp = datetime.fromisoformat(date)
        if anchor < stamp <= upper:
            fraction = float(np.isin(scl[i], GOOD_SCL).mean())
            candidates.append((-fraction, stamp, i))
    if not candidates:
        return None, "no_frame_in_post_anchor_window"
    negative_fraction, _, index = min(candidates)
    if -negative_fraction < MIN_VALID_FRACTION:
        return None, "insufficient_clear_pixels"
    return index, None


def read_case(payload, region):
    if region not in EVENT_ANCHORS:
        raise ValueError("Region has no frozen temporal policy")
    with h5py.File(BytesIO(payload), "r") as data:
        for band in RGB + ["MASK", "SCL"]:
            if data[band].shape != (15, 128, 128):
                raise ValueError("Unexpected data shape")
            dims = [list(data[band].dims[i].keys()) for i in range(3)]
            if dims != [["time"], ["x"], ["y"]]:
                raise ValueError("Unexpected stored axes")
        dates = dates_from_cf(
            data["time"][:],
            attribute_text(data["time"].attrs["units"]),
            attribute_text(data["time"].attrs.get("calendar", "standard")),
        )
        scl = data["SCL"][:]
        index, reason = choose_time(dates, scl, EVENT_ANCHORS[region])
        x, y = data["x"][:], data["y"][:]
        mask = data["MASK"][:]
        if not np.isin(mask, [0, 1]).all() or not np.all(mask == mask[0]):
            raise ValueError("Unexpected mask semantics")
        bbox = [
            float(x.min() - 5),
            float(y.min() - 5),
            float(x.max() + 5),
            float(y.max() + 5),
        ]
        meta = {
            "region": region,
            "crs": attribute_text(data.attrs["crs"]),
            "bbox": bbox,
            "event_anchor": EVENT_ANCHORS[region],
            "annotation_event_date": attribute_text(data.attrs.get("event_date", "")),
            "raw_positive_pixels": int((mask[0] == 1).sum()),
            "excluded_reason": reason,
        }
        if reason:
            return None, meta
        rgb = np.stack([data[band][index].T for band in RGB]).astype(np.float32)
        valid = np.isin(scl[index].T, GOOD_SCL)
        target = mask[index].T.astype(np.uint8)
        target[~valid] = 255
        if not np.isfinite(rgb).all():
            raise ValueError("Nonfinite raw RGB input")
        normalized = normalize_channels(rgb)
        meta.update(
            {
                "time_index": int(index),
                "acquisition_date": dates[index],
                "valid_pixels": int(valid.sum()),
                "valid_fraction": float(valid.mean()),
                "positive_pixels": int((target == 1).sum()),
                "input_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "block_id": f"{region}:{int(((bbox[0]+bbox[2])/2)//2560)}:{int(((bbox[1]+bbox[3])/2)//2560)}",
            }
        )
        return {"x": normalized, "y": target, "rgb": rgb}, meta


def rectangle_gap(a, b):
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return float(np.hypot(dx, dy))


def select_support(rows, seed=20261004):
    """Labels do not influence selection. West/east pools have a >=1280 m gap."""
    eligible = [
        r for r in rows if r["region"] == "chimanimani" and not r["excluded_reason"]
    ]
    boundary = float(np.median([(r["bbox"][0] + r["bbox"][2]) / 2 for r in eligible]))
    train_pool = sorted(
        [r for r in eligible if r["bbox"][2] <= boundary - 640],
        key=lambda r: r["sample"],
    )
    val_pool = sorted(
        [r for r in eligible if r["bbox"][0] >= boundary + 640],
        key=lambda r: r["sample"],
    )
    if len(train_pool) < 40 or len(val_pool) < 10:
        raise ValueError("Insufficient spatially separated support pools")
    rng = np.random.default_rng(seed)
    train = [
        train_pool[i] for i in sorted(rng.choice(len(train_pool), 40, replace=False))
    ]
    val = [val_pool[i] for i in sorted(rng.choice(len(val_pool), 10, replace=False))]
    gap = min(rectangle_gap(a["bbox"], b["bbox"]) for a in train for b in val)
    if gap < 1280 - 1e-6:
        raise ValueError("Spatial buffer violated")
    for subset in [train, val]:
        if not sum(r["positive_pixels"] for r in subset):
            raise ValueError(
                "Frozen label-blind draw has no foreground; do not silently redraw"
            )
    return (
        train,
        val,
        {
            "boundary_easting": boundary,
            "train_pool": len(train_pool),
            "val_pool": len(val_pool),
            "minimum_train_val_footprint_gap_m": gap,
            "draw_seed": seed,
            "selection_uses_labels": False,
        },
    )


def assert_training_regions(rows):
    if not rows or any(r["region"] != "chimanimani" for r in rows):
        raise ValueError("External test region cannot enter training or validation")


def bootstrap_blocks(rows, replicates=2000, seed=4242):
    """Conditional, within-region uncertainty; pixels are never independent draws."""
    from .core import metrics

    groups = {}
    for row in rows:
        groups.setdefault(row["block_id"], np.zeros((2, 2), dtype=np.int64))
        groups[row["block_id"]] += np.array(
            [[row["TN"], row["FP"]], [row["FN"], row["TP"]]], dtype=np.int64
        )
    keys = sorted(groups)
    n = len(keys)
    if n < 5:
        return {
            "block_count": n,
            "interval": None,
            "reason": "fewer_than_five_spatial_blocks",
        }
    counts = np.stack([groups[k] for k in keys])
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        total = counts[rng.integers(0, n, n)].sum(0)
        if total[1].sum() == 0:
            continue
        values.append(metrics(total)["Landslide_IoU"])
    return {
        "block_count": n,
        "block_size_m": 2560,
        "replicates": replicates,
        "valid_replicates": len(values),
        "interval": np.quantile(values, [0.025, 0.975]).tolist() if values else None,
        "interpretation": "within-event spatial block bootstrap; not uncertainty across independent events",
    }
