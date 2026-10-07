"""Shared preprocessing, channel definitions, metrics, and provenance."""

from pathlib import Path
import hashlib
import json
import numpy as np

# IARAI Landslide4Sense-2022: stored bands B1..B12 (B8A omitted), slope, DEM.
CHANNELS = {
    "full14": list(range(14)),
    "ms12": list(range(12)),
    "first3_legacy": [0, 1, 2],
    "first3_topo_legacy": [0, 1, 2, 12, 13],
    "rgb": [3, 2, 1],
    "rgb_topo": [3, 2, 1, 12, 13],
}
NORMALIZATION = "finite_zero_p1_p99_patch_band_zscore_v1"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_h5(path, key):
    import h5py

    with h5py.File(path, "r") as f:
        if key in f:
            return f[key][()]
        if len(f) == 1:
            return f[next(iter(f))][()]
        raise ValueError(f"Ambiguous HDF5 keys in {path}: {list(f)}")


def normalize(image):
    """Numerically matches the original final-model dataset implementation."""
    x = np.asarray(image, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D input, got {x.shape}")
    if x.shape[-1] == 14:
        x = x.transpose(2, 0, 1)
    elif x.shape[0] != 14:
        raise ValueError(f"Expected 14 native bands, got {x.shape}")
    return normalize_channels(x)


def normalize_channels(image):
    """The same patch/band transform for an explicitly ordered C,H,W tensor."""
    x = np.asarray(image, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError("Expected an explicitly ordered C,H,W array")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.zeros_like(x, dtype=np.float32)
    for c, band in enumerate(x):
        band = np.clip(band, np.percentile(band, 1), np.percentile(band, 99))
        std = band.std()
        out[c] = (band - band.mean()) / (1.0 if std < 1e-6 else std)
    return out


def confusion(prediction, target):
    p, y = np.asarray(prediction), np.asarray(target)
    if p.shape != y.shape:
        raise ValueError("Prediction and target shapes differ")
    valid = y != 255
    if not np.isin(y[valid], [0, 1]).all() or not np.isin(p[valid], [0, 1]).all():
        raise ValueError("Expected binary labels (or target ignore label 255)")
    return np.bincount((2 * y[valid] + p[valid]).astype(np.int64), minlength=4).reshape(
        2, 2
    )


def metrics(cm):
    tn, fp, fn, tp = np.asarray(cm, dtype=np.int64).ravel()
    div = lambda a, b: float(a / b) if b else 0.0
    iou, bg = div(tp, tp + fp + fn), div(tn, tn + fp + fn)
    return {
        "Landslide_IoU": iou,
        "Landslide_F1": div(2 * tp, 2 * tp + fp + fn),
        "Landslide_Precision": div(tp, tp + fp),
        "Landslide_Recall": div(tp, tp + fn),
        "mIoU": (iou + bg) / 2,
        "OA": div(tp + tn, tp + tn + fp + fn),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def pairs(root, split):
    names = {
        "train": ("TrainData", "mask"),
        "val": ("ValidData", "mask"),
        "test": ("TestData", "test"),
    }
    folder, mask_folder = names[split]
    images = sorted(
        (Path(root) / folder / "img").glob("image_*.h5"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    result = [
        (p, Path(root) / folder / mask_folder / p.name.replace("image_", "mask_"))
        for p in images
    ]
    if not result or any(not m.exists() for _, m in result):
        raise ValueError(f"Missing images or masks in {folder}")
    return result


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf8",
    )


def verify_selection_split(split):
    if split != "val":
        raise ValueError(
            "Checkpoint selection requires validation data; test scores cannot select a model."
        )
