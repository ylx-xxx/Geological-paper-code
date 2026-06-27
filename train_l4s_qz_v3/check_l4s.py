from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm


ROOT = Path("/root/autodl-tmp/datasetss/landslide4Sense")


def read_h5_first(path):
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        if len(keys) == 0:
            raise RuntimeError(f"No keys in {path}")
        key = keys[0]
        arr = f[key][()]
    return key, arr


def collect_pairs(split):
    if split == "train":
        img_dir = ROOT / "TrainData" / "img"
        mask_dir = ROOT / "TrainData" / "mask"
    elif split == "val":
        img_dir = ROOT / "ValidData" / "img"
        mask_dir = ROOT / "ValidData" / "mask"
    elif split == "test":
        img_dir = ROOT / "TestData" / "img"
        mask_dir = ROOT / "TestData" / "test"
    else:
        raise ValueError(split)

    pairs = []
    missing = []

    img_paths = sorted(img_dir.glob("image_*.h5"), key=lambda p: int(p.stem.split("_")[-1]))
    for img_path in img_paths:
        idx = img_path.stem.split("_")[-1]
        mask_path = mask_dir / f"mask_{idx}.h5"
        if mask_path.exists():
            pairs.append((img_path, mask_path))
        else:
            missing.append(img_path.name)

    extra_masks = []
    img_ids = {p.stem.split("_")[-1] for p in img_paths}
    for m in sorted(mask_dir.glob("mask_*.h5"), key=lambda p: int(p.stem.split("_")[-1])):
        mid = m.stem.split("_")[-1]
        if mid not in img_ids:
            extra_masks.append(m.name)

    return pairs, missing, extra_masks


def inspect_split(split):
    pairs, missing, extra_masks = collect_pairs(split)

    print("\n" + "=" * 100)
    print(f"{split.upper()} SPLIT")
    print("=" * 100)
    print(f"paired samples: {len(pairs)}")
    print(f"missing masks: {len(missing)}")
    print(f"extra masks: {len(extra_masks)}")

    if missing:
        print("missing examples:", missing[:10])
    if extra_masks:
        print("extra mask examples:", extra_masks[:20])

    if len(pairs) == 0:
        return

    img_key, img = read_h5_first(pairs[0][0])
    mask_key, mask = read_h5_first(pairs[0][1])

    print(f"sample image: {pairs[0][0].name}, key={img_key}, shape={img.shape}, dtype={img.dtype}")
    print(f"sample mask : {pairs[0][1].name}, key={mask_key}, shape={mask.shape}, dtype={mask.dtype}")
    print(f"image min/max/mean: {np.nanmin(img):.4f} / {np.nanmax(img):.4f} / {np.nanmean(img):.4f}")
    print(f"mask unique: {np.unique(mask)}")

    counts = np.zeros(2, dtype=np.int64)
    bad_masks = []

    for _, mask_path in tqdm(pairs, desc=f"count {split}"):
        _, m = read_h5_first(mask_path)
        u = np.unique(m)
        if not set(u.tolist()).issubset({0, 1}):
            bad_masks.append((mask_path.name, u.tolist()))
        counts[0] += np.sum(m == 0)
        counts[1] += np.sum(m == 1)

    total = counts.sum()
    print(f"pixel count non-landslide: {counts[0]} | ratio={counts[0] / total * 100:.4f}%")
    print(f"pixel count landslide    : {counts[1]} | ratio={counts[1] / total * 100:.4f}%")
    print(f"bad masks: {len(bad_masks)}")
    if bad_masks:
        print("bad mask examples:", bad_masks[:10])


if __name__ == "__main__":
    inspect_split("train")
    inspect_split("val")
    inspect_split("test")
