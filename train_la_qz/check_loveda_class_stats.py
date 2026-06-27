from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm


ROOT = Path("/root/autodl-tmp/datasetss/loveda")
NUM_CLASSES = 7
IGNORE_INDEX = 255


def collect(split):
    if split == "train":
        base = ROOT / "Train" / "Train"
    else:
        base = ROOT / "Val" / "Val"

    mask_paths = []
    for domain in ["Rural", "Urban"]:
        mask_paths += sorted((base / domain / "masks_png").glob("*.png"), key=lambda x: int(x.stem))

    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    image_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for p in tqdm(mask_paths, desc=split):
        mask = np.array(Image.open(p), dtype=np.int64)

        # LoveDA: 0 ignore, 1~7 valid
        mask = np.where(mask == 0, IGNORE_INDEX, mask - 1)

        valid = mask != IGNORE_INDEX
        for c in range(NUM_CLASSES):
            cnt = np.sum(mask[valid] == c)
            pixel_counts[c] += cnt
            if cnt > 0:
                image_counts[c] += 1

    return mask_paths, pixel_counts, image_counts


def report(split):
    paths, pixel_counts, image_counts = collect(split)
    total = pixel_counts.sum()

    print("\n" + "=" * 80)
    print(f"{split.upper()} SET")
    print(f"Images: {len(paths)}")
    print("=" * 80)

    for c in range(NUM_CLASSES):
        ratio = pixel_counts[c] / total * 100 if total > 0 else 0
        print(
            f"class_index={c} | original_label={c+1} | "
            f"pixel={pixel_counts[c]:>12} | pixel_ratio={ratio:>7.3f}% | "
            f"images_with_class={image_counts[c]:>5}"
        )

    print("=" * 80)


if __name__ == "__main__":
    report("train")
    report("val")
