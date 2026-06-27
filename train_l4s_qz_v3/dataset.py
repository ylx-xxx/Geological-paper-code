from pathlib import Path
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


def read_h5_first(path):
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        if len(keys) == 0:
            raise RuntimeError(f"No keys found in {path}")
        key = keys[0]
        arr = f[key][()]
    return arr


class Landslide4SenseDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        img_size=128,
        train=True,
        use_augmentation=True,
    ):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.train = train
        self.use_augmentation = use_augmentation and train

        if split == "train":
            self.img_dir = self.root / "TrainData" / "img"
            self.mask_dir = self.root / "TrainData" / "mask"
        elif split == "val":
            self.img_dir = self.root / "ValidData" / "img"
            self.mask_dir = self.root / "ValidData" / "mask"
        elif split == "test":
            self.img_dir = self.root / "TestData" / "img"
            self.mask_dir = self.root / "TestData" / "test"
        else:
            raise ValueError(f"Unsupported split: {split}")

        self.samples = self._collect_pairs()

        if len(self.samples) == 0:
            raise RuntimeError(f"No paired samples found in {self.img_dir}")

    def _collect_pairs(self):
        img_paths = sorted(
            self.img_dir.glob("image_*.h5"),
            key=lambda p: int(p.stem.split("_")[-1])
        )

        pairs = []
        for img_path in img_paths:
            idx = img_path.stem.split("_")[-1]
            mask_path = self.mask_dir / f"mask_{idx}.h5"
            if mask_path.exists():
                pairs.append((img_path, mask_path))

        return pairs

    def __len__(self):
        return len(self.samples)

    def _normalize_14band(self, img):
        """
        img: H,W,14 or 14,H,W
        return: 14,H,W float32
        """
        img = np.asarray(img, dtype=np.float32)

        if img.ndim != 3:
            raise RuntimeError(f"Image should be 3D, got shape {img.shape}")

        if img.shape[-1] == 14:
            img = np.transpose(img, (2, 0, 1))
        elif img.shape[0] == 14:
            pass
        else:
            raise RuntimeError(f"Cannot infer 14-band layout, got shape {img.shape}")

        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        # Per-band robust normalization.
        # This avoids relying on unknown absolute scales across Sentinel/Slope/DEM.
        out = np.zeros_like(img, dtype=np.float32)
        for c in range(img.shape[0]):
            band = img[c]
            p1 = np.percentile(band, 1)
            p99 = np.percentile(band, 99)
            band = np.clip(band, p1, p99)

            mean = band.mean()
            std = band.std()
            if std < 1e-6:
                std = 1.0
            out[c] = (band - mean) / std

        return out

    def _resize_if_needed(self, img, mask):
        # img: C,H,W, mask: H,W
        if img.shape[-2:] == (self.img_size, self.img_size):
            return img, mask

        img_t = torch.from_numpy(img).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()

        img_t = F.interpolate(
            img_t,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )

        mask_t = F.interpolate(
            mask_t,
            size=(self.img_size, self.img_size),
            mode="nearest",
        )

        img = img_t.squeeze(0).numpy()
        mask = mask_t.squeeze(0).squeeze(0).numpy().astype(np.int64)

        return img, mask

    def _augment(self, img, mask):
        # img: C,H,W, mask: H,W
        if random.random() < 0.5:
            img = img[:, :, ::-1].copy()
            mask = mask[:, ::-1].copy()

        if random.random() < 0.5:
            img = img[:, ::-1, :].copy()
            mask = mask[::-1, :].copy()

        if random.random() < 0.5:
            k = random.choice([1, 2, 3])
            img = np.rot90(img, k=k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k=k, axes=(0, 1)).copy()

        # Mild spectral noise
        if random.random() < 0.3:
            noise = np.random.normal(0.0, 0.02, size=img.shape).astype(np.float32)
            img = img + noise

        return img, mask

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = read_h5_first(img_path)
        mask = read_h5_first(mask_path).astype(np.int64)

        img = self._normalize_14band(img)

        if mask.ndim == 3:
            mask = np.squeeze(mask)
        mask = np.asarray(mask, dtype=np.int64)

        mask = np.where(mask > 0, 1, 0).astype(np.int64)

        img, mask = self._resize_if_needed(img, mask)

        if self.use_augmentation:
            img, mask = self._augment(img, mask)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()

        return img, mask
