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


def get_channel_indices(channel_mode):
    channel_mode = str(channel_mode).lower()

    if channel_mode == "rgb":
        return [0, 1, 2]

    if channel_mode == "ms12":
        return list(range(12))

    if channel_mode in ["rgb_topo", "rgb_dem_slope", "rgb_slope_dem"]:
        return [0, 1, 2, 12, 13]

    if channel_mode == "full14":
        return list(range(14))

    raise ValueError(
        f"Unsupported channel_mode={channel_mode}. "
        f"Use rgb, ms12, rgb_topo, or full14."
    )


def get_in_chans(channel_mode):
    return len(get_channel_indices(channel_mode))


class Landslide4SenseAblationDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        img_size=128,
        train=True,
        use_augmentation=True,
        channel_mode="full14",
    ):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.train = train
        self.use_augmentation = use_augmentation and train
        self.channel_mode = channel_mode
        self.channel_indices = get_channel_indices(channel_mode)

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

        if random.random() < 0.3:
            noise = np.random.normal(0.0, 0.02, size=img.shape).astype(np.float32)
            img = img + noise

        return img, mask

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = read_h5_first(img_path)
        mask = read_h5_first(mask_path).astype(np.int64)

        img = self._normalize_14band(img)
        img = img[self.channel_indices]

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
