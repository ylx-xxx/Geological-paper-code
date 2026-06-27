from pathlib import Path
from PIL import Image
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class LoveDADataset(Dataset):
    def __init__(self, root, split="train", img_size=512, train=True):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.train = train

        if split == "train":
            base = self.root / "Train" / "Train"
        else:
            base = self.root / "Val" / "Val"

        self.samples = []
        for domain in ["Rural", "Urban"]:
            img_dir = base / domain / "images_png"
            mask_dir = base / domain / "masks_png"
            imgs = sorted(img_dir.glob("*.png"), key=lambda x: int(x.stem))
            for img_path in imgs:
                mask_path = mask_dir / img_path.name
                if mask_path.exists():
                    self.samples.append((img_path, mask_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"No LoveDA samples found in {base}")

    def __len__(self):
        return len(self.samples)

    def _resize(self, img, mask):
        img = TF.resize(img, [self.img_size, self.img_size], interpolation=Image.BILINEAR)
        mask = TF.resize(mask, [self.img_size, self.img_size], interpolation=Image.NEAREST)
        return img, mask

    def _augment(self, img, mask):
        if random.random() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        if random.random() < 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        if random.random() < 0.5:
            angle = random.choice([90, 180, 270])
            img = TF.rotate(img, angle, interpolation=Image.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)
        return img, mask

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        img, mask = self._resize(img, mask)

        if self.train:
            img, mask = self._augment(img, mask)

        img = TF.to_tensor(img)
        img = TF.normalize(
            img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        mask = np.array(mask, dtype=np.int64)

        # LoveDA 原始标签常见为 0~7，其中 0 为 ignore，1~7 为有效类别
        # 这里转为：1~7 -> 0~6，原始0 -> 255 ignore
        mask = np.where(mask == 0, 255, mask - 1)
        mask = torch.from_numpy(mask).long()

        return img, mask
