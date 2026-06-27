from pathlib import Path
from PIL import Image, ImageFilter
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomResizedCrop


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

    def _train_resize_crop(self, img, mask):
        # LoveDA 原图通常较大，随机裁剪比直接全图 resize 更利于泛化
        i, j, h, w = RandomResizedCrop.get_params(
            img,
            scale=(0.55, 1.0),
            ratio=(0.75, 1.3333),
        )

        img = TF.resized_crop(
            img,
            i,
            j,
            h,
            w,
            size=[self.img_size, self.img_size],
            interpolation=Image.BILINEAR,
        )

        mask = TF.resized_crop(
            mask,
            i,
            j,
            h,
            w,
            size=[self.img_size, self.img_size],
            interpolation=Image.NEAREST,
        )

        return img, mask

    def _val_resize(self, img, mask):
        img = TF.resize(
            img,
            [self.img_size, self.img_size],
            interpolation=Image.BILINEAR,
        )
        mask = TF.resize(
            mask,
            [self.img_size, self.img_size],
            interpolation=Image.NEAREST,
        )
        return img, mask

    def _augment_geometric(self, img, mask):
        if random.random() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        if random.random() < 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        if random.random() < 0.4:
            angle = random.choice([90, 180, 270])
            img = TF.rotate(img, angle, interpolation=Image.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)

        return img, mask

    def _augment_color(self, img):
        if random.random() < 0.8:
            brightness = random.uniform(0.75, 1.25)
            contrast = random.uniform(0.75, 1.25)
            saturation = random.uniform(0.75, 1.25)
            hue = random.uniform(-0.03, 0.03)

            img = TF.adjust_brightness(img, brightness)
            img = TF.adjust_contrast(img, contrast)
            img = TF.adjust_saturation(img, saturation)
            img = TF.adjust_hue(img, hue)

        if random.random() < 0.2:
            radius = random.uniform(0.1, 1.0)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        return img

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.train:
            img, mask = self._train_resize_crop(img, mask)
            img, mask = self._augment_geometric(img, mask)
            img = self._augment_color(img)
        else:
            img, mask = self._val_resize(img, mask)

        img = TF.to_tensor(img)
        img = TF.normalize(
            img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        mask = np.array(mask, dtype=np.int64)

        # LoveDA: 0 ignore, 1~7 valid classes
        mask = np.where(mask == 0, 255, mask - 1)
        mask = torch.from_numpy(mask).long()

        return img, mask
