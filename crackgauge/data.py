"""Image/mask pairs for training and evaluation."""
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1).copy())


def pad_to(img: np.ndarray, multiple: int = 32) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = img.shape[:2]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    return img, (h, w)


class CrackDataset(Dataset):
    def __init__(self, split_dir: str | Path, crop: int | None = 512, train: bool = True):
        self.images = sorted((Path(split_dir) / "images").glob("*.jpg"))
        self.crop, self.train = crop, train

    def __len__(self):
        return len(self.images)

    def load(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        p = self.images[i]
        img = cv2.imread(str(p))
        mask = cv2.imread(str(p.parent.parent / "masks" / (p.stem + ".png")), cv2.IMREAD_GRAYSCALE)
        return img, (mask > 127).astype(np.float32)

    def __getitem__(self, i: int):
        img, mask = self.load(i)
        if self.train:
            h, w = mask.shape
            # prefer crops that contain crack pixels, so the model sees enough positives
            for _ in range(10):
                y, x = random.randint(0, h - self.crop), random.randint(0, w - self.crop)
                if mask[y:y + self.crop, x:x + self.crop].sum() > 50:
                    break
            img, mask = img[y:y + self.crop, x:x + self.crop], mask[y:y + self.crop, x:x + self.crop]
            if random.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            if random.random() < 0.5:
                img, mask = img[::-1], mask[::-1]
            if random.random() < 0.5:
                img, mask = np.rot90(img), np.rot90(mask)
            gain = random.uniform(0.75, 1.25)
            img = np.clip(img.astype(np.float32) * gain + random.uniform(-20, 20), 0, 255).astype(np.uint8)
        else:
            img, _ = pad_to(img)
            mask, _ = pad_to(mask)
        return to_tensor(np.ascontiguousarray(img)), torch.from_numpy(np.ascontiguousarray(mask))[None]
