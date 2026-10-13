"""Shrink the Mendeley concrete crack images and masks, and split them into train/val/test by a fixed seed.

Usage: python -m scripts.prepare_data D:/crack-data/raw/concrete D:/crack-data/prepared
"""
import json
import random
import sys
from pathlib import Path

import cv2

LONG_SIDE = 1024
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 7


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    names = sorted(p.name for p in (src / "rgb").glob("*.jpg"))
    random.Random(SEED).shuffle(names)
    n = len(names)
    cut1, cut2 = int(n * SPLIT["train"]), int(n * (SPLIT["train"] + SPLIT["val"]))
    splits = {"train": names[:cut1], "val": names[cut1:cut2], "test": names[cut2:]}
    for split, items in splits.items():
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "masks").mkdir(parents=True, exist_ok=True)
        for name in items:
            img = cv2.imread(str(src / "rgb" / name))
            mask = cv2.imread(str(src / "BW" / name), cv2.IMREAD_GRAYSCALE)
            h, w = img.shape[:2]
            s = LONG_SIDE / max(h, w)
            size = (round(w * s), round(h * s))
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            # masks are JPEGs with soft edges: area-resize then threshold keeps thin cracks connected
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)
            mask = ((mask > 80) * 255).astype("uint8")
            stem = Path(name).stem  # some source files are .JPG
            cv2.imwrite(str(dst / split / "images" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(dst / split / "masks" / f"{stem}.png"), mask)
    (dst / "split.json").write_text(json.dumps(splits, indent=1), encoding="utf-8")
    print({k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()
