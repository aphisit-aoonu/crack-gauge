"""Score the trained model on the held-out test photos it never saw. Threshold is picked on validation only.

Usage: python -m scripts.evaluate D:/crack-data/prepared D:/crack-data/runs/unet34/best.pt
Writes results/test_metrics.json
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from crackgauge.metrics import f1_from_counts, iou, tolerant_counts
from crackgauge.predict import Segmenter

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def score(seg: Segmenter, split_dir: Path) -> tuple[dict, list[float]]:
    probs, gts, secs = [], [], []
    for p in sorted((split_dir / "images").glob("*.jpg")):
        img = cv2.imread(str(p))
        gt = cv2.imread(str(split_dir / "masks" / f"{p.stem}.png"), cv2.IMREAD_GRAYSCALE) > 127
        t = time.time()
        probs.append(seg.probability(img))
        secs.append(time.time() - t)
        gts.append(gt)
    out = {}
    for th in THRESHOLDS:
        counts, ious = np.zeros(4), []
        for pr, gt in zip(probs, gts):
            pred = pr > th
            counts += tolerant_counts(pred, gt)
            ious.append(iou(pred, gt))
        p, r, f1 = f1_from_counts(*counts)
        out[th] = {"precision": p, "recall": r, "f1_tol2": f1, "iou": float(np.mean(ious))}
    return out, secs


def main() -> None:
    data, weights = Path(sys.argv[1]), sys.argv[2]
    seg = Segmenter(weights)
    val, _ = score(seg, data / "val")
    best_th = max(val, key=lambda t: val[t]["f1_tol2"])
    test, secs = score(seg, data / "test")
    result = {"threshold_from_val": best_th, "val": val[best_th], "test": test[best_th],
              "test_images": len(secs), "seconds_per_image": float(np.median(secs[1:]))}
    Path("results").mkdir(exist_ok=True)
    Path("results/test_metrics.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
