"""Check the millimetre measurement end to end on cracks of known width.

Real crack photos do not come with true widths, so this draws dark crack-like lines of exact width onto real
concrete taken from crack-free parts of the test photos, adds a coin-sized disc, then runs the full pipeline:
model -> mask -> width in pixels -> coin scale -> millimetres. It is a synthetic check and is reported as one.

Usage: python -m scripts.validate_width D:/crack-data/prepared D:/crack-data/runs/unet34/best.pt
Writes results/width_check.json and results/width_check.jpg
"""
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from crackgauge.measure import COIN_MM, find_coin, mm_per_px, refine_mask, widths
from crackgauge.predict import Segmenter

SIZE = 640
COIN_R = 60                    # px, so 1 px = 26 / 120 mm = 0.217 mm
TRUE_WIDTHS_PX = [4, 6, 9, 14, 20]
SEED = 3


def clean_patches(split_dir: Path, n: int) -> list[np.ndarray]:
    rng = random.Random(SEED)
    out = []
    for p in sorted((split_dir / "images").glob("*.jpg")):
        img = cv2.imread(str(p))
        m = cv2.imread(str(split_dir / "masks" / f"{p.stem}.png"), cv2.IMREAD_GRAYSCALE)
        h, w = m.shape
        for _ in range(30):
            y, x = rng.randint(0, h - SIZE // 2), rng.randint(0, w - SIZE // 2)
            if m[y:y + SIZE // 2, x:x + SIZE // 2].max() == 0:
                out.append(cv2.resize(img[y:y + SIZE // 2, x:x + SIZE // 2], (SIZE, SIZE)))
                break
        if len(out) == n:
            break
    return out


def draw_crack(img: np.ndarray, width_px: int, rng: random.Random) -> np.ndarray:
    """A wandering dark band of exact width, made of short straight pieces."""
    mask = np.zeros(img.shape[:2], np.uint8)
    y = rng.randint(200, 440)
    pts = [(40, y)]
    while pts[-1][0] < SIZE - 220:
        x0, y0 = pts[-1]
        pts.append((x0 + rng.randint(25, 45), int(np.clip(y0 + rng.randint(-14, 14), 80, SIZE - 80))))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        angle = np.degrees(np.arctan2(y1 - y0, x1 - x0))
        length = np.hypot(x1 - x0, y1 - y0) + width_px
        box = cv2.boxPoints((((x0 + x1) / 2, (y0 + y1) / 2), (length, width_px), angle))
        cv2.fillPoly(mask, [np.round(box).astype(np.int32)], 1)
    out = img.copy()
    shade = rng.randint(25, 60)
    out[mask > 0] = (out[mask > 0] * 0.25 + shade).astype(np.uint8)
    return out


def add_coin(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    c = (SIZE - 100, SIZE - 100)
    cv2.circle(out, c, COIN_R, (150, 160, 170), -1)
    cv2.circle(out, c, COIN_R - 8, (120, 130, 140), 3)
    return out


def main() -> None:
    data, weights = Path(sys.argv[1]), sys.argv[2]
    seg = Segmenter(weights)
    rng = random.Random(SEED)
    patches = clean_patches(data / "test", 8)
    rows, tiles = [], []
    for k, base in enumerate(patches):
        for w in TRUE_WIDTHS_PX:
            img = add_coin(draw_crack(base, w, rng))
            coin = find_coin(img)
            scale = mm_per_px(coin[2]) if coin else None
            mask = seg.mask(img)
            mask[img.shape[0] - 200:, img.shape[1] - 200:] = 0      # ignore the coin corner
            raw = widths(mask)
            mask = refine_mask(img, mask)
            r = widths(mask)
            true_mm = w * COIN_MM / (2 * COIN_R)
            got_mm = (r.median_px * scale) if (r and scale) else None
            rows.append({"patch": k, "true_px": w, "true_mm": round(true_mm, 3),
                         "measured_mm": None if got_mm is None else round(got_mm, 3),
                         "unrefined_mm": round(raw.median_px * scale, 3) if (raw and scale) else None,
                         "coin_radius_px": coin[2] if coin else None})
            if k == 0:
                vis = img.copy()
                vis[mask > 0] = (0, 0, 255)
                cv2.putText(vis, f"true {true_mm:.2f} mm  measured {got_mm or 0:.2f} mm", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                tiles.append(cv2.resize(vis, (400, 400)))
    found = [r for r in rows if r["measured_mm"] is not None]
    by_width = {}
    for w in TRUE_WIDTHS_PX:
        rs = [r for r in found if r["true_px"] == w]
        errs = [abs(r["measured_mm"] - r["true_mm"]) for r in rs]
        by_width[f"{w * COIN_MM / (2 * COIN_R):.2f}"] = {
            "n": len(rs), "mean_abs_error_mm": float(np.mean(errs)) if errs else None,
            "mean_measured_mm": float(np.mean([r["measured_mm"] for r in rs])) if rs else None}
    unrefined = [abs(r["unrefined_mm"] - r["true_mm"]) for r in rows if r["unrefined_mm"] is not None]
    summary = {"cases": len(rows), "measured": len(found),
               "mean_abs_error_mm_without_refinement": float(np.mean(unrefined)) if unrefined else None,
               "mean_abs_error_mm": float(np.mean([abs(r["measured_mm"] - r["true_mm"]) for r in found])),
               "by_true_width_mm": by_width, "rows": rows}
    Path("results").mkdir(exist_ok=True)
    Path("results/width_check.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    cv2.imwrite("results/width_check.jpg", np.hstack(tiles))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
