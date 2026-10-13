"""Copy a few held-out test photos into web/examples, plus one with a drawn coin to show the mm scale.

The coin in coin.jpg is drawn, not photographed: it shows how the scale works until a real photo with a coin
is added. The page and README say so.

Usage: python -m scripts.make_examples D:/crack-data/prepared
"""
import sys
from pathlib import Path

import cv2

PICK = 3
COIN_R = 65  # px on a 1024-wide close-up, roughly what a 26 mm coin is when the photo covers ~20 cm


def main() -> None:
    test = Path(sys.argv[1]) / "test"
    out = Path("web/examples")
    out.mkdir(parents=True, exist_ok=True)
    photos = sorted((test / "images").glob("*.jpg"))
    step = len(photos) // (PICK + 1)
    chosen = [photos[step * (i + 1)] for i in range(PICK)]
    for i, p in enumerate(chosen, 1):
        cv2.imwrite(str(out / f"{i}.jpg"), cv2.imread(str(p)), [cv2.IMWRITE_JPEG_QUALITY, 90])
    img = cv2.imread(str(chosen[0]))
    mask = cv2.imread(str(test / "masks" / f"{chosen[0].stem}.png"), cv2.IMREAD_GRAYSCALE)
    h, w = mask.shape
    # put the coin in the corner farthest from the crack
    corners = [(COIN_R + 30, COIN_R + 30), (w - COIN_R - 30, COIN_R + 30),
               (COIN_R + 30, h - COIN_R - 30), (w - COIN_R - 30, h - COIN_R - 30)]
    ys, xs = (mask > 0).nonzero()
    c = max(corners, key=lambda q: ((xs - q[0]) ** 2 + (ys - q[1]) ** 2).min())
    # two-tone like a real 10-baht coin: brass ring, silver centre about 0.7 of the diameter
    cv2.circle(img, c, COIN_R, (70, 150, 190), -1)
    cv2.circle(img, c, int(COIN_R * 0.7), (185, 190, 192), -1)
    cv2.putText(img, "10", (c[0] - 22, c[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (120, 125, 128), 3)
    cv2.imwrite(str(out / "coin.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print([p.name for p in chosen], "coin at", c)


if __name__ == "__main__":
    main()
