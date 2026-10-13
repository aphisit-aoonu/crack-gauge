"""Crack width from a crack mask, and the mm-per-pixel scale from a coin in the photo.

Width: thin the crack to a one-pixel centre line, then at each centre-line pixel the distance to the
nearest background pixel is half the local width.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import skeletonize

COIN_MM = 26.0  # Thai 10-baht coin diameter


@dataclass
class Widths:
    max_px: float
    p90_px: float
    median_px: float
    length_px: int
    centre: np.ndarray          # bool mask of the centre line
    width_map: np.ndarray       # width in px at each centre-line pixel, 0 elsewhere


def widths(mask: np.ndarray, min_length_px: int = 20) -> Widths | None:
    """None if there is no crack long enough to measure."""
    m = (mask > 0).astype(np.uint8)
    # drop tiny specks so they do not count as cracks
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) >= min_length_px:
            keep[lab == i] = 1
    if not keep.any():
        return None
    centre = skeletonize(keep.astype(bool))
    # distance from a pixel centre to the nearest background pixel centre; a line w px wide measures about w/2 + 0.5
    dist = cv2.distanceTransform(keep, cv2.DIST_L2, 5)
    w = np.maximum(2 * dist[centre] - 1, 1.0)
    width_map = np.zeros(m.shape, dtype=np.float32)
    width_map[centre] = w
    return Widths(float(w.max()), float(np.percentile(w, 90)), float(np.median(w)), int(centre.sum()), centre, width_map)


def fill_holes(m: np.ndarray, max_half_width_px: float = 16.0) -> np.ndarray:
    """On wide cracks the model can trace both edges and leave the dark middle out. Fill enclosed gaps that are
    narrow like a crack; a big island of concrete between two branching cracks is left alone."""
    closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    flood = np.pad(closed, 1).copy()
    cv2.floodFill(flood, None, (0, 0), 1)
    holes = (flood[1:-1, 1:-1] == 0).astype(np.uint8)
    depth = cv2.distanceTransform(holes, cv2.DIST_L2, 5)
    n, lab = cv2.connectedComponents(holes, connectivity=8)
    out = closed > 0
    for i in range(1, n):
        if depth[lab == i].max() <= max_half_width_px:
            out |= lab == i
    return out.astype(np.uint8)


def refine_mask(img_bgr: np.ndarray, mask: np.ndarray, band_px: int = 6) -> np.ndarray:
    """Tighten the model's mask to the dark pixels that are the actual gap.

    The model is trained to not miss cracks, so its masks run a few pixels wide. For measuring, look in a band
    around the predicted crack and keep only pixels darker than a threshold set from that band (Otsu)."""
    m = fill_holes((mask > 0).astype(np.uint8))
    if not m.any():
        return m
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
    band = cv2.dilate(m, k) > 0
    thr, _ = cv2.threshold(gray[band].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = (gray <= thr) & band
    # keep only dark pixels that belong to a predicted crack, not unrelated stains in the band
    n, lab = cv2.connectedComponents(dark.astype(np.uint8), connectivity=8)
    touching = np.unique(lab[(m > 0) & dark])
    return np.isin(lab, touching[touching > 0]).astype(np.uint8)


def find_coin(img_bgr: np.ndarray) -> tuple[int, int, int] | None:
    """(x, y, radius) of the most coin-like circle, or None."""
    gray = cv2.medianBlur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 5)
    h, w = gray.shape
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(h, w) / 4, param1=80, param2=30,
                               minRadius=int(min(h, w) * 0.02), maxRadius=int(min(h, w) * 0.25))
    if circles is None:
        return None
    x, y, r = circles[0][0]
    coin = refine_circle(img_bgr, x, y, r)
    # Hough always offers some circle, even on bare concrete: a real coin has an edge most of the way round
    return coin if rim_coverage(colour_edges(img_bgr), *coin) >= 0.6 else None


def rim_coverage(edges: np.ndarray, x: float, y: float, r: float) -> float:
    """Fraction of the circle's circumference (in 5 degree steps) that has an edge pixel on it."""
    ys, xs = np.nonzero(edges)
    on = np.abs(np.hypot(xs - x, ys - y) - r) < 2.5
    return np.unique(np.round(np.degrees(np.arctan2(ys[on] - y, xs[on] - x)) / 5)).size / 72


def colour_edges(img_bgr: np.ndarray) -> np.ndarray:
    """Edges from every colour channel: a brass rim on grey concrete can match it in brightness but not in colour."""
    blur = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    return np.maximum.reduce([cv2.Canny(blur[:, :, k], 40, 120) for k in range(3)])


def refine_circle(img_bgr: np.ndarray, x: float, y: float, r: float) -> tuple[int, int, int]:
    """Hough is only rough (about 10% off on radius). Fit a circle to the real edge pixels near it.
    The radius feeds straight into the mm scale, so this matters."""
    edges = colour_edges(img_bgr)
    ys, xs = np.nonzero(edges)
    d = np.hypot(xs - x, ys - y)
    # Hough may have locked onto the inner disc of a two-tone coin: take the outermost well-covered ring
    rings = np.arange(int(0.8 * r), int(1.7 * r))
    cover = np.array([np.unique(np.round(np.degrees(np.arctan2(ys[np.abs(d - q) < 1.5] - y,
                                                               xs[np.abs(d - q) < 1.5] - x)) / 5)).size / 72
                      for q in rings])
    if (cover > 0.6).any():
        r = float(rings[np.nonzero(cover > 0.6)[0].max()])
    # outer rim only: a 10-baht coin is two-tone and its inner disc edge sits at about 0.7 r
    near = (d > 0.85 * r) & (d < 1.2 * r)
    if near.sum() < 20:
        return int(round(x)), int(round(y)), int(round(r))
    px, py = xs[near].astype(float), ys[near].astype(float)
    # algebraic least-squares circle: x^2 + y^2 + D x + E y + F = 0
    a = np.column_stack([px, py, np.ones_like(px)])
    b = -(px ** 2 + py ** 2)
    (dd, ee, ff), *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = -dd / 2, -ee / 2
    rr = np.sqrt(cx ** 2 + cy ** 2 - ff)
    return int(round(cx)), int(round(cy)), int(round(rr))


def mm_per_px(coin_radius_px: float) -> float:
    return COIN_MM / (2 * coin_radius_px)


# Masonry crack categories by width (BRE Digest 251), used as a plain-language guide only.
CATEGORIES = [
    (0.1, "0", "hairline", "ผิวปูน"),
    (1.0, "1", "very slight", "เล็กมาก"),
    (5.0, "2", "slight", "เล็ก"),
    (15.0, "3", "moderate", "ปานกลาง ควรให้วิศวกรดู"),
    (25.0, "4", "severe", "รุนแรง ควรให้วิศวกรดูโดยเร็ว"),
    (float("inf"), "5", "very severe", "รุนแรงมาก ควรให้วิศวกรดูทันที"),
]


def category(width_mm: float) -> tuple[str, str, str]:
    for upper, code, en, th in CATEGORIES:
        if width_mm < upper:
            return code, en, th
    return CATEGORIES[-1][1:]
