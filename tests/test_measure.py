import cv2
import numpy as np
import pytest

from crackgauge.measure import category, find_coin, mm_per_px, refine_mask, widths
from crackgauge.metrics import f1_from_counts, iou, tolerant_counts


def band(w: float, angle_deg: float = 0.0, size: int = 400) -> np.ndarray:
    """A straight band exactly w pixels wide (cv2.line draws thicker than asked, so it is not used here)."""
    rect = ((size / 2, size / 2), (size * 0.8, w), angle_deg)
    m = np.zeros((size, size), np.uint8)
    cv2.fillPoly(m, [np.round(cv2.boxPoints(rect)).astype(np.int32)], 1)
    return m


@pytest.mark.parametrize("w", [3, 5, 9, 15])
def test_width_of_a_straight_band(w):
    m = np.zeros((200, 400), np.uint8)
    m[100:100 + w, 40:360] = 1
    assert widths(m).median_px == pytest.approx(w, abs=1.0)


@pytest.mark.parametrize("w", [5, 9])
def test_width_of_a_slanted_band(w):
    assert widths(band(w, 37)).median_px == pytest.approx(w, abs=1.5)


def test_specks_are_ignored():
    m = np.zeros((100, 100), np.uint8)
    m[50, 50] = 1
    assert widths(m) is None


def test_coin_is_found_and_scales_to_mm():
    img = np.full((600, 800, 3), 160, np.uint8)
    cv2.circle(img, (600, 400), 60, (90, 90, 90), -1)
    x, y, r = find_coin(img)
    assert (x, y) == pytest.approx((600, 400), abs=4)
    assert r == pytest.approx(60, abs=4)
    assert mm_per_px(r) == pytest.approx(26 / 120, rel=0.08)


def test_categories():
    assert category(0.05)[0] == "0"
    assert category(0.5)[0] == "1"
    assert category(3)[0] == "2"
    assert category(10)[0] == "3"
    assert category(30)[0] == "5"


def test_metrics():
    gt = np.zeros((50, 50), bool)
    gt[25, 5:45] = True
    shifted = np.roll(gt, 1, axis=0)
    assert iou(shifted, gt) == 0.0
    p, r, f1 = f1_from_counts(*tolerant_counts(shifted, gt, tol=2))
    assert f1 == pytest.approx(1.0)


def test_refine_mask_shrinks_an_overwide_mask_to_the_dark_gap():
    img = np.full((200, 400, 3), 170, np.uint8)
    img[96:104, 40:360] = 40                      # dark gap 8 px wide
    loose = np.zeros((200, 400), np.uint8)
    loose[92:108, 40:360] = 1                     # model mask 16 px wide
    assert widths(loose).median_px == pytest.approx(16, abs=1)
    assert widths(refine_mask(img, loose)).median_px == pytest.approx(8, abs=1)


def test_hollow_trace_of_a_wide_crack_is_filled_before_measuring():
    img = np.full((200, 400, 3), 170, np.uint8)
    img[90:110, 40:360] = 40                       # 20 px dark gap
    hollow = np.zeros((200, 400), np.uint8)
    hollow[88:93, 38:362] = 1                      # model traced only the two edges
    hollow[107:112, 38:362] = 1
    hollow[88:112, 38:43] = 1
    hollow[88:112, 357:362] = 1
    assert widths(refine_mask(img, hollow)).median_px == pytest.approx(20, abs=2)


def test_two_tone_coin_uses_the_outer_rim():
    img = np.full((600, 800, 3), 160, np.uint8)
    cv2.circle(img, (300, 300), 70, (70, 150, 190), -1)
    cv2.circle(img, (300, 300), 49, (185, 190, 192), -1)
    assert find_coin(img)[2] == pytest.approx(70, abs=3)


def test_no_coin_is_reported_on_a_faint_arc():
    # Hough will offer a circle for a partial arc; a coin needs an edge most of the way round
    img = np.full((400, 400, 3), 170, np.uint8)
    cv2.ellipse(img, (200, 200), (80, 80), 0, 0, 60, (60, 60, 60), 2)
    assert find_coin(img) is None


def test_island_between_two_cracks_is_not_filled():
    from crackgauge.measure import fill_holes
    m = np.zeros((300, 300), np.uint8)
    cv2.ellipse(m, (150, 150), (100, 60), 0, 0, 360, 1, 3)      # two branches around a patch of concrete
    assert fill_holes(m)[150, 150] == 0
    thin = np.zeros((300, 300), np.uint8)
    cv2.rectangle(thin, (50, 140), (250, 160), 1, 2)             # hollow middle of one wide crack
    assert fill_holes(thin)[150, 150] == 1
