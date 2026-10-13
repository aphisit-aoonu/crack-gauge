"""Pixel metrics for thin structures.

Strict IoU punishes a prediction that is one pixel off a 3-pixel-wide crack, so crack papers also report
F1 with a small distance tolerance. Both are reported here.
"""
import cv2
import numpy as np


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = (pred | gt).sum()
    return 1.0 if union == 0 else float((pred & gt).sum() / union)


def tolerant_counts(pred: np.ndarray, gt: np.ndarray, tol: int = 2) -> tuple[int, int, int, int]:
    """(matched predicted, all predicted, matched ground truth, all ground truth) within tol pixels."""
    pred, gt = pred.astype(np.uint8), gt.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    gt_near = cv2.dilate(gt, k) > 0
    pred_near = cv2.dilate(pred, k) > 0
    return int((pred.astype(bool) & gt_near).sum()), int(pred.sum()), int((gt.astype(bool) & pred_near).sum()), int(gt.sum())


def f1_from_counts(tp_p: int, n_p: int, tp_g: int, n_g: int) -> tuple[float, float, float]:
    precision = tp_p / n_p if n_p else 1.0
    recall = tp_g / n_g if n_g else 1.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1
