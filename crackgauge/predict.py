"""Run the trained model on one photo."""
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import pad_to, to_tensor
from .model import CrackUNet

LONG_SIDE = 1024
# picked on the validation photos (scripts.evaluate); the reported test numbers use the same value


class Segmenter:
    def __init__(self, weights: str | Path, threshold: float = 0.3, device: str | None = None):
        self.dev = device or ("cuda" if torch.cuda.device_count() > 0 else "cpu")
        self.model = CrackUNet(pretrained=False).to(self.dev).eval()
        self.model.load_state_dict(torch.load(weights, map_location=self.dev, weights_only=True))
        self.threshold = threshold

    @torch.no_grad()
    def probability(self, img_bgr: np.ndarray) -> np.ndarray:
        """Crack probability per pixel, same size as the input."""
        h, w = img_bgr.shape[:2]
        s = min(1.0, LONG_SIDE / max(h, w))
        small = cv2.resize(img_bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img_bgr
        padded, (sh, sw) = pad_to(small)
        prob = torch.sigmoid(self.model(to_tensor(padded)[None].to(self.dev)))[0, 0].cpu().numpy()[:sh, :sw]
        return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR) if s < 1 else prob

    def mask(self, img_bgr: np.ndarray) -> np.ndarray:
        return (self.probability(img_bgr) > self.threshold).astype(np.uint8)
