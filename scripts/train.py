"""Train the crack U-Net. Keeps the checkpoint with the best validation F1 (2 px tolerance).

Usage: python -m scripts.train D:/crack-data/prepared D:/crack-data/runs/unet34
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crackgauge.data import CrackDataset
from crackgauge.metrics import f1_from_counts, iou, tolerant_counts
from crackgauge.model import CrackUNet

EPOCHS = 50
# fp16 autocast gives NaN on GTX 16-series cards, so mixed precision stays off there
AMP = False
BATCH = 4
LR = 3e-4
SEED = 7


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps)).mean()


@torch.no_grad()
def evaluate(model, loader, dev, threshold=0.5):
    model.eval()
    counts, ious = np.zeros(4), []
    for img, mask in loader:
        with torch.autocast(dev, dtype=torch.float16, enabled=AMP and dev == "cuda"):
            prob = torch.sigmoid(model(img.to(dev))).float().cpu().numpy()[0, 0]
        gt = mask.numpy()[0, 0] > 0.5
        pred = prob > threshold
        counts += tolerant_counts(pred, gt)
        ious.append(iou(pred, gt))
    p, r, f1 = f1_from_counts(*counts)
    return {"precision": p, "recall": r, "f1_tol2": f1, "iou": float(np.mean(ious))}


def main() -> None:
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train = DataLoader(CrackDataset(data / "train"), batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True)
    val = DataLoader(CrackDataset(data / "val", crop=None, train=False), batch_size=1, num_workers=1)
    model = CrackUNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * len(train))
    scaler = torch.amp.GradScaler(enabled=AMP and dev == "cuda")
    pos_weight = torch.tensor([5.0], device=dev)  # cracks are ~1% of pixels
    best, log = -1.0, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, total = time.time(), 0.0
        for img, mask in train:
            img, mask = img.to(dev), mask.to(dev)
            with torch.autocast(dev, dtype=torch.float16, enabled=AMP and dev == "cuda"):
                logits = model(img)
                loss = F.binary_cross_entropy_with_logits(logits, mask, pos_weight=pos_weight) + dice_loss(logits.float(), mask)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        m = evaluate(model, val, dev)
        m.update(epoch=epoch, loss=total / len(train), seconds=round(time.time() - t0, 1))
        log.append(m)
        if m["f1_tol2"] > best:
            best = m["f1_tol2"]
            torch.save(model.state_dict(), out / "best.pt")
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}), flush=True)
        (out / "log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"best val F1 (2px) {best:.4f}")


if __name__ == "__main__":
    main()
