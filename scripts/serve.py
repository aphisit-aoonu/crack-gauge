"""Local web app: upload a crack photo (ideally with a 10-baht coin in it) and get the crack traced and measured.

Usage: python -m scripts.serve D:/crack-data/runs/unet34/best.pt   -> http://127.0.0.1:8820
"""
import base64
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from crackgauge.measure import category, find_coin, mm_per_px, refine_mask, widths
from crackgauge.predict import Segmenter

WEB = Path(__file__).resolve().parent.parent / "web"
MAX_SIDE = 2048


def overlay(img: np.ndarray, mask: np.ndarray, width_map: np.ndarray | None, coin) -> np.ndarray:
    """Crack in red, widest stretch highlighted, coin outlined."""
    out = img.copy()
    tint = out.copy()
    tint[mask > 0] = (40, 40, 230)
    out = cv2.addWeighted(tint, 0.75, out, 0.25, 0)
    if width_map is not None and width_map.max() > 0:
        y, x = np.unravel_index(np.argmax(width_map), width_map.shape)
        r = max(12, int(width_map.max() * 2))
        cv2.circle(out, (int(x), int(y)), r, (0, 220, 255), 3)
    if coin:
        cv2.circle(out, coin[:2], coin[2], (90, 230, 90), 3)
    return out


def analyse(img: np.ndarray, seg: Segmenter) -> dict:
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    t = time.time()
    raw = seg.mask(img)
    coin = find_coin(img)
    if coin:   # the coin rim can look like a curved crack
        cv2.circle(raw, coin[:2], int(coin[2] * 1.15), 0, -1)
    mask = refine_mask(img, raw)
    wd = widths(mask)
    result = {"seconds": round(time.time() - t, 2), "coin": coin is not None, "found": wd is not None}
    if wd:
        result.update(length_px=wd.length_px, max_px=round(wd.max_px, 1), p90_px=round(wd.p90_px, 1),
                      median_px=round(wd.median_px, 1))
        if coin:
            scale = mm_per_px(coin[2])
            p90_mm = wd.p90_px * scale
            code, en, th = category(p90_mm)
            result.update(mm_per_px=round(scale, 4), max_mm=round(wd.max_px * scale, 2), p90_mm=round(p90_mm, 2),
                          median_mm=round(wd.median_px * scale, 2), length_mm=round(wd.length_px * scale),
                          category=code, category_en=en, category_th=th)
    vis = overlay(img, mask, wd.width_map if wd else None, coin)
    ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    result["image"] = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    return result


def make_app(weights: str) -> FastAPI:
    seg = Segmenter(weights)
    app = FastAPI()

    @app.post("/api/measure")
    async def measure(file: UploadFile = File(...)):
        data = np.frombuffer(await file.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse({"error": "not an image"}, status_code=400)
        return analyse(img, seg)

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    app.mount("/", StaticFiles(directory=WEB), name="web")
    return app


if __name__ == "__main__":
    uvicorn.run(make_app(sys.argv[1]), host="127.0.0.1", port=8820, log_level="warning")
