"""Проба: цветовая маска кожи (YCrCb) поверх карты узора. Только для замеров."""
import csv, sys
from pathlib import Path
import cv2, numpy as np

for p in map(Path, sys.argv[1:]):
    bgr = cv2.imread(str(p))
    k = 2000 / bgr.shape[1]
    small = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    ycc = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = ycc[..., 0], ycc[..., 1].astype(int), ycc[..., 2].astype(int)
    skin = ((cr >= 140) & (cr <= 180) & (cb >= 85) & (cb <= 130) & (y > 60)).astype(np.float32)
    heat = np.zeros(small.shape[:2], np.float32)
    for r in csv.DictReader(open(p.parent / "out" / f"{p.stem}_blocks.csv")):
        x, yy, s = (int(int(r[c]) * k) for c in ("x", "y", "size"))
        frac = skin[yy:yy + s, x:x + s].mean() if s else 0
        score = float(r["coh"]) * min(float(r["peak"]) / 0.4, 1) * (1 if frac > 0.6 else 0)
        heat[yy:yy + s, x:x + s] = np.maximum(heat[yy:yy + s, x:x + s], score)
    col = cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    out = p.parent / "out" / "skin"
    out.mkdir(exist_ok=True)
    cv2.imwrite(str(out / f"{p.stem}_skin.jpg"), cv2.addWeighted(small, 0.55, col, 0.45, 0))
    print(p.name, "доля кожи в кадре", round(float(skin.mean()), 2))
