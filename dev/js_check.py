"""Прогоняет браузерное ядро (web/lib/core.js) через Node на фото и сохраняет результат.

Готовит сырые данные (полный кадр RGBA, рабочая копия 2000 px, маска кожи 256x256
от MediaPipe), запускает dev/js_run.mjs, собирает JPEG и карту.
Запуск: python dev/js_check.py samples/private/*.jpg -> samples/private/out/js/
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mp_test import seg, mpimg, SKIN  # noqa: E402


def skin_conf(rgb, vw, vh):
    """Как skinConfidence в web/app.js: кадр с полями до квадрата и куски 0,6 короткой стороны с шагом в половину."""
    H, W = rgb.shape[:2]
    kx = vw / W
    conf = np.zeros((vh, vw), np.float32)

    def run(sx, sy, side):
        canvas = np.zeros((side, side, 3), np.uint8)
        ix, iy = max(0, sx), max(0, sy)
        ex, ey = min(W, sx + side), min(H, sy + side)
        canvas[iy - sy:ey - sy, ix - sx:ex - sx] = rgb[iy:ey, ix:ex]
        small = cv2.resize(canvas, (256, 256), interpolation=cv2.INTER_AREA)
        r = seg.segment(mpimg(small))
        c = np.squeeze(sum(r.confidence_masks[i].numpy_view() for i in SKIN)).astype(np.float32)
        x0, x1 = int(ix * kx), min(vw, int(np.ceil(ex * kx)))
        y0, y1 = int(iy * kx), min(vh, int(np.ceil(ey * kx)))
        if x1 <= x0 or y1 <= y0:
            return
        xs = np.clip((((np.arange(x0, x1) + 0.5) / kx - sx) * 256 / side).astype(int), 0, 255)
        ys = np.clip((((np.arange(y0, y1) + 0.5) / kx - sy) * 256 / side).astype(int), 0, 255)
        conf[y0:y1, x0:x1] = np.maximum(conf[y0:y1, x0:x1], c[np.ix_(ys, xs)])

    side = max(W, H)
    run((W - side) // 2, (H - side) // 2, side)
    t = round(min(W, H) * 0.6)
    step = round(t / 2)

    def pos(n):
        a = list(range(0, max(0, n - t), step))
        a.append(max(0, n - t))
        return a

    for y in pos(H):
        for x in pos(W):
            run(x, y, t)
    return conf


KEEP = "--keep-far" in sys.argv
for p in map(Path, [a for a in sys.argv[1:] if not a.startswith("--")]):
    tmp = p.parent / "out" / "js" / p.stem
    tmp.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(p))
    H, W = bgr.shape[:2]
    vw, vh = 2000, round(H * 2000 / W)
    view = cv2.resize(bgr, (vw, vh), interpolation=cv2.INTER_AREA)
    conf = skin_conf(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), vw, vh)
    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA).tofile(tmp / "full.rgba")
    cv2.cvtColor(view, cv2.COLOR_BGR2RGBA).tofile(tmp / "view.rgba")
    conf.tofile(tmp / "conf.f32")
    (tmp / "meta.json").write_text(json.dumps({"W": W, "H": H, "vw": vw, "vh": vh}))
    out = subprocess.run(["node", "--max-old-space-size=4096", "dev/js_run.mjs", str(tmp)] + (["keepFar"] if KEEP else []),
                         capture_output=True, text=True, encoding="utf-8")
    if out.returncode:
        print(out.stderr[-2000:])
        sys.exit(1)
    print(p.name, out.stdout.strip())
    print(out.stderr.strip())
    res = np.fromfile(tmp / "out.rgba", np.uint8).reshape(H, W, 4)
    cv2.imwrite(str(tmp.parent / f"{p.stem}_js.jpg"), cv2.cvtColor(res, cv2.COLOR_RGBA2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    alpha = np.fromfile(tmp / "alpha.f32", np.float32).reshape(vh, vw)
    ov = cv2.resize(cv2.cvtColor(res, cv2.COLOR_RGBA2BGR), (vw, vh), interpolation=cv2.INTER_AREA).astype(np.float32)
    m = alpha > 0.5
    ov[m] = ov[m] * 0.45 + np.array([119, 132, 35], np.float32)
    for x0, y0, x1, y1 in json.loads((tmp / "residual.json").read_text()):
        ov[y0:y1, x0:x1] = (136, 46, 255)
    cv2.imwrite(str(tmp.parent / f"{p.stem}_js_map.jpg"), ov.astype(np.uint8))
    for f in ("full.rgba", "out.rgba"):
        (tmp / f).unlink()
