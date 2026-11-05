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

for p in map(Path, sys.argv[1:]):
    tmp = p.parent / "out" / "js" / p.stem
    tmp.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(p))
    H, W = bgr.shape[:2]
    vw, vh = 2000, round(H * 2000 / W)
    view = cv2.resize(bgr, (vw, vh), interpolation=cv2.INTER_AREA)
    rgb_small = cv2.cvtColor(cv2.resize(view, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    r = seg.segment(mpimg(rgb_small))
    conf = sum(r.confidence_masks[i].numpy_view() for i in SKIN).astype(np.float32)
    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA).tofile(tmp / "full.rgba")
    cv2.cvtColor(view, cv2.COLOR_BGR2RGBA).tofile(tmp / "view.rgba")
    conf.tofile(tmp / "conf.f32")
    (tmp / "meta.json").write_text(json.dumps({"W": W, "H": H, "vw": vw, "vh": vh}))
    out = subprocess.run(["node", "--max-old-space-size=4096", "dev/js_run.mjs", str(tmp)],
                         capture_output=True, text=True, encoding="utf-8")
    if out.returncode:
        print(out.stderr[-2000:])
        sys.exit(1)
    print(p.name, out.stdout.strip())
    print(out.stderr.strip())
    res = np.fromfile(tmp / "out.rgba", np.uint8).reshape(H, W, 4)
    cv2.imwrite(str(tmp.parent / f"{p.stem}_js.jpg"), cv2.cvtColor(res, cv2.COLOR_RGBA2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    ov = np.fromfile(tmp / "overlay.rgba", np.uint8).reshape(vh, vw, 4)
    cv2.imwrite(str(tmp.parent / f"{p.stem}_js_map.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGBA2BGR))
    for f in ("full.rgba", "out.rgba"):
        (tmp / f).unlink()
