"""Итоговая маска = кожа от модели + плитки с полосами, лежащие рядом с кожей.

Нужны результаты dev/ridgemap.py (<имя>_blocks.csv).
Запуск: python dev/mask_combo.py samples/private/*.jpg -> samples/private/out/combo/
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mp_test import skin_mask  # noqa: E402  (модель кожи из пробы MediaPipe)

NEAR = 0.04      # «рядом с кожей»: доля ширины кадра
THR = 0.5        # порог оценки полос

for p in map(Path, sys.argv[1:]):
    bgr = cv2.imread(str(p))
    k = 2000 / bgr.shape[1]
    view = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    skin = skin_mask(cv2.cvtColor(view, cv2.COLOR_BGR2RGB)) > 0.5
    r = int(NEAR * view.shape[1])
    near = cv2.dilate(skin.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    tex = np.zeros(skin.shape, bool)
    for row in csv.DictReader(open(p.parent / "out" / f"{p.stem}_blocks.csv")):
        if float(row["coh"]) * min(float(row["peak"]) / 0.4, 1) < THR:
            continue
        x, y, s = (int(int(row[c]) * k) for c in ("x", "y", "size"))
        tex[y:y + s, x:x + s] = True
    added = tex & near & ~skin
    final = skin | added
    tint = view.copy()
    tint[skin] = (0.5 * tint[skin] + [0, 127, 0]).astype(np.uint8)
    tint[added] = (0.5 * tint[added] + [0, 0, 127]).astype(np.uint8)
    out = p.parent / "out" / "combo"
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"{p.stem}_combo.jpg"), tint)
    print(p.name, "кожа", round(float(skin.mean()), 2), "добавлено полосами", round(float(added.mean()), 3))
