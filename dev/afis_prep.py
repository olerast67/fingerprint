"""Готовит вырезки подушечки для SourceAFIS: серый, выравнивание света, инверсия.

Делает: orig (эталон), other (тот же палец после поворота, уменьшения и JPEG,
как будто другой снимок), mirror (зеркальная копия: заведомо чужой узор).
Запуск: python dev/afis_prep.py <фото> x y w h  -> samples/private/out/afis/
"""
import sys
from pathlib import Path
import cv2, numpy as np



def prep(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = g - cv2.GaussianBlur(g, (0, 0), 25)
    g = 255 - cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)   # бороздки светлые, гребни тёмные
    g = cv2.createCLAHE(2.0, (8, 8)).apply(g.astype(np.uint8))
    return g


if __name__ == "__main__":
    src, x, y, w, h = sys.argv[1], *map(int, sys.argv[2:6])
    out = Path(src).parent / "out" / "afis"
    out.mkdir(parents=True, exist_ok=True)
    crop = cv2.imread(src)[y:y + h, x:x + w]
    cv2.imwrite(str(out / "orig.png"), prep(crop))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 6, 0.95)
    other = cv2.warpAffine(crop, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    ok, jpg = cv2.imencode(".jpg", other, [cv2.IMWRITE_JPEG_QUALITY, 70])
    cv2.imwrite(str(out / "other.png"), prep(cv2.imdecode(jpg, 1)))
    cv2.imwrite(str(out / "mirror.png"), prep(crop[:, ::-1]))
    print("готово:", out)
