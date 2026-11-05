"""Проба удаления узора на вырезке и подготовка к проверке SourceAFIS.

P: шаг линий в пикселях (по замеру ridgemap). Варианты:
- mild:   лёгкое размытие (сигма P/4), так делают простые «замазки»;
- strong: сильное размытие (сигма P) + мелкое зерно, это наш вариант;
- *_sharp: попытка злоумышленника вернуть детали резкостью (нерезкая маска).
Запуск: python dev/suppress_test.py <фото> x y w h P -> samples/private/out/afis/
"""
import sys
from pathlib import Path
import cv2, numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from afis_prep import prep  # noqa: E402

src, x, y, w, h, P = sys.argv[1], *map(int, sys.argv[2:7])
out = Path(src).parent / "out" / "afis"
crop = cv2.imread(src)[y:y + h, x:x + w].astype(np.float32)
rng = np.random.default_rng()


def blur(img, s):
    return cv2.GaussianBlur(img, (0, 0), s)


def grain(img):
    # зерно намного мельче линий: шаг 2-4 px против P, новых линий не рисует
    n = blur(rng.normal(0, 3.0, img.shape[:2]).astype(np.float32), 0.8)
    return img + n[..., None]


def sharpen(img, s, amount):
    return img + amount * (img - blur(img, s))


variants = {
    "mild": blur(crop, P / 4),
    "strong": grain(blur(crop, P)),
}
variants["mild_sharp"] = sharpen(variants["mild"], P / 4, 4.0)
variants["strong_sharp"] = sharpen(variants["strong"], P, 8.0)
for name, img in variants.items():
    img8 = np.clip(img, 0, 255).astype(np.uint8)
    ok, jpg = cv2.imencode(".jpg", img8, [cv2.IMWRITE_JPEG_QUALITY, 92])
    dec = cv2.imdecode(jpg, 1)
    cv2.imwrite(str(out / f"{name}.png"), prep(dec))
    cv2.imwrite(str(out / f"{name}_color.jpg"), dec)
print("готово")
