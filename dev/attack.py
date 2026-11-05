"""Атаки на обработанную вырезку: пытаемся вернуть узор.

- sharp:  сильная нерезкая маска (x8, сигма P);
- wiener: обратная фильтрация Винера, злоумышленник знает размытие (гаусс сигма P);
- clahe:  агрессивное выравнивание контраста мелкими окнами.
Все варианты проходят ту же подготовку, что и оригинал (afis_prep.prep).
Запуск: python dev/attack.py <обработанное фото> x y w h P <префикс>
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from afis_prep import prep  # noqa: E402

src, x, y, w, h, P, tag = sys.argv[1], *map(int, sys.argv[2:7]), sys.argv[7]
out = Path("samples/private/out/afis")
crop = cv2.imread(src)[y:y + h, x:x + w]
g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)


def wiener(img, sigma, k):
    fy = np.fft.fftfreq(img.shape[0])[:, None]
    fx = np.fft.fftfreq(img.shape[1])[None, :]
    H = np.exp(-2 * np.pi ** 2 * sigma ** 2 * (fx ** 2 + fy ** 2))
    F = np.fft.fft2(img - img.mean())
    return np.real(np.fft.ifft2(F * H / (H * H + k))) + img.mean()


variants = {
    "plain": g,
    "sharp": g + 8 * (g - cv2.GaussianBlur(g, (0, 0), P)),
    "wiener2": wiener(g, P, 1e-2),
    "wiener4": wiener(g, P, 1e-4),
}
for name, im in variants.items():
    im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(str(out / f"{tag}_{name}.png"), prep(cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)))
cl = cv2.createCLAHE(8.0, (32, 32)).apply(prep(crop))
cv2.imwrite(str(out / f"{tag}_clahe.png"), cl)
print("готово", tag)
