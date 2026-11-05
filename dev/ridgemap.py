"""Карта «похоже на папиллярный узор» по всему кадру (прототип слоя 2).

Для каждой плитки 128x128 (шаг 64) на нескольких уменьшениях кадра считаем:
- coh: когерентность, идут ли полосы в одну сторону (0..1);
- peak: во сколько раз самый сильный пик спектра в полосе периодов 5..20 px
  выше медианы этой полосы;
- period: шаг полос в пикселях полного кадра.

Запуск: python dev/ridgemap.py samples/private/*.jpg
Результат: samples/private/out/<имя>_map.jpg и <имя>_blocks.csv (в git не идут).
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

B, STEP = 128, 64
SCALES = (1, 2, 4)          # во сколько раз уменьшаем кадр
PMIN, PMAX = 5.0, 20.0      # периоды в пикселях уменьшенного кадра

fy, fx = np.meshgrid(np.fft.fftfreq(B), np.fft.fftfreq(B), indexing="ij")
RAD = np.hypot(fx, fy)
BAND = (RAD >= 1 / PMAX) & (RAD <= 1 / PMIN)
HANN = np.outer(np.hanning(B), np.hanning(B)).astype(np.float32)


def analyse(gray: np.ndarray, d: int):
    img = cv2.resize(gray, None, fx=1 / d, fy=1 / d, interpolation=cv2.INTER_AREA) if d > 1 else gray
    img = img.astype(np.float32)
    # убираем неравномерный свет: вычитаем сильно размытую копию
    hp = img - cv2.GaussianBlur(img, (0, 0), 8)
    # выравниваем контраст: резкие края не должны перекрикивать мелкую текстуру
    sd = np.sqrt(cv2.GaussianBlur(hp * hp, (0, 0), 8)) + 1.0
    hp = np.clip(hp / sd, -2, 2)
    g = cv2.GaussianBlur(hp, (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    h, w = img.shape
    rows = []
    for y in range(0, h - B + 1, STEP):
        for x in range(0, w - B + 1, STEP):
            bx, by = gx[y:y + B, x:x + B], gy[y:y + B, x:x + B]
            gxx, gyy, gxy = (bx * bx).sum(), (by * by).sum(), (bx * by).sum()
            den = gxx + gyy
            if den < 1e-3:
                continue
            coh = float(np.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2) / den)
            blk = hp[y:y + B, x:x + B]
            blk = (blk - blk.mean()) * HANN
            mag = np.abs(np.fft.fft2(blk))
            pw = mag * mag * BAND
            iy, ix = np.unravel_index(int(np.argmax(pw)), pw.shape)
            # доля энергии полосы в пятне вокруг пика и его зеркала:
            # у полос она высокая, у одиночного края размазана по линии
            near = (np.minimum(np.abs(np.arange(B)[:, None] - iy), B - np.abs(np.arange(B)[:, None] - iy)) ** 2
                    + np.minimum(np.abs(np.arange(B)[None, :] - ix), B - np.abs(np.arange(B)[None, :] - ix)) ** 2) <= 4
            my, mx = (-iy) % B, (-ix) % B
            near |= (np.minimum(np.abs(np.arange(B)[:, None] - my), B - np.abs(np.arange(B)[:, None] - my)) ** 2
                     + np.minimum(np.abs(np.arange(B)[None, :] - mx), B - np.abs(np.arange(B)[None, :] - mx)) ** 2) <= 4
            peak = float(pw[near].sum() / (pw.sum() + 1e-6))
            period = float(1 / RAD[iy, ix]) * d
            contrast = float(blk.std())
            rows.append((x * d, y * d, B * d, d, coh, peak, period, contrast))
    return rows


def main(paths):
    for p in map(Path, paths):
        bgr = cv2.imread(str(p))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        out = p.parent / "out"
        out.mkdir(exist_ok=True)
        allrows = []
        for d in SCALES:
            allrows += analyse(gray, d)
        with open(out / f"{p.stem}_blocks.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["x", "y", "size", "scale", "coh", "peak", "period_full_px", "contrast"])
            wr.writerows(allrows)
        # карта: красный = высокая когерентность и чёткий пик
        k = 2000 / bgr.shape[1]
        small = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        heat = np.zeros(small.shape[:2], np.float32)
        for x, y, s, d, coh, peak, period, c in allrows:
            score = coh * min(peak / 0.4, 1.0)
            x0, y0, x1, y1 = (int(v * k) for v in (x, y, x + s, y + s))
            heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1], score)
        col = cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(str(out / f"{p.stem}_map.jpg"), cv2.addWeighted(small, 0.55, col, 0.45, 0))
        top = sorted(allrows, key=lambda r: -r[4] * min(r[5] / 0.4, 1))[:5]
        print(p.name, len(allrows), "блоков; лучшие (x,y,scale,coh,peak,period):")
        for r in top:
            print("  ", r[0], r[1], r[3], round(r[4], 2), round(r[5], 2), round(r[6], 1))


if __name__ == "__main__":
    main(sys.argv[1:])
