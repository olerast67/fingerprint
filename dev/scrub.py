"""Эталонная обработка на Python: удаляет папиллярный узор с кожи на всём кадре.

Шаги:
1. Маска: кожа от MediaPipe + плитки с полосами рядом с кожей (как mask_combo).
2. Край маски уточняем GrabCut (отделяет предмет от фона по цвету) и
   направленным фильтром (guided filter): маска прилипает к контуру пальца.
3. Шаг линий P: медиана по плиткам внутри кожи, вдали от контура.
4. Гладкую кожу считаем только по пикселям кожи (нормированная свёртка),
   поэтому фон не затекает в палец. Размытие с сигмой P на уменьшенной копии.
5. Внутри маски заменяем пиксели гладкой кожей и добавляем мелкое зерно.
6. Собираем новый JPEG из пикселей: метаданные оригинала не переносятся.

Нужны результаты dev/ridgemap.py (<имя>_blocks.csv).
Запуск: python dev/scrub.py samples/private/*.jpg -> samples/private/out/scrub/
"""
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mp_test import skin_mask  # noqa: E402

NEAR, THR = 0.04, 0.5
MODE = "simple"   # "grabcut" или "simple" (переносится в браузер)


def ridge_blocks(p: Path):
    rows = []
    for r in csv.DictReader(open(p.parent / "out" / f"{p.stem}_blocks.csv")):
        if float(r["coh"]) * min(float(r["peak"]) / 0.4, 1) >= THR:
            rows.append((int(r["x"]), int(r["y"]), int(r["size"]), float(r["period_full_px"])))
    return rows


def guided(I, p, r, eps):
    box = lambda a: cv2.boxFilter(a, -1, (2 * r + 1, 2 * r + 1))
    mI, mp_ = box(I), box(p)
    a = (box(I * p) - mI * mp_) / (box(I * I) - mI * mI + eps)
    b = mp_ - a * mI
    return box(a) * I + box(b)


def build_mask(bgr, blocks):
    H, W = bgr.shape[:2]
    k = 2000 / W
    view = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    skin = skin_mask(cv2.cvtColor(view, cv2.COLOR_BGR2RGB)) > 0.5
    r = int(NEAR * view.shape[1])
    near = cv2.dilate(skin.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0
    inner = cv2.erode(skin.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
    tex = np.zeros(skin.shape, bool)
    periods, periods_any = [], []
    for x, y, s, per in blocks:
        x0, y0, x1, y1 = (int(v * k) for v in (x, y, x + s, y + s))
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        if near[y0:y1, x0:x1].any():
            tex[y0:y1, x0:x1] = True
            periods_any.append(per)
        if inner[min(cy, inner.shape[0] - 1), min(cx, inner.shape[1] - 1)]:
            periods.append(per)       # шаг линий только внутри кожи, вдали от контура
    hard = skin | (tex & near)
    ring = cv2.dilate(hard.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
    if MODE == "grabcut":
        # GrabCut уточняет край по цвету: квадраты плиток не должны залезать на фон
        gc = np.full(skin.shape, cv2.GC_BGD, np.uint8)
        gc[ring] = cv2.GC_PR_BGD
        gc[hard] = cv2.GC_PR_FGD
        gc[cv2.erode(skin.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0] = cv2.GC_FGD
        bgm, fgm = np.zeros((1, 65)), np.zeros((1, 65))
        cv2.grabCut(view, gc, None, bgm, fgm, 4, cv2.GC_INIT_WITH_MASK)
        fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.float32)
    else:
        # простая замена для браузера: спорный пиксель относим к коже или к фону
        # по расстоянию Махаланобиса до цвета кожи и цвета фона рядом (Lab)
        lab = cv2.cvtColor(view, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        def stats(sel):
            v = lab[sel.ravel()]
            return v.mean(0), np.linalg.inv(np.cov(v.T) + np.eye(3) * 4)
        def maha(m, ic):
            d = lab - m
            return np.einsum("ij,jk,ik->i", d, ic, d)
        ms, cs = stats(skin)
        mb, cb = stats(ring & ~hard)
        closer = (maha(ms, cs) < maha(mb, cb)).reshape(skin.shape)
        fg = (skin | (hard & closer)).astype(np.float32)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        fg = np.maximum(fg, skin.astype(np.float32))
    # подгонка края на половинном кадре и запас 2 px наружу
    half = cv2.resize(bgr, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    m = cv2.resize(fg, (W // 2, H // 2), interpolation=cv2.INTER_LINEAR)
    m = np.clip(guided(g, m, 6, 1e-4), 0, 1)
    m = cv2.dilate(np.clip((m - 0.35) / 0.3, 0, 1), np.ones((3, 3), np.uint8))
    alpha = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
    return alpha, (periods or periods_any), hard


def smooth_skin(img, alpha, sigma):
    # уменьшаем в 4 раза с защитой от муара (pyrDown сам сглаживает перед прореживанием)
    w = (alpha > 0.5).astype(np.float32)
    a, b = img * w[..., None], w
    for _ in range(2):
        a, b = cv2.pyrDown(a), cv2.pyrDown(b)
    s = sigma / 4
    a, b = cv2.GaussianBlur(a, (0, 0), s), cv2.GaussianBlur(b, (0, 0), s)
    sm = a / (b[..., None] + 1e-4)
    return cv2.resize(sm, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)


def scrub(path: Path, out_dir: Path):
    t = time.time()
    bgr = cv2.imread(str(path))
    alpha, periods, _ = build_mask(bgr, ridge_blocks(path))
    P = float(np.median(periods)) if periods else 18.0
    sigma = float(np.clip(P, 8, 40))
    img = bgr.astype(np.float32)
    sm = smooth_skin(img, alpha, sigma)
    rng = np.random.default_rng(int.from_bytes(np.random.bytes(8), "little"))
    grain = cv2.GaussianBlur(rng.normal(0, 3.0, alpha.shape).astype(np.float32), (0, 0), 0.8)
    res = img * (1 - alpha[..., None]) + (sm + grain[..., None]) * alpha[..., None]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{path.stem}_scrub.jpg"
    cv2.imwrite(str(out), np.clip(res, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])
    k = 2000 / bgr.shape[1]
    cv2.imwrite(str(out_dir / f"{path.stem}_alpha.jpg"),
                (cv2.resize(alpha, None, fx=k, fy=k) * 255).astype(np.uint8))
    print(f"{path.name}: P={P:.1f} сигма={sigma:.1f} доля маски={alpha.mean():.2f} {time.time() - t:.1f} с")
    return out


if __name__ == "__main__":
    for a in sys.argv[1:]:
        p = Path(a)
        scrub(p, p.parent / "out" / "scrub")
