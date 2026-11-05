"""Проба MediaPipe: руки (HandLandmarker) и кожа (SelfieMulticlass) на фото автора.

Руки ищем на копии 1280 px и на плитках 1024 px копии 2560 px (мелкие руки).
Кожу ищем на всём кадре и отдельно на вырезке вокруг найденных рук.
Запуск: python dev/mp_test.py samples/private/*.jpg  -> samples/private/out/mp/
"""
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

M = Path(__file__).parent / "models"
hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(M / "hand_landmarker.task")),
    num_hands=8, min_hand_detection_confidence=0.25, min_hand_presence_confidence=0.25))
seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=str(M / "selfie_multiclass_256x256.tflite")),
    output_category_mask=False, output_confidence_masks=True))
SKIN = (2, 3)  # кожа тела и кожа лица


def mpimg(rgb):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


def find_hands(rgb):
    """Возвращает список рамок (x0, y0, x1, y1) в долях кадра и точки кончиков."""
    boxes, tips = [], []

    def run(img, ox, oy, sx, sy):
        r = hands.detect(mpimg(img))
        for lm in r.hand_landmarks:
            xs = [ox + p.x * sx for p in lm]
            ys = [oy + p.y * sy for p in lm]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
            tips.extend((xs[i], ys[i]) for i in (4, 8, 12, 16, 20))

    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (1280, int(h * 1280 / w)), interpolation=cv2.INTER_AREA)
    run(small, 0, 0, 1, 1)
    big = cv2.resize(rgb, (2560, int(h * 2560 / w)), interpolation=cv2.INTER_AREA)
    bh, bw = big.shape[:2]
    T, S = 1024, 768
    for y in range(0, max(bh - T, 0) + 1, S):
        for x in range(0, max(bw - T, 0) + 1, S):
            run(big[y:y + T, x:x + T], x / bw, y / bh, T / bw, T / bh)
    return boxes, tips


def skin_mask(rgb):
    r = seg.segment(mpimg(cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)))
    conf = sum(r.confidence_masks[i].numpy_view() for i in SKIN)
    return cv2.resize(conf.astype(np.float32), (rgb.shape[1], rgb.shape[0]))


def main(paths):
  for p in map(Path, paths):
      rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
      k = 2000 / rgb.shape[1]
      view = cv2.resize(rgb, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
      vh, vw = view.shape[:2]
      boxes, tips = find_hands(rgb)
      whole = skin_mask(view)
      crop = np.zeros_like(whole)
      for x0, y0, x1, y1 in boxes:
          mx, my = (x1 - x0) * 0.3, (y1 - y0) * 0.3
          a, b = int(max(0, x0 - mx) * vw), int(max(0, y0 - my) * vh)
          c, d = int(min(1, x1 + mx) * vw), int(min(1, y1 + my) * vh)
          if c - a > 8 and d - b > 8:
              crop[b:d, a:c] = np.maximum(crop[b:d, a:c], skin_mask(view[b:d, a:c]))
      out = p.parent / "out" / "mp"
      out.mkdir(parents=True, exist_ok=True)
      for name, m in (("whole", whole), ("crop", crop)):
          tint = view.copy()
          tint[m > 0.5] = (0.5 * tint[m > 0.5] + [0, 127, 0]).astype(np.uint8)
          for x0, y0, x1, y1 in boxes:
              cv2.rectangle(tint, (int(x0 * vw), int(y0 * vh)), (int(x1 * vw), int(y1 * vh)), (255, 0, 0), 3)
          for x, y in tips:
              cv2.circle(tint, (int(x * vw), int(y * vh)), 10, (255, 255, 0), -1)
          cv2.imwrite(str(out / f"{p.stem}_{name}.jpg"), cv2.cvtColor(tint, cv2.COLOR_RGB2BGR))
      print(p.name, "рамок рук:", len(boxes), "кожа на всём кадре:", round(float((whole > 0.5).mean()), 2))


if __name__ == "__main__":
    main(sys.argv[1:])
