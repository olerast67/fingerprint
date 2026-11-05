// Фоновый поток: вся тяжёлая обработка, чтобы страница не зависала.
import { processImage } from "./lib/core.js";

self.onmessage = async (e) => {
  const { bitmap, view, vw, vh, conf } = e.data;
  try {
    const W = bitmap.width, H = bitmap.height;
    const canvas = new OffscreenCanvas(W, H);
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0);
    bitmap.close();
    const img = ctx.getImageData(0, 0, W, H);
    const res = processImage(img.data, W, H, view, vw, vh, conf, (stage, pct) => self.postMessage({ type: "progress", stage, pct }));
    ctx.putImageData(img, 0, 0);
    // файл собирается заново из пикселей: EXIF, миниатюра оригинала и прочие вложения не переносятся
    const blob = await canvas.convertToBlob({ type: "image/jpeg", quality: 0.92 });
    self.postMessage({ type: "done", blob, overlay: res.overlay, stats: res.stats }, [res.overlay.buffer]);
  } catch (err) {
    self.postMessage({ type: "error", message: String(err && err.message ? err.message : err) });
  }
};
