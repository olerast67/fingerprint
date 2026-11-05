// Фоновый поток: вся тяжёлая обработка, чтобы страница не зависала.
import { processImage, tint } from "./lib/core.js";

self.onmessage = async (e) => {
  const { bitmap, view, vw, vh, conf, opts } = e.data;
  try {
    const W = bitmap.width, H = bitmap.height;
    const canvas = new OffscreenCanvas(W, H);
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("Браузер не дал холст такого размера. Попробуйте уменьшить фото.");
    ctx.fillStyle = "#fff";                    // прозрачный PNG получает белый фон, а не чёрный
    ctx.fillRect(0, 0, W, H);
    ctx.drawImage(bitmap, 0, 0);
    bitmap.close();
    const img = ctx.getImageData(0, 0, W, H);
    const res = processImage(img.data, W, H, view, vw, vh, conf, (stage, pct) => self.postMessage({ type: "progress", stage, pct }), opts);
    ctx.putImageData(img, 0, 0);
    // файл собирается заново из пикселей: EXIF, миниатюра оригинала и прочие вложения не переносятся
    const blob = await canvas.convertToBlob({ type: "image/jpeg", quality: 0.92 });
    // карта строится из обработанного кадра: на исходном узор был бы виден
    const small = new OffscreenCanvas(vw, vh), sctx = small.getContext("2d");
    sctx.imageSmoothingQuality = "high";
    sctx.drawImage(canvas, 0, 0, vw, vh);
    const map = tint(sctx.getImageData(0, 0, vw, vh).data, vw, vh, res.alpha, res.residual);
    self.postMessage({ type: "done", blob, map, stats: res.stats }, [map.buffer]);
  } catch (err) {
    const memory = err instanceof RangeError || /memory|allocation/i.test(String(err && err.message));
    self.postMessage({ type: "error", memory, message: String(err && err.message ? err.message : err) });
  }
};
