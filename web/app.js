// Страница: выбор фото, модель кожи MediaPipe, передача в фоновый поток, показ результата.
const VIEW = 2000;
const IOS_LIMIT = 16_000_000;   // Safari на iPhone не создаёт холст больше ~16,7 Мп
const $ = (s) => document.querySelector(s);
const ui = {
  pick: $("#pick"), file: $("#file"), drop: $("#drop"),
  work: $("#work"), bar: $("#bar"), stage: $("#stage"),
  result: $("#result"), img: $("#shown"), summary: $("#summary"), download: $("#download"),
  tabs: [...document.querySelectorAll("[data-show]")], error: $("#error"), again: $("#again"),
  rMask: $("#r-mask"), rPeriod: $("#r-period"), rCheck: $("#r-check"), keepFar: $("#keepfar"),
  paint: $("#paint"), tools: $("#tools"), size: $("#size"), apply: $("#apply"), reset: $("#reset"),
  toolBtns: [...document.querySelectorAll("[data-tool]")],
};
// правки маски на рабочей копии: add закрашено кистью, remove стёрто ластиком
const edit = { tool: null, add: null, remove: null, vw: 0, vh: 0, keepFar: false };
let segmenter = null, urls = {}, busy = false, lastFile = null;
const isIOS = /iP(hone|ad|od)/.test(navigator.userAgent) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);

async function getSegmenter() {
  if (segmenter) return segmenter;
  // библиотеку грузим только после выбора фото: первый экран открывается без неё
  const { FilesetResolver, ImageSegmenter } = await import("./vendor/tasks-vision/vision_bundle.mjs");
  const base = new URL(".", import.meta.url);
  const files = await FilesetResolver.forVisionTasks(new URL("vendor/tasks-vision/wasm", base).href);
  segmenter = await ImageSegmenter.createFromOptions(files, {
    baseOptions: { modelAssetPath: new URL("models/selfie_multiclass_256x256.tflite", base).href, delegate: "CPU" },
    runningMode: "IMAGE", outputCategoryMask: false, outputConfidenceMasks: true,
  });
  return segmenter;
}

// модель кожи видит квадрат 256x256: даём ей весь кадр с полями и перекрывающиеся куски, берём максимум
function skinConfidence(seg, bitmap, vw, vh) {
  const W = bitmap.width, H = bitmap.height, kx = vw / W, conf = new Float32Array(vw * vh);
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const ctx = c.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  const run = (sx, sy, side) => {
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, 256, 256);
    const ix = Math.max(0, sx), iy = Math.max(0, sy), iw = Math.min(W, sx + side) - ix, ih = Math.min(H, sy + side) - iy;
    ctx.drawImage(bitmap, ix, iy, iw, ih, (ix - sx) * 256 / side, (iy - sy) * 256 / side, iw * 256 / side, ih * 256 / side);
    const r = seg.segment(c);
    const body = r.confidenceMasks[2].getAsFloat32Array(), face = r.confidenceMasks[3].getAsFloat32Array();
    const x0 = Math.floor(ix * kx), x1 = Math.min(vw, Math.ceil((ix + iw) * kx)), y0 = Math.floor(iy * kx), y1 = Math.min(vh, Math.ceil((iy + ih) * kx));
    for (let y = y0; y < y1; y++) {
      const v = Math.min(255, Math.max(0, Math.floor(((y + 0.5) / kx - sy) * 256 / side)));
      for (let x = x0; x < x1; x++) {
        const u = Math.min(255, Math.max(0, Math.floor(((x + 0.5) / kx - sx) * 256 / side))), q = v * 256 + u;
        const val = body[q] + face[q];
        if (val > conf[y * vw + x]) conf[y * vw + x] = val;
      }
    }
    r.close();
  };
  const side = Math.max(W, H);
  run((W - side) / 2, (H - side) / 2, side);
  const t = Math.round(Math.min(W, H) * 0.6), step = Math.round(t / 2);
  const pos = (n) => { const a = []; for (let p = 0; p + t < n; p += step) a.push(p); a.push(Math.max(0, n - t)); return a; };
  for (const y of pos(H)) for (const x of pos(W)) run(x, y, t);
  return conf;
}

function viewOf(bitmap) {
  const vw = Math.min(VIEW, bitmap.width), vh = Math.max(1, Math.round(bitmap.height * vw / bitmap.width));
  const c = document.createElement("canvas");
  c.width = vw; c.height = vh;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, vw, vh);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, vw, vh);
  return { data: ctx.getImageData(0, 0, vw, vh).data, vw, vh, canvas: c };
}

function resetEdits(vw, vh) {
  Object.assign(edit, { vw, vh, add: new Uint8Array(vw * vh), remove: new Uint8Array(vw * vh), dirty: false });
  ui.paint.width = vw; ui.paint.height = vh;
  drawEdits();
}

// мазок кистью или ластиком: круг в масках рабочей копии
function stroke(e) {
  const r = ui.paint.getBoundingClientRect(), k = edit.vw / r.width;
  const cx = (e.clientX - r.left) * k, cy = (e.clientY - r.top) * k, rad = +ui.size.value * k / 2;
  const on = edit.tool === "add" ? edit.add : edit.remove, off = edit.tool === "add" ? edit.remove : edit.add;
  for (let y = Math.max(0, Math.floor(cy - rad)); y < Math.min(edit.vh, Math.ceil(cy + rad)); y++)
    for (let x = Math.max(0, Math.floor(cx - rad)); x < Math.min(edit.vw, Math.ceil(cx + rad)); x++)
      if ((x - cx) ** 2 + (y - cy) ** 2 <= rad * rad) { on[y * edit.vw + x] = 1; off[y * edit.vw + x] = 0; }
  edit.dirty = true;
  if (!edit.raf) edit.raf = requestAnimationFrame(() => { edit.raf = 0; drawEdits(); });
}

function drawEdits() {
  if (!edit.vw || !edit.add) return;
  const ctx = ui.paint.getContext("2d"), img = ctx.createImageData(edit.vw, edit.vh), d = img.data;
  for (let i = 0; i < edit.add.length; i++) {
    if (edit.add[i]) { d[i * 4] = 63; d[i * 4 + 1] = 240; d[i * 4 + 2] = 216; d[i * 4 + 3] = 150; }
    else if (edit.remove[i]) { d[i * 4] = 6; d[i * 4 + 1] = 18; d[i * 4 + 2] = 17; d[i * 4 + 3] = 170; }
  }
  ctx.putImageData(img, 0, 0);
}

function setState(s) {
  ui.drop.hidden = s !== "idle";
  ui.work.hidden = s !== "work";
  ui.result.hidden = s !== "done";
  ui.error.hidden = s !== "error";
}

function show(which) {
  ui.img.src = urls[which];
  ui.img.alt = { after: "Фото после обработки", before: "Исходное фото, его не публикуйте", map: "Карта: бирюзовым обработанная кожа, пурпурным места для проверки" }[which];
  for (const t of ui.tabs) t.setAttribute("aria-pressed", String(t.dataset.show === which));
  ui.tools.hidden = which !== "map";
  ui.paint.hidden = which !== "map";
}

function readout(st) {
  ui.rMask.textContent = `${Math.round(st.maskFrac * 100)}%`;
  ui.rPeriod.textContent = st.hits ? `${Math.round(st.P)} пикс.` : "узор не найден";
  ui.rCheck.textContent = st.residual ? `проверить: ${st.residual}` : `чисто, ${Math.round(st.checkedFrac * 100)}% кожи`;
  ui.rCheck.classList.toggle("warn", !!st.residual);
}

function describe(st, scaled) {
  const parts = [];
  if (scaled) parts.push("Снимок уменьшен до 16 Мп: Safari на iPhone не обрабатывает кадры крупнее.");
  if (!st.maskFrac) parts.push("Кожи и папиллярного узора на снимке не нашлось, фото не изменено. Если на нём есть руки, не публикуйте снимок.");
  else if (!st.hits) parts.push("Похожего на папиллярный узор почти не нашлось, найденная кожа всё равно обработана.");
  if (st.farFrac > 0.002 && !st.keepFar) parts.push("Полосы нашлись и вне кожи: это может быть палец, который нейросеть не узнала, или дерево и ткань. Они тоже стёрты. Если там точно нет рук, их можно вернуть.");
  if (st.keepFar) parts.push("Полосы вне кожи оставлены как были.");
  if (st.residual) parts.push(`Мест для ручной проверки: ${st.residual}. Они отмечены пурпурным на карте. Если это не пальцы и не ладонь, всё в порядке. Если это палец, не публикуйте снимок.`);
  else if (st.maskFrac) parts.push("Повторная проверка следов узора не нашла.");
  return parts.join(" ");
}

async function handle(file, opts = {}) {
  if (busy || !file) return;
  busy = true;
  lastFile = file;
  ui.file.value = "";
  for (const u of Object.values(urls)) URL.revokeObjectURL(u);
  urls = {};
  setState("work");
  ui.bar.value = 0;
  ui.stage.textContent = "Открываю фото";
  let worker = null;
  try {
    if (typeof OffscreenCanvas === "undefined" || !("convertToBlob" in OffscreenCanvas.prototype))
      throw new Error("Этот браузер слишком старый для обработки. Нужен Chrome, Edge, Firefox или Safari 16.4 и новее.");
    let bitmap;
    try { bitmap = await createImageBitmap(file, { imageOrientation: "from-image" }); }
    catch { throw new Error("Браузер не смог открыть этот файл. Если это HEIC, сохраните фото как JPEG и попробуйте снова."); }
    let scaled = false;
    if (isIOS && bitmap.width * bitmap.height > IOS_LIMIT) {
      const f = Math.sqrt(IOS_LIMIT / (bitmap.width * bitmap.height));
      const w = Math.floor(bitmap.width * f), h = Math.floor(bitmap.height * f);
      bitmap.close();
      bitmap = await createImageBitmap(file, { imageOrientation: "from-image", resizeWidth: w, resizeHeight: h, resizeQuality: "high" });
      scaled = true;
    }
    ui.stage.textContent = "Загружаю модель кожи";
    const seg = await getSegmenter();
    ui.stage.textContent = "Ищу кожу";
    await new Promise((r) => setTimeout(r, 0));
    const view = viewOf(bitmap);
    const conf = skinConfidence(seg, bitmap, view.vw, view.vh);
    urls.before = URL.createObjectURL(file);
    worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    const done = await new Promise((resolve, reject) => {
      worker.onmessage = (e) => {
        const m = e.data;
        if (m.type === "progress") { ui.bar.value = m.pct; ui.stage.textContent = m.stage; }
        else if (m.type === "done") resolve(m);
        else reject(Object.assign(new Error(m.message), { memory: m.memory }));
      };
      worker.onerror = (e) => reject(new Error(e.message || "Ошибка в фоновом потоке"));
      if (edit.vw !== view.vw || edit.vh !== view.vh) resetEdits(view.vw, view.vh);
    edit.keepFar = !!opts.keepFar;
    const wopts = { keepFar: edit.keepFar, add: edit.dirty ? edit.add.slice() : null, remove: edit.dirty ? edit.remove.slice() : null };
    worker.postMessage({ bitmap, view: view.data, vw: view.vw, vh: view.vh, conf, opts: wopts }, [bitmap, view.data.buffer, conf.buffer]);
    });
    urls.after = URL.createObjectURL(done.blob);
    view.canvas.getContext("2d").putImageData(new ImageData(done.map, view.vw, view.vh), 0, 0);
    urls.map = URL.createObjectURL(await new Promise((r) => view.canvas.toBlob(r, "image/jpeg", 0.85)));
    ui.download.href = urls.after;
    ui.download.download = "foto-bez-otpechatkov.jpg";   // без имени оригинала: в нём бывают дата и время съёмки
    ui.summary.textContent = describe(done.stats, scaled);
    ui.keepFar.hidden = !(done.stats.farFrac > 0.002) || done.stats.keepFar;
    readout(done.stats);
    drawEdits();
    window.__lastStats = done.stats;
    show(done.stats.residual || !done.stats.maskFrac ? "map" : "after");
    setState("done");
  } catch (err) {
    ui.error.querySelector("p").textContent = err.memory || err instanceof RangeError
      ? "Не хватило памяти для такого большого снимка. Попробуйте на компьютере или уменьшите фото."
      : err.message;
    setState("error");
  } finally {
    if (worker) worker.terminate();
    busy = false;
  }
}

ui.file.addEventListener("change", () => { edit.vw = 0; handle(ui.file.files[0]); });
ui.pick.addEventListener("click", () => ui.file.click());
ui.again.addEventListener("click", () => setState("idle"));
ui.keepFar.addEventListener("click", () => handle(lastFile, { keepFar: true }));
ui.apply.addEventListener("click", () => handle(lastFile, { keepFar: edit.keepFar }));
ui.reset.addEventListener("click", () => { resetEdits(edit.vw, edit.vh); handle(lastFile, { keepFar: edit.keepFar }); });
for (const b of ui.toolBtns) b.addEventListener("click", () => {
  edit.tool = edit.tool === b.dataset.tool ? null : b.dataset.tool;
  for (const x of ui.toolBtns) x.setAttribute("aria-pressed", String(x.dataset.tool === edit.tool));
});
ui.paint.addEventListener("pointerdown", (e) => { if (!edit.tool) return; ui.paint.setPointerCapture(e.pointerId); stroke(e); });
ui.paint.addEventListener("pointermove", (e) => { if (edit.tool && e.buttons) stroke(e); });
for (const t of ui.tabs) t.addEventListener("click", () => show(t.dataset.show));
for (const ev of ["dragenter", "dragover"]) ui.drop.addEventListener(ev, (e) => { e.preventDefault(); ui.drop.classList.add("over"); });
for (const ev of ["dragleave", "drop"]) ui.drop.addEventListener(ev, () => ui.drop.classList.remove("over"));
ui.drop.addEventListener("drop", (e) => {
  e.preventDefault();
  const f = [...e.dataTransfer.files].find((x) => x.type.startsWith("image/"));
  if (f) { edit.vw = 0; handle(f); }
});
document.querySelectorAll(".retry").forEach((b) => b.addEventListener("click", () => setState("idle")));

// только для локальной проверки разработчиком: ?test=путь к фото на том же сервере
const params = new URLSearchParams(location.search), test = params.get("test");
if (test && ["localhost", "127.0.0.1"].includes(location.hostname)) {
  fetch(test).then((r) => r.blob()).then((b) => handle(new File([b], test.split("/").pop(), { type: b.type }), { keepFar: params.has("keepFar") }));
}
setState("idle");
