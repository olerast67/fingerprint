// Страница: выбор фото, модель кожи MediaPipe, передача в фоновый поток, показ результата.

const VIEW = 2000;
const $ = (s) => document.querySelector(s);
const ui = {
  pick: $("#pick"), file: $("#file"), drop: $("#drop"),
  work: $("#work"), bar: $("#bar"), stage: $("#stage"),
  result: $("#result"), img: $("#shown"), summary: $("#summary"), download: $("#download"),
  tabs: [...document.querySelectorAll("[data-show]")], error: $("#error"), again: $("#again"),
  rMask: $("#r-mask"), rPeriod: $("#r-period"), rCheck: $("#r-check"),
};
let segmenter = null, urls = {}, busy = false;

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

function skinConfidence(seg, bitmap) {
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const ctx = c.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, 256, 256);
  const r = seg.segment(c);
  const body = r.confidenceMasks[2].getAsFloat32Array(), face = r.confidenceMasks[3].getAsFloat32Array();
  const conf = new Float32Array(256 * 256);
  for (let i = 0; i < conf.length; i++) conf[i] = body[i] + face[i];
  r.close();
  return conf;
}

function viewOf(bitmap) {
  const vw = Math.min(VIEW, bitmap.width), vh = Math.round(bitmap.height * vw / bitmap.width);
  const c = document.createElement("canvas");
  c.width = vw; c.height = vh;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, vw, vh);
  return { data: ctx.getImageData(0, 0, vw, vh).data, vw, vh, canvas: c };
}

function setState(s) {
  ui.drop.hidden = s !== "idle";
  ui.work.hidden = s !== "work";
  ui.result.hidden = s !== "done";
  ui.error.hidden = s !== "error";
}

function show(which) {
  ui.img.src = urls[which];
  ui.img.alt = { after: "Фото после обработки", before: "Исходное фото", map: "Карта: бирюзовым обработанная кожа, пурпурным места для проверки" }[which];
  for (const t of ui.tabs) t.setAttribute("aria-pressed", String(t.dataset.show === which));
}

function readout(st) {
  ui.rMask.textContent = `${Math.round(st.maskFrac * 100)}%`;
  ui.rPeriod.textContent = `${Math.round(st.P)} пикс.`;
  ui.rCheck.textContent = st.residual ? `проверить: ${st.residual}` : "чисто";
  ui.rCheck.classList.toggle("warn", !!st.residual);
}

function describe(st) {
  const parts = [];
  if (!st.hits) parts.push("Похожего на папиллярный узор на этом снимке почти не нашлось, кожа всё равно обработана.");
  if (st.residual) parts.push(`Проверка нашла участков для ручной проверки: ${st.residual}. Они отмечены красным на карте. Если это не пальцы и не ладонь (татуировка, ткань, складка), всё в порядке. Если это палец, не публикуйте снимок.`);
  else parts.push("Повторная проверка следов узора не нашла.");
  return parts.join(" ");
}

async function handle(file) {
  if (busy || !file) return;
  busy = true;
  for (const u of Object.values(urls)) URL.revokeObjectURL(u);
  urls = {};
  setState("work");
  ui.bar.value = 0;
  ui.stage.textContent = "Открываю фото";
  try {
    let bitmap;
    try { bitmap = await createImageBitmap(file, { imageOrientation: "from-image" }); }
    catch { throw new Error("Браузер не смог открыть этот файл. Если это HEIC, сохраните фото как JPEG и попробуйте снова."); }
    ui.stage.textContent = "Загружаю модель кожи";
    const seg = await getSegmenter();
    ui.stage.textContent = "Ищу кожу";
    const conf = skinConfidence(seg, bitmap);
    const view = viewOf(bitmap);
    urls.before = URL.createObjectURL(file);
    const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    const done = await new Promise((resolve, reject) => {
      worker.onmessage = (e) => {
        const m = e.data;
        if (m.type === "progress") { ui.bar.value = m.pct; ui.stage.textContent = m.stage; }
        else if (m.type === "done") resolve(m);
        else reject(new Error(m.message));
      };
      worker.onerror = (e) => reject(new Error(e.message || "Ошибка в фоновом потоке"));
      worker.postMessage({ bitmap, view: view.data, vw: view.vw, vh: view.vh, conf }, [bitmap, view.data.buffer, conf.buffer]);
    });
    worker.terminate();
    urls.after = URL.createObjectURL(done.blob);
    const ctx = view.canvas.getContext("2d");
    ctx.putImageData(new ImageData(done.overlay, view.vw, view.vh), 0, 0);
    urls.map = URL.createObjectURL(await new Promise((r) => view.canvas.toBlob(r, "image/jpeg", 0.85)));
    const base = (file.name || "foto").replace(/\.[^.]+$/, "");
    ui.download.href = urls.after;
    ui.download.download = `${base}-bez-otpechatkov.jpg`;
    ui.summary.textContent = describe(done.stats);
    readout(done.stats);
    window.__lastStats = done.stats;
    show(done.stats.residual ? "map" : "after");
    setState("done");
  } catch (err) {
    ui.error.querySelector("p").textContent = err.message.includes("memory") || err.name === "RangeError"
      ? "Не хватило памяти для такого большого снимка. Попробуйте на компьютере или уменьшите фото."
      : err.message;
    setState("error");
  } finally {
    busy = false;
  }
}

ui.file.addEventListener("change", () => handle(ui.file.files[0]));
ui.pick.addEventListener("click", () => ui.file.click());
ui.again.addEventListener("click", () => { ui.file.value = ""; setState("idle"); });
for (const t of ui.tabs) t.addEventListener("click", () => show(t.dataset.show));
for (const ev of ["dragenter", "dragover"]) ui.drop.addEventListener(ev, (e) => { e.preventDefault(); ui.drop.classList.add("over"); });
for (const ev of ["dragleave", "drop"]) ui.drop.addEventListener(ev, () => ui.drop.classList.remove("over"));
ui.drop.addEventListener("drop", (e) => { e.preventDefault(); handle(e.dataTransfer.files[0]); });
document.querySelectorAll(".retry").forEach((b) => b.addEventListener("click", () => setState("idle")));

// только для локальной проверки разработчиком: ?test=путь к фото на том же сервере
const test = new URLSearchParams(location.search).get("test");
if (test && ["localhost", "127.0.0.1"].includes(location.hostname)) {
  fetch(test).then((r) => r.blob()).then((b) => handle(new File([b], test.split("/").pop(), { type: b.type })));
}
setState("idle");
