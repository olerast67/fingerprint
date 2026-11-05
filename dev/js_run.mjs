// Прогон браузерного ядра web/lib/core.js в Node на сырых данных от dev/js_check.py.
// Аргументы: <папка> [keepFar]
import { readFileSync, writeFileSync } from "node:fs";
import { processImage } from "../web/lib/core.js";

const dir = process.argv[2], keepFar = process.argv[3] === "keepFar";
const meta = JSON.parse(readFileSync(`${dir}/meta.json`, "utf8"));
const u8 = (f) => new Uint8ClampedArray(readFileSync(`${dir}/${f}`).buffer.slice(0));
const rgba = u8("full.rgba"), view = u8("view.rgba");
const cb = readFileSync(`${dir}/conf.f32`);
const conf = new Float32Array(cb.buffer.slice(cb.byteOffset, cb.byteOffset + cb.length));
const T0 = Date.now();
const res = processImage(rgba, meta.W, meta.H, view, meta.vw, meta.vh, conf,
  (s, p) => process.stderr.write(`${Date.now() - T0} мс ${p}% ${s}\n`), { keepFar });
writeFileSync(`${dir}/out.rgba`, rgba);
writeFileSync(`${dir}/alpha.f32`, Buffer.from(res.alpha.buffer));
writeFileSync(`${dir}/residual.json`, JSON.stringify(res.residual));
console.log(JSON.stringify(res.stats));
