// Прогон браузерного ядра web/lib/core.js в Node на сырых данных от dev/js_check.py.
import { readFileSync, writeFileSync } from "node:fs";
import { processImage } from "../web/lib/core.js";

const dir = process.argv[2];
const meta = JSON.parse(readFileSync(`${dir}/meta.json`, "utf8"));
const u8 = (f) => new Uint8ClampedArray(readFileSync(`${dir}/${f}`).buffer.slice(0));
const rgba = u8("full.rgba"), view = u8("view.rgba");
const cb = readFileSync(`${dir}/conf.f32`);
const conf = new Float32Array(cb.buffer, cb.byteOffset, cb.length / 4);
const T0 = Date.now();
const res = processImage(rgba, meta.W, meta.H, view, meta.vw, meta.vh, conf, (s, p) => process.stderr.write(`${Date.now() - T0} мс ${p}% ${s}\n`));
writeFileSync(`${dir}/out.rgba`, rgba);
writeFileSync(`${dir}/overlay.rgba`, res.overlay);
console.log(JSON.stringify(res.stats));
