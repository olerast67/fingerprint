// Ядро обработки: находит кожу с папиллярным узором и необратимо его стирает.
// Чистые функции над массивами пикселей, без DOM: работает в Web Worker и в Node.
// Повторяет эталон dev/scrub.py; пороги объяснены в docs/research/04-zamery.md.

export const B = 128, STEP = 64, SCALES = [1, 2, 4], PMIN = 5, PMAX = 20, THR = 0.5;
const VIEW = 2000;           // ширина рабочей копии для маски
const NEAR = 0.04;           // «рядом с кожей»: доля ширины кадра

// ---------- БПФ 128x128 ----------
const N = B, LOG = 7;
const REV = new Uint16Array(N);
for (let i = 0; i < N; i++) { let r = 0; for (let b = 0; b < LOG; b++) r |= ((i >> b) & 1) << (LOG - 1 - b); REV[i] = r; }
const COS = new Float32Array(N / 2), SIN = new Float32Array(N / 2);
for (let i = 0; i < N / 2; i++) { COS[i] = Math.cos(2 * Math.PI * i / N); SIN[i] = -Math.sin(2 * Math.PI * i / N); }
const freq = (i) => (i < N / 2 ? i : i - N) / N;
const HANN = new Float32Array(N * N), BAND = new Uint8Array(N * N), PERIOD = new Float32Array(N * N);
{
  const h = new Float32Array(N);
  for (let i = 0; i < N; i++) h[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (N - 1));
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const k = y * N + x, r = Math.hypot(freq(x), freq(y));
    HANN[k] = h[y] * h[x];
    BAND[k] = r >= 1 / PMAX && r <= 1 / PMIN ? 1 : 0;
    PERIOD[k] = r > 0 ? 1 / r : 0;
  }
}

function fft1(re, im, off, stride) {
  for (let i = 0; i < N; i++) {
    const j = REV[i];
    if (j > i) {
      const a = off + i * stride, b = off + j * stride;
      let t = re[a]; re[a] = re[b]; re[b] = t; t = im[a]; im[a] = im[b]; im[b] = t;
    }
  }
  for (let size = 2; size <= N; size <<= 1) {
    const half = size >> 1, step = N / size;
    for (let i = 0; i < N; i += size) for (let k = 0; k < half; k++) {
      const c = COS[k * step], s = SIN[k * step];
      const a = off + (i + k) * stride, b = off + (i + k + half) * stride;
      const tr = re[b] * c - im[b] * s, ti = re[b] * s + im[b] * c;
      re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
    }
  }
}
const RE = new Float32Array(N * N), IM = new Float32Array(N * N), PW = new Float32Array(N * N);
function fft2() {
  for (let y = 0; y < N; y++) fft1(RE, IM, y * N, 1);
  for (let x = 0; x < N; x++) fft1(RE, IM, x, N);
}

// ---------- размытие (три прохода ящиком ≈ гаусс) ----------
function boxSizes(sigma) {
  const n = 3, wIdeal = Math.sqrt(12 * sigma * sigma / n + 1);
  let wl = Math.floor(wIdeal); if (wl % 2 === 0) wl--;
  const m = Math.round((12 * sigma * sigma - n * wl * wl - 4 * n * wl - 3 * n) / (-4 * wl - 4));
  return [0, 1, 2].map((i) => ((i < m ? wl : wl + 2) - 1) / 2);
}
function boxH(src, dst, w, h, r) {
  const inv = 1 / (2 * r + 1);
  for (let y = 0; y < h; y++) {
    const o = y * w; let acc = 0;
    for (let i = -r - 1; i < r; i++) acc += src[o + Math.min(w - 1, Math.max(0, i))];
    for (let x = 0; x < w; x++) {
      acc += src[o + Math.min(w - 1, x + r)] - src[o + Math.max(0, x - r - 1)];
      dst[o + x] = acc * inv;
    }
  }
}
function boxV(src, dst, w, h, r) {
  const inv = 1 / (2 * r + 1);
  for (let x = 0; x < w; x++) {
    let acc = 0;
    for (let i = -r - 1; i < r; i++) acc += src[Math.min(h - 1, Math.max(0, i)) * w + x];
    for (let y = 0; y < h; y++) {
      acc += src[Math.min(h - 1, y + r) * w + x] - src[Math.max(0, y - r - 1) * w + x];
      dst[y * w + x] = acc * inv;
    }
  }
}
export function gaussBlur(a, w, h, sigma) {
  let s = Float32Array.from(a), t = new Float32Array(a.length);
  for (const r of boxSizes(sigma)) {
    if (r < 1) continue;
    boxH(s, t, w, h, r); boxV(t, s, w, h, r);
  }
  return s;
}

// ---------- карта полос (слой 2) ----------
function analyseScale(gray, w, h, d, include, out) {
  const M = 56, ROWS = 4;
  for (let by0 = 0; by0 + B <= h; by0 += STEP * ROWS) {
    const list = [];
    for (let by = by0; by < by0 + STEP * ROWS && by + B <= h; by += STEP)
      for (let bx = 0; bx + B <= w; bx += STEP) if (include(bx * d, by * d, B * d)) list.push(bx, by);
    if (!list.length) continue;
    let bxMin = w, bxMax = 0;
    for (let q = 0; q < list.length; q += 2) { bxMin = Math.min(bxMin, list[q]); bxMax = Math.max(bxMax, list[q]); }
    const x0 = Math.max(0, bxMin - M), x1 = Math.min(w, bxMax + B + M), sw = x1 - x0;
    const y0 = Math.max(0, by0 - M), y1 = Math.min(h, by0 + STEP * (ROWS - 1) + B + M), sh = y1 - y0, n = sw * sh;
    const g = new Float32Array(n);
    for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) g[y * sw + x] = gray[(y0 + y) * w + x0 + x];
    const bl = gaussBlur(g, sw, sh, 8), hp = new Float32Array(n), sq = new Float32Array(n);
    for (let i = 0; i < n; i++) { hp[i] = g[i] - bl[i]; sq[i] = hp[i] * hp[i]; }
    const sd = gaussBlur(sq, sw, sh, 8);
    for (let i = 0; i < n; i++) { const v = hp[i] / (Math.sqrt(sd[i]) + 1); hp[i] = v > 2 ? 2 : v < -2 ? -2 : v; }
    const sm = gaussBlur(hp, sw, sh, 1), gx = new Float32Array(n), gy = new Float32Array(n);
    for (let y = 1; y < sh - 1; y++) for (let x = 1; x < sw - 1; x++) {
      const o = y * sw + x;
      gx[o] = (sm[o - sw + 1] + 2 * sm[o + 1] + sm[o + sw + 1]) - (sm[o - sw - 1] + 2 * sm[o - 1] + sm[o + sw - 1]);
      gy[o] = (sm[o + sw - 1] + 2 * sm[o + sw] + sm[o + sw + 1]) - (sm[o - sw - 1] + 2 * sm[o - sw] + sm[o - sw + 1]);
    }
    for (let q = 0; q < list.length; q += 2) {
      const r = blockStats(hp, gx, gy, sq, sw, list[q] - x0, list[q + 1] - y0);
      if (r) out.push({ x: list[q] * d, y: list[q + 1] * d, s: B * d, coh: r.coh, peak: r.peak, period: r.period * d, amp: r.amp });
    }
  }
}

function blockStats(hp, gx, gy, sq, w, x, y) {
  let gxx = 0, gyy = 0, gxy = 0, m = 0, e = 0;
  for (let j = 0; j < B; j++) {
    let o = (y + j) * w + x;
    for (let i = 0; i < B; i++, o++) { const a = gx[o], b = gy[o]; gxx += a * a; gyy += b * b; gxy += a * b; m += hp[o]; e += sq[o]; }
  }
  // amp: сила мелких деталей в единицах яркости до выравнивания контраста
  const amp = Math.sqrt(e / (B * B));
  const den = gxx + gyy;
  if (den < 1e-3) return null;
  const coh = Math.sqrt((gxx - gyy) ** 2 + 4 * gxy * gxy) / den;
  // оценка = coh × (не больше 1), поэтому при coh < THR плитка заведомо не пройдёт: спектр не нужен
  if (coh < THR) return { coh, peak: 0, period: 0, amp };
  m /= B * B;
  for (let j = 0; j < B; j++) {
    const o = (y + j) * w + x;
    for (let i = 0; i < B; i++) { const k = j * N + i; RE[k] = (hp[o + i] - m) * HANN[k]; IM[k] = 0; }
  }
  fft2();
  let tot = 0, best = -1, bi = 0;
  for (let k = 0; k < N * N; k++) {
    if (!BAND[k]) { PW[k] = 0; continue; }
    const p = RE[k] * RE[k] + IM[k] * IM[k];
    PW[k] = p; tot += p;
    if (p > best) { best = p; bi = k; }
  }
  const iy = bi >> LOG, ix = bi & (N - 1);
  const around = (cy, cx) => {
    let s = 0;
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++)
      if (dx * dx + dy * dy <= 4) s += PW[((cy + dy) & (N - 1)) * N + ((cx + dx) & (N - 1))];
    return s;
  };
  const peak = (around(iy, ix) + around((-iy) & (N - 1), (-ix) & (N - 1))) / (tot + 1e-6);
  return { coh, peak, period: PERIOD[bi], amp };
}

export const score = (b) => b.coh * Math.min(b.peak / 0.4, 1);

function toGray(rgba, n) {
  const g = new Uint8Array(n);
  for (let i = 0, j = 0; i < n; i++, j += 4) g[i] = (0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2] + 0.5) | 0;
  return g;
}
function downscale(gray, w, h, d) {
  const dw = Math.floor(w / d), dh = Math.floor(h / d), out = new Uint8Array(dw * dh), inv = 1 / (d * d);
  for (let y = 0; y < dh; y++) for (let x = 0; x < dw; x++) {
    let s = 0;
    for (let j = 0; j < d; j++) { const o = (y * d + j) * w + x * d; for (let i = 0; i < d; i++) s += gray[o + i]; }
    out[y * dw + x] = (s * inv + 0.5) | 0;
  }
  return [out, dw, dh];
}
export function ridgeBlocks(rgba, W, H, include) {
  const gray = toGray(rgba, W * H), out = [];
  for (const d of SCALES) {
    const [g, w, h] = d === 1 ? [gray, W, H] : downscale(gray, W, H, d);
    analyseScale(g, w, h, d, include, out);
  }
  return out;
}

// ---------- маски на рабочей копии ----------
function distance(mask, w, h) {          // расстояние до ближайшего true (фаска 3-4), в пикселях
  const INF = 1e9, d = new Float32Array(w * h);
  for (let i = 0; i < d.length; i++) d[i] = mask[i] ? 0 : INF;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const o = y * w + x; let v = d[o];
    if (x > 0) v = Math.min(v, d[o - 1] + 3);
    if (y > 0) {
      v = Math.min(v, d[o - w] + 3);
      if (x > 0) v = Math.min(v, d[o - w - 1] + 4);
      if (x < w - 1) v = Math.min(v, d[o - w + 1] + 4);
    }
    d[o] = v;
  }
  for (let y = h - 1; y >= 0; y--) for (let x = w - 1; x >= 0; x--) {
    const o = y * w + x; let v = d[o];
    if (x < w - 1) v = Math.min(v, d[o + 1] + 3);
    if (y < h - 1) {
      v = Math.min(v, d[o + w] + 3);
      if (x < w - 1) v = Math.min(v, d[o + w + 1] + 4);
      if (x > 0) v = Math.min(v, d[o + w - 1] + 4);
    }
    d[o] = v;
  }
  for (let i = 0; i < d.length; i++) d[i] /= 3;
  return d;
}
const dilate = (m, w, h, r) => { const d = distance(m, w, h); return Uint8Array.from(d, (v) => (v <= r ? 1 : 0)); };
const erode = (m, w, h, r) => { const d = distance(Uint8Array.from(m, (v) => 1 - v), w, h); return Uint8Array.from(d, (v) => (v > r ? 1 : 0)); };

function resizeBilinear(src, sw, sh, dw, dh) {
  const out = new Float32Array(dw * dh);
  for (let y = 0; y < dh; y++) {
    const fy = Math.min(sh - 1, Math.max(0, (y + 0.5) * sh / dh - 0.5)), y0 = Math.floor(fy), y1 = Math.min(sh - 1, y0 + 1), ty = fy - y0;
    for (let x = 0; x < dw; x++) {
      const fx = Math.min(sw - 1, Math.max(0, (x + 0.5) * sw / dw - 0.5)), x0 = Math.floor(fx), x1 = Math.min(sw - 1, x0 + 1), tx = fx - x0;
      out[y * dw + x] = (src[y0 * sw + x0] * (1 - tx) + src[y0 * sw + x1] * tx) * (1 - ty) + (src[y1 * sw + x0] * (1 - tx) + src[y1 * sw + x1] * tx) * ty;
    }
  }
  return out;
}

function toLab(view, n) {
  const lab = new Float32Array(n * 3);
  const lin = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
  const f = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const LUT = new Float32Array(256); for (let i = 0; i < 256; i++) LUT[i] = lin(i);
  for (let i = 0; i < n; i++) {
    const r = LUT[view[i * 4]], g = LUT[view[i * 4 + 1]], b = LUT[view[i * 4 + 2]];
    const X = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047, Y = 0.2126 * r + 0.7152 * g + 0.0722 * b,
      Z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
    const fx = f(X), fy = f(Y), fz = f(Z);
    lab[i * 3] = (116 * fy - 16) * 2.55; lab[i * 3 + 1] = 500 * (fx - fy) + 128; lab[i * 3 + 2] = 200 * (fy - fz) + 128;
  }
  return lab;
}
function colorStats(lab, sel) {
  const m = [0, 0, 0], c = new Float64Array(9); let n = 0;
  for (let i = 0; i < sel.length; i++) if (sel[i]) { n++; for (let k = 0; k < 3; k++) m[k] += lab[i * 3 + k]; }
  if (n < 10) return null;
  for (let k = 0; k < 3; k++) m[k] /= n;
  for (let i = 0; i < sel.length; i++) if (sel[i]) {
    const d0 = lab[i * 3] - m[0], d1 = lab[i * 3 + 1] - m[1], d2 = lab[i * 3 + 2] - m[2], d = [d0, d1, d2];
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) c[a * 3 + b] += d[a] * d[b];
  }
  for (let k = 0; k < 9; k++) c[k] = c[k] / (n - 1) + (k % 4 === 0 ? 4 : 0);
  const [a, b, cc, d, e, f, g, h, i] = c, det = a * (e * i - f * h) - b * (d * i - f * g) + cc * (d * h - e * g);
  const inv = [e * i - f * h, cc * h - b * i, b * f - cc * e, f * g - d * i, a * i - cc * g, cc * d - a * f, d * h - e * g, b * g - a * h, a * e - b * d].map((v) => v / det);
  return { m, inv };
}
function maha(lab, i, s) {
  const d0 = lab[i * 3] - s.m[0], d1 = lab[i * 3 + 1] - s.m[1], d2 = lab[i * 3 + 2] - s.m[2], v = s.inv;
  return d0 * (v[0] * d0 + v[1] * d1 + v[2] * d2) + d1 * (v[3] * d0 + v[4] * d1 + v[5] * d2) + d2 * (v[6] * d0 + v[7] * d1 + v[8] * d2);
}
function boxMean(a, w, h, r) { const t = new Float32Array(a.length), o = new Float32Array(a.length); boxH(a, t, w, h, r); boxV(t, o, w, h, r); return o; }
function guided(I, p, w, h, r, eps) {
  const n = I.length, Ip = new Float32Array(n), II = new Float32Array(n);
  for (let i = 0; i < n; i++) { Ip[i] = I[i] * p[i]; II[i] = I[i] * I[i]; }
  const mI = boxMean(I, w, h, r), mp = boxMean(p, w, h, r), mIp = boxMean(Ip, w, h, r), mII = boxMean(II, w, h, r);
  const A = new Float32Array(n), Bc = new Float32Array(n);
  for (let i = 0; i < n; i++) { A[i] = (mIp[i] - mI[i] * mp[i]) / (mII[i] - mI[i] * mI[i] + eps); Bc[i] = mp[i] - A[i] * mI[i]; }
  const mA = boxMean(A, w, h, r), mB = boxMean(Bc, w, h, r), q = new Float32Array(n);
  for (let i = 0; i < n; i++) q[i] = mA[i] * I[i] + mB[i];
  return q;
}
function integral(m, w, h) {
  const s = new Float64Array((w + 1) * (h + 1));
  for (let y = 0; y < h; y++) { let row = 0; for (let x = 0; x < w; x++) { row += m[y * w + x]; s[(y + 1) * (w + 1) + x + 1] = s[y * (w + 1) + x + 1] + row; } }
  return (x0, y0, x1, y1) => {
    x0 = Math.max(0, x0); y0 = Math.max(0, y0); x1 = Math.min(w, x1); y1 = Math.min(h, y1);
    if (x1 <= x0 || y1 <= y0) return 0;
    return s[y1 * (w + 1) + x1] - s[y0 * (w + 1) + x1] - s[y1 * (w + 1) + x0] + s[y0 * (w + 1) + x0];
  };
}

// ---------- склейка на полном кадре ----------
function sampler(src, sw, sh, W, H) {      // билинейная выборка поля sw×sh в пикселе полного кадра
  const xi = new Int32Array(W), xt = new Float32Array(W), yi = new Int32Array(H), yt = new Float32Array(H);
  for (let x = 0; x < W; x++) { const f = Math.min(sw - 1.001, Math.max(0, (x + 0.5) * sw / W - 0.5)); xi[x] = f | 0; xt[x] = f - xi[x]; }
  for (let y = 0; y < H; y++) { const f = Math.min(sh - 1.001, Math.max(0, (y + 0.5) * sh / H - 0.5)); yi[y] = f | 0; yt[y] = f - yi[y]; }
  return { xi, xt, yi, yt, sw, sh };
}
const sample = (S, a, x, y, stride = 1, c = 0) => {
  const x0 = S.xi[x], tx = S.xt[x], y0 = S.yi[y], ty = S.yt[y], w = S.sw, o = (y0 * w + x0) * stride + c;
  return (a[o] * (1 - tx) + a[o + stride] * tx) * (1 - ty) + (a[o + w * stride] * (1 - tx) + a[o + (w + 1) * stride] * tx) * ty;
};

function makeRng() {
  const s = new Uint32Array(4);
  globalThis.crypto.getRandomValues(s);
  let [a, b, c, d] = s; if (!(a | b | c | d)) a = 1;
  return () => { const t = b << 9; let r = (b * 5); r = ((r << 7) | (r >>> 25)) * 9; c ^= a; d ^= b; b ^= c; a ^= d; c ^= t; d = (d << 11) | (d >>> 21); return (r >>> 0) / 4294967296; };
}

function scrubPass(rgba, W, H, alpha, vw, vh, sigma) {
  const S = sampler(alpha, vw, vh, W, H), q = 4, qw = Math.ceil(W / q), qh = Math.ceil(H / q);
  const acc = new Float32Array(qw * qh * 3), wt = new Float32Array(qw * qh);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (sample(S, alpha, x, y) <= 0.5) continue;
    const c = ((y / q) | 0) * qw + ((x / q) | 0), o = (y * W + x) * 4;
    acc[c * 3] += rgba[o]; acc[c * 3 + 1] += rgba[o + 1]; acc[c * 3 + 2] += rgba[o + 2]; wt[c] += 1;
  }
  const s = sigma / q, wb = gaussBlur(wt, qw, qh, s), sm = new Float32Array(qw * qh * 3);
  for (let ch = 0; ch < 3; ch++) {
    const one = new Float32Array(qw * qh);
    for (let i = 0; i < one.length; i++) one[i] = acc[i * 3 + ch];
    const b = gaussBlur(one, qw, qh, s);
    for (let i = 0; i < one.length; i++) sm[i * 3 + ch] = b[i] / (wb[i] + 1e-4);
  }
  const Q = sampler(sm, qw, qh, W, H), rnd = makeRng();
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const a = sample(S, alpha, x, y);
    if (a <= 0) continue;
    const o = (y * W + x) * 4, n = (rnd() - 0.5) * 4;      // мелкое зерно ±2, намного мельче линий
    for (let ch = 0; ch < 3; ch++) rgba[o + ch] = rgba[o + ch] * (1 - a) + (sample(Q, sm, x, y, 3, ch) + n) * a;
  }
}

// ---------- весь конвейер ----------
// rgba: Uint8ClampedArray полного кадра (меняется на месте); view: RGBA рабочей копии vw×vh;
// conf: Float32Array 256×256, уверенность «кожа» от MediaPipe SelfieMulticlass (классы 2+3).
export function processImage(rgba, W, H, view, vw, vh, conf, progress = () => {}) {
  const t0 = Date.now(), nv = vw * vh, k = vw / W;
  progress("Ищу кожу", 5);
  const skinF = resizeBilinear(conf, 256, 256, vw, vh), skin = Uint8Array.from(skinF, (v) => (v > 0.5 ? 1 : 0));
  const near = dilate(skin, vw, vh, Math.round(NEAR * vw)), inner = erode(skin, vw, vh, 15);
  const nearSum = integral(near, vw, vh);
  const includeNear = (x, y, s) => nearSum(Math.floor(x * k), Math.floor(y * k), Math.ceil((x + s) * k), Math.ceil((y + s) * k)) > 0;

  progress("Ищу папиллярный узор", 15);
  const blocks = ridgeBlocks(rgba, W, H, includeNear), hits = blocks.filter((b) => score(b) >= THR);
  const tex = new Uint8Array(nv), periods = [], periodsAny = [];
  for (const b of hits) {
    const x0 = Math.floor(b.x * k), y0 = Math.floor(b.y * k), x1 = Math.min(vw, Math.ceil((b.x + b.s) * k)), y1 = Math.min(vh, Math.ceil((b.y + b.s) * k));
    for (let y = y0; y < y1; y++) tex.fill(1, y * vw + x0, y * vw + x1);
    periodsAny.push(b.period);
    const cx = Math.min(vw - 1, (x0 + x1) >> 1), cy = Math.min(vh - 1, (y0 + y1) >> 1);
    if (inner[cy * vw + cx]) periods.push(b.period);
  }
  const hard = new Uint8Array(nv);
  for (let i = 0; i < nv; i++) hard[i] = skin[i] || (tex[i] && near[i]) ? 1 : 0;

  progress("Уточняю край кожи", 45);
  const ring = dilate(hard, vw, vh, 15), lab = toLab(view, nv);
  const bgSel = Uint8Array.from(ring, (v, i) => v && !hard[i]);
  const ss = colorStats(lab, skin), bs = colorStats(lab, bgSel);
  let fg = new Uint8Array(nv);
  for (let i = 0; i < nv; i++) fg[i] = skin[i] || (hard[i] && (!ss || !bs || maha(lab, i, ss) < maha(lab, i, bs))) ? 1 : 0;
  fg = dilate(erode(fg, vw, vh, 2), vw, vh, 2);
  for (let i = 0; i < nv; i++) if (skin[i]) fg[i] = 1;
  const I = new Float32Array(nv);
  for (let i = 0; i < nv; i++) I[i] = (0.299 * view[i * 4] + 0.587 * view[i * 4 + 1] + 0.114 * view[i * 4 + 2]) / 255;
  const g = guided(I, Float32Array.from(fg), vw, vh, 3, 1e-4);
  let alpha = new Float32Array(nv);
  for (let i = 0; i < nv; i++) alpha[i] = Math.min(1, Math.max(0, (g[i] - 0.35) / 0.3));
  { // запас 1 px наружу
    const a2 = new Float32Array(nv);
    for (let y = 0; y < vh; y++) for (let x = 0; x < vw; x++) {
      let m = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const yy = y + dy, xx = x + dx;
        if (yy >= 0 && yy < vh && xx >= 0 && xx < vw) m = Math.max(m, alpha[yy * vw + xx]);
      }
      a2[y * vw + x] = m;
    }
    alpha = a2;
  }
  periods.sort((a, b) => a - b);
  const list = periods.length ? periods : periodsAny.sort((a, b) => a - b);
  const P = list.length ? list[list.length >> 1] : 18, sigma = Math.min(40, Math.max(8, P));

  progress("Стираю узор", 60);
  scrubPass(rgba, W, H, alpha, vw, vh, sigma);

  progress("Проверяю результат", 80);
  // проверяем только плитки целиком внутри кожи: плитка с контуром пальца даёт ложную тревогу
  const deepSum = integral(Uint8Array.from(alpha, (v) => (v > 0.9 ? 1 : 0)), vw, vh);
  const deep = (x, y, s) => {
    const x0 = Math.floor(x * k), y0 = Math.floor(y * k), x1 = Math.ceil((x + s) * k), y1 = Math.ceil((y + s) * k);
    return x1 <= vw && y1 <= vh && deepSum(x0, y0, x1, y1) >= (x1 - x0) * (y1 - y0);
  };
  // остатком считаем полосы с шагом от 0,5P до 3P: крупнее размытие с сигмой P сохраняет намеренно
  const isResidual = (b) => score(b) >= THR && b.period >= 0.5 * P && b.period <= 3 * P;
  let residual = ridgeBlocks(rgba, W, H, deep).filter(isResidual);
  const firstResidual = residual.length;
  let s2 = sigma;
  for (let pass = 0; pass < 2 && residual.length; pass++) {
    progress("Дочищаю остатки", 88 + pass * 5);
    // палец ближе к камере даёт линии крупнее: размываем по шагу самих остатков
    s2 = Math.min(60, Math.max(s2 * 1.5, ...residual.map((b) => b.period)));
    const extra = new Float32Array(nv);
    for (const b of residual) {
      const x0 = Math.floor(b.x * k), y0 = Math.floor(b.y * k), x1 = Math.min(vw, Math.ceil((b.x + b.s) * k)), y1 = Math.min(vh, Math.ceil((b.y + b.s) * k));
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) extra[y * vw + x] = alpha[y * vw + x];
    }
    scrubPass(rgba, W, H, extra, vw, vh, s2);
    residual = ridgeBlocks(rgba, W, H, deep).filter(isResidual);
  }

  const overlay = new Uint8ClampedArray(view);
  for (let i = 0; i < nv; i++) if (alpha[i] > 0.5) { overlay[i * 4] = overlay[i * 4] * 0.45 + 35; overlay[i * 4 + 1] = overlay[i * 4 + 1] * 0.45 + 132; overlay[i * 4 + 2] = overlay[i * 4 + 2] * 0.45 + 119; }
  for (const b of residual) {
    const x0 = Math.floor(b.x * k), y0 = Math.floor(b.y * k), x1 = Math.min(vw, Math.ceil((b.x + b.s) * k)), y1 = Math.min(vh, Math.ceil((b.y + b.s) * k));
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { const o = (y * vw + x) * 4; overlay[o] = 255; overlay[o + 1] = 46; overlay[o + 2] = 136; }
  }
  progress("Готово", 100);
  let maskFrac = 0; for (let i = 0; i < nv; i++) maskFrac += alpha[i] > 0.5 ? 1 : 0;
  return {
    overlay,
    stats: { P, sigma, sigmaMax: s2, maskFrac: maskFrac / nv, blocks: blocks.length, hits: hits.length, firstResidual, residual: residual.length,
      residualBlocks: residual.map((b) => ({ x: b.x, y: b.y, s: b.s, period: +b.period.toFixed(1), amp: +b.amp.toFixed(2), coh: +b.coh.toFixed(2), peak: +b.peak.toFixed(2) })), ms: Date.now() - t0 },
  };
}

export { VIEW };
