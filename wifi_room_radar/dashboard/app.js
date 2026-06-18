/* wifi-room-radar live dashboard.
 *
 * Vanilla ES module, zero dependencies, works offline. Talks to the server
 * over a websocket at /ws; message envelope is
 *   {"type": "info"|"state", "data": {...}}
 * where "state" data fields are exactly the SensingState dataclass fields
 * (see wifi_room_radar/types.py).
 *
 * Structure: small renderer classes (Spectrogram, RoomMap, LineChart) that
 * each own one canvas, a tiny state store, and a websocket client with
 * exponential-backoff reconnect. Rendering is requestAnimationFrame-driven
 * but only repaints when new data arrived or the window resized.
 */

const $ = (id) => document.getElementById(id);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

const MONO = '11px ui-monospace, "Cascadia Code", Consolas, monospace';
const MONO_SMALL = '10px ui-monospace, "Cascadia Code", Consolas, monospace';

/* Canvas colour palette (kept in sync with style.css custom properties). */
const C = {
  grid: 'rgba(118, 131, 154, 0.16)',
  frame: 'rgba(219, 228, 240, 0.22)',
  text: '#76839a',
  ink: '#dbe4f0',
  accent: '#34e0b4',
  cyan: '#4fc3f7',
  warn: '#ffb648',
  track: '#7df0ff',
  ghost: 'rgba(219, 228, 240, 0.55)',
};

/* Room-level activity label → display glyph (deliberately ascii-ish to
 * match the control-room voice). Unknown labels fall back to the dot. */
const ACTIVITY_GLYPHS = {
  idle: '·',
  micro: '~',
  walking: '>>',
  gesturing: '*',
};

const TRAIL_KEEP_S = 20; // seconds of per-track position history to draw
const TRAIL_DROP_S = 5; // forget a track's trail after this long unseen
const VITALS_MIN_CONF = 0.3; // below this, per-track vitals are noise

/* ------------------------------------------------------------------ */
/* Viridis-like colormap: piecewise-linear LUT over 9 anchor colours.  */
/* ------------------------------------------------------------------ */

const VIRIDIS = (() => {
  const stops = [
    [0.0, 68, 1, 84], [0.125, 72, 40, 120], [0.25, 59, 82, 139],
    [0.375, 46, 110, 142], [0.5, 33, 145, 140], [0.625, 53, 183, 121],
    [0.75, 94, 201, 98], [0.875, 170, 220, 50], [1.0, 253, 231, 37],
  ];
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let j = 0;
    while (j < stops.length - 2 && t > stops[j + 1][0]) j++;
    const [t0, r0, g0, b0] = stops[j];
    const [t1, r1, g1, b1] = stops[j + 1];
    const u = clamp((t - t0) / (t1 - t0), 0, 1);
    lut[i * 3] = r0 + (r1 - r0) * u;
    lut[i * 3 + 1] = g0 + (g1 - g0) * u;
    lut[i * 3 + 2] = b0 + (b1 - b0) * u;
  }
  return lut;
})();

function viridisRgb(t) {
  const i = clamp(Math.round(t * 255), 0, 255) * 3;
  return [VIRIDIS[i], VIRIDIS[i + 1], VIRIDIS[i + 2]];
}

/* Resize a canvas backing store to its CSS size * devicePixelRatio and
 * return [ctx, cssWidth, cssHeight] with the transform set so drawing
 * happens in CSS pixels. */
function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 300;
  const h = canvas.clientHeight || 150;
  const bw = Math.max(1, Math.round(w * dpr));
  const bh = Math.max(1, Math.round(h * dpr));
  if (canvas.width !== bw || canvas.height !== bh) {
    canvas.width = bw;
    canvas.height = bh;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [ctx, w, h];
}

/* ------------------------------------------------------------------ */
/* Spectrogram: scrolling Doppler waterfall.                            */
/* New columns are painted at the right edge of a fixed-width offscreen */
/* buffer (1 px per state update) which is then stretched to the panel. */
/* ------------------------------------------------------------------ */

class Spectrogram {
  constructor(canvas, historyCols = 480) {
    this.canvas = canvas;
    this.cols = historyCols;
    this.buf = document.createElement('canvas');
    this.nbins = 0;
    this.freqs = null;
    this.dbMin = NaN; // adaptive display range (EMA of per-column extrema)
    this.dbMax = NaN;
  }

  /* Append one doppler column (dB magnitudes, fft-shifted neg..pos). */
  push(freqs, column) {
    const n = column.length;
    if (!n) return;
    this.freqs = freqs;
    const b = this.buf.getContext('2d');
    if (n !== this.nbins) {
      this.nbins = n;
      this.buf.width = this.cols;
      this.buf.height = n;
      b.fillStyle = '#000';
      b.fillRect(0, 0, this.cols, n);
    }
    let lo = Infinity, hi = -Infinity;
    for (const v of column) {
      if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
    }
    if (!Number.isFinite(lo)) return;
    if (!Number.isFinite(this.dbMin)) { this.dbMin = lo; this.dbMax = hi; }
    // Track the noise floor slowly and peaks faster so transient motion
    // bursts brighten without permanently rescaling the whole waterfall.
    this.dbMin += (lo - this.dbMin) * 0.02;
    this.dbMax += (hi - this.dbMax) * 0.08;
    if (this.dbMax - this.dbMin < 20) {
      const mid = (this.dbMax + this.dbMin) / 2;
      this.dbMin = mid - 10;
      this.dbMax = mid + 10;
    }
    b.drawImage(this.buf, -1, 0); // scroll history one column left
    const img = b.createImageData(1, this.nbins);
    const span = this.dbMax - this.dbMin;
    for (let i = 0; i < n; i++) {
      const v = column[n - 1 - i]; // row 0 of the canvas = most positive Hz
      const t = clamp((v - this.dbMin) / span, 0, 1);
      const ix = clamp(Math.round(t * 255), 0, 255) * 3;
      const k = i * 4;
      img.data[k] = VIRIDIS[ix];
      img.data[k + 1] = VIRIDIS[ix + 1];
      img.data[k + 2] = VIRIDIS[ix + 2];
      img.data[k + 3] = 255;
    }
    b.putImageData(img, this.cols - 1, 0);
  }

  draw() {
    const [ctx, w, h] = fitCanvas(this.canvas);
    ctx.clearRect(0, 0, w, h);
    const ml = 78, mr = 64, mt = 10, mb = 12;
    const pw = Math.max(10, w - ml - mr);
    const ph = Math.max(10, h - mt - mb);

    ctx.fillStyle = 'rgba(0, 0, 0, 0.35)';
    ctx.fillRect(ml, mt, pw, ph);
    if (this.nbins > 0) {
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(this.buf, 0, 0, this.cols, this.nbins, ml, mt, pw, ph);
    }

    // 0 Hz centre line.
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.30)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(ml, mt + ph / 2);
    ctx.lineTo(ml + pw, mt + ph / 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Frequency axis.
    ctx.font = MONO;
    ctx.fillStyle = C.text;
    ctx.textAlign = 'right';
    const fTop = this.freqs && this.freqs.length ? this.freqs[this.freqs.length - 1] : 0;
    const fBot = this.freqs && this.freqs.length ? this.freqs[0] : 0;
    ctx.fillText(`+${Math.abs(fTop).toFixed(0)} Hz`, ml - 8, mt + 10);
    ctx.fillText('approaching', ml - 8, mt + 22);
    ctx.fillText('0 Hz', ml - 8, mt + ph / 2 + 4);
    ctx.fillText(`−${Math.abs(fBot).toFixed(0)} Hz`, ml - 8, mt + ph - 12);
    ctx.fillText('receding', ml - 8, mt + ph);

    // dB colour scale.
    const cbx = ml + pw + 14, cbw = 10;
    for (let i = 0; i < ph; i++) {
      const [r, g, b] = viridisRgb(1 - i / ph);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(cbx, mt + i, cbw, 1.5);
    }
    ctx.strokeStyle = C.frame;
    ctx.strokeRect(cbx + 0.5, mt + 0.5, cbw, ph);
    ctx.textAlign = 'left';
    if (Number.isFinite(this.dbMax)) {
      ctx.fillText(`${this.dbMax.toFixed(0)} dB`, cbx + cbw + 5, mt + 10);
      ctx.fillText(`${this.dbMin.toFixed(0)} dB`, cbx + cbw + 5, mt + ph);
    } else {
      ctx.fillText('dB', cbx + cbw + 5, mt + 10);
    }
    ctx.strokeRect(ml + 0.5, mt + 0.5, pw, ph);
  }
}

/* ------------------------------------------------------------------ */
/* RoomMap: plan view of the room with occupancy heatmap, radio        */
/* markers, confirmed tracks and ground-truth ghosts. Grid row 0 sits  */
/* at y = 0 (bottom of the room) so the map is drawn with y pointing   */
/* up, matching the metre coordinate convention.                       */
/* ------------------------------------------------------------------ */

class RoomMap {
  constructor(canvas) {
    this.canvas = canvas;
    this.cells = document.createElement('canvas'); // 1 px per grid cell
  }

  /* Look up a geometry key in info, tolerating nesting under "radio"
   * or "geometry" depending on how the provider assembled its info. */
  static geo(info, key) {
    if (!info) return null;
    if (info[key] != null) return info[key];
    if (info.radio && info.radio[key] != null) return info.radio[key];
    if (info.geometry && info.geometry[key] != null) return info.geometry[key];
    return null;
  }

  /* Greedy spatial clustering: antenna elements within `eps` metres of a
   * cluster's running centroid belong to the same physical node. Used to
   * collapse multi-node rx_positions (a flat list of element positions
   * across all nodes) into one labelled marker per box. */
  static clusterNodes(points, eps = 0.2) {
    const acc = []; // {sx, sy, n} running sums per cluster
    for (const p of points) {
      if (!p || p.length < 2 || !Number.isFinite(p[0]) || !Number.isFinite(p[1])) continue;
      let hit = null;
      for (const c of acc) {
        if (Math.hypot(p[0] - c.sx / c.n, p[1] - c.sy / c.n) <= eps) { hit = c; break; }
      }
      if (hit) { hit.sx += p[0]; hit.sy += p[1]; hit.n += 1; }
      else acc.push({ sx: p[0], sy: p[1], n: 1 });
    }
    return acc.map((c) => [c.sx / c.n, c.sy / c.n]);
  }

  draw(state, info, trails) {
    const [ctx, w, h] = fitCanvas(this.canvas);
    ctx.clearRect(0, 0, w, h);
    if (!state) return;
    const size = state.room_size || [0, 0];
    const W = size[0], D = size[1];
    if (!(W > 0 && D > 0)) {
      ctx.font = MONO;
      ctx.fillStyle = C.text;
      ctx.textAlign = 'center';
      ctx.fillText('room geometry unknown', w / 2, h / 2);
      return;
    }
    const pad = 26;
    const s = Math.min((w - 2 * pad) / W, (h - 2 * pad) / D);
    const rw = W * s, rh = D * s;
    const ox = (w - rw) / 2, oy = (h - rh) / 2;
    const px = (x, y) => [ox + x * s, oy + (D - y) * s]; // y axis points up

    ctx.fillStyle = 'rgba(255, 255, 255, 0.02)';
    ctx.fillRect(ox, oy, rw, rh);

    // Occupancy heatmap: paint the grid at native resolution offscreen,
    // then stretch with smoothing for a soft heat-blob look.
    const g = state.occupancy_grid;
    if (g && g.length && g[0].length) {
      const rows = g.length, cols = g[0].length;
      if (this.cells.width !== cols || this.cells.height !== rows) {
        this.cells.width = cols;
        this.cells.height = rows;
      }
      const cctx = this.cells.getContext('2d');
      const img = cctx.createImageData(cols, rows);
      for (let r = 0; r < rows; r++) {
        const ro = rows - 1 - r; // grid row 0 = bottom of room = bottom of map
        for (let c = 0; c < cols; c++) {
          const v = clamp(g[r][c], 0, 1);
          const ix = clamp(Math.round(v * 255), 0, 255) * 3;
          const k = (ro * cols + c) * 4;
          img.data[k] = VIRIDIS[ix];
          img.data[k + 1] = VIRIDIS[ix + 1];
          img.data[k + 2] = VIRIDIS[ix + 2];
          img.data[k + 3] = Math.round(225 * Math.pow(v, 0.7));
        }
      }
      cctx.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(this.cells, ox, oy, rw, rh);
    }

    // Subtle metre grid + axis labels.
    ctx.strokeStyle = C.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = 1; x < W; x += 1) {
      const [X] = px(x, 0);
      ctx.moveTo(X, oy);
      ctx.lineTo(X, oy + rh);
    }
    for (let y = 1; y < D; y += 1) {
      const Y = px(0, y)[1];
      ctx.moveTo(ox, Y);
      ctx.lineTo(ox + rw, Y);
    }
    ctx.stroke();
    ctx.font = MONO;
    ctx.fillStyle = C.text;
    ctx.textAlign = 'center';
    for (let x = 0; x <= W + 1e-9; x += 1) {
      ctx.fillText(String(x), px(x, 0)[0], oy + rh + 14);
    }
    ctx.textAlign = 'right';
    for (let y = 0; y <= D + 1e-9; y += 1) {
      ctx.fillText(String(y), ox - 6, px(0, y)[1] + 4);
    }
    ctx.strokeStyle = C.frame;
    ctx.strokeRect(ox + 0.5, oy + 0.5, rw, rh);

    // Radio markers (when the provider exposes geometry in info).
    const tx = RoomMap.geo(info, 'tx_pos');
    if (tx && tx.length >= 2) this.marker(ctx, px(tx[0], tx[1]), C.warn, 'TX', 'diamond');
    const rxs = RoomMap.geo(info, 'rx_positions');
    if (Array.isArray(rxs) && rxs.length) {
      // Show every antenna element faintly, plus one labelled marker per
      // node (RX1, RX2, ... when more than one node is contributing).
      ctx.fillStyle = 'rgba(79, 195, 247, 0.45)';
      for (const p of rxs) {
        if (p && p.length >= 2) {
          const [ex, ey] = px(p[0], p[1]);
          ctx.fillRect(ex - 1.5, ey - 1.5, 3, 3);
        }
      }
      const nodes = RoomMap.clusterNodes(rxs);
      nodes.forEach((n, i) => {
        this.marker(ctx, px(n[0], n[1]), C.cyan, nodes.length > 1 ? `RX${i + 1}` : 'RX', 'square');
      });
    }

    // Track trails: fading polylines of recent positions (client-side
    // history, drawn beneath ghosts and the live track dots).
    if (trails && trails.size) {
      const now = state.timestamp;
      ctx.lineCap = 'round';
      ctx.lineWidth = 1.4;
      ctx.strokeStyle = C.track;
      for (const tr of trails.values()) {
        const pts = tr.pts;
        for (let i = 1; i < pts.length; i++) {
          const age = now - pts[i].t;
          ctx.globalAlpha = 0.45 * clamp(1 - age / TRAIL_KEEP_S, 0, 1);
          const [x0, y0] = px(pts[i - 1].x, pts[i - 1].y);
          const [x1, y1] = px(pts[i].x, pts[i].y);
          ctx.beginPath();
          ctx.moveTo(x0, y0);
          ctx.lineTo(x1, y1);
          ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;
    }

    // Ground-truth ghosts (sim only): hollow dashed circles.
    const gt = state.ground_truth;
    if (gt && Array.isArray(gt.people)) {
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = C.ghost;
      ctx.lineWidth = 1.2;
      for (const p of gt.people) {
        const [X, Y] = px(p.x, p.y);
        ctx.beginPath();
        ctx.arc(X, Y, Math.max(6, 0.22 * s), 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(X, Y, 1.5, 0, Math.PI * 2);
        ctx.fillStyle = C.ghost;
        ctx.fill();
      }
      ctx.setLineDash([]);
    }

    // Confirmed tracks: bright glow circles, id labels, velocity vectors.
    for (const t of state.tracks || []) {
      const [X, Y] = px(t.x, t.y);
      const speed = Math.hypot(t.vx, t.vy);
      if (speed > 0.05) {
        const [X2, Y2] = px(t.x + t.vx * 0.8, t.y + t.vy * 0.8); // 0.8 s lead
        ctx.strokeStyle = C.track;
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.moveTo(X, Y);
        ctx.lineTo(X2, Y2);
        ctx.stroke();
        const a = Math.atan2(Y2 - Y, X2 - X);
        ctx.fillStyle = C.track;
        ctx.beginPath();
        ctx.moveTo(X2, Y2);
        ctx.lineTo(X2 - 7 * Math.cos(a - 0.4), Y2 - 7 * Math.sin(a - 0.4));
        ctx.lineTo(X2 - 7 * Math.cos(a + 0.4), Y2 - 7 * Math.sin(a + 0.4));
        ctx.closePath();
        ctx.fill();
      }
      const r = Math.max(6, 0.18 * s);
      ctx.save();
      ctx.shadowColor = C.track;
      ctx.shadowBlur = 14;
      ctx.globalAlpha = 0.35 + 0.65 * clamp(t.confidence, 0, 1);
      ctx.fillStyle = C.track;
      ctx.beginPath();
      ctx.arc(X, Y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
      ctx.fillStyle = '#06121a';
      ctx.font = `bold ${MONO}`;
      ctx.textAlign = 'center';
      ctx.fillText(String(t.track_id), X, Y + 3.5);

      // Per-person vitals tag: shown only when this track's breathing
      // estimate is usable; heart rate joins in when it is too.
      const br = t.breathing;
      if (br && br.confidence >= VITALS_MIN_CONF && br.rate_bpm > 0) {
        const hb = t.heartbeat;
        const bits = [];
        if (hb && hb.confidence >= VITALS_MIN_CONF && hb.rate_bpm > 0) {
          bits.push(`♥ ${hb.rate_bpm.toFixed(0)}`);
        }
        bits.push(`⌁ ${br.rate_bpm.toFixed(0)} bpm`);
        const lbl = bits.join(' | ');
        ctx.font = MONO_SMALL;
        ctx.textAlign = 'left';
        const tw = ctx.measureText(lbl).width;
        let lx = X + r + 8;
        if (lx + tw + 6 > w) lx = Math.max(2, X - r - 10 - tw); // flip near the right edge
        const ly = Math.max(14, Y - r - 6);
        ctx.fillStyle = 'rgba(11, 14, 19, 0.78)';
        ctx.fillRect(lx - 3, ly - 9, tw + 6, 13);
        ctx.lineWidth = 1;
        ctx.strokeStyle = 'rgba(125, 240, 255, 0.30)';
        ctx.strokeRect(lx - 2.5, ly - 8.5, tw + 5, 12);
        ctx.fillStyle = C.ink;
        ctx.fillText(lbl, lx, ly + 1);
      }
    }
  }

  marker(ctx, [X, Y], color, label, shape) {
    ctx.fillStyle = color;
    ctx.beginPath();
    if (shape === 'diamond') {
      ctx.moveTo(X, Y - 6);
      ctx.lineTo(X + 6, Y);
      ctx.lineTo(X, Y + 6);
      ctx.lineTo(X - 6, Y);
      ctx.closePath();
      ctx.fill();
    } else {
      ctx.fillRect(X - 4, Y - 4, 8, 8);
    }
    if (label) {
      ctx.font = MONO;
      ctx.textAlign = 'left';
      ctx.fillText(label, X + 8, Y + 4);
    }
  }
}

/* ------------------------------------------------------------------ */
/* LineChart: smooth single-series chart with auto y-scaling, used for */
/* the breathing waveform and the heart-rate trend sparkline.          */
/* ------------------------------------------------------------------ */

class LineChart {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.o = Object.assign({ stroke: C.accent, fill: true, lineWidth: 1.8 }, opts);
  }

  draw(samples, label) {
    const [ctx, w, h] = fitCanvas(this.canvas);
    ctx.clearRect(0, 0, w, h);
    ctx.font = MONO;
    if (!samples || samples.length < 2) {
      ctx.fillStyle = C.text;
      ctx.textAlign = 'center';
      ctx.fillText('no signal', w / 2, h / 2);
      return;
    }
    const m = { l: 10, r: 10, t: 16, b: 10 };
    let lo = Infinity, hi = -Infinity;
    for (const v of samples) {
      if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
    }
    if (!Number.isFinite(lo)) return;
    if (hi - lo < 1e-9) { hi += 0.5; lo -= 0.5; }
    const padY = (hi - lo) * 0.12;
    lo -= padY;
    hi += padY;
    const n = samples.length;
    const X = (i) => m.l + (w - m.l - m.r) * (i / (n - 1));
    const Y = (v) => m.t + (h - m.t - m.b) * (1 - (v - lo) / (hi - lo));

    // Smooth path through quadratic midpoints.
    ctx.beginPath();
    ctx.moveTo(X(0), Y(samples[0]));
    for (let i = 1; i < n - 1; i++) {
      ctx.quadraticCurveTo(X(i), Y(samples[i]), (X(i) + X(i + 1)) / 2, (Y(samples[i]) + Y(samples[i + 1])) / 2);
    }
    ctx.lineTo(X(n - 1), Y(samples[n - 1]));
    ctx.strokeStyle = this.o.stroke;
    ctx.lineWidth = this.o.lineWidth;
    ctx.lineJoin = 'round';
    ctx.stroke();

    if (this.o.fill) {
      ctx.lineTo(X(n - 1), h - m.b);
      ctx.lineTo(X(0), h - m.b);
      ctx.closePath();
      const gr = ctx.createLinearGradient(0, m.t, 0, h - m.b);
      gr.addColorStop(0, this.o.stroke + '3a'); // 6-digit hex + alpha byte
      gr.addColorStop(1, this.o.stroke + '00');
      ctx.fillStyle = gr;
      ctx.fill();
    }

    if (label) {
      ctx.fillStyle = C.ink;
      ctx.textAlign = 'right';
      ctx.fillText(label, w - m.r, m.t - 4);
    }
  }
}

/* ------------------------------------------------------------------ */
/* Timeline: rolling 5-minute strip with three lanes — presence (filled */
/* band), motion level (area, 0..1) and breathing rate (line, only      */
/* while a confident estimate exists). Samples land in a ~1 Hz ring     */
/* buffer keyed by wall-clock time, so stream stalls show up as gaps    */
/* instead of stretched data.                                           */
/* ------------------------------------------------------------------ */

class Timeline {
  constructor(canvas, windowS = 300) {
    this.canvas = canvas;
    this.windowS = windowS;
    this.samples = []; // {t, presence, motion, bpm|null}, oldest first
    this._lastSample = -Infinity;
  }

  /* Sample the state at ~1 Hz; safe to call on every websocket message. */
  push(s) {
    const now = performance.now() / 1000;
    if (now - this._lastSample < 0.95) return;
    this._lastSample = now;
    const br = s.breathing;
    const usable = br && br.confidence >= VITALS_MIN_CONF && br.rate_bpm > 0;
    this.samples.push({
      t: now,
      presence: !!s.presence,
      motion: clamp(s.motion_level || 0, 0, 1),
      bpm: usable ? br.rate_bpm : null,
    });
    while (this.samples.length && now - this.samples[0].t > this.windowS + 2) {
      this.samples.shift();
    }
  }

  draw() {
    const [ctx, w, h] = fitCanvas(this.canvas);
    ctx.clearRect(0, 0, w, h);
    ctx.font = MONO;
    const ml = 70, mr = 56, mt = 8, mb = 16;
    const pw = Math.max(10, w - ml - mr);
    const ph = Math.max(10, h - mt - mb);
    const now = performance.now() / 1000;
    const X = (t) => ml + pw * clamp(1 - (now - t) / this.windowS, 0, 1);

    ctx.fillStyle = 'rgba(0, 0, 0, 0.25)';
    ctx.fillRect(ml, mt, pw, ph);

    // Lane geometry, top to bottom.
    const lanes = {
      presence: { y: mt + 2, h: ph * 0.16 },
      motion: { y: mt + ph * 0.26, h: ph * 0.32 },
      breath: { y: mt + ph * 0.66, h: ph * 0.32 },
    };

    // Minute grid + minimal time axis.
    ctx.strokeStyle = C.grid;
    ctx.beginPath();
    for (let m = 1; m < this.windowS / 60; m++) {
      const gx = ml + pw * ((m * 60) / this.windowS);
      ctx.moveTo(gx, mt);
      ctx.lineTo(gx, mt + ph);
    }
    ctx.stroke();
    ctx.fillStyle = C.text;
    ctx.textAlign = 'center';
    ctx.fillText(`-${Math.round(this.windowS / 60)}m`, ml, mt + ph + 12);
    ctx.fillText('now', ml + pw, mt + ph + 12);

    // Lane captions.
    ctx.textAlign = 'right';
    ctx.fillText('presence', ml - 8, lanes.presence.y + lanes.presence.h / 2 + 4);
    ctx.fillText('motion', ml - 8, lanes.motion.y + lanes.motion.h / 2 + 4);
    ctx.fillText('breath', ml - 8, lanes.breath.y + lanes.breath.h / 2 + 4);

    const S = this.samples;
    const GAP = 3; // s between samples beyond which the series breaks

    // Presence: filled band while true.
    ctx.fillStyle = 'rgba(52, 224, 180, 0.5)';
    for (let i = 0; i < S.length; i++) {
      if (!S[i].presence) continue;
      const t0 = S[i].t;
      const t1 = i + 1 < S.length ? Math.min(S[i + 1].t, t0 + GAP) : Math.min(now, t0 + GAP);
      ctx.fillRect(X(t0), lanes.presence.y, Math.max(1, X(t1) - X(t0)), lanes.presence.h);
    }

    // Motion: area + line over a fixed 0..1 scale.
    const my = (v) => lanes.motion.y + lanes.motion.h * (1 - clamp(v, 0, 1));
    this.series(ctx, S, (p) => p.motion, my, GAP, X, C.cyan, lanes.motion.y + lanes.motion.h);

    // Breathing: line auto-scaled over observed bpm, broken while unreliable.
    let lo = Infinity, hi = -Infinity;
    for (const p of S) {
      if (p.bpm != null) { if (p.bpm < lo) lo = p.bpm; if (p.bpm > hi) hi = p.bpm; }
    }
    if (Number.isFinite(lo)) {
      if (hi - lo < 4) { const mid = (hi + lo) / 2; lo = mid - 2; hi = mid + 2; }
      const by = (v) => lanes.breath.y + lanes.breath.h * (1 - clamp((v - lo) / (hi - lo), 0, 1));
      this.series(ctx, S, (p) => p.bpm, by, GAP, X, C.accent, null);
    }

    // Current values at the right edge of each lane.
    const last = S.length ? S[S.length - 1] : null;
    const fresh = last && now - last.t < GAP * 2;
    ctx.textAlign = 'left';
    ctx.fillStyle = fresh && last.presence ? C.accent : C.text;
    ctx.fillText(fresh ? (last.presence ? 'OCC' : 'CLR') : '—', ml + pw + 8, lanes.presence.y + lanes.presence.h / 2 + 4);
    ctx.fillStyle = fresh ? C.cyan : C.text;
    ctx.fillText(fresh ? `${Math.round(last.motion * 100)}%` : '—', ml + pw + 8, lanes.motion.y + lanes.motion.h / 2 + 4);
    ctx.fillStyle = fresh && last.bpm != null ? C.accent : C.text;
    ctx.fillText(fresh && last.bpm != null ? last.bpm.toFixed(1) : '—', ml + pw + 8, lanes.breath.y + lanes.breath.h / 2 + 4);

    ctx.strokeStyle = C.frame;
    ctx.strokeRect(ml + 0.5, mt + 0.5, pw, ph);
  }

  /* Stroke one lane's series, breaking the path on missing samples or
   * sampling gaps; optionally fill the area down to `baseY`. */
  series(ctx, S, val, Y, gap, X, color, baseY) {
    let run = []; // [x, y] points of the current contiguous segment
    const flush = () => {
      if (run.length < 2) { run = []; return; }
      ctx.beginPath();
      ctx.moveTo(run[0][0], run[0][1]);
      for (let i = 1; i < run.length; i++) ctx.lineTo(run[i][0], run[i][1]);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.4;
      ctx.lineJoin = 'round';
      ctx.stroke();
      if (baseY != null) {
        ctx.lineTo(run[run.length - 1][0], baseY);
        ctx.lineTo(run[0][0], baseY);
        ctx.closePath();
        ctx.fillStyle = color + '26'; // 6-digit hex + alpha byte
        ctx.fill();
      }
      run = [];
    };
    for (let i = 0; i < S.length; i++) {
      const v = val(S[i]);
      if (v == null || !Number.isFinite(v)) { flush(); continue; }
      if (run.length && S[i].t - S[i - 1].t > gap) flush();
      run.push([X(S[i].t), Y(v)]);
    }
    flush();
  }
}

/* ------------------------------------------------------------------ */
/* State store + DOM bindings.                                          */
/* ------------------------------------------------------------------ */

const store = {
  info: null,
  state: null,
  heartHist: [], // client-side bpm history for the trend sparkline
  trails: new Map(), // track_id -> {pts: [{x, y, t}], last} for the room map
  alerts: [], // last received alerts list (for the 1 Hz title/age ticker)
  uiRate: 0, // measured state messages per second (EMA)
  _lastMsg: null,
  _trailClock: null, // last stream timestamp seen by updateTrails
};

const spectro = new Spectrogram($('spectro-canvas'));
const roomMap = new RoomMap($('room-canvas'));
const breathChart = new LineChart($('breath-canvas'), { stroke: C.accent });
const heartChart = new LineChart($('heart-canvas'), { stroke: '#ff7d92' });
const timeline = new Timeline($('timeline-canvas'));

let dirty = true;
window.addEventListener('resize', () => { dirty = true; });

function onInfo(info) {
  store.info = info;
  renderSourceInfo(info);
  renderFooter();
  dirty = true;
}

function onState(s) {
  const now = performance.now();
  if (store._lastMsg != null) {
    const dt = (now - store._lastMsg) / 1000;
    if (dt > 0) {
      store.uiRate = store.uiRate ? store.uiRate * 0.85 + 0.15 / dt : 1 / dt;
    }
  }
  store._lastMsg = now;
  store.state = s;
  if (s.heartbeat && s.heartbeat.rate_bpm > 0) {
    store.heartHist.push(s.heartbeat.rate_bpm);
    if (store.heartHist.length > 300) store.heartHist.shift();
  }
  if (s.doppler_column && s.doppler_column.length) {
    spectro.push(s.doppler_freqs || [], s.doppler_column);
  }
  updateTrails(s);
  timeline.push(s);
  document.body.classList.add('has-data');
  renderCards(s);
  renderAlerts(s);
  renderFooter();
  dirty = true;
}

/* Maintain per-track position history for the room-map trails. Keyed by
 * stream time; if the timestamp jumps backwards (server restart) the
 * history is dropped instead of drawing bogus cross-room lines. */
function updateTrails(s) {
  const now = s.timestamp;
  if (!Number.isFinite(now)) return;
  if (store._trailClock != null && now < store._trailClock - 1) store.trails.clear();
  store._trailClock = now;
  for (const t of s.tracks || []) {
    if (t == null || t.track_id == null) continue;
    let tr = store.trails.get(t.track_id);
    if (!tr) {
      tr = { pts: [], last: now };
      store.trails.set(t.track_id, tr);
    }
    tr.last = now;
    tr.pts.push({ x: t.x, y: t.y, t: now });
  }
  for (const [id, tr] of store.trails) {
    while (tr.pts.length && now - tr.pts[0].t > TRAIL_KEEP_S) tr.pts.shift();
    if (now - tr.last > TRAIL_DROP_S) store.trails.delete(id);
  }
}

function renderSourceInfo(info) {
  const bits = [];
  if (info.type) bits.push(info.type);
  if (info.sample_rate != null) bits.push(`${Number(info.sample_rate).toFixed(0)} pkt/s`);
  if (info.n_rx != null && info.n_tx != null) bits.push(`${info.n_rx}×${info.n_tx} ant`);
  if (info.n_subcarriers != null) bits.push(`${info.n_subcarriers} sub`);
  if (info.carrier_freq_ghz != null) bits.push(`${Number(info.carrier_freq_ghz).toFixed(2)} GHz`);
  $('source-info').textContent = bits.join(' · ') || 'no source';

  // Multi-node setups advertise n_nodes; single-node (or older) servers
  // omit it and the chip stays hidden.
  const n = Number(info.n_nodes);
  const nodesEl = $('node-count');
  if (Number.isFinite(n) && n > 1) {
    nodesEl.textContent = `${n} nodes`;
    nodesEl.hidden = false;
  } else {
    nodesEl.hidden = true;
  }
}

function setVital(prefix, v) {
  const bpmEl = $(`${prefix}-bpm`);
  const confEl = $(`${prefix}-conf`);
  const barEl = $(`${prefix}-conf-bar`);
  const card = $(`card-${prefix}`);
  if (v && v.rate_bpm > 0) {
    bpmEl.textContent = v.rate_bpm.toFixed(1);
    const conf = clamp(v.confidence || 0, 0, 1);
    confEl.textContent = `${Math.round(conf * 100)}%`;
    barEl.style.width = `${Math.round(conf * 100)}%`;
    card.classList.toggle('lowconf', conf < 0.3);
  } else {
    bpmEl.textContent = '—';
    confEl.textContent = '—';
    barEl.style.width = '0%';
    card.classList.add('lowconf');
  }
}

function renderCards(s) {
  $('presence-value').textContent = s.presence ? 'OCCUPIED' : 'CLEAR';
  $('card-presence').classList.toggle('on', !!s.presence);

  const pct = Math.round(clamp(s.motion_level || 0, 0, 1) * 100);
  $('motion-pct').textContent = `${pct}%`;
  $('motion-bar').style.width = `${pct}%`;
  $('motion-badge').classList.toggle('show', !!s.motion_detected);
  $('card-motion').classList.toggle('on', !!s.motion_detected);

  // Room-level activity label (older servers omit the field entirely).
  const act = typeof s.activity === 'string' ? s.activity : null;
  $('activity-glyph').textContent = (act && ACTIVITY_GLYPHS[act]) || ACTIVITY_GLYPHS.idle;
  $('activity-value').textContent = act || '—';
  const actCard = $('card-activity');
  actCard.classList.remove('act-idle', 'act-micro', 'act-walking', 'act-gesturing');
  if (act && ACTIVITY_GLYPHS[act]) actCard.classList.add(`act-${act}`);

  setVital('breath', s.breathing);
  setVital('heart', s.heartbeat);

  const sv = s.subvocal;
  const score = sv ? clamp(sv.activity_score || 0, 0, 1) : 0;
  $('subvocal-score').textContent = sv ? `${Math.round(score * 100)}%` : '—';
  $('subvocal-bar').style.width = `${Math.round(score * 100)}%`;
  $('card-subvocal').classList.toggle('active', !!(sv && sv.active));
}

function renderFooter() {
  const s = store.state, i = store.info;
  $('foot-ts').textContent = s ? `t = ${s.timestamp.toFixed(1)} s` : 't = —';
  $('foot-rate').textContent = store.uiRate ? `ui ${store.uiRate.toFixed(1)} Hz` : 'ui — Hz';
  if (i) {
    const bits = [i.type || 'source'];
    if (i.sample_rate != null) bits.push(`${Number(i.sample_rate).toFixed(0)} pkt/s`);
    if (i.n_rx != null) bits.push(`${i.n_rx} rx`);
    if (i.n_subcarriers != null) bits.push(`${i.n_subcarriers} subcarriers`);
    if (i.bandwidth_mhz != null) bits.push(`${Number(i.bandwidth_mhz).toFixed(0)} MHz`);
    $('foot-src').textContent = bits.join(' · ');
  }
}

/* ------------------------------------------------------------------ */
/* Alerts: full-width pulsing banner above the cards, plus a flashing   */
/* <title> while any alert is active. Alert "since" is on the stream    */
/* clock, so ages are computed against the stream time, extrapolated    */
/* by wall time between messages.                                       */
/* ------------------------------------------------------------------ */

const BASE_TITLE = document.title;

function alertLabel(type) {
  if (type === 'fall') return 'FALL DETECTED';
  if (type === 'breathing_stopped') return 'BREATHING STOPPED';
  return String(type || 'ALERT').replace(/_/g, ' ').toUpperCase();
}

function titleLabel(type) {
  return type === 'fall' ? 'FALL' : alertLabel(type);
}

function fmtAge(sec) {
  const v = Math.max(0, Math.floor(sec));
  const mm = String(Math.floor(v / 60)).padStart(2, '0');
  const ss = String(v % 60).padStart(2, '0');
  return `${mm}:${ss}`;
}

/* Best estimate of "now" on the stream clock: last state timestamp plus
 * wall time elapsed since that message arrived (capped so a stalled
 * stream doesn't run the counters away). */
function streamNow() {
  const s = store.state;
  if (!s || !Number.isFinite(s.timestamp)) return null;
  const dt = store._lastMsg != null ? (performance.now() - store._lastMsg) / 1000 : 0;
  return s.timestamp + clamp(dt, 0, 60);
}

function renderAlerts(s) {
  const alerts = Array.isArray(s && s.alerts) ? s.alerts : [];
  store.alerts = alerts;
  const banner = $('alert-banner');
  document.body.classList.toggle('alerting', alerts.length > 0);
  if (!alerts.length) {
    banner.hidden = true;
    banner.textContent = '';
    return;
  }
  // The list is tiny, so rebuild rather than diff.
  banner.hidden = false;
  banner.textContent = '';
  for (const a of alerts) {
    const el = document.createElement('div');
    el.className = 'alert';
    const title = document.createElement('span');
    title.className = 'alert-title';
    title.textContent = `⚠ ${alertLabel(a && a.type)}`;
    const msg = document.createElement('span');
    msg.className = 'alert-msg';
    msg.textContent = (a && a.message) || '';
    const age = document.createElement('span');
    age.className = 'alert-age';
    if (a && Number.isFinite(a.since)) age.dataset.since = String(a.since);
    el.append(title, msg, age);
    banner.appendChild(el);
  }
  refreshAlertAges();
}

function refreshAlertAges() {
  const now = streamNow();
  for (const el of document.querySelectorAll('.alert-age')) {
    const since = Number(el.dataset.since);
    el.textContent = now != null && Number.isFinite(since) ? fmtAge(now - since) : '';
  }
}

/* 1 Hz housekeeping: alert age counters, title flash, and a repaint so
 * the timeline keeps scrolling between (or without) state messages. */
let titleFlip = false;
setInterval(() => {
  if (store.alerts.length) {
    refreshAlertAges();
    titleFlip = !titleFlip;
    document.title = titleFlip
      ? `⚠ ${titleLabel(store.alerts[0] && store.alerts[0].type)} — wifi-room-radar`
      : BASE_TITLE;
  } else if (document.title !== BASE_TITLE) {
    document.title = BASE_TITLE;
    titleFlip = false;
  }
  dirty = true;
}, 1000);

/* ------------------------------------------------------------------ */
/* Render loop (repaints only when something changed).                  */
/* ------------------------------------------------------------------ */

function frame() {
  if (dirty) {
    dirty = false;
    const s = store.state;
    spectro.draw();
    roomMap.draw(s, store.info, store.trails);
    const br = s && s.breathing;
    breathChart.draw(
      br ? br.waveform : null,
      br && br.rate_bpm > 0 ? `${br.rate_bpm.toFixed(1)} bpm` : null,
    );
    const hb = s && s.heartbeat;
    heartChart.draw(
      store.heartHist.length > 1 ? store.heartHist : null,
      hb && hb.rate_bpm > 0 ? `${hb.rate_bpm.toFixed(1)} bpm` : null,
    );
    timeline.draw();
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

/* ------------------------------------------------------------------ */
/* Websocket client with exponential-backoff auto-reconnect.            */
/* ------------------------------------------------------------------ */

let backoff = 500; // ms; grows to 8 s, resets on successful connect

function setConn(ok, text) {
  $('conn-dot').classList.toggle('ok', ok);
  $('conn-text').textContent = text;
}

function connect() {
  setConn(false, 'connecting');
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  let ws;
  try {
    ws = new WebSocket(`${proto}://${location.host}/ws`);
  } catch (e) {
    setTimeout(connect, backoff);
    backoff = Math.min(backoff * 1.7, 8000);
    return;
  }
  ws.onopen = () => {
    backoff = 500;
    setConn(true, 'live');
  };
  ws.onmessage = (ev) => {
    let msg;
    try {
      msg = JSON.parse(ev.data);
    } catch (e) {
      return;
    }
    if (msg.type === 'info') onInfo(msg.data);
    else if (msg.type === 'state') onState(msg.data);
  };
  ws.onclose = () => {
    setConn(false, 'reconnecting');
    setTimeout(connect, backoff);
    backoff = Math.min(backoff * 1.7, 8000);
  };
  ws.onerror = () => {
    try { ws.close(); } catch (e) { /* already closed */ }
  };
}
connect();
