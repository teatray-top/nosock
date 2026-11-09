const NSChart = (() => {
  const cssv = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
  const touchFirst = matchMedia("(pointer: coarse)").matches;
  const FONT = '"Pretendard Variable", Pretendard, system-ui, sans-serif';

  function rgba(c, a) {
    const m = /^#([0-9a-f]{6})$/i.exec(c);
    if (!m) return c;
    const n = parseInt(m[1], 16);
    return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
  }

  function compact(v) {
    if (v == null || !isFinite(v)) return "-";
    const a = Math.abs(v), s = v < 0 ? "−" : "";
    if (a >= 1e8) return s + (a / 1e8).toLocaleString("ko-KR", { maximumFractionDigits: a >= 1e10 ? 0 : 2 }) + "억";
    if (a >= 1e4) return s + (a / 1e4).toLocaleString("ko-KR", { maximumFractionDigits: a >= 1e6 ? 0 : 1 }) + "만";
    return s + Math.round(a).toLocaleString("ko-KR");
  }

  function niceTicks(lo, hi, n) {
    const span = hi - lo;
    if (!(span > 0)) return [lo];
    const raw = span / n, mag = 10 ** Math.floor(Math.log10(raw)), f = raw / mag;
    const step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return out;
  }

  const RANGES = [["1M", 31, "1개월"], ["3M", 92, "3개월"], ["1Y", 366, "1년"], ["3Y", 1096, "3년"], ["ALL", Infinity, "전체"]];
  const dayNum = (d) => Date.UTC(+d.slice(0, 4), +d.slice(5, 7) - 1, +d.slice(8, 10)) / 864e5;

  function create(host, cfg) {
    if (host._chart) host._chart.destroy();
    host.innerHTML = "";
    const root = document.createElement("div");
    root.className = "viz";
    root.innerHTML = `<div class="viz-read"><span class="viz-day"></span><div class="viz-legend" role="group" aria-label="선 켜고 끄기"></div></div>
      <div class="viz-plot" tabindex="0" role="img"></div>
      <div class="viz-nav" aria-hidden="true"></div>
      <div class="viz-ctl"><div class="viz-ranges" role="group" aria-label="보이는 기간"></div><span class="viz-hint">${touchFirst ? "두 손가락으로 확대" : "휠로 확대, 끌어서 이동"}</span></div>`;
    host.appendChild(root);
    const q = (s) => root.querySelector(s);
    const plotEl = q(".viz-plot"), navEl = q(".viz-nav"), dayEl = q(".viz-day"), legEl = q(".viz-legend"), rangeEl = q(".viz-ranges");
    plotEl.setAttribute("aria-label", cfg.label || "추이 차트");

    let days = [], series = [], n = 0, dn = [], dow = [];
    let view = [0, 0];
    const hidden = new Set();
    let cursor = null;
    let col = {};
    const readCols = () => {
      col = { ink: cssv("--ink"), ink2: cssv("--ink-2"), ink3: cssv("--ink-3"), line: cssv("--line"), bg: cssv("--bg"), surface: cssv("--surface"), up: cssv("--up"), down: cssv("--down") };
      varCache.clear();
    };
    const varCache = new Map();
    const cv = (name) => { if (!varCache.has(name)) varCache.set(name, cssv(name) || col.ink); return varCache.get(name); };
    readCols();
    const colorOf = (s) => cv(s.color);

    const PAD = { l: 0, r: 52, t: 10, b: 24 };
    let W = plotEl.clientWidth || 300, H = plotEl.clientHeight || 240, NW = W, NH = navEl.clientHeight || 34;
    const stage = new Konva.Stage({ container: plotEl, width: W, height: H });
    const dataLayer = new Konva.Layer({ listening: false }), curLayer = new Konva.Layer({ listening: false });
    stage.add(dataLayer, curLayer);
    const nav = new Konva.Stage({ container: navEl, width: NW, height: NH });
    const navBase = new Konva.Layer({ listening: false }), navWin = new Konva.Layer({ listening: false });
    nav.add(navBase, navWin);

    const pw = () => W - PAD.l - PAD.r, ph = () => H - PAD.t - PAD.b;
    const X = (i) => PAD.l + ((i - view[0]) / Math.max(1e-9, view[1] - view[0])) * pw();
    const idxAt = (x, v = view) => v[0] + ((x - PAD.l) / pw()) * (v[1] - v[0]);
    const NX = (i) => (n <= 1 ? 0 : (i / (n - 1)) * NW);
    const minSpan = () => Math.min(Math.max(1, n - 1), Math.max(4, Math.round((n - 1) / 400)));
    let yr = { lo: 0, hi: 1 };

    function visibleRange() {
      return [Math.max(0, Math.floor(view[0])), Math.min(n - 1, Math.ceil(view[1]))];
    }

    function computeY() {
      const [a, b] = visibleRange();
      let lo = Infinity, hi = -Infinity;
      for (const s of series) {
        if (hidden.has(s.key)) continue;
        for (let i = a; i <= b; i++) {
          const v = s.values[i];
          if (v == null || !isFinite(v)) continue;
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
      }
      if (!isFinite(lo)) { lo = 0; hi = 1; }
      if (hi === lo) { hi += Math.abs(hi) * 0.05 || 1; lo -= Math.abs(lo) * 0.05 || 1; }
      const pad = (hi - lo) * 0.05;
      const nonneg = lo >= 0;
      hi += pad;
      lo -= pad;
      if (nonneg && lo < 0) lo = 0;
      yr = { lo, hi };
    }
    const Y = (v) => PAD.t + (1 - (v - yr.lo) / (yr.hi - yr.lo)) * ph();

    function xTicks(a, b) {
      const maxN = Math.max(2, Math.floor(pw() / 66));
      const levels = [
        [(i) => days[i].slice(0, 4) !== days[i - 1].slice(0, 4), (d) => d.slice(0, 4)],
        [(i) => days[i].slice(0, 7) !== days[i - 1].slice(0, 7) && +days[i].slice(5, 7) % 3 === 1, (d) => d.slice(2, 4) + "." + d.slice(5, 7)],
        [(i) => days[i].slice(0, 7) !== days[i - 1].slice(0, 7), (d) => d.slice(2, 4) + "." + d.slice(5, 7)],
        [(i) => dow[i] < dow[i - 1] || dn[i] - dn[i - 1] >= 7, (d) => d.slice(5, 7) + "." + d.slice(8, 10)],
        [(i) => days[i].slice(0, 10) !== days[i - 1].slice(0, 10), (d) => d.slice(5, 7) + "." + d.slice(8, 10)],
        [(i) => days[i].length > 10 && days[i].slice(11, 13) !== days[i - 1].slice(11, 13), (d) => d.slice(11, 13) + "시"],
        [() => true, (d) => (d.length > 10 ? d.slice(11, 16) : d.slice(5, 7) + "." + d.slice(8, 10))],
      ];
      let pick = null;
      for (let L = levels.length - 1; L >= 0; L--) {
        const [test, fmt] = levels[L];
        const idx = [];
        for (let i = Math.max(1, a); i <= b; i++) if (test(i)) idx.push(i);
        if (idx.length <= maxN && idx.length >= 2) { pick = [idx, fmt]; break; }
        if (L === 0) {
          const stride = Math.ceil(idx.length / maxN);
          pick = [idx.filter((_, k) => k % stride === 0), fmt];
        }
      }
      return pick;
    }

    function drawData(ctx) {
      const c = ctx._context;
      if (n < 1) return;
      computeY();
      const [a, b] = visibleRange();
      const top = PAD.t, bot = PAD.t + ph(), left = PAD.l, right = PAD.l + pw();
      c.font = `500 11px ${FONT}`;
      c.textBaseline = "middle";
      c.textAlign = "right";
      const ticks = niceTicks(yr.lo, yr.hi, ph() > 200 ? 5 : 4);
      for (const t of ticks) {
        const y = Math.round(Y(t)) + 0.5;
        if (y < top - 1 || y > bot + 1) continue;
        c.strokeStyle = col.line;
        c.lineWidth = 1;
        c.setLineDash([]);
        c.beginPath();
        c.moveTo(left, y);
        c.lineTo(right, y);
        c.stroke();
        c.fillStyle = col.ink3;
        c.fillText(compact(t), W - 2, Math.min(Math.max(y, top + 6), bot - 6));
      }
      const xt = xTicks(a, b);
      c.textAlign = "center";
      c.textBaseline = "alphabetic";
      c.fillStyle = col.ink3;
      if (xt) {
        let lastR = -Infinity;
        for (const i of xt[0]) {
          const x = X(i), label = xt[1](days[i]), w = c.measureText(label).width;
          const cx = Math.min(Math.max(x, left + w / 2), right - w / 2);
          if (cx - w / 2 < lastR + 8) continue;
          c.strokeStyle = col.line;
          c.beginPath();
          c.moveTo(Math.round(x) + 0.5, bot);
          c.lineTo(Math.round(x) + 0.5, bot + 4);
          c.stroke();
          c.fillText(label, cx, H - 4);
          lastR = cx + w / 2;
        }
      }
      c.save();
      c.beginPath();
      c.rect(left, top - 3, pw(), ph() + 6);
      c.clip();
      const S = (k) => series.find((s) => s.key === k);
      const vals = (s, i) => { let v = s.values[i]; let j = i; while ((v == null || !isFinite(v)) && j > 0) v = s.values[--j]; return v == null ? yr.lo : v; };
      if (cfg.band && !hidden.has(cfg.band.top) && !hidden.has(cfg.band.base) && S(cfg.band.top) && S(cfg.band.base)) {
        const T = S(cfg.band.top), D = S(cfg.band.base);
        const bandPath = () => {
          c.beginPath();
          for (let i = a; i <= b; i++) c.lineTo(X(i), Y(vals(T, i)));
          for (let i = b; i >= a; i--) c.lineTo(X(i), Y(vals(D, i)));
          c.closePath();
        };
        for (const [edge, color] of [[top - 3, col.up], [bot + 3, col.down]]) {
          c.save();
          c.beginPath();
          c.moveTo(X(a), edge);
          for (let i = a; i <= b; i++) c.lineTo(X(i), Y(vals(D, i)));
          c.lineTo(X(b), edge);
          c.closePath();
          c.clip();
          bandPath();
          c.fillStyle = rgba(color, 0.13);
          c.fill();
          c.restore();
        }
      }
      for (const s of series) {
        if (hidden.has(s.key)) continue;
        const color = colorOf(s);
        c.setLineDash(s.style === "dash" ? [4, 4] : []);
        c.lineJoin = "round";
        c.lineCap = "round";
        if (s.style === "step") {
          c.beginPath();
          c.moveTo(X(a), bot + 3);
          let prev = vals(s, a);
          for (let i = a; i <= b; i++) {
            const v = vals(s, i);
            c.lineTo(X(i), Y(prev));
            c.lineTo(X(i), Y(v));
            prev = v;
          }
          c.lineTo(X(b), bot + 3);
          c.closePath();
          c.fillStyle = rgba(color, s.fillAlpha ?? 0.2);
          c.fill();
          c.beginPath();
          prev = vals(s, a);
          c.moveTo(X(a), Y(prev));
          for (let i = a; i <= b; i++) {
            const v = vals(s, i);
            c.lineTo(X(i), Y(prev));
            c.lineTo(X(i), Y(v));
            prev = v;
          }
          c.strokeStyle = s.stroke ? cv(s.stroke) : color;
          c.lineWidth = 1.5;
          c.stroke();
          continue;
        }
        c.beginPath();
        let started = false;
        for (let i = a; i <= b; i++) {
          const v = s.values[i];
          if (v == null || !isFinite(v)) continue;
          if (!started) { c.moveTo(X(i), Y(v)); started = true; } else c.lineTo(X(i), Y(v));
        }
        c.strokeStyle = color;
        c.lineWidth = s.width || 2;
        c.stroke();
      }
      c.restore();
      c.setLineDash([]);
    }

    function drawCursor(ctx) {
      if (cursor == null || n < 1) return;
      const c = ctx._context;
      const i = Math.round(cursor);
      const x = Math.round(X(i)) + 0.5;
      c.strokeStyle = col.ink2;
      c.lineWidth = 1;
      c.setLineDash([2, 3]);
      c.beginPath();
      c.moveTo(x, PAD.t);
      c.lineTo(x, PAD.t + ph());
      c.stroke();
      c.setLineDash([]);
      for (const s of series) {
        if (hidden.has(s.key)) continue;
        const v = s.values[i];
        if (v == null || !isFinite(v)) continue;
        c.beginPath();
        c.arc(x, Y(v), 4, 0, Math.PI * 2);
        c.fillStyle = col.bg;
        c.fill();
        c.lineWidth = 2;
        c.strokeStyle = s.stroke ? cv(s.stroke) : colorOf(s);
        c.stroke();
      }
    }

    function drawNav(ctx) {
      const c = ctx._context;
      if (n < 2) return;
      const s = series.find((x) => x.key === (cfg.navKey || series[0].key));
      let lo = Infinity, hi = -Infinity;
      for (const v of s.values) if (v != null && isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      if (!(hi > lo)) { hi = lo + 1; }
      const NY = (v) => 3 + (1 - (v - lo) / (hi - lo)) * (NH - 6);
      c.beginPath();
      c.moveTo(0, NH);
      s.values.forEach((v, i) => { if (v != null && isFinite(v)) c.lineTo(NX(i), NY(v)); });
      c.lineTo(NW, NH);
      c.closePath();
      c.fillStyle = rgba(col.ink3, 0.18);
      c.fill();
      c.beginPath();
      s.values.forEach((v, i) => { if (v != null && isFinite(v)) (i ? c.lineTo : c.moveTo).call(c, NX(i), NY(v)); });
      c.strokeStyle = col.ink3;
      c.lineWidth = 1;
      c.stroke();
    }

    function drawNavWin(ctx) {
      const c = ctx._context;
      if (n < 2) return;
      const x0 = NX(view[0]), x1 = NX(view[1]);
      c.fillStyle = rgba(col.bg, 0.7);
      c.fillRect(0, 0, x0, NH);
      c.fillRect(x1, 0, NW - x1, NH);
      c.strokeStyle = col.ink2;
      c.lineWidth = 1.25;
      c.strokeRect(x0 + 0.75, 0.75, Math.max(2, x1 - x0 - 1.5), NH - 1.5);
      c.fillStyle = col.ink;
      for (const x of [x0, x1]) {
        c.beginPath();
        c.roundRect ? c.roundRect(x - 3, NH / 2 - 9, 6, 18, 3) : c.rect(x - 3, NH / 2 - 9, 6, 18);
        c.fill();
      }
    }

    const dataShape = new Konva.Shape({ sceneFunc: drawData, listening: false });
    const curShape = new Konva.Shape({ sceneFunc: drawCursor, listening: false });
    const navShape = new Konva.Shape({ sceneFunc: drawNav, listening: false });
    const navWinShape = new Konva.Shape({ sceneFunc: drawNavWin, listening: false });
    dataLayer.add(dataShape);
    curLayer.add(curShape);
    navBase.add(navShape);
    navWin.add(navWinShape);

    let raf = 0, dirtyData = false, dirtyNav = false;
    function schedule(data, navb) {
      dirtyData = dirtyData || data;
      dirtyNav = dirtyNav || navb;
      if (raf) return;
      raf = requestAnimationFrame(() => {
        raf = 0;
        if (dirtyData) dataLayer.batchDraw();
        if (dirtyNav) navBase.batchDraw();
        dirtyData = dirtyNav = false;
        curLayer.batchDraw();
        navWin.batchDraw();
        readout();
        markRange();
      });
    }

    const fmt = cfg.fmt || compact;
    function readout() {
      if (!n) { dayEl.textContent = ""; return; }
      const i = cursor == null ? visibleRange()[1] : Math.round(cursor);
      const d = days[i];
      dayEl.textContent = `${d.slice(0, 4)}.${d.slice(5, 7)}.${d.slice(8, 10)}${d.length > 10 ? " " + d.slice(11, 16) : ""}`;
      for (const s of series) {
        const b = legEl.querySelector(`[data-k="${s.key}"] b`);
        if (b) b.textContent = fmt(s.values[i]);
      }
    }

    function buildLegend() {
      legEl.innerHTML = "";
      for (const s of series) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "lg";
        btn.dataset.k = s.key;
        btn.setAttribute("aria-pressed", String(!hidden.has(s.key)));
        btn.innerHTML = `<i class="sw sw-${s.style || "line"}" style="--c:var(${s.color})${s.stroke ? `;--c2:var(${s.stroke})` : ""}"></i><span>${s.name}</span><b></b>`;
        btn.onclick = () => {
          if (hidden.has(s.key)) hidden.delete(s.key); else hidden.add(s.key);
          btn.setAttribute("aria-pressed", String(!hidden.has(s.key)));
          schedule(true, false);
        };
        legEl.appendChild(btn);
      }
      if (cfg.band) {
        const sp = document.createElement("span");
        sp.className = "lg lg-band";
        sp.innerHTML = `<i class="sw sw-band"></i><span>평가 손익</span>`;
        legEl.appendChild(sp);
      }
    }

    function buildRanges() {
      rangeEl.innerHTML = "";
      if (n < 2) return;
      const span = dn[n - 1] - dn[0];
      for (const [k, d, label] of RANGES) {
        if (d !== Infinity && d >= span * 0.9) continue;
        const btn = document.createElement("button");
        btn.type = "button";
        btn.dataset.r = k;
        btn.textContent = label;
        btn.onclick = () => animateTo(rangeView(d));
        rangeEl.appendChild(btn);
      }
    }
    function rangeView(d) {
      if (d === Infinity) return [0, n - 1];
      const from = dn[n - 1] - d;
      let i = n - 1;
      while (i > 0 && dn[i - 1] >= from) i--;
      return [Math.min(i, n - 1 - minSpan()), n - 1];
    }
    function markRange() {
      for (const btn of rangeEl.querySelectorAll("button")) {
        const r = RANGES.find((x) => x[0] === btn.dataset.r);
        const v = rangeView(r[1]);
        btn.setAttribute("aria-pressed", String(Math.abs(v[0] - view[0]) < 0.5 && Math.abs(v[1] - view[1]) < 0.5));
      }
    }

    function setView(a, b, keepCursor) {
      if (n < 2) { view = [0, Math.max(0, n - 1)]; schedule(true, false); return; }
      let span = Math.min(n - 1, Math.max(minSpan(), b - a));
      a = Math.max(0, Math.min(a, n - 1 - span));
      view = [a, a + span];
      if (!keepCursor) cursor = cursor == null ? null : Math.max(view[0], Math.min(view[1], cursor));
      schedule(true, false);
    }

    let anim = 0;
    function animateTo(target) {
      cancelAnimationFrame(anim);
      if (reduced()) { setView(target[0], target[1]); return; }
      const from = view.slice(), t0 = performance.now(), dur = 280;
      const step = (now) => {
        const t = Math.min(1, (now - t0) / dur), e = 1 - (1 - t) ** 3;
        setView(from[0] + (target[0] - from[0]) * e, from[1] + (target[1] - from[1]) * e);
        if (t < 1) anim = requestAnimationFrame(step);
      };
      anim = requestAnimationFrame(step);
    }

    function zoomAt(x, factor) {
      const span = view[1] - view[0];
      const ns = Math.min(n - 1, Math.max(minSpan(), span * factor));
      const anchor = idxAt(x);
      const frac = (x - PAD.l) / pw();
      setView(anchor - frac * ns, anchor - frac * ns + ns);
    }

    const lx = (e) => e.clientX - plotEl.getBoundingClientRect().left;
    const pts = new Map();
    let gesture = null, lastTap = 0;
    plotEl.addEventListener("pointerdown", (e) => {
      if (n < 2) return;
      plotEl.setPointerCapture(e.pointerId);
      pts.set(e.pointerId, { x: lx(e), y: e.clientY });
      cancelAnimationFrame(anim);
      if (e.pointerType === "mouse") {
        if (e.button === 0) gesture = { type: "pan", x0: lx(e), v0: view.slice() };
        return;
      }
      if (pts.size === 2) {
        const [p, r] = [...pts.values()];
        const mid = (p.x + r.x) / 2;
        gesture = { type: "pinch", d0: Math.max(24, Math.hypot(p.x - r.x, p.y - r.y)), span0: view[1] - view[0], anchor: idxAt(mid), v0: view.slice() };
        cursor = null;
        schedule(false, false);
      } else if (pts.size === 1) {
        const now = performance.now();
        if (now - lastTap < 300) { animateTo([0, n - 1]); lastTap = 0; } else lastTap = now;
        gesture = { type: "scrub" };
        cursor = Math.max(view[0], Math.min(view[1], idxAt(lx(e))));
        schedule(false, false);
      }
    });
    plotEl.addEventListener("pointermove", (e) => {
      if (n < 2) return;
      if (pts.has(e.pointerId)) pts.set(e.pointerId, { x: lx(e), y: e.clientY });
      if (e.pointerType === "mouse") {
        if (gesture && gesture.type === "pan" && e.buttons & 1) {
          const dx = lx(e) - gesture.x0;
          const span = gesture.v0[1] - gesture.v0[0];
          const sh = (-dx / pw()) * span;
          setView(gesture.v0[0] + sh, gesture.v0[1] + sh, true);
          plotEl.style.cursor = "grabbing";
        }
        cursor = Math.max(view[0], Math.min(view[1], idxAt(lx(e))));
        schedule(false, false);
        return;
      }
      if (gesture && gesture.type === "pinch" && pts.size >= 2) {
        const [p, r] = [...pts.values()];
        const d = Math.max(24, Math.hypot(p.x - r.x, p.y - r.y));
        const span = Math.min(n - 1, Math.max(minSpan(), gesture.span0 * (gesture.d0 / d)));
        const mid = (p.x + r.x) / 2;
        const a = gesture.anchor - ((mid - PAD.l) / pw()) * span;
        setView(a, a + span, true);
      } else if (gesture && gesture.type === "scrub") {
        cursor = Math.max(view[0], Math.min(view[1], idxAt(lx(e))));
        schedule(false, false);
      }
    });
    const end = (e) => {
      pts.delete(e.pointerId);
      if (e.pointerType === "mouse") { gesture = null; plotEl.style.cursor = ""; return; }
      if (gesture && gesture.type === "pinch") gesture = pts.size === 1 ? { type: "after" } : null;
      if (!pts.size) { gesture = null; cursor = null; schedule(false, false); }
    };
    plotEl.addEventListener("pointerup", end);
    plotEl.addEventListener("pointercancel", end);
    plotEl.addEventListener("pointerleave", (e) => { if (e.pointerType === "mouse" && !(e.buttons & 1)) { cursor = null; schedule(false, false); } });
    plotEl.addEventListener("wheel", (e) => {
      if (n < 2) return;
      e.preventDefault();
      cancelAnimationFrame(anim);
      const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
      if (Math.abs(e.deltaX) > Math.abs(dy)) {
        const sh = (e.deltaX / pw()) * (view[1] - view[0]);
        setView(view[0] + sh, view[1] + sh, true);
      } else zoomAt(lx(e), Math.exp(dy * 0.0016));
    }, { passive: false });
    plotEl.addEventListener("dblclick", () => animateTo([0, n - 1]));
    plotEl.addEventListener("keydown", (e) => {
      const span = view[1] - view[0];
      const k = { ArrowLeft: () => setView(view[0] - span * 0.15, view[1] - span * 0.15), ArrowRight: () => setView(view[0] + span * 0.15, view[1] + span * 0.15), "+": () => zoomAt(PAD.l + pw() / 2, 0.75), "=": () => zoomAt(PAD.l + pw() / 2, 0.75), "-": () => zoomAt(PAD.l + pw() / 2, 1.33), "0": () => animateTo([0, n - 1]) }[e.key];
      if (k) { e.preventDefault(); k(); }
    });

    let navDrag = null;
    const nx = (e) => e.clientX - navEl.getBoundingClientRect().left;
    navEl.addEventListener("pointerdown", (e) => {
      if (n < 2) return;
      navEl.setPointerCapture(e.pointerId);
      cancelAnimationFrame(anim);
      const x = nx(e), x0 = NX(view[0]), x1 = NX(view[1]);
      const grab = touchFirst ? 18 : 9;
      const span = view[1] - view[0];
      if (Math.abs(x - x0) <= grab && Math.abs(x - x0) <= Math.abs(x - x1)) navDrag = { mode: "l" };
      else if (Math.abs(x - x1) <= grab) navDrag = { mode: "r" };
      else {
        if (x < x0 || x > x1) { const c = (x / NW) * (n - 1); setView(c - span / 2, c + span / 2); }
        navDrag = { mode: "move", x0: x, v0: view.slice() };
      }
    });
    navEl.addEventListener("pointermove", (e) => {
      if (!navDrag) return;
      const x = nx(e), i = (Math.max(0, Math.min(NW, x)) / NW) * (n - 1);
      if (navDrag.mode === "move") {
        const sh = ((x - navDrag.x0) / NW) * (n - 1);
        setView(navDrag.v0[0] + sh, navDrag.v0[1] + sh);
      } else if (navDrag.mode === "l") setView(Math.min(i, view[1] - minSpan()), view[1]);
      else setView(view[0], Math.max(i, view[0] + minSpan()));
    });
    const navEnd = () => (navDrag = null);
    navEl.addEventListener("pointerup", navEnd);
    navEl.addEventListener("pointercancel", navEnd);

    function resize() {
      const w = plotEl.clientWidth, h = plotEl.clientHeight, w2 = navEl.clientWidth, h2 = navEl.clientHeight;
      if (!w || (w === W && h === H && w2 === NW && h2 === NH)) return;
      W = w; H = h; NW = w2; NH = h2;
      stage.size({ width: W, height: H });
      nav.size({ width: NW, height: NH });
      schedule(true, true);
    }
    const ro = new ResizeObserver(() => resize());
    ro.observe(root);
    const mq = matchMedia("(prefers-color-scheme: dark)");
    const onTheme = () => { readCols(); schedule(true, true); };
    mq.addEventListener("change", onTheme);

    function update(next) {
      const wasFull = n < 2 || (view[0] <= 0.5 && view[1] >= n - 1.5);
      const atEnd = n >= 2 && view[1] >= n - 1.5;
      const span = view[1] - view[0];
      days = next.days;
      series = next.series;
      n = days.length;
      dn = days.map(dayNum);
      dow = days.map((d) => new Date(d + "T00:00:00Z").getUTCDay());
      buildLegend();
      buildRanges();
      root.classList.toggle("viz-empty", n < 2);
      if (wasFull || n < 2) view = [0, Math.max(0, n - 1)];
      else if (atEnd) setView(n - 1 - span, n - 1);
      else setView(view[0], view[1]);
      if (cfg.initial && wasFull && n >= 2) view = rangeView(cfg.initial);
      cursor = null;
      schedule(true, true);
    }

    const api = {
      update,
      destroy() { ro.disconnect(); mq.removeEventListener("change", onTheme); cancelAnimationFrame(raf); cancelAnimationFrame(anim); stage.destroy(); nav.destroy(); host._chart = null; },
      get view() { return view.slice(); },
      zoom(f) { zoomAt(PAD.l + pw() / 2, f); },
    };
    host._chart = api;
    update({ days: cfg.days, series: cfg.series });
    return api;
  }

  return { create, compact };
})();
