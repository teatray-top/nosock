const B = "/nosock";
const KR = { "069500": "KODEX 200", "229200": "코스닥150", "091160": "반도체 (국내)", "091170": "은행", "114800": "인버스", "148070": "국고채 10년", "114260": "국고채 3년", "153130": "단기채권", "0167A0": "AI반도체 TOP2" };
const US = { SPY: "S&P500", QQQ: "나스닥100", IWM: "러셀2000", DIA: "다우30", XLK: "기술", SMH: "반도체", XLF: "금융", XLE: "에너지", XLV: "헬스케어", XLU: "유틸리티", XLP: "필수소비재", EFA: "선진국", EEM: "신흥국", TLT: "미국채 20년+", IEF: "미국채 7-10년", SHY: "미국채 1-3년", GLD: "금", SLV: "은", USO: "원유", SH: "S&P500 인버스", UUP: "달러" };
const BOOKS = {
  U29_topk: { name: "상위 K 모멘텀", desc: "20거래일마다 29개 종목을 모멘텀 점수로 줄 세워 위에서 K개를 같은 비중으로 삽니다." },
  U29_dual: { name: "듀얼 모멘텀", desc: "상위 K와 같게 고르되, 12-1개월 수익이 0 이하인 종목은 사지 않고 그 몫을 현금으로 둡니다." },
};
const nm = (c) => KR[c] || US[c] || c;
const won = (x) => (x == null || isNaN(x) ? "-" : (Math.round(x) || 0).toLocaleString("ko-KR"));
const signed = (x) => (x == null || isNaN(x) ? "-" : (x >= 0 ? "+" : "−") + won(Math.abs(x)));
const cmp = (x) => NSChart.compact(x);
const pct = (x) => (x == null || isNaN(x) ? "-" : (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(1) + "%");
const cls = (x) => (x == null || isNaN(x) || Math.abs(x) < 1e-9 ? "" : x > 0 ? "up" : "down");
const dot = (d) => (d ? `${d.slice(0, 4)}.${d.slice(5, 7)}.${d.slice(8, 10)}` : "");
const WD = ["일", "월", "화", "수", "목", "금", "토"];
const toD = (s) => new Date(s + "T00:00:00Z");
const iso = (d) => d.toISOString().slice(0, 10);
const wd = (s) => WD[toD(s).getUTCDay()];
const addDays = (s, n) => { const d = toD(s); d.setUTCDate(d.getUTCDate() + n); return iso(d); };
const addYears = (s, n) => { const d = toD(s); d.setUTCFullYear(d.getUTCFullYear() + n); return iso(d); };
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const store = { get: (k, d) => { try { return localStorage.getItem(k) ?? d; } catch (_) { return d; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch (_) {} } };
let S = null;
let book = store.get("book", "U29_topk");
if (!BOOKS[book]) book = "U29_topk";
let counted = false;

async function api(path, opt) {
  const r = await fetch(B + path, Object.assign({ credentials: "same-origin", headers: { "Content-Type": "application/json" } }, opt || {}));
  if (r.status === 401) { showLogin(); throw new Error("로그인이 필요합니다"); }
  const t = await r.text();
  try { return JSON.parse(t); } catch (_) { throw new Error(`서버 응답 ${r.status}${r.headers.get("x-honeypot") ? ", 허니팟에서 막힘" : ""}`); }
}
function showLogin() { $("login").hidden = false; $("app").hidden = true; }

function liveSeries(rows, pick) {
  return {
    days: rows.map((r) => r.day),
    series: [
      { key: "total", name: "합계", color: "--ink", values: rows.map((r) => pick(r).total), width: 2.25 },
      { key: "deposits", name: "넣은 돈", color: "--ink-3", style: "dash", width: 1.5, values: rows.map((r) => pick(r).deposits) },
      { key: "vault", name: "금고", color: "--vault", stroke: "--vault-ink", fillAlpha: 0.5, style: "step", values: rows.map((r) => pick(r).vault) },
    ],
  };
}

function bookRows(b = book) { return ((S && S.mom_history) || []).filter((r) => r[b]); }
function current(b = book) { return S && S.mom_last && S.mom_last.books ? S.mom_last.books[b] : null; }
function lastK(b) { const d = bookRows(b).filter((r) => r[b].rebalance).pop(); return d ? d[b].rebalance.k : null; }

function countUp(elm, to) {
  const paint = (v) => (elm.innerHTML = `${won(v)}<small>원</small>`);
  if (counted || matchMedia("(prefers-reduced-motion: reduce)").matches) { paint(to); return; }
  counted = true;
  const t0 = performance.now(), dur = 700;
  const step = (now) => {
    const t = Math.min(1, (now - t0) / dur);
    paint(to * (1 - (1 - t) ** 4));
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function payoutsOf(rows, b = book) {
  const out = [];
  let prev = 0;
  for (const r of rows) {
    const v = r[b].vault || 0;
    if (v > prev + 0.5) out.push({ day: r.day, amount: v - prev, total: v });
    prev = v;
  }
  return out;
}

let liveChart = null;
let liveChartMode = null;
let liveMode = "won";
try { liveMode = localStorage.getItem("ns-live-mode") === "ret" ? "ret" : "won"; } catch (e) {}
function twr(marks) {
  let acc = 1;
  return marks.map((x, i) => {
    if (i > 0) {
      const p = marks[i - 1];
      const flow = x.contributed != null && p.contributed != null ? x.contributed - p.contributed : 0;
      const base = p.equity + (p.vault || 0) + flow;
      if (base > 0) acc *= (x.equity + (x.vault || 0)) / base;
    }
    return 100 * (acc - 1);
  });
}
const retFmt = (v) => (v == null || !isFinite(v) ? "-" : Math.abs(v) < 0.005 ? "0.00%" : (v > 0 ? "+" : "−") + Math.abs(v).toFixed(2) + "%");
const retAxis = (t, step) => {
  const d = Math.max(0, Math.min(3, Math.ceil(-Math.log10(step) - 1e-9)));
  return Math.abs(t) < step / 2 ? "0%" : (t > 0 ? "+" : "−") + Math.abs(t).toFixed(d) + "%";
};
function setLiveMode(m) {
  liveMode = m === "ret" ? "ret" : "won";
  try { localStorage.setItem("ns-live-mode", liveMode); } catch (e) {}
  renderNow();
}
const kstStamp = (ts) => (ts ? ts.slice(0, 16).replace("T", " ") : "");
const LV = () => (S && S.live) || {};
function todayKST() { return new Date(Date.now() + 9 * 3600e3).toISOString().slice(0, 10); }
const net0 = (x) => x.equity + (x.vault || 0) - (x.contributed || 0);
function renderNow() {
  const lv = LV();
  const lm = lv.last_mark, plan = lv.plan;
  const rl = $("runline");
  const recentErr = (lv.errors || []).filter((e) => Date.now() - Date.parse(e.ts) < 3600e3);
  if (!lv.last_start) rl.innerHTML = `<i></i><span>자동 매매가 아직 시작되지 않았습니다.</span>`;
  else if (lv.last_start.kind === "stop") rl.innerHTML = `<i></i><span><b>멈춤</b> ${kstStamp(lv.last_start.ts)}</span>`;
  else rl.innerHTML = `<i class="${recentErr.length ? "err" : "on"}"></i><span><b>${lv.live ? "자동 실주문 켜짐" : "드라이런 (주문 안 냄)"}</b>${lm ? `, 마지막 확인 ${kstStamp(lm.ts).slice(5)}` : ""}${freshening ? ", 잔고 확인 중" : ""}${recentErr.length ? `, 최근 오류 ${recentErr.length}건` : ""}</span>`;
  $("mode").textContent = lv.live ? "실계좌" : "드라이런";
  $("mode").classList.toggle("live", !!lv.live);
  $("asof").textContent = lm ? `${kstStamp(lm.ts).slice(5).replace("-", ".")} 기준` : "기록 대기";
  if (lm) countUp($("vault"), lm.equity); else $("vault").innerHTML = `-`;
  const marks = lv.marks || [];
  const today = todayKST();
  const firstToday = marks.find((x) => x.ts.slice(0, 10) === today);
  const pay = $("pay");
  const cw = (v) => cls(Math.round(v) || 0), cp = (v) => cls(Math.round(v * 100) / 100 || 0);
  const pv = (w, p) => `<span class="${cw(w)}">${signed(w)}원</span><small class="${cp(p)}">${retFmt(p)}</small>`;
  $("pnl-all").innerHTML = "-";
  $("pnl-today").innerHTML = "-";
  $("pnl-all-from").textContent = "";
  $("pnl-today-from").textContent = "";
  $("pnl-split").innerHTML = "";
  if (lm && marks.length) {
    const r = twr(marks), last = r[r.length - 1], n0 = net0(lm);
    $("pnl-all").innerHTML = pv(n0, last);
    const sp = lv.pnl_split;
    if (sp && sp.ts === lm.ts) {
      const h = Math.round(sp.hold), s = Math.round(sp.sold || 0), f = Math.round(n0) - h - s;
      $("pnl-split").innerHTML = [["보유 종목", h], ...(s ? [["판 종목", s]] : []), ["환율", f]].map(([k, v]) => `<span>${k} <b class="${cls(v)}">${signed(v)}원</b></span>`).join("");
    }
    $("pnl-all-from").textContent = `${dot(marks[0].ts.slice(0, 10)).slice(5)}부터`;
    if (firstToday && lm.ts !== firstToday.ts) {
      const k = marks.indexOf(firstToday);
      $("pnl-today").innerHTML = pv(n0 - net0(firstToday), 100 * ((1 + last / 100) / (1 + r[k] / 100) - 1));
      $("pnl-today-from").textContent = `${firstToday.ts.slice(11, 16)}부터`;
    }
    pay.innerHTML = "";
  } else pay.innerHTML = `<p class="pay">자동 매매를 켜면 도비 장부를 평가해 여기에 보여 줍니다.</p>`;
  $("acct").textContent = won(lm && lm.vault != null ? lm.vault : lv.vault ? lv.vault.vault : 0) + "원";
  $("dep").textContent = lm ? won(lm.krw) + "원" : "-";
  $("profit").textContent = lm && lm.usd != null ? "$" + Number(lm.usd).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "-";

  const fxb = $("fxbox");
  fxb.innerHTML = "";
  for (const f of ((S && S.fx) || []).filter((f) => !f.done)) {
    const box = document.createElement("div");
    box.className = "fx";
    const toUsd = f.direction === "KRW->USD";
    box.innerHTML = `<div class="fx-top"><span>환전 요청${f.live ? "" : " (드라이런)"}</span><span>${esc(f.day)}</span></div>
      <div class="fx-amt">${won(f.amount_won)}원<span>${toUsd ? "원화 → 달러" : "달러 → 원화"}</span></div>
      <p>${esc(f.body || `약 $${(f.amount_usd || 0).toFixed(2)}`)}</p>`;
    const b = document.createElement("button");
    b.className = "btn";
    b.type = "button";
    b.textContent = "환전했어요";
    b.onclick = () => fxDone(b, f.id);
    box.appendChild(b);
    fxb.appendChild(box);
  }
  const fc = S && S.fx_check;
  if (fc && fc.ts && fc.ts.slice(0, 10) === todayKST()) {
    const box = document.createElement("div");
    box.className = "fx";
    box.innerHTML = `<div class="fx-top"><span>환전 확인</span><span>${esc(fc.ts.slice(11, 16))}</span></div><p>${esc(fc.body || "")}</p>`;
    const last = ((S && S.fx) || []).filter((f) => f.day === fc.ts.slice(0, 10)).slice(-1)[0];
    if (last && fc.left >= 0.01) {
      const b = document.createElement("button");
      b.className = "btn";
      b.type = "button";
      b.textContent = "더 바꿨어요";
      b.onclick = () => fxDone(b, last.id);
      box.appendChild(b);
    }
    fxb.appendChild(box);
  }

  const wonData = { days: marks.map((x) => x.ts.slice(0, 16)), series: [{ key: "equity", name: "도비 자산", color: "--ink", width: 2.25, values: marks.map((x) => x.equity) }, { key: "contributed", name: "넣은 돈", color: "--ink-3", width: 1.25, style: "step", values: marks.map((x) => (x.contributed == null ? x.equity : x.contributed)) }] };
  const mode = liveMode;
  const data = mode === "ret" ? { days: wonData.days, series: [{ key: "ret", name: "수익률 (입금 제외)", color: "--ink", width: 2.25, values: twr(marks) }] } : wonData;
  const cfg = mode === "ret" ? { label: "도비 수익률 추이, 입금 제외", fmt: retFmt, yfmt: retAxis } : { label: "도비 자산 추이", fmt: (v) => (v == null || !isFinite(v) ? "-" : won(v) + "원") };
  for (const b of document.querySelectorAll("#live-mode button")) b.setAttribute("aria-pressed", String(b.dataset.m === mode));
  if (!liveChart || !$("live-chart")._chart || liveChartMode !== mode) {
    liveChart = NSChart.create($("live-chart"), Object.assign(cfg, data));
    liveChartMode = mode;
  } else liveChart.update(data);

  const hb = $("holds");
  const tgt = lv.target;
  const w = Object.assign({}, (tgt && tgt.weights) || (plan && plan.weights) || {});
  const subOf = {};
  for (const [o, s] of Object.entries((tgt && tgt.subs) || {})) if (s && w[o]) { w[s] = (w[s] || 0) + w[o]; delete w[o]; subOf[s] = o; }
  const val = (lm && lm.value) || {};
  const eq = lm ? lm.equity : 0;
  $("plan-asof").textContent = tgt ? `${kstStamp(tgt.ts).slice(5).replace("-", ".")} ${tgt.phase === "us" ? "미국장" : "국내장"} 주문 기준` : plan ? `${dot(plan.decided_on)} 종가로 계산` : "";
  const codes = [...new Set([...Object.keys(w), ...Object.keys(val)])].sort((a, b) => (w[b] || 0) - (w[a] || 0) || (val[b] || 0) - (val[a] || 0));
  const top = Math.max(0.0001, ...Object.values(w), ...codes.map((c) => (eq ? (val[c] || 0) / eq : 0)));
  if (!codes.length) hb.innerHTML = `<li class="empty">첫 리밸런싱 계획이 나오면 종목별 목표 비율과 실제 보유가 여기에 나옵니다.</li>`;
  else hb.innerHTML = `<li class="wlegend" style="border:0;padding:0"><span><i></i>지금 보유</span><span><i class="over"></i>목표 초과분</span><span><i class="t"></i>목표</span></li>` + codes.map((c) => {
    const cur = eq ? (val[c] || 0) / eq : 0, tg = w[c] || 0;
    const p = (v) => ((100 * v) / top).toFixed(2);
    const over = cur > tg ? `<em style="left:${p(tg)}%;width:${p(cur - tg)}%"></em>` : "";
    const mark = tg > 0 ? `<u style="left:${p(tg)}%"></u>` : "";
    return `<li><div class="h-row"><span class="h-name">${esc(nm(c))}</span><span class="h-code">${esc(c)}${subOf[c] ? ` ${esc(nm(subOf[c]))} 대신` : ""}</span><span class="h-val">${won(val[c] || 0)}원</span></div><div class="h-bar"><i><b style="width:${p(Math.min(cur, tg))}%"></b>${over}${mark}</i><span style="min-width:86px">${(100 * cur).toFixed(1)}% / ${(100 * tg).toFixed(1)}%</span></div></li>`;
  }).join("");

  const ol = $("lastdec");
  const ords = (lv.orders || []).filter((o) => o.ts.slice(0, 10) === today);
  ol.innerHTML = ords.length ? ords.slice().reverse().map(orderRow).join("") : `<li class="empty" style="display:block">오늘 낸 주문이 없습니다. 국내는 09:00, 미국은 미국장이 열리자마자(지금은 23:30) 주문합니다.</li>`;
}

async function fxDone(b, id) {
  b.disabled = true;
  b.textContent = "잔고 확인 중";
  try {
    await api("/api/fx/done", { method: "POST", body: JSON.stringify({ id }) });
    load();
    setTimeout(load, 2500);
    setTimeout(load, 6000);
  } catch (e) {
    b.disabled = false;
    b.textContent = "다시 눌러 주세요";
  }
}

function orderRow(o, withDate) {
  const side = o.side === "BUY" ? "매수" : o.side === "SELL" ? "매도" : "";
  const req = o.request || {};
  const amt = req.amount ? `$${req.amount}` : req.quantity ? `${req.quantity}주` : o.quantity ? `${o.quantity}주` : o.amount ? `$${o.amount}` : "";
  let st = "", c = "";
  if (o.kind === "order") { st = o.status === "FILLED" ? "체결" : o.status === "PARTIAL_FILLED" ? "일부 체결" : o.status; c = o.status === "FILLED" ? "ok" : "bad"; }
  else if (o.kind === "dry_order") { st = "드라이런"; c = "dry"; }
  else if (o.kind === "order_error") { st = "오류"; c = "bad"; }
  else if (o.kind === "skip") { st = "건너뜀"; c = "dry"; }
  else if (o.kind === "scale_us") { st = "달러 부족"; c = "bad"; }
  else if (o.kind === "refuse") { st = "중단"; c = "bad"; }
  else if (o.kind === "fx_missing") { st = "환전 없음"; c = "bad"; }
  const detail = o.kind === "order" && o.avg_price ? `평균 ${o.avg_price}, 체결 ${o.filled_qty}` : o.reason || [o.code_detail ? `${o.error} ${o.code_detail}` : o.error, o.message].filter(Boolean).join(" ");
  const tm = withDate ? `${o.ts.slice(5, 10).replace("-", ".")}<br>${o.ts.slice(11, 16)}` : o.ts.slice(11, 16);
  return `<li><span class="tm">${tm}</span><span>${o.code ? esc(nm(o.code)) + " " : ""}${side} ${esc(amt)}</span><span class="st ${c}">${esc(st)}</span>${detail ? `<span class="sub2">${esc(detail)}</span>` : ""}</li>`;
}

const pc1 = (x) => `${(100 * (x || 0)).toFixed(1)}%`;
function decRow(day, d) {
  const li = document.createElement("li");
  if (d.rule === "allocator") {
    const tw = Object.entries(d.targets || {}).filter(([, w]) => w > 0).sort((a, b) => b[1] - a[1]);
    const top = tw.slice(0, 3).map(([c, w]) => `${esc(nm(c))} ${Math.round(100 * w)}%`).join(", ");
    const main = `<b>${tw.length ? top : "전부 현금"}</b>${tw.length > 3 ? `<span class="muted">외 ${tw.length - 3}종목${d.cash_weight > 0.005 ? `, 현금 ${Math.round(100 * d.cash_weight)}%` : ""}</span>` : ""}${d.vault_paid > 0.5 ? `<mark>금고 +${cmp(d.vault_paid)}</mark>` : ""}`;
    li.innerHTML = `<button type="button"><span class="d-date"><b>${day.slice(5).replace("-", ".")}</b><span>${day.slice(0, 4)}</span></span><span class="d-main">${main}</span><span class="d-k">${(d.candidates || []).length ? `${tw.length}종목` : "균등"}</span><i class="chev"></i></button>`;
    li.firstChild.onclick = () => openSheet(`${dot(day)} 리밸런싱`, rationaleModel(d));
    return li;
  }
  const picks = (d.candidates || []).filter((c) => c.selected).map((c) => esc(nm(c.code)));
  const main = `<b>${picks.length ? picks.join(", ") : "전부 현금"}</b>${picks.length && d.cash_weight > 0.001 ? `<span class="muted">현금 ${Math.round(100 * d.cash_weight)}%</span>` : ""}${d.vault_paid > 0.5 ? `<mark>금고 +${cmp(d.vault_paid)}</mark>` : ""}`;
  li.innerHTML = `<button type="button"><span class="d-date"><b>${day.slice(5).replace("-", ".")}</b><span>${day.slice(0, 4)}</span></span><span class="d-main">${main}</span><span class="d-k">${d.rule === "equal" ? "균등" : "K " + d.k}</span><i class="chev"></i></button>`;
  li.firstChild.onclick = () => openSheet(`${dot(day)} 리밸런싱`, rationale(d));
  return li;
}

function rationaleModel(d) {
  const tg = d.targets || {}, hb = d.holdings_before || {}, eq = d.equity || 1;
  const before = (c) => (hb[c] ? hb[c].value / eq : 0);
  const cands = d.candidates || [];
  const pay = d.vault_paid > 0.5 ? ` 이날 순이익 ${won(d.profit)}원이 이전 고점을 넘어 금고에 ${won(d.vault_paid)}원을 옮겼습니다.` : "";
  if (!cands.length) return `<p class="note">이 해에는 학습할 과거 데이터가 모자라 후보 전체에 같은 비율을 둡니다.${pay}</p>`;
  const codes = [...new Set([...cands.map((c) => c.code), ...Object.keys(hb)])];
  const up = codes.filter((c) => (tg[c] || 0) - before(c) > 0.01), down = codes.filter((c) => (tg[c] || 0) - before(c) < -0.01);
  const line = (label, list) => (list.length ? `<div><em>${label}</em><b>${list.map((c) => esc(nm(c))).join(", ")}</b></div>` : "");
  const first = !Object.keys(hb).length;
  const rows = cands.slice().sort((a, b) => (tg[b.code] || 0) - (tg[a.code] || 0)).map((c, i) => `<div class="cand${(tg[c.code] || 0) > 0 ? " pick" : ""}"><span class="nm"><i>${i + 1}</i><span>${esc(nm(c.code))}</span></span><span class="n">${pc1(before(c.code))}</span><span class="n"><b>${pc1(tg[c.code])}</b></span><span class="n">${c.score.toFixed(2)}</span></div>`).join("");
  return `<p class="note">20거래일마다 모델이 전날 종가로 후보마다 점수를 매기고, 점수가 높을수록 목표 비율을 크게 줍니다. 점수가 1 높으면 비율이 약 2.7배입니다. 가진 비율이 목표보다 적으면 사고 많으면 팔아서 맞춥니다.${first ? " 첫 리밸런싱이라 전부 새로 삽니다." : ""}${d.cash_weight > 0.005 ? ` 현금 비율은 ${Math.round(100 * d.cash_weight)}%입니다.` : ""}${pay}</p>
    ${first ? "" : `<div class="changes">${line("늘림", up)}${line("줄임", down)}</div>`}
    <div class="cands"><div class="cand head"><span>종목</span><span class="n">전</span><span class="n">목표</span><span class="n">점수</span></div>${rows}</div>`;
}

function rationale(d) {
  const why = d.rule === "equal" ? "후보 전체에 같은 비중을 둡니다." : `6개월 수익과 12-1개월 수익을 종목끼리 비교한 점수가 높은 ${d.k}개에 ${Math.round(100 / d.k)}%씩 넣습니다.${d.rule === "dual" ? " 12-1개월 수익이 0 이하인 종목은 사지 않고 그 몫을 현금으로 둡니다." : ""}`;
  const tg = d.targets || {};
  const before = Object.keys(d.holdings_before || {});
  const after = Object.keys(tg).filter((c) => tg[c] > 0);
  const bought = after.filter((c) => !before.includes(c)), sold = before.filter((c) => !after.includes(c)), kept = after.filter((c) => before.includes(c));
  const line = (label, list) => (list.length ? `<div><em>${label}</em><b>${list.map((c) => esc(nm(c))).join(", ")}</b></div>` : "");
  const rows = (d.candidates || []).map((c) => `<div class="cand${c.selected ? " pick" : ""}"><span class="nm"><i>${c.rank ?? ""}</i><span>${esc(nm(c.code))}</span>${c.selected && tg[c.code] ? `<b class="w">${Math.round(100 * tg[c.code])}%</b>` : ""}</span><span class="n ${cls(c.mom126_pct)}">${pct(c.mom126_pct)}</span><span class="n ${cls(c.mom12_1_pct)}">${pct(c.mom12_1_pct)}</span><span class="n">${c.z == null ? "-" : c.z.toFixed(2)}</span>${!c.selected && c.reason && c.reason !== "순위 밖" ? `<span class="why">${esc(c.reason)}</span>` : ""}</div>`).join("");
  return `<p class="note">${why}${d.vault_paid > 0.5 ? ` 이날 순이익 ${won(d.profit)}원이 이전 고점을 넘어 금고에 ${won(d.vault_paid)}원을 옮겼습니다.` : ""}</p>
    <div class="changes">${line("새로 삼", bought)}${line("유지", kept)}${line("팖", sold)}</div>
    <div class="cands"><div class="cand head"><span>종목</span><span class="n">6개월</span><span class="n">12-1개월</span><span class="n">점수</span></div>${rows}</div>`;
}

let sheetPushed = false;
function openSheet(title, html, cls) {
  $("sheet-title").textContent = title;
  $("sheet-body").innerHTML = html;
  const panel = $("sheet").querySelector(".panel");
  panel.className = "panel" + (cls ? " " + cls : "");
  $("sheet").hidden = false;
  panel.scrollTop = 0;
  if (!sheetPushed) { history.pushState({ sheet: true }, ""); sheetPushed = true; }
  $("sheet-close").focus({ preventScroll: true });
}
function closeSheet() { if (sheetPushed) history.back(); else $("sheet").hidden = true; }
$("sheet-close").onclick = closeSheet;
$("sheet").onclick = (e) => { if (e.target.id === "sheet") closeSheet(); };
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("sheet").hidden) closeSheet(); });
window.addEventListener("popstate", () => { if (sheetPushed) { sheetPushed = false; $("sheet").hidden = true; } });

const FEAT = { m20: "20일 수익", m126: "6개월 수익", m12_1: "12-1개월 수익", vol60: "60일 변동성", volr: "변동성 확대(20/60일)", dd252: "1년 고점 대비", gap200: "200일선 괴리" };
const STATEF = { s_vol20: "시장 20일 변동성", s_trend200: "시장 200일 추세", s_dd252: "시장 1년 고점 대비", s_breadth: "200일선 위 종목 비율" };
const fmtFeat = (k, v) => (v == null || isNaN(v) ? "-" : k === "volr" ? v.toFixed(2) + "배" : k === "vol60" || k === "s_vol20" || k === "s_breadth" ? (100 * v).toFixed(0) + "%" : pct(100 * (k === "dd252" || k === "s_dd252" ? v : Math.expm1(v))));
function candSheet(x) {
  const rows = Object.keys(FEAT).map((k) => `<dt>${FEAT[k]}</dt><dd>${fmtFeat(k, x.raw[k])} <span class="${cls(x.contrib[k])}" style="font-size:12px">(${x.contrib[k] >= 0 ? "+" : ""}${x.contrib[k].toFixed(2)})</span></dd>`).join("");
  return `<p class="note">점수 ${x.score.toFixed(2)}, 비율 ${(100 * x.weight).toFixed(1)}%. 괄호는 그 특징이 점수에 더하거나 뺀 값입니다. 점수가 1 높으면 비율이 약 2.7배가 됩니다.</p><dl class="kv" style="margin-top:10px">${rows}</dl>`;
}
function renderWhy() {
  const lv = LV(), plan = lv.plan;
  $("why-lead").textContent = "모델이 후보(29종목과 도비가 가진 종목) 각각에 점수를 매기고, 점수가 높을수록 비율을 크게 줍니다. 현금도 후보 하나로 같이 경쟁합니다. 해마다 과거 데이터로만 다시 학습합니다.";
  const ul = $("declist");
  if (!plan || !plan.rationale) { ul.innerHTML = `<li class="empty">첫 계획이 나오면 모델이 무엇을 보고 비율을 정했는지 여기에 나옵니다.</li>`; return; }
  const r = plan.rationale;
  const bmax = Math.max(0.01, ...Object.values(r.beta || {}).map(Math.abs));
  const beta = Object.entries(r.beta || {}).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).map(([k, v]) => `<div><span>${FEAT[k] || k}<span class="muted" style="font-size:12px"> ${v >= 0 ? "높을수록 더 삼" : "높을수록 덜 삼"}</span></span><span><b style="${v >= 0 ? "left:50%" : "right:50%"};width:${(50 * Math.abs(v)) / bmax}%;background:${v >= 0 ? "var(--up)" : "var(--down)"}"></b></span><span class="${cls(v)}">${v >= 0 ? "+" : ""}${v.toFixed(2)}</span></div>`).join("");
  const cands = (r.candidates || []).slice().sort((a, b) => b.weight - a.weight);
  ul.innerHTML = `<li style="border:0"><h2>이번 학습에서 본 것<small>${dot(plan.decided_on)} 종가, 학습 ${r.trained_rows}구간</small></h2><div class="beta">${beta}</div><p class="note">시장 상태: ${Object.entries(r.state || {}).map(([k, v]) => `${STATEF[k]} ${fmtFeat(k, v)}`).join(", ")}. 현금 비율 ${(100 * (r.cash_weight || 0)).toFixed(0)}%.</p></li>
    <li style="border:0;margin-top:22px"><h2>종목별 점수와 비율<small>누르면 자세히</small></h2></li>`;
  for (const x of cands) {
    const li = document.createElement("li");
    li.innerHTML = `<button type="button"><span class="d-date"><b>${(100 * x.weight).toFixed(1)}%</b><span>${x.rank}위</span></span><span class="d-main"><b>${esc(nm(x.code))}</b><span class="muted">${esc(x.code)}${x.outside ? ", 학습 범위 밖" : ""}</span></span><span class="d-k">점수 ${x.score.toFixed(2)}</span><i class="chev"></i></button>`;
    li.firstChild.onclick = () => openSheet(`${nm(x.code)} (${x.code})`, candSheet(x));
    ul.appendChild(li);
  }
}

const SC_WHO = { desk: "시나리오 데스크", "codex-seoha": "서버 Codex 이슈", dot: "도트 이슈" };
const fp = (f, d = 2) => (f == null || isNaN(f) ? "-" : Number((100 * Math.abs(f)).toFixed(d)) === 0 ? (0).toFixed(d) + "%" : (f > 0 ? "+" : "−") + (100 * Math.abs(f)).toFixed(d) + "%");
const fr = (f) => (f == null ? "-" : Math.round(100 * f) + "%");
const cl2 = (f) => cls(f == null ? null : Number((100 * f).toFixed(2)));
const stamp = (ts) => (ts ? `${dot(ts.slice(0, 10))} ${ts.slice(11, 16)}` : "");

function scoreHTML(sc) {
  if (!sc) return `<section class="block"><h2>예측 점수판</h2><p class="empty">매일 07:10(미국장 마감 뒤)과 16:10(국내장 마감 뒤)에 처음 계산되면 여기에 나옵니다.</p></section>`;
  const d = sc.desk || {}, t = d.total || {};
  const by = (sc.calls && sc.calls.by_author) || {};
  const order = Object.keys(by).sort((a, b) => (a === "desk" ? -1 : b === "desk" ? 1 : a.localeCompare(b)));
  const who = order.map((au) => {
    const a = by[au];
    const done = a.matured ? `방향 적중 ${fr(a.hit_rate)}, 그 방향으로 샀다면 비용 후 평균 ${fp(a.net_mean, 1)}` : "아직 없음";
    const open = a.open ? `지금까지 방향 적중 ${fr(a.open_hit_rate)}, 비용 후 평균 ${fp(a.open_net_mean, 1)}` : "아직 없음";
    const vs = a.base_n ? `. 같은 ${a.base_n}건을 과거 평균으로 예측했다면 평균 오차 ${(100 * a.base_mae).toFixed(1)}%p` : "";
    const mag = a.mag_n ? `<p class="note">예상 크기와 실제의 평균 차이 ${fp(a.mag_bias, 1)}, 평균 오차 ${(100 * a.mag_mae).toFixed(1)}%p (${a.mag_n}건)${vs}</p>` : "";
    return `<li><div class="sc-who"><b>${esc(SC_WHO[au] || au)}</b><span>${a.all ?? a.calls}건 중 ${a.calls}건 채점</span></div><dl class="kv sc-kv"><dt>기간 끝남 ${a.matured}건</dt><dd>${done}</dd><dt>진행 중 ${a.open}건</dt><dd>${open}</dd></dl>${mag}</li>`;
  }).join("");
  const items = (sc.calls && sc.calls.items) || [];
  const dk = items.filter((x) => x.author === "desk");
  const lastAt = dk.reduce((m, x) => (x.at > m ? x.at : m), "");
  const latest = dk.filter((x) => x.at === lastAt).sort((a, b) => Math.abs(b.magnitude) - Math.abs(a.magnitude));
  const state = (x) => (!x.entry_day ? "시작 전" : x.matured ? (x.hit ? "맞음" : "틀림") : `${x.days}/${x.horizon}일`);
  const withBase = latest.filter((x) => x.base != null);
  const avg = (xs) => xs.reduce((s, v) => s + v, 0) / xs.length;
  const baseNote = withBase.length ? `<p class="note" style="margin-top:8px">이 예측 평균 ${pct(avg(withBase.map((x) => x.magnitude)))}, 같은 종목들의 과거 10년 20거래일 평균 ${pct(avg(withBase.map((x) => x.base)))}. 데스크 예상은 과거 평균 ± 표준편차의 1/4 안으로 보정됩니다.</p>` : "";
  const rows = latest.map((x) => `<div><span>${esc(nm(x.code))}</span><span class="${cls(x.magnitude)}">${pct(x.magnitude)}</span><span class="${x.entry_day ? cls(x.ret) : ""}">${x.entry_day ? fp(x.ret, 1) : "-"}</span><span class="${x.matured ? (x.hit ? "up" : "down") : "muted"}">${state(x)}</span></div>`).join("");
  return `<section class="block"><h2>예측 점수판<small>${stamp(sc.asof)} 계산</small></h2>
    <dl class="kv"><dt>데스크 비율 조정을 따랐다면 (비용 후)</dt><dd class="${cl2(t.net)}">${fp(t.net)}</dd><dt>조정으로 번 것</dt><dd class="${cl2(t.gross)}">${fp(t.gross)}</dd><dt>거래 비용</dt><dd>${fp(-(t.cost || 0))}</dd><dt>계산한 날</dt><dd>${(d.daily || []).length}일</dd></dl>
    <ul class="sc-list">${who}</ul>
    ${latest.length ? `<h2 style="margin-top:22px">데스크 최근 예측과 실제<small>${stamp(lastAt)} 예측, 20거래일 기준</small></h2><div class="sc-table"><div class="sc-head"><span>종목</span><span>예상</span><span>실제</span><span>상태</span></div>${rows}</div>${baseNote}` : ""}
    <p class="note" style="margin-top:10px">예측이 나온 다음 장 시가에 샀다고 보고 계산하고, 수수료와 호가 차이를 비용으로 뺍니다. 실제는 시가부터 마지막 종가까지의 수익률입니다(국내 ${dot(sc.last_close && sc.last_close.KR)}, 미국 ${dot(sc.last_close && sc.last_close.US)} 종가까지). 이슈 예측 중 방향을 '혼조'나 '불분명'으로 적은 것은 채점하지 않습니다.</p></section>`;
}

function renderReport() {
  const lv = LV();
  $("report-lead").textContent = "도비 장부의 자동 매매 기록입니다. 계좌의 다른 돈과 종목은 건드리지 않습니다.";
  const box = $("report");
  const ords = (lv.orders || []).slice().reverse();
  const v = lv.vault;
  const plans = (lv.plans || []).slice().reverse();
  box.innerHTML = scoreHTML(S && S.scorecard) + `<section class="block"><h2>체결과 주문<small>${ords.length}건</small></h2><ul class="olist">${ords.length ? ords.slice(0, 60).map((o) => orderRow(o, true)).join("") : '<li class="empty" style="display:block">아직 주문이 없습니다.</li>'}</ul></section>
    <section class="block"><h2>금고</h2>${v ? `<dl class="kv"><dt>금고 누적</dt><dd>${won(v.vault)}원</dd><dt>넣은 돈</dt><dd>${won(v.principal)}원</dd><dt>판 것의 손익 누적</dt><dd class="${cls(v.profit)}">${signed(v.profit)}원</dd>${v.tax_deduction != null ? `<dt>올해 미국 양도차익</dt><dd class="${cls(v.tax_year_gain || 0)}">${signed(v.tax_year_gain || 0)}원</dd><dt>미국 양도세 적립</dt><dd>${won(v.tax_reserve || 0)}원</dd>` : ""}</dl>` : '<p class="empty">아직 판 종목이 없습니다.</p>'}<p class="note" style="margin-top:10px">팔아서 생긴 손익 누적이 이전 최고를 넘으면 넘은 만큼의 절반을 금고로 떼어 둡니다. 금고 돈은 계좌에 그대로 있고 도비는 쓰지 않습니다. 손실은 남긴 이익에서 먼저 빠집니다. 미국 종목은 올해 원화 기준 양도차익이 ${won(v && v.tax_deduction != null ? v.tax_deduction : 2500000)}원을 넘으면 넘은 만큼의 22%를 양도세로 따로 떼어 두고, 남은 이익으로 금고를 계산합니다.</p></section>
    <section class="block"><h2>계획 이력</h2>${plans.length ? `<dl class="kv">${plans.map((p) => `<dt>${dot(p.decided_on)} 종가</dt><dd>${Object.keys(p.weights).length}종목</dd>`).join("")}</dl>` : '<p class="empty">아직 계획이 없습니다.</p>'}</section>
    ${(lv.errors || []).length ? `<section class="block"><h2>최근 오류</h2><dl class="kv">${lv.errors.slice().reverse().map((e) => `<dt>${kstStamp(e.ts)}</dt><dd class="down">${esc(e.error)} ${esc(e.where || "")}</dd>`).join("")}</dl></section>` : ""}`;
}

const DATA_LABEL = { historical: "과거 시세", bootstrap: "과거 섞기", bear: "하락장", sideways: "횡보장", volatile: "고변동" };
const DATA_NOTE = {
  historical: "실제 시세를 그대로 씁니다.",
  bootstrap: "과거 수익률을 20거래일 묶음으로 무작위로 이어 붙여 새 시세를 만듭니다.",
  bear: "섞은 시세에서 주식형 종목 수익률을 연 20%씩 깎습니다.",
  sideways: "섞은 시세에서 종목마다 평균 수익률을 빼서 추세를 없앱니다.",
  volatile: "평균은 그대로 두고 흔들림을 1.5배로 키웁니다.",
};
const MODE_LABEL = { model: "비율 모델", topk: "모멘텀 집중(예전)", dual: "모멘텀 + 하락 시 현금(예전)", equal: "29종 균등" };
const MODE_NOTE = { model: "도비와 같은 비율 모델입니다. 해마다 그 전 데이터로만 학습한 모델이 후보마다 비율을 정하고 20거래일마다 맞춥니다. 같은 기간 29종 균등 분배와 같이 보여 줍니다. 2012년 전에는 학습할 데이터가 모자라 균등으로 둡니다.", topk: "", dual: "", equal: "" };
const runTitle = (p) => `${DATA_LABEL[p.data] || p.data}, ${MODE_LABEL[p.mode] || p.mode}`;
const runSub = (p) => [p.data === "historical" ? `${dot(p.start)} ~ ${dot(p.end)}` : `${p.paths}개 경로, ${p.years}년`, p.mode === "equal" || p.mode === "model" ? "" : `K ${p.k === "auto" ? "자동" : p.k}`].filter(Boolean).join(", ");

let runsTimer = 0;
async function loadRuns() {
  clearTimeout(runsTimer);
  const d = await api("/api/bt/list");
  const ul = $("runs");
  ul.innerHTML = "";
  if (!d.runs.length) ul.innerHTML = `<li class="empty">아직 실행한 모의투자가 없습니다. 조건을 고르고 실행해 보세요.</li>`;
  for (const r of d.runs) {
    const p = r.params || {};
    let res;
    if (r.status === "done" && r.summary) {
      const t = r.summary.total;
      res = typeof t === "number" ? `${cmp(t)}<small>합계</small>` : t && t.median != null ? `${cmp(t.median)}<small>중앙값</small>` : "완료";
    } else res = `<span class="state ${r.status === "failed" ? "fail" : "busy"}">${r.status === "failed" ? "실패" : r.status === "running" ? "계산 중" : "대기"}</span>`;
    const li = document.createElement("li");
    li.innerHTML = `<button type="button" ${r.status === "done" ? "" : "disabled"}><span class="run-main"><b>${esc(runTitle(p))}</b><span>${esc(runSub(p))}</span></span><span class="run-res">${res}</span>${r.status === "done" ? '<i class="chev"></i>' : ""}</button>`;
    if (r.status === "done") li.firstChild.onclick = () => { fromList = true; location.hash = "run=" + encodeURIComponent(r.id); };
    ul.appendChild(li);
  }
  if (d.runs.some((r) => r.status === "queued" || r.status === "running")) runsTimer = setTimeout(() => loadRuns().catch(() => {}), 4000);
}

function bars(rows, dep) {
  const max = Math.max(...rows.map((r) => r.total), dep || 0) * 1.02;
  const mark = dep ? `<u style="left:${((100 * dep) / max).toFixed(2)}%"></u>` : "";
  return `<div class="bars">${rows.map((r) => `<div class="bar-row${r.me ? " mine" : ""}"><span class="lab">${esc(r.label)}</span><span class="val">${cmp(r.total)}${r.vault ? `<mark>금고 ${cmp(r.vault)}</mark>` : ""}</span><span class="track"><b style="width:${((100 * r.total) / max).toFixed(2)}%"></b>${mark}</span></div>`).join("")}</div>${dep ? `<div class="bars-note"><i></i>넣은 돈 ${cmp(dep)}</div>` : ""}`;
}

function strip(paths) {
  const vals = paths.flatMap((p) => [p.total, p.equal_total, p.deposits]).filter((v) => v != null);
  const lo = Math.min(...vals), hi = Math.max(...vals), span = hi - lo || 1;
  const X = (v) => ((100 * (v - lo)) / span).toFixed(2) + "%";
  const dep = paths[0].deposits;
  const dots = paths.map((p) => `<span class="dt e" style="left:${X(p.equal_total)};top:50px" title="균등 분배 ${cmp(p.equal_total)}"></span><span class="dt s" style="left:${X(p.total)};top:20px" title="이 전략 ${cmp(p.total)}"></span>`).join("");
  return `<div class="strip" role="img" aria-label="경로별 마지막 날 합계"><span class="lane" style="top:20px"></span><span class="lane" style="top:50px"></span><span class="dep" style="left:${X(dep)}"></span>${dots}</div>
    <div class="strip-key"><span>${cmp(lo)}</span><span class="k"><span><i class="s"></i>이 전략</span><span><i class="e"></i>균등 분배</span><span><i class="d"></i>넣은 돈</span></span><span>${cmp(hi)}</span></div>`;
}

function decisionList(host, list) {
  const ul = document.createElement("ul");
  ul.className = "rows";
  const all = list.slice().reverse();
  let shown = 0;
  const more = document.createElement("button");
  more.type = "button";
  more.className = "btn quiet wide more";
  const page = () => {
    all.slice(shown, shown + 20).forEach((d) => ul.appendChild(decRow(d.day, d)));
    shown = Math.min(all.length, shown + 20);
    more.hidden = shown >= all.length;
    more.textContent = `${all.length - shown}회 더 보기`;
  };
  more.onclick = page;
  host.append(ul, more);
  page();
}

let fromList = false;
function back() { if (fromList) { fromList = false; history.back(); } else go("sim"); }
let runReq = 0;
async function showRun(id) {
  const box = $("sim-run");
  const my = ++runReq;
  box.innerHTML = `<button class="back" type="button">모의투자</button><p class="empty">결과를 불러오는 중입니다.</p>`;
  box.querySelector(".back").onclick = back;
  let r;
  try { r = await api("/api/bt/get?id=" + encodeURIComponent(id)); } catch (e) { if (my === runReq) box.querySelector(".empty").textContent = "불러오지 못했습니다: " + e.message; return; }
  if (my !== runReq) return;
  const p = r.params || {};
  const same = (m) => m;
  if (r.kind === "historical") {
    const s = r.summary, b = r.benchmarks, first = r.series[0], lastD = r.series[r.series.length - 1];
    const rows = [{ label: "이 전략", total: s.total, vault: s.vault, me: true }];
    if (b.equal && p.mode !== "equal") rows.push({ label: "같은 후보 균등 분배", total: b.equal.total, vault: b.equal.vault });
    if (b.hold_069500) rows.push({ label: "KODEX 200만 보유", total: b.hold_069500.total });
    if (b.hold_SPY) rows.push({ label: "S&P500만 보유", total: b.hold_SPY.total });
    const ky = r.k_by_year ? Object.entries(r.k_by_year).map(([y, v]) => `<span>${y}<b>K ${v.k}</b></span>`).join("") : "";
    box.innerHTML = `<button class="back" type="button">모의투자</button>
      <h1>${esc(runTitle(p))}</h1>
      <p class="sub">${dot(first.day)} ~ ${dot(lastD.day)}, 하루 ${cmp(p.deposit || 10000)}원${p.realistic === false ? "" : ", 현실 조건"}${p.mode === "equal" || p.mode === "model" ? "" : `, K ${p.k === "auto" ? "연도별 자동" : p.k}`}</p>
      <div class="run-grid"><div>
        <div id="run-chart"></div>
      </div><div>
        <section class="block"><h2>마지막 날 합계</h2>${bars(rows, s.deposits)}</section>
        <div class="grid2">
          <div><span>넣은 돈</span><b>${cmp(s.deposits)}</b></div>
          <div><span>금고</span><b>${cmp(s.vault)}</b></div>
          <div><span>낸 세금</span><b>${cmp(s.taxes)}</b></div>
          <div><span>순이익 최악 낙폭</span><b class="${cls(s.worst_profit_drawdown)}">${cmp(s.worst_profit_drawdown)}</b></div>
          <div><span>금고 적립</span><b>${s.payouts}회</b></div>
          <div><span>계산 시간</span><b>${r.elapsed_s != null ? r.elapsed_s.toFixed(0) + "초" : "-"}</b></div>
        </div>
      </div></div>
      ${ky ? `<section class="block"><h2>연도별 K</h2><div class="kyears">${ky}</div><p class="note" style="margin-top:8px">해마다 직전 3년을 K 2, 3, 5, 8로 돌려 합계가 가장 컸던 K를 그해에 씁니다.</p></section>` : ""}
      <section class="block"><h2>리밸런싱<small>${r.decisions.length}회</small></h2><div id="run-dec"></div></section>`;
    NSChart.create($("run-chart"), Object.assign({ band: { top: "total", base: "deposits" }, label: "모의투자 평가 추이" }, liveSeries(r.series, same)));
    decisionList($("run-dec"), r.decisions);
  } else {
    const d = r.distribution, n = r.paths.length;
    box.innerHTML = `<button class="back" type="button">모의투자</button>
      <h1>${esc(runTitle(p))}</h1>
      <p class="sub">가상 경로 ${n}개, 경로마다 ${p.years}년, 하루 ${cmp(p.deposit || 10000)}원${p.mode === "model" ? ", 비율 모델" : `, K ${p.k_used ?? p.k}`}</p>
      <section class="block"><h2>경로별 마지막 날 합계</h2>${strip(r.paths)}</section>
      <div class="grid2">
        <div><span>하위 10%</span><b>${cmp(d.total.p10)}</b></div>
        <div><span>중앙값</span><b>${cmp(d.total.median)}</b></div>
        <div><span>상위 10%</span><b>${cmp(d.total.p90)}</b></div>
        <div><span>금고 중앙값</span><b>${cmp(d.vault.median)}</b></div>
        <div><span>균등 분배보다 많은 경로</span><b>${d.beats_equal} / ${n}</b></div>
        <div><span>넣은 돈보다 적은 경로</span><b class="${d.loses_money ? "down" : ""}">${d.loses_money} / ${n}</b></div>
      </div>
      <section class="block"><h2>첫 경로 흐름<small>날짜는 가상</small></h2><div id="run-chart"></div></section>
      <section class="block"><h2>첫 경로 리밸런싱<small>${r.example.decisions.length}회</small></h2><div id="run-dec"></div></section>`;
    NSChart.create($("run-chart"), Object.assign({ band: { top: "total", base: "deposits" }, label: "첫 가상 경로 평가 추이" }, liveSeries(r.example.series, same)));
    decisionList($("run-dec"), r.example.decisions);
  }
  box.querySelector(".back").onclick = back;
}

const MIN_DAY = "2006-01-02", LAST_DAY = "2026-11-05";
const clampDay = (s) => (s < MIN_DAY ? MIN_DAY : s > LAST_DAY ? LAST_DAY : s);
const SIM = { data: "historical", mode: "model", k: "auto", preset: "2016", start: "2016-01-04", end: LAST_DAY, paths: 20, years: 10, deposit: 10000, realistic: true };
const OPTS = {
  data: Object.entries(DATA_LABEL),
  k: [["auto", "자동"], ["2", "2개"], ["3", "3개"], ["5", "5개"], ["8", "8개"]],
  preset: [["1", "최근 1년"], ["3", "최근 3년"], ["5", "최근 5년"], ["2016", "2016년부터"], ["custom", "직접"]],
  deposit: [[10000, "1만 원"], [30000, "3만 원"], [50000, "5만 원"], [100000, "10만 원"]],
};
function presetRange(v) {
  if (v === "2016") return ["2016-01-04", LAST_DAY];
  return [clampDay(addDays(addYears(LAST_DAY, -Number(v)), 1)), LAST_DAY];
}
function applyPreset(v) {
  SIM.preset = v;
  if (v === "custom") return;
  [SIM.start, SIM.end] = presetRange(v);
}
function presetOf(a, b) {
  for (const [v] of OPTS.preset) if (v !== "custom") { const [x, y] = presetRange(v); if (x === a && y === b) return v; }
  return "custom";
}

function lastOfMonth(y, m) { return new Date(Date.UTC(y, m, 0)).getUTCDate(); }
function parseDay(raw, which) {
  const t = raw.trim().replace(/\s+/g, "").replace(/일$/, "").replace(/\.$/, "");
  if (!t) return { err: "날짜를 입력하세요." };
  const cands = [];
  let r;
  if ((r = /^(\d{4})(\d{2})(\d{2})$/.exec(t)) || (r = /^(\d{4})[.\-/년](\d{1,2})[.\-/월](\d{1,2})$/.exec(t))) cands.push([+r[1], +r[2], +r[3]]);
  if ((r = /^(\d{2})[.\-/](\d{1,2})[.\-/](\d{1,2})$/.exec(t)) || (r = /^(\d{2})(\d{2})(\d{2})$/.exec(t))) cands.push([2000 + +r[1], +r[2], +r[3]]);
  if ((r = /^(\d{4})[.\-/년]?(\d{1,2})월?$/.exec(t))) { const y = +r[1], m = +r[2]; cands.push([y, m, m >= 1 && m <= 12 ? (which === "end" ? lastOfMonth(y, m) : 1) : 1]); }
  if (!cands.length) return { err: "2016.06.03이나 20160603처럼 입력하세요." };
  let firstErr = null;
  for (const [y, m, d] of cands) {
    const dt = new Date(Date.UTC(y, m - 1, d));
    if (dt.getUTCFullYear() !== y || dt.getUTCMonth() !== m - 1 || dt.getUTCDate() !== d) { firstErr = firstErr || `${y}년 ${m}월 ${d}일은 없는 날짜입니다.`; continue; }
    const day = iso(dt);
    if (day < MIN_DAY || day > LAST_DAY) { firstErr = firstErr || `${dot(MIN_DAY)}부터 ${dot(LAST_DAY)} 사이로 입력하세요. 시세가 그 사이에만 있습니다.`; continue; }
    return { day };
  }
  return { err: firstErr };
}
function spanText(a, b) {
  const days = Math.round((toD(b) - toD(a)) / 864e5) + 1;
  const y = days / 365.25;
  return `${days.toLocaleString("ko-KR")}일, 약 ${y >= 1 ? y.toFixed(1) + "년" : Math.max(1, Math.round(days / 30.44)) + "개월"}`;
}
function commitDate(which) {
  const inp = $("d-" + which);
  const r = parseDay(inp.value, which);
  const err = $("d-err");
  if (r.err) {
    $("df-" + which).classList.add("bad");
    err.textContent = (which === "start" ? "시작일: " : "끝일: ") + r.err;
    err.hidden = false;
    return false;
  }
  $("df-" + which).classList.remove("bad");
  err.hidden = true;
  const other = which === "start" ? "end" : "start";
  const a = which === "start" ? r.day : SIM.start, b = which === "end" ? r.day : SIM.end;
  if (a >= b) {
    $("df-" + which).classList.add("bad");
    err.textContent = "시작일은 끝일보다 앞이어야 합니다.";
    err.hidden = false;
    return false;
  }
  $("df-" + other).classList.remove("bad");
  SIM[which] = r.day;
  SIM.preset = presetOf(SIM.start, SIM.end);
  renderSim(true);
  return true;
}
for (const w of ["start", "end"]) {
  const inp = $("d-" + w);
  inp.placeholder = w === "start" ? "2016.01.04" : dot(LAST_DAY);
  inp.addEventListener("change", () => commitDate(w));
  inp.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); if (commitDate(w)) inp.blur(); } });
  inp.addEventListener("input", () => $("df-" + w).classList.remove("bad"));
}
document.querySelectorAll("[data-cal]").forEach((b) => (b.onclick = () => openCal(b.dataset.cal)));

const CAL = { active: "start", start: "", end: "", view: "", mode: "days" };
const chev = (d) => `<svg class="ico" viewBox="0 0 24 24"><path d="${d < 0 ? "M15 6l-6 6 6 6" : "M9 6l6 6-6 6"}"/></svg>`;
function openCal(which) {
  Object.assign(CAL, { active: which, start: SIM.start, end: SIM.end, mode: "days" });
  CAL.view = (which === "start" ? SIM.start : SIM.end).slice(0, 7);
  openSheet("기간 고르기", `<div id="cal"></div>`, "narrow");
  drawCal(CAL[which]);
}
function ym(y, m) { return `${y}-${String(m).padStart(2, "0")}`; }
function shiftMonth(v, n) { let [y, m] = v.split("-").map(Number); m += n; while (m > 12) { m -= 12; y++; } while (m < 1) { m += 12; y--; } return ym(y, m); }
function drawCal(focusDay) {
  const box = $("cal");
  if (!box) return;
  const [vy, vm] = CAL.view.split("-").map(Number);
  const minM = MIN_DAY.slice(0, 7), maxM = LAST_DAY.slice(0, 7);
  const prevOk = CAL.mode === "days" ? CAL.view > minM : CAL.mode === "months" ? vy > +MIN_DAY.slice(0, 4) : false;
  const nextOk = CAL.mode === "days" ? CAL.view < maxM : CAL.mode === "months" ? vy < +LAST_DAY.slice(0, 4) : false;
  const title = CAL.mode === "years" ? "연도 고르기" : CAL.mode === "months" ? `${vy}년` : `${vy}년 ${vm}월`;
  let body = "";
  if (CAL.mode === "days") {
    const first = new Date(Date.UTC(vy, vm - 1, 1)).getUTCDay(), n = lastOfMonth(vy, vm);
    const inView = (s) => s && s.slice(0, 7) === CAL.view;
    let target = focusDay && inView(focusDay) ? focusDay : inView(CAL[CAL.active]) ? CAL[CAL.active] : null;
    const cells = [];
    for (let i = 0; i < first; i++) cells.push(`<span class="day blank"></span>`);
    for (let d = 1; d <= n; d++) {
      const s = `${CAL.view}-${String(d).padStart(2, "0")}`;
      const off = s < MIN_DAY || s > LAST_DAY;
      if (!target && !off) target = s;
      const dow = (first + d - 1) % 7;
      const c = ["day", dow === 0 || dow === 6 ? "we" : "", s === CAL.start ? "s" : "", s === CAL.end ? "e" : "", CAL.end && s > CAL.start && s < CAL.end ? "in" : "", s === LAST_DAY ? "last" : ""].filter(Boolean).join(" ");
      cells.push(`<button type="button" class="${c}" data-d="${s}" ${off ? "disabled" : ""} tabindex="-1" aria-label="${vy}년 ${vm}월 ${d}일 ${WD[dow]}요일${s === LAST_DAY ? ", 마지막 시세" : ""}" aria-pressed="${s === CAL.start || s === CAL.end}"><span>${d}</span></button>`);
    }
    body = `<div class="cal-grid" role="grid" aria-label="${title}">${WD.map((w) => `<span class="dow">${w}</span>`).join("")}${cells.join("")}</div>`;
    CAL._target = target;
  } else if (CAL.mode === "years") {
    const ys = [];
    for (let y = +MIN_DAY.slice(0, 4); y <= +LAST_DAY.slice(0, 4); y++) ys.push(`<button type="button" data-y="${y}" aria-current="${y === vy}">${y}</button>`);
    body = `<div class="cal-ys">${ys.join("")}</div>`;
  } else {
    const ms = [];
    for (let m = 1; m <= 12; m++) { const v = ym(vy, m); ms.push(`<button type="button" data-m="${v}" aria-current="${v === CAL.view}" ${v < minM || v > maxM ? "disabled" : ""}>${m}월</button>`); }
    body = `<div class="cal-ys">${ms.join("")}</div>`;
  }
  box.innerHTML = `<div class="cal-tabs">${["start", "end"].map((k) => `<button type="button" data-tab="${k}" aria-pressed="${CAL.active === k}"><span>${k === "start" ? "시작일" : "끝일"}${CAL.active === k ? " 고르는 중" : ""}</span><b>${CAL[k] ? dot(CAL[k]) : "고르세요"}</b></button>`).join("")}</div>
    <div class="cal-head"><button class="cal-nav" type="button" data-step="-1" aria-label="이전" ${prevOk ? "" : "disabled"}>${chev(-1)}</button><button class="cal-title" type="button" aria-expanded="${CAL.mode !== "days"}">${title}</button><button class="cal-nav" type="button" data-step="1" aria-label="다음" ${nextOk ? "" : "disabled"}>${chev(1)}</button></div>
    ${body}
    <div class="cal-foot"><p>${CAL.end ? spanText(CAL.start, CAL.end) : "끝일을 고르세요"}</p><button class="btn" type="button" data-apply ${CAL.end ? "" : "disabled"}>적용</button></div>`;
  box.querySelectorAll("[data-tab]").forEach((b) => (b.onclick = () => { CAL.active = b.dataset.tab; CAL.mode = "days"; CAL.view = (CAL[CAL.active] || CAL.start).slice(0, 7); drawCal(CAL[CAL.active] || null); }));
  box.querySelectorAll("[data-step]").forEach((b) => (b.onclick = () => {
    const n = +b.dataset.step;
    if (CAL.mode === "days") CAL.view = shiftMonth(CAL.view, n);
    else CAL.view = ym(vy + n, vm);
    drawCal();
  }));
  box.querySelector(".cal-title").onclick = () => { CAL.mode = CAL.mode === "days" ? "years" : CAL.mode === "years" ? "days" : "years"; drawCal(); };
  box.querySelectorAll("[data-y]").forEach((b) => (b.onclick = () => { const y = +b.dataset.y; let v = ym(y, vm); if (v < minM) v = minM; if (v > maxM) v = maxM; CAL.view = v; CAL.mode = "months"; drawCal(); }));
  box.querySelectorAll("[data-m]").forEach((b) => (b.onclick = () => { CAL.view = b.dataset.m; CAL.mode = "days"; drawCal(); }));
  box.querySelectorAll("[data-d]").forEach((b) => (b.onclick = () => {
    const s = b.dataset.d;
    if (CAL.active === "start") { CAL.start = s; if (CAL.end && CAL.end <= s) CAL.end = ""; CAL.active = "end"; }
    else if (s <= CAL.start) { CAL.start = s; CAL.end = ""; }
    else CAL.end = s;
    drawCal(s);
  }));
  const grid = box.querySelector(".cal-grid");
  if (grid) {
    const t = grid.querySelector(`[data-d="${CAL._target}"]`);
    if (t) { t.tabIndex = 0; if (focusDay) t.focus({ preventScroll: true }); }
    grid.addEventListener("keydown", (e) => {
      const cur = document.activeElement && document.activeElement.dataset.d;
      if (!cur) return;
      const mv = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -7, ArrowDown: 7 }[e.key];
      let next = null;
      if (mv) next = addDays(cur, mv);
      else if (e.key === "PageUp" || e.key === "PageDown") { const v = shiftMonth(cur.slice(0, 7), e.key === "PageUp" ? -1 : 1); const [y, m] = v.split("-").map(Number); next = `${v}-${String(Math.min(+cur.slice(8), lastOfMonth(y, m))).padStart(2, "0")}`; }
      if (!next) return;
      e.preventDefault();
      next = clampDay(next);
      CAL.view = next.slice(0, 7);
      drawCal(next);
    });
  }
  box.querySelector("[data-apply]").onclick = () => {
    SIM.start = CAL.start;
    SIM.end = CAL.end;
    SIM.preset = presetOf(SIM.start, SIM.end);
    $("df-start").classList.remove("bad");
    $("df-end").classList.remove("bad");
    $("d-err").hidden = true;
    closeSheet();
    renderSim();
  };
}

function renderSim(keepFocus) {
  document.querySelectorAll("#simform .chips").forEach((box) => {
    const name = box.dataset.name;
    box.innerHTML = "";
    for (const [v, label] of OPTS[name]) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = label;
      b.setAttribute("role", "radio");
      b.setAttribute("aria-checked", String(String(SIM[name]) === String(v)));
      b.onclick = () => {
        if (name === "preset") { if (v === "custom") { SIM.preset = "custom"; renderSim(); $("d-start").focus(); return; } applyPreset(v); $("d-err").hidden = true; $("df-start").classList.remove("bad"); $("df-end").classList.remove("bad"); }
        else SIM[name] = v;
        renderSim();
      };
      box.appendChild(b);
    }
  });
  for (const w of ["start", "end"]) { const inp = $("d-" + w); if (document.activeElement !== inp || keepFocus) inp.value = dot(SIM[w]); }
  $("d-note").textContent = `${spanText(SIM.start, SIM.end)} 동안 돌립니다.`;
  const steppers = { paths: [1, 200, (v) => (v >= 20 ? 10 : 1), (v) => `${v}개`], years: [1, 30, () => 1, (v) => `${v}년`] };
  document.querySelectorAll("#simform .stepper").forEach((box) => {
    const name = box.dataset.name, [lo, hi, step, fmt] = steppers[name];
    box.innerHTML = "";
    const dec = document.createElement("button"), inc = document.createElement("button"), lab = document.createElement("span");
    dec.type = inc.type = "button";
    dec.textContent = "−";
    inc.textContent = "+";
    dec.setAttribute("aria-label", "줄이기");
    inc.setAttribute("aria-label", "늘리기");
    dec.onclick = () => { SIM[name] = Math.max(lo, SIM[name] - step(SIM[name] - 1)); renderSim(); };
    inc.onclick = () => { SIM[name] = Math.min(hi, SIM[name] + step(SIM[name])); renderSim(); };
    lab.textContent = fmt(SIM[name]);
    box.append(dec, lab, inc);
  });
  $("realistic").setAttribute("aria-checked", String(SIM.realistic));
  const hist = SIM.data === "historical";
  document.querySelectorAll("#simform .hist").forEach((x) => (x.hidden = !hist));
  document.querySelectorAll("#simform .syn").forEach((x) => (x.hidden = hist));
  $("set-k").hidden = true;
  $("data-note").textContent = DATA_NOTE[SIM.data];
  $("mode-note").textContent = MODE_NOTE[SIM.mode];
}
$("realistic").onclick = () => { SIM.realistic = !SIM.realistic; renderSim(); };
renderSim();
$("simform").onsubmit = async (ev) => {
  ev.preventDefault();
  if (SIM.data === "historical" && (!commitDate("start") || !commitDate("end"))) return;
  const p = { data: SIM.data, mode: SIM.mode, k: SIM.k, deposit: SIM.deposit, realistic: SIM.realistic };
  if (p.data === "historical") Object.assign(p, { start: SIM.start, end: SIM.end });
  else Object.assign(p, { paths: SIM.paths, years: SIM.years, seed: Math.floor(Math.random() * 100000) });
  const btn = ev.target.querySelector("button[type=submit]");
  btn.disabled = true;
  $("simmsg").textContent = "실행 대기열에 넣는 중입니다.";
  try {
    const r = await api("/api/bt/run", { method: "POST", body: JSON.stringify(p) });
    $("simmsg").textContent = r.id ? "실행했습니다. 끝나면 실행 기록에서 눌러 보세요." : r.error || "실행하지 못했습니다.";
    loadRuns().catch(() => {});
  } catch (e) { $("simmsg").textContent = "실행하지 못했습니다: " + e.message; }
  btn.disabled = false;
};

const VIEWS = ["now", "why", "report", "sim"];
function go(v) { history.replaceState(history.state, "", "#" + v); route(); }
function route() {
  const h = location.hash.slice(1);
  let v = VIEWS.includes(h) ? h : "now", run = null;
  if (h.startsWith("run=")) { v = "sim"; run = decodeURIComponent(h.slice(4)); }
  VIEWS.forEach((x) => ($("v-" + x).hidden = x !== v));
  document.querySelectorAll("#nav button").forEach((b) => (b.dataset.v === v ? b.setAttribute("aria-current", "page") : b.removeAttribute("aria-current")));
  $("sim-home").hidden = !!run;
  $("sim-run").hidden = !run;
  if (run) showRun(run);
  else if (v === "sim") loadRuns().catch(() => {});
  scrollTo(0, 0);
}
document.querySelectorAll("#nav button").forEach((b) => (b.onclick = () => go(b.dataset.v)));
window.addEventListener("hashchange", route);

function render() {
  renderNow();
  renderWhy();
  renderReport();
}

async function load() {
  try {
    S = await api("/api/state");
    $("login").hidden = true;
    $("app").hidden = false;
    $("me").textContent = (S.account || "?").slice(0, 1).toUpperCase();
    render();
  } catch (e) {}
}

$("me").onclick = () => {
  const inApp = !!window.NoSockApp;
  openSheet("계정과 앱", `<dl class="kv"><dt>계정</dt><dd>${esc(S && S.account ? S.account : "-")}</dd><dt>앱</dt><dd>${inApp ? esc(window.NoSockApp.version()) : "웹 브라우저"}</dd><dt>장부 기준일</dt><dd>${S && S.mom_last ? dot(S.mom_last.day) : "-"}</dd></dl>
    <div class="acct-list">${inApp ? '<button class="btn quiet wide" id="a-upd" type="button">업데이트 확인</button>' : ""}<button class="btn quiet wide" id="a-push" type="button">테스트 알림 보내기</button><p class="note" id="a-msg" role="status">알림은 휴대폰의 NoSock 앱으로 갑니다.</p><button class="btn quiet wide" id="a-out" type="button">로그아웃</button></div>`);
  if (inApp) $("a-upd").onclick = () => window.NoSockApp.checkUpdate();
  $("a-push").onclick = async () => {
    try { await api("/api/push/test", { method: "POST", body: "{}" }); if (window.NoSockApp) setTimeout(() => window.NoSockApp.pollNow(), 500); $("a-msg").textContent = window.NoSockApp ? "보냈습니다. 곧 앱 알림이 뜹니다." : "보냈습니다. 휴대폰의 NoSock 앱에 15분 안에 알림이 갑니다."; }
    catch (e) { $("a-msg").textContent = "보내지 못했습니다: " + e.message; }
  };
  $("a-out").onclick = async () => { await fetch(B + "/login", { method: "POST", credentials: "same-origin", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ logout: true }) }); location.replace(location.pathname); };
};

$("loginform").onsubmit = async (ev) => {
  ev.preventDefault();
  $("lmsg").textContent = "확인 중입니다.";
  const r = await fetch(B + "/login", { method: "POST", credentials: "same-origin", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id: $("lid").value.trim(), password: $("lpw").value }) });
  const d = await r.json().catch(() => ({}));
  if (r.ok) { $("lpw").value = ""; $("lmsg").textContent = ""; load().then(freshen); } else $("lmsg").textContent = d.error || "로그인하지 못했습니다.";
};

let freshening = false, freshAt = 0;
async function freshen() {
  if (!S || freshening || Date.now() - freshAt < 10000) return;
  freshAt = Date.now();
  const lastTs = () => (S && S.live && S.live.last_mark ? S.live.last_mark.ts : "");
  const before = lastTs();
  freshening = true;
  renderNow();
  try {
    await api("/api/live/refresh", { method: "POST", body: "{}" });
    for (const ms of [1500, 2000, 3000, 4000]) {
      await new Promise((ok) => setTimeout(ok, ms));
      await load();
      if (lastTs() && lastTs() !== before) break;
    }
  } catch (e) {}
  freshening = false;
  renderNow();
}

route();
load().then(freshen);
setInterval(load, 60000);
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") load().then(freshen); });
