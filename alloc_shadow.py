import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ft"))
import numpy as np
import lightgbm as lgb
import learn_inverse as LI
import harvest_v8 as H8
import msma_v1 as M1
import ms_role_v2 as R
import ms_alloc_v1 as A1

KST = ZoneInfo("Asia/Seoul")
CODES = list(M1.CLASS)
SIZING = "slots2_equity_half"
SLOTS = 2
BUFFER = 0.02
ANCHOR = M1.END
CUT = "2026-01-01"
COUNT = 200
MODEL_DIR = os.path.join(HERE, "ft", "results", "alloc_shadow_model_v1")
META_SHA = "4496210ea1894a3a5fadb48b961506d7daadac8d30c17652d68f9dee4de9f09a"
LEDGER = os.path.join(HERE, "runs", "alloc_shadow_ledger.jsonl")
SNAP_DIR = os.path.join(HERE, "data", "kr_etf", "alloc_shadow_snaps")
RUN_AT = (15, 55)
COMPLETE_AFTER = "15:31:00"
DEPOSIT = R.DEPOSIT
VAULT_SHARE = R.VAULT_SHARE
FEE = LI.FEE
IRC = os.path.join(HERE, "irc.py")


def sha_b(b):
    return hashlib.sha256(b).hexdigest()


def sha_f(p):
    return sha_b(open(p, "rb").read())


def now_kst():
    return datetime.now(KST)


def cost(day, p):
    return FEE + H8.tick(day, p) / p


def append(row, path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")


def last_state(path):
    if not os.path.exists(path):
        return None
    last = None
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("kind") == "alloc_day":
            last = r
    return last


def pinned():
    recs = R.load()
    out = {}
    for c in CODES:
        r = recs[c]
        n = sum(1 for d in r["days"] if d <= ANCHOR)
        out[c] = dict(days=list(r["days"][:n]), **{k: [float(x) for x in r[k][:n]] for k in ("O", "H", "L", "C")})
    return out


def next_gap(day):
    d = date.fromisoformat(day)
    k = 1
    while (d + timedelta(days=k)).weekday() >= 5:
        k += 1
    return float(k)


def features_live(rec, code, last_gap):
    days = rec["days"]
    n = len(days)
    O, H, L, C = (np.asarray(rec[k], float) for k in ("O", "H", "L", "C"))
    r = np.r_[np.nan, np.diff(np.log(C))]
    rs = lambda w: np.array([np.std(r[max(1, i - w + 1):i + 1]) if i >= w else np.nan for i in range(n)])
    v20, v60 = rs(20), rs(60)
    S, _, _ = M1.signals(np.array(days), C[:, None])
    eq = np.nanmean(np.tanh(np.stack([S[:, :, M1.SIG.index(s)] for s in M1.EQ_SIGNALS], axis=2)), axis=2)[:, 0]
    hi = np.array([C[max(0, i - 249):i + 1].max() for i in range(n)])
    lc = np.log(C)
    lag = lambda k: np.r_[[np.nan] * k, lc[k:] - lc[:-k]]
    gaps = np.array([(date.fromisoformat(days[i + 1]) - date.fromisoformat(days[i])).days if i + 1 < n else last_gap for i in range(n)], float)
    dow = np.array([date.fromisoformat(x).weekday() for x in days], float)
    cls = np.zeros((n, len(M1.CLASSES)))
    cls[:, M1.CLASSES.index(M1.CLASS[code])] = 1.0
    F = np.column_stack([lag(1), lag(5), lag(20), lag(60), v20, v60, v20 / v60, C / hi - 1, eq, dow, gaps, cls])
    return F, v20


def build():
    recs = R.load()
    inst = {c: R.instrument(recs[c], c) for c in CODES}
    samp = A1.samples(inst)
    X, Y = [], []
    for c, ins in inst.items():
        for t, idx, Rr, ex in samp[c]:
            if ins["days"][t] < CUT and ex.max() < len(ins["days"]) and ins["days"][int(ex.max())] < CUT and not np.isnan(Rr).any():
                X.append(R.rows(ins["F"][t], R.GRID[idx], ins["v20"][t]))
                Y.append(R.reward(Rr, R.KAPPA))
    m = lgb.train(LI.PARAMS, lgb.Dataset(np.vstack(X), label=np.concatenate(Y)), num_boost_round=LI.ROUNDS)
    ms = m.model_to_string().encode("utf-8")
    check = {}
    for c, ins in inst.items():
        t = ins["days"].index(ANCHOR)
        p = m.predict(R.rows(ins["F"][t], R.GRID, ins["v20"][t]))
        a = int(np.argmax(p))
        check[c] = dict(a=a, v=float(p[a]))
    os.makedirs(MODEL_DIR, exist_ok=False)
    mf = os.path.join(MODEL_DIR, "model.txt")
    with open(mf, "xb") as f:
        f.write(ms)
    files = {k: f"ft/{k}.py" for k in ("ms_role_v2", "ms_alloc_v1", "msma_v1", "learn_inverse", "harvest_v8")}
    meta = dict(version="alloc-shadow-model-v1", cut=CUT, rows=int(sum(len(y) for y in Y)), model=dict(file="model.txt", sha256=sha_b(ms)), etf_manifest_sha256=sha_f(os.path.join(R.ETF_DIR, "manifest.json")), code_sha256={k: sha_f(os.path.join(HERE, v)) for k, v in files.items()}, anchor=ANCHOR, anchor_check=check, policy=dict(sizing=SIZING, buffer=BUFFER, kappa=R.KAPPA, vault_share=VAULT_SHARE, deposit=DEPOSIT, grid=len(R.GRID)), spec="ft/results/ms_alloc_spec_v2.json")
    mb = json.dumps(meta, ensure_ascii=False, indent=1).encode("utf-8")
    with open(os.path.join(MODEL_DIR, "meta.json"), "xb") as f:
        f.write(mb)
    print("meta sha256", sha_b(mb))


def load_model():
    mb = open(os.path.join(MODEL_DIR, "meta.json"), "rb").read()
    if sha_b(mb) != META_SHA:
        raise SystemExit("model meta hash differs")
    meta = json.loads(mb)
    b = open(os.path.join(MODEL_DIR, meta["model"]["file"]), "rb").read()
    if sha_b(b) != meta["model"]["sha256"]:
        raise SystemExit("model file hash differs")
    return lgb.Booster(model_str=b.decode("utf-8")), meta


def read_only_client():
    import toss
    t = toss.Toss()
    if not t.tok or t.exp < time.time() + 120:
        raise RuntimeError("cached token missing or expiring; not issuing")

    def refuse():
        raise RuntimeError("token issue refused")

    t._issue = refuse
    return t


def bar_complete(day, fetched):
    return day < fetched[:10] or (day == fetched[:10] and fetched[11:19] >= COMPLETE_AFTER)


def fetch_all(client_factory):
    t = client_factory()
    snap = {}
    for c in CODES:
        snap[c] = {}
        for adj in (False, True):
            r = t.candles(c, "1d", count=COUNT, adjusted=adj)
            snap[c]["adj" if adj else "raw"] = r["candles"]
            time.sleep(0.2)
    return snap, now_kst().isoformat(timespec="milliseconds")


def to_rows(candles, fetched):
    out = {}
    for x in candles:
        d = x["timestamp"][:10]
        if bar_complete(d, fetched):
            out[d] = (float(x["openPrice"]), float(x["highPrice"]), float(x["lowPrice"]), float(x["closePrice"]))
    return out


def splice(pin, adj_rows):
    if ANCHOR not in adj_rows:
        raise RuntimeError("anchor day missing from the adjusted fetch")
    s = pin["C"][-1] / adj_rows[ANCHOR][3]
    rec = {k: list(v) for k, v in pin.items()}
    for d in sorted(x for x in adj_rows if x > ANCHOR):
        o, h, l, c = adj_rows[d]
        rec["days"].append(d)
        for k, v in zip("OHLC", (o, h, l, c)):
            rec[k].append(v * s)
    return rec


def new_state():
    return dict(pool=0.0, vault=0.0, deposits=0.0, realized=0.0, hwm=0.0, trades=0, wins=0, losses=0, open={}, pending={})


def settle_bar(st, c, bar, D):
    events = []
    o_, h_, l_, c_ = bar
    if c in st["pending"]:
        od = st["pending"].pop(c)
        if od["d"] == 0:
            pin = o_
        elif l_ > od["limit"]:
            st["pool"] += od["reserve"]
            events.append(dict(code=c, kind="unfilled", limit=od["limit"]))
            pin = None
        else:
            pin = o_ if o_ < od["limit"] else od["limit"]
        if pin is not None:
            cin = cost(D, pin)
            sh = od["shares"]
            avail = st["pool"] + od["reserve"]
            while sh > 0 and sh * pin * (1 + cin) > avail:
                sh -= 1
            if sh == 0:
                st["pool"] = avail
                events.append(dict(code=c, kind="cancel_gap", open=o_))
            else:
                paid = sh * pin * (1 + cin)
                st["pool"] = avail - paid
                st["open"][c] = dict(shares=sh, pin=pin, paid=paid, d=od["d"], tp=od["tp"], sl=od["sl"], hz=od["hz"], up=pin * (1 + od["tp"] + 2 * cin) if od["tp"] > 0 else None, dn=pin * (1 - od["sl"]) if od["sl"] > 0 else None, first_day=D, sessions=0)
                st["trades"] += 1
                events.append(dict(code=c, kind="buy", shares=sh, price=pin, paid=paid))
    if c in st["open"]:
        p = st["open"][c]
        p["sessions"] += 1
        on_fill = p["first_day"] == D
        first = (not on_fill) or p["d"] == 0
        px, why = None, None
        if p["dn"] is not None and l_ <= p["dn"]:
            px, why = (min(p["dn"], o_) if (not on_fill and o_ < p["dn"]) else p["dn"]), "stop"
        elif p["up"] is not None and h_ >= p["up"] and first:
            px, why = (max(p["up"], o_) if (not on_fill and o_ > p["up"]) else p["up"]), "take_profit"
        elif p["sessions"] >= p["hz"]:
            px, why = c_, "cycle_end"
        if px is not None:
            proceeds = p["shares"] * px * (1 - cost(D, px))
            prof = proceeds - p["paid"]
            st["realized"] += prof
            v = VAULT_SHARE * (st["realized"] - st["hwm"]) if st["realized"] > st["hwm"] else 0.0
            st["hwm"] = max(st["hwm"], st["realized"])
            st["vault"] += v
            st["pool"] += proceeds - v
            st["wins" if prof > 0 else "losses"] += 1
            del st["open"][c]
            events.append(dict(code=c, kind="sell", why=why, shares=p["shares"], price=px, proceeds=proceeds, profit=prof, to_vault=v))
    return events


def decide(st, D, model, feats, closes):
    ranked = []
    for c in CODES:
        if c in st["open"] or c in st["pending"] or c not in feats:
            continue
        Fv, v20 = feats[c]
        if np.isnan(Fv).any() or not v20 > 0:
            continue
        p = model.predict(R.rows(Fv, R.GRID, v20))
        a = int(np.argmax(p))
        ranked.append((float(p[a]), c, a))
    ranked.sort(key=lambda z: -z[0])
    equity = st["pool"] + sum(x["paid"] for x in st["open"].values()) + sum(x["reserve"] for x in st["pending"].values())
    orders, skipped = [], []
    for v, c, a in ranked:
        if v <= 0 or len(st["open"]) + len(st["pending"]) >= SLOTS:
            break
        if c not in closes:
            skipped.append(c)
            continue
        d, tp, sl, hz = (float(z) for z in R.GRID[a])
        rc = closes[c]
        ref = rc * (1 + BUFFER) if d == 0 else rc * (1 - d)
        budget = min(st["pool"], equity / SLOTS)
        sh = math.floor(budget / ref)
        while sh > 0 and sh * ref * (1 + cost(D, ref)) > st["pool"]:
            sh -= 1
        if sh < 1:
            skipped.append(c)
            continue
        reserve = sh * ref * (1 + cost(D, ref))
        st["pool"] -= reserve
        od = dict(a=a, v=v, d=d, tp=tp, sl=sl, hz=int(hz), shares=sh, ref=ref, limit=rc * (1 - d) if d > 0 else None, reserve=reserve, placed_day=D)
        st["pending"][c] = od
        orders.append(dict(code=c, **od))
    return ranked[:6], orders, skipped


def mark(st, closes):
    return st["pool"] + sum(od["reserve"] for od in st["pending"].values()) + sum(p["shares"] * closes.get(c, p["pin"]) for c, p in st["open"].items())


def process(today, snap, fetched, model, prev, ledger, snap_info, pins):
    raw = {c: to_rows(snap[c]["raw"], fetched) for c in CODES}
    adj = {c: to_rows(snap[c]["adj"], fetched) for c in CODES}
    st = json.loads(json.dumps(prev["state"])) if prev else new_state()
    start = prev["day"] if prev else None
    days = sorted(set(d for c in CODES for d in raw[c] if (start is None and d == today) or (start is not None and start < d <= today)))
    if not days:
        row = dict(kind="alloc_skip", day=today, reason="no completed bar for today")
        append(row, ledger)
        return [row]
    out = []
    recs = {c: splice(pins[c], adj[c]) for c in CODES}
    for D in days:
        ev = []
        for c in CODES:
            if D in raw[c]:
                ev += settle_bar(st, c, raw[c][D], D)
        st["pool"] += DEPOSIT
        st["deposits"] += DEPOSIT
        closes = {c: raw[c][D][3] for c in CODES if D in raw[c]}
        forward = D == today
        top, orders, skipped = [], [], []
        if forward:
            feats = {}
            for c in CODES:
                rc = recs[c]
                if D not in rc["days"]:
                    continue
                i = rc["days"].index(D)
                sub = {k: v[:i + 1] for k, v in rc.items()}
                F, v20 = features_live(sub, c, next_gap(D))
                feats[c] = (F[-1], v20[-1])
            top, orders, skipped = decide(st, D, model, feats, closes)
        eqv = mark(st, closes)
        row = dict(kind="alloc_day", policy=SIZING, day=D, decision_kind="forward" if forward else "backfill_no_trade", events=ev, top=[dict(code=c, v=v, action=[float(z) for z in R.GRID[a]]) for v, c, a in top], orders=orders, skipped_for_price=skipped, account_marked=eqv, vault=st["vault"], deposits=st["deposits"], total=eqv + st["vault"], state=st, snap=snap_info)
        out.append(row)
        st = json.loads(json.dumps(st))
    for r in out:
        append(r, ledger)
    return out


def run_once(today=None, ledger=LEDGER, client_factory=read_only_client, snap_dir=SNAP_DIR, quiet=False, quiet_skip=False):
    today = today or now_kst().strftime("%Y-%m-%d")
    prev = last_state(ledger)
    if prev and prev["day"] >= today:
        return []
    model, meta = load_model()
    pins = pinned()
    try:
        snap, fetched = fetch_all(client_factory)
    except Exception as e:
        row = dict(kind="alloc_skip", day=today, reason="fetch failed: " + type(e).__name__)
        append(row, ledger)
        if not quiet and not quiet_skip:
            subprocess.run([sys.executable, "-X", "utf8", IRC, "say", message(row, {})], cwd=HERE, timeout=60)
        return [row]
    os.makedirs(snap_dir, exist_ok=True)
    sb = json.dumps(dict(fetched_at=fetched, candles=snap), ensure_ascii=False).encode("utf-8")
    sp = os.path.join(snap_dir, f"{today}_{fetched[11:19].replace(':', '')}.json")
    with open(sp, "xb") as f:
        f.write(sb)
    info = dict(model_meta_sha256=META_SHA, snapshot=(os.path.relpath(sp, HERE) if os.path.splitdrive(sp)[0].lower() == os.path.splitdrive(HERE)[0].lower() else sp).replace("\\", "/"), snapshot_sha256=sha_b(sb), fetched_at=fetched)
    rows = process(today, snap, fetched, model, prev, ledger, info, pins)
    if not quiet:
        names = json.load(open(os.path.join(R.ETF_DIR, "manifest.json"), encoding="utf-8"))["names"]
        for r in rows:
            if r["kind"] == "alloc_skip" and quiet_skip:
                continue
            subprocess.run([sys.executable, "-X", "utf8", IRC, "say", message(r, names)], cwd=HERE, timeout=60)
    return rows


def message(r, names):
    if r["kind"] != "alloc_day":
        return f"[AI배분 모의 {r['day']}] 건너뜀: {r['reason']}"
    w = lambda x: f"{x:,.0f}원"
    parts = []
    for e in r["events"]:
        nm = names[e["code"]][:12]
        if e["kind"] == "buy":
            parts.append(f"매수 {nm} {e['shares']}주 @{e['price']:,.0f}")
        elif e["kind"] == "sell":
            parts.append(f"매도 {nm} {e['shares']}주 @{e['price']:,.0f} ({ {'stop': '손절', 'take_profit': '익절', 'cycle_end': '기간만료'}[e['why']] }, 손익 {e['profit']:+,.0f}원, 금고 +{e['to_vault']:,.0f})")
        elif e["kind"] == "unfilled":
            parts.append(f"미체결 {nm}")
        else:
            parts.append(f"취소 {nm}")
    od = [f"{names[o['code']][:12]} {o['shares']}주 {'시가' if o['d'] == 0 else '지정가 ' + format(o['limit'], ',.0f')} 익절 {'없음' if o['tp'] == 0 else '+' + format(100 * o['tp'], 'g') + '%'} 손절 {'없음' if o['sl'] == 0 else '-' + format(100 * o['sl'], 'g') + '%'} 최대 {o['hz']}일" for o in r["orders"]]
    tag = " 소급기록·주문 없음" if r["decision_kind"] != "forward" else ""
    return f"[AI배분 모의 {r['day']}{tag}] 입금 누계 {w(r['deposits'])} → 금고 {w(r['vault'])} + 계좌 {w(r['account_marked'])} = {w(r['total'])}. 오늘: {', '.join(parts) or '체결 없음'}. 다음 주문: {'; '.join(od) or '없음'}" + (f" (1주 가격 초과로 건너뜀: {', '.join(names[c][:10] for c in r['skipped_for_price'])})" if r["skipped_for_price"] else "")


def next_run(now):
    t = now.replace(hour=RUN_AT[0], minute=RUN_AT[1], second=0, microsecond=0)
    if now >= t:
        t += timedelta(days=1)
    while t.weekday() >= 5:
        t += timedelta(days=1)
    return t


def loop():
    attempts = {}
    while True:
        now = now_kst()
        today = now.strftime("%Y-%m-%d")
        prev = last_state(LEDGER)
        due = now.weekday() < 5 and (now.hour, now.minute) >= RUN_AT and not (prev and prev["day"] >= today) and attempts.get(today, 0) < 3
        if due:
            attempts[today] = attempts.get(today, 0) + 1
            try:
                run_once(today, quiet_skip=attempts[today] > 1)
            except Exception as e:
                append(dict(kind="alloc_error", day=today, error=type(e).__name__), LEDGER)
        t = next_run(now_kst())
        time.sleep(max(30.0, min(3600.0, (t - now_kst()).total_seconds())))


def selftest():
    import tempfile
    tmp = tempfile.mkdtemp(prefix="TEST_FIXTURE_alloc_")
    led = os.path.join(tmp, "ledger.jsonl")
    checks = []
    recs = R.load()
    pins = pinned()
    model, meta = load_model()
    ok = True
    for c in CODES:
        ins = R.instrument(recs[c], c)
        F, v20 = features_live(pins[c], c, 1.0)
        same = np.array_equal(np.nan_to_num(F, nan=-9e9), np.nan_to_num(ins["F"], nan=-9e9)) and np.array_equal(np.nan_to_num(v20, nan=-9e9), np.nan_to_num(ins["v20"], nan=-9e9))
        p = model.predict(R.rows(F[-1], R.GRID, v20[-1]))
        a = int(np.argmax(p))
        ok = ok and same and a == meta["anchor_check"][c]["a"] and abs(float(p[a]) - meta["anchor_check"][c]["v"]) < 1e-12
    checks.append(("live features equal the backtest features and the frozen model reproduces the anchor decisions", ok))
    raw_man = json.load(open(os.path.join(R.ETF_DIR, "manifest.json"), encoding="utf-8"))
    rawrec = {c: json.load(open(os.path.join(R.ETF_DIR, raw_man["files"][f"{c}_raw"]["file"]), encoding="utf-8")) for c in CODES}
    rawrec = {c: (v if v["days"] else recs[c]) for c, v in rawrec.items()}

    class Fake:
        def __init__(self, upto):
            self.upto = upto

        def candles(self, sym, interval, count=200, before=None, adjusted=False):
            src = recs[sym] if adjusted else rawrec[sym]
            ds = [i for i, d in enumerate(src["days"]) if d <= self.upto][-count:]
            return {"candles": [{"timestamp": src["days"][i] + "T00:00:00+09:00", "openPrice": str(src["O"][i]), "highPrice": str(src["H"][i]), "lowPrice": str(src["L"][i]), "closePrice": str(src["C"][i])} for i in ds]}

    r1 = run_once("2026-11-06", led, lambda: Fake("2026-11-06"), os.path.join(tmp, "snap"), quiet=True)
    checks.append(("first run records one forward day with one deposit", len(r1) == 1 and r1[0]["day"] == "2026-11-06" and r1[0]["decision_kind"] == "forward" and r1[0]["deposits"] == DEPOSIT and not r1[0]["events"]))
    checks.append(("first run reserves at most the pool", r1[0]["state"]["pool"] >= -1e-9 and abs(r1[0]["account_marked"] - DEPOSIT) < 1e-9))
    r2 = run_once("2026-11-06", led, lambda: Fake("2026-11-06"), os.path.join(tmp, "snap"), quiet=True)
    checks.append(("same day is idempotent", r2 == []))
    r3 = run_once("2026-11-09", led, lambda: Fake("2026-11-06"), os.path.join(tmp, "snap"), quiet=True)
    checks.append(("no completed bar for today is a skip", len(r3) == 1 and r3[0]["kind"] == "alloc_skip"))
    st = new_state()
    st["pool"] = 100000.0
    st["pending"]["X"] = dict(d=0.0, tp=0.02, sl=0.04, hz=5, shares=9, limit=None, reserve=9 * 10200.0 * (1 + cost("2026-11-10", 10200.0)))
    st["pool"] -= st["pending"]["X"]["reserve"]
    D = "2026-11-10"
    e1 = settle_bar(st, "X", (10000.0, 10000.0 * (1 + 0.02 + 2 * cost(D, 10000.0)) + 1, 9900.0, 10050.0), D)
    pin, cin = 10000.0, cost(D, 10000.0)
    up = pin * (1 + 0.02 + 2 * cin)
    proceeds = 9 * up * (1 - cost(D, up))
    prof = proceeds - 9 * pin * (1 + cin)
    checks.append(("open buy can take profit on the fill day and half the profit goes to the vault", [x["kind"] for x in e1] == ["buy", "sell"] and e1[1]["why"] == "take_profit" and abs(st["vault"] - 0.5 * prof) < 1e-9 and abs(st["pool"] + st["vault"] - (100000.0 + prof)) < 1e-6))
    st2 = new_state()
    st2["pending"]["Y"] = dict(d=0.03, tp=0.05, sl=0.0, hz=20, shares=3, limit=9700.0, reserve=3 * 9700.0 * (1 + cost(D, 9700.0)))
    e2 = settle_bar(st2, "Y", (9900.0, 9950.0, 9750.0, 9800.0), D)
    checks.append(("dip order not touched returns its reserve", [x["kind"] for x in e2] == ["unfilled"] and abs(st2["pool"] - 3 * 9700.0 * (1 + cost(D, 9700.0))) < 1e-9 and not st2["open"]))
    st3 = new_state()
    st3["pending"]["Z"] = dict(d=0.03, tp=0.01, sl=0.04, hz=20, shares=3, limit=9700.0, reserve=3 * 9700.0 * (1 + cost(D, 9700.0)))
    e3 = settle_bar(st3, "Z", (9900.0, 11000.0, 9650.0, 9800.0), D)
    checks.append(("dip fill day: no take profit, position stays open", [x["kind"] for x in e3] == ["buy"] and "Z" in st3["open"] and st3["open"]["Z"]["pin"] == 9700.0))
    e4 = settle_bar(st3, "Z", (9000.0, 9100.0, 8900.0, 9000.0), "2026-11-11")
    checks.append(("stop on a gap below fills at the open and the vault is unchanged", e4[0]["why"] == "stop" and e4[0]["price"] == 9000.0 and st3["vault"] == 0.0 and st3["losses"] == 1))
    lost = -e4[0]["profit"]
    st3["pending"]["W"] = dict(d=0.0, tp=0.0, sl=0.0, hz=1, shares=10, limit=None, reserve=0.0)
    st3["pool"] += 10 * 20000.0
    e5 = settle_bar(st3, "W", (10000.0, 10100.0, 9990.0, 10000.0 * (1 + cost("2026-11-12", 10000.0)) * (1 + cost("2026-11-12", 11000.0)) + 1000.0), "2026-11-12")
    gain = e5[1]["profit"]
    checks.append(("after a loss, only net profit above the previous high goes to the vault (50%)", gain > lost and abs(e5[1]["to_vault"] - 0.5 * (gain - lost)) < 1e-6 and abs(st3["vault"] - 0.5 * (gain - lost)) < 1e-6))
    st4 = dict(st3)
    st4["pending"] = {"V": dict(d=0.0, tp=0.0, sl=0.0, hz=1, shares=1, limit=None, reserve=0.0)}
    st4["pool"] += 20000.0
    e6 = settle_bar(st4, "V", (10000.0, 10000.0, 9000.0, 9000.0), "2026-11-13")
    checks.append(("a later loss leaves the vault alone and is earned back before the next payment", e6[1]["to_vault"] == 0.0 and st4["vault"] == st3["vault"] and st4["realized"] < st4["hwm"]))
    rng = np.random.default_rng(7)
    eq_ok, n_eq = True, 0
    days = [str(date(2026, 3, 2) + timedelta(days=i)) for i in range(60)]
    Cs = 10000.0 * np.exp(np.cumsum(rng.normal(0, 0.012, 60)))
    Os = Cs * np.exp(rng.normal(0, 0.004, 60))
    Hs = np.maximum(Os, Cs) * np.exp(np.abs(rng.normal(0, 0.006, 60)))
    Ls = np.minimum(Os, Cs) * np.exp(-np.abs(rng.normal(0, 0.006, 60)))
    ins = dict(days=days, O=Os, H=Hs, L=Ls, C=Cs)
    for a in rng.choice(len(R.GRID), size=120, replace=False):
        t = 10
        d, tp, sl, hz = (float(z) for z in R.GRID[a])
        Rb, exb = R.outcome(ins, t, R.GRID[a])
        s = new_state()
        lim = Cs[t] * (1 - d) if d > 0 else None
        s["pending"]["Q"] = dict(d=d, tp=tp, sl=sl, hz=int(hz), shares=1, limit=lim, reserve=1e9)
        s["pool"] = 0.0
        got = None
        for j in range(t + 1, 60):
            ev = settle_bar(s, "Q", (Os[j], Hs[j], Ls[j], Cs[j]), days[j])
            for e in ev:
                if e["kind"] == "unfilled":
                    got = (0.0, j)
                if e["kind"] == "sell":
                    got = (math.log(e["proceeds"] / (e["proceeds"] - e["profit"])), j)
            if got:
                break
        n_eq += 1
        eq_ok = eq_ok and got is not None and abs(got[0] - Rb) < 1e-12 and got[1] == exb
    checks.append((f"shadow fills reproduce the backtest outcome for {n_eq} random actions", eq_ok))
    stx = new_state()
    stx["pool"] = 5_000_000.0
    fx = {}
    for c in CODES:
        F, v20 = features_live(pins[c], c, 1.0)
        fx[c] = (F[-1], v20[-1])
    closes_x = {c: pins[c]["C"][-1] for c in CODES}
    topx, ordx, skx = decide(stx, ANCHOR, model, fx, closes_x)
    best2 = [c for c in sorted(CODES, key=lambda c: -meta["anchor_check"][c]["v"]) if meta["anchor_check"][c]["v"] > 0][:2]
    within = all(o["shares"] * o["ref"] <= 2_500_000.0 + 1e-6 for o in ordx)
    checks.append(("two slots: the two best positive instruments get half the equity each", sorted(o["code"] for o in ordx) == sorted(best2) and not skx and within and abs(stx["pool"] + sum(o["reserve"] for o in ordx) - 5_000_000.0) < 1e-6))
    msg = message(r1[0], raw_man["names"])
    checks.append(("message renders", msg.startswith("[AI배분 모의 2026-11-06]")))
    for name, okk in checks:
        print("PASS" if okk else "FAIL", name)
    print("fixture", tmp)
    print("first-day orders", [(o["code"], o["shares"], o["d"], o["tp"], o["sl"], o["hz"]) for o in r1[0]["orders"]], "top", [(x["code"], round(x["v"], 4)) for x in r1[0]["top"]])
    return all(ok for _, ok in checks)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        build()
    elif cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    elif cmd == "once":
        run_once()
    elif cmd == "loop":
        loop()
    else:
        raise SystemExit("usage: build | selftest | once | loop")
