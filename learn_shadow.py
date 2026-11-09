import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ft"))
import numpy as np
import lightgbm as lgb
import learn_inverse as LI
import compare_trend_v2 as CT

KST = ZoneInfo("Asia/Seoul")
L, S = "069500", "114800"
MODEL_DIR = os.path.join(HERE, "ft", "results", "learn_shadow_models_v1")
META_SHA = "3beea6ed9456fa663dffdccf2f1da0e70a3435fa2ad2dcd569d4c3d773e11467"
LIVE_RAW = os.path.join(HERE, "data", "kr_etf", "live_daily_raw.json")
LIVE_ADJ = os.path.join(HERE, "data", "kr_etf", "live_daily_adj.json")
SNAP_DIR = os.path.join(HERE, "data", "kr_etf", "shadow_114800")
LEDGER = os.path.join(HERE, "runs", "learn_shadow_ledger.jsonl")
RUN_AT = (15, 50)
CAPITAL = 100_000_000.0
FEE, TICK = LI.FEE, LI.TICK
BOOKS = ("hold", "ribbon", "learn_long", "learn_inv", "kfft_W128_K3", "vguard_L2")
KFFT = (128, 3, 0.01)
LEV = "122630"
LEV_SNAP_DIR = os.path.join(HERE, "data", "kr_etf", "shadow_122630")
HARVEST = {"harvest_A": False, "harvest_C": True}
H_TP, H_SL, VAULT_SHARE = 0.10, 0.10, 0.5
SKIM = {"skim_R1": 0.10}
CPPI = {"floor95_m3": (0.95, 3), "floor90_m3": (0.90, 3)}
CPPI_BAND, CPPI_G = 0.05, 0.10
IRC = os.path.join(HERE, "irc.py")


def sha_b(b):
    return hashlib.sha256(b).hexdigest()


def now_kst():
    return datetime.now(KST)


def load_models():
    mb = open(os.path.join(MODEL_DIR, "meta.json"), "rb").read()
    if sha_b(mb) != META_SHA:
        raise SystemExit("model meta hash differs")
    meta = json.loads(mb)
    out = {}
    for name in ("long", "inv"):
        b = open(os.path.join(MODEL_DIR, meta["files"][name]["file"]), "rb").read()
        if sha_b(b) != meta["files"][name]["sha256"]:
            raise SystemExit(f"model file hash differs: {name}")
        out[name] = lgb.Booster(model_str=b.decode("utf-8"))
    return out, meta


def load_series(raw_path, adj_path):
    raw = json.load(open(raw_path, encoding="utf-8"))
    adj = json.load(open(adj_path, encoding="utf-8"))
    if raw["days"] != adj["days"]:
        raise SystemExit("raw/adjusted calendars differ")
    return raw, adj


def features_at(adjC, i):
    F = LI.features([None] * len(adjC), dict(Ca=[float(x) for x in adjC]))
    return F[i]


def ribbon_at(adjC, i):
    C = [float(x) for x in adjC[:i + 1]]
    e = {n: CT.ema(C, n)[-1] for n in (8, 21, 50, 200)}
    return L if e[8] > e[21] > e[50] > e[200] else None


def kfft_at(rawC, i):
    import kalman_fft_v1 as K1
    y = np.log(np.asarray(rawC[:i + 1], float))
    lev, _ = K1.kalman(y, np.zeros(len(y), dtype=int), KFFT[2])
    f = K1.fft_features(lev, np.arange(len(y)), KFFT[0], KFFT[1])
    v = f[i, 5]
    return L if (not np.isnan(v) and v > 0) else None


def vguard_at(adjC, i):
    import vswing_v1 as VS
    f = VS.features(np.asarray(adjC[:i + 1], float))
    w, _ = VS.targets(f, "L2_trend_crashguard")
    return L if w[-1] > 0 else None


def policy_step(state, pl, ps, allow_inv):
    v = {None: 0.0, L: pl}
    if allow_inv:
        v[S] = ps
    best = max(v, key=lambda a: v[a])
    if state in v and v[state] >= v[best] - LI.BAND:
        best = state
    return best


def eq_at(adjC, i):
    import msma_v1 as M1
    A = np.asarray(adjC[:i + 1], float)
    Sg, _, _ = M1.signals(np.arange(len(A)), A[:, None])
    v = np.nanmean(np.tanh(np.stack([Sg[:, :, M1.SIG.index(x)] for x in M1.EQ_SIGNALS], axis=2)), axis=2)[-1, 0]
    return float(v) if np.isfinite(v) else None


def fetch_daily(client_factory, sym):
    t = client_factory()
    r = t.candles(sym, "1d", count=30)
    fetched = now_kst().isoformat(timespec="milliseconds")
    rows = {c["timestamp"][:10]: (float(c["openPrice"]), float(c["closePrice"])) for c in r["candles"]}
    return rows, r["candles"], fetched


def fetch_inverse(client_factory):
    t = client_factory()
    r = t.candles(S, "1d", count=30)
    fetched = now_kst().isoformat(timespec="milliseconds")
    rows = {c["timestamp"][:10]: (float(c["openPrice"]), float(c["closePrice"])) for c in r["candles"]}
    return rows, r["candles"], fetched


def read_only_client():
    import toss
    t = toss.Toss()
    if not t.tok or t.exp < time.time() + 120:
        raise RuntimeError("cached token missing or expiring; not issuing")

    def refuse():
        raise RuntimeError("token issue refused")

    t._issue = refuse
    return t


def last_state(path):
    if not os.path.exists(path):
        return None
    last = None
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("kind") == "shadow_day":
            last = r
    return last


def append(row, path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def settle(book, target, o_long, o_inv):
    cash, ql, qi = book["cash"], book["q_long"], book["q_inv"]
    fills = []
    if ql > 0 and target != L:
        px = o_long - TICK
        cash += ql * px * (1 - FEE)
        fills.append(dict(side="sell", sym=L, qty=ql, px=px))
        ql = 0.0
    if qi > 0 and target != S:
        px = o_inv - TICK
        cash += qi * px * (1 - FEE)
        fills.append(dict(side="sell", sym=S, qty=qi, px=px))
        qi = 0.0
    if target == L and ql == 0 and cash > 0:
        px = o_long + TICK
        ql = cash / (px * (1 + FEE))
        fills.append(dict(side="buy", sym=L, qty=ql, px=px))
        cash = 0.0
    if target == S and qi == 0 and cash > 0:
        px = o_inv + TICK
        qi = cash / (px * (1 + FEE))
        fills.append(dict(side="buy", sym=S, qty=qi, px=px))
        cash = 0.0
    return dict(book, cash=cash, q_long=ql, q_inv=qi), fills


def harvest_settle(book, o_long, o_inv):
    b = dict(book)
    fills = []
    act = b.get("pending")
    if act == "sell" and b["leg"]:
        o = o_long if b["leg"] == L else o_inv
        q = b["q_long"] if b["leg"] == L else b["q_inv"]
        px = o - TICK
        proceeds = q * px * (1 - FEE)
        profit = proceeds - b["basis"]
        tv = VAULT_SHARE * profit if profit > 0 else 0.0
        b["vault"] += tv
        b["cash"] += proceeds - tv
        fills.append(dict(side="sell", sym=b["leg"], qty=q, px=px, profit=profit, to_vault=tv))
        b.update(q_long=0.0, q_inv=0.0, leg=None, basis=0.0, entry=None)
    elif act in (L, S) and b["leg"] is None and b["cash"] > 0:
        o = o_long if act == L else o_inv
        px = o + TICK
        q = b["cash"] / (px * (1 + FEE))
        b.update(basis=b["cash"], entry=b["cash"] / q, leg=act, cash=0.0)
        b["q_long" if act == L else "q_inv"] = q
        fills.append(dict(side="buy", sym=act, qty=q, px=px))
    b["pending"] = None
    return b, fills


def harvest_decide(book, allow_inv, c_long, c_inv, pl, ps):
    if book["leg"]:
        c = c_long if book["leg"] == L else c_inv
        return "sell" if c >= book["entry"] * (1 + H_TP) or c <= book["entry"] * (1 - H_SL) else None
    cands = [(0.0, None), (pl, L)] + ([(ps, S)] if allow_inv else [])
    best = max(cands, key=lambda x: x[0])
    return best[1] if best[0] > LI.BAND else None


def skim_settle(book, o_long):
    b = dict(book)
    fills = []
    act = b.get("pending")
    if act == "buy" and b["q_long"] == 0 and b["cash"] > 0:
        px = o_long + TICK
        q = b["cash"] / (px * (1 + FEE))
        b.update(q_long=q, basis=b["cash"], cash=0.0)
        fills.append(dict(side="buy", sym=L, qty=q, px=px))
    elif isinstance(act, list) and act[0] == "skim" and b["q_long"] > 0:
        px = o_long - TICK
        sq = act[1] * b["q_long"]
        proceeds = sq * px * (1 - FEE)
        b["vault"] += proceeds
        b["q_long"] -= sq
        b["basis"] = b["q_long"] * o_long
        fills.append(dict(side="skim", sym=L, qty=sq, px=px, to_vault=proceeds))
    b["pending"] = None
    return b, fills


def skim_decide(book, g, c_long):
    if book["q_long"] == 0:
        return "buy" if book["cash"] > 0 else None
    v = book["q_long"] * c_long
    p = v - book["basis"]
    return ["skim", VAULT_SHARE * p / v] if p >= g * book["basis"] else None


def cppi_settle(book, o_long):
    b = dict(book)
    fills = []
    act = b.get("pending")
    if isinstance(act, list) and act[0] == "skim" and b["q_long"] > 0:
        px = o_long - TICK
        sq = act[1] * b["q_long"]
        proceeds = sq * px * (1 - FEE)
        b["vault"] += proceeds
        b["q_long"] -= sq
        b["basis"] = b["q_long"] * o_long
        fills.append(dict(side="skim", sym=L, qty=sq, px=px, to_vault=proceeds))
    elif isinstance(act, list) and act[0] == "trade" and act[1] > 0 and b["cash"] > 0:
        px = o_long + TICK
        spend = min(b["cash"], act[1] * px * (1 + FEE))
        bq = spend / (px * (1 + FEE))
        b["q_long"] += bq
        b["cash"] -= spend
        b["basis"] += spend
        fills.append(dict(side="buy", sym=L, qty=bq, px=px))
    elif isinstance(act, list) and act[0] == "trade" and act[1] < 0 and b["q_long"] > 0:
        px = o_long - TICK
        sq = min(b["q_long"], -act[1])
        proceeds = sq * px * (1 - FEE)
        part = b["basis"] * sq / b["q_long"]
        profit = proceeds - part
        tv = VAULT_SHARE * profit if profit > 0 else 0.0
        b["vault"] += tv
        b["cash"] += proceeds - tv
        b["basis"] -= part
        b["q_long"] -= sq
        fills.append(dict(side="sell", sym=L, qty=sq, px=px, profit=profit, to_vault=tv))
    b["pending"] = None
    return b, fills


def cppi_decide(book, floor, m, c_long):
    risky = book["q_long"] * c_long
    if book["q_long"] > 0 and risky - book["basis"] >= CPPI_G * book["basis"]:
        return ["skim", VAULT_SHARE * (risky - book["basis"]) / risky]
    acc = book["cash"] + risky
    W = acc + book["vault"]
    target = min(acc, m * max(0.0, W - floor * CAPITAL))
    if abs(target - risky) > CPPI_BAND * W or (target == 0.0 and book["q_long"] > 0):
        dq = target / c_long - book["q_long"]
        if dq != 0:
            return ["trade", dq]
    return None


def process(today, raw, adj, inv_rows, models, prev, ledger, snap_info, lev_rows=None):
    lev_rows = lev_rows or {}
    days = raw["days"]
    if not days or days[-1] < today:
        return [dict(kind="shadow_skip", day=today, reason="069500 close for today not in the loop series", series_last=days[-1] if days else None)]
    start = days.index(prev["day"]) + 1 if prev else days.index(today)
    out = []
    state = prev
    for i in range(start, days.index(today) + 1):
        d = days[i]
        if d not in inv_rows:
            out.append(dict(kind="shadow_skip", day=d, reason="114800 candle missing"))
            break
        o_long, c_long = float(raw["O"][i]), float(raw["C"][i])
        o_inv, c_inv = inv_rows[d]
        books, fills = {}, {}
        for b in BOOKS:
            if state is None:
                books[b] = dict(cash=CAPITAL, q_long=0.0, q_inv=0.0, target=None)
                fills[b] = []
            else:
                nb, f = settle(state["books"][b], state["books"][b]["target"], o_long, o_inv)
                books[b], fills[b] = nb, f
        late = d < today
        f = features_at(adj["C"][:i + 1], i)
        X = np.asarray([f])
        pl, ps = float(models["long"].predict(X)[0]), float(models["inv"].predict(X)[0])
        targets = dict(hold=L, ribbon=ribbon_at(adj["C"], i), kfft_W128_K3=kfft_at(raw["C"], i), vguard_L2=vguard_at(adj["C"], i))
        targets["learn_long"] = policy_step(state["books"]["learn_long"]["target"] if state else None, pl, ps, False)
        targets["learn_inv"] = policy_step(state["books"]["learn_inv"]["target"] if state else None, pl, ps, True)
        decided = dict(targets)
        if late:
            targets = {b: (L if books[b]["q_long"] > 0 else S if books[b]["q_inv"] > 0 else None) for b in BOOKS}
        for b in BOOKS:
            books[b]["target"] = targets[b]
            books[b]["value"] = books[b]["cash"] + books[b]["q_long"] * c_long + books[b]["q_inv"] * c_inv
            books[b]["return_pct"] = 100 * (books[b]["value"] / CAPITAL - 1)
        for b, allow_inv in HARVEST.items():
            if state is None or b not in state["books"]:
                hb, hf = dict(cash=CAPITAL, q_long=0.0, q_inv=0.0, leg=None, basis=0.0, entry=None, vault=0.0, pending=None), []
            else:
                hb, hf = harvest_settle(state["books"][b], o_long, o_inv)
            decided[b] = harvest_decide(hb, allow_inv, c_long, c_inv, pl, ps)
            hb["pending"] = None if late else decided[b]
            hb["equity"] = hb["cash"] + hb["q_long"] * c_long + hb["q_inv"] * c_inv
            hb["value"] = hb["equity"] + hb["vault"]
            hb["vault_pct"] = 100 * hb["vault"] / CAPITAL
            hb["return_pct"] = 100 * (hb["value"] / CAPITAL - 1)
            books[b], fills[b] = hb, hf
        for b, g in SKIM.items():
            if state is None or b not in state["books"]:
                kb, kf = dict(cash=CAPITAL, q_long=0.0, q_inv=0.0, basis=0.0, vault=0.0, pending=None), []
            else:
                kb, kf = skim_settle(state["books"][b], o_long)
            decided[b] = skim_decide(kb, g, c_long)
            kb["pending"] = None if late else decided[b]
            kb["equity"] = kb["cash"] + kb["q_long"] * c_long
            kb["value"] = kb["equity"] + kb["vault"]
            kb["vault_pct"] = 100 * kb["vault"] / CAPITAL
            kb["return_pct"] = 100 * (kb["value"] / CAPITAL - 1)
            books[b], fills[b] = kb, kf
        for b, (fl, mm) in CPPI.items():
            if state is None or b not in state["books"]:
                pb, pf = dict(cash=CAPITAL, q_long=0.0, q_inv=0.0, basis=0.0, vault=0.0, pending=None), []
            else:
                pb, pf = cppi_settle(state["books"][b], o_long)
            decided[b] = cppi_decide(pb, fl, mm, c_long)
            pb["pending"] = None if late else decided[b]
            pb["equity"] = pb["cash"] + pb["q_long"] * c_long
            pb["value"] = pb["equity"] + pb["vault"]
            pb["vault_pct"] = 100 * pb["vault"] / CAPITAL
            pb["return_pct"] = 100 * (pb["value"] / CAPITAL - 1)
            pb["exposure_pct"] = 100 * pb["q_long"] * c_long / pb["value"]
            books[b], fills[b] = pb, pf
        row = dict(kind="shadow_day", day=d, recorded_at=now_kst().isoformat(timespec="seconds"), decision_kind="backfill_no_trade" if late else "forward", backfill_decisions=decided if late else None, prices=dict(o_long=o_long, c_long=c_long, o_inv=o_inv, c_inv=c_inv), features=dict(zip(LI.FEATS, f)), pred_long=pl, pred_inv=ps, fills=fills, books=books, inputs=snap_info, note="decision at this close executes at the next session open; values marked at this close; fractional shares, fee 0.015% + 5 won per fill")
        out.append(row)
        state = row
    for r in out:
        append(r, ledger)
    return out


def run_once(today=None, raw_path=LIVE_RAW, adj_path=LIVE_ADJ, ledger=LEDGER, client_factory=read_only_client, snap_dir=SNAP_DIR, quiet=False):
    today = today or now_kst().strftime("%Y-%m-%d")
    prev = last_state(ledger)
    if prev and prev["day"] >= today:
        return []
    models, meta = load_models()
    raw, adj = load_series(raw_path, adj_path)
    try:
        inv_rows, candles, fetched = fetch_inverse(client_factory)
    except Exception as e:
        row = dict(kind="shadow_skip", day=today, reason="114800 fetch failed: " + type(e).__name__)
        append(row, ledger)
        return [row]
    inv_rows = {d: v for d, v in inv_rows.items() if d < fetched[:10] or (d == fetched[:10] and fetched[11:19] >= "15:31:00")}
    os.makedirs(snap_dir, exist_ok=True)
    sb = json.dumps(dict(symbol=S, fetched_at=fetched, candles=candles), ensure_ascii=False).encode("utf-8")
    sp = os.path.join(snap_dir, f"{today}_{fetched[11:19].replace(':', '')}.json")
    with open(sp, "xb") as f:
        f.write(sb)
    snap_info = dict(models_meta_sha256=META_SHA, live_raw_sha256=sha_b(open(raw_path, "rb").read()), live_adj_sha256=sha_b(open(adj_path, "rb").read()), inverse_snapshot=sp.replace("\\", "/"), inverse_snapshot_sha256=sha_b(sb))
    rows = process(today, raw, adj, inv_rows, models, prev, ledger, snap_info)
    if not quiet:
        for r in rows:
            if r["kind"] == "shadow_day":
                bk = r["books"]
                name = {L: "069500", S: "114800", None: "현금"}
                tag = " 소급기록·새 판단 미체결" if r.get("decision_kind") == "backfill_no_trade" else ""
                msg = f"[학습 그림자 {r['day']}{tag}] 예측 069500 {100 * r['pred_long']:+.2f}% 인버스 {100 * r['pred_inv']:+.2f}% → 다음 판단 학습 {name[bk['learn_long']['target']]} / 학습+인버스 {name[bk['learn_inv']['target']]} / 리본 {name[bk['ribbon']['target']]}. 가상 수익(1억, 비용 후) 학습 {bk['learn_long']['return_pct']:+.2f}% 학습+인버스 {bk['learn_inv']['return_pct']:+.2f}% 리본 {bk['ribbon']['return_pct']:+.2f}% 보유 {bk['hold']['return_pct']:+.2f}% 칼만FFT {bk['kfft_W128_K3']['return_pct']:+.2f}%(다음 {name[bk['kfft_W128_K3']['target']]}) 변동대응 {bk['vguard_L2']['return_pct']:+.2f}%(다음 {name[bk['vguard_L2']['target']]}). 현금생산 A 금고 {bk['harvest_A']['vault_pct']:.2f}% 합계 {bk['harvest_A']['return_pct']:+.2f}% / C(인버스) 금고 {bk['harvest_C']['vault_pct']:.2f}% 합계 {bk['harvest_C']['return_pct']:+.2f}% / 보유+이익떼기 금고 {bk['skim_R1']['vault_pct']:.2f}% 합계 {bk['skim_R1']['return_pct']:+.2f}% / 바닥95% 합계 {bk['floor95_m3']['return_pct']:+.2f}% 주식 {bk['floor95_m3']['exposure_pct']:.0f}% / 바닥90% 합계 {bk['floor90_m3']['return_pct']:+.2f}% 주식 {bk['floor90_m3']['exposure_pct']:.0f}%"
                subprocess.run([sys.executable, "-X", "utf8", IRC, "say", msg], cwd=HERE, timeout=60)
            else:
                subprocess.run([sys.executable, "-X", "utf8", IRC, "say", f"[학습 그림자 {r['day']}] 건너뜀: {r['reason']}"], cwd=HERE, timeout=60)
    return rows


def next_run(now):
    t = now.replace(hour=RUN_AT[0], minute=RUN_AT[1], second=0, microsecond=0)
    if now >= t:
        t += timedelta(days=1)
    while t.weekday() >= 5:
        t += timedelta(days=1)
    return t


def loop():
    while True:
        now = now_kst()
        today = now.strftime("%Y-%m-%d")
        prev = last_state(LEDGER)
        due = now.weekday() < 5 and (now.hour, now.minute) >= RUN_AT and not (prev and prev["day"] >= today)
        if due:
            try:
                run_once(today)
            except Exception as e:
                append(dict(kind="shadow_error", day=today, error=type(e).__name__), LEDGER)
        t = next_run(now_kst())
        time.sleep(max(30.0, min(3600.0, (t - now_kst()).total_seconds())))


def selftest():
    import tempfile
    tmp = tempfile.mkdtemp(prefix="TEST_FIXTURE_shadow_")
    adj = json.load(open(LI.ADJ_FILE, encoding="utf-8"))[L]
    rawf = json.load(open(os.path.join(HERE, "data", "kr_etf", "kodex_daily_raw.json"), encoding="utf-8"))[L]
    n = rawf["days"].index("2026-11-05") + 1
    days = rawf["days"][:n]
    rp, ap = os.path.join(tmp, "raw.json"), os.path.join(tmp, "adj.json")
    led = os.path.join(tmp, "ledger.jsonl")

    def write(m):
        json.dump(dict(days=days[:m], O=rawf["O"][:m], C=rawf["C"][:m]), open(rp, "w", encoding="utf-8"))
        json.dump(dict(days=days[:m], O=adj["O"][:m], C=adj["C"][:m]), open(ap, "w", encoding="utf-8"))

    iv = json.load(open(os.path.join(LI.INV_DIR_DEFAULT, "daily_raw.json"), encoding="utf-8"))[S]
    inv = {d: (float(o), float(c)) for d, o, c in zip(iv["days"], iv["O"], iv["C"])}

    lvr = json.load(open(os.path.join(HERE, "data", "kr_etf", "yf_lev_20261108_000349", "122630_raw.json"), encoding="utf-8"))
    levd = {d: (float(o), float(c)) for d, o, c in zip(lvr["days"], lvr["O"], lvr["C"])}

    class Fake:
        def __init__(self, upto):
            self.upto = upto

        def candles(self, sym, interval, count=30):
            src = levd if sym == LEV else inv
            ds = [d for d in sorted(src) if d <= self.upto][-count:]
            return {"candles": [{"timestamp": d + "T00:00:00+09:00", "openPrice": str(src[d][0]), "closePrice": str(src[d][1])} for d in ds]}

    checks = []
    write(n - 3)
    d0 = days[n - 4]
    r = run_once(d0, rp, ap, led, lambda: Fake(d0), os.path.join(tmp, "snap"), quiet=True)
    checks.append(("first run decides only", len(r) == 1 and r[0]["kind"] == "shadow_day" and all(not v for v in r[0]["fills"].values()) and r[0]["books"]["hold"]["value"] == CAPITAL))
    r2 = run_once(d0, rp, ap, led, lambda: Fake(d0), os.path.join(tmp, "snap"), quiet=True)
    checks.append(("same day is idempotent", r2 == []))
    write(n)
    d1 = days[n - 1]
    r3 = run_once(d1, rp, ap, led, lambda: Fake(d1), os.path.join(tmp, "snap"), quiet=True)
    checks.append(("catch-up processes missed days in order", [x["day"] for x in r3] == days[n - 3:n]))
    last = r3[-1]
    checks.append(("11-05 predictions equal the frozen model check", last["day"] == "2026-11-05" and last["pred_long"] == -0.00542095336063337 and last["pred_inv"] == 0.025339779548631997))
    h = r3[0]["books"]["hold"]
    o = float(rawf["O"][n - 3])
    checks.append(("hold bought at next open +5 won with fee", abs(h["q_long"] - CAPITAL / ((o + TICK) * (1 + FEE))) < 1e-9 and h["cash"] == 0.0))
    vals_ok = all(abs(x["books"][b]["value"] - (x["books"][b]["cash"] + x["books"][b]["q_long"] * x["prices"]["c_long"] + x["books"][b]["q_inv"] * x["prices"]["c_inv"])) < 1e-6 for x in r3 for b in BOOKS)
    checks.append(("values equal cash + marks", vals_ok))
    write(n)
    r4 = run_once("2026-11-09", rp, ap, led, lambda: Fake("2026-11-06"), os.path.join(tmp, "snap"), quiet=True)
    checks.append(("missing 069500 close for today is a skip", len(r4) == 1 and r4[0]["kind"] == "shadow_skip"))
    b0 = dict(cash=0.0, q_long=10.0, q_inv=0.0, target=L)
    b1, f1 = settle(b0, S, 100.0, 50.0)
    checks.append(("switch sells long then buys inverse with the proceeds", abs(b1["cash"]) < 1e-12 and b1["q_long"] == 0 and abs(b1["q_inv"] - 10 * 95 * (1 - FEE) / (55 * (1 + FEE))) < 1e-12 and [x["side"] for x in f1] == ["sell", "buy"]))
    hb = dict(cash=1000.0, q_long=0.0, q_inv=0.0, leg=None, basis=0.0, entry=None, vault=0.0, pending=L)
    hb, hf = harvest_settle(hb, 95.0, 50.0)
    q = 1000.0 / (100.0 * (1 + FEE))
    checks.append(("harvest buy at open +5 won with fee", hb["leg"] == L and abs(hb["q_long"] - q) < 1e-12 and hb["cash"] == 0.0))
    hb["pending"] = harvest_decide(hb, False, 111.0, 50.0, 0.0, 0.0)
    checks.append(("harvest take-profit triggers at +10% of entry", hb["pending"] == "sell"))
    hb2, hf2 = harvest_settle(hb, 125.0, 50.0)
    proceeds = q * 120.0 * (1 - FEE)
    checks.append(("harvest sale moves half the profit to the vault", abs(hb2["vault"] - 0.5 * (proceeds - 1000.0)) < 1e-9 and abs(hb2["cash"] - (proceeds - hb2["vault"])) < 1e-9 and hb2["leg"] is None))
    hb3 = dict(hb2, pending=None)
    checks.append(("harvest C picks the inverse when its prediction is best", harvest_decide(hb3, True, 0, 0, 0.001, 0.02) == S and harvest_decide(hb3, False, 0, 0, 0.001, 0.02) == L))
    hb4 = dict(cash=0.0, q_long=10.0, q_inv=0.0, leg=L, basis=1000.0, entry=100.0, vault=0.0, pending="sell")
    hb4, _ = harvest_settle(hb4, 85.0, 50.0)
    checks.append(("harvest loss keeps the vault unchanged", hb4["vault"] == 0.0 and abs(hb4["cash"] - 10 * 80 * (1 - FEE)) < 1e-9))
    late_rows = [x for x in r3 if x["day"] < d1]
    hold_keep = all(x["decision_kind"] == "backfill_no_trade" and all(x["books"][b]["target"] == (L if x["books"][b]["q_long"] > 0 else S if x["books"][b]["q_inv"] > 0 else None) for b in BOOKS) and all(x["books"][b]["pending"] is None for b in HARVEST) for x in late_rows)
    checks.append(("catch-up days are backfill: decisions recorded, no new orders", len(late_rows) == 2 and hold_keep and r3[-1]["decision_kind"] == "forward"))
    checks.append(("no fill at an open that follows a late decision", all(not v for v in r3[-1]["fills"].values()) and all(not v for v in r3[-2]["fills"].values())))
    rows_h = [x for x in r3 if "harvest_A" in x["books"]]
    checks.append(("harvest books recorded every shadow day", len(rows_h) == len(r3) and all(abs(x["books"]["harvest_A"]["value"] - (x["books"]["harvest_A"]["equity"] + x["books"]["harvest_A"]["vault"])) < 1e-6 for x in rows_h)))
    kb = dict(cash=1000.0, q_long=0.0, q_inv=0.0, basis=0.0, vault=0.0, pending="buy")
    kb, kf = skim_settle(kb, 95.0)
    kq = 1000.0 / (100.0 * (1 + FEE))
    checks.append(("skim buy at open +5 won with fee", abs(kb["q_long"] - kq) < 1e-12 and kb["basis"] == 1000.0 and kb["cash"] == 0.0 and [x["side"] for x in kf] == ["buy"]))
    checks.append(("skim waits below +10% of basis", skim_decide(kb, 0.10, 105.0) is None))
    kb["pending"] = skim_decide(kb, 0.10, 111.0)
    kv = kq * 111.0
    kf_ok = kb["pending"][0] == "skim" and abs(kb["pending"][1] - 0.5 * (kv - 1000.0) / kv) < 1e-12
    checks.append(("skim decides half the profit at +10%", kf_ok))
    kb2, kf2 = skim_settle(json.loads(json.dumps(kb)), 125.0)
    sq = kb["pending"][1] * kq
    checks.append(("skim sells the slice at open -5 won into the vault and resets the basis", abs(kb2["vault"] - sq * 120.0 * (1 - FEE)) < 1e-9 and abs(kb2["q_long"] - (kq - sq)) < 1e-12 and abs(kb2["basis"] - (kq - sq) * 125.0) < 1e-9 and kb2["cash"] == 0.0))
    rows_k = [x for x in r3 if "skim_R1" in x["books"]]
    checks.append(("skim book recorded every day, no pending on catch-up days", len(rows_k) == len(r3) and all(abs(x["books"]["skim_R1"]["value"] - (x["books"]["skim_R1"]["equity"] + x["books"]["skim_R1"]["vault"])) < 1e-6 for x in rows_k) and all(x["books"]["skim_R1"]["pending"] is None for x in late_rows) and r3[0]["books"]["skim_R1"]["q_long"] > 0))
    pb = dict(cash=CAPITAL, q_long=0.0, q_inv=0.0, basis=0.0, vault=0.0, pending=None)
    d0p = cppi_decide(pb, 0.90, 3, 100.0)
    checks.append(("floor book first target is m x cushion of capital", d0p[0] == "trade" and abs(d0p[1] - 0.30 * CAPITAL / 100.0) < 1e-6))
    pb["pending"] = d0p
    pb, pf = cppi_settle(pb, 95.0)
    bq = min(CAPITAL, d0p[1] * 100.0 * (1 + FEE)) / (100.0 * (1 + FEE))
    checks.append(("floor buy at open +5 won with fee", abs(pb["q_long"] - bq) < 1e-6 and abs(pb["cash"] + pb["basis"] - CAPITAL) < 1e-6 and [x["side"] for x in pf] == ["buy"]))
    checks.append(("floor no trade inside the 5% band", cppi_decide(pb, 0.90, 3, 101.0) is None))
    crash = cppi_decide(pb, 0.90, 3, 100.0 * (1 - 0.34))
    checks.append(("floor sells everything when the cushion is gone", crash[0] == "trade" and abs(crash[1] + pb["q_long"]) < 1e-6))
    up = cppi_decide(pb, 0.90, 3, 112.0)
    checks.append(("floor book skims half the profit at +10%", up[0] == "skim" and abs(up[1] - 0.5 * (pb["q_long"] * 112.0 - pb["basis"]) / (pb["q_long"] * 112.0)) < 1e-12))
    pb2 = dict(pb, pending=["trade", -pb["q_long"] / 2])
    pb2, _ = cppi_settle(pb2, 125.0)
    sold = pb["q_long"] / 2
    prof = sold * 120.0 * (1 - FEE) - pb["basis"] / 2
    checks.append(("floor de-risk sale sends half the realized profit to the vault", abs(pb2["vault"] - 0.5 * prof) < 1e-6 and abs(pb2["basis"] - pb["basis"] / 2) < 1e-6))
    rows_p = [x for x in r3 if "floor90_m3" in x["books"] and "floor95_m3" in x["books"]]
    checks.append(("floor books recorded every day, no pending on catch-up days", len(rows_p) == len(r3) and all(abs(x["books"][b]["value"] - (x["books"][b]["equity"] + x["books"][b]["vault"])) < 1e-6 for x in rows_p for b in CPPI) and all(x["books"][b]["pending"] is None for x in late_rows for b in CPPI) and r3[0]["books"]["floor90_m3"]["q_long"] > 0))
    import kalman_fft_v1 as K1
    yC = np.log(np.asarray(rawf["C"][:n], float))
    levf, _ = K1.kalman(yC, np.zeros(n, dtype=int), KFFT[2])
    ff = K1.fft_features(levf, np.arange(n), KFFT[0], KFFT[1])
    probe = [n - 1, n - 40, n - 400, n - 2000]
    checks.append(("kfft target is causal and equals the full-series research signal", all(kfft_at(rawf["C"], i) == (L if ff[i, 5] > 0 else None) for i in probe)))
    checks.append(("kfft book recorded every day", all("kfft_W128_K3" in x["books"] for x in r3)))
    import msma_v1 as M1
    Aall = np.asarray(adj["C"][:n], float)
    Sg, _, _ = M1.signals(np.arange(n), Aall[:, None])
    eqf = np.nanmean(np.tanh(np.stack([Sg[:, :, M1.SIG.index(x)] for x in M1.EQ_SIGNALS], axis=2)), axis=2)[:, 0]
    checks.append(("lev EQ score is causal and equals the full-series research score", all(abs(eq_at(adj["C"], i) - eqf[i]) < 1e-9 for i in (n - 1, n - 60, n - 700))))
    import vswing_v1 as VS
    fv = VS.features(np.asarray(adj["C"][:n], float))
    wv, _ = VS.targets(fv, "L2_trend_crashguard")
    checks.append(("vguard target is causal and equals the full-series research rule", all(vguard_at(adj["C"], i) == (L if wv[i] > 0 else None) for i in (n - 1, n - 30, n - 150, n - 900))))
    checks.append(("vguard book recorded every day", all("vguard_L2" in x["books"] for x in r3)))
    for name, ok in checks:
        print("PASS" if ok else "FAIL", name)
    print("all passed", all(ok for _, ok in checks), "fixture dir", tmp)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "selftest":
        selftest()
    elif cmd == "once":
        print(json.dumps(run_once(), ensure_ascii=False)[:2000])
    elif cmd == "loop":
        loop()
    else:
        print("usage: learn_shadow.py selftest | once | loop")
