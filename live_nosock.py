import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ft"))
sys.dont_write_bytecode = True
import numpy as np

KST = ZoneInfo("Asia/Seoul")
NY = ZoneInfo("America/New_York")
RUNS = os.environ.get("NOSOCK_RUNS", os.path.join(HERE, "runs"))
LOG = os.environ.get("NOSOCK_LOG", os.path.join(RUNS, "live_nosock.jsonl"))
STATE = os.environ.get("NOSOCK_STATE", os.path.join(RUNS, "live_nosock_state.json"))
STOP = os.path.join(RUNS, "live_nosock.stop")
MAX_EQUITY = float(os.environ.get("LIVE_MAX_EQUITY", 5_000_000))
DAILY_DEPOSIT = float(os.environ.get("NOSOCK_DAILY_DEPOSIT", 10_000))
SEED = float(os.environ.get("NOSOCK_SEED", 0))
BUFFER = 0.01
KR_FEE = 0.003
MIN_US_USD = 1.0
CYCLE = 20
KR_PREP, KR_OPEN, KR_LAST = (8, 40), (9, 6), (15, 15)
US_OPEN_NY, US_LAST_NY = (9, 30), (15, 50)
US_TAX, US_DEDUCTION = 0.22, 2_500_000.0
FINAL = ("FILLED", "CANCELED", "REJECTED")


def live_on():
    return os.environ.get("DOBBY_LIVE") == "1"


def now():
    return datetime.now(KST)


def log(kind, **kw):
    row = dict(ts=now().isoformat(timespec="seconds"), kind=kind, live=live_on(), **kw)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
    return row


def load_state():
    try:
        return json.load(open(STATE, encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(s):
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=1, default=float)
    os.replace(tmp, STATE)


def listfile(name):
    p = os.path.join(RUNS, name)
    return [x for x in open(p, encoding="utf-8").read().split() if x] if os.path.exists(p) else []


def mk(markets, c):
    return markets.get(c) or ("KR" if c[:1].isdigit() else "US")


def fetch_long(t, c, pages=2):
    out, before = [], None
    for _ in range(pages):
        cs = t.candles(c, "1d", count=200, before=before, adjusted=True)["candles"]
        if not cs:
            break
        out += cs
        if len(cs) < 200:
            break
        before = min(x["timestamp"] for x in cs)
        time.sleep(0.2)
    return out


def build_market(t, extra=()):
    import mom_shadow as MS
    kr, kus, us, fx_hist = MS.pinned()
    cut = lambda d: {c: MS.cut(v, MS.ANCHOR) for c, v in d.items()}
    kr, kus, us = cut(kr), cut(kus), cut(us)
    fx_hist = {d: v for d, v in fx_hist.items() if d <= MS.ANCHOR}
    snap, fx_now, fetched = MS.fetch_all(lambda: t, list(kr) + list(kus), list(us))
    rows_kr = {c: MS.to_rows(snap[c], fetched, "KR") for c in list(kr) + list(kus)}
    rows_us = {c: MS.to_rows(snap[c], fetched, "US") for c in us}
    for r in MS.read_rows(MS.FX_LOG, "fx"):
        fx_hist[r["us_day"]] = r["midRate"]
    last_us = max((max(v) for v in rows_us.values() if v), default=None)
    if last_us and last_us > MS.ANCHOR:
        fx_hist.setdefault(last_us, fx_now["midRate"])
    kr = {c: MS.splice(v, rows_kr[c]) for c, v in kr.items()}
    kus = {c: MS.splice(v, rows_kr[c]) for c, v in kus.items()}
    us = {c: MS.splice(v, rows_us[c]) for c, v in us.items()}
    inst, groups = MS.build(kr, kus, us, fx_hist)
    import ms_alloc_v7 as A7
    from nosock import run_backtest as RB
    ext = {}
    for c in extra:
        if c in inst:
            continue
        try:
            rows = MS.to_rows(fetch_long(t, c), fetched, "KR" if c[:1].isdigit() else "US")
        except Exception as e:
            log("extra_fetch_error", code=c, error=type(e).__name__)
            continue
        ds = sorted(rows)
        if len(ds) < 260:
            log("extra_short_history", code=c, days=len(ds))
            continue
        ext[c] = dict(days=ds, **{k: [rows[d][j] for d in ds] for j, k in enumerate("OHLC")})
    if ext:
        us_days = sorted(set(d for v in us.values() for d in v["days"]) | set(d for c, v in ext.items() if not c[:1].isdigit() for d in v["days"]))
        fxm = MS.fx_ff(us_days, fx_hist)
        for c, rec in ext.items():
            inst[c] = MS.make_ins(rec, c, "KR", A7.kr_cost) if c[:1].isdigit() else MS.make_ins(rec, c, "US", A7.us_cost_maker(RB.US_FEE, RB.US_FX), fxm)
    for c, ins in inst.items():
        ins["cost_maker"] = A7.kr_cost if ins["market"] == "KR" else A7.us_cost_maker(RB.US_FEE, RB.US_FX)
    return inst, groups, fx_now, fetched


def decide(inst, groups, D, extra=()):
    import ms_alloc_v3 as A3
    from nosock import allocator as AL
    from nosock import engine as E
    from nosock import run_backtest as RB
    from nosock import selector as SEL
    codes = groups["U29"]
    m = E.Market(inst, codes, {})
    m.raw = A3.load_raw()
    mom = RB.strategy_for(inst, codes, "dual", m)
    st = SEL.State(m, mom)
    feats = AL.Features(m, st)
    rows = [r for r in AL.dataset(m, feats, "2006-01-01", D) if r["t_end"] <= D]
    pol, lam, lam_scores = AL.fit_auto(rows, D)
    extra = [c for c in extra if c in inst and c not in codes]
    w, why = AL.Allocator(m, feats, pol).decide(D, None, extra)
    why["lambda"] = lam
    why["lambda_scores"] = {str(k): v for k, v in lam_scores.items()}
    why["trained_rows"] = len(rows)
    why["last_label_end"] = rows[-1]["t_end"] if rows else None
    why["outside"] = [x["code"] for x in why.get("candidates", []) if x.get("outside")]
    markets = {c: inst[c]["market"] for c in list(codes) + extra}
    return w, why, markets, m


def kr_sessions_after(inst, d0):
    return [d for d in inst["069500"]["days"] if d > d0]


def new_book(t):
    adopt = set(listfile("adopt.txt"))
    b = dict(krw=SEED, usd=0.0, qty={}, cost={}, cost_krw={}, vault_krw=0.0, vault_usd=0.0, tax={}, tax_usd=0.0, realized=0.0, hwm=0.0, deposits=0.0, contributed=SEED, pending={})
    items = [it for it in t.holdings()["items"] if it["symbol"] in adopt and float(it["quantity"]) > 0]
    if items:
        px = {p["symbol"]: float(p["lastPrice"]) for p in t.prices([it["symbol"] for it in items])}
        rate = float(t.fx("USD", "KRW")["midRate"])
        for it in items:
            s, q = it["symbol"], float(it["quantity"])
            avg = float(it.get("averagePurchasePrice") or 0.0)
            us = not s[:1].isdigit()
            b["qty"][s] = q
            b["cost"][s] = q * avg
            if us:
                b["cost_krw"][s] = q * avg * rate
            val = q * px.get(s, avg) * (rate if us else 1.0)
            b["contributed"] += val
            log("adopt", code=s, qty=q, avg_price=avg, value_krw=val, note="이전 도비가 산 보유분을 장부로 가져옴")
    return b


def book_equity(b, px, fx, markets):
    val = {}
    for s, q in b["qty"].items():
        p = px.get(s)
        if p is not None:
            val[s] = q * p * (fx if mk(markets, s) == "US" else 1.0)
    return val, sum(val.values()) + b["krw"] + b["usd"] * fx


class Account:
    def __init__(self, t, st, codes, markets):
        self.t = t
        self.st = st
        self.codes = set(codes)
        self.markets = markets

    def snapshot(self):
        t, b = self.t, self.st["book"]
        actual = {it["symbol"]: float(it["quantity"]) for it in t.holdings()["items"]}
        for s in list(b["qty"]):
            a = actual.get(s, 0.0)
            if b["qty"][s] > a + 1e-6 and not b["pending"]:
                log("book_clamp", code=s, book=b["qty"][s], account=a, note="계좌 보유가 장부보다 적어 장부를 줄임")
                if a <= 1e-9:
                    b["qty"].pop(s)
                    b["cost"].pop(s, None)
                    b.get("cost_krw", {}).pop(s, None)
                else:
                    b["cost"][s] = b["cost"].get(s, 0.0) * a / b["qty"][s]
                    if s in b.get("cost_krw", {}):
                        b["cost_krw"][s] *= a / b["qty"][s]
                    b["qty"][s] = a
        syms = sorted(set(b["qty"]) | self.codes)
        px = {}
        for i in range(0, len(syms), 20):
            for p in t.prices(syms[i:i + 20]):
                px[p["symbol"]] = float(p["lastPrice"])
        acct_krw = float(t.buying_power("KRW")["cashBuyingPower"])
        acct_usd = float(t.buying_power("USD")["cashBuyingPower"])
        fx = t.fx("USD", "KRW")
        rate = float(fx["midRate"])
        buy_rate = max(rate, float(fx.get("rate") or rate))
        val, equity = book_equity(b, px, rate, self.markets)
        vault = b["vault_krw"] + b["vault_usd"] * rate
        return dict(held=dict(b["qty"]), px=px, krw=b["krw"], usd=b["usd"], fx=rate, fx_buy=buy_rate, value=val, equity=equity, vault=vault, tax_reserve=b.get("tax_usd", 0.0) * rate, acct_krw=acct_krw, acct_usd=acct_usd)


def make_plan(weights, snap, markets, rebalance, us_only=False):
    fx = snap["fx"]
    B = snap["equity"]
    inv = max(0.0, B * (1 - BUFFER))
    tgt = {c: w * inv for c, w in weights.items()}
    h = dict(snap["value"])
    krw, usd = snap["krw"], snap["usd"]
    kr, us = [], []
    if rebalance:
        for c in sorted(h):
            px = snap["px"].get(c)
            if not px:
                continue
            m = mk(markets, c)
            if us_only and m != "US":
                continue
            ex = h[c] - tgt.get(c, 0.0)
            q0 = snap["held"].get(c, 0.0)
            if m == "KR":
                n = int(q0) if c not in weights else min(int(q0), int(math.floor(ex / px + 0.5)))
                if n > 0:
                    kr.append(dict(code=c, side="SELL", qty=n, px=px, cur_qty=q0))
                    h[c] -= n * px
                    krw += n * px * (1 - KR_FEE)
            else:
                full = c not in weights
                ex_usd = ex / fx
                if full or ex_usd >= MIN_US_USD:
                    q = q0 if full else min(q0, math.floor(ex_usd / px * 1e6) / 1e6)
                    if q > 0:
                        us.append(dict(code=c, side="SELL", qty=q, usd=round(q * px, 2), px=px, cur_qty=q0, full_exit=full))
                        h[c] -= q * px * fx
                        usd += q * px
    cash_tgt = B - sum(tgt.values())
    spend = max(0.0, krw + usd * fx - cash_tgt)
    if us_only:
        spend = min(spend, usd * fx)
    kr_left = krw
    buys = {}
    while True:
        best, bd = None, 0.0
        for c, T in tgt.items():
            px = snap["px"].get(c)
            m = mk(markets, c)
            if not px or (us_only and m != "US"):
                continue
            d = T - h.get(c, 0.0)
            if d <= bd:
                continue
            if m == "KR":
                if d < 0.5 * px or px * (1 + KR_FEE) > min(spend, kr_left):
                    continue
            elif MIN_US_USD * fx > spend:
                continue
            best, bd = c, d
        if best is None:
            break
        px = snap["px"][best]
        if mk(markets, best) == "KR":
            n = max(1, int(math.floor(bd / px + 0.5)))
            n = min(n, int(math.floor(min(spend, kr_left) / (px * (1 + KR_FEE)))))
            cost = n * px * (1 + KR_FEE)
            h[best] = h.get(best, 0.0) + n * px
            spend -= cost
            kr_left -= cost
            buys[best] = buys.get(best, 0) + n
        else:
            a = math.floor(min(max(MIN_US_USD * fx, bd), spend) / fx * 100) / 100
            if a < MIN_US_USD:
                break
            h[best] = h.get(best, 0.0) + a * fx
            spend -= a * fx
            buys[best] = buys.get(best, 0.0) + a
    for c, v in sorted(buys.items(), key=lambda kv: -tgt.get(kv[0], 0.0)):
        px = snap["px"][c]
        if mk(markets, c) == "KR":
            kr.append(dict(code=c, side="BUY", qty=v, px=px, target_krw=tgt.get(c, 0.0), cur_qty=snap["held"].get(c, 0.0)))
        else:
            us.append(dict(code=c, side="BUY", usd=round(v, 2), px=px, target_usd=tgt.get(c, 0.0) / fx, cur_qty=snap["held"].get(c, 0.0), full_exit=False))
    us_buy = sum(x["usd"] for x in us if x["side"] == "BUY")
    us_sell = sum(x["usd"] for x in us if x["side"] == "SELL")
    need_usd = max(0.0, us_buy - us_sell - snap["usd"])
    kr_buy = sum(x["qty"] * x["px"] for x in kr if x["side"] == "BUY")
    kr_sell = sum(x["qty"] * x["px"] for x in kr if x["side"] == "SELL")
    return dict(investable=inv, equity=B, targets_krw=tgt, kr=kr, us=us, need_usd=round(need_usd * 1.01, 2) if need_usd > 0.5 else 0.0, kr_buy=kr_buy, kr_sell=kr_sell, us_buy=us_buy, us_sell=us_sell, rebalance=rebalance, idle_krw=max(0.0, spend))


def realize(st, p_cur, cur, fx, code):
    b = st["book"]
    p_krw = p_cur * (fx if cur == "usd" else 1.0)
    N = b["realized"] + p_krw
    pay_krw = 0.5 * max(0.0, N - b["hwm"])
    b["hwm"] = max(b["hwm"], N)
    b["realized"] = N
    pay = pay_krw / (fx if cur == "usd" else 1.0)
    if pay > 0:
        b[cur] -= pay
        b["vault_" + cur] += pay
    vault = b["vault_krw"] + b["vault_usd"] * fx
    ty = (b.get("tax") or {}).get(str(now().astimezone(NY).year)) or {}
    log("vault", code=code, realized_trade=p_krw, profit=N, hwm=b["hwm"], paid=pay_krw, vault=vault, vault_krw=b["vault_krw"], vault_usd=b["vault_usd"], principal=b["contributed"],
        tax_reserve=b.get("tax_usd", 0.0) * fx, tax_year_gain=ty.get("gain", 0.0), tax_accrued=ty.get("accrued", 0.0), tax_deduction=us_deduction())


def us_deduction():
    try:
        return float(open(os.path.join(RUNS, "us_tax_deduction.txt"), encoding="utf-8").read().strip().replace(",", ""))
    except (OSError, ValueError):
        return US_DEDUCTION


def tax_fields(b):
    b.setdefault("cost_krw", {})
    b.setdefault("tax", {})
    b.setdefault("tax_usd", 0.0)
    return b


def accrue_us_tax(st, gain_krw, fx, code):
    b = tax_fields(st["book"])
    y = str(now().astimezone(NY).year)
    T = b["tax"].setdefault(y, dict(gain=0.0, accrued=0.0, usd=0.0))
    T["gain"] += gain_krw
    ded = us_deduction()
    acc = US_TAX * max(0.0, T["gain"] - ded)
    d_usd = (acc - T["accrued"]) / fx
    if d_usd < 0:
        d_usd = -min(-d_usd, T["usd"])
    b["usd"] -= d_usd
    b["tax_usd"] += d_usd
    T["usd"] += d_usd
    T["accrued"] = acc
    log("us_tax", year=y, code=code, gain_krw=gain_krw, year_gain=T["gain"], deduction=ded, accrued=acc, moved_usd=d_usd, reserve_usd=T["usd"], reserve_total_usd=b["tax_usd"])
    return d_usd * fx


def settle_tax(st):
    b = st.get("book")
    if not b or not b.get("tax"):
        return
    paid = set(listfile("tax_paid.txt"))
    for y, T in b["tax"].items():
        if y in paid and not T.get("paid"):
            b["tax_usd"] -= T["usd"]
            log("us_tax_paid", year=y, accrued=T["accrued"], released_usd=T["usd"], note="주인님이 낸 양도세만큼 적립금을 도비 장부에서 뺌")
            T.update(paid=True, released_usd=T["usd"], usd=0.0)
            save_state(st)


def apply_fill(st, code, side, market, q, gross, fee, fx):
    b = tax_fields(st["book"])
    cur = "krw" if market == "KR" else "usd"
    if q <= 0:
        return
    if side == "BUY":
        if cur == "usd":
            b["cost_krw"][code] = b["cost_krw"].get(code, b["cost"].get(code, 0.0) * fx) + (gross + fee) * fx
        b[cur] -= gross + fee
        b["qty"][code] = b["qty"].get(code, 0.0) + q
        b["cost"][code] = b["cost"].get(code, 0.0) + gross + fee
    else:
        q0 = b["qty"].get(code, 0.0)
        frac = min(1.0, q / q0) if q0 > 0 else 0.0
        basis = b["cost"].get(code, 0.0) * frac
        basis_krw = b["cost_krw"].get(code, b["cost"].get(code, 0.0) * fx) * frac
        got = gross - fee
        b[cur] += got
        left = q0 - q
        if left <= 1e-9:
            b["qty"].pop(code, None)
            b["cost"].pop(code, None)
            b["cost_krw"].pop(code, None)
        else:
            b["qty"][code] = left
            b["cost"][code] = b["cost"].get(code, 0.0) - basis
            if code in b["cost_krw"]:
                b["cost_krw"][code] -= basis_krw
        if cur == "usd":
            gain_krw = got * fx - basis_krw
            taxed = accrue_us_tax(st, gain_krw, fx, code)
            realize(st, (gain_krw - taxed) / fx, cur, fx, code)
        else:
            realize(st, got - basis, cur, fx, code)


def exec_totals(o, px_hint):
    ex = o.get("execution") or {}
    q = float(ex.get("filledQuantity") or 0.0)
    avg = float(ex.get("averageFilledPrice") or px_hint or 0.0)
    gross = float(ex["filledAmount"]) if ex.get("filledAmount") not in (None, "") else q * avg
    fee = float(ex.get("commission") or 0.0) + float(ex.get("tax") or 0.0)
    return q, gross, fee


def wait_order(t, oid, timeout=90):
    t0 = time.time()
    o = t.order(oid)
    while time.time() - t0 < timeout and o["status"] not in FINAL:
        time.sleep(2)
        o = t.order(oid)
    return o


def place(t, st, cycle, x, market, fx, **kw):
    cid = f"ns-{cycle}-{x['code']}-{x['side'][0]}-{int(time.time())}"
    if not live_on():
        return log("dry_order", cycle=cycle, code=x["code"], side=x["side"], client_order_id=cid, **kw)
    try:
        r = t.place(x["code"], x["side"], "MARKET", client_order_id=cid, **kw)
    except Exception as e:
        return log("order_error", cycle=cycle, code=x["code"], side=x["side"], client_order_id=cid, error=type(e).__name__, code_detail=getattr(e, "code", None), status=getattr(e, "status", None), request=kw)
    oid = r["orderId"]
    try:
        o = wait_order(t, oid)
    except Exception as e:
        st["book"]["pending"][oid] = dict(code=x["code"], side=x["side"], market=market, q=0.0, gross=0.0, fee=0.0, px=x["px"])
        save_state(st)
        return log("order_error", cycle=cycle, code=x["code"], side=x["side"], client_order_id=cid, order_id=oid, error=type(e).__name__, where="order_status", request=kw)
    q, gross, fee = exec_totals(o, x["px"])
    apply_fill(st, x["code"], x["side"], market, q, gross, fee, fx)
    if o["status"] not in FINAL:
        st["book"]["pending"][oid] = dict(code=x["code"], side=x["side"], market=market, q=q, gross=gross, fee=fee, px=x["px"])
    save_state(st)
    return log("order", cycle=cycle, code=x["code"], side=x["side"], client_order_id=cid, order_id=oid, status=o["status"], filled_qty=q, avg_price=(gross / q if q else None), filled_amount=gross, fee=fee, request=kw)


def reconcile(t, st, fx):
    pend = st.get("book", {}).get("pending") or {}
    for oid in list(pend):
        p = pend[oid]
        try:
            o = t.order(oid)
        except Exception as e:
            log("error", error=type(e).__name__, where="reconcile", order_id=oid)
            continue
        q, gross, fee = exec_totals(o, p["px"])
        dq = q - p["q"]
        if dq > 1e-9:
            apply_fill(st, p["code"], p["side"], p["market"], dq, gross - p["gross"], fee - p["fee"], fx)
            log("order", code=p["code"], side=p["side"], order_id=oid, status=o["status"], filled_qty=dq, filled_amount=gross - p["gross"], fee=fee - p["fee"], note="늦게 체결된 분량")
            p.update(q=q, gross=gross, fee=fee)
        if o["status"] in FINAL:
            pend.pop(oid)
        save_state(st)


def open_order_symbols(t):
    try:
        return {o.get("symbol") for o in t.orders("OPEN").get("orders", [])}
    except Exception:
        return set()


def mark(st, acct, phase, snap=None):
    s = snap or acct.snapshot()
    b = st["book"]
    return log("mark", phase=phase, cycle=st.get("current"), equity=s["equity"], krw=s["krw"], usd=s["usd"], fx=s["fx"], held=s["held"], value=s["value"], vault=s["vault"], tax_reserve=s.get("tax_reserve", 0.0), contributed=b["contributed"], deposits=b["deposits"], realized=b["realized"])


SCEN_LOG = os.environ.get("NOSOCK_SCENARIO_LOG", os.path.join(RUNS, "scenario_desk.jsonl"))
SCEN_MAX_AGE_H = float(os.environ.get("NOSOCK_SCENARIO_MAX_AGE_H", 20))
SCEN_ON = os.environ.get("NOSOCK_SCENARIO_LIVE", "1") == "1"
RETARGET = 0.10


def latest_scenario():
    if not os.path.exists(SCEN_LOG):
        return None
    best = None
    with open(SCEN_LOG, encoding="utf-8") as f:
        for line in f:
            if '"kind": "scenario"' not in line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("kind") == "scenario" and r.get("opt_ok") and r.get("scenarios") and r.get("probabilities"):
                best = r
    if not best:
        return None
    ts = datetime.fromisoformat(best["ts"])
    if (now() - ts).total_seconds() > SCEN_MAX_AGE_H * 3600:
        return None
    return best


def effective_weights(base, markets):
    if not SCEN_ON:
        return dict(base), None
    row = latest_scenario()
    if not row:
        return dict(base), None
    import scenario_desk as SD
    codes = sorted(set(base) | {c for s in row["scenarios"] for c in (s.get("returns_pct") or {}) if c in markets})
    try:
        p, R = SD.validate(dict(scenarios=row["scenarios"]), codes)
        w0 = np.array([base.get(c, 0.0) for c in codes])
        cap = SD.current_cap()
        w, ok = SD.optimize(w0, p, R, cap, SD.PER_NAME, codes, row.get("vols"))
    except Exception as e:
        log("scenario_apply_error", error=type(e).__name__)
        return dict(base), None
    if not ok:
        log("scenario_apply_error", error="optimizer_not_converged", desk_ts=row["ts"])
        return dict(base), None
    eff = {c: float(x) for c, x in zip(codes, w) if x >= 0.002}
    tilt = {c: round(eff.get(c, 0.0) - base.get(c, 0.0), 4) for c in codes if abs(eff.get(c, 0.0) - base.get(c, 0.0)) > 0.002}
    return eff, dict(desk_ts=row["ts"], cap=cap, limit="risk_budget" if row.get("vols") and all(c in row["vols"] for c in codes) else "per_name", trigger=row.get("trigger"), source=row.get("source", "desk"), issue_ids=row.get("issue_ids"), moved=round(sum(v for v in tilt.values() if v > 0), 4), tilt=tilt)


def target_weights(st, markets, phase, rebalance):
    base = st["cycles"][st["current"]]["weights"]
    weights, info = effective_weights(base, markets)
    prev = st.get("last_target")
    diff = sum(abs(weights.get(c, 0.0) - prev.get(c, 0.0)) for c in set(weights) | set(prev)) if prev else 0.0
    if not rebalance and prev and diff > RETARGET:
        rebalance = True
        log("scenario_rebalance", phase=phase, diff=round(diff, 4), desk_ts=(info or {}).get("desk_ts"), note="시나리오가 바뀌어 목표 비율이 크게 달라져 오늘 팔고 사서 맞춤")
    st["last_target"] = weights
    log("weights_used", phase=phase, source="scenario" if info else "model", scenario=info, rebalance=rebalance, weights={c: round(v, 4) for c, v in sorted(weights.items(), key=lambda kv: -kv[1])})
    return weights, rebalance


def run_kr(t, st, acct, day, rebalance):
    cyc = st["current"]
    weights, rebalance = target_weights(st, acct.markets, "kr", rebalance)
    snap = acct.snapshot()
    plan = make_plan(weights, snap, acct.markets, rebalance)
    log("day_plan", cycle=cyc, day=day, phase="kr", rebalance=rebalance, equity=snap["equity"], kr=plan["kr"], us=plan["us"], need_usd=plan["need_usd"], idle_krw=plan["idle_krw"])
    busy = open_order_symbols(t)
    kr = [x for x in plan["kr"] if x["code"] not in busy]
    for x in [x for x in kr if x["side"] == "SELL"]:
        sq = float(t.get("/api/v1/sellable-quantity", account=True, symbol=x["code"])["sellableQuantity"]) if live_on() else x["qty"]
        q = int(min(x["qty"], sq, st["book"]["qty"].get(x["code"], 0.0)))
        if q > 0:
            place(t, st, cyc, x, "KR", snap["fx"], quantity=q)
    acct_krw = float(t.buying_power("KRW")["cashBuyingPower"])
    avail = min(st["book"]["krw"], acct_krw)
    for x in [x for x in kr if x["side"] == "BUY"]:
        afford = int(math.floor(max(0.0, avail) / (x["px"] * (1 + KR_FEE))))
        q = min(x["qty"], afford)
        if q <= 0:
            log("skip", cycle=cyc, code=x["code"], reason="원화 부족", need=x["qty"] * x["px"], krw=avail)
            continue
        place(t, st, cyc, x, "KR", snap["fx"], quantity=q)
        avail -= q * x["px"] * (1 + KR_FEE)
    snap2 = acct.snapshot()
    plan2 = make_plan(weights, snap2, acct.markets, rebalance)
    st["fx_pending"] = None
    if plan2["need_usd"] > 0:
        need = plan2["need_usd"]
        b = st["book"]
        free_usd = max(0.0, snap2["acct_usd"] - b["usd"] - b["vault_usd"] - b.get("tax_usd", 0.0))
        short = math.ceil(max(0.0, need - free_usd) * 100) / 100
        st["fx_pending"] = dict(day=day, usd=need, rate=snap2["fx_buy"])
        if short > 0:
            won = int(math.ceil(short * snap2["fx_buy"] / 100.0) * 100)
            log("fx_request", id=f"live-{day}", cycle=cyc, day=day, direction="KRW->USD", amount_usd=short, amount_won=won, need_usd=need, account_free_usd=round(free_usd, 2), fx_mid=snap2["fx"], body=f"도비가 오늘 밤 미국 종목을 사려면 ${need:,.2f}가 필요한데, 계좌에서 쓸 수 있는 달러는 ${free_usd:,.2f}라 ${short:,.2f}가 모자랍니다. 미국장이 열리는 {us_open_kst(day)} 전에 원화 {won:,}원을 달러로 바꿔 주세요.")
        else:
            log("fx_ready", day=day, need_usd=need, account_free_usd=round(free_usd, 2), note=f"계좌 달러 ${free_usd:,.2f}로 충분해 환전 요청 없음. {us_open_kst(day)} 미국장에서 ${need:,.2f}를 도비 장부로 옮김")
    save_state(st)
    mark(st, acct, "kr", snap2)
    return plan2


def credit_fx(st, snap):
    p = st.get("fx_pending")
    b = st["book"]
    if not p:
        return 0.0
    free_usd = snap["acct_usd"] - b["usd"] - b["vault_usd"] - b.get("tax_usd", 0.0)
    got = min(p["usd"], max(0.0, free_usd), max(0.0, b["krw"]) / p["rate"])
    got = math.floor(got * 100) / 100
    if got > 0:
        b["usd"] += got
        b["krw"] -= got * p["rate"]
        log("fx_credit", usd=got, won=got * p["rate"], rate=p["rate"], requested=p["usd"], note="계좌에 생긴 달러를 도비 장부로 옮김")
    else:
        log("fx_missing", requested=p["usd"], account_usd=snap["acct_usd"], note="환전된 달러가 없어 미국 매수를 건너뜀")
    st["fx_pending"] = None
    return got


def run_us(t, st, acct, day, rebalance):
    cyc = st["current"]
    weights, rebalance = target_weights(st, acct.markets, "us", rebalance)
    snap = acct.snapshot()
    credit_fx(st, snap)
    snap = acct.snapshot()
    plan = make_plan(weights, snap, acct.markets, rebalance, us_only=True)
    log("day_plan", cycle=cyc, day=day, phase="us", rebalance=rebalance, equity=snap["equity"], us=plan["us"])
    busy = open_order_symbols(t)
    us = [x for x in plan["us"] if x["code"] not in busy]
    for x in [x for x in us if x["side"] == "SELL"]:
        sq = float(t.get("/api/v1/sellable-quantity", account=True, symbol=x["code"])["sellableQuantity"]) if live_on() else x["cur_qty"]
        q = min(sq, x["qty"], st["book"]["qty"].get(x["code"], 0.0))
        q = math.floor(q * 1e6) / 1e6
        if q > 0:
            place(t, st, cyc, x, "US", snap["fx"], quantity=f"{q:.6f}".rstrip("0").rstrip("."))
    acct_usd = float(t.buying_power("USD")["cashBuyingPower"]) if live_on() else snap["acct_usd"]
    avail = max(0.0, min(st["book"]["usd"], acct_usd - st["book"]["vault_usd"] - st["book"].get("tax_usd", 0.0)) * 0.995)
    buys = [x for x in us if x["side"] == "BUY"]
    want = sum(x["usd"] for x in buys)
    scale = min(1.0, avail / want) if want > 0 else 0.0
    if buys and scale < 1.0:
        log("scale_us", cycle=cyc, usd=avail, want=want, scale=scale, reason="달러 부족: 있는 만큼만 비율대로 삽니다")
    for x in buys:
        amt = math.floor(x["usd"] * scale * 100) / 100
        if amt < MIN_US_USD:
            log("skip", cycle=cyc, code=x["code"], reason="1달러 미만", usd=amt)
            continue
        place(t, st, cyc, x, "US", snap["fx"], amount=f"{amt:.2f}")
    save_state(st)
    mark(st, acct, "us")


def in_window(n, a, b):
    m = n.hour * 60 + n.minute
    lo, hi = a[0] * 60 + a[1], b[0] * 60 + b[1]
    return lo <= m <= hi if lo <= hi else (m >= lo or m <= hi)


def ny_minutes(n):
    ny = n.astimezone(NY)
    return ny.weekday(), ny.hour * 60 + ny.minute


def us_window(n):
    wd, m = ny_minutes(n)
    return wd < 5 and US_OPEN_NY[0] * 60 + US_OPEN_NY[1] <= m <= US_LAST_NY[0] * 60 + US_LAST_NY[1]


def us_mark_window(n):
    wd, m = ny_minutes(n)
    return wd < 5 and 9 * 60 + 20 <= m <= 16 * 60 + 5


def us_open_kst(day):
    y, mo, d = map(int, day.split("-"))
    return datetime(y, mo, d, US_OPEN_NY[0], US_OPEN_NY[1], tzinfo=NY).astimezone(KST).strftime("%H:%M")


def near_open(n):
    wd, m = ny_minutes(n)
    return in_window(n, (9, 5), (9, 8)) or (wd < 5 and 9 * 60 + 29 <= m <= 9 * 60 + 32)


def to_kst(ts):
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(KST)
    except Exception:
        return None


def kr_open_today(t):
    w = to_kst(t.prices(["069500"])[0].get("timestamp"))
    n = now()
    return w is not None and w.date() == n.date() and w.hour * 60 + w.minute >= 9 * 60


def us_open_now(t):
    w = to_kst(t.prices(["SPY"])[0].get("timestamp"))
    return w is not None and timedelta(0) <= (now() - w) < timedelta(minutes=15)


def candidates_extra(st):
    return sorted(set(st.get("extra", [])) | set(listfile("candidates_extra.txt")) | set(listfile("adopt.txt")) | set(st.get("book", {}).get("qty", {})))


def prepare(t, st, today):
    extra = candidates_extra(st)
    inst, groups, fx_now, fetched = build_market(t, extra)
    D = max(d for d in inst["069500"]["days"] if d < today)
    cyc = st.get("current")
    p = dict(day=today, D=D, fresh=False)
    if not cyc or len(kr_sessions_after(inst, st["cycles"][cyc]["decided_on"])) >= CYCLE:
        w, why, markets, m = decide(inst, groups, D, extra)
        p.update(fresh=True, weights=w, why=why, markets=markets, fetched=fetched, fx_mid=fx_now["midRate"])
    st["prepared"] = p
    log("prepared", day=today, decided_on=D, fresh=p["fresh"])
    save_state(st)
    return p


def commit(st, p, today):
    if not p["fresh"]:
        return False
    st.setdefault("cycles", {})[today] = dict(decided_on=p["D"], weights=p["weights"], markets=p["markets"], created=now().isoformat(timespec="seconds"))
    st["current"] = today
    st["markets"] = p["markets"]
    st["rebalance_for"] = today
    log("plan", cycle=today, decided_on=p["D"], fetched_at=p["fetched"], fx_mid=p["fx_mid"], weights=p["weights"], cash_weight=p["why"].get("cash_weight"), rationale=p["why"])
    return True


def once_amount(day):
    p = os.path.join(RUNS, "deposit_once.txt")
    if not os.path.exists(p):
        return None
    try:
        v = float(open(p, encoding="utf-8").read().strip().replace(",", ""))
    except (OSError, ValueError):
        log("deposit_once_bad", day=day, note="deposit_once.txt 금액을 읽지 못해 평소대로 1만원만 꺼냄")
        return None
    os.replace(p, f"{p}.{day}.done")
    return v


def deposit(st, day, acct_krw):
    b = st["book"]
    free = acct_krw - b["krw"] - b["vault_krw"]
    once = once_amount(day)
    want = max(DAILY_DEPOSIT, once) if once else DAILY_DEPOSIT
    amt = float(int(max(0.0, min(want, free))))
    b["krw"] += amt
    b["deposits"] += amt
    b["contributed"] += amt
    if amt < want:
        log("deposit_short", day=day, amount=amt, wanted=want, free_krw=free, note="계좌에 남은 원화가 모자라 있는 만큼만 꺼냄")
    log("deposit", day=day, amount=amt, krw=b["krw"], deposits=b["deposits"], once=once)
    return amt


def step(t, st, n, marks_state):
    today = n.strftime("%Y-%m-%d")
    if st.get("book", {}).get("pending"):
        reconcile(t, st, float(t.fx("USD", "KRW")["midRate"]))
    settle_tax(st)
    if n.weekday() < 5 and in_window(n, KR_PREP, KR_LAST) and st.get("checked_day") != today and (st.get("prepared") or {}).get("day") != today:
        prepare(t, st, today)
    if n.weekday() < 5 and in_window(n, KR_OPEN, KR_LAST) and st.get("checked_day") != today and kr_open_today(t):
        if "book" not in st:
            st["book"] = new_book(t)
            st["extra"] = sorted(set(st.get("extra", [])) | set(st["book"]["qty"]))
            log("vault", profit=0.0, paid=0.0, vault=0.0, principal=st["book"]["contributed"], note="도비 장부 시작")
        p = st.get("prepared")
        if not p or p.get("day") != today:
            p = prepare(t, st, today)
        commit(st, p, today)
        deposit(st, today, float(t.buying_power("KRW")["cashBuyingPower"]))
        st["checked_day"] = today
        save_state(st)
    cyc = st.get("current")
    if cyc and n.weekday() < 5 and in_window(n, KR_OPEN, KR_LAST) and st.get("checked_day") == today and st.get("kr_done_for") != today:
        markets = st["markets"]
        acct = Account(t, st, list(markets), markets)
        snap = acct.snapshot()
        if snap["equity"] > MAX_EQUITY:
            log("refuse", reason=f"도비 자산 {snap['equity']:,.0f}원이 상한 {MAX_EQUITY:,.0f}원을 넘음", equity=snap["equity"])
            return "exit"
        run_kr(t, st, acct, today, st.get("rebalance_for") == today)
        st["kr_done_for"] = today
        save_state(st)
    day = st.get("checked_day")
    if day and us_window(n) and st.get("kr_done_for") == day and st.get("us_done_for") != day and us_open_now(t):
        markets = st["markets"]
        acct = Account(t, st, list(markets), markets)
        run_us(t, st, acct, day, st.get("rebalance_for") == day)
        st["us_done_for"] = day
        save_state(st)
    if "book" in st and time.time() - marks_state.get("last", 0.0) > 600 and (in_window(n, (9, 0), (15, 35)) or us_mark_window(n)) and st.get("markets"):
        acct = Account(t, st, list(st["markets"]), st["markets"])
        mark(st, acct, "tick")
        save_state(st)
        marks_state["last"] = time.time()
    return None


def loop():
    from toss import Toss
    t = Toss()
    t.account = t.accounts()[0]["accountSeq"]
    st = load_state()
    log("start", max_equity=MAX_EQUITY, daily_deposit=DAILY_DEPOSIT, seed=SEED, adopt=listfile("adopt.txt"), state_cycle=st.get("current"))
    ms = {}
    while True:
        if os.path.exists(STOP):
            log("stop", reason="stop file")
            return
        try:
            if step(t, st, now(), ms) == "exit":
                log("stop", reason="refused")
                return
        except Exception as e:
            log("error", error=type(e).__name__, where=traceback.extract_tb(e.__traceback__)[-1].name, detail=str(getattr(e, "code", "")))
            time.sleep(60)
        time.sleep(5 if near_open(now()) else 30)


def plan_only():
    from toss import Toss
    t = Toss()
    t.account = t.accounts()[0]["accountSeq"]
    st = load_state()
    first = "book" not in st
    if first:
        st["book"] = new_book(t)
        free = float(t.buying_power("KRW")["cashBuyingPower"])
        st["book"]["krw"] += float(int(max(0.0, min(DAILY_DEPOSIT, free))))
    extra = candidates_extra(st)
    inst, groups, fx_now, fetched = build_market(t, extra)
    D = max(d for d in inst["069500"]["days"] if d < now().strftime("%Y-%m-%d"))
    w, why, markets, m = decide(inst, groups, D, extra)
    acct = Account(t, st, list(markets), markets)
    snap = acct.snapshot()
    plan = make_plan(w, snap, markets, True)
    print(f"결정 기준일 {D}, 학습 {why['trained_rows']}개 구간, 현금 {100 * why['cash_weight']:.0f}%, 범위 밖 후보 {why['outside']}")
    for c, x in sorted(w.items(), key=lambda kv: -kv[1]):
        print(f"  {c:7s} {100 * x:5.1f}%")
    for x in why["candidates"]:
        if x.get("outside"):
            print(f"  범위 밖 {x['code']}: 점수 {x['score']:.2f}, {x['rank']}위, 비율 {100 * x['weight']:.1f}%")
    print(f"도비 장부{' (첫날 가정)' if first else ''}: 자산 {snap['equity']:,.0f}원 = 원화 {snap['krw']:,.0f} + 달러 ${snap['usd']:,.2f} + 보유 {sum(snap['value'].values()):,.0f}, 금고 {snap['vault']:,.0f}원, 환율 {snap['fx']:,.1f}")
    for x in plan["kr"]:
        print(f"  국내 {x['side']} {x['code']} {x['qty']}주 @ {x['px']:,.0f}")
    for x in plan["us"]:
        print(f"  미국 {x['side']} {x['code']} ${x['usd']:,.2f} @ ${x['px']:,.2f}")
    print(f"환전 필요 ${plan['need_usd']:,.2f}, 못 쓰고 남는 돈 {plan['idle_krw']:,.0f}원")
    log("plan_preview", decided_on=D, weights=w, plan=plan, equity=snap["equity"], first_day=first)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if cmd == "plan":
        plan_only()
    elif cmd == "run":
        loop()
