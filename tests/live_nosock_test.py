import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
tmp = tempfile.mkdtemp(prefix="nosock_test_")
os.environ["NOSOCK_LOG"] = os.path.join(tmp, "log.jsonl")
os.environ["NOSOCK_STATE"] = os.path.join(tmp, "state.json")
os.environ["NOSOCK_SCENARIO_LOG"] = os.path.join(tmp, "scen_default.jsonl")
os.environ.pop("NOSOCK_SEED", None)
import live_nosock as L

L.RUNS = tmp
L.RESEND_FOR = 0
open(os.path.join(tmp, "adopt.txt"), "w").write("FIVN\n")
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  {detail}"))


def rows():
    return [json.loads(x) for x in open(os.environ["NOSOCK_LOG"], encoding="utf-8")]


class FakeToss:
    def __init__(self, holdings, px, krw, usd, fx=1400.0):
        self.h = dict(holdings)
        self.px = dict(px)
        self.krw, self.usd, self.rate = krw, usd, fx
        self.placed = []
        self.n = 0
        self.last = {}
        self.reject = {}

    def holdings(self):
        return dict(items=[dict(symbol=s, quantity=str(q), averagePurchasePrice="36.42" if s == "FIVN" else "1") for s, q in self.h.items() if q > 0])

    def prices(self, syms):
        return [dict(symbol=s, lastPrice=str(self.px[s]), timestamp="2026-11-09T09:05:00+09:00") for s in syms if s in self.px]

    def buying_power(self, cur):
        return dict(cashBuyingPower=str(self.krw if cur == "KRW" else self.usd))

    def fx(self, a, b):
        return dict(midRate=str(self.rate), rate=str(self.rate * 1.01))

    def orders(self, status):
        return dict(orders=[])

    def get(self, path, account=False, symbol=None):
        return dict(sellableQuantity=str(self.h.get(symbol, 0)))

    def place(self, symbol, side, order_type, client_order_id=None, quantity=None, amount=None):
        if symbol in self.reject:
            raise self.reject[symbol]
        self.n += 1
        p = self.px[symbol]
        kr = symbol[:1].isdigit()
        if amount is not None:
            q = float(amount) / p
            gross = float(amount)
        else:
            q = float(quantity)
            gross = q * p
        fee = gross * (0.00015 if kr else 0.001)
        if side == "BUY":
            if kr:
                self.krw -= gross + fee
            else:
                self.usd -= gross + fee
        else:
            if kr:
                self.krw += gross - fee
            else:
                self.usd += gross - fee
        self.h[symbol] = self.h.get(symbol, 0) + (q if side == "BUY" else -q)
        self.placed.append(dict(symbol=symbol, side=side, quantity=quantity, amount=amount))
        oid = f"o{self.n}"
        self.last[oid] = dict(status="FILLED", execution=dict(filledQuantity=str(q), averageFilledPrice=str(p), filledAmount=str(gross), commission=str(fee), tax="0"))
        return dict(orderId=oid)

    def order(self, oid):
        return self.last[oid]


markets = {"069500": "KR", "114800": "KR", "153130": "KR", "SPY": "US", "GLD": "US", "SMH": "US", "FIVN": "US"}
px = {"069500": 79000.0, "114800": 6000.0, "153130": 110000.0, "SPY": 600.0, "GLD": 250.0, "SMH": 300.0, "FIVN": 36.0}

t = FakeToss({"FIVN": 0.22, "005930": 10}, px, krw=3_000_000.0, usd=500.0)
b = L.new_book(t)
check("book starts from seed, not account cash", b["krw"] == 0.0 and b["usd"] == 0.0, b)
check("adopted FIVN into book", abs(b["qty"].get("FIVN", 0) - 0.22) < 1e-9 and abs(b["cost"]["FIVN"] - 0.22 * 36.42) < 1e-9, b)
check("owner's other holding not adopted", "005930" not in b["qty"])
st = dict(book=b)
st["book"]["krw"] += 10_000
acct = L.Account(t, st, list(markets), markets)
snap = acct.snapshot()
check("bot equity = book only (10,000 + FIVN)", abs(snap["equity"] - (10_000 + 0.22 * 36 * 1400)) < 1e-6, snap["equity"])
check("account cash not in bot equity", snap["equity"] < 30_000)

w = {"SMH": 0.30, "SPY": 0.25, "069500": 0.20, "114800": 0.15, "FIVN": 0.02}
plan = L.make_plan(w, snap, markets, rebalance=True)
us = {(x["code"], x["side"]): x for x in plan["us"]}
kr = {(x["code"], x["side"]): x for x in plan["kr"]}
check("rebalance trims FIVN to target", ("FIVN", "SELL") in us and not us[("FIVN", "SELL")]["full_exit"], plan["us"])
check("KODEX200 lot too big for small book -> not bought", ("069500", "BUY") not in kr, plan["kr"])
check("cheap KR lot (inverse 6,000) bought", ("114800", "BUY") in kr and kr[("114800", "BUY")]["qty"] == 1, plan["kr"])
spent = sum(x["qty"] * x["px"] * 1.003 for x in plan["kr"] if x["side"] == "BUY") + sum(x["usd"] for x in plan["us"] if x["side"] == "BUY") * 1400 - sum(x["usd"] for x in plan["us"] if x["side"] == "SELL") * 1400
check("plan never spends more than bot cash", spent <= 10_000 + 1e-6, spent)
check("US buys at least $1 each", all(x["usd"] >= 1.0 for x in plan["us"] if x["side"] == "BUY"), plan["us"])
plan_nr = L.make_plan(w, snap, markets, rebalance=False)
check("non-rebalance day: big overweight (FIVN 52% vs 2%) sold the same day", any(x["code"] == "FIVN" and x["side"] == "SELL" for x in plan_nr["us"]), plan_nr["us"])
snap_d = dict(snap, held={"SMH": 0.023102}, px=dict(px), value={"SMH": 0.023102 * 300 * 1400}, usd=0.0, krw=297.0, equity=10_000.0)
pd_nr = L.make_plan({"SMH": 0.97}, snap_d, markets, rebalance=False)
pd_r = L.make_plan({"SMH": 0.97}, snap_d, markets, rebalance=True)
check("non-rebalance day: small drift (1%p over target) kept", not any(x["side"] == "SELL" for x in pd_nr["us"]), pd_nr["us"])
check("rebalance day: same small drift trimmed if at least $1", any(x["side"] == "SELL" for x in pd_r["us"]) == (0.023102 * 300 * 1400 - 0.97 * 9900 >= 1400), pd_r["us"])
w2 = {"SMH": 0.5, "SPY": 0.48}
plan_x = L.make_plan(w2, snap, markets, rebalance=True)
check("weight 0 -> full exit", any(x["code"] == "FIVN" and x["full_exit"] for x in plan_x["us"]), plan_x["us"])

os.environ.pop("DOBBY_LIVE", None)
st = dict(book=L.new_book(t), current="c1", cycles={"c1": dict(weights=w)})
st["book"]["krw"] += 10_000
acct = L.Account(t, st, list(markets), markets)
L.run_kr(t, st, acct, "2026-11-09", True)
check("dry run places nothing", t.placed == [], t.placed)
check("dry run logs dry_order", any(r["kind"] == "dry_order" for r in rows()))
check("account already holds enough USD -> no fx request, fx_ready logged", not any(r["kind"] == "fx_request" for r in rows()) and any(r["kind"] == "fx_ready" and 0 < r["need_usd"] < 10 for r in rows()), [r for r in rows() if r["kind"] in ("fx_request", "fx_ready")])
check("fx pending kept for crediting at US open", st.get("fx_pending") and st["fx_pending"]["usd"] > 0, st.get("fx_pending"))
need0 = None
for acct_usd, label in ((0.0, "no USD in account -> request full need"), (None, "partial USD -> request only the shortfall")):
    acct_usd = round(need0 / 2, 2) if acct_usd is None else acct_usd
    open(os.environ["NOSOCK_LOG"], "w").close()
    tf = FakeToss({"FIVN": 0.22}, px, krw=3_000_000.0, usd=acct_usd)
    sf = dict(book=L.new_book(tf), current="c1", cycles={"c1": dict(weights=w)})
    sf["book"]["krw"] += 10_000
    L.run_kr(tf, sf, L.Account(tf, sf, list(markets), markets), "2026-11-09", True)
    fr = [r for r in rows() if r["kind"] == "fx_request"]
    ok = len(fr) == 1 and abs(fr[0]["amount_usd"] - max(0.0, fr[0]["need_usd"] - acct_usd)) < 0.011 and fr[0]["amount_won"] >= fr[0]["amount_usd"] * 1400 and "모자랍니다" in fr[0]["body"]
    check(label, ok, (acct_usd, fr))
    need0 = fr[0]["need_usd"] if fr else 0.0

os.environ["DOBBY_LIVE"] = "1"
t2 = FakeToss({"FIVN": 0.22, "005930": 10}, px, krw=3_000_000.0, usd=500.0)
st2 = dict(book=L.new_book(t2), current="c1", cycles={"c1": dict(weights=w)})
st2["book"]["krw"] += 10_000
acct2 = L.Account(t2, st2, list(markets), markets)
L.run_kr(t2, st2, acct2, "2026-11-09", True)
kr_orders = [p for p in t2.placed]
check("live KR: buys only bot money", sum(float(p["quantity"]) * px[p["symbol"]] for p in kr_orders if p["side"] == "BUY") <= 10_000, kr_orders)
check("live KR: owner 005930 untouched", t2.h["005930"] == 10)
check("book KRW decreased by fill + fee", st2["book"]["krw"] < 10_000 and st2["book"]["qty"].get("114800") == 1, st2["book"])
check("fx pending recorded", st2.get("fx_pending") and st2["fx_pending"]["usd"] > 0, st2.get("fx_pending"))
need = st2["fx_pending"]["usd"]
L.run_us(t2, st2, acct2, "2026-11-09", True)
check("US: fx credited from account usd", any(r["kind"] == "fx_credit" for r in rows()))
us_orders = [p for p in t2.placed if not p["symbol"].isdigit()]
check("US: FIVN trimmed", any(p["symbol"] == "FIVN" and p["side"] == "SELL" for p in us_orders), us_orders)
spent_us = sum(float(p["amount"]) for p in us_orders if p["side"] == "BUY")
check("US: buys within bot usd (fx credit + FIVN proceeds)", spent_us <= need + 0.22 * 36 + 0.01, (spent_us, need))
check("US: owner usd mostly untouched", t2.usd > 480, t2.usd)
check("book usd never negative", st2["book"]["usd"] > -0.01, st2["book"]["usd"])
check("vault row logged on FIVN loss sale, nothing paid", any(r["kind"] == "vault" and r.get("code") == "FIVN" and r["paid"] == 0 for r in rows()))

st3 = dict(book=dict(krw=0.0, usd=0.0, qty={"SMH": 1.0}, cost={"SMH": 200.0}, vault_krw=0.0, vault_usd=0.0, realized=0.0, hwm=0.0, deposits=0.0, contributed=0.0, pending={}))
L.apply_fill(st3, "SMH", "SELL", "US", 1.0, 300.0, 0.3, 1400.0)
gain = 300.0 - 0.3 - 200.0
check("vault: half of realized gain in USD", abs(st3["book"]["vault_usd"] - gain / 2) < 1e-9 and abs(st3["book"]["usd"] - gain / 2 - 200.0) < 1e-9, st3["book"])
st3["book"]["qty"]["SMH"] = 1.0
st3["book"]["cost"]["SMH"] = 300.0
L.apply_fill(st3, "SMH", "SELL", "US", 1.0, 250.0, 0.0, 1400.0)
v1 = st3["book"]["vault_usd"]
st3["book"]["qty"]["SMH"] = 1.0
st3["book"]["cost"]["SMH"] = 250.0
L.apply_fill(st3, "SMH", "SELL", "US", 1.0, 290.0, 0.0, 1400.0)
check("vault: loss first eats profit, pays only above high-water mark", abs(st3["book"]["vault_usd"] - v1) < 1e-9, (v1, st3["book"]["vault_usd"]))
snapv = dict(held={}, px=px, krw=100_000.0, usd=0.0, fx=1400.0, value={}, equity=100_000.0, vault=50_000.0)
pv = L.make_plan({"114800": 1.0}, snapv, markets, True)
check("vault not in bot equity -> not invested", sum(x["qty"] * x["px"] for x in pv["kr"]) <= 100_000, pv["kr"])

t4 = FakeToss({}, px, krw=1_000_000.0, usd=0.0)
st4 = dict(book=dict(krw=0.0, usd=0.0, qty={"SPY": 1.0}, cost={"SPY": 600.0}, vault_krw=0.0, vault_usd=0.0, realized=0.0, hwm=0.0, deposits=0.0, contributed=0.0, pending={}))
L.Account(t4, st4, list(markets), markets).snapshot()
check("book clamped when owner sold shares", "SPY" not in st4["book"]["qty"], st4["book"])

sd = dict(book=dict(krw=5_000.0, usd=0.0, qty={}, cost={}, vault_krw=3_000.0, vault_usd=0.0, realized=0.0, hwm=0.0, deposits=0.0, contributed=0.0, pending={}))
check("deposit: full 10,000 when account has free cash", L.deposit(sd, "d1", 100_000.0) == 10_000 and sd["book"]["krw"] == 15_000)
check("deposit: only what is free (account 22,000 - bot 15,000 - vault 3,000)", L.deposit(sd, "d2", 22_000.0) == 4_000 and sd["book"]["krw"] == 19_000)
check("deposit: nothing when no free cash", L.deposit(sd, "d3", 20_000.0) == 0 and sd["book"]["krw"] == 19_000)
check("deposit_short logged", sum(r["kind"] == "deposit_short" for r in rows()) == 2)
scen_log = os.path.join(tmp, "scen.jsonl")
L.SCEN_LOG = scen_log
base_w = {"SMH": 0.3, "SPY": 0.3, "114800": 0.1, "GLD": 0.1}
mk_mk = dict(markets)
w_none, i_none = L.effective_weights(base_w, mk_mk)
check("scenario: no desk result -> model weights", w_none == base_w and i_none is None)
row = dict(ts=L.now().isoformat(timespec="seconds"), kind="scenario", opt_ok=True, trigger="urgent:test", issue_ids=["i1"], probabilities=[0.7, 0.3],
           scenarios=[dict(name="down", probability=0.7, returns_pct={"SMH": -12.0, "SPY": -5.0, "GLD": 6.0, "114800": 5.0}), dict(name="up", probability=0.3, returns_pct={"SMH": 8.0, "SPY": 3.0, "GLD": -1.0, "114800": -3.0})])
open(scen_log, "w", encoding="utf-8").write(json.dumps(row) + chr(10))
w_s, i_s = L.effective_weights(base_w, mk_mk)
up = {c: w_s.get(c, 0) - base_w.get(c, 0) for c in set(w_s) | set(base_w)}
check("scenario: fresh desk result tilts weights", i_s is not None and i_s["issue_ids"] == ["i1"] and w_s.get("SMH", 0) < base_w["SMH"], (w_s, i_s))
check("scenario: per-name increase <= 15%p", max(up.values()) <= 0.15 + 1e-6, up)
check("scenario: total moved <= 30%", sum(v for v in up.values() if v > 0) <= 0.30 + 1e-6, up)
check("scenario: no leverage", sum(w_s.values()) <= 1 + 1e-6, sum(w_s.values()))
import scenario_desk as SDk
SDk.RUNS = tmp
open(os.path.join(tmp, "scenario_cap.txt"), "w").write("0.175")
w_c, i_c = L.effective_weights(base_w, mk_mk)
up_c = {c: w_c.get(c, 0) - base_w.get(c, 0) for c in set(w_c) | set(base_w)}
check("scenario: cap file 17.5% respected", i_c["cap"] == 0.175 and sum(v for v in up_c.values() if v > 0) <= 0.175 + 1e-6, (i_c.get("cap"), up_c))
os.remove(os.path.join(tmp, "scenario_cap.txt"))
import numpy as npt
cs_t = ["069500", "114800", "SMH"]
p_t, R_t = SDk.validate(dict(scenarios=[dict(probability=1.0, returns_pct={"069500": -10.0, "114800": 25.0, "SMH": -5.0})]), cs_t)
check("inverse: AI number replaced by -underlying - drag", abs(R_t[0, 1] - (0.10 - SDk.INV_DRAG_20D)) < 1e-9, R_t)
w_n = SDk.net_inverse(npt.array([0.05, 0.12, 0.1]), cs_t)
check("inverse: long and inverse of same index are netted", abs(w_n[0]) < 1e-12 and abs(w_n[1] - 0.07) < 1e-12, w_n)
w_h, i_h = L.effective_weights({"069500": 0.3, "114800": 0.0, "SMH": 0.3}, {"069500": "KR", "114800": "KR", "SMH": "US"})
check("inverse: never holds index long and its inverse together", not (w_h.get("069500", 0) > 0.002 and w_h.get("114800", 0) > 0.002), w_h)
cs_v = ["SMH", "SPY", "153130", "GLD"]
w0_v = npt.array([0.3, 0.3, 0.0, 0.1])
p_v, R_v = SDk.validate(dict(scenarios=[dict(probability=0.7, returns_pct={"SMH": -15.0, "SPY": -6.0, "153130": 0.5, "GLD": 3.0}), dict(probability=0.3, returns_pct={"SMH": 6.0, "SPY": 3.0, "153130": 0.3, "GLD": -1.0})]), cs_v)
vols_v = {"SMH": 0.45, "SPY": 0.18, "153130": 0.01, "GLD": 0.16}
w_v, ok_v = SDk.optimize(w0_v, p_v, R_v, 0.175, 0.15, cs_v, vols_v)
up_v = npt.clip(w_v - w0_v, 0, None)
check("risk budget: low-vol asset may take more than 15%p", up_v[2] > 0.15 + 1e-6 or up_v.sum() <= 0.175 + 1e-6, (w_v, up_v))
check("risk budget: vol-weighted buys within budget", float(npt.array([vols_v[c] for c in cs_v]) @ up_v) <= 0.175 * SDk.SIGMA_REF + 1e-6, up_v)
check("risk budget: total moved <= cap", up_v.sum() <= 0.175 + 1e-6, up_v)
old = dict(row, ts=(L.now() - L.timedelta(hours=30)).isoformat(timespec="seconds"))
open(scen_log, "w", encoding="utf-8").write(json.dumps(old) + chr(10))
check("scenario: stale (>20h) desk result ignored", L.effective_weights(base_w, mk_mk)[1] is None)
open(scen_log, "w", encoding="utf-8").write(json.dumps(row) + chr(10))
st_r = dict(current="c1", cycles={"c1": dict(weights=base_w)}, last_target=dict(base_w))
w_t, reb = L.target_weights(st_r, mk_mk, "kr", False)
check("scenario: big target change -> same-day rebalance (sells allowed)", reb is True and any(json.loads(x)["kind"] == "scenario_rebalance" for x in open(os.environ["NOSOCK_LOG"], encoding="utf-8")))
w_t2, reb2 = L.target_weights(st_r, mk_mk, "us", False)
check("scenario: unchanged target -> no extra rebalance", reb2 is False)
open(scen_log, "w").close()
kd = lambda *a: L.datetime(*a, tzinfo=L.KST)
check("US window opens 23:30 KST in winter", L.us_window(kd(2026, 11, 9, 23, 30)) and not L.us_window(kd(2026, 11, 9, 23, 29)))
check("US window opens 22:30 KST in summer (DST)", L.us_window(kd(2026, 3, 10, 22, 30)) and not L.us_window(kd(2026, 3, 10, 22, 29)))
check("US window spans midnight, closed Saturday NY", L.us_window(kd(2026, 11, 10, 2, 0)) and not L.us_window(kd(2026, 11, 14, 23, 40)))
check("FX deadline text follows DST", L.us_open_kst("2026-11-09") == "23:30" and L.us_open_kst("2026-03-10") == "22:30")

L.time.sleep = lambda x: None
days = ["2026-11-0%d" % d for d in range(2, 7)] + ["2026-11-%02d" % d for d in range(9, 31) if L.datetime(2026, 11, d).weekday() < 5] + ["2026-12-%02d" % d for d in range(1, 32) if L.datetime(2026, 12, d).weekday() < 5]
calls = dict(build=0, decide=0, extra=None)


def fake_build_market(t, extra=()):
    calls["build"] += 1
    calls["extra"] = list(extra)
    today = L.now().strftime("%Y-%m-%d")
    return {"069500": dict(days=[d for d in days if d <= today])}, dict(U29=[c for c in markets if c != "FIVN"]), dict(midRate=1400.0), "x"


def fake_decide(inst, groups, D, extra=()):
    calls["decide"] += 1
    return {"SMH": 0.3, "SPY": 0.3, "114800": 0.2, "FIVN": 0.05}, dict(cash_weight=0.15, outside=list(extra)), markets, None


L.build_market = fake_build_market
L.decide = fake_decide
T = FakeToss({"FIVN": 0.22, "005930": 10}, px, krw=2_000_000.0, usd=300.0)
sim_now = [None]
L.now = lambda: sim_now[0]
L.kr_open_today = lambda t: True
L.us_open_now = lambda t: True
open(os.environ["NOSOCK_LOG"], "w").close()
st = {}
ms = {}


def at(y, mo, d, h, mi):
    sim_now[0] = L.datetime(y, mo, d, h, mi, tzinfo=L.KST)
    return L.step(T, st, sim_now[0], ms)


at(2026, 11, 9, 8, 30)
check("step: nothing before 08:40", calls["build"] == 0 and not T.placed)
L.kr_open_today = lambda t: False
at(2026, 11, 9, 8, 45)
check("step: plan computed before open, no orders", calls["decide"] == 1 and not T.placed and "current" not in st, (calls, st.get("current")))
at(2026, 11, 9, 9, 0)
check("step: market not open yet -> no commit", "current" not in st and not T.placed)
L.kr_open_today = lambda t: True
at(2026, 11, 9, 9, 3)
check("step: open but LP quote-free 09:00-09:05 -> no commit, no orders", "current" not in st and not T.placed)
at(2026, 11, 9, 9, 6)
check("step: at 09:06, prepared plan committed without recompute", calls["decide"] == 1 and calls["build"] == 1 and st["current"] == "2026-11-09" and "book" in st, calls)
check("step: FIVN passed to model as candidate", "FIVN" in (calls["extra"] or []), calls["extra"])
check("step: first deposit 10,000", st["book"]["deposits"] == 10_000)
kr1 = [p for p in T.placed if p["symbol"].isdigit()]
check("step: KR buys never exceed bot KRW", sum(float(p["quantity"]) * px[p["symbol"]] for p in kr1) <= 10_000, kr1)
n_before = len(T.placed)
at(2026, 11, 9, 9, 7)
at(2026, 11, 9, 13, 0)
check("step: no repeat KR orders same day", len(T.placed) == n_before and calls["build"] == 1, (len(T.placed), calls))
at(2026, 11, 9, 23, 29)
check("step: no US orders before 23:30", len(T.placed) == n_before)
at(2026, 11, 9, 23, 30)
us1 = [p for p in T.placed if not p["symbol"].isdigit()]
check("step: US orders right at 23:30 open", len(us1) > 0, T.placed)
n_after_us = len(T.placed)
at(2026, 11, 10, 0, 30)
check("step: no repeat US orders", len(T.placed) == n_after_us)
at(2026, 11, 10, 9, 6)
check("step: next day deposit, no new plan", calls["decide"] == 1 and st["book"]["deposits"] == 20_000, (calls, st["book"]["deposits"]))
check("step: next day tops up without selling", all(p["side"] == "BUY" for p in T.placed[n_after_us:]), T.placed[n_after_us:])
at(2026, 11, 10, 23, 40)
check("step: US top-up runs daily", len(T.placed) > n_after_us, len(T.placed))
k = 0
for d in days:
    if d <= "2026-11-10":
        continue
    y, mo, dd = map(int, d.split("-"))
    at(y, mo, dd, 9, 6)
    at(y, mo, dd, 23, 40)
    k += 1
    if calls["decide"] > 1:
        break
check("step: new plan after 20 KR sessions", calls["decide"] == 2 and st["current"] != "2026-11-09", (calls, st.get("current"), k))
check("step: owner's 005930 never traded", all(p["symbol"] != "005930" for p in T.placed))
check("step: owner's cash mostly untouched", T.krw > 2_000_000 - 22 * 10_000 - 1, T.krw)
T2 = FakeToss({}, px, krw=0.0, usd=0.0)
st5 = dict(book=dict(krw=0.0, usd=0.0, qty={}, cost={}, vault_krw=0.0, vault_usd=0.0, realized=0.0, hwm=0.0, deposits=0.0, contributed=0.0, pending={}))
check("empty account: no deposit, no orders", L.deposit(st5, "d", 0.0) == 0)
bk = st["book"]
_, eq = L.book_equity(bk, T.px, 1400.0, markets)
check("step: book cash >= 0", bk["krw"] > -1 and bk["usd"] > -0.01, (bk["krw"], bk["usd"]))
check("step: book matches account for bot symbols", all(T.h.get(s, 0) + 1e-6 >= q for s, q in bk["qty"].items()), (bk["qty"], T.h))
from datetime import datetime as _dt

real_now = L.now
L.now = lambda: _dt(2026, 11, 9, 23, 30, tzinfo=L.KST)
stt = dict(book=dict(krw=0.0, usd=20_000.0, qty={}, cost={}, vault_krw=0.0, vault_usd=0.0, realized=0.0, hwm=0.0, deposits=0.0, contributed=0.0, pending={}))
bt = stt["book"]
L.apply_fill(stt, "QQQ", "BUY", "US", 10.0, 5000.0, 0.0, 1300.0)
check("tax: KRW basis recorded at buy fx", abs(bt["cost_krw"]["QQQ"] - 6_500_000) < 1e-6, bt["cost_krw"])
L.apply_fill(stt, "QQQ", "SELL", "US", 10.0, 5000.0, 0.0, 1400.0)
check("tax: fx gain counts as taxable KRW gain", abs(bt["tax"]["2026"]["gain"] - 500_000) < 1e-6 and abs(bt["realized"] - 500_000) < 1e-6, bt["tax"])
check("tax: under deduction -> no reserve", bt["tax_usd"] == 0 and bt["tax"]["2026"]["accrued"] == 0, bt["tax"])
usd_before = bt["usd"]
L.apply_fill(stt, "QQQ", "BUY", "US", 10.0, 5000.0, 0.0, 1400.0)
L.apply_fill(stt, "QQQ", "SELL", "US", 10.0, 8000.0, 0.0, 1400.0)
acc = 0.22 * (4_700_000 - 2_500_000)
check("tax: 22% of year gain above 2.5M reserved", abs(bt["tax"]["2026"]["accrued"] - acc) < 1e-6 and abs(bt["tax_usd"] - acc / 1400) < 1e-6, bt["tax"])
check("tax: reserve leaves bot usd", abs(bt["usd"] - (usd_before - 5000 + 8000 - acc / 1400 - (bt["vault_usd"] - 500_000 * 0.5 / 1400))) < 1e-6, bt)
check("tax: realized and vault are after tax", abs(bt["realized"] - (4_700_000 - acc)) < 1e-6 and abs(bt["vault_usd"] * 1400 - 0.5 * (4_700_000 - acc)) < 1e-3, (bt["realized"], bt["vault_usd"] * 1400))
L.apply_fill(stt, "QQQ", "BUY", "US", 10.0, 8000.0, 0.0, 1400.0)
L.apply_fill(stt, "QQQ", "SELL", "US", 10.0, 6000.0, 0.0, 1400.0)
check("tax: later loss same year releases reserve", bt["tax"]["2026"]["accrued"] == 0 and abs(bt["tax_usd"]) < 1e-9 and abs(bt["tax"]["2026"]["gain"] - 1_900_000) < 1e-6, bt["tax"])
open(os.path.join(tmp, "us_tax_deduction.txt"), "w").write("0")
L.apply_fill(stt, "QQQ", "BUY", "US", 1.0, 100.0, 0.0, 1400.0)
L.apply_fill(stt, "QQQ", "SELL", "US", 1.0, 100.0, 0.0, 1400.0)
check("tax: deduction file overrides 2.5M (shared with other overseas gains)", abs(bt["tax"]["2026"]["accrued"] - 0.22 * 1_900_000) < 1e-6, bt["tax"])
os.remove(os.path.join(tmp, "us_tax_deduction.txt"))
L.apply_fill(stt, "QQQ", "BUY", "US", 1.0, 100.0, 0.0, 1400.0)
L.apply_fill(stt, "QQQ", "SELL", "US", 1.0, 100.0, 0.0, 1400.0)
check("tax: removing file restores 2.5M deduction", bt["tax"]["2026"]["accrued"] == 0 and abs(bt["tax_usd"]) < 1e-9, bt["tax"])
L.apply_fill(stt, "QQQ", "BUY", "US", 10.0, 5000.0, 0.0, 1400.0)
L.apply_fill(stt, "QQQ", "SELL", "US", 10.0, 9000.0, 0.0, 1400.0)
res = bt["tax_usd"]
sfx = dict(book=dict(krw=1_000_000.0, usd=0.0, qty={}, cost={}, vault_krw=0.0, vault_usd=0.0, tax_usd=res, realized=0.0, hwm=0.0, deposits=0.0, contributed=0.0, pending={}), fx_pending=dict(usd=50.0, rate=1400.0))
check("tax: reserve dollars never credited as fx conversion", res > 0 and L.credit_fx(sfx, dict(acct_usd=res)) == 0.0, res)
L.now = lambda: _dt(2027, 1, 1, 0, 30, tzinfo=L.KST)
stn = dict(book=dict(krw=0.0, usd=100.0, qty={"SPY": 1.0}, cost={"SPY": 100.0}, cost_krw={"SPY": 140_000.0}, vault_krw=0.0, vault_usd=0.0, realized=0.0, hwm=0.0, deposits=0.0, contributed=0.0, pending={}))
L.apply_fill(stn, "SPY", "SELL", "US", 1.0, 110.0, 0.0, 1400.0)
check("tax: year follows the New York trade date", list(stn["book"]["tax"]) == ["2026"], stn["book"]["tax"])
L.now = lambda: _dt(2027, 5, 31, 12, 0, tzinfo=L.KST)
open(os.path.join(tmp, "tax_paid.txt"), "w").write("2026\n")
L.settle_tax(stt)
check("tax: paid year releases reserve to owner", bt["tax"]["2026"].get("paid") and abs(bt["tax_usd"]) < 1e-9 and bt["tax"]["2026"]["usd"] == 0, bt["tax"])
L.settle_tax(stt)
check("tax: settle is idempotent", sum(r["kind"] == "us_tax_paid" for r in rows()) == 1)
os.remove(os.path.join(tmp, "tax_paid.txt"))
skr = dict(book=dict(krw=0.0, usd=0.0, qty={"114800": 2.0}, cost={"114800": 10_000.0}, vault_krw=0.0, vault_usd=0.0, realized=0.0, hwm=0.0, deposits=0.0, contributed=0.0, pending={}))
L.apply_fill(skr, "114800", "SELL", "KR", 2.0, 12_000.0, 0.0, 1400.0)
check("tax: KR sale leaves US tax untouched", skr["book"]["tax"] == {} and skr["book"]["tax_usd"] == 0 and abs(skr["book"]["realized"] - 2_000) < 1e-9, skr["book"])
L.now = real_now
once_p = os.path.join(tmp, "deposit_once.txt")
open(once_p, "w").write("30,000\n")
so = dict(book=dict(krw=0.0, usd=0.0, qty={}, cost={}, vault_krw=0.0, vault_usd=0.0, realized=0.0, hwm=0.0, deposits=0.0, contributed=0.0, pending={}))
check("deposit once: first day takes 30,000", L.deposit(so, "2026-11-09", 30_000.0) == 30_000 and so["book"]["krw"] == 30_000, so["book"])
check("deposit once: file consumed", not os.path.exists(once_p) and os.path.exists(once_p + ".2026-11-09.done"))
check("deposit once: next day back to 10,000", L.deposit(so, "2026-11-10", 50_000.0) == 10_000 and so["book"]["krw"] == 40_000, so["book"])
open(once_p, "w").write("30000")
short0 = sum(r["kind"] == "deposit_short" for r in rows())
check("deposit once: never more than free cash", L.deposit(so, "2026-11-11", 65_000.0) == 25_000 and sum(r["kind"] == "deposit_short" for r in rows()) == short0 + 1, so["book"])
open(once_p, "w").write("abc")
check("deposit once: unreadable file -> normal 10,000", L.deposit(so, "2026-11-12", 200_000.0) == 10_000 and any(r["kind"] == "deposit_once_bad" for r in rows()))
os.remove(once_p)
mk_h = {"114800": "KR", "SPY": "US", "GLD": "US"}
mk_h.update({f"U{i:02d}": "US" for i in range(26)})
def hsnap(held_q, krw):
    v = {"114800": held_q * 14300.0} if held_q else {}
    p = {"114800": 14300.0, "SPY": 600.0, "GLD": 250.0}
    p.update({f"U{i:02d}": 100.0 for i in range(26)})
    return dict(held={"114800": held_q} if held_q else {}, px=p, krw=krw, usd=0.0, fx=1360.0, value=v, equity=krw + sum(v.values()), vault=0.0)
def kr_side(plan, side):
    return sum(x["qty"] for x in plan["kr"] if x["side"] == side)
w_case = {"114800": 0.0374}
w_case.update({f"U{i:02d}": 0.036 for i in range(26)})
ph = L.make_plan(w_case, hsnap(0, 38_460.0), mk_h, True)
check("KR lot 10x target, cash available (11-09 bank case) -> not bought", kr_side(ph, "BUY") == 0, ph["kr"])
ph = L.make_plan({"114800": 0.20, "SPY": 0.15, "GLD": 0.15}, hsnap(0, 38_460.0), mk_h, True)
check("KR target 0.53 share -> 1 bought", kr_side(ph, "BUY") == 1, ph["kr"])
ph = L.make_plan({"114800": 0.17, "SPY": 0.15, "GLD": 0.15}, hsnap(0, 38_460.0), mk_h, True)
check("KR target 0.45 share -> not bought", kr_side(ph, "BUY") == 0, ph["kr"])
ph = L.make_plan({"114800": 0.40, "SPY": 0.1}, hsnap(0, 60_000.0), mk_h, True)
check("KR target 1.66 shares -> 2 bought", kr_side(ph, "BUY") == 2, ph["kr"])
w_case1 = dict(w_case)
ph = L.make_plan(w_case1, hsnap(1.0, 24_160.0), mk_h, True)
check("held 1 share vs target 0.1 share -> sold on rebalance", kr_side(ph, "SELL") == 1, ph["kr"])
ph = L.make_plan(w_case1, hsnap(1.0, 24_160.0), mk_h, False)
check("held 1 share (59% of book) vs target 0.1 share -> sold even on a non-rebalance day", kr_side(ph, "SELL") == 1, ph["kr"])
ph = L.make_plan({"114800": 0.22, "SPY": 0.5}, hsnap(1.0, 24_160.0), mk_h, True)
check("held 1 share vs target 0.59 share -> kept", kr_side(ph, "SELL") == 0, ph["kr"])
check("KR trades start after LP quote-free first 5 minutes", L.KR_OPEN >= (9, 5), L.KR_OPEN)
nov9 = [("USO", 2.28, 2.2845), ("SMH", 1.14, 1.1473), ("XLV", 1.05, 1.0546), ("XLE", 1.0, 1.0025), ("XLK", 1.0, 0.8453), ("XLP", 1.0, 0.7356), ("XLF", 1.0, 0.7145), ("EEM", 1.0, 0.6936), ("EFA", 1.0, 0.6817), ("GLD", 1.0, 0.6667)]
nb = [dict(code=c, side="BUY", usd=u, target_usd=tu) for c, u, tu in nov9]
fit = L.fit_us_buys(nb, 11.4624)
check("11-09 case: $0.008 short -> all 10 still bought", len(fit) == 10, fit)
check("11-09 case: every order >= $1 and total within dollars", all(a >= 1.0 for _, a in fit) and sum(a for _, a in fit) <= 11.4624, [(x["code"], a) for x, a in fit])
check("11-09 case: $1 orders untouched, cut taken from bigger ones", all(a == 1.0 for x, a in fit if x["usd"] == 1.0) and dict((x["code"], a) for x, a in fit)["USO"] < 2.28, [(x["code"], a) for x, a in fit])
fit = L.fit_us_buys(nb, 11.47)
check("enough dollars -> amounts unchanged", [a for _, a in sorted(fit, key=lambda z: z[0]["code"])] == [u for _, u in sorted([(c, u) for c, u, _ in nov9])], fit)
fit = L.fit_us_buys(nb, 5.0)
codes = {x["code"] for x, _ in fit}
check("$5 for $11.47 -> smallest targets dropped first, total within", "USO" in codes and "GLD" not in codes and sum(a for _, a in fit) <= 5.0 and all(a >= 1.0 for _, a in fit), [(x["code"], a) for x, a in fit])
check("under $1 -> nothing", L.fit_us_buys(nb, 0.99) == [])

import numpy as _np
import scenario_desk as _SD
real_eff = L.effective_weights


def desk_nets_only(b, m):
    cs = sorted(b)
    w = _SD.net_inverse(_np.array([b[c] for c in cs]), cs)
    return {c: float(x) for c, x in zip(cs, w) if x > 0}, dict(desk_ts="t")


L.effective_weights = desk_nets_only
w11 = {"069500": 0.0606, "229200": 0.0227, "091160": 0.1007, "091170": 0.0429, "114800": 0.0169, "153130": 0.0317, "SPY": 0.05, "SH": 0.0168}
st_o = dict(current="c1", cycles={"c1": dict(weights=dict(w11))}, kr_trend=dict(state=0.0))
wo, _ = L.target_weights(st_o, dict(markets, **{"091160": "KR", "091170": "KR", "153130": "KR", "229200": "KR", "SH": "US"}), "kr", False)
L.effective_weights = real_eff
check("model weights go to orders as the model set them (old kr_trend in state ignored, no hand-made inverse)", all(abs(wo.get(c, 0) - w11[c]) < 1e-9 for c in ("229200", "091160", "091170", "153130")) and "114800" not in wo, wo)
check("desk netting of KR index vs inverse still applies", abs(wo.get("069500", 0) - (0.0606 - 0.0169)) < 1e-9, wo)
check("desk netting of US index vs inverse still applies", abs(wo.get("SPY", 0) - (0.05 - 0.0168)) < 1e-9 and "SH" not in wo, wo)
check("no hand-made KR trend rule left in the runner", not any(hasattr(L, n) for n in ("kr_xform", "kr_trend_state", "KR_RISK")))
mk_k = dict(markets, **{"091170": "KR"})
snap_k = dict(held={"091170": 1.0}, px={"091170": 14120.0, "114800": 6000.0, "SPY": 600.0}, krw=9_000.0, usd=0.0, fx=1400.0, value={"091170": 14120.0}, equity=23_120.0, vault=0.0)
pk = L.make_plan({"091170": 0.0429, "SPY": 0.3, "114800": 0.0169}, snap_k, mk_k, rebalance=False)
check("bank share far over the model target is sold on a non-rebalance day (R1)", any(x["code"] == "091170" and x["side"] == "SELL" for x in pk["kr"]), pk["kr"])

import time as _t
L.MARK_FLAG = os.path.join(tmp, "mark_now.flag")
views = lambda: sum(1 for r in rows() if r["kind"] == "mark" and r.get("phase") == "view")
v0 = views()
at(2026, 11, 15, 3, 0)
check("Sunday night, no refresh request -> no balance check", views() == v0)
open(L.MARK_FLAG, "w").close()
at(2026, 11, 15, 3, 1)
check("refresh request -> balance checked at once, request cleared", views() == v0 + 1 and not os.path.exists(L.MARK_FLAG))
open(L.MARK_FLAG, "w").close()
t0 = _t.time()
L.nap(30)
check("wait loop wakes within a second on a refresh request", _t.time() - t0 < 1.5)
os.remove(L.MARK_FLAG)
t0 = _t.time()
L.nap(2)
check("wait loop sleeps its full time without a request", _t.time() - t0 >= 1.9)

L.FX_FLAG = os.path.join(tmp, "fx_check.flag")


def fx_book(usd, krw):
    return dict(krw=krw, usd=usd, qty={}, cost={}, cost_krw={}, vault_krw=0.0, vault_usd=0.0, tax={}, tax_usd=0.0, realized=0.0, hwm=0.0, deposits=0.0, contributed=0.0, pending={})


def fx_rows(kind):
    return [r for r in rows() if r["kind"] == kind]


tfx = FakeToss({}, px, krw=50_000.0, usd=17.43)
sx = dict(book=fx_book(1.4214, 28_004.0), markets={"SPY": "US"}, fx_pending=dict(day="2026-11-10", usd=17.09, rate=1346.8))
ax = L.Account(tfx, sx, ["SPY"], {"SPY": "US"})
n_cr, n_ck = len(fx_rows("fx_credit")), len(fx_rows("fx_check"))
L.check_fx(sx, ax)
ck = fx_rows("fx_check")[-1]
check("FX check (11-10 case): $16.00 found in the account moved to the book at once", abs(sx["book"]["usd"] - 17.4214) < 1e-9 and abs(sx["book"]["krw"] - (28_004.0 - 16.0 * 1346.8)) < 1e-6 and len(fx_rows("fx_credit")) == n_cr + 1, sx["book"])
check("FX check: shortfall $1.09 kept pending and told in won", sx["fx_pending"] and abs(sx["fx_pending"]["usd"] - 1.09) < 1e-9 and ck["got"] == 16.0 and abs(ck["left"] - 1.09) < 1e-9 and ck["won_left"] == 1500 and "더 바꾸면" in ck["body"], (sx.get("fx_pending"), ck))
tfx.usd += 1.09
L.check_fx(sx, ax)
ck = fx_rows("fx_check")[-1]
check("FX check: second press after topping up credits the rest and clears the request", sx["fx_pending"] is None and abs(sx["book"]["usd"] - 18.5114) < 1e-9 and "다 찼습니다" in ck["body"], (sx.get("fx_pending"), ck))
n_miss = len(fx_rows("fx_missing"))
check("FX: US open after early credit -> nothing more to credit, no 'missing' alarm", L.credit_fx(sx, ax.snapshot()) == 0.0 and len(fx_rows("fx_missing")) == n_miss)
sy = dict(book=fx_book(1.4214, 28_004.0), markets={"SPY": "US"}, fx_pending=dict(day="2026-11-10", usd=17.09, rate=1346.8))
ty = FakeToss({}, px, krw=50_000.0, usd=17.43)
ay = L.Account(ty, sy, ["SPY"], {"SPY": "US"})
L.check_fx(sy, ay)
L.credit_fx(sy, ay.snapshot())
check("FX: part credited early, nothing more by US open -> no 'missing' alarm, request closed, buys use what came", sy["fx_pending"] is None and len(fx_rows("fx_missing")) == n_miss and abs(sy["book"]["usd"] - 17.4214) < 1e-9, sy)
sz = dict(book=fx_book(1.4214, 28_004.0), markets={"SPY": "US"}, fx_pending=dict(day="2026-11-10", usd=5.0, rate=1400.0))
tz = FakeToss({}, px, krw=50_000.0, usd=1.4214)
az = L.Account(tz, sz, ["SPY"], {"SPY": "US"})
L.check_fx(sz, az)
ck = fx_rows("fx_check")[-1]
check("FX check before converting: nothing credited, request kept, owner told the amount", ck["got"] == 0.0 and sz["fx_pending"]["usd"] == 5.0 and ck["won_left"] == 7000 and "안 보입니다" in ck["body"], ck)
L.credit_fx(sz, az.snapshot())
check("FX: never converted -> US open logs 'missing' as before", len(fx_rows("fx_missing")) == n_miss + 1 and sz["fx_pending"] is None)
sn = dict(book=fx_book(1.4214, 28_004.0), markets={"SPY": "US"})
L.check_fx(sn, L.Account(tz, sn, ["SPY"], {"SPY": "US"}))
check("FX check with no open request -> says so, book untouched", "기다리는 환전 요청이 없습니다" in fx_rows("fx_check")[-1]["body"] and sn["book"]["usd"] == 1.4214)
n_ck = len(fx_rows("fx_check"))
open(L.FX_FLAG, "w").close()
at(2026, 11, 15, 3, 2)
check("'환전했어요' flag -> runner checks at once, flag cleared", len(fx_rows("fx_check")) == n_ck + 1 and not os.path.exists(L.FX_FLAG))
open(L.FX_FLAG, "w").close()
t0 = _t.time()
L.nap(30)
check("wait loop wakes within a second on '환전했어요'", _t.time() - t0 < 1.5)
os.remove(L.FX_FLAG)

sp = {"091160": 92580.0, "0167A0": 12460.0, "069500": 79000.0, "SPY": 600.0}
mk_s = {"091160": "KR", "069500": "KR", "SPY": "US"}
ws = {"091160": 0.1007, "069500": 0.0606, "SPY": 0.1}


def sub_snap(eq, held=None, value=None):
    value = value or {}
    return dict(held=held or {}, px=dict(sp), value=value, krw=eq - sum(value.values()), usd=0.0, fx=1400.0, equity=eq)


xs, su = L.kr_subs(ws, sub_snap(62_700.0))
check("sub: semis target 6,251 < half of 92,580 -> its weight goes to 0167A0, other weights and the total unchanged", "091160" not in xs and abs(xs["0167A0"] - 0.1007) < 1e-12 and xs["069500"] == 0.0606 and xs["SPY"] == 0.1 and su["091160"]["sub"] == "0167A0" and su["091160"]["target_krw"] == 6251, (xs, su))
check("sub: the model's own weights are not modified", ws == {"091160": 0.1007, "069500": 0.0606, "SPY": 0.1})
kb = {x["code"]: x["qty"] for x in L.make_plan(xs, sub_snap(62_700.0), mk_s, rebalance=True)["kr"] if x["side"] == "BUY"}
check("sub: tomorrow's case (book 62,700, target 6,251 >= half of 12,460) buys 1 share of 0167A0, none of 091160", kb.get("0167A0") == 1 and "091160" not in kb, kb)
x52, _ = L.kr_subs(ws, sub_snap(52_700.0))
kb = {x["code"] for x in L.make_plan(x52, sub_snap(52_700.0), mk_s, rebalance=True)["kr"] if x["side"] == "BUY"}
check("sub: today's case (book 52,700, target 5,254 < half of 12,460) -> mapped but nothing bought, same half-share rule", "0167A0" in x52 and "0167A0" not in kb and "091160" not in kb, kb)
xb, sb_ = L.kr_subs(ws, sub_snap(500_000.0))
check("sub: book 500,000 (target 49,846 >= half of 92,580) -> original 091160 used, no substitute", xb == ws and sb_ == {}, (xb, sb_))
xh, sh = L.kr_subs(ws, sub_snap(150_000.0, held={"091160": 1.0}, value={"091160": 92_580.0}))
check("sub: original already held -> keep the original even if its target is small (no back-and-forth)", xh == ws and sh == {}, (xh, sh))
xm, sm = L.kr_subs(ws, dict(sub_snap(62_700.0), px={c: v for c, v in sp.items() if c != "0167A0"}))
check("sub: no price for the substitute -> original kept", xm == ws and sm == {}, (xm, sm))
xw0, sw0 = L.kr_subs({"069500": 0.5, "SPY": 0.4}, sub_snap(62_700.0))
check("sub: model gives semis 0 -> nothing substituted", "0167A0" not in xw0 and sw0 == {}, (xw0, sw0))
s500 = sub_snap(500_000.0, held={"0167A0": 4.0}, value={"0167A0": 4 * 12_460.0})
xr, _ = L.kr_subs(ws, s500)
pr = L.make_plan(xr, s500, mk_s, rebalance=False)
kr_r = {(x["code"], x["side"]): x["qty"] for x in pr["kr"]}
check("sub: book grows past the original's half-share point -> sell all 0167A0, buy 091160", kr_r.get(("0167A0", "SELL")) == 4 and kr_r.get(("091160", "BUY")) == 1, pr["kr"])
ta = FakeToss({}, sp, krw=62_700.0, usd=0.0)
check("sub: account snapshot prices the substitute even before it is held", "0167A0" in L.Account(ta, dict(book=fx_book(0.0, 62_700.0)), list(mk_s), mk_s).snapshot()["px"])
prev_live, prev_scen = os.environ.get("DOBBY_LIVE"), L.SCEN_ON
os.environ["DOBBY_LIVE"] = "1"
L.SCEN_ON = False
tk_ = FakeToss({}, sp, krw=62_700.0, usd=0.0)
sk = dict(book=fx_book(0.0, 62_700.0), current="c1", cycles={"c1": dict(weights=dict(ws))})
L.run_kr(tk_, sk, L.Account(tk_, sk, list(mk_s), mk_s), "2026-11-11", False)
ks = [r for r in rows() if r["kind"] == "kr_substitute"][-1]
kr_placed = [p for p in tk_.placed if p["symbol"][:1].isdigit()]
check("sub: run_kr orders 1 share of 0167A0, logs the swap, book holds it, model target still says 091160", kr_placed == [dict(symbol="0167A0", side="BUY", quantity=1, amount=None)] and ks["subs"]["091160"]["sub"] == "0167A0" and sk["book"]["qty"].get("0167A0") == 1 and "091160" in sk["last_target"] and "0167A0" not in sk["last_target"], (tk_.placed, ks, sk["book"]["qty"]))
snap_k = L.Account(tk_, sk, list(mk_s), mk_s).snapshot()
check("sub: held 0167A0 valued in won in the bot equity", abs(snap_k["value"]["0167A0"] - 12_460.0) < 1e-6, snap_k["value"])
if prev_live is None:
    os.environ.pop("DOBBY_LIVE", None)
else:
    os.environ["DOBBY_LIVE"] = prev_live
L.SCEN_ON = prev_scen

from toss import ApiError
prev_live, prev_scen = os.environ.get("DOBBY_LIVE"), L.SCEN_ON
os.environ["DOBBY_LIVE"] = "1"
L.SCEN_ON = False
bp = {"SPY": 600.0, "GLD": 250.0, "SMH": 300.0, "UUP": 27.0}
bm = {c: "US" for c in bp}
bw = {c: 0.2 for c in bp}


def blk_run(day, reject=None, st=None, kr=False):
    tb = FakeToss({}, bp, krw=0.0, usd=20.0)
    tb.reject = reject or {}
    sb = st or dict(book=fx_book(20.0, 0.0), current="c1", cycles={"c1": dict(weights=dict(bw))})
    n0 = len(rows())
    (L.run_kr if kr else L.run_us)(tb, sb, L.Account(tb, sb, list(bm), bm), day, True)
    return tb, sb, rows()[n0:]


tb, sb, new = blk_run("2026-11-10", {"UUP": ApiError(422, "prerequisite-required", "해외 파생 ETF 위험 고지 동의가 필요합니다", "rq1")})
oe = [r for r in new if r["kind"] == "order_error"]
check("reject: Toss's reason text and request id are logged on the failed order", len(oe) == 1 and oe[0]["code"] == "UUP" and oe[0]["code_detail"] == "prerequisite-required" and "위험 고지" in oe[0]["message"] and oe[0]["request_id"] == "rq1", oe)
check("reject: other names filled, only UUP refused -> UUP blocked for 7 days", sorted(p["symbol"] for p in tb.placed) == ["GLD", "SMH", "SPY"] and sb["blocked"]["UUP"]["until"] == "2026-11-17" and any(r["kind"] == "blocked" and r["code"] == "UUP" for r in new), (tb.placed, sb.get("blocked")))
sb["book"]["usd"] += 20.0
tb2, sb, new = blk_run("2026-11-11", st=sb)
check("reject: next night UUP is not ordered, a skip row says why and until when", "UUP" not in [p["symbol"] for p in tb2.placed] and any(r["kind"] == "skip" and r["code"] == "UUP" and "2026-11-17" in r["reason"] and "위험 고지" in r["reason"] for r in new), (tb2.placed, [r for r in new if r["kind"] == "skip"]))
sk2 = dict(book=fx_book(0.0, 30_000.0), current="c1", cycles={"c1": dict(weights=dict(bw))}, blocked=dict(sb["blocked"]))
_, sk2, new = blk_run("2026-11-12", st=sk2, kr=True)
dp = [r for r in new if r["kind"] == "day_plan"][-1]
check("reject: KR session no longer asks to convert dollars for UUP", not any(x["code"] == "UUP" for x in dp["us"]), dp["us"])
tb3, sb, new = blk_run("2026-11-17", st=sb)
check("reject: after 7 days UUP is tried again", "UUP" in [p["symbol"] for p in tb3.placed], tb3.placed)
tb4, sb4, new = blk_run("2026-11-10", {c: ApiError(422, "http", "") for c in bp})
check("reject: every order refused (account/time problem) -> nothing blocked", not sb4.get("blocked") and not tb4.placed, sb4.get("blocked"))
tb5, sb5, new = blk_run("2026-11-10", {"UUP": ApiError(422, "insufficient-buying-power", "주문 가능 금액이 부족합니다")})
check("reject: not-enough-cash refusal is not a reason to block the name", not sb5.get("blocked"), sb5.get("blocked"))
tb6, sb6, new = blk_run("2026-11-10", {"UUP": ApiError(500, "http", "server error")})
check("reject: server error (not 422) does not block", not sb6.get("blocked"), sb6.get("blocked"))
held = dict(book=dict(fx_book(20.0, 0.0), qty={"UUP": 0.5}), blocked={"UUP": dict(day="2026-11-10", until="2026-11-17")})
check("reject: a blocked name already held keeps its target (never force-sold)", "UUP" in L.drop_blocked(held, bw, "2026-11-11")[0])
if prev_live is None:
    os.environ.pop("DOBBY_LIVE", None)
else:
    os.environ["DOBBY_LIVE"] = prev_live
L.SCEN_ON = prev_scen


class TokT:
    def __init__(self, exp, fail=False):
        self.tok, self.exp, self.fail, self.issued = "a", exp, fail, 0

    def _load_token(self):
        pass

    def _issue(self):
        if self.fail:
            raise RuntimeError("oauth down")
        self.issued += 1
        self.tok, self.exp = "b", clk[0] + 86400


clk = [1_000_000.0]
tk, ms = TokT(clk[0] + 10 * 3600), {}
check("token with 10h left -> not renewed", not L.keep_token(tk, ms, lambda: clk[0]) and tk.issued == 0)
tk.exp = clk[0] + 2 * 3600
check("token with 2h left -> renewed so the desk never finds it expired", L.keep_token(tk, ms, lambda: clk[0]) and tk.issued == 1 and tk.exp > clk[0] + 23 * 3600)
tk2, ms2 = TokT(clk[0] + 60, fail=True), {}
check("renew failure -> logged, not raised", not L.keep_token(tk2, ms2, lambda: clk[0]) and any(r["kind"] == "token_renew_error" for r in rows()))
clk[0] += 60
tk2.fail = False
check("after a failure, waits 10 min before retrying", not L.keep_token(tk2, ms2, lambda: clk[0]) and tk2.issued == 0)
clk[0] += 600
check("retries after 10 min", L.keep_token(tk2, ms2, lambda: clk[0]) and tk2.issued == 1)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
