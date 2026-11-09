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
        self.n += 1
        p = self.px[symbol]
        kr = symbol.isdigit()
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
check("non-rebalance day never sells", all(x["side"] == "BUY" for x in plan_nr["kr"] + plan_nr["us"]), plan_nr)
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
check("held 1 share -> kept on non-rebalance day", kr_side(ph, "SELL") == 0, ph["kr"])
ph = L.make_plan({"114800": 0.22, "SPY": 0.5}, hsnap(1.0, 24_160.0), mk_h, True)
check("held 1 share vs target 0.59 share -> kept", kr_side(ph, "SELL") == 0, ph["kr"])
check("KR trades start after LP quote-free first 5 minutes", L.KR_OPEN >= (9, 5), L.KR_OPEN)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
