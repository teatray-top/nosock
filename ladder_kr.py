import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "kr_etf", "kodex_daily_raw.json")
OUT = os.path.join(HERE, "runs", "ladder_kr_2026-11-06_v4.json")
SYM = "069500"
DEPOSIT, FEE, TICK = 10000.0, 0.00015, 5.0
MA_N, DIP, DIP_GAP = 200, -0.04, 20
LAST = "2026-11-05"
PERIODS = {"full": ("2003-08-01", LAST), "common": ("2024-04-01", LAST), "bull_2025": ("2025-01-02", "2025-12-30"), "crash_2026": ("2026-07-08", LAST)}


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def fee(x):
    return float(np.floor(x * FEE))


def load():
    r = json.load(open(RAW, encoding="utf-8"))[SYM]
    keep = [i for i, d in enumerate(r["days"]) if d <= LAST]
    days = [r["days"][i] for i in keep]
    O = np.array([r["O"][i] for i in keep], float)
    C = np.array([r["C"][i] for i in keep], float)
    return days, O, C


def signals(C):
    n = len(C)
    ma = np.full(n, np.nan)
    for i in range(MA_N - 1, n):
        ma[i] = C[i - MA_N + 1:i + 1].mean()
    above = C > ma
    r1 = np.full(n, np.nan)
    r1[1:] = C[1:] / C[:-1] - 1
    dip = r1 <= DIP
    dip_first = np.zeros(n, bool)
    for i in np.flatnonzero(dip):
        dip_first[i] = not dip[max(0, i - DIP_GAP):i].any()
    return dict(ma=ma, above=above, dip_first=dip_first)


def run(rung, days, O, C, sig, start, end):
    idx = [i for i, d in enumerate(days) if start <= d <= end]
    cash, qty, cost, deposits = 0.0, 0, 0.0, 0.0
    buys, sells, hist = 0, 0, []
    buy_until = -1
    last_i = idx[-1] if idx else -1
    for i in idx:
        deposits += DEPOSIT
        cash += DEPOSIT
        if i + 1 >= len(days) or i == last_i:
            hist.append((days[i], cash + qty * C[i], deposits, qty * C[i] / (cash + qty * C[i]) if cash + qty * C[i] > 0 else 0.0))
            break
        want_buy, want_sell = False, False
        ma_ok = np.isfinite(sig["ma"][i])
        if rung == "R0_cash":
            pass
        elif rung == "R1_buyhold":
            want_buy = True
        elif rung == "R2_buy_above_ma200":
            want_buy = bool(ma_ok and C[i] > sig["ma"][i])
        elif rung == "R3_ma200_in_out":
            if ma_ok:
                want_buy = bool(C[i] > sig["ma"][i])
                want_sell = bool(C[i] < sig["ma"][i])
        elif rung == "R4_buy_after_dip":
            if sig["dip_first"][i]:
                buy_until = i + DIP_GAP - 1
            want_buy = i <= buy_until
        o_next = O[i + 1]
        if want_sell and qty > 0:
            px = o_next - TICK
            proceeds = qty * px - fee(qty * px)
            cash += proceeds
            qty, cost = 0, 0.0
            sells += 1
        if want_buy:
            px = o_next + TICK
            q = int(cash // px)
            while q > 0 and q * px + fee(q * px) > cash:
                q -= 1
            if q >= 1:
                c = q * px + fee(q * px)
                cash -= c
                qty += q
                cost += c
                buys += 1
        value = cash + qty * C[i + 1]
        hist.append((days[i + 1], value, deposits, qty * C[i + 1] / value if value > 0 else 0.0))
    if not hist:
        return None
    adj = np.array([h[1] - h[2] for h in hist])
    peak = np.maximum.accumulate(np.concatenate([[0.0], adj]))[1:]
    return dict(rung=rung, start=days[idx[0]], first_mark=hist[0][0], end=hist[-1][0], sessions=len(hist), deposits=deposits, pnl=float(adj[-1]), pnl_pct_of_deposits=float(adj[-1] / deposits) if deposits else None, max_dd_won=float((adj - peak).min()), exposure_avg=float(np.mean([h[3] for h in hist])), buys=buys, sells=sells, qty_end=qty, final_value=hist[-1][1])


RUNGS = ["R0_cash", "R1_buyhold", "R2_buy_above_ma200", "R3_ma200_in_out", "R4_buy_after_dip"]
PREV = {"R1_buyhold": "R0_cash", "R2_buy_above_ma200": "R1_buyhold", "R3_ma200_in_out": "R2_buy_above_ma200", "R4_buy_after_dip": "R1_buyhold"}


def main():
    days, O, C = load()
    sig = signals(C)
    out = dict(meta=dict(version=2, script_sha256=sha(os.path.abspath(__file__)), raw_sha256=sha(RAW), sym=SYM, deposit=DEPOSIT, fee=FEE, tick=TICK, ma_n=MA_N, dip=DIP, dip_gap=DIP_GAP, periods=PERIODS, rule_text=dict(common="deposit 10,000 on every session in the period incl. the last; decide on close of session i, act at open of session i+1 (buy +tick, sell -tick, fee floored to won); value marked at close of i+1; raw (unadjusted) prices", R0_cash="deposit only", R1_buyhold="buy max whole shares whenever cash allows; never sell", R2_buy_above_ma200="R1 but buy only when close > 200-session MA (strict)", R3_ma200_in_out="buy when close > MA200, sell all when close < MA200, do nothing when equal", R4_buy_after_dip="R1 but buy only on the first -4% day (no -4% day in the prior 20 sessions) and the following 19 sessions = 20 decision sessions")), periods={})
    lines = []
    for pname, (s, e) in PERIODS.items():
        res = {r: run(r, days, O, C, sig, s, e) for r in RUNGS}
        out["periods"][pname] = res
        lines.append(f"[{pname} {res['R1_buyhold']['start']}~{res['R1_buyhold']['end']} {res['R1_buyhold']['sessions']}세션 입금 {res['R1_buyhold']['deposits']:,.0f}]")
        for r in RUNGS:
            m = res[r]
            prev = res[PREV[r]] if r in PREV else None
            delta = f" (직전 축 대비 {m['pnl'] - prev['pnl']:+,.0f})" if prev else ""
            lines.append(f"  {r:22s} 손익 {m['pnl']:+12,.0f} ({m['pnl_pct_of_deposits']*100:+6.1f}%) 낙폭 {m['max_dd_won']:+12,.0f} 노출 {m['exposure_avg']*100:3.0f}% 매수 {m['buys']:4d} 매도 {m['sells']:3d}{delta}")
    out["summary_lines"] = lines
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n".join(lines))
    print("saved", OUT, "sha", sha(OUT))


def selftest():
    n = 500
    days = [f"2020-{1 + i // 28:02d}-{1 + i % 28:02d}" if i < 336 else f"2021-{1 + (i - 336) // 28:02d}-{1 + (i - 336) % 28:02d}" for i in range(n)]
    C = np.full(n, 10000.0)
    C[:250] = np.linspace(10000.0, 10500.0, 250)
    C[250:300] = 9000.0
    C[300:] = 11000.0
    O = C.copy()
    sig = signals(C)
    r0 = run("R0_cash", days, O, C, sig, days[0], days[-1])
    assert r0["pnl"] == 0.0 and r0["buys"] == 0 and r0["max_dd_won"] == 0.0 and r0["exposure_avg"] == 0.0
    r1 = run("R1_buyhold", days, O, C, sig, days[0], days[-1])
    assert r1["buys"] > 0 and r1["sells"] == 0 and r1["exposure_avg"] > 0.9 and r1["qty_end"] > 0
    i_first = 0
    px = 10000.0 + TICK
    assert r1["pnl"] > 0, r1
    r3 = run("R3_ma200_in_out", days, O, C, sig, days[0], days[-1])
    assert r3["sells"] >= 1 and r3["buys"] >= 1, r3
    assert sig["dip_first"][250] and not sig["dip_first"][251]
    r4 = run("R4_buy_after_dip", days, O, C, sig, days[0], days[-1])
    assert 1 <= r4["buys"] <= 20 and r4["sells"] == 0, r4
    rich = run("R4_buy_after_dip", days, O, C, sig, days[0], days[-1])
    sig_one = dict(sig, dip_first=np.zeros(n, bool))
    sig_one["dip_first"][300] = True
    big = np.full(n, 100.0)
    r4b = run("R4_buy_after_dip", days, big, big, sig_one, days[0], days[-1])
    assert r4b["buys"] == 20, r4b["buys"]
    eq = np.full(n, 10000.0)
    sig_eq = signals(eq)
    r3e = run("R3_ma200_in_out", days, eq, eq, sig_eq, days[0], days[-1])
    assert r3e["buys"] == 0 and r3e["sells"] == 0, r3e
    r1c = run("R1_buyhold", days, O, C, sig, days[0], days[-1])
    assert r1c["deposits"] == DEPOSIT * n and r1c["sessions"] == n, (r1c["deposits"], r1c["sessions"])
    rp = run("R1_buyhold", days, O, C, sig, days[100], days[300])
    assert rp["start"] == days[100] and rp["end"] == days[300] and rp["sessions"] == 201 and rp["deposits"] == 201 * DEPOSIT and rp["first_mark"] == days[101], (rp["start"], rp["end"], rp["sessions"], rp["first_mark"])
    assert rp["buys"] <= 200
    flat = np.full(n, 10000.0)
    rf = run("R4_buy_after_dip", days, flat, flat, signals(flat), days[0], days[-1])
    assert rf["buys"] == 0 and rf["pnl"] == 0.0, rf
    empty = run("R1_buyhold", days, O, C, sig, "2030-01-01", "2030-12-31")
    assert empty is None
    print("selftest ok", dict(sha=sha(os.path.abspath(__file__))))


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    else:
        main()
