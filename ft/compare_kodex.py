import glob
import hashlib
import json
import math
import os
import sys
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = os.path.join(HERE, "user_data", "data_kodex", "source_snapshot", "069500.csv")
ADJ = r"G:\NoSock\stock\penny\data\kr_etf\kodex_daily_adj.json"
ADJ_SHA = "844b27891727b1faa227ea3259cb47eee258f8bf85ec30a3fc6a41b5a299a609"
FEE = 0.00015
CAPITAL = 10_000_000
SCALE = 10_000
WINDOWS = {"A": ("2022-11-23", "2025-12-30"), "B": ("2026-01-12", "2026-11-05")}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def fee(x):
    return x * FEE


def sessions():
    d = pd.read_csv(SNAP).sort_values("ts")
    d["day"] = d["ts"].str[:10]
    g = d.groupby("day")
    s = pd.DataFrame(dict(open=g["open"].first().astype(float), close=g["close"].last().astype(float)))
    return s


def ema_last_series(closes, n):
    a = 2.0 / (n + 1)
    out, e = [], None
    for x in closes:
        e = x if e is None else a * x + (1 - a) * e
        out.append(e)
    return out


def ribbon_weights():
    if sha(ADJ) != ADJ_SHA:
        raise SystemExit("adjusted file hash changed")
    adj = json.load(open(ADJ, encoding="utf-8"))["069500"]
    days = [d for d in adj["days"] if d <= "2026-11-05"]
    C = [float(c) for c in adj["C"][:len(days)]]
    if days[0] != "2002-10-14":
        raise SystemExit("origin")
    e = {n: ema_last_series(C, n) for n in (8, 21, 50, 200)}
    return {d: (1 if e[8][i] > e[21][i] > e[50][i] > e[200][i] else 0) for i, d in enumerate(days)}


def metrics(equity, expo, trips):
    eq = pd.Series(equity)
    dd = (eq / eq.cummax() - 1).min()
    return dict(final_equity=round(float(eq.iloc[-1])), return_pct=round(100 * (eq.iloc[-1] / CAPITAL - 1), 2), max_drawdown_pct=round(100 * float(dd), 2), avg_exposure=round(float(pd.Series(expo).mean()), 3), round_trips=trips)


def run_hold(s, slip):
    o = s["open"].iloc[0] + slip
    q = int(CAPITAL // (o * (1 + FEE)))
    cash = CAPITAL - q * o - fee(q * o)
    eq = [cash + q * c for c in s["close"]]
    ex = [q * c / (cash + q * c) for c in s["close"]]
    return metrics(eq, ex, 1)


def run_ribbon(s, w, slip):
    days = list(s.index)
    all_days = sorted(w)
    prev = {d: all_days[all_days.index(d) - 1] for d in days}
    cash, q, trips = float(CAPITAL), 0, 0
    eq, ex = [], []
    for d in days:
        target = w[prev[d]]
        o = s.at[d, "open"]
        if target == 1 and q == 0:
            px = o + slip
            n = int(cash // (px * (1 + FEE)))
            if n > 0:
                cash -= n * px + fee(n * px)
                q = n
        elif target == 0 and q > 0:
            px = o - slip
            cash += q * px - fee(q * px)
            q = 0
            trips += 1
        c = s.at[d, "close"]
        eq.append(cash + q * c)
        ex.append(q * c / (cash + q * c))
    if q > 0:
        trips += 1
    return metrics(eq, ex, trips)


def run_replay(s, window, strat, slip):
    z = sorted(glob.glob(os.path.join(HERE, "results", "rung2_v1", window, strat, "*.zip")))[-1]
    with zipfile.ZipFile(z) as f:
        n = [x for x in f.namelist() if x.endswith(".json") and not x.endswith("_config.json")][0]
        st = json.loads(f.read(n))["strategy"][strat]
    ev = []
    for t in st["trades"]:
        od = pd.Timestamp(t["open_date"]).tz_convert("Asia/Seoul").strftime("%Y-%m-%d")
        cd = pd.Timestamp(t["close_date"]).tz_convert("Asia/Seoul").strftime("%Y-%m-%d")
        ev.append((od, 0, t))
        ev.append((cd, 1, t))
    cash_u = float(st["starting_balance"])
    min_cash = cash_u
    pos = {}
    eq, ex = [], []
    by_day = {}
    for d, k, t in ev:
        by_day.setdefault(d, []).append((k, t))
    for d in s.index:
        for k, t in sorted(by_day.get(d, []), key=lambda x: (pd.Timestamp(x[1]["open_date"] if x[0] == 0 else x[1]["close_date"]), x[0])):
            a = t["amount"]
            if k == 0:
                px = t["open_rate"] + slip / SCALE
                cash_u -= a * px * (1 + t["fee_open"])
                pos[id(t)] = a
            else:
                px = t["close_rate"] - slip / SCALE
                cash_u += a * px * (1 - t["fee_close"])
                pos.pop(id(t), None)
            min_cash = min(min_cash, cash_u)
        c_u = s.at[d, "close"] / SCALE
        held = sum(pos.values())
        eq.append((cash_u + held * c_u) * SCALE)
        ex.append(held * c_u / (cash_u + held * c_u))
    out = metrics(eq, ex, len(st["trades"]))
    out["engine_final_balance_won"] = round(st["final_balance"] * SCALE)
    out["parity_ok"] = None if slip else abs(out["final_equity"] - out["engine_final_balance_won"]) <= 1
    out["min_cash_won"] = round(min_cash * SCALE)
    out["result_zip"] = os.path.relpath(z, HERE)
    return out


def main(out_path):
    if sha(SNAP)[:8] != "ee10cfe3":
        raise SystemExit("snapshot hash changed")
    s_all = sessions()
    w = ribbon_weights()
    res = {}
    for name, (a, b) in WINDOWS.items():
        s = s_all.loc[a:b]
        for slip in (0, 5):
            key = f"{name}_slip{slip}"
            res[key] = dict(window=[s.index[0], s.index[-1]], sessions=len(s), start_open=float(s["open"].iloc[0]), end_close=float(s["close"].iloc[-1]), hold=run_hold(s, slip), ribbon=run_ribbon(s, w, slip), AdxSmas=run_replay(s, name, "AdxSmas", slip), mabStra=run_replay(s, name, "mabStra", slip))
    rec = dict(version="kodex-same-basis-compare-v1.1", labels=dict(slip5="FIXED-FILL COST SENSITIVITY: an adverse 5 won on every fill with the original quantities kept, so replay cash can go negative; not an executable variant; parity is not checked for it", end_basis="hold and ribbon are valued at the last session close without liquidation; the freqtrade replays end with the engine's forced exit at the last 1h/4h bar", reading="return leadership and drawdown leadership are reported separately", data_check="2026-07-30 -> 07-31 +24.2% KODEX raw close jump is present in the Toss daily file, the Toss 1m snapshot and Yahoo Finance 069500.KS (87,635 -> 108,820); Samsung Electronics 207,000 -> 262,500 in both sources"), code_sha256=sha(os.path.abspath(__file__)), spec_sha256=sha(os.path.join(HERE, "compare_spec_v1.json")), snapshot_sha256=sha(SNAP), adjusted_sha256=ADJ_SHA, results=res)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    for k, v in res.items():
        print(f"== {k} {v['window'][0]}..{v['window'][1]} sessions {v['sessions']} open {v['start_open']:.0f} close {v['end_close']:.0f}")
        for m in ("hold", "ribbon", "AdxSmas", "mabStra"):
            r = v[m]
            extra = f" parity {r['parity_ok']} engine {r['engine_final_balance_won']:,} min_cash {r['min_cash_won']:,}" if "parity_ok" in r else ""
            print(f"  {m:8s} final {r['final_equity']:12,d} return {r['return_pct']:8.2f}% maxDD {r['max_drawdown_pct']:7.2f}% exposure {r['avg_exposure']:.2f} trips {r['round_trips']}{extra}")
    print("saved", out_path, sha(out_path)[:8])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "compare_kodex_v1_1.json"))
