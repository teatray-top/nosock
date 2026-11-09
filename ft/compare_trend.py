import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = {"069500": (os.path.join(HERE, "user_data", "data_kodex", "source_snapshot", "069500.csv"), "ee10cfe3"), "229200": (os.path.join(HERE, "user_data", "data_kosdaq150_v1", "source_snapshot", "229200.csv"), "5213ea89")}
ADJ = r"G:\NoSock\stock\penny\data\kr_etf\kodex_daily_adj.json"
ADJ_SHA = "844b27891727b1faa227ea3259cb47eee258f8bf85ec30a3fc6a41b5a299a609"
ORIGIN = {"069500": "2002-10-14", "229200": "2015-10-01"}
LAST_COMPLETE = "2026-11-05"
FEE = 0.00015
CAPITAL = 10_000_000
WINDOWS = {"A": ("2022-11-23", "2025-12-30"), "B": ("2026-01-12", "2026-11-05")}
METHODS = ("hold", "ribbon", "close_gt_ema200", "ema50_gt_ema200", "donchian20_exit10")
PARITY = os.path.join(HERE, "results", "compare_kodex_v1_1.json")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sessions(sym):
    p, pre = SNAP[sym]
    if sha(p)[:8] != pre:
        raise SystemExit(f"snapshot hash changed for {sym}")
    d = pd.read_csv(p).sort_values("ts")
    d["day"] = d["ts"].str[:10]
    g = d.groupby("day")
    return pd.DataFrame(dict(open=g["open"].first().astype(float), close=g["close"].last().astype(float)))


def ema(xs, n):
    a = 2.0 / (n + 1)
    out, e = [], None
    for x in xs:
        e = x if e is None else a * x + (1 - a) * e
        out.append(e)
    return out


def weights(sym):
    if sha(ADJ) != ADJ_SHA:
        raise SystemExit("adjusted file hash changed")
    adj = json.load(open(ADJ, encoding="utf-8"))[sym]
    days = [d for d in adj["days"] if d <= LAST_COMPLETE]
    C = [float(c) for c in adj["C"][:len(days)]]
    if days[0] != ORIGIN[sym]:
        raise SystemExit("origin")
    e = {n: ema(C, n) for n in (8, 21, 50, 200)}
    w = {m: {} for m in METHODS if m != "hold"}
    held = 0
    for i, d in enumerate(days):
        w["ribbon"][d] = 1 if e[8][i] > e[21][i] > e[50][i] > e[200][i] else 0
        w["close_gt_ema200"][d] = 1 if C[i] > e[200][i] else 0
        w["ema50_gt_ema200"][d] = 1 if e[50][i] > e[200][i] else 0
        if i >= 20:
            if held == 0 and C[i] > max(C[i - 20:i]):
                held = 1
            elif held == 1 and C[i] < min(C[i - 10:i]):
                held = 0
        w["donchian20_exit10"][d] = held
    return w, days


def metrics(equity, expo, trips, trades):
    eq = pd.Series(equity)
    dd = (eq / eq.cummax() - 1).min()
    return dict(final_equity=round(float(eq.iloc[-1])), return_pct=round(100 * (eq.iloc[-1] / CAPITAL - 1), 2), max_drawdown_pct=round(100 * float(dd), 2), avg_exposure=round(float(pd.Series(expo).mean()), 3), round_trips=trips, trades=trades)


def run_hold(s, slip):
    o = s["open"].iloc[0] + slip
    q = int(CAPITAL // (o * (1 + FEE)))
    cash = CAPITAL - q * o - q * o * FEE
    eq = [cash + q * c for c in s["close"]]
    ex = [q * c / (cash + q * c) for c in s["close"]]
    return metrics(eq, ex, 1, [[s.index[0], "buy", q, o]])


def run_weights(s, w, all_days, slip):
    days = list(s.index)
    prev = {d: all_days[all_days.index(d) - 1] for d in days}
    cash, q, trips, trades = float(CAPITAL), 0, 0, []
    eq, ex = [], []
    for d in days:
        target = w[prev[d]]
        o = s.at[d, "open"]
        if target == 1 and q == 0:
            px = o + slip
            n = int(cash // (px * (1 + FEE)))
            if n > 0:
                cash -= n * px + n * px * FEE
                q = n
                trades.append([d, "buy", n, px])
        elif target == 0 and q > 0:
            px = o - slip
            cash += q * px - q * px * FEE
            trades.append([d, "sell", q, px])
            q = 0
            trips += 1
        c = s.at[d, "close"]
        eq.append(cash + q * c)
        ex.append(q * c / (cash + q * c))
    if q > 0:
        trips += 1
    return metrics(eq, ex, trips, trades)


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    res = {}
    for sym in SNAP:
        s_all = sessions(sym)
        w, all_days = weights(sym)
        for name, (a, b) in WINDOWS.items():
            s = s_all.loc[a:b]
            for slip in (0, 5):
                key = f"{sym}_{name}_slip{slip}"
                r = dict(window=[s.index[0], s.index[-1]], sessions=len(s), start_open=float(s["open"].iloc[0]), end_close=float(s["close"].iloc[-1]), hold=run_hold(s, slip))
                for m in METHODS[1:]:
                    r[m] = run_weights(s, w[m], all_days, slip)
                res[key] = r
    old = json.load(open(PARITY, encoding="utf-8"))["results"]
    parity = {}
    for k in ("A_slip0", "A_slip5", "B_slip0", "B_slip5"):
        for m in ("hold", "ribbon"):
            a, b = old[k][m], res["069500_" + k][m]
            parity[f"{k}.{m}"] = all(a[f] == b[f] for f in ("final_equity", "return_pct", "max_drawdown_pct", "avg_exposure", "round_trips"))
    rec = dict(version="trend-rules-compare-v1", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), snapshots={k: sha(v[0]) for k, v in SNAP.items()}, adjusted_sha256=ADJ_SHA, parity_with_compare_kodex_v1_1=parity, labels=spec["labels"], results=res)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print("parity 069500 hold/ribbon vs compare_kodex_v1_1:", all(parity.values()), {k: v for k, v in parity.items() if not v})
    for k, v in res.items():
        print(f"== {k} {v['window'][0]}..{v['window'][1]} sessions {v['sessions']}")
        for m in METHODS:
            r = v[m]
            print(f"  {m:18s} return {r['return_pct']:8.2f}% maxDD {r['max_drawdown_pct']:7.2f}% exposure {r['avg_exposure']:.2f} trips {r['round_trips']}")
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
