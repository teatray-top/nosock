import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = {"069500": (os.path.join(HERE, "user_data", "data_kodex", "source_snapshot", "069500.csv"), "ee10cfe3"), "229200": (os.path.join(HERE, "user_data", "data_kosdaq150_v1", "source_snapshot", "229200.csv"), "5213ea89")}
CAL = os.path.join(HERE, "session_calendar_kr_etf_f1_v1.json")
ADJ = r"G:\NoSock\stock\penny\data\kr_etf\kodex_daily_adj.json"
ADJ_SHA = "844b27891727b1faa227ea3259cb47eee258f8bf85ec30a3fc6a41b5a299a609"
ORIGIN = {"069500": "2002-10-14", "229200": "2015-10-01"}
LAST_COMPLETE = "2026-11-05"
FEE = 0.00015
SLIP = 5
CAPITAL = 100_000_000
WINDOWS = {"A": ("2022-11-23", "2025-12-30"), "B": ("2026-01-12", "2026-11-05")}
METHODS = ("hold", "ribbon", "close_gt_ema200", "ema50_gt_ema200", "donchian20_exit10")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sessions(sym, cal):
    p, pre = SNAP[sym]
    if sha(p)[:8] != pre:
        raise SystemExit(f"snapshot hash changed for {sym}")
    d = pd.read_csv(p).sort_values("ts")
    d["day"] = d["ts"].str[:10]
    d["hm"] = d["ts"].str[11:16]
    out = {}
    for day, g in d.groupby("day"):
        cont_end = cal[day]["continuous_end"]
        g = g.reset_index(drop=True)
        first = g.iloc[0]
        pos = g[(g["volume"] > 0) & (g["hm"] <= cont_end)]
        fill = None if pos.empty else (float(pos.iloc[0]["open"]), pos.iloc[0]["hm"], bool(pos.index[0] != 0))
        out[day] = dict(close=float(g.iloc[-1]["close"]), first_open=float(first["open"]), fill=fill)
    return out


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


def simulate(sess, days, target_of):
    cash, q, trades, closed, delayed, skipped = float(CAPITAL), 0, [], 0, 0, 0
    equity = []
    for d in days:
        s = sess[d]
        t = target_of(d)
        if (t == 1 and q == 0) or (t == 0 and q > 0):
            if s["fill"] is None:
                skipped += 1
            else:
                px0, hm, late = s["fill"]
                delayed += int(late)
                if t == 1:
                    px = px0 + SLIP
                    n = int(cash // (px * (1 + FEE)))
                    if n > 0:
                        cash -= n * px * (1 + FEE)
                        q = n
                        trades.append([d, hm, "buy", n, px])
                else:
                    px = px0 - SLIP
                    cash += q * px * (1 - FEE)
                    trades.append([d, hm, "sell", q, px])
                    q = 0
                    closed += 1
        equity.append((d, cash + q * s["close"]))
    last = sess[days[-1]]["close"]
    final = cash + q * last * (1 - FEE)
    return dict(equity=equity, final=final, trades=trades, closed=closed, open_at_end=q > 0, delayed_fills=delayed, skipped_no_volume=skipped, avg_exposure=None)


def report(sim, hold_final, days, sess):
    eq = [CAPITAL] + [v for _, v in sim["equity"]]
    s = pd.Series(eq)
    dd = float((s / s.cummax() - 1).min())
    years = {}
    prev = float(CAPITAL)
    for i, (d, v) in enumerate(sim["equity"]):
        y = d[:4]
        last_of_year = i == len(sim["equity"]) - 1 or sim["equity"][i + 1][0][:4] != y
        if last_of_year:
            end = sim["final"] if i == len(sim["equity"]) - 1 else v
            years[y] = round(100 * (end / prev - 1), 2)
            prev = end
    ret = 100 * (sim["final"] / CAPITAL - 1)
    return dict(final_won=round(sim["final"]), return_pct=round(ret, 2), vs_hold_pct=round(ret - 100 * (hold_final / CAPITAL - 1), 2), max_drawdown_daily_pct=round(100 * dd, 2), completed_trades=sim["closed"], open_at_end=sim["open_at_end"], by_year_pct=years, delayed_fills=sim["delayed_fills"], skipped_no_volume=sim["skipped_no_volume"], trades=sim["trades"])


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    if sha(CAL) != spec["inputs"]["calendar_sha256"]:
        raise SystemExit("calendar hash differs")
    cal = json.load(open(CAL, encoding="utf-8"))["days"]
    res = {}
    for sym in SNAP:
        sess = sessions(sym, cal)
        w, all_days = weights(sym)
        prev = {d: all_days[all_days.index(d) - 1] for d in sess if d in all_days and all_days.index(d) > 0}
        for name, (a, b) in WINDOWS.items():
            days = [d for d in sorted(sess) if a <= d <= b]
            first = days[0]
            sims = {"hold": simulate(sess, days, lambda d: 1)}
            for m in METHODS[1:]:
                sims[m] = simulate(sess, days, lambda d, m=m: w[m][prev[d]])
            hold_final = sims["hold"]["final"]
            res[f"{sym}_{name}"] = dict(window=[first, days[-1]], sessions=len(days), methods={m: report(sims[m], hold_final, days, sess) for m in METHODS})
    rec = dict(version="trend-rules-compare-v2", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=dict(snapshots={k: sha(v[0]) for k, v in SNAP.items()}, adjusted_sha256=ADJ_SHA, calendar_sha256=sha(CAL)), labels=spec["labels"], results=res)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    for k, v in res.items():
        print(f"== {k} {v['window'][0]}..{v['window'][1]} sessions {v['sessions']}")
        for m in METHODS:
            r = v["methods"][m]
            print(f"  {m:18s} return {r['return_pct']:8.2f}% vs_hold {r['vs_hold_pct']:8.2f} maxDD {r['max_drawdown_daily_pct']:7.2f}% closed {r['completed_trades']:3d} open {r['open_at_end']!s:5s} delayed {r['delayed_fills']} skipped {r['skipped_no_volume']} years {r['by_year_pct']}")
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
