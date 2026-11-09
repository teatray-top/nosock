import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pandas as pd
import compare_trend_v2 as CT

L, S = "069500", "114800"
RAW = r"G:\NoSock\stock\penny\data\kr_etf\kodex_daily_raw.json"
DAILY_FROM = "2009-10-01"
LAST = CT.LAST_COMPLETE
FEE, SLIP, CAPITAL = CT.FEE, CT.SLIP, CT.CAPITAL
METHODS = ("hold", "ribbon", "ribbon_inv", "ribbon_inv_flip", "donchian20_exit10", "donchian_ls")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def targets():
    if sha(CT.ADJ) != CT.ADJ_SHA:
        raise SystemExit("adjusted file hash changed")
    adj = json.load(open(CT.ADJ, encoding="utf-8"))[L]
    days = [d for d in adj["days"] if d <= LAST]
    C = [float(c) for c in adj["C"][:len(days)]]
    if days[0] != CT.ORIGIN[L]:
        raise SystemExit("origin")
    e = {n: CT.ema(C, n) for n in (8, 21, 50, 200)}
    tg = {m: {} for m in METHODS}
    lo, ls = 0, 0
    for i, d in enumerate(days):
        bull = e[8][i] > e[21][i] > e[50][i] > e[200][i]
        bear = e[8][i] < e[21][i] < e[50][i] < e[200][i]
        tg["hold"][d] = L
        tg["ribbon"][d] = L if bull else None
        tg["ribbon_inv"][d] = L if bull else (S if bear else None)
        tg["ribbon_inv_flip"][d] = L if bull else S
        if i >= 20:
            hi20, lo20, hi10, lo10 = max(C[i - 20:i]), min(C[i - 20:i]), max(C[i - 10:i]), min(C[i - 10:i])
            if lo == 0 and C[i] > hi20:
                lo = 1
            elif lo == 1 and C[i] < lo10:
                lo = 0
            if ls == 0:
                ls = 1 if C[i] > hi20 else (-1 if C[i] < lo20 else 0)
            elif ls == 1 and C[i] < lo10:
                ls = 0
            elif ls == -1 and C[i] > hi10:
                ls = 0
        tg["donchian20_exit10"][d] = L if lo == 1 else None
        tg["donchian_ls"][d] = L if ls == 1 else (S if ls == -1 else None)
    return tg, days


def minute_sessions(path, cal):
    d = pd.read_csv(path).sort_values("ts")
    d["day"] = d["ts"].str[:10]
    d["hm"] = d["ts"].str[11:16]
    out = {}
    for day, g in d.groupby("day"):
        if day not in cal:
            continue
        ce = cal[day]["continuous_end"]
        pos = [(r.hm, float(r.open)) for r in g.itertuples() if r.volume > 0 and r.hm <= ce]
        out[day] = dict(close=float(g.iloc[-1]["close"]), pos=pos)
    return out


def first_fill(s, after):
    for hm, o in s["pos"]:
        if after is None or hm > after:
            return o, hm
    return None


def simulate(days, sess, target_of, fill_of):
    cash, q = float(CAPITAL), {L: 0, S: 0}
    trades, skipped, held = [], 0, {L: 0, S: 0, None: 0}
    equity = []
    for d in days:
        t = target_of(d)
        after = None
        for leg in (L, S):
            if q[leg] > 0 and leg != t:
                f = fill_of(leg, d, after)
                if f is None:
                    skipped += 1
                    continue
                px = f[0] - SLIP
                cash += q[leg] * px * (1 - FEE)
                trades.append([d, f[1], "sell", leg, q[leg], px])
                q[leg] = 0
                after = f[1]
        if t is not None and q[t] == 0 and q[L if t == S else S] == 0:
            f = fill_of(t, d, after)
            if f is None:
                skipped += 1
            else:
                px = f[0] + SLIP
                n = int(cash // (px * (1 + FEE)))
                if n > 0:
                    cash -= n * px * (1 + FEE)
                    q[t] = n
                    trades.append([d, f[1], "buy", t, n, px])
        held[L if q[L] else (S if q[S] else None)] += 1
        equity.append((d, cash + q[L] * sess[L][d]["close"] + q[S] * sess[S][d]["close"]))
    final = cash + sum(q[k] * sess[k][days[-1]]["close"] * (1 - FEE) for k in (L, S))
    return dict(equity=equity, final=final, trades=trades, skipped=skipped, days_long=held[L], days_inverse=held[S], days_cash=held[None])


def report(sim, hold_final):
    s = pd.Series([CAPITAL] + [v for _, v in sim["equity"]])
    dd = float((s / s.cummax() - 1).min())
    years, prev = {}, float(CAPITAL)
    eq = sim["equity"]
    for i, (d, v) in enumerate(eq):
        if i == len(eq) - 1 or eq[i + 1][0][:4] != d[:4]:
            end = sim["final"] if i == len(eq) - 1 else v
            years[d[:4]] = round(100 * (end / prev - 1), 2)
            prev = end
    ret = 100 * (sim["final"] / CAPITAL - 1)
    return dict(final_won=round(sim["final"]), return_pct=round(ret, 2), vs_hold_pct=round(ret - 100 * (hold_final / CAPITAL - 1), 2), max_drawdown_daily_pct=round(100 * dd, 2), buys=sum(1 for t in sim["trades"] if t[2] == "buy"), inverse_buys=sum(1 for t in sim["trades"] if t[2] == "buy" and t[3] == S), days_long=sim["days_long"], days_inverse=sim["days_inverse"], days_cash=sim["days_cash"], skipped=sim["skipped"], by_year_pct=years, trades=sim["trades"])


def run_block(days, sess, fill_of, tg, prev):
    sims = {m: simulate(days, sess, (lambda d, m=m: tg[m][prev[d]]), fill_of) for m in METHODS}
    hf = sims["hold"]["final"]
    return dict(window=[days[0], days[-1]], sessions=len(days), methods={m: report(sims[m], hf) for m in METHODS})


def main(data_dir, out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    inv_csv, inv_raw = os.path.join(data_dir, S + ".csv"), os.path.join(data_dir, "daily_raw.json")
    for p, k in ((inv_csv, "inverse_minute_sha256"), (inv_raw, "inverse_daily_raw_sha256"), (RAW, "kodex_daily_raw_sha256"), (CT.CAL, "calendar_sha256"), (CT.SNAP[L][0], "kodex_minute_sha256")):
        if sha(p) != spec["inputs"][k]:
            raise SystemExit(f"input hash differs: {k}")
    cal = json.load(open(CT.CAL, encoding="utf-8"))["days"]
    tg, all_days = targets()
    prev = {d: all_days[i - 1] for i, d in enumerate(all_days) if i > 0}
    msess = {L: minute_sessions(CT.SNAP[L][0], cal), S: minute_sessions(inv_csv, cal)}
    res = {}
    for name, (a, b) in CT.WINDOWS.items():
        days = [d for d in sorted(msess[L]) if a <= d <= b]
        missing = [d for d in days if d not in msess[S]]
        if missing:
            raise SystemExit(f"inverse minutes missing on {len(missing)} days, first {missing[:3]}")
        res[f"minute_{name}"] = run_block(days, msess, lambda leg, d, after: first_fill(msess[leg][d], after), tg, prev)
    kr = json.load(open(RAW, encoding="utf-8"))[L]
    ir = json.load(open(inv_raw, encoding="utf-8"))[S]
    dsess = {L: {}, S: {}}
    for sym, src in ((L, kr), (S, ir)):
        for i, d in enumerate(src["days"]):
            if DAILY_FROM <= d <= LAST:
                dsess[sym][d] = dict(open=float(src["O"][i]), close=float(src["C"][i]), vol=float(src["V"][i]))
    ddays = sorted(set(dsess[L]) & set(dsess[S]) & set(prev))
    only = sorted(set(dsess[L]) ^ set(dsess[S]))
    res["daily_open"] = run_block(ddays, dsess, lambda leg, d, after: (dsess[leg][d]["open"], "09:00") if dsess[leg][d]["vol"] > 0 and dsess[leg][d]["open"] > 0 else None, tg, prev)
    res["daily_open"]["fill_note"] = "daily open approximation for both legs; a switch sells and buys at the same session open"
    res["daily_open"]["calendar_mismatch_days"] = only
    rec = dict(version="inverse-compare-v1", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], results=res)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    for k, v in res.items():
        print(f"== {k} {v['window'][0]}..{v['window'][1]} sessions {v['sessions']}")
        for m in METHODS:
            r = v["methods"][m]
            print(f"  {m:18s} return {r['return_pct']:8.2f}% vs_hold {r['vs_hold_pct']:8.2f} maxDD {r['max_drawdown_daily_pct']:7.2f}% buys {r['buys']:3d} inv {r['inverse_buys']:3d} days L/S/C {r['days_long']}/{r['days_inverse']}/{r['days_cash']} skip {r['skipped']} years {r['by_year_pct']}")
    print("calendar mismatch days (daily)", len(res["daily_open"]["calendar_mismatch_days"]))
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:5])
