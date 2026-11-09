import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
import math
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import pandas as pd
import learn_inverse as LI
import learn_inverse_synth_v3 as V3
import compare_trend_v2 as CT
import harvest_v2 as H2
import harvest_v4 as H4
import turning_points_v1 as TPN

H = 20
BAR_MIN = 30
TEST_YEARS = (2023, 2024, 2025, 2026)
EVAL_FROM, EVAL_TO = "2023-01-02", "2026-11-05"
K_TP, K_SL, K_TRAIL = 1.6, 1.6, 0.8
CAPITAL = 100_000_000.0
VAULT_SHARE = 0.5
LONG, INV = "069500", "114800"
NFEATS = ["r1", "r5", "r20", "r60", "r120", "r250", "dd250", "ma20", "ma50", "ma200", "volratio", "e8_21", "e21_50", "e50_200", "up20"]
INV_MIN = os.path.join(os.path.dirname(HERE), "data", "kr_etf", "inverse_114800_20261107_151154", "114800.csv")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def norm_features(C):
    c = pd.Series(np.asarray(C, dtype=float))
    lc = np.log(c)
    lr = lc.diff()
    s = lr.rolling(20).std(ddof=0)
    s60 = lr.rolling(60).std(ddof=0)
    cols = {}
    for n in (1, 5, 20, 60, 120, 250):
        cols[f"r{n}"] = (lc - lc.shift(n)) / (s * math.sqrt(n))
    cols["dd250"] = (lc - np.log(c.rolling(250).max())) / (s * math.sqrt(250))
    for k in (20, 50, 200):
        cols[f"ma{k}"] = (lc - np.log(c.rolling(k).mean())) / (s * math.sqrt(k))
    cols["volratio"] = s / s60
    e = {n: c.ewm(alpha=2.0 / (n + 1), adjust=False).mean() for n in (8, 21, 50, 200)}
    cols["e8_21"] = np.log(e[8] / e[21]) / (s * math.sqrt(8))
    cols["e21_50"] = np.log(e[21] / e[50]) / (s * math.sqrt(21))
    cols["e50_200"] = np.log(e[50] / e[200]) / (s * math.sqrt(50))
    cols["up20"] = (lr > 0).astype(float).rolling(20).mean()
    X = pd.DataFrame(cols)[NFEATS].to_numpy()
    ok = np.isfinite(X).all(axis=1) & (s.to_numpy() > 0)
    ok[:250] = False
    return X, s.to_numpy(), ok


def daily_training(days, k, cutoff):
    X, s, ok = norm_features(k["Ca"])
    rows, ys = [], []
    for i in range(len(days) - H - 1):
        if ok[i] and i + H + 1 <= cutoff:
            rows.append(i)
            ys.append(math.log(k["Oa"][i + H + 1] / k["Oa"][i + 1]) / (s[i] * math.sqrt(H)))
    return X[rows], np.asarray(ys)


def synth_training():
    man = json.load(open(os.path.join(V3.PATHS, "manifest.json"), encoding="utf-8"))
    Xs, ys = [], []
    for e in man["paths"]:
        if e["split"] != "train":
            continue
        p = V3.load_path(e)
        O, C = np.asarray(p["k"]["O"]), p["k"]["C"]
        X, s, ok = norm_features(C)
        n = len(C)
        idx = np.arange(n - H - 1)
        m = ok[idx]
        idx = idx[m]
        Xs.append(X[idx])
        ys.append(np.log(O[idx + H + 1] / O[idx + 1]) / (s[idx] * math.sqrt(H)))
    return np.vstack(Xs), np.concatenate(ys)


def load_minutes(path, cal):
    d = pd.read_csv(path)
    d["day"] = d["ts"].str[:10]
    d["hm"] = d["ts"].str[11:16]
    out = {}
    for day, g in d.groupby("day"):
        if day not in cal or not (EVAL_FROM <= day <= EVAL_TO or day < EVAL_FROM):
            continue
        out[day] = g[["hm", "open", "close", "volume"]].to_numpy()
    return out


def build_bars(mins, cal):
    bars, dropped = [], []
    for day in sorted(mins):
        m = mins[day]
        info = cal[day]
        h0, m0 = map(int, info["session_start"].split(":"))
        start = h0 * 60 + m0
        lh, lm = map(int, info["last"].split(":"))
        last = lh * 60 + lm
        expected = last - start
        if len(m) != expected:
            dropped.append(dict(day=day, minutes=len(m), expected=expected))
            continue
        ce = info["continuous_end"]
        nb = math.ceil(expected / BAR_MIN)
        for b in range(nb):
            lo, hi = start + BAR_MIN * b, min(start + BAR_MIN * (b + 1), last)
            sel = [r for r in m if lo < int(r[0][:2]) * 60 + int(r[0][3:]) <= hi]
            fills = [(r[0], float(r[1])) for r in sel if float(r[3]) > 0 and r[0] <= ce]
            bars.append(dict(day=day, bar=b, end=f"{hi // 60:02d}:{hi % 60:02d}", close=float(sel[-1][2]), fill=fills[0] if fills else None))
    return bars, dropped


def next_fill(bars):
    nf = [None] * len(bars)
    nxt = None
    for i in range(len(bars) - 1, -1, -1):
        nf[i] = nxt
        if bars[i]["fill"] is not None:
            nxt = (i, bars[i]["fill"][1])
    return nf


def run_harvest(kind, exitk, closes, s_at, pred_ret, fills_l, fills_i, closes_i, cost_thr, n, start):
    cash, vault = CAPITAL, 0.0
    leg, q, basis, entry, top, sig = None, 0.0, 0.0, None, None, None
    sales, cost_paid, trades = 0, 0.0, []
    series = {}
    peak, mdd = CAPITAL, 0.0
    pending = None
    for i in range(start, n):
        if pending and pending[0] <= i:
            _, act, px, _dec = pending
            if act == "sell" and leg:
                c = LI.FEE + LI.TICK / px
                proceeds = q * px * (1 - c)
                cost_paid += q * px * c
                profit = proceeds - basis
                tv = VAULT_SHARE * profit if profit > 0 else 0.0
                vault += tv
                cash += proceeds - tv
                sales += 1
                trades.append(dict(side="sell", leg=leg, bar=i, price=px, profit=profit, to_vault=tv))
                leg, q, basis, entry, top, sig = None, 0.0, 0.0, None, None, None
            elif act in (LONG, INV) and leg is None and cash > 0:
                c = LI.FEE + LI.TICK / px
                q = cash * (1 - c) / px
                cost_paid += cash * c
                basis, entry, cash, leg = cash, cash / q, 0.0, act
                top = entry
                sig = s_at(pending[3]) * math.sqrt(H)
                trades.append(dict(side="buy", leg=leg, bar=i, price=px))
            pending = None
        cl = closes[i] if leg != INV else closes_i[i]
        tot = cash + (q * cl if leg else 0.0) + vault
        series[i] = tot
        peak = max(peak, tot)
        mdd = min(mdd, tot / peak - 1)
        if pending or i >= n - 1:
            continue
        act = None
        if kind == "hold":
            if leg is None and i == start:
                act = LONG
        elif leg:
            top = max(top, cl)
            if cl >= entry * (1 + K_TP * sig):
                act = "sell"
            elif exitk == "X0" and cl <= entry * (1 - K_SL * sig):
                act = "sell"
            elif exitk == "X1" and cl <= top * (1 - K_TRAIL * sig):
                act = "sell"
        else:
            pr = pred_ret(i)
            if pr is not None:
                if pr > cost_thr(i, LONG):
                    act = LONG
                elif kind == "C" and -pr > cost_thr(i, INV):
                    act = INV
        if act:
            f = (fills_i if act == INV or (act == "sell" and leg == INV) else fills_l)[i]
            if f is not None:
                pending = (f[0], act, f[1], i)
    if leg:
        px = closes[n - 1] if leg != INV else closes_i[n - 1]
        c = LI.FEE + LI.TICK / px
        cost_paid += q * px * c
        eq = cash + q * px * (1 - c)
    else:
        eq = cash
    final = eq + vault
    mdd = min(mdd, final / peak - 1)
    series[n - 1] = final
    return dict(vault_pct=100 * vault / CAPITAL, equity_pct=100 * eq / CAPITAL, total_pct=100 * (final / CAPITAL - 1), mdd_total=100 * mdd, sales=sales, cost_pct=100 * cost_paid / CAPITAL), series, trades


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, p in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), p)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    days, k, inv = LI.load(LI.INV_DIR_DEFAULT)
    sX, sy = synth_training()
    cal = json.load(open(CT.CAL, encoding="utf-8"))["days"]
    bars, dropped = build_bars(load_minutes(CT.SNAP[LONG][0], cal), cal)
    ibars, idropped = build_bars(load_minutes(INV_MIN, cal), cal)
    key = lambda b: (b["day"], b["bar"])
    imap = {key(b): b for b in ibars}
    bars = [b for b in bars if key(b) in imap]
    ib = [imap[key(b)] for b in bars]
    bc = [b["close"] for b in bars]
    ic = [b["close"] for b in ib]
    BX, Bs, Bok = norm_features(bc)
    fl, fi = next_fill(bars), next_fill(ib)
    dX, ds, dok = norm_features(k["Ca"])
    models, fold_info = {}, []
    pred_bar, pred_day = {}, {}
    for y in TEST_YEARS:
        e = next(j for j, d in enumerate(days) if d >= f"{y}-01-01")
        cutoff = e - 1
        X, yv = daily_training(days, k, cutoff)
        ds_ = LI.lgb.Dataset(np.vstack([X, sX]), label=np.concatenate([yv, sy]), free_raw_data=False)
        m = LI.lgb.train(LI.PARAMS, ds_, num_boost_round=LI.ROUNDS)
        models[y] = m
        bi = [i for i, b in enumerate(bars) if b["day"][:4] == str(y) and Bok[i]]
        if bi:
            p = m.predict(BX[bi])
            for i, v in zip(bi, p):
                pred_bar[i] = float(v)
        di = [i for i, d in enumerate(days) if d[:4] == str(y) and dok[i] and d <= EVAL_TO]
        if di:
            p = m.predict(dX[di])
            for i, v in zip(di, p):
                pred_day[i] = float(v)
        fold_info.append(dict(year=y, cutoff_day=days[cutoff], real_rows=int(len(X)), synth_rows=int(len(sX)), bar_preds=len(bi), day_preds=len(di)))
    nb = len(bars)
    real_norm = []
    for i in sorted(pred_bar):
        a, b = fl[i], None
        if a is not None:
            j = a[0] + H
            if j < nb and fl[j - 1] is not None:
                b = fl[j - 1]
        if a is not None and b is not None:
            real_norm.append((pred_bar[i], math.log(b[1] / a[1]) / (Bs[i] * math.sqrt(H))))
    rn = np.asarray(real_norm)
    ic_bar = dict(corr=float(np.corrcoef(rn[:, 0], rn[:, 1])[0, 1]), sign_hit=float(np.mean((rn[:, 0] > 0) == (rn[:, 1] > 0))), base=float(np.mean(rn[:, 1] > 0)), n=int(len(rn)))
    start_bar = next(i for i, b in enumerate(bars) if b["day"] >= EVAL_FROM)
    band = LI.BAND
    cost_thr_b = lambda i, leg: 2 * (LI.FEE + LI.TICK / (bc[i] if leg == LONG else ic[i])) + band
    pr_b = lambda i: pred_bar[i] * Bs[i] * math.sqrt(H) if i in pred_bar else None
    res, series, trades = {}, {}, {}
    for name, kind, ex in (("bar30_hold", "hold", None), ("bar30_A_X0", "A", "X0"), ("bar30_A_X1", "A", "X1"), ("bar30_C_X0", "C", "X0"), ("bar30_C_X1", "C", "X1")):
        r, s_, t = run_harvest(kind, ex, bc, lambda i: Bs[i], pr_b, fl, fi, ic, cost_thr_b, nb, start_bar)
        res[name], series[name], trades[name] = r, s_, [dict(x, day=bars[x["bar"]]["day"], end=bars[x["bar"]]["end"]) for x in t]
    raw = json.load(open(os.path.join(os.path.dirname(HERE), "data", "kr_etf", "kodex_daily_raw.json"), encoding="utf-8"))[LONG]
    iv = json.load(open(H2.INV_DAILY, encoding="utf-8"))[INV]
    imapd = {d: (float(o), float(c)) for d, o, c in zip(iv["days"], iv["O"], iv["C"])}
    dd_ = [d for d in raw["days"] if d <= EVAL_TO]
    st = dd_.index(EVAL_FROM) - 1
    keep = list(range(st, len(dd_)))
    sub = [dd_[i] for i in keep]
    dO = [float(raw["O"][i]) for i in keep]
    dC = [float(raw["C"][i]) for i in keep]
    iO = [imapd[dd_[i]][0] for i in keep]
    iC = [imapd[dd_[i]][1] for i in keep]
    dfl = [(j + 1, dO[j + 1]) if j + 1 < len(keep) else None for j in range(len(keep))]
    dfi = [(j + 1, iO[j + 1]) if j + 1 < len(keep) else None for j in range(len(keep))]
    gidx = {d: i for i, d in enumerate(days)}
    pr_d = lambda j: pred_day[gidx[sub[j]]] * ds[gidx[sub[j]]] * math.sqrt(H) if gidx[sub[j]] in pred_day else None
    s_d = lambda j: ds[gidx[sub[j]]]
    cost_thr_d = lambda j, leg: 2 * (LI.FEE + LI.TICK / (dC[j] if leg == LONG else iC[j])) + band
    for name, kind, ex in (("day_norm_A_X0", "A", "X0"), ("day_norm_A_X1", "A", "X1"), ("day_norm_C_X0", "C", "X0"), ("day_norm_C_X1", "C", "X1")):
        r, s_, t = run_harvest(kind, ex, dC, s_d, pr_d, dfl, dfi, iC, cost_thr_d, len(keep), 0)
        res[name], series[name], trades[name] = r, s_, [dict(x, day=sub[x["bar"]]) for x in t]
    A = [float(x) for x in json.load(open(CT.ADJ, encoding="utf-8"))[LONG]["C"][:len(dd_)]]
    e = {m: CT.ema(A, m) for m in (8, 21, 50, 200)}
    g0 = json.load(open(H2.PRED_FILE, encoding="utf-8"))["preds"]
    px = {LONG: dict(O=dO, C=dC), INV: dict(O=iO, C=iC)}
    for p in H4.policies():
        if p["name"] in ("hold", "ribbon", "A_long_X0_sl10", "A_long_X1_trail5"):
            r, s_, _ = H4.run(p, px, lambda leg, j: LI.FEE + LI.TICK / px[leg]["O"][j], 0, len(sub), lambda j: tuple(g0[sub[j]]) if sub[j] in g0 else (None, None), lambda j: e[8][keep[j]] > e[21][keep[j]] > e[50][keep[j]] > e[200][keep[j]], lambda j: e[8][keep[j]] < e[21][keep[j]] < e[50][keep[j]] < e[200][keep[j]], lambda j, leg: e[8][keep[j]] < e[21][keep[j]])
            res["day_G0_" + p["name"]], series["day_G0_" + p["name"]] = r, s_
    day_val = {}
    for name, s_ in series.items():
        if name.startswith("bar30"):
            v = {}
            for i, val in s_.items():
                v[bars[i]["day"]] = val
            day_val[name] = v
        else:
            day_val[name] = {sub[j]: val for j, val in s_.items()}
    pts = [(kd, i) for kd, i in TPN.zigzag(A, TPN.ZIG) if EVAL_FROM <= dd_[i] <= EVAL_TO]
    turn = []
    for kd, i in pts:
        j = keep.index(i)
        a, b = sub[max(0, j - TPN.BEFORE)], sub[min(len(sub) - 1, j + TPN.AFTER)]
        row = dict(kind=kd, day=dd_[i], window=[a, b])
        for name, v in day_val.items():
            da = max(x for x in v if x <= a) if any(x <= a for x in v) else min(v)
            db = max(x for x in v if x <= b)
            row[name] = 100 * (v[db] / v[da] - 1)
        turn.append(row)
    tsum = {kd: {name: dict(lost=int(sum(r[name] < 0 for r in turn if r["kind"] == kd)), worst=float(min(r[name] for r in turn if r["kind"] == kd)), n=int(sum(1 for r in turn if r["kind"] == kd))) for name in day_val} for kd in ("peak", "trough") if any(r["kind"] == kd for r in turn)}
    rec = dict(version="dt-scale-v1", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], bars=dict(count=nb, dropped_days_069500=dropped, dropped_days_114800=idropped, first=bars[0]["day"], last=bars[-1]["day"]), folds=fold_info, ic_bar30=ic_bar, results=res, turning_points=turn, turning_summary=tsum, trades=trades)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print("bars", nb, "dropped days", len(dropped), len(idropped), "| folds", fold_info)
    print("30m out-of-sample (daily-trained, vol-normalized):", ic_bar)
    for n_, r in res.items():
        print(f"{n_:30s} vault {r['vault_pct']:6.1f}% total {r['total_pct']:7.1f}% mdd {r['mdd_total']:6.1f} sales {r['sales']:4d} cost {r['cost_pct']:5.2f}%")
    for kd, s_ in tsum.items():
        print(kd, json.dumps({n_: (v["lost"], v["n"], round(v["worst"], 1)) for n_, v in s_.items()}))
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
