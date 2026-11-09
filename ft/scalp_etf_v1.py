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
import lightgbm as lgb
import learn_inverse as LI
import compare_trend_v2 as CT

CAL = CT.CAL
LONG_MIN = CT.SNAP["069500"][0]
INV_MIN = os.path.join(os.path.dirname(HERE), "data", "kr_etf", "inverse_114800_20261107_151154", "114800.csv")
HORIZONS = (10, 30, 60)
CAPITAL = 100_000_000.0
VAULT_SHARE = 0.5
TEST_FROM = "2023-07-01"
EVAL_TO = "2026-11-05"
SEEDS = 20
FEATS = ["r1", "r3", "r5", "r10", "r15", "r30", "r60", "vol15", "vol60", "volratio", "vshare5_60", "flow15", "tod", "d_open", "d_prev", "pos_range", "prev_ret", "ret5d", "ret20d", "prev_range", "bull", "bear"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def hm2m(s):
    return int(s[:2]) * 60 + int(s[3:5])


def load():
    cal = json.load(open(CAL, encoding="utf-8"))["days"]
    a = pd.read_csv(LONG_MIN)
    b = pd.read_csv(INV_MIN).rename(columns={"open": "io", "high": "ih", "low": "il", "close": "ic", "volume": "iv"})
    d = a.merge(b, on="ts", how="inner")
    d["day"] = d["ts"].str[:10]
    d["hm"] = d["ts"].str[11:16]
    d = d[d["day"].isin(cal.keys()) & (d["day"] <= EVAL_TO)]
    keep, dropped = [], []
    for day, g in d.groupby("day"):
        info = cal[day]
        exp = hm2m(info["last"]) - hm2m(info["session_start"])
        if len(g) != exp:
            dropped.append(day)
            continue
        keep.append(day)
    d = d[d["day"].isin(keep)].sort_values("ts").reset_index(drop=True)
    d["mod"] = d["hm"].map(hm2m)
    d["ce"] = d["day"].map(lambda x: hm2m(cal[x]["continuous_end"]))
    d["start"] = d["day"].map(lambda x: hm2m(cal[x]["session_start"]))
    return d, dropped


def daily_context(d):
    adj = json.load(open(CT.ADJ, encoding="utf-8"))["069500"]
    A = [float(x) for x in adj["C"]]
    e = {k: CT.ema(A, k) for k in (8, 21, 50, 200)}
    pos = {day: i for i, day in enumerate(adj["days"])}
    days = sorted(d["day"].unique())
    last = d.groupby("day").agg(close=("close", "last"), hi=("high", "max"), lo=("low", "min"))
    out = {}
    for k, day in enumerate(days):
        if k < 21:
            continue
        p = days[k - 1]
        c = last["close"]
        i = pos[p]
        out[day] = dict(prev_ret=math.log(c[p] / c[days[k - 2]]), ret5d=math.log(c[p] / c[days[k - 6]]), ret20d=math.log(c[p] / c[days[k - 21]]), prev_range=math.log(last["hi"][p] / last["lo"][p]), bull=float(e[8][i] > e[21][i] > e[50][i] > e[200][i]), bear=float(e[8][i] < e[21][i] < e[50][i] < e[200][i]), prev_close=float(c[p]))
    return out


def features(d, ctx):
    lc = np.log(d["close"].to_numpy(float))
    same = (d["day"] == d["day"].shift(1)).to_numpy()
    r1 = np.where(same, np.diff(lc, prepend=np.nan), np.nan)
    f = pd.DataFrame(index=d.index)
    s = pd.Series(lc)
    for k in (1, 3, 5, 10, 15, 30, 60):
        f[f"r{k}"] = s - s.shift(k)
    r1s = pd.Series(r1)
    f["vol15"] = r1s.rolling(15, min_periods=10).std()
    f["vol60"] = r1s.rolling(60, min_periods=40).std()
    v = d["volume"].astype(float)
    f["volratio"] = f["vol15"] / f["vol60"]
    f["vshare5_60"] = v.rolling(5).sum() / (v.rolling(60).sum() / 12.0)
    sv = np.sign(r1s.fillna(0.0)) * v
    f["flow15"] = sv.rolling(15).sum() / v.rolling(15).sum()
    f["tod"] = (d["mod"] - d["start"]).astype(float)
    g = d.groupby("day")
    f["d_open"] = lc - np.log(g["open"].transform("first").to_numpy(float))
    hi = g["high"].cummax().to_numpy(float)
    lo = g["low"].cummin().to_numpy(float)
    f["pos_range"] = np.where(hi > lo, (d["close"].to_numpy(float) - lo) / np.maximum(hi - lo, 1e-9), 0.5)
    cx = d["day"].map(lambda x: ctx.get(x))
    for k in ("prev_ret", "ret5d", "ret20d", "prev_range", "bull", "bear"):
        f[k] = cx.map(lambda z: z[k] if z else np.nan)
    pc = cx.map(lambda z: z["prev_close"] if z else np.nan).to_numpy(float)
    f["d_prev"] = lc - np.log(pc)
    return f[FEATS]


def fill_index(d):
    ok = ((d["volume"] > 0) & (d["iv"] > 0) & (d["mod"] <= d["ce"])).to_numpy()
    day = d["day"].to_numpy()
    n = len(d)
    nxt = np.full(n + 1, -1)
    for i in range(n - 1, -1, -1):
        nxt[i] = i if ok[i] else (nxt[i + 1] if i + 1 < n and day[i + 1] == day[i] else -1)
    return nxt[:n], day


def labels(d, H):
    nxt, day = fill_index(d)
    n = len(d)
    O, IO = d["open"].to_numpy(float), d["io"].to_numpy(float)
    Hh, Ll = d["high"].to_numpy(float), d["low"].to_numpy(float)
    ent = np.full(n, -1)
    ext = np.full(n, -1)
    for i in range(n - 1):
        if day[i + 1] != day[i]:
            continue
        e = nxt[i + 1]
        if e < 0 or e + H >= n or day[e + H] != day[i]:
            continue
        x = nxt[e + H]
        if x < 0:
            continue
        ent[i], ext[i] = e, x
    v = ent >= 0
    yl = np.full(n, np.nan)
    yi = np.full(n, np.nan)
    yv = np.full(n, np.nan)
    yl[v] = np.log(O[ext[v]] / O[ent[v]])
    yi[v] = np.log(IO[ext[v]] / IO[ent[v]])
    for i in np.nonzero(v)[0]:
        yv[i] = math.log(Hh[ent[i]:ext[i]].max() / Ll[ent[i]:ext[i]].min())
    return ent, ext, yl, yi, yv


def folds(days):
    qs = []
    y, q = 2023, 3
    while True:
        a = f"{y}-{3 * (q - 1) + 1:02d}-01"
        y2, q2 = (y, q + 1) if q < 4 else (y + 1, 1)
        b = f"{y2}-{3 * (q2 - 1) + 1:02d}-01"
        if a > EVAL_TO:
            break
        qs.append((a, b))
        y, q = y2, q2
    return qs


def fit(X, y):
    return lgb.train(LI.PARAMS, lgb.Dataset(X, label=y, free_raw_data=False), num_boost_round=LI.ROUNDS)


def cost_at(px):
    return LI.FEE + LI.TICK / px


def simulate(rows, ent, ext, d, pred, mode, rng=None, count=None):
    O, IO = d["open"].to_numpy(float), d["io"].to_numpy(float)
    cash, vault, trades = CAPITAL, 0.0, []
    busy_until = -1
    cand = list(rows)
    if mode == "random":
        picks = sorted(rng.choice(len(cand), size=min(count[0], len(cand)), replace=False))
        cand = [cand[k] for k in picks]
        legs = rng.random(len(cand)) < count[1]
    for k, i in enumerate(cand):
        if ent[i] < 0 or ent[i] <= busy_until:
            continue
        if mode == "model":
            p = pred[i]
            tl = 2 * cost_at(O[ent[i]])
            ti = 2 * cost_at(IO[ent[i]])
            if p > tl:
                leg = "L"
            elif -p > ti:
                leg = "I"
            else:
                continue
        else:
            leg = "L" if legs[k] else "I"
        P = O if leg == "L" else IO
        pin, pout = P[ent[i]], P[ext[i]]
        cin, cout = cost_at(pin), cost_at(pout)
        q = cash * (1 - cin) / pin
        proceeds = q * pout * (1 - cout)
        profit = proceeds - cash
        tv = VAULT_SHARE * profit if profit > 0 else 0.0
        vault += tv
        trades.append(dict(i=int(i), leg=leg, entry=int(ent[i]), exit=int(ext[i]), net=proceeds / cash - 1, to_vault=tv))
        cash = proceeds - tv
        busy_until = ext[i]
    return dict(total_pct=100 * ((cash + vault) / CAPITAL - 1), vault_pct=100 * vault / CAPITAL, equity_pct=100 * cash / CAPITAL, trades=len(trades), mean_net_bp=1e4 * float(np.mean([t["net"] for t in trades])) if trades else None, win=float(np.mean([t["net"] > 0 for t in trades])) if trades else None, long_share=float(np.mean([t["leg"] == "L" for t in trades])) if trades else None), trades


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, p in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), p)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    d, dropped = load()
    ctx = daily_context(d)
    X = features(d, ctx).to_numpy(float)
    okx = ~np.isnan(X).any(axis=1)
    days = d["day"].to_numpy()
    fl = folds(sorted(set(days)))
    O = d["open"].to_numpy(float)
    res = dict(rows=int(len(d)), days=int(len(set(days))), dropped_days=dropped, folds=fl, horizons={})
    print("rows", len(d), "days", len(set(days)), "dropped", len(dropped), "folds", len(fl), flush=True)
    for H in HORIZONS:
        ent, ext, yl, yi, yv = labels(d, H)
        lab = okx & ~np.isnan(yl)
        pred = np.full(len(d), np.nan)
        pvol = np.full(len(d), np.nan)
        for a, b in fl:
            tr = lab & (days < a)
            te = lab & (days >= a) & (days < b)
            if te.sum() == 0:
                continue
            m = fit(X[tr], yl[tr])
            pred[te] = m.predict(X[te])
            mv = fit(X[tr], yv[tr])
            pvol[te] = mv.predict(X[te])
        te_all = lab & (days >= TEST_FROM)
        p, y, yI = pred[te_all], yl[te_all], yi[te_all]
        cl = np.array([2 * cost_at(x) for x in O[ent[te_all]]])
        ci = np.array([2 * cost_at(x) for x in d["io"].to_numpy(float)[ent[te_all]]])
        dec = pd.qcut(p, 10, labels=False, duplicates="drop")
        dect = [dict(decile=int(k), n=int((dec == k).sum()), pred_bp=1e4 * float(p[dec == k].mean()), long_gross_bp=1e4 * float(y[dec == k].mean()), long_net_bp=1e4 * float((y - cl)[dec == k].mean()), inv_gross_bp=1e4 * float(yI[dec == k].mean()), inv_net_bp=1e4 * float((yI - ci)[dec == k].mean())) for k in sorted(set(dec))]
        diag = dict(n=int(te_all.sum()), corr=float(np.corrcoef(p, y)[0, 1]), sign_hit=float(np.mean(np.sign(p) == np.sign(y))), up_base=float(np.mean(y > 0)), vol_corr=float(np.corrcoef(pvol[te_all], yv[te_all])[0, 1]), vol_naive_corr=float(np.corrcoef(X[te_all][:, FEATS.index("vol15")], yv[te_all])[0, 1]), mean_cost_long_bp=1e4 * float(cl.mean()), mean_cost_inv_bp=1e4 * float(ci.mean()), share_pred_above_cost=float(np.mean(p > cl)), share_pred_below_minus_cost=float(np.mean(-p > ci)), deciles=dect)
        per_fold, all_model_trades = [], []
        for a, b in fl:
            rows = np.nonzero(lab & (days >= a) & (days < b))[0]
            if len(rows) == 0:
                continue
            r, t = simulate(rows, ent, ext, d, pred, "model")
            all_model_trades += t
            rnd, rnd_n = [], []
            if r["trades"]:
                for sd in range(SEEDS):
                    rr, _ = simulate(rows, ent, ext, d, pred, "random", np.random.default_rng(1000 * H + sd), (r["trades"], r["long_share"]))
                    rnd.append(rr["total_pct"])
                    rnd_n.append(rr["trades"])
            first, last = rows[0], rows[-1]
            hold = 100 * (O[last] / O[first] - 1)
            per_fold.append(dict(test=[a, b], model=r, random_mean_total_pct=float(np.mean(rnd)) if rnd else None, random_totals=rnd, random_trades_mean=float(np.mean(rnd_n)) if rnd_n else None, hold_pct=hold, days=int(len(set(days[rows])))))
        full_rows = np.nonzero(te_all)[0]
        full, full_t = simulate(full_rows, ent, ext, d, pred, "model")
        res["horizons"][str(H)] = dict(diagnostics=diag, folds=per_fold, full_window=full, trades=full_t)
        print(f"H{H} corr {diag['corr']:.4f} hit {diag['sign_hit']:.4f} base {diag['up_base']:.4f} volcorr {diag['vol_corr']:.3f} naive {diag['vol_naive_corr']:.3f} cost L {diag['mean_cost_long_bp']:.1f}bp I {diag['mean_cost_inv_bp']:.1f}bp above {diag['share_pred_above_cost']:.4f} below {diag['share_pred_below_minus_cost']:.4f}", flush=True)
        for r in dect:
            print(f"  dec {r['decile']} pred {r['pred_bp']:6.2f} L gross {r['long_gross_bp']:6.2f} net {r['long_net_bp']:6.2f} | I gross {r['inv_gross_bp']:6.2f} net {r['inv_net_bp']:6.2f}", flush=True)
        for f in per_fold:
            m = f["model"]
            print(f"  fold {f['test'][0]} trades {m['trades']:4d} mean {m['mean_net_bp'] if m['mean_net_bp'] is None else round(m['mean_net_bp'], 2)}bp total {m['total_pct']:7.2f}% vault {m['vault_pct']:6.2f}% random {f['random_mean_total_pct'] if f['random_mean_total_pct'] is None else round(f['random_mean_total_pct'], 2)} hold {f['hold_pct']:6.2f}", flush=True)
        print(f"  FULL trades {full['trades']} mean {full['mean_net_bp']}bp total {full['total_pct']:.2f}% vault {full['vault_pct']:.2f}% win {full['win']} long {full['long_share']}", flush=True)
    rec = dict(version="scalp-etf-v1", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], feats=FEATS, results=res)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
