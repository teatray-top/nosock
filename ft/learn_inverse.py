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
import numpy as np
import lightgbm as lgb
import compare_trend_v2 as CT
import compare_inverse as CI

L, S = CI.L, CI.S
H = 20
FEE, TICK = CT.FEE, CT.SLIP
LAST = CT.LAST_COMPLETE
FIRST_YEAR, LAST_YEAR = 2013, 2026
FEAT_START = 250
PARAMS = dict(objective="regression", num_leaves=7, max_depth=3, learning_rate=0.05, min_data_in_leaf=200, lambda_l2=1.0, bagging_fraction=1.0, feature_fraction=1.0, deterministic=True, force_col_wise=True, num_threads=1, seed=0, verbose=-1)
ROUNDS = 200
BAND = 2 * (FEE + TICK / 80000.0) * 2
FEATS = ["r1", "r5", "r20", "r60", "r120", "r250", "dd250", "ma20", "ma50", "ma200", "vol20", "vol60", "volratio", "e8_21", "e21_50", "e50_200", "up20"]
ADJ_FILE = CT.ADJ
INV_DIR_DEFAULT = r"G:\NoSock\stock\penny\data\kr_etf\inverse_114800_20261107_151154"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load(inv_dir):
    raw = json.load(open(CI.RAW, encoding="utf-8"))[L]
    adj = json.load(open(ADJ_FILE, encoding="utf-8"))[L]
    days = [d for d in raw["days"] if d <= LAST]
    if [d for d in adj["days"] if d <= LAST] != days or days[0] != CT.ORIGIN[L]:
        raise SystemExit("069500 raw/adjusted calendars differ")
    n = len(days)
    k = dict(O=[float(x) for x in raw["O"][:n]], C=[float(x) for x in raw["C"][:n]], Oa=[float(x) for x in adj["O"][:n]], Ca=[float(x) for x in adj["C"][:n]])
    iv = json.load(open(os.path.join(inv_dir, "daily_raw.json"), encoding="utf-8"))[S]
    inv = {d: float(o) for d, o in zip(iv["days"], iv["O"]) if d <= LAST}
    return days, k, inv


def ema(xs, n):
    return CT.ema(xs, n)


def features(days, k):
    C = k["Ca"]
    e = {m: ema(C, m) for m in (8, 21, 50, 200)}
    lr = [0.0] + [math.log(C[i] / C[i - 1]) for i in range(1, len(C))]
    out = {}
    for i in range(FEAT_START, len(C)):
        w250 = C[i - 249:i + 1]
        r20 = lr[i - 19:i + 1]
        r60 = lr[i - 59:i + 1]
        sd = lambda xs: float(np.std(xs))
        v20, v60 = sd(r20), sd(r60)
        out[i] = [math.log(C[i] / C[i - m]) for m in (1, 5, 20, 60, 120, 250)] + [C[i] / max(w250) - 1, C[i] / (sum(C[i - 19:i + 1]) / 20) - 1, C[i] / (sum(C[i - 49:i + 1]) / 50) - 1, C[i] / (sum(C[i - 199:i + 1]) / 200) - 1, v20, v60, v20 / v60 if v60 > 0 else 1.0, e[8][i] / e[21][i] - 1, e[21][i] / e[50][i] - 1, e[50][i] / e[200][i] - 1, sum(1 for x in r20 if x > 0) / 20.0]
    return out


def label_long(k, i):
    b, s = k["O"][i + 1] + TICK, k["O"][i + H + 1] - TICK
    fac = (k["Oa"][i + H + 1] / k["O"][i + H + 1]) / (k["Oa"][i + 1] / k["O"][i + 1])
    return s * (1 - FEE) / (b * (1 + FEE)) * fac - 1


def label_inv(days, inv, i):
    d1, d2 = days[i + 1], days[i + H + 1]
    if d1 not in inv or d2 not in inv:
        return None
    b, s = inv[d1] + TICK, inv[d2] - TICK
    return s * (1 - FEE) / (b * (1 + FEE)) - 1


def folds(days):
    out = []
    n = len(days)
    for y in range(FIRST_YEAR, LAST_YEAR + 1):
        e = next(j for j, d in enumerate(days) if d >= f"{y}-01-01")
        nxt = next((j for j, d in enumerate(days) if d >= f"{y + 1}-01-01"), None)
        c = e - 1
        last_dec = (nxt - 2) if nxt is not None else n - 1
        out.append(dict(year=y, cutoff=c, cutoff_day=days[c], first_dec=c, last_dec=last_dec, train_last=c - H - 1))
    return out


def fit(X, y):
    ds = lgb.Dataset(np.asarray(X, dtype=np.float64), label=np.asarray(y, dtype=np.float64), free_raw_data=False)
    return lgb.train(PARAMS, ds, num_boost_round=ROUNDS)


def build(days, k, inv):
    F = features(days, k)
    preds, fold_info = {}, []
    for f in folds(days):
        jl = [j for j in range(FEAT_START, f["train_last"] + 1)]
        js = [j for j in jl if label_inv(days, inv, j) is not None]
        for j in jl + js:
            if j + H + 1 > f["cutoff"]:
                raise SystemExit("unmatured label in training")
        ml = fit([F[j] for j in jl], [label_long(k, j) for j in jl])
        ms = fit([F[j] for j in js], [label_inv(days, inv, j) for j in js])
        dec = list(range(f["first_dec"], f["last_dec"] + 1))
        Xd = np.asarray([F[i] for i in dec], dtype=np.float64)
        pl, ps = ml.predict(Xd), ms.predict(Xd)
        for i, a, b in zip(dec, pl, ps):
            preds[days[i]] = (float(a), float(b))
        ylong = [label_long(k, j) for j in jl]
        yinv = [label_inv(days, inv, j) for j in js]
        fold_info.append(dict(year=f["year"], cutoff_day=f["cutoff_day"], train_long=len(jl), train_inv=len(js), train_inv_first=days[js[0]] if js else None, label_long_mean=float(np.mean(ylong)), label_long_pos=float(np.mean([v > 0 for v in ylong])), label_inv_mean=float(np.mean(yinv)) if yinv else None, decisions=len(dec), pred_long_mean=float(np.mean(pl)), pred_inv_mean=float(np.mean(ps)), pred_long_pos=float(np.mean(pl > 0)), pred_inv_gt_long_and_0=float(np.mean((ps > pl) & (ps > 0))), importance_long=dict(zip(FEATS, [int(v) for v in ml.feature_importance("gain")])), importance_inv=dict(zip(FEATS, [int(v) for v in ms.feature_importance("gain")]))))
    return preds, fold_info


def policy(preds, allow_inv):
    out, state = {}, None
    for d in sorted(preds):
        pl, ps = preds[d]
        v = {None: 0.0, L: pl}
        if allow_inv:
            v[S] = ps
        best = max(v, key=lambda a: v[a])
        if state in v and v[state] >= v[best] - BAND:
            best = state
        out[d] = best
        state = best
    return out


def main(out_path, spec_path, spec_sha, inv_dir=INV_DIR_DEFAULT):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for p, key in ((CI.RAW, "kodex_daily_raw_sha256"), (ADJ_FILE, "kodex_daily_adj_sha256"), (os.path.join(inv_dir, "daily_raw.json"), "inverse_daily_raw_sha256"), (os.path.join(inv_dir, S + ".csv"), "inverse_minute_sha256"), (CT.CAL, "calendar_sha256"), (CT.SNAP[L][0], "kodex_minute_sha256"), (os.path.join(HERE, "compare_inverse.py"), "compare_inverse_sha256"), (os.path.join(HERE, "compare_trend_v2.py"), "compare_trend_v2_sha256")):
        if sha(p) != spec["inputs"][key]:
            raise SystemExit(f"input hash differs: {key}")
    days, k, inv = load(inv_dir)
    preds, fold_info = build(days, k, inv)
    tg_rule, all_days = CI.targets()
    tg = {"hold": tg_rule["hold"], "ribbon": tg_rule["ribbon"], "ribbon_inv": tg_rule["ribbon_inv"], "learn_long": policy(preds, False), "learn_inv": policy(preds, True)}
    methods = list(tg)
    prev = {d: all_days[i - 1] for i, d in enumerate(all_days) if i > 0}
    cal = json.load(open(CT.CAL, encoding="utf-8"))["days"]
    msess = {L: CI.minute_sessions(CT.SNAP[L][0], cal), S: CI.minute_sessions(os.path.join(inv_dir, S + ".csv"), cal)}
    res = {}

    def block(days_b, sess, fill_of):
        sims = {m: CI.simulate(days_b, sess, (lambda d, m=m: tg[m][prev[d]]), fill_of) for m in methods}
        hf = sims["hold"]["final"]
        return dict(window=[days_b[0], days_b[-1]], sessions=len(days_b), methods={m: CI.report(sims[m], hf) for m in methods})

    for name, (a, b) in CT.WINDOWS.items():
        dd = [d for d in sorted(msess[L]) if a <= d <= b and prev.get(d) in preds]
        res[f"minute_{name}"] = block(dd, msess, lambda leg, d, after: CI.first_fill(msess[leg][d], after))
    kr = json.load(open(CI.RAW, encoding="utf-8"))[L]
    ir = json.load(open(os.path.join(inv_dir, "daily_raw.json"), encoding="utf-8"))[S]
    dsess = {L: {}, S: {}}
    for sym, src in ((L, kr), (S, ir)):
        for i, d in enumerate(src["days"]):
            if d <= LAST:
                dsess[sym][d] = dict(open=float(src["O"][i]), close=float(src["C"][i]), vol=float(src["V"][i]))
    ddays = sorted(d for d in set(dsess[L]) & set(dsess[S]) if prev.get(d) in preds)
    res["daily_open"] = block(ddays, dsess, lambda leg, d, after: (dsess[leg][d]["open"], "09:00") if dsess[leg][d]["vol"] > 0 and dsess[leg][d]["open"] > 0 else None)
    last_day = max(preds)
    rec = dict(version="learn-inverse-v1", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], params=PARAMS, rounds=ROUNDS, horizon=H, band=BAND, features=FEATS, folds=fold_info, latest_decision=dict(day=last_day, pred_long=preds[last_day][0], pred_inv=preds[last_day][1], learn_long=tg["learn_long"][last_day], learn_inv=tg["learn_inv"][last_day]), results=res)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    for fi in fold_info:
        print(f"fold {fi['year']} train L {fi['train_long']} S {fi['train_inv']} | label L mean {100 * fi['label_long_mean']:+.2f}% pos {100 * fi['label_long_pos']:.0f}% S mean {100 * (fi['label_inv_mean'] or 0):+.2f}% | pred L mean {100 * fi['pred_long_mean']:+.2f}% pos {100 * fi['pred_long_pos']:.0f}% S>L,0 {100 * fi['pred_inv_gt_long_and_0']:.0f}%")
    for kk, v in res.items():
        print(f"== {kk} {v['window'][0]}..{v['window'][1]} sessions {v['sessions']}")
        for m in methods:
            r = v["methods"][m]
            print(f"  {m:11s} return {r['return_pct']:8.2f}% vs_hold {r['vs_hold_pct']:8.2f} maxDD {r['max_drawdown_daily_pct']:7.2f}% buys {r['buys']:3d} inv {r['inverse_buys']:3d} days L/S/C {r['days_long']}/{r['days_inverse']}/{r['days_cash']} years {r['by_year_pct']}")
    print("latest", rec["latest_decision"])
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
