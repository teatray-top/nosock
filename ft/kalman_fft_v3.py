import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
import sys
from collections import deque

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import learn_inverse as LI
import scenario_test_v1 as SC
import harvest_v5 as H5
import harvest_v8 as H8
import kalman_fft_v1 as K1
import kalman_fft_v2 as K2

CAPITAL = H5.CAPITAL
VAULT_SHARE = H5.VAULT_SHARE
CFGS = [(w, k, lam) for lam in K2.LAMBDAS for w in K2.WINDOWS for k in K2.HARMONICS]
ONLINE_N = 500
MIN_HIST = 100


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def run(target, ok, O, IO, cost):
    n = len(target)
    cash, vault, leg, q, basis = CAPITAL, 0.0, None, 0.0, 0.0
    nets = []
    series = np.full(n, np.nan)
    peak, mdd, low = CAPITAL, 0.0, CAPITAL

    def close(e):
        nonlocal cash, vault, leg, q, basis
        P = O if leg == "L" else IO
        proceeds = q * P[e] * (1 - cost(leg, e))
        profit = proceeds - basis
        tv = VAULT_SHARE * profit if profit > 0 else 0.0
        vault += tv
        cash += proceeds - tv
        nets.append((leg, proceeds / basis - 1))
        leg, q, basis = None, 0.0, 0.0

    for t in range(n - 1):
        want = target[t] if ok[t] else None
        if want != leg:
            if leg is not None:
                close(t + 1)
            if want is not None:
                P = O if want == "L" else IO
                q = cash * (1 - cost(want, t + 1)) / P[t + 1]
                basis, cash, leg = cash, 0.0, want
    if leg is not None:
        close(n - 1)
    r = [x[1] for x in nets]
    return dict(total_pct=100 * ((cash + vault) / CAPITAL - 1), vault_pct=100 * vault / CAPITAL, trades=len(nets), mean_net_bp=1e4 * float(np.mean(r)) if r else None)


def run_marked(target, ok, O, C, IO, IC, cost, start):
    n = len(target)
    cash, vault, leg, q, basis = CAPITAL, 0.0, None, 0.0, 0.0
    peak, mdd, low = CAPITAL, 0.0, CAPITAL
    trades = 0
    for t in range(start, n - 1):
        want = target[t] if ok[t] else None
        if want != leg:
            e = t + 1
            if leg is not None:
                P = O if leg == "L" else IO
                proceeds = q * P[e] * (1 - cost(leg, e))
                profit = proceeds - basis
                tv = VAULT_SHARE * profit if profit > 0 else 0.0
                vault += tv
                cash += proceeds - tv
                leg, q, basis = None, 0.0, 0.0
                trades += 1
            if want is not None:
                P = O if want == "L" else IO
                q = cash * (1 - cost(want, e)) / P[e]
                basis, cash, leg = cash, 0.0, want
        M = (C if leg == "L" else IC) if leg else None
        tot = cash + (q * M[t + 1] if leg else 0.0) + vault
        peak = max(peak, tot)
        mdd = min(mdd, tot / peak - 1)
        low = min(low, tot)
    if leg is not None:
        M = C if leg == "L" else IC
        P = M
        proceeds = q * P[n - 1] * (1 - cost(leg, n - 1))
        profit = proceeds - basis
        tv = VAULT_SHARE * profit if profit > 0 else 0.0
        vault += tv
        cash += proceeds - tv
        trades += 1
    tot = cash + vault
    return dict(total_pct=100 * (tot / CAPITAL - 1), vault_pct=100 * vault / CAPITAL, mdd_pct=100 * min(mdd, tot / peak - 1), lowest_pct=100 * (min(low, tot) / CAPITAL - 1), trades=trades)


def online_targets(O, C, mode, rule="d1"):
    y = np.log(C)
    pos = np.arange(len(C))
    lev = {lam: K2.kalman_all(y, lam)[0] for lam in K2.LAMBDAS}
    feats = [K1.fft_features(lev[c[2]], pos, c[0], c[1]) for c in CFGS]
    tgs = [K2.targets(f, rule, mode) for f in feats]
    nr = np.r_[np.diff(np.log(O))[1:], [np.nan, np.nan]][:len(O)]
    dq = [deque(maxlen=ONLINE_N) for _ in CFGS]
    sums = np.zeros(len(CFGS))
    tg = np.array([None] * len(C), dtype=object)
    for t in range(len(C)):
        if t >= 2:
            s = t - 2
            for ci in range(len(CFGS)):
                v = feats[ci][s, 5]
                if np.isnan(v) or np.isnan(nr[s]):
                    continue
                h = 1.0 if np.sign(v) == np.sign(nr[s]) else 0.0
                if len(dq[ci]) == ONLINE_N:
                    sums[ci] -= dq[ci][0]
                dq[ci].append(h)
                sums[ci] += h
        sc = np.array([sums[ci] / len(dq[ci]) if len(dq[ci]) >= MIN_HIST else -1 for ci in range(len(CFGS))])
        b = int(sc.argmax())
        if sc[b] >= 0:
            tg[t] = tgs[b][t]
    return tg


def shifted(tg, ok, seed):
    idx = np.nonzero(ok)[0]
    tr = tg.copy()
    if len(idx) > 10:
        rng = np.random.default_rng(seed)
        tr[idx] = np.roll(tg[idx], int(rng.integers(len(idx) // 10, 9 * len(idx) // 10)))
    return tr


def window_eval(O, C, IO, IC, cost, start, seeds=5):
    out = {}
    ok = np.zeros(len(C), bool)
    ok[start:] = True
    hold = np.array(["L"] * len(C), dtype=object)
    out["hold"] = run_marked(hold, ok, O, C, IO, IC, cost, start)
    for mode in ("long_only", "long_inverse"):
        tg = online_targets(O, C, mode)
        okm = ok & np.array([x is not None or True for x in tg])
        r = run_marked(tg, ok, O, C, IO, IC, cost, start)
        rnd = [run_marked(shifted(tg, ok, 900 + s), ok, O, C, IO, IC, cost, start)["total_pct"] for s in range(seeds)]
        out[f"kfft_online_d1_{mode}"] = dict(r, random_shift_mean_total_pct=float(np.mean(rnd)))
    return out


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, p in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), p)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    res = {}
    raw = json.load(open(os.path.join(os.path.dirname(HERE), "data", "kr_etf", "kodex_daily_raw.json"), encoding="utf-8"))[H5.LONG]
    days = raw["days"]
    a = days.index(next(d for d in days if d >= "2007-01-02"))
    b = days.index(next(d for d in days if d >= "2012-12-28"))
    pre = 400
    O = np.asarray(raw["O"][a - pre:b + 1], float)
    C = np.asarray(raw["C"][a - pre:b + 1], float)
    IO, IC = SC.inverse_path(O, C)
    cost = lambda leg, j: LI.FEE + LI.TICK / (O[j] if leg == "L" else IO[j])
    res["crisis_2007_2012"] = dict(window=[days[a], days[b]], note="history from 400 sessions before the window for the filter and the online selection; synthetic -1x inverse", results=window_eval(O, C, IO, IC, cost, pre))
    print("CRISIS", {k: (round(v["total_pct"], 1), round(v.get("random_shift_mean_total_pct", float("nan")), 1), round(v["lowest_pct"], 1), v["trades"]) for k, v in res["crisis_2007_2012"]["results"].items()}, flush=True)
    dd, Od, Cd, IOd, ICd, tf = K2.daily_bars()
    st = int(np.argmax(dd >= tf))
    cost2 = lambda leg, j: LI.FEE + (LI.TICK if leg == "L" else H8.tick(dd[j], IOd[j])) / (Od[j] if leg == "L" else IOd[j])
    res["real_2013_2026"] = dict(window=[dd[st], dd[-1]], results=window_eval(Od, Cd, IOd, ICd, cost2, st))
    print("REAL", {k: (round(v["total_pct"], 1), round(v.get("random_shift_mean_total_pct", float("nan")), 1), round(v["lowest_pct"], 1), v["trades"]) for k, v in res["real_2013_2026"]["results"].items()}, flush=True)
    rows = []
    for trend in SC.TRENDS:
        for case in SC.LIQ_CASES:
            for seed in range(SC.SEEDS):
                O_, C_, liq = SC.generate(trend, case, seed)
                IO_, IC_ = SC.inverse_path(O_, C_)
                c3 = lambda leg, j: LI.FEE + liq[j]["cost_bp"] / 10000.0
                r = window_eval(np.asarray(O_, float), np.asarray(C_, float), IO_, IC_, c3, SC.WARM, seeds=2)
                rows.append(dict(trend=trend, liquidity=case, seed=seed, res=r))
    summ = {}
    for t in list(SC.TRENDS) + ["all"]:
        ps = [r for r in rows if t == "all" or r["trend"] == t]
        summ[t] = {}
        for k in ps[0]["res"]:
            v = [r["res"][k]["total_pct"] for r in ps]
            summ[t][k] = dict(median_total_pct=float(np.median(v)), made_money=int(sum(x > 0 for x in v)), beat_hold=int(sum(r["res"][k]["total_pct"] > r["res"]["hold"]["total_pct"] for r in ps)), median_lowest_pct=float(np.median([r["res"][k]["lowest_pct"] for r in ps])), median_random_shift=float(np.median([r["res"][k].get("random_shift_mean_total_pct", np.nan) for r in ps])) if k != "hold" else None, paths=len(ps))
            s = summ[t][k]
            print(f"SYN {t:8s} {k:28s} median {s['median_total_pct']:7.1f} random {s['median_random_shift']} made {s['made_money']}/{s['paths']} beat hold {s['beat_hold']} lowest {s['median_lowest_pct']:6.1f}", flush=True)
    res["synthetic"] = dict(summary=summ, paths=rows)
    rec = dict(version="kalman-fft-v3", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], results=res)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
