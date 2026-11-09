import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
import math
import sys
import warnings
from datetime import date

sys.dont_write_bytecode = True
warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import pandas as pd
import lightgbm as lgb
import learn_inverse as LI
import scalp_etf_v1 as S1

TPS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.07)
SLS = (0.01, 0.03, 0.05)
ACTIONS = [(tp, sl) for tp in TPS for sl in SLS]
DECIDE_EVERY = 30
MAX_SESSIONS = 10
SEEDS = 20
CAPITAL = 100_000_000.0
FEATS = ["r5", "r30", "r120", "r_day", "r_1d", "r_5d", "v30", "v120", "v_today", "v_1d", "v_5d", "v_20d", "vr_short", "vr_day", "v_change", "gap_today", "tod", "dow", "next_gap_days", "pos_day_range", "dist_hi5", "dist_lo5", "mom20d", "mom60d"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load():
    d, _ = S1.load()
    d = d[(d["mod"] <= d["ce"]) & (d["volume"] > 0)].reset_index(drop=True)
    return d


def build(d):
    day = d["day"].to_numpy()
    O, H, Lo, C = (d[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    n = len(d)
    lc = np.log(C)
    same = np.r_[False, day[1:] == day[:-1]]
    r1 = np.where(same, np.r_[np.nan, np.diff(lc)], np.nan)
    udays = sorted(set(day))
    di = {u: k for k, u in enumerate(udays)}
    dk = np.array([di[x] for x in day])
    first = np.r_[0, np.nonzero(np.diff(dk))[0] + 1]
    last = np.r_[first[1:] - 1, n - 1]
    dclose = C[last]
    dopen = O[first]
    dvol = pd.Series(r1).groupby(dk).std().to_numpy()
    nd = len(udays)
    gaps_after = [(date.fromisoformat(udays[k + 1]) - date.fromisoformat(udays[k])).days if k + 1 < nd else 1 for k in range(nd)]
    s = pd.Series(lc)
    F = np.full((n, len(FEATS)), np.nan)
    col = {f: j for j, f in enumerate(FEATS)}
    for L, nm in ((5, "r5"), (30, "r30"), (120, "r120")):
        F[:, col[nm]] = (s - s.shift(L)).to_numpy()
    rs = pd.Series(r1)
    F[:, col["v30"]] = rs.rolling(30, min_periods=20).std().to_numpy()
    F[:, col["v120"]] = rs.rolling(120, min_periods=80).std().to_numpy()
    F[:, col["v_today"]] = rs.groupby(dk).transform(lambda x: x.expanding(10).std()).to_numpy()
    pd_close = np.r_[np.nan, dclose[:-1]]
    F[:, col["r_day"]] = lc - np.log(dopen[dk])
    F[:, col["gap_today"]] = np.log(dopen / pd_close)[dk]
    for nm, k in (("r_1d", 1), ("r_5d", 5)):
        ref = np.r_[[np.nan] * (k + 1), dclose[:-(k + 1)]] if nd > k + 1 else np.full(nd, np.nan)
        F[:, col[nm]] = lc - np.log(ref[dk])
    v1 = np.r_[np.nan, dvol[:-1]]
    v5 = np.array([np.nanmean(dvol[max(0, k - 5):k]) if k >= 2 else np.nan for k in range(nd)])
    v20 = np.array([np.nanmean(dvol[max(0, k - 20):k]) if k >= 5 else np.nan for k in range(nd)])
    F[:, col["v_1d"]] = v1[dk]
    F[:, col["v_5d"]] = v5[dk]
    F[:, col["v_20d"]] = v20[dk]
    F[:, col["vr_short"]] = F[:, col["v30"]] / F[:, col["v120"]]
    F[:, col["vr_day"]] = v5[dk] / v20[dk]
    F[:, col["v_change"]] = F[:, col["v_today"]] / v1[dk]
    F[:, col["tod"]] = (d["mod"] - d["start"]).to_numpy(float)
    F[:, col["dow"]] = np.array([date.fromisoformat(u).weekday() for u in udays], float)[dk]
    F[:, col["next_gap_days"]] = np.array(gaps_after, float)[dk]
    hi = pd.Series(H).groupby(dk).cummax().to_numpy()
    lo = pd.Series(Lo).groupby(dk).cummin().to_numpy()
    F[:, col["pos_day_range"]] = np.where(hi > lo, (C - lo) / np.maximum(hi - lo, 1e-9), 0.5)
    dhi = pd.Series(pd.Series(H).groupby(dk).max().to_numpy())
    dlo = pd.Series(pd.Series(Lo).groupby(dk).min().to_numpy())
    hi5 = dhi.shift(1).rolling(5, min_periods=3).max().to_numpy()
    lo5 = dlo.shift(1).rolling(5, min_periods=3).min().to_numpy()
    F[:, col["dist_hi5"]] = lc - np.log(hi5[dk])
    F[:, col["dist_lo5"]] = lc - np.log(lo5[dk])
    m20 = np.r_[[np.nan] * 21, dclose[:-21]] if nd > 21 else np.full(nd, np.nan)
    m60 = np.r_[[np.nan] * 61, dclose[:-61]] if nd > 61 else np.full(nd, np.nan)
    F[:, col["mom20d"]] = (lc - np.log(m20[dk]))
    F[:, col["mom60d"]] = (lc - np.log(m60[dk]))
    mod, ce, start = d["mod"].to_numpy(), d["ce"].to_numpy(), d["start"].to_numpy()
    dec = np.nonzero(((mod - start) % DECIDE_EVERY == 0) & (mod <= ce - DECIDE_EVERY) & (dk >= 61))[0]
    dec = dec[dec + 1 < n]
    return dict(day=day, dk=dk, first=first, last=last, O=O, H=H, L=Lo, C=C, F=F, dec=dec, udays=udays)


def outcomes(b):
    O, H, Lo, C, dk, last = b["O"], b["H"], b["L"], b["C"], b["dk"], b["last"]
    n = len(C)
    R = np.full((len(b["dec"]), len(ACTIONS)), np.nan)
    X = np.full((len(b["dec"]), len(ACTIONS)), -1, dtype=np.int64)
    first_min = np.r_[True, dk[1:] != dk[:-1]]
    for q, i in enumerate(b["dec"]):
        e = i + 1
        end_day = min(dk[e] + MAX_SESSIONS - 1, dk[-1])
        stop = last[end_day]
        if stop <= e:
            continue
        pin = O[e]
        cin = LI.FEE + LI.TICK / pin
        h = H[e:stop + 1]
        l = Lo[e:stop + 1]
        o = O[e:stop + 1]
        fm = first_min[e:stop + 1]
        for a, (tp, sl) in enumerate(ACTIONS):
            up = pin * (1 + tp + 2 * cin)
            dn = pin * (1 - sl)
            hit_up = h >= up
            hit_dn = l <= dn
            ku = int(np.argmax(hit_up)) if hit_up.any() else None
            kd = int(np.argmax(hit_dn)) if hit_dn.any() else None
            if kd is not None and (ku is None or kd <= ku):
                k = kd
                px = min(dn, o[k]) if fm[k] and o[k] < dn else dn
                px -= LI.TICK
            elif ku is not None:
                k = ku
                px = max(up, o[k]) if fm[k] and o[k] > up else up
            else:
                k = stop - e
                px = C[stop] - LI.TICK
            cout = LI.FEE
            R[q, a] = math.log(px * (1 - cout) / (pin * (1 + cin)))
            X[q, a] = e + k
    return R, X


def quarters():
    out = []
    for y in (2024, 2025, 2026):
        for qq in range(4):
            a = f"{y}-{3 * qq + 1:02d}-01"
            b = f"{y + (1 if qq == 3 else 0)}-{(3 * qq + 4 - 1) % 12 + 1:02d}-01"
            if a <= S1.EVAL_TO:
                out.append((a, b))
    return out


def design(Fd):
    A = np.array(ACTIONS)
    m = len(Fd)
    Xs = np.repeat(Fd, len(ACTIONS), axis=0)
    Xa = np.tile(A, (m, 1))
    return np.hstack([Xs, Xa])


def simulate(b, dec_days, pick, R, X, idx_rows):
    cash = CAPITAL
    busy = -1
    trades, chosen = [], []
    for q in idx_rows:
        i = b["dec"][q]
        if i <= busy:
            continue
        a = pick(q)
        if a is None or np.isnan(R[q, a]):
            continue
        cash *= math.exp(R[q, a])
        trades.append(R[q, a])
        chosen.append(a)
        busy = X[q, a]
    return dict(total_pct=100 * (cash / CAPITAL - 1), trades=len(trades), mean_bp=1e4 * float(np.mean(trades)) if trades else None, win=float(np.mean(np.array(trades) > 0)) if trades else None), chosen


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, pth in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), pth)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    d = load()
    b = build(d)
    R, X = outcomes(b)
    Fd = b["F"][b["dec"]]
    dday = b["day"][b["dec"]]
    print("decision points", len(b["dec"]), "actions", len(ACTIONS), "outcome NaN share", float(np.isnan(R).mean()), flush=True)
    res = dict(quarters=[], actions=ACTIONS)
    pred = np.full(R.shape, np.nan)
    for a_, b_ in quarters():
        tr = np.nonzero((dday < a_) & ~np.isnan(R).any(axis=1))[0]
        last_ok = [q for q in tr if X[q].max() < np.searchsorted(b["day"], a_)]
        tr = np.array(last_ok, dtype=int)
        te = np.nonzero((dday >= a_) & (dday < b_))[0]
        if len(te) == 0 or len(tr) < 500:
            continue
        m = lgb.train(LI.PARAMS, lgb.Dataset(design(Fd[tr]), label=R[tr].ravel()), num_boost_round=LI.ROUNDS)
        pred[te] = m.predict(design(Fd[te])).reshape(len(te), len(ACTIONS))
        fixed = int(np.nanargmax(np.nanmean(R[tr], axis=0)))
        res["quarters"].append(dict(test=[a_, b_], train_points=int(len(tr)), test_points=int(len(te)), fixed_best_action=ACTIONS[fixed]))
        print("fold", a_, "train", len(tr), "test", len(te), "fixed best", ACTIONS[fixed], flush=True)
    te_all = np.nonzero(~np.isnan(pred).all(axis=1))[0]
    out = {}
    pick_rl = lambda q: int(np.nanargmax(pred[q])) if np.nanmax(pred[q]) > 0 else None
    r, ch = simulate(b, dday, pick_rl, R, X, te_all)
    out["RL_fitted_Q"] = r
    hist = {}
    for a in ch:
        hist[str(ACTIONS[a])] = hist.get(str(ACTIONS[a]), 0) + 1
    out["RL_fitted_Q"]["chosen_actions"] = hist
    fixed_by_q = {}
    for qd in res["quarters"]:
        for q in np.nonzero((dday >= qd["test"][0]) & (dday < qd["test"][1]))[0]:
            fixed_by_q[q] = ACTIONS.index(tuple(qd["fixed_best_action"]))
    out["fixed_best_in_train"], _ = simulate(b, dday, lambda q: fixed_by_q.get(q), R, X, te_all)
    rnd = []
    for sd in range(SEEDS):
        rng = np.random.default_rng(900 + sd)
        rr, _ = simulate(b, dday, lambda q: int(rng.integers(len(ACTIONS))), R, X, te_all)
        rnd.append(rr["total_pct"])
    out["random_action_mean_total_pct"] = float(np.mean(rnd))
    out["random_action_totals"] = rnd
    for a, act in enumerate(ACTIONS):
        rr, _ = simulate(b, dday, lambda q, a=a: a, R, X, te_all)
        out[f"always_{act}"] = rr
    i0, i1 = b["dec"][te_all[0]] + 1, len(b["C"]) - 1
    out["hold_pct"] = 100 * (b["C"][i1] / b["O"][i0] - 1)
    corr = []
    for q in te_all:
        if not np.isnan(R[q]).any():
            corr.append(np.corrcoef(pred[q], R[q])[0, 1])
    out["mean_within_point_rank_corr_pred_vs_real"] = float(np.nanmean(corr))
    best_real = np.nanargmax(R[te_all], axis=1)
    best_pred = np.nanargmax(pred[te_all], axis=1)
    out["share_pred_best_equals_real_best"] = float(np.mean(best_real == best_pred))
    out["share_real_best_by_action"] = {str(ACTIONS[a]): float(np.mean(best_real == a)) for a in range(len(ACTIONS))}
    print("RESULTS", json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "chosen_actions"}) for k, v in out.items() if not k.startswith("always_") and k != "random_action_totals"}, default=float), flush=True)
    print("RL chosen actions", hist, flush=True)
    print("always-action totals", {k[7:]: round(v["total_pct"], 1) for k, v in out.items() if k.startswith("always_")}, flush=True)
    res["test"] = out
    rec = dict(version="rl-band-v1", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], features=FEATS, results=res)
    with open(out_path, "x", encoding="utf-8") as fo:
        json.dump(rec, fo, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
