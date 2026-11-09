import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import pandas as pd
import lightgbm as lgb
import learn_inverse as LI
import scalp_etf_v1 as S1
import harvest_v8 as H8

THETAS = (0.003, 0.005)
CAPITAL = 100_000_000.0
VAULT_SHARE = 0.5
SEEDS = 20
PARAMS = dict(LI.PARAMS, objective="binary")
MARGINS = (0.1, 0.2)
SFEATS = ["since_pivot", "from_pivot", "last_leg", "last_leg_len", "run_max_dd", "run_min_up", "leg_dir"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def hindsight_legs(C, day, theta):
    n = len(C)
    leg = np.zeros(n)
    pivots = []
    start = 0
    while start < n:
        end = start
        while end + 1 < n and day[end + 1] == day[start]:
            end += 1
        seg = C[start:end + 1]
        piv = [0]
        direction = 0
        ext = 0
        for i in range(1, len(seg)):
            if direction >= 0:
                if seg[i] > seg[ext]:
                    ext = i
                elif seg[i] <= seg[ext] * (1 - theta) and (direction == 1 or seg[ext] >= seg[piv[-1]] * (1 + theta) or direction == 0):
                    if direction == 0 and seg[ext] < seg[piv[-1]] * (1 + theta):
                        direction = -1
                        ext = i
                        continue
                    piv.append(ext)
                    direction = -1
                    ext = i
                    continue
            if direction <= 0:
                if seg[i] < seg[ext]:
                    ext = i
                elif seg[i] >= seg[ext] * (1 + theta):
                    if direction == 0:
                        direction = 1
                        ext = i
                        continue
                    piv.append(ext)
                    direction = 1
                    ext = i
        piv.append(len(seg) - 1)
        piv = sorted(set(piv))
        for a, b in zip(piv[:-1], piv[1:]):
            leg[start + a:start + b] = 1.0 if seg[b] > seg[a] else 0.0
        pivots += [start + p for p in piv]
        start = end + 1
    return leg, pivots


def causal_state(C, day, theta):
    n = len(C)
    out = np.zeros((n, len(SFEATS)))
    conf = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and day[j + 1] == day[i]:
            j += 1
        direction, ext, piv, piv_t, last_leg, last_len = 0, i, C[i], i, 0.0, 0.0
        hi, lo = C[i], C[i]
        for t in range(i, j + 1):
            c = C[t]
            hi, lo = max(hi, c), min(lo, c)
            if direction >= 0 and c > C[ext]:
                ext = t if direction == 1 else ext
            if direction <= 0 and c < C[ext]:
                ext = t if direction == -1 else ext
            if direction == 0:
                if c >= lo * (1 + theta):
                    direction, last_leg, last_len, piv, piv_t, ext = 1, c / lo - 1, 0.0, lo, t, t
                elif c <= hi * (1 - theta):
                    direction, last_leg, last_len, piv, piv_t, ext = -1, c / hi - 1, 0.0, hi, t, t
            elif direction == 1:
                if c > C[ext]:
                    ext = t
                if c <= C[ext] * (1 - theta):
                    last_leg, last_len = C[ext] / piv - 1, float(ext - piv_t)
                    direction, piv, piv_t, ext = -1, C[ext], ext, t
            else:
                if c < C[ext]:
                    ext = t
                if c >= C[ext] * (1 + theta):
                    last_leg, last_len = C[ext] / piv - 1, float(ext - piv_t)
                    direction, piv, piv_t, ext = 1, C[ext], ext, t
            conf[t] = direction
            out[t] = [t - piv_t, c / piv - 1, last_leg, last_len, c / C[ext] - 1 if direction == 1 else 0.0, c / C[ext] - 1 if direction == -1 else 0.0, float(direction)]
        i = j + 1
    return out, conf


def simulate(d, ent_ok, target, nxt, O, IO):
    cash, vault = CAPITAL, 0.0
    leg, q, basis = None, 0.0, 0.0
    trades = []
    day = d["day"].to_numpy()
    days_sorted = None
    n = len(d)
    i = 0
    pending = None
    for t in range(n - 1):
        want = target[t] if ent_ok[t] else None
        last_of_day = t + 1 >= n or day[t + 1] != day[t]
        if leg is not None and (want != leg or last_of_day or not ent_ok[t]):
            e = nxt[t + 1] if not last_of_day and t + 1 < n else -1
            if e < 0:
                e = t
            P = O if leg == "L" else IO
            px = P[e]
            c = LI.FEE + H8.tick(day[e], px) / px if leg == "I" else LI.FEE + LI.TICK / px
            proceeds = q * px * (1 - c)
            profit = proceeds - basis
            tv = VAULT_SHARE * profit if profit > 0 else 0.0
            vault += tv
            cash += proceeds - tv
            trades.append(dict(leg=leg, exit=int(e), net=proceeds / basis - 1))
            leg, q, basis = None, 0.0, 0.0
        if leg is None and want is not None and not last_of_day:
            e = nxt[t + 1]
            if e < 0:
                continue
            P = O if want == "L" else IO
            px = P[e]
            c = LI.FEE + H8.tick(day[e], px) / px if want == "I" else LI.FEE + LI.TICK / px
            q = cash * (1 - c) / px
            basis, cash, leg = cash, 0.0, want
            trades[-1:] = trades[-1:]
    tot = cash + vault
    nets = [x["net"] for x in trades]
    return dict(total_pct=100 * (tot / CAPITAL - 1), vault_pct=100 * vault / CAPITAL, trades=len(trades), mean_net_bp=1e4 * float(np.mean(nets)) if nets else None, win=float(np.mean([x > 0 for x in nets])) if nets else None)


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, p in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), p)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    d, dropped = S1.load()
    ctx = S1.daily_context(d)
    base = S1.features(d, ctx).to_numpy(float)
    C = d["close"].to_numpy(float)
    O, IO = d["open"].to_numpy(float), d["io"].to_numpy(float)
    day = d["day"].to_numpy()
    nxt, _ = S1.fill_index(d)
    cont = (d["mod"] <= d["ce"]).to_numpy()
    fl = S1.folds(None)
    res = dict(rows=int(len(d)), days=int(len(set(day))), thetas={})
    for th in THETAS:
        leg, piv = hindsight_legs(C, day, th)
        sf, conf = causal_state(C, day, th)
        X = np.hstack([base, sf])
        ok = ~np.isnan(base).any(axis=1) & cont
        p = np.full(len(d), np.nan)
        for a, b in fl:
            tr = ok & (day < a)
            te = ok & (day >= a) & (day < b)
            if te.sum() == 0:
                continue
            m = lgb.train(PARAMS, lgb.Dataset(X[tr], label=leg[tr]), num_boost_round=LI.ROUNDS)
            p[te] = m.predict(X[te])
        test = ok & (day >= S1.TEST_FROM)
        auc_model = S1_auc(p[test], leg[test])
        auc_causal = S1_auc((conf[test] + 1) / 2, leg[test])
        leg_sizes = []
        pv = [x for x in piv if day[x] >= S1.TEST_FROM]
        for x0, x1 in zip(pv[:-1], pv[1:]):
            if day[x0] == day[x1]:
                leg_sizes.append(abs(C[x1] / C[x0] - 1))
        lag = []
        for x in pv:
            if not (0 < x < len(d) - 1) or day[x - 1] != day[x]:
                continue
            want_up = leg[x] == 1.0
            for y in range(x, min(len(d), x + 120)):
                if day[y] != day[x]:
                    break
                if (conf[y] == 1) == want_up and conf[y] != 0 and (y == x or conf[y - 1] != conf[y]):
                    lag.append((y - x, abs(C[y] / C[x] - 1)))
                    break
        out = dict(n_test=int(test.sum()), auc_model=auc_model, auc_causal_zigzag=auc_causal, up_leg_share=float(leg[test].mean()), legs_per_day=len(leg_sizes) / max(1, len(set(day[test]))), median_leg_pct=100 * float(np.median(leg_sizes)) if leg_sizes else None, causal_lag_minutes_median=float(np.median([z[0] for z in lag])) if lag else None, causal_lag_pct_median=100 * float(np.median([z[1] for z in lag])) if lag else None, policies={})
        print(f"theta {th}: aucModel {auc_model:.4f} aucCausal {auc_causal:.4f} upShare {out['up_leg_share']:.3f} legs/day {out['legs_per_day']:.1f} medLeg {out['median_leg_pct']:.3f}% causal lag {out['causal_lag_minutes_median']} min {out['causal_lag_pct_median']:.3f}%", flush=True)
        targets = {}
        targets["causal_zigzag"] = np.where(conf == 1, "L", np.where(conf == -1, "I", None))
        for mg in MARGINS:
            tg = np.array([None] * len(d), dtype=object)
            state = None
            for t in range(len(d)):
                if t > 0 and day[t] != day[t - 1]:
                    state = None
                if np.isnan(p[t]):
                    tg[t] = None
                    continue
                if p[t] > 0.5 + mg:
                    state = "L"
                elif p[t] < 0.5 - mg:
                    state = "I"
                tg[t] = state
            targets[f"model_m{int(mg * 100)}"] = tg
        for name, tg in targets.items():
            fold_rows = []
            for a, b in fl:
                sel = (day >= a) & (day < b)
                if not sel.any():
                    continue
                idx = np.nonzero(sel)[0]
                lo, hi = idx[0], idx[-1] + 1
                dd = d.iloc[lo:hi]
                r = simulate(dd, ok[lo:hi], tg[lo:hi], np.where(nxt[lo:hi] >= 0, nxt[lo:hi] - lo, -1), O[lo:hi], IO[lo:hi])
                rnd = []
                for sd in range(SEEDS):
                    rng = np.random.default_rng(7000 + sd)
                    flips = np.array(tg[lo:hi], dtype=object)
                    perm = np.array(["L" if z else "I" for z in rng.random(hi - lo) < 0.5], dtype=object)
                    rt = np.where(pd.isnull(flips), None, perm)
                    blocks = np.cumsum(np.r_[True, flips[1:] != flips[:-1]])
                    first = {}
                    for k2, bl in enumerate(blocks):
                        first.setdefault(bl, rt[k2])
                    rt2 = np.array([first[bl] if flips[k2] is not None else None for k2, bl in enumerate(blocks)], dtype=object)
                    rr = simulate(dd, ok[lo:hi], rt2, np.where(nxt[lo:hi] >= 0, nxt[lo:hi] - lo, -1), O[lo:hi], IO[lo:hi])
                    rnd.append(rr["total_pct"])
                fold_rows.append(dict(test=[a, b], model=r, random_mean_total_pct=float(np.mean(rnd)), random_totals=rnd))
            tot = float(np.prod([1 + f["model"]["total_pct"] / 100 for f in fold_rows]) * 100 - 100)
            rtot = float(np.prod([1 + f["random_mean_total_pct"] / 100 for f in fold_rows]) * 100 - 100)
            ntr = sum(f["model"]["trades"] for f in fold_rows)
            mbp = np.mean([f["model"]["mean_net_bp"] for f in fold_rows if f["model"]["mean_net_bp"] is not None])
            beat = sum(f["model"]["total_pct"] > f["random_mean_total_pct"] for f in fold_rows)
            out["policies"][name] = dict(folds=fold_rows, chained_total_pct=tot, random_chained_total_pct=rtot, trades=ntr, mean_fold_net_bp=float(mbp), folds_beating_random=beat, folds=len(fold_rows)) if False else dict(fold_results=fold_rows, chained_total_pct=tot, random_chained_total_pct=rtot, trades=ntr, mean_fold_net_bp=float(mbp), folds_beating_random=int(beat), n_folds=len(fold_rows))
            print(f"  {name:14s} trades {ntr:6d} mean {mbp:7.2f}bp chained {tot:8.2f}% random {rtot:8.2f}% beat {beat}/{len(fold_rows)}", flush=True)
        res["thetas"][str(th)] = out
    rec = dict(version="turn-scalp-v1", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], results=res)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


def S1_auc(p, y):
    m = ~np.isnan(p)
    p, y = p[m], y[m]
    order = np.argsort(p, kind="mergesort")
    r = np.empty(len(p))
    r[order] = np.arange(1, len(p) + 1)
    s = pd.Series(p)
    r = s.rank(method="average").to_numpy()
    pos = y == 1
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


if __name__ == "__main__":
    main(*sys.argv[1:4])
