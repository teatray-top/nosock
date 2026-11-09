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
import pandas as pd
import lightgbm as lgb
from numpy.lib.stride_tricks import sliding_window_view
import learn_inverse as LI
import scalp_etf_v1 as S1
import harvest_v8 as H8

WINDOWS = (16, 32, 64, 128)
HARMONICS = (1, 3)
LAMBDAS = (0.01, 0.1)
AHEAD = 60
HORIZONS = (10, 30)
ONLINE_N = 500
CAPITAL = 100_000_000.0
VAULT_SHARE = 0.5
SEEDS = 20
CHUNK = 20000


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def configs():
    return [(w, k, lam) for lam in LAMBDAS for w in WINDOWS for k in HARMONICS]


def cname(c):
    return f"W{c[0]}_K{c[1]}_L{c[2]}"


def kalman(y, day, lam):
    n = len(y)
    lev = np.empty(n)
    slo = np.empty(n)
    r = 1e-8
    q1, q2 = lam * r, lam * lam * r / 10.0
    l = s = 0.0
    p00 = p01 = p11 = 0.0
    for t in range(n):
        if t == 0 or day[t] != day[t - 1]:
            l, s = y[t], 0.0
            p00, p01, p11 = 100 * r, 0.0, r
        else:
            lp, sp = l + s, s
            a00 = p00 + 2 * p01 + p11 + q1
            a01 = p01 + p11
            a11 = p11 + q2
            S = a00 + r
            k0, k1 = a00 / S, a01 / S
            e = y[t] - lp
            l, s = lp + k0 * e, sp + k1 * e
            p00, p01, p11 = (1 - k0) * a00, (1 - k0) * a01, a11 - k1 * a01
        lev[t], slo[t] = l, s
    return lev, slo


def fft_features(lev, pos, W, K):
    n = len(lev)
    out = np.full((n, 9), np.nan)
    j = np.arange(W, dtype=float)
    jc = j - j.mean()
    vj = (jc ** 2).sum()
    fut = np.arange(W - 1, W + AHEAD, dtype=float)
    win = sliding_window_view(lev, W)
    rows = np.nonzero(pos >= W - 1)[0]
    for c0 in range(0, len(rows), CHUNK):
        rr = rows[c0:c0 + CHUNK]
        X = win[rr - W + 1]
        m = X.mean(axis=1, keepdims=True)
        b = ((X - m) * jc).sum(axis=1, keepdims=True) / vj
        a = m - b * j.mean()
        R = X - (a + b * j)
        F = np.fft.rfft(R, axis=1)
        amp = np.abs(F[:, 1:])
        top = np.argsort(-amp, axis=1)[:, :K] + 1
        Fk = np.take_along_axis(F, top, axis=1)
        scale = np.where(top == W // 2, 1.0 / W, 2.0 / W)
        A = np.abs(Fk) * scale
        ph = np.angle(Fk)
        curve = a + b * fut[None, :]
        d1 = np.repeat(b, len(fut), axis=1)
        d2 = np.zeros_like(curve)
        for q in range(K):
            w_ = 2 * np.pi * top[:, q:q + 1] / W
            arg = w_ * fut[None, :] + ph[:, q:q + 1]
            curve = curve + A[:, q:q + 1] * np.cos(arg)
            d1 = d1 - A[:, q:q + 1] * w_ * np.sin(arg)
            d2 = d2 - A[:, q:q + 1] * w_ * w_ * np.cos(arg)
        s2 = np.sign(d2)
        iflip = (s2[:, 1:] != s2[:, :1]) & (s2[:, 1:] != 0)
        ihas = iflip.any(axis=1)
        ifirst = np.where(ihas, iflip.argmax(axis=1) + 1, AHEAD)
        move = curve - curve[:, :1]
        dv = np.diff(curve, axis=1)
        sgn = np.sign(dv)
        flip = (sgn[:, 1:] != sgn[:, :1]) & (sgn[:, 1:] != 0)
        has = flip.any(axis=1)
        first = np.where(has, flip.argmax(axis=1) + 1, AHEAD)
        etype = np.where(sgn[:, 0] > 0, 1.0, -1.0)
        emove = move[np.arange(len(rr)), first]
        out[rr] = np.column_stack([move[:, 10], move[:, 30], first.astype(float), etype, emove, d1[:, 0], d2[:, 0], ifirst.astype(float), np.sign(d2[:, 0])])
    return out


def simulate(rows, ent, ext, day, O, IO, pred, rng=None, count=None):
    cash, vault, trades = CAPITAL, 0.0, []
    busy = -1
    cand = list(rows)
    legs = None
    if rng is not None:
        k = min(count[0], len(cand))
        pick = sorted(rng.choice(len(cand), size=k, replace=False))
        cand = [cand[x] for x in pick]
        legs = rng.random(len(cand)) < count[1]
    cost = lambda leg, e: LI.FEE + (LI.TICK if leg == "L" else H8.tick(day[e], IO[e])) / (O[e] if leg == "L" else IO[e])
    for z, i in enumerate(cand):
        if ent[i] < 0 or ent[i] <= busy:
            continue
        if rng is None:
            p = pred[i]
            if np.isnan(p):
                continue
            if p > 2 * cost("L", ent[i]):
                leg = "L"
            elif -p > 2 * cost("I", ent[i]):
                leg = "I"
            else:
                continue
        else:
            leg = "L" if legs[z] else "I"
        P = O if leg == "L" else IO
        cin, cout = cost(leg, ent[i]), cost(leg, ext[i])
        q = cash * (1 - cin) / P[ent[i]]
        proceeds = q * P[ext[i]] * (1 - cout)
        profit = proceeds - cash
        tv = VAULT_SHARE * profit if profit > 0 else 0.0
        vault += tv
        trades.append((leg, proceeds / cash - 1))
        cash = proceeds - tv
        busy = ext[i]
    nets = [x[1] for x in trades]
    return dict(total_pct=100 * ((cash + vault) / CAPITAL - 1), vault_pct=100 * vault / CAPITAL, trades=len(trades), mean_net_bp=1e4 * float(np.mean(nets)) if nets else None, long_share=float(np.mean([x[0] == "L" for x in trades])) if trades else None)


def state_targets(f, rule):
    d1, d2 = f[:, 5], f[:, 6]
    tg = np.array([None] * len(f), dtype=object)
    v = ~np.isnan(d1)
    if rule == "phase":
        tg[v & (d1 > 0) & (d2 > 0)] = "L"
        tg[v & (d1 < 0) & (d2 < 0)] = "I"
    else:
        tg[v & (d1 > 0)] = "L"
        tg[v & (d1 < 0)] = "I"
    return tg


def run_states(target, ok, day, nxt, lastfill, O, IO, flip=None):
    cash, vault, leg, q, basis, cur = CAPITAL, 0.0, None, 0.0, 0.0, None
    nets, seg = [], -1
    n = len(target)
    cost = lambda lg, e: LI.FEE + (LI.TICK if lg == "L" else H8.tick(day[e], IO[e])) / (O[e] if lg == "L" else IO[e])

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
        same = day[t + 1] == day[t]
        want = target[t] if (ok[t] and same) else None
        if want == cur:
            continue
        if leg is not None:
            e = nxt[t + 1] if same and nxt[t + 1] >= 0 else (lastfill[t] if lastfill[t] >= 0 else t)
            close(e)
        cur = None
        if want is not None and same and nxt[t + 1] >= 0:
            e = nxt[t + 1]
            seg += 1
            real = (("I" if want == "L" else "L") if flip[seg % len(flip)] else want) if flip is not None else want
            P = O if real == "L" else IO
            q = cash * (1 - cost(real, e)) / P[e]
            basis, cash, leg, cur = cash, 0.0, real, want
    if leg is not None:
        close(lastfill[n - 1] if lastfill[n - 1] >= 0 else n - 1)
    r = [x[1] for x in nets]
    return dict(total_pct=100 * ((cash + vault) / CAPITAL - 1), vault_pct=100 * vault / CAPITAL, trades=len(nets), mean_net_bp=1e4 * float(np.mean(r)) if r else None, win=float(np.mean([x > 0 for x in r])) if r else None, long_share=float(np.mean([x[0] == "L" for x in nets])) if nets else None)


def evaluate_states(name, target, okm, day, fl, nxt, lastfill, O, IO):
    folds, chain, rchain, ntr, beat = [], 1.0, 1.0, 0, 0
    for a, b in fl:
        idx = np.nonzero((day >= a) & (day < b))[0]
        if len(idx) == 0:
            continue
        lo, hi = idx[0], idx[-1] + 1
        sl = slice(lo, hi)
        nx = np.where(nxt[sl] >= 0, nxt[sl] - lo, -1)
        lf = np.where(lastfill[sl] >= 0, lastfill[sl] - lo, -1)
        r = run_states(target[sl], okm[sl], day[sl], nx, lf, O[sl], IO[sl])
        rnd = []
        for sd in range(SEEDS):
            fp = np.random.default_rng(31 + sd).random(max(1, r["trades"] + 5)) < 0.5
            rnd.append(run_states(target[sl], okm[sl], day[sl], nx, lf, O[sl], IO[sl], flip=fp)["total_pct"])
        rm = float(np.mean(rnd))
        chain *= 1 + r["total_pct"] / 100
        rchain *= 1 + rm / 100
        ntr += r["trades"]
        beat += r["total_pct"] > rm
        folds.append(dict(test=[a, b], model=r, random_direction_mean_total_pct=rm))
    return dict(name=name, chained_total_pct=100 * (chain - 1), random_direction_chained_total_pct=100 * (rchain - 1), trades=ntr, trades_per_day=ntr / max(1, len(set(day[day >= S1.TEST_FROM]))), folds_beating_random=int(beat), n_folds=len(folds), folds=folds)


def evaluate(name, pred, lab, days, fl, ent, ext, day, O, IO, H, y):
    te = lab & (days >= S1.TEST_FROM) & ~np.isnan(pred)
    corr = float(np.corrcoef(pred[te], y[te])[0, 1]) if te.sum() > 10 else None
    hit = float(np.mean(np.sign(pred[te]) == np.sign(y[te]))) if te.sum() else None
    folds, chain, rchain, ntr, beat = [], 1.0, 1.0, 0, 0
    for a, b in fl:
        rows = np.nonzero(lab & (days >= a) & (days < b))[0]
        if len(rows) == 0:
            continue
        r = simulate(rows, ent, ext, day, O, IO, pred)
        rnd = []
        if r["trades"]:
            for sd in range(SEEDS):
                rnd.append(simulate(rows, ent, ext, day, O, IO, pred, np.random.default_rng(97 * H + sd), (r["trades"], r["long_share"]))["total_pct"])
        rm = float(np.mean(rnd)) if rnd else 0.0
        chain *= 1 + r["total_pct"] / 100
        rchain *= 1 + rm / 100
        ntr += r["trades"]
        beat += r["total_pct"] > rm
        folds.append(dict(test=[a, b], model=r, random_mean_total_pct=rm))
    return dict(name=name, H=H, corr=corr, sign_hit=hit, up_base=float(np.mean(y[te] > 0)) if te.sum() else None, chained_total_pct=100 * (chain - 1), random_chained_total_pct=100 * (rchain - 1), trades=ntr, folds_beating_random=int(beat), n_folds=len(folds), folds=folds)


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
    day = d["day"].to_numpy()
    pos = d.groupby("day").cumcount().to_numpy()
    y_log = np.log(d["close"].to_numpy(float))
    O, IO = d["open"].to_numpy(float), d["io"].to_numpy(float)
    fl = S1.folds(None)
    cfgs = configs()
    lev = {lam: kalman(y_log, day, lam) for lam in LAMBDAS}
    feats, names = [], []
    for c in cfgs:
        f = fft_features(lev[c[2]][0], pos, c[0], c[1])
        feats.append(f)
        names += [f"{cname(c)}_{s}" for s in ("move10", "move30", "t_ext", "ext_type", "ext_move", "d1", "d2", "t_infl", "d2_sign")]
        print("features", cname(c), "valid", int((~np.isnan(f[:, 0])).sum()), flush=True)
    for lam in LAMBDAS:
        feats.append(np.column_stack([lev[lam][1], y_log - lev[lam][0]]))
        names += [f"kal_slope_L{lam}", f"kal_resid_L{lam}"]
    FX = np.hstack(feats)
    out = dict(rows=int(len(d)), configs=[cname(c) for c in cfgs], results=[], state_results=[])
    nxt, _ = S1.fill_index(d)
    fillable = nxt == np.arange(len(d))
    lastfill = np.full(len(d), -1)
    lf = -1
    for t in range(len(d)):
        if t > 0 and day[t] != day[t - 1]:
            lf = -1
        if fillable[t]:
            lf = t
        lastfill[t] = lf
    mod, ce = d["mod"].to_numpy(), d["ce"].to_numpy()
    okm = (mod <= ce - 3) & ~np.isnan(base).any(axis=1) & (day >= S1.TEST_FROM)
    for ci, c in enumerate(cfgs):
        for rule in ("phase", "d1"):
            tg = state_targets(feats[ci], rule)
            r = evaluate_states(f"{rule}_{cname(c)}", tg, okm, day, fl, nxt, lastfill, O, IO)
            out["state_results"].append(r)
            print(f"STATE {r['name']:24s} trades {r['trades']:7d} ({r['trades_per_day']:.1f}/day) chained {r['chained_total_pct']:9.2f}% random-direction {r['random_direction_chained_total_pct']:9.2f}% beat {r['folds_beating_random']}/{r['n_folds']}", flush=True)
    for H in HORIZONS:
        ent, ext, yl, yi, yv = S1.labels(d, H)
        lab = ~np.isnan(yl) & ~np.isnan(base).any(axis=1)
        col = 0 if H == 10 else 1
        for ci, c in enumerate(cfgs):
            pr = feats[ci][:, col].copy()
            r = evaluate(f"fixed_{cname(c)}", pr, lab, day, fl, ent, ext, day, O, IO, H, yl)
            out["results"].append(r)
            print(f"H{H} {r['name']:22s} corr {r['corr']:.4f} hit {r['sign_hit']:.4f} base {r['up_base']:.4f} trades {r['trades']:6d} chained {r['chained_total_pct']:8.2f}% random {r['random_chained_total_pct']:8.2f}% beat {r['folds_beating_random']}/{r['n_folds']}", flush=True)
        known = sorted([(ext[i], i) for i in np.nonzero(lab)[0]])
        dq = [deque(maxlen=ONLINE_N) for _ in cfgs]
        sums = np.zeros(len(cfgs))
        kp = 0
        pr = np.full(len(d), np.nan)
        chosen = np.full(len(d), -1)
        for t in range(len(d)):
            while kp < len(known) and known[kp][0] <= t:
                i = known[kp][1]
                for ci in range(len(cfgs)):
                    v = feats[ci][i, col]
                    if np.isnan(v):
                        continue
                    h = 1.0 if np.sign(v) == np.sign(yl[i]) else 0.0
                    if len(dq[ci]) == ONLINE_N:
                        sums[ci] -= dq[ci][0]
                    dq[ci].append(h)
                    sums[ci] += h
                kp += 1
            if not lab[t]:
                continue
            score = np.array([sums[ci] / len(dq[ci]) if len(dq[ci]) >= 100 else -1 for ci in range(len(cfgs))])
            best = int(score.argmax())
            if score[best] < 0:
                continue
            chosen[t] = best
            pr[t] = feats[best][t, col]
        r = evaluate("online_select", pr, lab, day, fl, ent, ext, day, O, IO, H, yl)
        te = lab & (day >= S1.TEST_FROM) & (chosen >= 0)
        r["chosen_share"] = {cname(cfgs[ci]): float(np.mean(chosen[te] == ci)) for ci in range(len(cfgs))}
        out["results"].append(r)
        print(f"H{H} online_select           corr {r['corr']:.4f} hit {r['sign_hit']:.4f} trades {r['trades']:6d} chained {r['chained_total_pct']:8.2f}% random {r['random_chained_total_pct']:8.2f}% beat {r['folds_beating_random']}/{r['n_folds']} chosen {sorted(r['chosen_share'].items(), key=lambda z: -z[1])[:3]}", flush=True)
        for tag, X in (("lgbm_base_only", base), ("lgbm_base_plus_kalman_fft", np.hstack([base, FX]))):
            pr = np.full(len(d), np.nan)
            imp = np.zeros(X.shape[1])
            for a, b in fl:
                tr = lab & (day < a)
                tt = lab & (day >= a) & (day < b)
                if tt.sum() == 0:
                    continue
                m = lgb.train(LI.PARAMS, lgb.Dataset(X[tr], label=yl[tr], free_raw_data=False), num_boost_round=LI.ROUNDS)
                pr[tt] = m.predict(X[tt])
                imp += m.feature_importance("gain")
            r = evaluate(tag, pr, lab, day, fl, ent, ext, day, O, IO, H, yl)
            nm = S1.FEATS + (names if X.shape[1] > len(S1.FEATS) else [])
            r["top_features"] = sorted(zip(nm, imp.tolist()), key=lambda z: -z[1])[:12]
            out["results"].append(r)
            print(f"H{H} {tag:24s} corr {r['corr']:.4f} hit {r['sign_hit']:.4f} trades {r['trades']:6d} chained {r['chained_total_pct']:8.2f}% random {r['random_chained_total_pct']:8.2f}% beat {r['folds_beating_random']}/{r['n_folds']} top {[z[0] for z in r['top_features'][:5]]}", flush=True)
    rec = dict(version="kalman-fft-v1", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], results=out)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
