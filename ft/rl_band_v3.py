import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
import math
import sys
import warnings

sys.dont_write_bytecode = True
warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import lightgbm as lgb
import learn_inverse as LI
import rl_band_v1 as R1
import rl_band_v2 as R2

DS = np.round(np.arange(0.0, 0.1001, 0.005), 4)
TPS = np.round(np.arange(0.0025, 0.1501, 0.005), 4)
SLS = np.array([0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.0])
CARRY = np.array([0.0, 1.0, 2.0])
GRID = np.array([(d, tp, sl, cr) for d in DS for tp in TPS for sl in SLS for cr in CARRY])
SAMPLES = 150
WAIT_SESSIONS = 2
MAX_SESSIONS = 10
SEEDS = 20
CAPITAL = 100_000_000.0


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def outcome_set(b, q, acts):
    O, H, L, C, dk, last = b["O"], b["H"], b["L"], b["C"], b["dk"], b["last"]
    first_min = b["first_min"]
    eod, gapd = b["eod"], b["gapd"]
    i = b["dec"][q]
    e0 = i + 1
    p0 = C[i]
    wait_end = last[min(dk[e0] + WAIT_SESSIONS - 1, dk[-1])]
    lw = L[e0:wait_end + 1]
    runmin = np.minimum.accumulate(lw)
    out = np.zeros(len(acts))
    exits = np.full(len(acts), -1, dtype=np.int64)
    cache = {}
    for a, (d, tp, sl, cr) in enumerate(acts):
        if d == 0:
            k = e0
            pin = O[e0]
        else:
            lvl = p0 * (1 - d)
            j = int(np.searchsorted(-runmin, -lvl, side="left"))
            if j >= len(runmin):
                out[a] = 0.0
                exits[a] = wait_end
                continue
            k = e0 + j
            pin = min(lvl, O[k]) if (first_min[k] and O[k] < lvl) else lvl
            pin += LI.TICK
        cin = LI.FEE + LI.TICK / pin
        lim = min(dk[k] + MAX_SESSIONS - 1, dk[-1])
        if cr == 0.0:
            stop = eod[dk[k]]
        elif cr == 1.0:
            dd = dk[k]
            while dd < lim and gapd[dd] <= 1:
                dd += 1
            stop = eod[dd] if gapd[dd] > 1 else last[lim]
        else:
            stop = last[lim]
        if stop <= k:
            out[a] = 0.0
            exits[a] = k
            continue
        ck = (k, int(stop))
        if ck not in cache:
            hh = H[k:stop + 1].copy()
            ll = L[k:stop + 1].copy()
            if d != 0:
                hh[0] = -np.inf
            cache[ck] = (np.maximum.accumulate(hh), np.minimum.accumulate(ll), stop)
        rmax, rmin, stop = cache[ck]
        up = pin * (1 + tp + 2 * cin)
        ku = int(np.searchsorted(rmax, up, side="left"))
        kd = int(np.searchsorted(-rmin, -pin * (1 - sl), side="left")) if sl > 0 else len(rmin)
        if kd < len(rmin) and kd <= ku:
            kk = k + kd
            lvl = pin * (1 - sl)
            px = (min(lvl, O[kk]) if (first_min[kk] and O[kk] < lvl) else lvl) - LI.TICK
        elif ku < len(rmax):
            kk = k + ku
            px = max(up, O[kk]) if (first_min[kk] and O[kk] > up) else up
        else:
            kk = stop
            px = C[stop] - LI.TICK
        out[a] = math.log(px * (1 - LI.FEE) / (pin * (1 + cin)))
        exits[a] = kk
    return out, exits


def rows(Fs, acts, v20):
    A = np.asarray(acts)
    scale = np.where(v20 > 0, v20 * math.sqrt(380), np.nan)
    return np.hstack([np.repeat(Fs[None, :], len(A), axis=0), A, A[:, :3] / scale])


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, pth in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), pth)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    d = R1.load()
    b = R1.build(d)
    b["first_min"] = np.r_[True, b["dk"][1:] != b["dk"][:-1]]
    hm = d["hm"].to_numpy()
    ce_hm = d["ce"].to_numpy()
    mod = d["mod"].to_numpy()
    nd = int(b["dk"][-1]) + 1
    eod = np.array(b["last"], dtype=np.int64)
    okk = mod <= ce_hm - 1
    for k_ in range(len(mod)):
        if okk[k_]:
            eod[b["dk"][k_]] = k_
    b["eod"] = eod
    from datetime import date as _date
    ud = b["udays"]
    b["gapd"] = np.array([(_date.fromisoformat(ud[k_ + 1]) - _date.fromisoformat(ud[k_])).days if k_ + 1 < nd else 1 for k_ in range(nd)])
    F = b["F"][b["dec"]]
    v20 = F[:, R1.FEATS.index("v_20d")]
    dday = b["day"][b["dec"]]
    rng = np.random.default_rng(12345)
    samp_idx = np.zeros((len(b["dec"]), SAMPLES), dtype=np.int64)
    samp_r = np.zeros((len(b["dec"]), SAMPLES))
    samp_x = np.zeros((len(b["dec"]), SAMPLES), dtype=np.int64)
    zero_d = np.nonzero(GRID[:, 0] == 0)[0]
    for q in range(len(b["dec"])):
        idx = np.r_[rng.choice(zero_d, size=30, replace=False), rng.choice(len(GRID), size=SAMPLES - 30, replace=False)]
        r, x = outcome_set(b, q, GRID[idx])
        samp_idx[q], samp_r[q], samp_x[q] = idx, r, x
        if q % 2000 == 0:
            print("outcomes", q, "/", len(b["dec"]), flush=True)
    res = dict(quarters=[], grid_size=int(len(GRID)))
    chosen = {}
    sim_cash, rl_trades = CAPITAL, []
    fixed_cash, fixed_trades = CAPITAL, []
    rnd_cash = np.full(SEEDS, CAPITAL)
    rngs = [np.random.default_rng(700 + s) for s in range(SEEDS)]
    busy_rl = busy_fx = -1
    busy_rnd = np.full(SEEDS, -1)
    first_test = None
    for a_, b_ in R1.quarters():
        start_min = int(np.searchsorted(b["day"], a_))
        tr = np.nonzero((dday < a_))[0]
        tr = np.array([q for q in tr if samp_x[q].max() < start_min], dtype=int)
        te = np.nonzero((dday >= a_) & (dday < b_))[0]
        if len(te) == 0 or len(tr) < 1000:
            continue
        if first_test is None:
            first_test = te[0]
        Xtr = np.vstack([rows(F[q], GRID[samp_idx[q]], v20[q]) for q in tr])
        ytr = samp_r[tr].ravel()
        m = lgb.train(LI.PARAMS, lgb.Dataset(Xtr, label=ytr), num_boost_round=LI.ROUNDS)
        sums = np.zeros(len(GRID))
        cnts = np.zeros(len(GRID))
        np.add.at(sums, samp_idx[tr].ravel(), samp_r[tr].ravel())
        np.add.at(cnts, samp_idx[tr].ravel(), 1)
        mean = np.where(cnts >= 30, sums / np.maximum(cnts, 1), -np.inf)
        fixed = int(np.argmax(mean))
        res["quarters"].append(dict(test=[a_, b_], train_points=int(len(tr)), fixed_best=GRID[fixed].tolist(), fixed_best_train_mean_bp=1e4 * float(mean[fixed])))
        print("fold", a_, "train points", len(tr), "rows", len(ytr), "fixed best", GRID[fixed].tolist(), round(1e4 * float(mean[fixed]), 1), "bp", flush=True)
        for q in te:
            i = b["dec"][q]
            if i > busy_rl:
                p = m.predict(rows(F[q], GRID, v20[q]))
                a = int(np.argmax(p))
                if p[a] > 0:
                    r, x = outcome_set(b, q, GRID[a:a + 1])
                    sim_cash *= math.exp(r[0])
                    if not (GRID[a][0] > 0 and r[0] == 0.0 and x[0] == b["last"][min(b["dk"][i + 1] + WAIT_SESSIONS - 1, b["dk"][-1])]):
                        rl_trades.append(r[0])
                    busy_rl = x[0]
                    key = (round(float(GRID[a][0]), 4), round(float(GRID[a][1]), 4), round(float(GRID[a][2]), 4), int(GRID[a][3]))
                    chosen[key] = chosen.get(key, 0) + 1
            if i > busy_fx:
                r, x = outcome_set(b, q, GRID[fixed:fixed + 1])
                fixed_cash *= math.exp(r[0])
                fixed_trades.append(r[0])
                busy_fx = x[0]
            for s in range(SEEDS):
                if i > busy_rnd[s]:
                    a = int(rngs[s].integers(len(GRID)))
                    r, x = outcome_set(b, q, GRID[a:a + 1])
                    rnd_cash[s] *= math.exp(r[0])
                    busy_rnd[s] = x[0]
        print("   cumulative after", b_, "RL", round(100 * (sim_cash / CAPITAL - 1), 1), "% fixed", round(100 * (fixed_cash / CAPITAL - 1), 1), "% random mean", round(100 * (float(np.mean(rnd_cash)) / CAPITAL - 1), 1), "%", flush=True)
    i0 = b["dec"][first_test] + 1
    hold = 100 * (b["C"][-1] / b["O"][i0] - 1)
    top = sorted(chosen.items(), key=lambda z: -z[1])[:15]
    ds = np.array([k[0] for k, v in chosen.items() for _ in range(v)])
    tps = np.array([k[1] for k, v in chosen.items() for _ in range(v)])
    sls = np.array([k[2] for k, v in chosen.items() for _ in range(v)])
    crs = np.array([k[3] for k, v in chosen.items() for _ in range(v)])
    res["test"] = dict(rl_total_pct=100 * (sim_cash / CAPITAL - 1), rl_trades=len(rl_trades), rl_mean_bp=1e4 * float(np.mean(rl_trades)) if rl_trades else None, rl_win=float(np.mean(np.array(rl_trades) > 0)) if rl_trades else None, fixed_total_pct=100 * (fixed_cash / CAPITAL - 1), fixed_trades=len(fixed_trades), random_total_pct=[100 * (c / CAPITAL - 1) for c in rnd_cash], random_mean_total_pct=100 * (float(np.mean(rnd_cash)) / CAPITAL - 1), hold_pct=hold, chosen_top=[(list(k), v) for k, v in top], chosen_buy_dip_quantiles=np.percentile(ds, [10, 50, 90]).tolist() if len(ds) else None, chosen_tp_quantiles=np.percentile(tps, [10, 50, 90]).tolist() if len(tps) else None, chosen_sl_share_none=float(np.mean(sls == 0)) if len(sls) else None, chosen_carry_share={"intraday_only": float(np.mean(crs == 0)) if len(crs) else None, "weekday_nights": float(np.mean(crs == 1)) if len(crs) else None, "all_nights_10_sessions": float(np.mean(crs == 2)) if len(crs) else None})
    print("TEST", json.dumps({k: v for k, v in res["test"].items() if k not in ("random_total_pct", "chosen_top")}, default=float), flush=True)
    print("TOP CHOICES (buy dip, take profit above cost, stop (0 = none), carry 0 intraday / 1 weekday nights / 2 all):", top, flush=True)
    rec = dict(version="rl-band-v3", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], results=res)
    with open(out_path, "x", encoding="utf-8") as fo:
        json.dump(rec, fo, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
