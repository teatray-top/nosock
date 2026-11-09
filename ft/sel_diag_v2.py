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
import ms_role_v2 as R
import ms_alloc_v7 as A7
import sel_diag_v1 as D1

H = D1.H
VOL_FLOOR = 0.005
SHRINK_K = 250.0
HOLD20 = int(np.nonzero((R.GRID[:, 1] == 0) & (R.GRID[:, 3] == 20))[0][0])
XS_NAMES = ("mom126", "mom252_21", "rev20", "lowvol60")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def forward_floor(ins):
    O, C = ins["O"], ins["C"]
    n = len(C)
    f = np.full(n, np.nan)
    fr = np.full(n, np.nan)
    for t in range(n - H - 1):
        f[t] = math.log(C[t + H] / O[t + 1])
        v = ins["v20"][t]
        if v > 0:
            fr[t] = f[t] / (max(v, VOL_FLOOR) * math.sqrt(H))
    return f, fr


def by_date(inst, codes):
    idx = {}
    for c in codes:
        for i, d in enumerate(inst[c]["days"]):
            idx.setdefault(d, []).append((c, i))
    return idx


def xs_rank_features(inst, codes, simple, dates):
    out = {c: np.full((len(inst[c]["C"]), len(XS_NAMES) + 1), np.nan) for c in codes}
    for d, lst in dates.items():
        for k, name in enumerate(list(XS_NAMES) + ["v20"]):
            vals = [(c, i, (inst[c]["v20"][i] if name == "v20" else simple[c][name][i])) for c, i in lst]
            vals = [(c, i, v) for c, i, v in vals if not np.isnan(v)]
            if len(vals) < D1.MIN_N:
                continue
            r = D1.rankdata(np.array([v for _, _, v in vals]))
            r = r / (len(vals) - 1)
            for (c, i, _), rr in zip(vals, r):
                out[c][i, k] = rr
    return out


def composite(inst, codes, simple, dates):
    out = {c: np.full(len(inst[c]["C"]), np.nan) for c in codes}
    for d, lst in dates.items():
        zs = []
        for name in ("mom126", "mom252_21"):
            vals = np.array([simple[c][name][i] for c, i in lst])
            ok = ~np.isnan(vals)
            if ok.sum() < D1.MIN_N:
                zs = None
                break
            z = np.full(len(vals), np.nan)
            z[ok] = (vals[ok] - vals[ok].mean()) / (vals[ok].std() + 1e-12)
            zs.append(z)
        if zs is None:
            continue
        m = np.nanmean(np.vstack(zs), axis=0)
        for (c, i), v in zip(lst, m):
            out[c][i] = v
    return out


def xs_targets(inst, codes, fwd, dates):
    y = {c: np.full(len(inst[c]["C"]), np.nan) for c in codes}
    for d, lst in dates.items():
        vals = [(c, i, fwd[c][1][i]) for c, i in lst if not np.isnan(fwd[c][1][i])]
        if len(vals) < D1.MIN_N:
            continue
        m = np.mean([v for _, _, v in vals])
        for c, i, v in vals:
            y[c][i] = v - m
    return y


def xs_model(inst, codes, xsf, y):
    pred = {c: np.full(len(inst[c]["C"]), np.nan) for c in codes}
    shr = {c: np.full(len(inst[c]["C"]), np.nan) for c in codes}
    for yr in R.TEST_YEARS:
        cut = f"{yr}-01-01"
        X, Y, cnt = [], [], {}
        for c in codes:
            ins = inst[c]
            n = len(ins["C"])
            for t in range(260, n - H - 1):
                if ins["days"][t + H] >= cut:
                    break
                row = np.r_[ins["F"][t], xsf[c][t]]
                if np.isnan(row).any() or np.isnan(y[c][t]):
                    continue
                X.append(row)
                Y.append(y[c][t])
                cnt[c] = cnt.get(c, 0) + 1
        m = lgb.train(LI.PARAMS, lgb.Dataset(np.array(X, dtype=np.float32), label=np.array(Y)), num_boost_round=LI.ROUNDS)
        for c in codes:
            ins = inst[c]
            ts = [t for t, d in enumerate(ins["days"]) if d[:4] == str(yr) and t >= 260]
            if not ts:
                continue
            Xt = np.array([np.r_[ins["F"][t], xsf[c][t]] for t in ts], dtype=np.float32)
            p = m.predict(Xt)
            bad = np.isnan(Xt).any(axis=1)
            p[bad] = np.nan
            w = cnt.get(c, 0) / (cnt.get(c, 0) + SHRINK_K)
            for t, v in zip(ts, p):
                pred[c][t] = v
                shr[c][t] = v * w
        print("   xs model year", yr, "rows", len(Y), flush=True)
    return pred, shr


def q_scores(inst, codes, samp):
    qmax = {c: np.full(len(inst[c]["C"]), np.nan) for c in codes}
    qh = {c: np.full(len(inst[c]["C"]), np.nan) for c in codes}
    for yr in R.TEST_YEARS:
        cut = f"{yr}-01-01"
        X, Y = [], []
        for c in codes:
            ins = inst[c]
            for t, idx, Rr, ex in samp[c]:
                if ins["days"][t] < cut and ex.max() < len(ins["days"]) and ins["days"][int(ex.max())] < cut and not np.isnan(Rr).any():
                    X.append(R.rows(ins["F"][t], R.GRID[idx], ins["v20"][t]).astype(np.float32))
                    Y.append(R.reward(Rr, R.KAPPA))
        m = lgb.train(LI.PARAMS, lgb.Dataset(np.vstack(X), label=np.concatenate(Y)), num_boost_round=LI.ROUNDS)
        del X
        for c in codes:
            ins = inst[c]
            n = len(ins["C"])
            for t, d in enumerate(ins["days"]):
                if d[:4] != str(yr) or t < 260 or t + 1 >= n or np.isnan(ins["F"][t]).any():
                    continue
                p = m.predict(R.rows(ins["F"][t], R.GRID, ins["v20"][t]))
                qmax[c][t] = float(p.max())
                qh[c][t] = float(p[HOLD20])
        print("   q model year", yr, flush=True)
    return qmax, qh


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, pth in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), pth)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    inst, groups = D1.build_all()
    fwd = {c: forward_floor(ins) for c, ins in inst.items()}
    simple = {c: D1.simple_scores(ins) for c, ins in inst.items()}
    samp = A7.samples(inst)
    out = {}
    rng = np.random.default_rng(199)
    for g, codes in groups.items():
        dates = by_date(inst, codes)
        xsf = xs_rank_features(inst, codes, simple, dates)
        y = xs_targets(inst, codes, fwd, dates)
        xs_pred, xs_shr = xs_model(inst, codes, xsf, y)
        qmax, qh = q_scores(inst, codes, samp)
        sets = dict(mom126={c: simple[c]["mom126"] for c in codes}, composite_mom=composite(inst, codes, simple, dates), q_max_over_actions=qmax, q_hold20_only=qh, xs_model=xs_pred, xs_model_shrunk=xs_shr, random={c: rng.standard_normal(len(inst[c]["C"])) for c in codes})
        out[g] = D1.evaluate(inst, codes, sets, fwd)
        print(g, "instruments", len(codes), flush=True)
        for k, v in out[g].items():
            print(f"   {k:20s} IC {v['ic_mean']:+.4f} (t {v['ic_t_nonoverlap']:+.2f}, positive days {100 * v['ic_positive_share']:.0f}%) floored risk-adj IC {v['ic_riskadj_mean']:+.4f} | top-minus-bottom quintile {v['top_minus_bottom_quintile_20d_pct']:+.2f}% per 20d, top2-minus-median {v['top2_minus_median_20d_pct']:+.2f}%", flush=True)
    rec = dict(version="sel-diag-v2", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], horizon=H, vol_floor=VOL_FLOOR, shrink_k=SHRINK_K, results=out)
    with open(out_path, "x", encoding="utf-8") as fo:
        json.dump(rec, fo, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
