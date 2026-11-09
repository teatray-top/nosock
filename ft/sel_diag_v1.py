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
import msma_v1 as M1
import ms_role_v2 as R
import ms_alloc_v7 as A7
import ms_alloc_v9 as A9

H = 20
MIN_N = 5
US_FEE, US_FX = 0.001, 0.0005


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def rankdata(x):
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    xs = x[o]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    d = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan


def build_all():
    man_us, us = A7.load_us()
    fxrec = us["KRW_X_adj"]
    fx = {d: c for d, c in zip(fxrec["days"], fxrec["C"]) if c > 500}
    fx_days = sorted(fx)
    us_days = sorted(set(d for k, v in us.items() if k.endswith("_adj") and not k.startswith("KRW") for d in v["days"]))
    fx_ff, last, j = {}, None, 0
    for d in us_days:
        while j < len(fx_days) and fx_days[j] <= d:
            last = fx[fx_days[j]]
            j += 1
        if last is not None:
            fx_ff[d] = last
    man_kus, kus = A9.load_krus()
    classes = sorted(set(M1.CLASSES) | set(man_us["classes"].values()) | set(man_kus["classes"].values()))
    recs = R.load()
    inst = {}
    for c in M1.CLASS:
        inst[c] = A7.instrument_x(recs[c], c, M1.CLASS[c], classes, "KR", A7.kr_cost)
    for c, cls in man_kus["classes"].items():
        inst[c] = A7.instrument_x(kus[f"{c}_adj"], c, cls, classes, "KR", A7.kr_cost)
    for c, cls in man_us["classes"].items():
        inst[c] = A7.instrument_x(us[f"{c}_adj"], c, cls, classes, "US", A7.us_cost_maker(US_FEE, US_FX), fx_ff)
    groups = dict(U15=list(M1.CLASS), U28=list(M1.CLASS) + list(man_kus["classes"]), U36=list(M1.CLASS) + list(man_us["classes"]))
    return inst, groups


def simple_scores(ins):
    C = ins["C"]
    n = len(C)
    lc = np.log(C)
    r = np.r_[np.nan, np.diff(lc)]
    out = {k: np.full(n, np.nan) for k in ("mom126", "mom252_21", "riskadj126", "rev20", "lowvol60")}
    for t in range(252, n):
        out["mom126"][t] = lc[t] - lc[t - 126]
        out["mom252_21"][t] = lc[t - 21] - lc[t - 252]
        sd = np.std(r[t - 125:t + 1])
        out["riskadj126"][t] = (lc[t] - lc[t - 126]) / (sd * math.sqrt(126)) if sd > 0 else np.nan
        out["rev20"][t] = -(lc[t] - lc[t - 20])
        sd60 = np.std(r[t - 59:t + 1])
        out["lowvol60"][t] = -sd60
    return out


def forward(ins):
    O, C = ins["O"], ins["C"]
    n = len(C)
    f = np.full(n, np.nan)
    fr = np.full(n, np.nan)
    for t in range(n - H - 1):
        f[t] = math.log(C[t + H] / O[t + 1])
        v = ins["v20"][t]
        fr[t] = f[t] / (v * math.sqrt(H)) if v > 0 else np.nan
    return f, fr


def model_scores(inst, codes, samp):
    sub = {c: inst[c] for c in codes}
    scores = {c: np.full(len(inst[c]["C"]), np.nan) for c in codes}
    for y in R.TEST_YEARS:
        cut = f"{y}-01-01"
        X, Y = [], []
        for c in codes:
            ins = sub[c]
            for t, idx, Rr, ex in samp[c]:
                if ins["days"][t] < cut and ex.max() < len(ins["days"]) and ins["days"][int(ex.max())] < cut and not np.isnan(Rr).any():
                    X.append(R.rows(ins["F"][t], R.GRID[idx], ins["v20"][t]).astype(np.float32))
                    Y.append(R.reward(Rr, R.KAPPA))
        m = lgb.train(LI.PARAMS, lgb.Dataset(np.vstack(X), label=np.concatenate(Y)), num_boost_round=LI.ROUNDS)
        del X
        for c in codes:
            ins = sub[c]
            n = len(ins["C"])
            ts = [t for t, d in enumerate(ins["days"]) if d[:4] == str(y) and t >= 260 and t + 1 < n and not np.isnan(ins["F"][t]).any()]
            for t in ts:
                p = m.predict(R.rows(ins["F"][t], R.GRID, ins["v20"][t]))
                scores[c][t] = float(p.max())
        print("   model year", y, "codes", len(codes), flush=True)
    return scores


def evaluate(inst, codes, score_sets, fwd):
    days = sorted(set(d for c in codes for d in inst[c]["days"] if d >= "2016-01-01"))
    pos = {c: {d: i for i, d in enumerate(inst[c]["days"])} for c in codes}
    res = {}
    for name, sc in score_sets.items():
        ics, icr, spread, top2, nday, step_ic = [], [], [], [], [], []
        for k, D in enumerate(days):
            xs, ys, yr = [], [], []
            for c in codes:
                i = pos[c].get(D)
                if i is None:
                    continue
                s, f, fr = sc[c][i], fwd[c][0][i], fwd[c][1][i]
                if np.isnan(s) or np.isnan(f) or np.isnan(fr):
                    continue
                xs.append(s)
                ys.append(f)
                yr.append(fr)
            if len(xs) < MIN_N:
                continue
            xs, ys, yr = np.array(xs), np.array(ys), np.array(yr)
            ic = spearman(xs, ys)
            ics.append(ic)
            icr.append(spearman(xs, yr))
            o = np.argsort(-xs)
            q = max(1, len(xs) // 5)
            spread.append(ys[o[:q]].mean() - ys[o[-q:]].mean())
            top2.append(ys[o[:2]].mean() - np.median(ys))
            nday.append(len(xs))
            if k % H == 0:
                step_ic.append(ic)
        ics, step_ic = np.array(ics), np.array(step_ic)
        res[name] = dict(days=len(ics), instruments_mean=float(np.mean(nday)) if nday else None, ic_mean=float(np.nanmean(ics)), ic_riskadj_mean=float(np.nanmean(icr)), ic_t_nonoverlap=float(np.nanmean(step_ic) / (np.nanstd(step_ic) / math.sqrt(np.sum(~np.isnan(step_ic))))) if len(step_ic) > 2 else None, ic_positive_share=float(np.mean(ics > 0)), top_minus_bottom_quintile_20d_pct=100 * float(np.mean(spread)), top2_minus_median_20d_pct=100 * float(np.mean(top2)))
    return res


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, pth in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), pth)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    inst, groups = build_all()
    fwd = {c: forward(ins) for c, ins in inst.items()}
    simple = {c: simple_scores(ins) for c, ins in inst.items()}
    samp = A7.samples(inst)
    out = {}
    rng = np.random.default_rng(99)
    for g, codes in groups.items():
        ms = model_scores(inst, codes, samp)
        sets = {k: {c: simple[c][k] for c in codes} for k in ("mom126", "mom252_21", "riskadj126", "rev20", "lowvol60")}
        sets["model_kappa_qmax"] = ms
        sets["random"] = {c: rng.standard_normal(len(inst[c]["C"])) for c in codes}
        out[g] = evaluate(inst, codes, sets, fwd)
        print(g, "instruments", len(codes), flush=True)
        for k, v in out[g].items():
            print(f"   {k:18s} IC {v['ic_mean']:+.4f} (t {v['ic_t_nonoverlap']:+.2f}, positive days {100 * v['ic_positive_share']:.0f}%) risk-adj IC {v['ic_riskadj_mean']:+.4f} | top-minus-bottom quintile {v['top_minus_bottom_quintile_20d_pct']:+.2f}% per 20d, top2-minus-median {v['top2_minus_median_20d_pct']:+.2f}% | days {v['days']} n {v['instruments_mean']:.1f}", flush=True)
    rec = dict(version="sel-diag-v1", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], horizon=H, results=out)
    with open(out_path, "x", encoding="utf-8") as fo:
        json.dump(rec, fo, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
