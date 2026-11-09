from datetime import date, timedelta

import numpy as np

EQUITY_CLASSES = {"kr_eq", "kr_sector", "us_eq", "us_small", "us_sector", "dev_eq", "em_eq", "us_tech", "us_semi", "us_energy", "us_health", "us_reit", "us_div", "silver", "oil"}
REGIMES = ("bootstrap", "bear", "sideways", "volatile")


def joint_returns(inst, codes):
    common = sorted(set.intersection(*[set(inst[c]["days"]) for c in codes]))
    pos = {c: {d: i for i, d in enumerate(inst[c]["days"])} for c in codes}
    rc = np.zeros((len(common) - 1, len(codes)))
    ro = np.zeros((len(common) - 1, len(codes)))
    for j, c in enumerate(codes):
        ins = inst[c]
        for t in range(1, len(common)):
            i0, i1 = pos[c][common[t - 1]], pos[c][common[t]]
            rc[t - 1, j] = np.log(ins["C"][i1] / ins["C"][i0])
            ro[t - 1, j] = np.log(ins["O"][i1] / ins["C"][i0])
    return common, rc, ro


def weekdays(start, n):
    out, d = [], date.fromisoformat(start)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def generate(inst, codes, classes, regime, n_days, seed, block=20, start="2000-01-03"):
    common, rc, ro = joint_returns(inst, codes)
    rng = np.random.default_rng(seed)
    T = len(rc)
    idx = []
    while len(idx) < n_days:
        s = int(rng.integers(0, T - block))
        idx.extend(range(s, s + block))
    idx = np.array(idx[:n_days])
    c_ret, o_ret = rc[idx].copy(), ro[idx].copy()
    eq = np.array([classes.get(c) in EQUITY_CLASSES for c in codes])
    if regime == "bear":
        c_ret[:, eq] += -0.20 / 252
    elif regime == "sideways":
        c_ret -= rc.mean(axis=0)
    elif regime == "volatile":
        mu = rc.mean(axis=0)
        c_ret = mu + 1.5 * (c_ret - mu)
        o_ret = 1.5 * o_ret
    days = weekdays(start, n_days + 1)
    out = {}
    for j, c in enumerate(codes):
        C = np.empty(n_days + 1)
        O = np.empty(n_days + 1)
        C[0] = O[0] = 10_000.0
        for t in range(n_days):
            O[t + 1] = C[t] * np.exp(o_ret[t, j])
            C[t + 1] = C[t] * np.exp(c_ret[t, j])
        H = np.maximum(O, C)
        L = np.minimum(O, C)
        src = inst[c]
        ins = dict(code=c, market=src["market"], days=list(days), O=O, H=H, L=L, C=C, fxv=np.full(n_days + 1, 1400.0))
        ins["cost"] = src["cost_maker"](ins) if "cost_maker" in src else src["cost"]
        out[c] = ins
    return out, dict(regime=regime, seed=seed, block=block, sample_days=len(common), sample_from=common[0], sample_to=common[-1])
