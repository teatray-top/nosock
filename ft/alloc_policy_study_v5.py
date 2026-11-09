import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import numpy as np
import ms_alloc_v3 as A3
import mom_shadow as MS
import dd_study_v1 as DD
import alloc_policy_study_v4 as V4
from nosock import allocator as AL
from nosock import engine as E
from nosock import run_backtest as RB
from nosock import selector as SEL

OUT = os.path.join(HERE, "results", "alloc_policy_study_v5.json")
DEF = ("153130", "114260", "SHY", "114800", "SH")
NS = len(AL.STATE_F)
LAMS = AL.LAMS


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Pol2:
    def __init__(self, p, smu, ssd, idx):
        self.p, self.smu, self.ssd, self.idx = p, smu, ssd, idx

    def weights(self, f):
        p = self.p
        sz = np.clip((f["s"] - self.smu) / self.ssd, -4, 4)
        a = 1 / (1 + math.exp(-(float(sz @ p["ga"]) + p["ca"])))
        Z = f["Z"][:, self.idx]
        isd = np.array([c in DEF for c in f["codes"]])
        w = {}
        if (~isd).any():
            s = Z[~isd] @ p["br"]
            e = np.exp(s - s.max())
            e /= e.sum()
            for c, x in zip(np.array(f["codes"])[~isd], e):
                w[c] = a * float(x)
        sd = np.r_[Z[isd] @ p["bd"], float(sz @ p["gd"] + p["cd"])]
        e = np.exp(sd - sd.max())
        e /= e.sum()
        for c, x in zip(np.array(f["codes"])[isd], e[:-1]):
            w[c] = (1 - a) * float(x)
        w = {c: x for c, x in w.items() if x >= AL.MIN_W}
        return w, 1 - sum(w.values())


def fit(rows, lam, g, idx):
    import torch
    S = np.array([r["s"] for r in rows])
    smu, ssd = S.mean(axis=0), S.std(axis=0) + 1e-9
    Sz = np.clip((S - smu) / ssd, -4, 4)
    nmax = max(len(r["codes"]) for r in rows)
    T, P = len(rows), len(idx)
    X = np.zeros((T, nmax, P))
    R = np.ones((T, nmax))
    K = np.zeros((T, nmax))
    Mr = np.zeros((T, nmax), bool)
    Md = np.zeros((T, nmax), bool)
    for t, r in enumerate(rows):
        n = len(r["codes"])
        X[t, :n] = r["Z"][:, idx]
        R[t, :n], K[t, :n] = r["R"], r["cost"]
        d = np.array([c in DEF for c in r["codes"]])
        Mr[t, :n], Md[t, :n] = ~d, d
    x_t, r_t, k_t, mr, md, s_t = (torch.tensor(a) for a in (X, R, K, Mr, Md, Sz))
    z = lambda n: torch.zeros(n, dtype=torch.float64, requires_grad=True)
    br, bd, ga, gd, ca, cd = z(P), z(P), z(NS), z(NS), z(1), z(1)
    opt = torch.optim.LBFGS([br, bd, ga, gd, ca, cd], max_iter=300, line_search_fn="strong_wolfe")
    gross = r_t * (1 - k_t)

    def closure():
        opt.zero_grad()
        a = torch.sigmoid((s_t * ga).sum(-1) + ca)
        sr = (x_t * br).sum(-1).masked_fill(~mr, -1e9)
        wr = torch.softmax(sr, dim=1)
        sd = torch.cat([(x_t * bd).sum(-1).masked_fill(~md, -1e9), (s_t * gd).sum(-1, keepdim=True) + cd], dim=1)
        wd = torch.softmax(sd, dim=1)
        risky = (wr * gross).sum(1)
        dfn = (wd[:, :-1] * gross).sum(1) + wd[:, -1]
        port = a * risky + (1 - a) * dfn
        loss = -V4.util(port, g).mean() + lam * sum((v ** 2).sum() for v in (br, bd, ga, gd))
        loss.backward()
        return loss

    opt.step(closure)
    d = lambda v: v.detach().numpy().copy()
    return Pol2(dict(br=d(br), bd=d(bd), ga=d(ga), gd=d(gd), ca=float(ca.detach()[0]), cd=float(cd.detach()[0])), smu, ssd, idx)


def score(pol, rows, g):
    u = []
    for r in rows:
        w, _ = pol.weights(dict(codes=r["codes"], Z=r["Z"], s=r["s"]))
        port = sum(w.get(c, 0.0) * R * (1 - k) for c, R, k in zip(r["codes"], r["R"], r["cost"])) + (1 - sum(w.values()))
        u.append(V4.util_np(port, g))
    return float(np.mean(u)) if u else -1e9


def fit_auto(rows, y, g, idx):
    cut = f"{y - 3}-01-01"
    fr = [r for r in rows if r["t_end"] < cut]
    va = [r for r in rows if r["t"] >= cut]
    if len(fr) >= 100 and len(va) >= 30:
        sc = {lam: score(fit(fr, lam, g, idx), va, g) for lam in LAMS}
        lam = max(sc, key=sc.get)
    else:
        lam = 1e-2
    return fit(rows, lam, g, idx), lam


def main():
    t0 = time.time()
    inst, groups, names, classes = RB.load_market()
    codes = groups["U29"]
    m = E.Market(inst, codes, names)
    m.raw = A3.load_raw()
    st = SEL.State(m, RB.strategy_for(inst, codes, "dual", m))
    feats = V4.Features4(m, st)
    base = list(range(len(AL.ASSET_F)))
    full = list(range(len(V4.F4)))
    variants = [(f"v5 {fn} H{H} g{g}", H, g, idx) for fn, idx in (("base", base), ("full", full)) for H in (20, 5) for g in (1, 3)]
    data = {}
    for H in (20, 5):
        AL.H = H
        data[H] = AL.dataset(m, feats, "2006-01-01", MS.ANCHOR, step=5)
    AL.H = 20
    res = dict(anchor=MS.ANCHOR, defensive=DEF, variants={}, runs={})
    pols = {}
    for name, H, g, idx in variants:
        rows = data[H]
        P, oos = {}, {}
        for y in range(2012, int(MS.ANCHOR[:4]) + 2):
            tr = [r for r in rows if r["t_end"] < f"{y}-01-01"]
            pol, lam = fit_auto(tr, y, g, idx)
            if y > int(MS.ANCHOR[:4]):
                P["latest"] = pol
                continue
            P[y] = pol
            te = [r for r in rows if r["t"][:4] == str(y)]
            oos[y] = dict(lam=lam, g=score(pol, te, 1), eq=float(np.mean([math.log(float(np.mean(r["R"] * (1 - r["cost"])))) for r in te])))
        pols[name] = P
        wins = sum(v["g"] > v["eq"] for v in oos.values())
        res["variants"][name] = dict(oos=oos, wins=wins)
        log(f"{name}: OOS log growth {np.mean([v['g'] for v in oos.values()]):+.4f} vs equal {np.mean([v['eq'] for v in oos.values()]):+.4f}, beats equal {wins}/{len(oos)}")
    windows = (("2016~", "2016-01-04"), ("2020~", "2020-01-02"), ("2026-06~", "2026-06-01"), ("2026-10~", "2026-10-01"))
    for wname, start in windows:
        row = {}
        for name, H, g, idx in variants:
            E.H = H
            r = E.run(m, V4.Strat(m, feats, {k: v for k, v in pols[name].items() if k != "latest"}), start, MS.ANCHOR, lambda D: 1, whole=True, tax=True, record=True)
            mt = DD.metrics(r)
            row[name] = dict(total=r["total"], pct=100 * (r["total"] / r["deposits"] - 1), maxdd=mt["twr_max_dd"])
        E.H = 20
        r = E.run(m, RB.strategy_for(inst, codes, "equal", m), start, MS.ANCHOR, lambda D: 1, whole=True, tax=True, record=True)
        row["equal"] = dict(total=r["total"], pct=100 * (r["total"] / r["deposits"] - 1), maxdd=DD.metrics(r)["twr_max_dd"])
        res["runs"][wname] = row
        log(f"{wname}: " + " | ".join(f"{k} {v['total'] / 1e4:,.1f}만 {v['pct']:+.1f}% dd{100 * v['maxdd']:.0f}%" for k, v in row.items()))
    f = feats.at(MS.ANCHOR)
    for name, H, g, idx in variants:
        w, cash = pols[name]["latest"].weights(f)
        dfn = sum(x for c, x in w.items() if c in DEF) + cash
        top = sorted(w.items(), key=lambda kv: -kv[1])[:6]
        res.setdefault("today", {})[name] = dict(top=top, cash=cash, defensive=dfn)
        log(f"today {name}: defensive {100 * dfn:.0f}% (cash {100 * cash:.0f}%) " + " ".join(f"{c} {100 * x:.0f}%" for c, x in top))
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=float)
    os.replace(tmp, OUT)
    log(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
