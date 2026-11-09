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
import alloc_policy_study_v5 as V5
from nosock import allocator as AL
from nosock import engine as E
from nosock import macro as MC
from nosock import run_backtest as RB
from nosock import selector as SEL

OUT = os.path.join(HERE, "results", "alloc_policy_study_v6.json")
DEF = V5.DEF
LAMS = AL.LAMS


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Features6(V4.Features4):
    def __init__(self, m, state, macro, use):
        super().__init__(m, state)
        self.macro, self.use = macro, use

    def at(self, D, extra=()):
        f = super().at(D)
        if f is None:
            return None
        if not self.use:
            return f
        x = self.macro.at(D)
        if x is None:
            return None
        return dict(f, s=np.r_[f["s"], x])


class Pol(V5.Pol2):
    pass


def fit(rows, lam, g, idx):
    import torch
    S = np.array([r["s"] for r in rows])
    NS = S.shape[1]
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
        wr = torch.softmax((x_t * br).sum(-1).masked_fill(~mr, -1e9), dim=1)
        wd = torch.softmax(torch.cat([(x_t * bd).sum(-1).masked_fill(~md, -1e9), (s_t * gd).sum(-1, keepdim=True) + cd], dim=1), dim=1)
        port = a * (wr * gross).sum(1) + (1 - a) * ((wd[:, :-1] * gross).sum(1) + wd[:, -1])
        loss = -V4.util(port, g).mean() + lam * sum((v ** 2).sum() for v in (br, bd, ga, gd))
        loss.backward()
        return loss

    opt.step(closure)
    d = lambda v: v.detach().numpy().copy()
    return Pol(dict(br=d(br), bd=d(bd), ga=d(ga), gd=d(gd), ca=float(ca.detach()[0]), cd=float(cd.detach()[0])), smu, ssd, idx)


def fit_auto(rows, y, g, idx):
    cut = f"{y - 3}-01-01"
    fr = [r for r in rows if r["t_end"] < cut]
    va = [r for r in rows if r["t"] >= cut]
    if len(fr) >= 100 and len(va) >= 30:
        sc = {lam: V5.score(fit(fr, lam, g, idx), va, g) for lam in LAMS}
        lam = max(sc, key=sc.get)
    else:
        lam = 1e-2
    return fit(rows, lam, g, idx), lam


def gate_path(pol, feats, m, start, end):
    out = []
    for D in [d for d in m.days if start <= d <= end and d in m.kr_days][::5]:
        f = feats.at(D)
        if f is None:
            continue
        w, cash = pol.weights(f)
        out.append((D, sum(x for c, x in w.items() if c in DEF) + cash))
    return out


def main():
    t0 = time.time()
    inst, groups, names, classes = RB.load_market()
    codes = groups["U29"]
    m = E.Market(inst, codes, names)
    m.raw = A3.load_raw()
    st = SEL.State(m, RB.strategy_for(inst, codes, "dual", m))
    spy = inst["SPY"]
    macro = MC.Macro(spy["days"], spy["fxv"])
    base = list(range(len(AL.ASSET_F)))
    feats = {u: Features6(m, st, macro, u) for u in (False, True)}
    variants = [("v5 g3 (금리 없음)", False, 20, 3), ("v6 g3 (금리 등)", True, 20, 3), ("v6 g1 (금리 등)", True, 20, 1), ("v6 g6 (금리 등)", True, 20, 6), ("v6 H5 g3 (금리 등)", True, 5, 3)]
    data = {}
    for use in (False, True):
        for H in (20, 5):
            AL.H = H
            data[(use, H)] = AL.dataset(m, feats[use], "2006-01-01", MS.ANCHOR, step=5)
    AL.H = 20
    log("datasets " + ", ".join(f"{k}:{len(v)}" for k, v in data.items()))
    res = dict(anchor=MS.ANCHOR, macro=list(MC.MACRO_F), variants={}, runs={})
    pols = {}
    for name, use, H, g in variants:
        rows = data[(use, H)]
        P, oos = {}, {}
        for y in range(2012, int(MS.ANCHOR[:4]) + 2):
            tr = [r for r in rows if r["t_end"] < f"{y}-01-01"]
            pol, lam = fit_auto(tr, y, g, base)
            if y > int(MS.ANCHOR[:4]):
                P["latest"] = pol
                continue
            P[y] = pol
            te = [r for r in rows if r["t"][:4] == str(y)]
            oos[y] = dict(lam=lam, g=V5.score(pol, te, 1), eq=float(np.mean([math.log(float(np.mean(r["R"] * (1 - r["cost"])))) for r in te])))
        pols[name] = P
        wins = sum(v["g"] > v["eq"] for v in oos.values())
        res["variants"][name] = dict(oos=oos, wins=wins)
        log(f"{name}: OOS log growth {np.mean([v['g'] for v in oos.values()]):+.4f} vs equal {np.mean([v['eq'] for v in oos.values()]):+.4f}, beats equal {wins}/{len(oos)}")
    windows = (("2016~", "2016-01-04"), ("2020~", "2020-01-02"), ("2022~", "2022-01-03"), ("2026-06~", "2026-06-01"), ("2026-10~", "2026-10-01"))
    for wname, start in windows:
        row = {}
        for name, use, H, g in variants:
            E.H = H
            r = E.run(m, V4.Strat(m, feats[use], {k: v for k, v in pols[name].items() if k != "latest"}), start, MS.ANCHOR, lambda D: 1, whole=True, tax=True, record=True)
            row[name] = dict(total=r["total"], pct=100 * (r["total"] / r["deposits"] - 1), maxdd=DD.metrics(r)["twr_max_dd"])
        E.H = 20
        r = E.run(m, RB.strategy_for(inst, codes, "equal", m), start, MS.ANCHOR, lambda D: 1, whole=True, tax=True, record=True)
        row["균등"] = dict(total=r["total"], pct=100 * (r["total"] / r["deposits"] - 1), maxdd=DD.metrics(r)["twr_max_dd"])
        res["runs"][wname] = row
        log(f"{wname}: " + " | ".join(f"{k} {v['total'] / 1e4:,.1f}만 {v['pct']:+.1f}% dd{100 * v['maxdd']:.0f}%" for k, v in row.items()))
    for name, use, H, g in variants:
        f = feats[use].at(MS.ANCHOR)
        w, cash = pols[name]["latest"].weights(f)
        dfn = sum(x for c, x in w.items() if c in DEF) + cash
        top = sorted(w.items(), key=lambda kv: -kv[1])[:6]
        res.setdefault("today", {})[name] = dict(top=top, cash=cash, defensive=dfn)
        log(f"today {name}: defensive {100 * dfn:.0f}% (cash {100 * cash:.0f}%) " + " ".join(f"{c} {100 * x:.0f}%" for c, x in top))
        if use:
            pol = pols[name]["latest"]
            ga = pol.p["ga"]
            labels = list(AL.STATE_F) + list(MC.MACRO_F)
            log("  gate weights (+ = more invested): " + " ".join(f"{l} {v:+.2f}" for l, v in sorted(zip(labels, ga), key=lambda kv: -abs(kv[1]))))
    for name in ("v6 g3 (금리 등)",):
        use = True
        path = []
        for y in range(2016, int(MS.ANCHOR[:4]) + 1):
            path += gate_path(pols[name][y], feats[use], m, f"{y}-01-01", f"{y}-12-31")
        res["gate_path"] = path
        hi = [p for p in path if p[1] > 0.3]
        log(f"{name}: weeks with defensive>30%: {len(hi)}/{len(path)}; " + ", ".join(f"{d} {100 * v:.0f}%" for d, v in hi[-12:]))
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=float)
    os.replace(tmp, OUT)
    log(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
