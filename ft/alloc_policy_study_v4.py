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
import alloc_policy_study_v3 as V3
from nosock import allocator as AL
from nosock import engine as E
from nosock import run_backtest as RB
from nosock import selector as SEL

OUT = os.path.join(HERE, "results", "alloc_policy_study_v4.json")
LAMS = AL.LAMS
ABS = (("m20", 0.08), ("m126", 0.2), ("m12_1", 0.3), ("gap200", 0.1), ("dd252", 0.2))
F4 = AL.ASSET_F + tuple("abs_" + k for k, _ in ABS) + ("beta60",)
NS = len(AL.STATE_F)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Features4(V3.Features3):
    def at(self, D, extra=()):
        f = super().at(D)
        if f is None:
            return None
        raw = f["raw"]
        cols = [np.clip(raw[:, AL.ASSET_F.index(k)] / s, -4, 4) for k, s in ABS]
        Z = np.hstack([f["Z"][:, :len(AL.ASSET_F)], np.array(cols).T, f["Z"][:, -1:]])
        return dict(f, Z=Z)


def util(port, g):
    import torch
    return torch.log(port) if g == 1 else (port.pow(1 - g) - 1) / (1 - g)


def util_np(p, g):
    return math.log(p) if g == 1 else (p ** (1 - g) - 1) / (1 - g)


class Pol:
    def __init__(self, beta, gamma, c0, smu, ssd, idx):
        self.beta, self.gamma, self.c0, self.smu, self.ssd, self.idx = beta, gamma, c0, smu, ssd, idx

    def weights(self, f):
        sz = np.clip((f["s"] - self.smu) / self.ssd, -4, 4)
        sc = f["Z"][:, self.idx] @ self.beta
        z = np.r_[sc, float(sz @ self.gamma + self.c0)]
        z = z - z.max()
        a = np.exp(z)
        a = a / a.sum()
        w = {c: float(x) for c, x in zip(f["codes"], a[:-1]) if x >= AL.MIN_W}
        return w, float(a[-1])


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
    M = np.zeros((T, nmax), bool)
    for t, r in enumerate(rows):
        n = len(r["codes"])
        X[t, :n] = r["Z"][:, idx]
        R[t, :n], K[t, :n], M[t, :n] = r["R"], r["cost"], True
    x_t, r_t, k_t, m_t, s_t = (torch.tensor(a) for a in (X, R, K, M, Sz))
    beta = torch.zeros(P, dtype=torch.float64, requires_grad=True)
    gamma = torch.zeros(NS, dtype=torch.float64, requires_grad=True)
    c0 = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([beta, gamma, c0], max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        sc = (x_t * beta).sum(-1).masked_fill(~m_t, -1e9)
        cash = (s_t * gamma).sum(-1, keepdim=True) + c0
        a = torch.softmax(torch.cat([sc, cash], dim=1), dim=1)
        port = (a[:, :-1] * r_t * (1 - k_t)).sum(1) + a[:, -1]
        loss = -util(port, g).mean() + lam * ((beta ** 2).sum() + (gamma ** 2).sum())
        loss.backward()
        return loss

    opt.step(closure)
    return Pol(beta.detach().numpy().copy(), gamma.detach().numpy().copy(), float(c0.detach()[0]), smu, ssd, idx)


def score(pol, rows, g):
    u = []
    for r in rows:
        w, _ = pol.weights(dict(codes=r["codes"], Z=r["Z"], s=r["s"]))
        port = sum(w.get(c, 0.0) * R * (1 - k) for c, R, k in zip(r["codes"], r["R"], r["cost"])) + (1 - sum(w.values()))
        u.append(util_np(port, g))
    return float(np.mean(u)) if u else -1e9


def logg(pol, rows):
    return score(pol, rows, 1)


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


class Strat:
    def __init__(self, m, feats, pols):
        self.m, self.feats, self.pols = m, feats, pols

    def decide(self, D, k):
        pol = self.pols.get(int(D[:4])) if isinstance(self.pols, dict) else self.pols
        f = self.feats.at(D)
        if pol is None or f is None:
            el = self.m.eligible(D)
            return ({c: 1.0 / len(el) for c in el} if el else {}), dict(rule="allocator", candidates=[])
        w, cash = pol.weights(f)
        return w, dict(rule="allocator", k=len(w), cash_weight=max(0.0, 1 - sum(w.values())), candidates=[])


def main():
    t0 = time.time()
    inst, groups, names, classes = RB.load_market()
    codes = groups["U29"]
    m = E.Market(inst, codes, names)
    m.raw = A3.load_raw()
    st = SEL.State(m, RB.strategy_for(inst, codes, "dual", m))
    feats = Features4(m, st)
    base = list(range(len(AL.ASSET_F)))
    full = list(range(len(F4)))
    variants = [("v2", 20, 1, base)] + [(f"v4 H{H} g{g}", H, g, full) for H in (20, 5) for g in (1, 3, 6)]
    data = {}
    for H in (20, 5):
        AL.H = H
        data[H] = AL.dataset(m, feats, "2006-01-01", MS.ANCHOR, step=5)
        log(f"H{H} dataset {len(data[H])} rows")
    AL.H = 20
    res = dict(anchor=MS.ANCHOR, features=list(F4), variants={}, runs={})
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
            oos[y] = dict(lam=lam, g=logg(pol, te), eq=float(np.mean([math.log(float(np.mean(r["R"] * (1 - r["cost"])))) for r in te])))
        pols[name] = P
        wins = sum(v["g"] > v["eq"] for v in oos.values())
        res["variants"][name] = dict(H=H, g=g, oos=oos, wins=wins, pooled=float(np.mean([v["g"] for v in oos.values()])), pooled_eq=float(np.mean([v["eq"] for v in oos.values()])))
        log(f"{name}: OOS log growth per window {res['variants'][name]['pooled']:+.4f} vs equal {res['variants'][name]['pooled_eq']:+.4f}, beats equal {wins}/{len(oos)} years, lams " + ",".join(f"{v['lam']:g}" for v in oos.values()))
    windows = (("2016~", "2016-01-04"), ("2020~", "2020-01-02"), ("2026-06~", "2026-06-01"), ("2026-10~", "2026-10-01"))
    for wname, start in windows:
        row = {}
        for name, H, g, idx in variants:
            E.H = H
            r = E.run(m, Strat(m, feats, {k: v for k, v in pols[name].items() if k != "latest"}), start, MS.ANCHOR, lambda D: 1, whole=True, tax=True, record=True)
            mt = DD.metrics(r)
            row[name] = dict(total=r["total"], deposits=r["deposits"], pct=100 * (r["total"] / r["deposits"] - 1), maxdd=mt["twr_max_dd"], vault=r["vault"])
        E.H = 20
        r = E.run(m, RB.strategy_for(inst, codes, "equal", m), start, MS.ANCHOR, lambda D: 1, whole=True, tax=True, record=True)
        mt = DD.metrics(r)
        row["equal"] = dict(total=r["total"], deposits=r["deposits"], pct=100 * (r["total"] / r["deposits"] - 1), maxdd=mt["twr_max_dd"], vault=r["vault"])
        res["runs"][wname] = row
        log(f"{wname}: " + " | ".join(f"{k} {v['total'] / 1e4:,.1f}만 {v['pct']:+.1f}% dd{100 * v['maxdd']:.0f}%" for k, v in row.items()))
    f = feats.at(MS.ANCHOR)
    for name, H, g, idx in variants:
        w, cash = pols[name]["latest"].weights(f)
        top = sorted(w.items(), key=lambda kv: -kv[1])[:6]
        res.setdefault("today", {})[name] = dict(top=top, cash=1 - sum(w.values()), n=len(w))
        log(f"today {name}: cash {100 * (1 - sum(w.values())):.0f}% n={len(w)} " + " ".join(f"{c} {100 * x:.0f}%" for c, x in top))
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=float)
    os.replace(tmp, OUT)
    log(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
