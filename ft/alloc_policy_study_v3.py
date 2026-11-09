import bisect
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
from nosock import allocator as AL
from nosock import engine as E
from nosock import run_backtest as RB
from nosock import selector as SEL
from nosock import synth as SY

OUT = os.path.join(HERE, "results", "alloc_policy_study_v3.json")
F3 = AL.ASSET_F + ("beta60",)
NF, NS = len(F3), len(AL.STATE_F)
LAMS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Features3(AL.Features):
    def __init__(self, m, state):
        super().__init__(m, state)
        n = len(m.days)
        r, eq = state.r, state.eq_ret
        self.beta = np.full((n, len(m.codes)), np.nan)
        for t in range(60, n):
            e = eq[t - 59:t + 1]
            v = e.var()
            if v <= 0:
                continue
            x = r[t - 59:t + 1]
            self.beta[t] = ((x - x.mean(axis=0)) * (e - e.mean())[:, None]).mean(axis=0) / v
        self.col = {c: j for j, c in enumerate(m.codes)}

    def at(self, D, extra=()):
        f = super().at(D)
        if f is None:
            return None
        t = bisect.bisect_right(self.m.days, D) - 1
        b = np.array([self.beta[t, self.col[c]] for c in f["codes"]])
        b = np.where(np.isnan(b), 1.0, b)
        zb = np.clip((b - 0.7) / 0.5, -4, 4)
        f = dict(f)
        f["Z"] = np.hstack([f["Z"], zb[:, None]])
        f["raw"] = np.hstack([f["raw"], b[:, None]])
        return f


def design(Z, sz, inter):
    if not inter:
        return Z
    return np.hstack([Z, (Z[:, :, None] * sz[None, None, :]).reshape(Z.shape[0], -1)])


class Policy3:
    def __init__(self, beta, gamma, c0, smu, ssd, inter):
        self.beta, self.gamma, self.c0, self.smu, self.ssd, self.inter = beta, gamma, c0, smu, ssd, inter

    def weights(self, f):
        sz = np.clip((f["s"] - self.smu) / self.ssd, -4, 4)
        sc = design(f["Z"], sz, self.inter) @ self.beta
        z = np.r_[sc, float(sz @ self.gamma + self.c0)]
        z = z - z.max()
        a = np.exp(z)
        a = a / a.sum()
        w = {c: float(x) for c, x in zip(f["codes"], a[:-1]) if x >= AL.MIN_W}
        return w, float(a[-1]), sc, sz


def fit3(rows, lam, inter, nf):
    import torch
    S = np.array([r["s"] for r in rows])
    smu, ssd = S.mean(axis=0), S.std(axis=0) + 1e-9
    Sz = np.clip((S - smu) / ssd, -4, 4)
    nmax = max(len(r["codes"]) for r in rows)
    T = len(rows)
    P = nf + (nf * NS if inter else 0)
    X = np.zeros((T, nmax, P))
    R = np.ones((T, nmax))
    K = np.zeros((T, nmax))
    M = np.zeros((T, nmax), bool)
    for t, r in enumerate(rows):
        n = len(r["codes"])
        X[t, :n] = design(r["Z"][:, :nf], Sz[t], inter)
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
        loss = -torch.log(port).mean() + lam * ((beta ** 2).sum() + (gamma ** 2).sum())
        loss.backward()
        return loss

    opt.step(closure)
    return Policy3(beta.detach().numpy().copy(), gamma.detach().numpy().copy(), float(c0.detach()[0]), smu, ssd, inter)


def view(r, nf):
    return dict(codes=r["codes"], Z=r["Z"][:, :nf], s=r["s"])


def growth(pol, rows, nf):
    g = []
    for r in rows:
        w, _, _, _ = pol.weights(view(r, nf))
        port = sum(w.get(c, 0.0) * R * (1 - k) for c, R, k in zip(r["codes"], r["R"], r["cost"])) + (1 - sum(w.values()))
        g.append(math.log(port))
    return float(np.mean(g)) if g else -1e9


def fit_auto3(rows, y, inter, nf):
    cut = f"{y - 3}-01-01"
    fr = [r for r in rows if r["t_end"] < cut]
    va = [r for r in rows if r["t"] >= cut]
    if len(fr) >= 100 and len(va) >= 30:
        sc = {lam: growth(fit3(fr, lam, inter, nf), va, nf) for lam in LAMS}
        lam = max(sc, key=sc.get)
    else:
        lam = AL.LAMBDA
    return fit3(rows, lam, inter, nf), lam


class Alloc3:
    def __init__(self, m, feats, pols, nf):
        self.m, self.feats, self.pols, self.nf = m, feats, pols, nf

    def decide(self, D, k):
        pol = self.pols.get(int(D[:4])) if isinstance(self.pols, dict) else self.pols
        f = self.feats.at(D)
        if pol is None or f is None:
            el = self.m.eligible(D)
            return ({c: 1.0 / len(el) for c in el} if el else {}), dict(rule="allocator", note="학습 전 균등", candidates=[])
        w, cash, sc, sz = pol.weights(dict(codes=f["codes"], Z=f["Z"][:, :self.nf], s=f["s"]))
        return w, dict(rule="allocator", k=len(w), cash_weight=max(0.0, 1 - sum(w.values())), candidates=[])


def pnl(r):
    return dict(total=r["total"], deposits=r["deposits"], pnl=r["total"] - r["deposits"], pct=100 * (r["total"] / r["deposits"] - 1), vault=r["vault"], worst=r["worst_profit_drawdown"])


def main():
    t0 = time.time()
    inst, groups, names, classes = RB.load_market()
    codes = groups["U29"]
    m = E.Market(inst, codes, names)
    m.raw = A3.load_raw()
    st = SEL.State(m, RB.strategy_for(inst, codes, "dual", m))
    feats = Features3(m, st)
    rows = AL.dataset(m, feats, "2006-01-01", MS.ANCHOR)
    log(f"dataset {len(rows)} rows")
    variants = {"v2": (False, len(AL.ASSET_F)), "v3b": (False, NF), "v3i": (True, NF)}
    pols = {k: {} for k in variants}
    latest = {}
    oos = {k: {} for k in variants}
    for y in range(2012, int(MS.ANCHOR[:4]) + 2):
        tr = [r for r in rows if r["t_end"] < f"{y}-01-01"]
        te = [r for r in rows if r["t"][:4] == str(y)]
        for k, (inter, nf) in variants.items():
            pol, lam = fit_auto3(tr, y, inter, nf)
            if y <= int(MS.ANCHOR[:4]):
                pols[k][y] = pol
                g = growth(pol, te, nf) if te else None
                eqg = float(np.mean([math.log(float(np.mean(r["R"] * (1 - r["cost"])))) for r in te])) if te else None
                oos[k][y] = dict(lam=lam, g=g, eq=eqg)
            else:
                latest[k] = pol
        log(f"{y} " + " ".join(f"{k} lam={oos[k][y]['lam']:g} g={oos[k][y]['g']:+.4f}" for k in variants if y in oos[k]) + (f" eq={oos['v2'][y]['eq']:+.4f}" if y in oos["v2"] else " latest fitted"))
    res = dict(anchor=MS.ANCHOR, oos=oos, runs={}, bear={})
    for k in variants:
        ys = [v for v in oos[k].values() if v["g"] is not None]
        res.setdefault("pooled", {})[k] = dict(g=float(np.mean([v["g"] for v in ys])), eq=float(np.mean([v["eq"] for v in ys])), wins=sum(v["g"] > v["eq"] for v in ys), n=len(ys))
        log(f"pooled {k}: {res['pooled'][k]}")
    windows = (("2016~", "2016-01-04"), ("2020~", "2020-01-02"), ("2026-06~", "2026-06-01"), ("2026-10~", "2026-10-01"))
    for H in (20, 5):
        E.H = H
        for wname, start in windows:
            row = {}
            for k, (inter, nf) in variants.items():
                row[k] = pnl(E.run(m, Alloc3(m, feats, pols[k], nf), start, MS.ANCHOR, lambda D: 1, whole=True, tax=True, record=False))
            row["equal"] = pnl(E.run(m, RB.strategy_for(inst, codes, "equal", m), start, MS.ANCHOR, lambda D: 1, whole=True, tax=True, record=False))
            res["runs"][f"H{H} {wname}"] = row
            log(f"H{H} {wname}: " + " | ".join(f"{k} {v['total'] / 1e4:,.1f}만 ({v['pct']:+.1f}%)" for k, v in row.items()))
    E.H = 20
    for regime in ("bear", "volatile"):
        out = {k: [] for k in list(variants) + ["equal"]}
        for i in range(5):
            si, meta = SY.generate(inst, codes, classes, regime, 300 + 5 * 252, 7000 + i)
            sm = E.Market(si, codes, names)
            sst = SEL.State(sm, RB.strategy_for(si, codes, "dual", sm))
            sf = Features3(sm, sst)
            a, b = si[codes[0]]["days"][300], si[codes[0]]["days"][-1]
            for k, (inter, nf) in variants.items():
                out[k].append(E.run(sm, Alloc3(sm, sf, latest[k], nf), a, b, lambda D: 1, whole=True, tax=True, record=False)["total"])
            out["equal"].append(E.run(sm, RB.strategy_for(si, codes, "equal", sm), a, b, lambda D: 1, whole=True, tax=True, record=False)["total"])
        res["bear"][regime] = {k: dict(median=float(np.median(v)), all=[float(x) for x in v]) for k, v in out.items()}
        log(f"{regime} 5 paths x 5y median: " + " | ".join(f"{k} {np.median(v) / 1e4:,.0f}만" for k, v in out.items()) + " | v3i beats v2 " + str(sum(x > y for x, y in zip(out["v3i"], out["v2"]))) + "/5, v3i beats equal " + str(sum(x > y for x, y in zip(out["v3i"], out["equal"]))) + "/5")
    lp = latest["v3i"]
    res["latest_v3i"] = dict(beta=[float(x) for x in lp.beta], gamma=[float(x) for x in lp.gamma], c0=lp.c0, smu=[float(x) for x in lp.smu], ssd=[float(x) for x in lp.ssd], features=list(F3), state=list(AL.STATE_F))
    f = feats.at(MS.ANCHOR)
    for k, (inter, nf) in variants.items():
        w, cash, _, _ = latest[k].weights(dict(codes=f["codes"], Z=f["Z"][:, :nf], s=f["s"]))
        top = sorted(w.items(), key=lambda kv: -kv[1])[:6]
        res.setdefault("today", {})[k] = dict(top=top, cash=1 - sum(w.values()))
        log(f"today {k}: cash {100 * (1 - sum(w.values())):.0f}% " + " ".join(f"{names.get(c, c)} {100 * x:.0f}%" for c, x in top))
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=float)
    os.replace(tmp, OUT)
    log(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
