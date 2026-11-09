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
import alloc_policy_study_v6 as V6
from nosock import allocator as AL
from nosock import engine as E
from nosock import macro as MC
from nosock import run_backtest as RB
from nosock import selector as SEL

OUT = os.path.join(HERE, "results", "alloc_policy_study_v7.json")
DEF = V6.DEF


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Features7(V4.Features4):
    def __init__(self, m, state, macro, diff, use_macro, use_diff):
        super().__init__(m, state)
        self.macro, self.diff, self.um, self.ud = macro, diff, use_macro, use_diff

    def at(self, D, extra=()):
        f = super().at(D)
        if f is None:
            return None
        parts = [f["s"]]
        if self.um:
            x = self.macro.at(D)
            if x is None:
                return None
            parts.append(x)
        if self.ud:
            x = self.diff.at(D)
            if x is None:
                return None
            parts.append(x)
        return dict(f, s=np.concatenate(parts))


def main():
    t0 = time.time()
    inst, groups, names, classes = RB.load_market()
    codes = groups["U29"]
    m = E.Market(inst, codes, names)
    m.raw = A3.load_raw()
    st = SEL.State(m, RB.strategy_for(inst, codes, "dual", m))
    macro = MC.Macro(inst["SPY"]["days"], inst["SPY"]["fxv"])
    diff = MC.RateDiff(inst["148070"]["days"], inst["148070"]["C"])
    base = list(range(len(AL.ASSET_F)))
    cfg = {"none": (False, False), "us": (True, False), "us+diff": (True, True), "diff": (False, True)}
    feats = {k: Features7(m, st, macro, diff, *v) for k, v in cfg.items()}
    variants = [(f"{k} g{g}", k, g) for k in ("none", "us", "diff", "us+diff") for g in (1, 3)]
    data = {k: AL.dataset(m, feats[k], "2006-01-01", MS.ANCHOR, step=5) for k in cfg}
    log("datasets " + ", ".join(f"{k}:{len(v)} from {v[0]['t']}" for k, v in data.items()))
    res = dict(anchor=MS.ANCHOR, state=list(AL.STATE_F) + list(MC.MACRO_F) + list(MC.DIFF_F), variants={}, runs={})
    pols = {}
    for name, k, g in variants:
        rows = data[k]
        P, oos = {}, {}
        for y in range(2012, int(MS.ANCHOR[:4]) + 2):
            tr = [r for r in rows if r["t_end"] < f"{y}-01-01"]
            pol, lam = V6.fit_auto(tr, y, g, base)
            if y > int(MS.ANCHOR[:4]):
                P["latest"] = (pol, lam)
                continue
            P[y] = pol
            te = [r for r in rows if r["t"][:4] == str(y)]
            oos[y] = dict(lam=lam, g=V6.V5.score(pol, te, 1), eq=float(np.mean([math.log(float(np.mean(r["R"] * (1 - r["cost"])))) for r in te])))
        pols[name] = P
        wins = sum(v["g"] > v["eq"] for v in oos.values())
        res["variants"][name] = dict(oos=oos, wins=wins)
        log(f"{name}: OOS log growth {np.mean([v['g'] for v in oos.values()]):+.4f} vs equal {np.mean([v['eq'] for v in oos.values()]):+.4f}, beats equal {wins}/{len(oos)}")
    windows = (("2016~", "2016-01-04"), ("2020~", "2020-01-02"), ("2022~", "2022-01-03"), ("2026-06~", "2026-06-01"), ("2026-10~", "2026-10-01"))
    E.H = 20
    for wname, start in windows:
        row = {}
        for name, k, g in variants:
            r = E.run(m, V4.Strat(m, feats[k], {y: v for y, v in pols[name].items() if y != "latest"}), start, MS.ANCHOR, lambda D: 1, whole=True, tax=True, record=True)
            row[name] = dict(total=r["total"], pct=100 * (r["total"] / r["deposits"] - 1), maxdd=DD.metrics(r)["twr_max_dd"])
        r = E.run(m, RB.strategy_for(inst, codes, "equal", m), start, MS.ANCHOR, lambda D: 1, whole=True, tax=True, record=True)
        row["equal"] = dict(total=r["total"], pct=100 * (r["total"] / r["deposits"] - 1), maxdd=DD.metrics(r)["twr_max_dd"])
        res["runs"][wname] = row
        log(f"{wname}: " + " | ".join(f"{n} {v['total'] / 1e4:,.1f}만 {v['pct']:+.1f}% dd{100 * v['maxdd']:.0f}%" for n, v in row.items()))
    for name, k, g in variants:
        pol, lam = pols[name]["latest"]
        f = feats[k].at(MS.ANCHOR)
        w, cash = pol.weights(f)
        dfn = sum(x for c, x in w.items() if c in DEF) + cash
        labels = list(AL.STATE_F) + (list(MC.MACRO_F) if cfg[k][0] else []) + (list(MC.DIFF_F) if cfg[k][1] else [])
        ga = pol.p["ga"]
        res.setdefault("today", {})[name] = dict(defensive=dfn, lam=lam, gate=dict(zip(labels, map(float, ga))), ca=pol.p["ca"])
        log(f"today {name}: lam {lam:g} defensive {100 * dfn:.0f}% ca {pol.p['ca']:+.2f} gate: " + " ".join(f"{l} {v:+.3f}" for l, v in sorted(zip(labels, ga), key=lambda kv: -abs(kv[1]))[:6]))
        y26 = pols[name].get(2026)
        if y26 is not None:
            path = V6.gate_path(y26, feats[k], m, "2026-05-01", MS.ANCHOR)
            log(f"  2026 policy defensive by week: " + " ".join(f"{d[5:]} {100 * v:.0f}%" for d, v in path))
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=float)
    os.replace(tmp, OUT)
    log(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
