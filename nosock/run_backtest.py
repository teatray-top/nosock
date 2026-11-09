import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ft"))
sys.dont_write_bytecode = True
import numpy as np
import ms_alloc_v3 as A3
import ms_alloc_v7 as A7
import ms_role_v2 as R
import mom_shadow as MS
import sel_diag_v1 as D1
import sel_diag_v2 as D2
from nosock import engine as E
from nosock import synth as S

US_FEE, US_FX = 0.001, 0.0005


def load_market():
    kr, kus, us, fx = MS.pinned()
    inst, groups = MS.build({c: MS.cut(v, MS.ANCHOR) for c, v in kr.items()}, {c: MS.cut(v, MS.ANCHOR) for c, v in kus.items()}, {c: MS.cut(v, MS.ANCHOR) for c, v in us.items()}, {d: v for d, v in fx.items() if d <= MS.ANCHOR}, end=MS.ANCHOR)
    names = json.load(open(os.path.join(R.ETF_DIR, "manifest.json"), encoding="utf-8"))["names"]
    man_us, _ = A7.load_us()
    names = dict(names, **man_us["names"])
    classes = dict(json.load(open(os.path.join(R.ETF_DIR, "manifest.json"), encoding="utf-8")).get("classes", {}))
    import msma_v1 as M1
    classes.update(M1.CLASS)
    classes.update(man_us["classes"])
    for c, ins in inst.items():
        ins["cost_maker"] = A7.kr_cost if ins["market"] == "KR" else A7.us_cost_maker(US_FEE, US_FX)
    return inst, groups, names, classes


def strategy_for(inst, codes, mode, market):
    simple = {c: D1.simple_scores(inst[c]) for c in codes}
    score = D2.composite(inst, codes, simple, D2.by_date(inst, codes))
    return E.Momentum(market, score, {c: simple[c]["mom252_21"] for c in codes}, {c: simple[c]["mom126"] for c in codes}, mode)


POLICIES = os.path.join(HERE, "policies_v1.json")


def model_strategy(inst, codes, market, latest=False):
    from nosock import allocator as AL
    from nosock import selector as SEL
    st = SEL.State(market, strategy_for(inst, codes, "dual", market))
    feats = AL.Features(market, st)
    P = json.load(open(POLICIES, encoding="utf-8"))["years"]
    mk = lambda v: AL.Policy(np.array(v["beta"]), np.array(v["gamma"]), v["c0"], np.array(v["smu"]), np.array(v["ssd"]))
    if latest:
        return AL.Allocator(market, feats, mk(P[max(P, key=int)]))
    return AL.Allocator(market, feats, {int(y): mk(v) for y, v in P.items()})


def strategy_of(inst, codes, mode, market, latest=False):
    return model_strategy(inst, codes, market, latest) if mode == "model" else strategy_for(inst, codes, mode, market)


def hold_one(market, code, start, end, deposit):
    ins = market.inst[code]
    units, dep = 0.0, 0.0
    for D in sorted(d for d in market.days if start <= d <= end and d in market.kr_days):
        o, i, d_exec = market.next_open(code, D)
        if o is None or d_exec > end:
            continue
        units += deposit / (o * (1 + ins["cost"](i, o)))
        dep += deposit
    c, _ = market.close_at(code, end)
    return dict(deposits=dep, total=units * c if c else 0.0)


def weekly(marks):
    out, last = [], None
    for x in marks:
        wk = x["day"][:4] + str(int(x["day"][5:7]) * 5 + int(x["day"][8:10]) // 7)
        if wk != last:
            out.append(x)
            last = wk
        else:
            out[-1] = x
    return out


def slim(decisions):
    for d in decisions:
        if d.get("rule") == "allocator":
            d["candidates"] = [dict(code=c["code"], score=round(c["score"], 3), weight=round(c["weight"], 4), rank=c["rank"], selected=c["selected"], m126=round(c["raw"]["m126"], 4), m12_1=round(c["raw"]["m12_1"], 4)) for c in d.get("candidates", [])]
            d.pop("state_z", None)
    return decisions


def summarize(r, label):
    vy = {}
    for p in r["payouts"]:
        vy[p["day"][:4]] = vy.get(p["day"][:4], 0.0) + p["amount"]
    return dict(label=label, deposits=r["deposits"], vault=r["vault"], account=r["account"], total=r["total"], taxes=r["taxes"], worst_profit_drawdown=r["worst_profit_drawdown"], payouts=len(r["payouts"]), vault_by_year=vy)


def historical(p, inst, groups, names):
    codes = groups[p.get("universe", "U29")]
    m = E.Market(inst, codes, names)
    m.raw = A3.load_raw()
    s = strategy_of(inst, codes, p["mode"], m)
    start, end = p.get("start", "2016-01-01"), p.get("end", MS.ANCHOR)
    years = list(range(int(start[:4]), int(end[:4]) + 1))
    kinfo = None
    if p["mode"] in ("equal", "model"):
        k_of = lambda D: 1
    elif str(p.get("k", "auto")) == "auto":
        kinfo = E.choose_k(m, s, years)
        k_of = lambda D: kinfo[int(D[:4])]["k"]
    else:
        k_of = lambda D, k=int(p["k"]): k
    real = bool(p.get("realistic", True))
    dep = float(p.get("deposit", E.DEPOSIT))
    r = E.run(m, s, start, end, k_of, whole=real, tax=real, record=True, deposit=dep)
    eq = E.run(m, strategy_for(inst, codes, "equal", m), start, end, lambda D: 1, whole=real, tax=real, record=False, deposit=dep)
    bench = dict(equal=summarize(eq, "15종 균등" if False else "같은 종목 균등 분배"), hold_069500=hold_one(m, "069500", start, end, dep), hold_SPY=hold_one(m, "SPY", start, end, dep) if "SPY" in codes else None)
    return dict(kind="historical", params=p, summary=summarize(r, "전략"), benchmarks=bench, k_by_year=kinfo, series=weekly(r["marks"]), decisions=slim(r["decisions"]), fills=r["fills"], payouts=r["payouts"], holdings=r["holdings"], names={c: names.get(c, c) for c in codes})


def synthetic(p, inst, groups, names, classes):
    codes = groups[p.get("universe", "U29")]
    n_paths = int(p.get("paths", 20))
    years = int(p.get("years", 10))
    k = 3 if str(p.get("k", "auto")) == "auto" else int(p["k"])
    real = bool(p.get("realistic", True))
    dep = float(p.get("deposit", E.DEPOSIT))
    warm = 300
    rows, first = [], None
    for i in range(n_paths):
        si, meta = S.generate(inst, codes, classes, p.get("regime", "bootstrap"), warm + years * 252, int(p.get("seed", 1)) * 1000 + i)
        m = E.Market(si, codes, names)
        start = si[codes[0]]["days"][warm]
        end = si[codes[0]]["days"][-1]
        r = E.run(m, strategy_of(si, codes, p["mode"], m, latest=True), start, end, lambda D: k if p["mode"] not in ("equal", "model") else 1, whole=real, tax=real, record=(i == 0), deposit=dep)
        e = E.run(m, strategy_for(si, codes, "equal", m), start, end, lambda D: 1, whole=real, tax=real, record=False, deposit=dep)
        rows.append(dict(path=i, total=r["total"], vault=r["vault"], deposits=r["deposits"], worst=r["worst_profit_drawdown"], payouts=len(r["payouts"]), equal_total=e["total"], equal_vault=e["vault"]))
        if i == 0:
            first = dict(series=weekly(r["marks"]), decisions=slim(r["decisions"]), fills=r["fills"], payouts=r["payouts"], meta=meta)
    arr = lambda k_: np.array([x[k_] for x in rows])
    q = lambda a: dict(p10=float(np.percentile(a, 10)), median=float(np.median(a)), p90=float(np.percentile(a, 90)))
    dist = dict(total=q(arr("total")), vault=q(arr("vault")), worst=q(arr("worst")), equal_total=q(arr("equal_total")), beats_equal=int(sum(x["total"] > x["equal_total"] for x in rows)), loses_money=int(sum(x["total"] < x["deposits"] for x in rows)))
    return dict(kind="synthetic", params=dict(p, k_used=k), paths=rows, distribution=dist, example=first, names={c: names.get(c, c) for c in codes})


def main(params_path, out_path):
    p = json.load(open(params_path, encoding="utf-8"))
    t0 = time.time()
    inst, groups, names, classes = load_market()
    res = historical(p, inst, groups, names) if p.get("data", "historical") == "historical" else synthetic(p, inst, groups, names, classes)
    res["elapsed_s"] = round(time.time() - t0, 1)
    res["data_until"] = MS.ANCHOR
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, default=float)
    os.replace(tmp, out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
