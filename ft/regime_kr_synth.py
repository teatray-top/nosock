import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import numpy as np
import regime_compare as RC
import ms_alloc_v3 as A3
from nosock import engine as E
from nosock import run_backtest as RB
from nosock import synth as S

KR_RISK = {"069500", "229200", "091160", "091170"}
OUT = os.path.join(os.path.dirname(HERE), "ft", "results", "regime_kr_synth_v1.json")


def trend_state(days, closes, n_ma, lo=0.96, hi=1.01):
    st, out = 1.0, {}
    for k, D in enumerate(days):
        if k + 1 >= n_ma:
            ma = float(np.mean(closes[k + 1 - n_ma:k + 1]))
            if st == 1.0 and closes[k] < ma * lo:
                st = 0.0
            elif st == 0.0 and closes[k] > ma * hi:
                st = 1.0
        out[D] = st
    return out


def make_xform(state, to_inverse):
    def f(D, te):
        if state.get(D, 1.0) >= 1.0:
            return te
        out = dict(te)
        freed = sum(out.pop(c, 0.0) for c in KR_RISK if c in out)
        if freed <= 0:
            return out
        out["114800"] = out.get("114800", 0.0) + freed * to_inverse
        out["153130"] = out.get("153130", 0.0) + freed * (1 - to_inverse)
        return out
    return f


def variants(m):
    ins = m.inst["069500"]
    days, closes = ins["days"], np.asarray(ins["C"], float)
    s60 = trend_state(days, closes, 60)
    s200 = trend_state(days, closes, 200, 0.98, 1.01)
    return [("R1", None), ("K60 절반 인버스", make_xform(s60, 0.5)), ("K60 단기채", make_xform(s60, 0.0)), ("K200 절반 인버스", make_xform(s200, 0.5))]


def one(m, strat, start, end, xf):
    marks, info = RC.run_x(m, strat, start, end, decide="h20", band=0.02, xform=xf)
    r = RC.stats(marks, info)
    return dict(cagr=r["cagr"], mdd=r["mdd"], worst20=r["worst20"], cost=r["cost"], trades=r["trades"])


def main():
    t0 = time.time()
    inst, groups, names, classes = RB.load_market()
    codes = groups["U29"]
    res = dict(historical={}, synthetic={})
    m = E.Market(inst, codes, names)
    m.raw = A3.load_raw()
    model = RB.model_strategy(inst, codes, m)
    for start in ("2009-05-01", "2017-01-02"):
        for name, xf in variants(m):
            r = one(m, model, start, max(m.days), xf)
            res["historical"][f"{start} {name}"] = r
            print(f"[hist {start}] {name}: CAGR {100 * r['cagr']:+.2f}% MDD {100 * r['mdd']:.1f}% worst20 {100 * r['worst20']:.1f}%", flush=True)
    n_paths = int(os.environ.get("PATHS", "30"))
    years, warm = 10, 300
    for regime in S.REGIMES:
        rows = []
        for i in range(n_paths):
            si, meta = S.generate(inst, codes, classes, regime, warm + years * 252, 7000 + i)
            ms = E.Market(si, codes, names)
            strat = RB.model_strategy(si, codes, ms, latest=True)
            start, end = si[codes[0]]["days"][warm], si[codes[0]]["days"][-1]
            row = {}
            for name, xf in variants(ms):
                row[name] = one(ms, strat, start, end, xf)
            rows.append(row)
        names_v = list(rows[0])
        summ = {}
        for nv in names_v:
            c = np.array([r[nv]["cagr"] for r in rows])
            d = np.array([r[nv]["mdd"] for r in rows])
            w = np.array([r[nv]["worst20"] for r in rows])
            summ[nv] = dict(cagr_p10=float(np.percentile(c, 10)), cagr_med=float(np.median(c)), cagr_p90=float(np.percentile(c, 90)), mdd_med=float(np.median(d)), mdd_p10=float(np.percentile(d, 10)), worst20_med=float(np.median(w)))
            if nv != "R1":
                dc = c - np.array([r["R1"]["cagr"] for r in rows])
                dd = d - np.array([r["R1"]["mdd"] for r in rows])
                summ[nv].update(better_cagr=int((dc > 0).sum()), better_mdd=int((dd > 0).sum()), both_better=int(((dc > 0) & (dd > 0)).sum()), dcagr_med=float(np.median(dc)), dmdd_med=float(np.median(dd)))
        res["synthetic"][regime] = dict(paths=n_paths, years=years, summary=summ)
        for nv, sm in summ.items():
            extra = f" | 수익 나아진 경로 {sm['better_cagr']}/{n_paths}, 낙폭 나아진 경로 {sm['better_mdd']}/{n_paths}, 중앙 차이 수익 {100 * sm['dcagr_med']:+.2f}%p 낙폭 {100 * sm['dmdd_med']:+.1f}%p" if nv != "R1" else ""
            print(f"[{regime}] {nv}: CAGR 중앙 {100 * sm['cagr_med']:+.2f}% (p10 {100 * sm['cagr_p10']:+.2f}%, p90 {100 * sm['cagr_p90']:+.2f}%) MDD 중앙 {100 * sm['mdd_med']:.1f}% (p10 {100 * sm['mdd_p10']:.1f}%) 최악20일 중앙 {100 * sm['worst20_med']:.1f}%{extra}", flush=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", OUT, round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
