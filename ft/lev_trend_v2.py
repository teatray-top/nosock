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
import learn_inverse as LI
import scenario_test_v1 as SC
import lev_trend_v1b as T
import voltime_v1 as VT

CAPITAL = T.CAPITAL
VOL_TARGET = 0.35
STRONG = 0.3
HYST = 0.1
EBAND = 0.25


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def exposures(eq, vol_ann):
    n = len(eq)
    E = {}
    base = np.where(~np.isnan(eq) & (eq > 0), 2.0, 0.0)
    E["S4_eqsignal_2x"] = base
    vt25 = np.where(np.isnan(vol_ann), 2.0, np.minimum(2.0, 0.25 / np.maximum(vol_ann, 1e-6)))
    vt = np.where(np.isnan(vol_ann), 2.0, np.minimum(2.0, VOL_TARGET / np.maximum(vol_ann, 1e-6)))
    E["V1a_vol_target25"] = np.where(base > 0, vt25, 0.0)
    E["V1b_vol_target35"] = np.where(base > 0, vt, 0.0)
    E["V2_strength"] = np.where(~np.isnan(eq) & (eq > STRONG), 2.0, np.where(~np.isnan(eq) & (eq > 0), 1.0, 0.0))
    E["V3_vol35_and_strength"] = np.minimum(E["V1b_vol_target35"], E["V2_strength"])
    h = np.zeros(n)
    state = 0.0
    for i in range(n):
        if not np.isnan(eq[i]):
            if eq[i] > HYST:
                state = 2.0
            elif eq[i] < -HYST:
                state = 0.0
        h[i] = state
    E["V4_hysteresis_2x"] = h
    E["V5_hysteresis_vol35"] = np.where(h > 0, vt, 0.0)
    return E


def run_expo(E, PO, PC, O1, C1, start, end, cost2, cost1):
    cash, q2 = CAPITAL, 0.0
    peak, mdd, low = CAPITAL, 0.0, CAPITAL
    vals, cur, switches = [], 0.0, 0
    for t in range(start, end):
        e = t + 1
        tot_t = cash + q2 * PC[t]
        want = float(E[t])
        if abs(want - cur) > EBAND or (want == 0 and q2 > 0):
            target_val = tot_t * min(want, 2.0) / 2.0
            have = q2 * PO[e]
            if target_val > have:
                spend = min(cash, (target_val - have) * (1 + cost2(e)))
                q2 += spend / (PO[e] * (1 + cost2(e)))
                cash -= spend
            else:
                sq = min(q2, (have - target_val) / PO[e])
                cash += sq * PO[e] * (1 - cost2(e))
                q2 -= sq
            cur = want
            switches += 1
        tot = cash + q2 * PC[e]
        vals.append(tot)
        peak = max(peak, tot)
        mdd = min(mdd, tot / peak - 1)
        low = min(low, tot)
    v = np.asarray(vals)
    return v, dict(total_pct=100 * (v[-1] / CAPITAL - 1), cagr_pct=100 * ((v[-1] / CAPITAL) ** (252 / len(v)) - 1), mdd_pct=100 * mdd, lowest_pct=100 * (low / CAPITAL - 1), rebalances=switches, mean_exposure=float(np.mean(E[start:end])))


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, pth in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), pth)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    days, O, C, AO, A, LO, LC = T.load()
    drag, cal = T.calibrate(days, A, AO, LC)
    PO, PC, splice = T.spliced(days, AO, A, LO, LC, drag)
    eq, sma = T.signals_069500(days, A)
    vol = VT.ewma_vol(list(A)) * math.sqrt(252)
    E = exposures(eq, vol)
    E["S0_hold_1x"] = np.ones(len(days))
    cost2 = lambda e: LI.FEE + LI.TICK / PO[e]
    windows = {"2003-10..2012-12 (out of period)": ("2003-10-01", "2012-12-28"), "2013-01..2026-11": ("2013-01-02", T.M1.END), "2003-10..2026-11 (full)": ("2003-10-01", T.M1.END)}
    res = {}
    for wn, (a, b) in windows.items():
        st = int(np.argmax(days >= a)) - 1
        en = len(days) - 1 if b >= days[-1] else int(np.argmax(days > b)) - 1
        res[wn] = {}
        for nm, ex in E.items():
            v, r = run_expo(ex, PO, PC, O, C, st, en, cost2, None) if nm != "S0_hold_1x" else T.run(np.ones(len(days), bool), O, C, st, en, lambda e: LI.FEE + LI.TICK / O[e])
            r["years"] = T.yearly(v, days, st)
            r["worst_year_pct"] = min(r["years"].values())
            r["losing_years"] = [y for y, x in r["years"].items() if x < 0]
            pk = np.maximum.accumulate(v)
            dd = v / pk - 1
            tr = int(np.argmin(dd))
            p0 = max(k for k in range(tr + 1) if v[k] == pk[tr])
            r["mdd_window"] = [str(days[st + 1 + p0]), str(days[st + 1 + tr])]
            res[wn][nm] = r
            print(f"{wn[:18]:18s} {nm:26s} total {r['total_pct']:9.1f}% cagr {r['cagr_pct']:6.2f}% mdd {r['mdd_pct']:6.1f} ({r['mdd_window'][0]}..{r['mdd_window'][1]}) worstyr {r['worst_year_pct']:6.1f} lost {len(r['losing_years'])} expo {r.get('mean_exposure', 1.0):.2f}", flush=True)
    syn = {}
    for trend in SC.TRENDS:
        for case in SC.LIQ_CASES:
            for seed in range(SC.SEEDS):
                Os, Cs, liq = SC.generate(trend, case, seed)
                Os, Cs = np.asarray(Os, float), np.asarray(Cs, float)
                so, sc = T.synth_2x(Os, Cs, drag)
                eqs, _ = T.signals_069500(np.arange(len(Cs)), Cs)
                vs = VT.ewma_vol(list(Cs)) * math.sqrt(252)
                Es = exposures(eqs, vs)
                cst = lambda e: LI.FEE + liq[e]["cost_bp"] / 10000.0
                for nm, ex in Es.items():
                    _, r = run_expo(ex, so, sc, Os, Cs, T.SC.WARM, len(Cs) - 1, cst, None)
                    syn.setdefault(trend, {}).setdefault(nm, []).append((r["total_pct"], r["mdd_pct"]))
                _, r = T.run(np.ones(len(Cs), bool), Os, Cs, T.SC.WARM, len(Cs) - 1, cst)
                syn.setdefault(trend, {}).setdefault("hold_1x", []).append((r["total_pct"], r["mdd_pct"]))
    ss = {}
    for t, d in syn.items():
        ss[t] = {nm: dict(median_total_pct=float(np.median([x[0] for x in v])), median_mdd_pct=float(np.median([x[1] for x in v])), worst_total_pct=float(min(x[0] for x in v))) for nm, v in d.items()}
        print("SYN", t, {nm: (round(x["median_total_pct"], 1), round(x["median_mdd_pct"], 1)) for nm, x in ss[t].items()}, flush=True)
    rec = dict(version="lev-trend-v2", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], results=res, synthetic_summary=ss)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
