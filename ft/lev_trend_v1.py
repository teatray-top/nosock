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
import compare_trend_v2 as CT
import scenario_test_v1 as SC
import msma_v1 as M1

LEV_DIR = os.path.join(os.path.dirname(HERE), "data", "kr_etf", "yf_lev_20261108_000349")
CAPITAL = 100_000_000.0
VAULT_SHARE, SKIM_G = 0.5, 0.10
CAL_FROM, CAL_TO = "2010-02-23", "2012-12-28"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load():
    raw = json.load(open(os.path.join(os.path.dirname(HERE), "data", "kr_etf", "kodex_daily_raw.json"), encoding="utf-8"))["069500"]
    adj = json.load(open(CT.ADJ, encoding="utf-8"))["069500"]
    days = [d for d in raw["days"] if d <= M1.END]
    n = len(days)
    O = np.asarray(raw["O"][:n], float)
    C = np.asarray(raw["C"][:n], float)
    A = np.asarray(adj["C"][:n], float)
    AO = np.asarray(adj["O"][:n], float)
    man = json.load(open(os.path.join(LEV_DIR, "manifest.json"), encoding="utf-8"))
    info = man["files"]["122630_adj"]
    p = os.path.join(LEV_DIR, info["file"])
    if sha(p) != info["sha256"]:
        raise SystemExit("122630 hash differs")
    lv = json.load(open(p, encoding="utf-8"))
    lm = {d: (o, c) for d, o, c in zip(lv["days"], lv["O"], lv["C"])}
    LO = np.array([lm[d][0] if d in lm else np.nan for d in days])
    LC = np.array([lm[d][1] if d in lm else np.nan for d in days])
    return np.array(days), O, C, AO, A, LO, LC


def synth_2x(AO, A, drag):
    n = len(A)
    so, sc = np.empty(n), np.empty(n)
    so[0] = sc[0] = 10000.0
    for t in range(1, n):
        so[t] = sc[t - 1] * (1 + 2 * (AO[t] / A[t - 1] - 1))
        sc[t] = sc[t - 1] * (1 + 2 * (A[t] / A[t - 1] - 1) - drag)
    return so, sc


def calibrate(days, A, AO, LC):
    m = (days >= CAL_FROM) & (days <= CAL_TO) & ~np.isnan(LC)
    idx = np.nonzero(m)[0]
    real = np.log(LC[idx[-1]] / LC[idx[0]])
    best, bd = None, 0.0
    for d in np.linspace(0, 0.0004, 401):
        _, sc = synth_2x(AO, A, d)
        e = abs(np.log(sc[idx[-1]] / sc[idx[0]]) - real)
        if best is None or e < best:
            best, bd = e, d
    rr = np.diff(np.log(LC[idx]))
    _, sc = synth_2x(AO, A, bd)
    rs = np.diff(np.log(sc[idx]))
    return bd, dict(daily_drag=bd, annual_drag_pct=100 * bd * 252, corr_daily=float(np.corrcoef(rr, rs)[0, 1]), window=[CAL_FROM, CAL_TO], residual_log=float(best))


def spliced(days, AO, A, LO, LC, drag):
    so, sc = synth_2x(AO, A, drag)
    first = int(np.argmax(~np.isnan(LC)))
    k = LC[first] / sc[first]
    PO = np.where(np.arange(len(days)) < first, so * k, LO)
    PC = np.where(np.arange(len(days)) < first, sc * k, LC)
    return PO, PC, days[first]


def signals_069500(days, A):
    S, sd60, r = M1.signals(days, A[:, None])
    eq = np.nanmean(np.tanh(np.stack([S[:, :, M1.SIG.index(s)] for s in M1.EQ_SIGNALS], axis=2)), axis=2)[:, 0]
    sma = np.full(len(A), np.nan)
    cs = np.cumsum(np.insert(A, 0, 0.0))
    sma[199:] = (cs[200:] - cs[:-200]) / 200.0
    return eq, sma


def run(on, PO, PC, start, end, cost, vault_on=False):
    cash, q, vault, basis = CAPITAL, 0.0, 0.0, None
    peak, mdd, low, trades = CAPITAL, 0.0, CAPITAL, 0
    vals = []
    for t in range(start, end):
        want = bool(on[t])
        e = t + 1
        if want and q == 0 and cash > 0:
            c = cost(e)
            q = cash * (1 - c) / PO[e]
            basis = cash
            cash = 0.0
            trades += 1
        elif not want and q > 0:
            c = cost(e)
            proceeds = q * PO[e] * (1 - c)
            profit = proceeds - basis
            tv = VAULT_SHARE * profit if (vault_on and profit > 0) else 0.0
            vault += tv
            cash += proceeds - tv
            q = 0.0
        elif vault_on and q > 0 and q * PC[t] >= basis * (1 + SKIM_G):
            c = cost(e)
            gain = q * PC[t] - basis
            sq = VAULT_SHARE * gain / PC[t]
            vault += sq * PO[e] * (1 - c)
            q -= sq
            basis = q * PO[e]
        tot = cash + q * PC[e] + vault
        vals.append(tot)
        peak = max(peak, tot)
        mdd = min(mdd, tot / peak - 1)
        low = min(low, tot)
    v = np.asarray(vals)
    return v, dict(total_pct=100 * (v[-1] / CAPITAL - 1), cagr_pct=100 * ((v[-1] / CAPITAL) ** (252 / len(v)) - 1), mdd_pct=100 * mdd, lowest_pct=100 * (low / CAPITAL - 1), vault_pct=100 * vault / CAPITAL, entries=trades, time_in_market=float(np.mean(on[start:end])))


def yearly(v, days, start):
    dd = days[start + 1:start + 1 + len(v)]
    out, prev = {}, CAPITAL
    for y in sorted(set(x[:4] for x in dd)):
        last = max(k for k, x in enumerate(dd) if x[:4] == y)
        out[y] = 100 * (v[last] / prev - 1)
        prev = v[last]
    return out


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, pth in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), pth)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    days, O, C, AO, A, LO, LC = load()
    drag, cal = calibrate(days, A, AO, LC)
    print("calibration", cal, flush=True)
    PO2, PC2, splice = spliced(days, AO, A, LO, LC, drag)
    eq, sma = signals_069500(days, A)
    above = ~np.isnan(sma) & (A > sma)
    pos_eq = ~np.isnan(eq) & (eq > 0)
    cost1 = lambda e: LI.FEE + LI.TICK / O[e]
    cost2 = lambda e: LI.FEE + LI.TICK / PO2[e]
    allon = np.ones(len(days), bool)
    strategies = {"S0_hold_1x": (allon, "1x"), "S1_hold_2x": (allon, "2x"), "S2_sma200_2x": (above, "2x"), "S3_sma200_1x": (above, "1x"), "S4_eqsignal_2x": (pos_eq, "2x"), "S5_eqsignal_1x": (pos_eq, "1x")}
    windows = {"2003-10..2012-12 (synthetic 2x, out of period)": ("2003-10-01", "2012-12-28"), "2013-01..2026-11 (real 122630)": ("2013-01-02", M1.END), "2003-10..2026-11 (full, spliced)": ("2003-10-01", M1.END)}
    res = {}
    for wn, (a, b) in windows.items():
        st = int(np.argmax(days >= a)) - 1
        en = len(days) - 1 if b >= days[-1] else int(np.argmax(days > b)) - 1
        res[wn] = {}
        for nm, (on, lev) in strategies.items():
            for vault_on in (False, True):
                if vault_on and nm not in ("S2_sma200_2x", "S4_eqsignal_2x", "S0_hold_1x"):
                    continue
                PO_, PC_, cst = (O, C, cost1) if lev == "1x" else (PO2, PC2, cost2)
                v, r = run(on, PO_, PC_, st, en, cst, vault_on)
                r["years"] = yearly(v, days, st)
                r["worst_year_pct"] = min(r["years"].values())
                r["losing_years"] = [y for y, x in r["years"].items() if x < 0]
                key = nm + ("_vault" if vault_on else "")
                res[wn][key] = r
                print(f"{wn[:22]:22s} {key:22s} total {r['total_pct']:9.1f}% cagr {r['cagr_pct']:6.2f}% mdd {r['mdd_pct']:6.1f} lowest {r['lowest_pct']:6.1f} worstyr {r['worst_year_pct']:6.1f} lost {len(r['losing_years'])} vault {r['vault_pct']:.1f}% in-mkt {r['time_in_market']:.2f} entries {r['entries']}", flush=True)
    syn = []
    for trend in SC.TRENDS:
        for case in SC.LIQ_CASES:
            for seed in range(SC.SEEDS):
                Os, Cs, liq = SC.generate(trend, case, seed)
                Os, Cs = np.asarray(Os, float), np.asarray(Cs, float)
                so, sc = synth_2x(Os, Cs, drag)
                sm = np.full(len(Cs), np.nan)
                cs_ = np.cumsum(np.insert(Cs, 0, 0.0))
                sm[199:] = (cs_[200:] - cs_[:-200]) / 200.0
                ab = ~np.isnan(sm) & (Cs > sm)
                cst = lambda e: LI.FEE + liq[e]["cost_bp"] / 10000.0
                row = dict(trend=trend)
                for nm, (on, PO_, PC_) in {"hold_1x": (np.ones(len(Cs), bool), Os, Cs), "hold_2x": (np.ones(len(Cs), bool), so, sc), "sma200_2x": (ab, so, sc), "sma200_1x": (ab, Os, Cs)}.items():
                    _, r = run(on, PO_, PC_, SC.WARM, len(Cs) - 1, cst)
                    row[nm] = dict(total_pct=r["total_pct"], mdd_pct=r["mdd_pct"])
                syn.append(row)
    ss = {}
    for t in list(SC.TRENDS) + ["all"]:
        ps = [x for x in syn if t == "all" or x["trend"] == t]
        ss[t] = {nm: dict(median_total_pct=float(np.median([x[nm]["total_pct"] for x in ps])), median_mdd_pct=float(np.median([x[nm]["mdd_pct"] for x in ps])), worst_total_pct=float(min(x[nm]["total_pct"] for x in ps)), beat_hold_1x=int(sum(x[nm]["total_pct"] > x["hold_1x"]["total_pct"] for x in ps)), paths=len(ps)) for nm in ("hold_1x", "hold_2x", "sma200_2x", "sma200_1x")}
        print("SYN", t, {nm: (round(v["median_total_pct"], 1), round(v["median_mdd_pct"], 1), round(v["worst_total_pct"], 1), v["beat_hold_1x"]) for nm, v in ss[t].items()}, flush=True)
    rec = dict(version="lev-trend-v1", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], calibration=cal, splice_day=splice, results=res, synthetic_summary=ss)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
