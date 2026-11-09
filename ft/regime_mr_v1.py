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
import vswing_v1 as VS

CAPITAL = 100_000_000.0
VAULT_SHARE = 0.5
TURB = 1.5
ZT = 1.0
RUNGS = [("R0_hold", None, None, False), ("R1_mr_z1_next_open", "z1", "open", False), ("R2_mr_z1_close_auction", "z1", "close", False), ("R3_mr_z3_next_open", "z3", "open", False), ("R4_mr_z3_close_auction", "z3", "close", False), ("R5_mr_z1_close_auction_vault", "z1", "close", True), ("R6_vguard_L2_ref", "vguard", "open", False)]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def regime(A):
    A = np.asarray(A, float)
    n = len(A)
    r = np.r_[np.nan, np.diff(np.log(A))]
    s5 = np.array([np.std(r[max(1, i - 4):i + 1]) if i >= 5 else np.nan for i in range(n)])
    s60 = np.array([np.std(r[max(1, i - 59):i + 1]) if i >= 60 else np.nan for i in range(n)])
    med = np.array([np.nanmedian(s5[max(0, i - 249):i + 1]) if i >= 260 else np.nan for i in range(n)])
    turb = s5 > TURB * med
    z1 = r / s60
    z3 = np.r_[[np.nan] * 3, np.log(A[3:] / A[:-3])] / (s60 * math.sqrt(3))
    return dict(turb=turb, ready=~np.isnan(med) & ~np.isnan(s60), z1=z1, z3=z3)


def targets(rg, sig):
    n = len(rg["turb"])
    w = np.ones(n)
    cur = 1.0
    for i in range(n):
        if not rg["ready"][i]:
            w[i] = 1.0
            continue
        if not rg["turb"][i]:
            cur = 1.0
        else:
            z = rg[sig][i]
            if z < -ZT:
                cur = 1.0
            elif z > ZT:
                cur = 0.0
        w[i] = cur
    return w


def run(w, O, C, start, end, cost, at_close, vault_on):
    cash, q, b, vault = CAPITAL, 0.0, 0.0, 0.0
    vals, peak, mdd, low, trades = [], CAPITAL, 0.0, CAPITAL, 0
    for t in range(start, end):
        fi = t if at_close else t + 1
        P = C[t] if at_close else O[t + 1]
        acc = cash + q * C[t]
        want_q = w[t] * acc / C[t]
        dq = want_q - q
        if abs(dq * C[t]) >= 0.01 * acc or (w[t] == 0 and q > 0):
            c = cost(fi, P)
            if dq < 0:
                sq = min(q, -dq)
                proceeds = sq * P * (1 - c)
                part = b * sq / q
                profit = proceeds - part
                tv = VAULT_SHARE * profit if (vault_on and profit > 0) else 0.0
                vault += tv
                cash += proceeds - tv
                q -= sq
                b -= part
                trades += 1
            elif dq > 0 and cash > 0:
                spend = min(cash, dq * P * (1 + c))
                q += spend / (P * (1 + c))
                cash -= spend
                b += spend
                trades += 1
        tot = cash + q * C[t + 1] + vault
        vals.append(tot)
        peak = max(peak, tot)
        mdd = min(mdd, tot / peak - 1)
        low = min(low, tot)
    v = np.asarray(vals)
    return v, dict(total_pct=100 * (v[-1] / CAPITAL - 1), cagr_pct=100 * ((v[-1] / CAPITAL) ** (252 / len(v)) - 1), mdd_pct=100 * mdd, lowest_pct=100 * (low / CAPITAL - 1), vault_pct=100 * vault / CAPITAL, trades=trades, mean_exposure=float(np.mean(w[start:end])))


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, pth in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), pth)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    raw = json.load(open(os.path.join(os.path.dirname(HERE), "data", "kr_etf", "kodex_daily_raw.json"), encoding="utf-8"))["069500"]
    adj = json.load(open(CT.ADJ, encoding="utf-8"))["069500"]
    days = np.array([d for d in raw["days"] if d <= M1.END])
    n = len(days)
    O, C = np.asarray(raw["O"][:n], float), np.asarray(raw["C"][:n], float)
    A = np.asarray(adj["C"][:n], float)
    rg = regime(A)
    vg, _ = VS.targets(VS.features(A), "L2_trend_crashguard")
    cost = lambda i, P: LI.FEE + LI.TICK / P
    W = {}
    for name, sig, fill, vault in RUNGS:
        W[name] = np.ones(n) if sig is None else (vg if sig == "vguard" else targets(rg, sig))
    windows = {"2003-10..2012-12": ("2003-10-01", "2012-12-28"), "2013-01..2026-11": ("2013-01-02", M1.END), "2026 only": ("2026-01-02", M1.END), "2003-10..2026-11 full": ("2003-10-01", M1.END)}
    res = {}
    for wn, (a, b) in windows.items():
        st = int(np.argmax(days >= a)) - 1
        en = n - 1 if b >= days[-1] else int(np.argmax(days > b)) - 1
        res[wn] = {}
        for name, sig, fill, vault in RUNGS:
            v, r = run(W[name], O, C, st, en, cost, fill == "close", vault)
            r["years"] = VS.yearly(v, days, st)
            r["losing_years"] = [y for y, x in r["years"].items() if x < 0]
            res[wn][name] = r
            print(f"{wn:22s} {name:30s} total {r['total_pct']:8.1f}% cagr {r['cagr_pct']:6.2f}% mdd {r['mdd_pct']:6.1f} lost {len(r['losing_years'])} vault {r['vault_pct']:.1f}% expo {r['mean_exposure']:.2f} trades {r['trades']}", flush=True)
    syn = {}
    for trend in SC.TRENDS:
        for case in SC.LIQ_CASES:
            for seed in range(SC.SEEDS):
                Os, Cs, liq = SC.generate(trend, case, seed)
                Os, Cs = np.asarray(Os, float), np.asarray(Cs, float)
                rgs = regime(Cs)
                cst = lambda i, P: LI.FEE + liq[i]["cost_bp"] / 10000.0
                for name, sig, fill, vault in RUNGS:
                    if sig == "vguard":
                        continue
                    ws = np.ones(len(Cs)) if sig is None else targets(rgs, sig)
                    _, r = run(ws, Os, Cs, SC.WARM, len(Cs) - 1, cst, fill == "close", vault)
                    syn.setdefault(trend, {}).setdefault(name, []).append(r["total_pct"])
    ss = {t: {nm: dict(median_total_pct=float(np.median(v)), beat_hold=int(sum(x > h for x, h in zip(v, syn[t]["R0_hold"])))) for nm, v in d.items()} for t, d in syn.items()}
    for t, d in ss.items():
        print("SYN", t, {nm: (round(x["median_total_pct"], 1), x["beat_hold"]) for nm, x in d.items()}, flush=True)
    rec = dict(version="regime-mr-v1", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], results=res, synthetic_summary=ss)
    with open(out_path, "x", encoding="utf-8") as fo:
        json.dump(rec, fo, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
