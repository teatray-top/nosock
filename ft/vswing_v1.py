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
import voltime_v1 as VT
import harvest_v8 as H8
import turning_learn_v1 as TL

CAPITAL = 100_000_000.0
VAULT_SHARE = 0.5
Z_HI, Z_LO = 2.0, 0.0
SPIKE = 1.5
PEAK_DROP = 0.85
INV_EQ = -0.2
LADDER = ["L0_hold", "L1_trend", "L2_trend_crashguard", "L3_plus_swing", "L4_plus_vault", "L5_plus_inverse"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def ema(x, n):
    a = 2.0 / (n + 1)
    out = np.empty(len(x))
    v = x[0]
    for i, z in enumerate(x):
        v = z if i == 0 else a * z + (1 - a) * v
        out[i] = v
    return out


def features(A):
    A = np.asarray(A, float)
    S, _, _ = M1.signals(np.arange(len(A)), A[:, None])
    eq = np.nanmean(np.tanh(np.stack([S[:, :, M1.SIG.index(s)] for s in M1.EQ_SIGNALS], axis=2)), axis=2)[:, 0]
    sig = VT.ewma_vol(list(A))
    med = VT.expanding_median(sig)
    e20, e50 = ema(A, 20), ema(A, 50)
    sma20 = np.array([A[max(0, i - 19):i + 1].mean() for i in range(len(A))])
    sd20 = np.array([A[max(0, i - 19):i + 1].std() for i in range(len(A))])
    z = (A - sma20) / np.where(sd20 > 0, sd20, np.nan)
    peak10 = np.array([np.nanmax(sig[max(0, i - 10):i + 1]) if not np.all(np.isnan(sig[max(0, i - 10):i + 1])) else np.nan for i in range(len(A))])
    return dict(eq=eq, sig=sig, med=med, z=z, below50=A < e50, spike=sig > SPIKE * med, peaked=sig < PEAK_DROP * peak10)


def targets(f, rung):
    n = len(f["eq"])
    w = np.zeros(n)
    leg = np.array(["L"] * n, dtype=object)
    guard = False
    cur = 0.0
    for i in range(n):
        eq = f["eq"][i]
        if np.isnan(eq) or np.isnan(f["sig"][i]):
            w[i] = 1.0 if rung == "L0_hold" else 0.0
            continue
        if rung == "L0_hold":
            w[i] = 1.0
            continue
        up = eq > 0
        if rung == "L1_trend":
            w[i] = 1.0 if up else 0.0
            continue
        if not np.isnan(f["med"][i]) and f["spike"][i] and f["below50"][i]:
            guard = True
        if guard:
            if f["peaked"][i] and up:
                guard = False
            else:
                w[i] = 0.0
                cur = 0.0
                continue
        if not up:
            cur = 0.0
            if rung == "L5_plus_inverse" and eq < INV_EQ and not f["spike"][i]:
                w[i] = 1.0
                leg[i] = "I"
            else:
                w[i] = 0.0
            continue
        if rung == "L2_trend_crashguard":
            w[i] = 1.0
            continue
        if cur == 0.0:
            cur = 1.0
        if f["z"][i] > Z_HI:
            cur = 0.5
        elif f["z"][i] < Z_LO:
            cur = 1.0
        w[i] = cur
    return w, leg


def run(w, leg, O, C, IO, IC, start, end, cost, vault_on):
    cash, qL, qI, bL, bI, vault = CAPITAL, 0.0, 0.0, 0.0, 0.0, 0.0
    vals, peak, mdd, low, trades = [], CAPITAL, 0.0, CAPITAL, 0
    for t in range(start, end):
        e = t + 1
        acc = cash + qL * C[t] + qI * IC[t]
        wantL = w[t] if leg[t] == "L" else 0.0
        wantI = w[t] if leg[t] == "I" else 0.0
        for side in ("sell", "buy"):
            for nm in ("L", "I"):
                P, q, b = (O, qL, bL) if nm == "L" else (IO, qI, bI)
                want = wantL if nm == "L" else wantI
                Cx = C[t] if nm == "L" else IC[t]
                tgt_q = want * acc / Cx
                dq = tgt_q - q
                if abs(dq * Cx) < 0.01 * acc and not (want == 0 and q > 0):
                    continue
                c = cost(nm, e)
                if side == "sell" and dq < 0:
                    sq = min(q, -dq)
                    proceeds = sq * P[e] * (1 - c)
                    part = b * sq / q if q > 0 else 0.0
                    profit = proceeds - part
                    tv = VAULT_SHARE * profit if (vault_on and profit > 0) else 0.0
                    vault += tv
                    cash += proceeds - tv
                    q -= sq
                    b -= part
                    trades += 1
                elif side == "buy" and dq > 0:
                    spend = min(cash, dq * P[e] * (1 + c))
                    if spend <= 0:
                        continue
                    q += spend / (P[e] * (1 + c))
                    cash -= spend
                    b += spend
                    trades += 1
                if nm == "L":
                    qL, bL = q, b
                else:
                    qI, bI = q, b
        tot = cash + qL * C[e] + qI * IC[e] + vault
        vals.append(tot)
        peak = max(peak, tot)
        mdd = min(mdd, tot / peak - 1)
        low = min(low, tot)
    v = np.asarray(vals)
    return v, dict(total_pct=100 * (v[-1] / CAPITAL - 1), cagr_pct=100 * ((v[-1] / CAPITAL) ** (252 / len(v)) - 1), mdd_pct=100 * mdd, lowest_pct=100 * (low / CAPITAL - 1), vault_pct=100 * vault / CAPITAL, trades=trades, mean_exposure=float(np.mean(w[start:end])))


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
    raw = json.load(open(os.path.join(os.path.dirname(HERE), "data", "kr_etf", "kodex_daily_raw.json"), encoding="utf-8"))["069500"]
    adj = json.load(open(CT.ADJ, encoding="utf-8"))["069500"]
    days = np.array([d for d in raw["days"] if d <= M1.END])
    n = len(days)
    O, C = np.asarray(raw["O"][:n], float), np.asarray(raw["C"][:n], float)
    A = np.asarray(adj["C"][:n], float)
    IO, IC, splice = TL.spliced_inverse(list(days), dict(O=list(O), C=list(C)), 0, n - 1)
    IO, IC = np.asarray(IO), np.asarray(IC)
    f = features(A)
    cost = lambda nm, e: LI.FEE + (LI.TICK / O[e] if nm == "L" else H8.tick(days[e], IO[e]) / IO[e])
    windows = {"2003-10..2012-12 (out of period)": ("2003-10-01", "2012-12-28"), "2013-01..2026-11": ("2013-01-02", M1.END), "2026 only (volatile bull)": ("2026-01-02", M1.END), "2003-10..2026-11 (full)": ("2003-10-01", M1.END)}
    res = {}
    for wn, (a, b) in windows.items():
        st = int(np.argmax(days >= a)) - 1
        en = n - 1 if b >= days[-1] else int(np.argmax(days > b)) - 1
        res[wn] = {}
        for rung in LADDER:
            w, leg = targets(f, rung if rung != "L4_plus_vault" else "L3_plus_swing")
            v, r = run(w, leg, O, C, IO, IC, st, en, cost, rung in ("L4_plus_vault", "L5_plus_inverse"))
            r["years"] = yearly(v, days, st)
            r["losing_years"] = [y for y, x in r["years"].items() if x < 0]
            pk = np.maximum.accumulate(v)
            dd = v / pk - 1
            tr = int(np.argmin(dd))
            p0 = max(k for k in range(tr + 1) if v[k] == pk[tr])
            r["mdd_window"] = [str(days[st + 1 + p0]), str(days[st + 1 + tr])]
            res[wn][rung] = r
            print(f"{wn[:20]:20s} {rung:22s} total {r['total_pct']:8.1f}% cagr {r['cagr_pct']:6.2f}% mdd {r['mdd_pct']:6.1f} ({r['mdd_window'][0]}..{r['mdd_window'][1]}) lost {len(r['losing_years'])} vault {r['vault_pct']:.1f}% expo {r['mean_exposure']:.2f} trades {r['trades']}", flush=True)
    syn = {}
    for trend in SC.TRENDS:
        for case in SC.LIQ_CASES:
            for seed in range(SC.SEEDS):
                Os, Cs, liq = SC.generate(trend, case, seed)
                Os, Cs = np.asarray(Os, float), np.asarray(Cs, float)
                IOs, ICs = SC.inverse_path(Os, Cs)
                fs = features(Cs)
                cst = lambda nm, e: LI.FEE + liq[e]["cost_bp"] / 10000.0
                for rung in LADDER:
                    w, leg = targets(fs, rung if rung != "L4_plus_vault" else "L3_plus_swing")
                    _, r = run(w, leg, Os, Cs, IOs, ICs, SC.WARM, len(Cs) - 1, cst, rung in ("L4_plus_vault", "L5_plus_inverse"))
                    syn.setdefault(trend, {}).setdefault(rung, []).append((r["total_pct"], r["mdd_pct"]))
    ss = {}
    for t in list(SC.TRENDS) + ["all"]:
        rows = {rung: sum((syn[x][rung] for x in SC.TRENDS), []) if t == "all" else syn[t][rung] for rung in LADDER}
        ss[t] = {rung: dict(median_total_pct=float(np.median([x[0] for x in v])), median_mdd_pct=float(np.median([x[1] for x in v])), worst_total_pct=float(min(x[0] for x in v)), beat_hold=int(sum(a[0] > b[0] for a, b in zip(v, rows["L0_hold"])))) for rung, v in rows.items()}
        print("SYN", t, {r_: (round(x["median_total_pct"], 1), round(x["median_mdd_pct"], 1), x["beat_hold"]) for r_, x in ss[t].items()}, flush=True)
    rec = dict(version="vswing-v1", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], splice_day=splice, results=res, synthetic_summary=ss)
    with open(out_path, "x", encoding="utf-8") as fo:
        json.dump(rec, fo, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
