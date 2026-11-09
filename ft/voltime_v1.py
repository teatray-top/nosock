import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
import math
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import learn_inverse as LI
import compare_trend_v2 as CT
import scenario_test_v1 as SC
import harvest_v5 as H5
import turning_points_v1 as TP

CAPITAL = H5.CAPITAL
VAULT_SHARE = H5.VAULT_SHARE
LONG = H5.LONG
LAMBDA = 0.94
WBAND = 0.10
SKIM_G = 0.10


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def policies():
    return [dict(name="R0_hold", kind="skim"), dict(name="R1_skim10", kind="skim", g=SKIM_G), dict(name="V1_vol_x1.0", kind="vol", k=1.0), dict(name="V1_vol_x0.75", kind="vol", k=0.75)]


def ewma_vol(C):
    r = np.diff(np.log(np.asarray(C, float)), prepend=np.nan)
    v = np.full(len(C), np.nan)
    s2 = None
    for i in range(1, len(C)):
        s2 = r[i] ** 2 if s2 is None else LAMBDA * s2 + (1 - LAMBDA) * r[i] ** 2
        if i >= 20:
            v[i] = math.sqrt(s2)
    return v


def expanding_median(v):
    out = np.full(len(v), np.nan)
    seen = []
    import bisect
    for i, x in enumerate(v):
        if not np.isnan(x):
            bisect.insort(seen, x)
        if len(seen) >= 250:
            m = len(seen)
            out[i] = seen[m // 2] if m % 2 else 0.5 * (seen[m // 2 - 1] + seen[m // 2])
    return out


def run_weight(px, cost, start, n, w_at, keep=False):
    P = px[LONG]
    cash, vault, q, B = CAPITAL, 0.0, 0.0, 0.0
    sales, buys, cost_paid, trades = 0, 0, 0.0, []
    series = {start: CAPITAL}
    peak, mdd, low = CAPITAL, 0.0, CAPITAL
    wsum, wn = 0.0, 0
    for i in range(start, n - 1):
        j = i + 1
        c_i, o, c = P["C"][i], P["O"][j], cost(LONG, j)
        risky = q * c_i
        if q > 0 and risky - B >= SKIM_G * B:
            f = VAULT_SHARE * (risky - B) / risky
            sq = f * q
            proceeds = sq * o * (1 - c)
            cost_paid += sq * o * c
            vault += proceeds
            q -= sq
            B = q * o
            sales += 1
            if keep:
                trades.append(dict(side="skim", decision_index=i, fill_index=j, price=o, qty=sq, to_vault=proceeds))
        else:
            acc = cash + risky
            w = w_at(i)
            cur = risky / acc if acc > 0 else 0.0
            if abs(w - cur) > WBAND or (w >= 0.999 and cur < 0.999 and cash > 0.01 * acc and abs(w - cur) > 0.02):
                dq = w * acc / c_i - q
                if dq > 0:
                    spend = min(cash, dq * o * (1 + c))
                    if spend > 0:
                        bq = spend / (o * (1 + c))
                        q += bq
                        cash -= spend
                        B += spend
                        cost_paid += bq * o * c
                        buys += 1
                        if keep:
                            trades.append(dict(side="buy", decision_index=i, fill_index=j, price=o, qty=bq, spend=spend, w=w))
                elif dq < 0 and q > 0:
                    sq = min(q, -dq)
                    proceeds = sq * o * (1 - c)
                    cost_paid += sq * o * c
                    part = B * sq / q
                    profit = proceeds - part
                    tv = VAULT_SHARE * profit if profit > 0 else 0.0
                    vault += tv
                    cash += proceeds - tv
                    B -= part
                    q -= sq
                    sales += 1
                    if keep:
                        trades.append(dict(side="sell", decision_index=i, fill_index=j, price=o, qty=sq, proceeds=proceeds, profit=profit, to_vault=tv, w=w))
        tot = cash + q * P["C"][j] + vault
        acc_j = cash + q * P["C"][j]
        wsum += q * P["C"][j] / acc_j if acc_j > 0 else 0.0
        wn += 1
        series[j] = tot
        peak = max(peak, tot)
        mdd = min(mdd, tot / peak - 1)
        low = min(low, tot)
    c = cost(LONG, n - 1)
    cost_paid += q * P["C"][n - 1] * c
    eq_end = cash + q * P["C"][n - 1] * (1 - c)
    final = eq_end + vault
    mdd = min(mdd, final / peak - 1)
    low = min(low, final)
    series[n - 1] = final
    return dict(vault_pct=100 * vault / CAPITAL, equity_pct=100 * eq_end / CAPITAL, total_pct=100 * (final / CAPITAL - 1), mdd_total=100 * mdd, lowest_total_pct=100 * (low / CAPITAL - 1), mean_weight=wsum / max(wn, 1), sales=sales, buys=buys, cost_pct=100 * cost_paid / CAPITAL), series, trades


def run_pol(p, px, cost, start, n, vol, med, keep=False):
    if p["kind"] == "vol":
        w_at = lambda i: 1.0 if np.isnan(vol[i]) or np.isnan(med[i]) else min(1.0, p["k"] * med[i] / vol[i])
        return run_weight(px, cost, start, n, w_at, keep)
    r, s, t, _ = H5.run_skim(p, px, cost, start, n, lambda i: False, lambda i: (None, None), keep)
    vals = [s[k] for k in sorted(s)]
    return dict(r, lowest_total_pct=100 * (min(vals) / CAPITAL - 1), mean_weight=None, buys=1), s, t


def real_eval(pols):
    raw = json.load(open(os.path.join(os.path.dirname(HERE), "data", "kr_etf", "kodex_daily_raw.json"), encoding="utf-8"))[LONG]
    adj = json.load(open(CT.ADJ, encoding="utf-8"))[LONG]
    days = [d for d in raw["days"] if d <= H5.H2.REAL_TO]
    A = [float(x) for x in adj["C"][:len(days)]]
    vol_all = ewma_vol(A)
    med_all = expanding_median(vol_all)
    sub, px, cost, bull, bear, brk, pred, Ak = H5.load_real()
    off = days.index(sub[0])
    vol, med = vol_all[off:], med_all[off:]
    res, series, trades, years = {}, {}, {}, {}
    for p in pols:
        r, s, t = run_pol(p, px, cost, 0, len(sub), vol, med, keep=True)
        res[p["name"]], series[p["name"]] = r, s
        trades[p["name"]] = [dict(x, decision_day=sub[x["decision_index"]], fill_day=sub[x["fill_index"]]) for x in t]
        years[p["name"]] = H5.yearly(s, sub)
        res[p["name"]]["years_lost_money"] = [y for y, v in years[p["name"]].items() if v < 0]
        res[p["name"]]["worst_year_pct"] = min(years[p["name"]].values())
    pts = TP.zigzag(Ak, TP.ZIG)
    turn = []
    for kind, i in pts:
        a, b = max(1, i - TP.BEFORE), min(len(sub) - 1, i + TP.AFTER)
        row = dict(kind=kind, day=sub[i], window=[sub[a], sub[b]], market_pct=100 * (px[LONG]["C"][b] / px[LONG]["C"][a] - 1))
        for p in pols:
            row[p["name"]] = 100 * (series[p["name"]][b] / series[p["name"]][a] - 1)
        turn.append(row)
    tsum = {}
    for kind in ("peak", "trough"):
        rs = [r for r in turn if r["kind"] == kind]
        tsum[kind] = {p["name"]: dict(mean=float(np.mean([r[p["name"]] for r in rs])), worst=float(min(r[p["name"]] for r in rs)), lost=int(sum(r[p["name"]] < 0 for r in rs)), n=len(rs)) for p in pols}
    return res, trades, years, turn, tsum, dict(first_decision=sub[0], last_day=sub[-1], vol_median_at_start=float(med[0]))


def crisis2008(pols):
    raw = json.load(open(os.path.join(os.path.dirname(HERE), "data", "kr_etf", "kodex_daily_raw.json"), encoding="utf-8"))[LONG]
    adj = json.load(open(CT.ADJ, encoding="utf-8"))[LONG]
    days = raw["days"]
    A = [float(x) for x in adj["C"]]
    vol_all = ewma_vol(A)
    med_all = expanding_median(vol_all)
    a, b = days.index(next(d for d in days if d >= "2007-01-02")), days.index(next(d for d in days if d >= "2012-12-28"))
    keep = list(range(a, b + 1))
    px = {LONG: dict(O=[float(raw["O"][i]) for i in keep], C=[float(raw["C"][i]) for i in keep])}
    cost = lambda leg, j: LI.FEE + LI.TICK / px[leg]["O"][j]
    vol, med = vol_all[a:b + 1], med_all[a:b + 1]
    out = {}
    for p in pols:
        r, s, _ = run_pol(p, px, cost, 0, len(keep), vol, med)
        out[p["name"]] = dict(r, years=H5.yearly(s, [days[i] for i in keep]))
    return dict(window=[days[a], days[b]], results=out)


def synth_eval(pols):
    rows = []
    for trend in SC.TRENDS:
        for case in SC.LIQ_CASES:
            for seed in range(SC.SEEDS):
                O, C, liq = SC.generate(trend, case, seed)
                Cl = [float(x) for x in C]
                vol = ewma_vol(Cl)
                med = expanding_median(vol)
                px = {LONG: dict(O=[float(x) for x in O], C=Cl)}
                cost = lambda leg, j: LI.FEE + liq[j]["cost_bp"] / 10000.0
                res = {}
                for p in pols:
                    r, _, _ = run_pol(p, px, cost, SC.WARM, len(Cl), vol, med)
                    res[p["name"]] = r
                rows.append(dict(trend=trend, liquidity=case, seed=seed, res=res))
    return rows


def summarize(rows, pols):
    out = {}
    for t in list(SC.TRENDS) + ["all"]:
        ps = [r for r in rows if t == "all" or r["trend"] == t]
        out[t] = {}
        for p in pols:
            v = lambda k: [r["res"][p["name"]][k] for r in ps]
            out[t][p["name"]] = dict(median_total_pct=float(np.median(v("total_pct"))), worst_total_pct=float(min(v("total_pct"))), median_lowest_pct=float(np.median(v("lowest_total_pct"))), median_mdd=float(np.median(v("mdd_total"))), median_vault_pct=float(np.median(v("vault_pct"))), made_money=int(sum(x > 0 for x in v("total_pct"))), median_minus_hold=float(np.median([r["res"][p["name"]]["total_pct"] - r["res"]["R0_hold"]["total_pct"] for r in ps])), paths=len(ps))
    return out


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, p in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), p)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    pols = policies()
    real, trades, years, turn, tsum, info = real_eval(pols)
    print("REAL", info, flush=True)
    for p in pols:
        r = real[p["name"]]
        print(f"{p['name']:14s} vault {r['vault_pct']:6.1f} total {r['total_pct']:7.1f} lowest {r['lowest_total_pct']:6.1f} mdd {r['mdd_total']:6.1f} worstyear {r['worst_year_pct']:6.1f} lostyears {r['years_lost_money']} w {r['mean_weight']} sales {r['sales']} cost {r['cost_pct']:.2f} | peaks lost {tsum['peak'][p['name']]['lost']}/{tsum['peak'][p['name']]['n']} worst {tsum['peak'][p['name']]['worst']:.1f}", flush=True)
    for p in pols:
        print("  years", p["name"], {y: round(v, 1) for y, v in years[p["name"]].items()}, flush=True)
    cr = crisis2008(pols)
    print("CRISIS", cr["window"], flush=True)
    for p in pols:
        r = cr["results"][p["name"]]
        print(f"  {p['name']:14s} total {r['total_pct']:7.1f} lowest {r['lowest_total_pct']:6.1f} mdd {r['mdd_total']:6.1f} years {({y: round(v, 1) for y, v in r['years'].items()})}", flush=True)
    rows = synth_eval(pols)
    summ = summarize(rows, pols)
    rec = dict(version="voltime-v1", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], policies=pols, real=dict(info=info, results=real, years=years, turning_points=turn, turning_summary=tsum), real_trades=trades, crisis2008=cr, synthetic_summary=summ, synthetic_paths=rows)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False)
    for t in list(SC.TRENDS) + ["all"]:
        for p in pols:
            s = summ[t][p["name"]]
            print(f"SYN {t:8s} {p['name']:14s} total {s['median_total_pct']:7.1f} worst {s['worst_total_pct']:7.1f} lowest {s['median_lowest_pct']:6.1f} mdd {s['median_mdd']:6.1f} made {s['made_money']}/{s['paths']} -hold {s['median_minus_hold']:6.1f}")
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
