import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
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
BAND = 0.05
SKIM_G = 0.10


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def policies():
    out = [dict(name="R0_hold", kind="skim"), dict(name="R1_skim10", kind="skim", g=SKIM_G)]
    for f in (0.95, 0.90):
        for m in (3, 5):
            out.append(dict(name=f"P_floor{int(f * 100)}_m{m}", kind="cppi", floor=f, m=m, g=SKIM_G))
    return out


def run_cppi(pol, px, cost, start, n, keep=False):
    P = px[LONG]
    cash, vault, q, B = CAPITAL, 0.0, 0.0, 0.0
    F = pol["floor"] * CAPITAL
    m, g = pol["m"], pol["g"]
    sales, buys, cost_paid, trades = 0, 0, 0.0, []
    series = {start: CAPITAL}
    peak, mdd, low = CAPITAL, 0.0, CAPITAL
    breach = 0
    for i in range(start, n - 1):
        j = i + 1
        c_i, o, c = P["C"][i], P["O"][j], cost(LONG, j)
        risky = q * c_i
        if q > 0 and risky - B >= g * B:
            f = VAULT_SHARE * (risky - B) / risky
            sq = f * q
            proceeds = sq * o * (1 - c)
            cost_paid += sq * o * c
            vault += proceeds
            q -= sq
            B = q * o
            sales += 1
            if keep:
                trades.append(dict(side="skim", decision_index=i, fill_index=j, price=o, qty=sq, proceeds=proceeds, to_vault=proceeds))
        else:
            acc = cash + risky
            W = acc + vault
            target = min(acc, m * max(0.0, W - F))
            if abs(target - risky) > BAND * W or (target == 0.0 and q > 0):
                dq = target / c_i - q
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
                            trades.append(dict(side="buy", decision_index=i, fill_index=j, price=o, qty=bq, spend=spend))
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
                        trades.append(dict(side="sell", decision_index=i, fill_index=j, price=o, qty=sq, proceeds=proceeds, profit=profit, to_vault=tv))
        tot = cash + q * P["C"][j] + vault
        series[j] = tot
        peak = max(peak, tot)
        mdd = min(mdd, tot / peak - 1)
        low = min(low, tot)
        breach += tot < F - 0.005 * CAPITAL
    c = cost(LONG, n - 1)
    cost_paid += q * P["C"][n - 1] * c
    eq_end = cash + q * P["C"][n - 1] * (1 - c)
    final = eq_end + vault
    mdd = min(mdd, final / peak - 1)
    low = min(low, final)
    series[n - 1] = final
    out = dict(vault_pct=100 * vault / CAPITAL, equity_pct=100 * eq_end / CAPITAL, total_pct=100 * (final / CAPITAL - 1), mdd_total=100 * mdd, lowest_total_pct=100 * (low / CAPITAL - 1), days_below_floor=int(breach), sales=sales, buys=buys, cost_pct=100 * cost_paid / CAPITAL)
    return out, series, trades


def run_pol(p, px, cost, start, n, keep=False):
    if p["kind"] == "cppi":
        return run_cppi(p, px, cost, start, n, keep)
    r, s, t, _ = H5.run_skim(p, px, cost, start, n, lambda i: False, lambda i: (None, None), keep)
    vals = [s[k] for k in sorted(s)]
    r = dict(r, lowest_total_pct=100 * (min(vals) / CAPITAL - 1), days_below_floor=None, buys=1)
    return r, s, t


def real_eval(pols):
    sub, px, cost, bull, bear, brk, pred, A = H5.load_real()
    res, series, trades, years = {}, {}, {}, {}
    for p in pols:
        r, s, t = run_pol(p, px, cost, 0, len(sub), keep=True)
        res[p["name"]], series[p["name"]] = r, s
        trades[p["name"]] = [dict(x, decision_day=sub[x["decision_index"]], fill_day=sub[x["fill_index"]]) for x in t]
        years[p["name"]] = H5.yearly(s, sub)
    for p in pols:
        ys = years[p["name"]]
        res[p["name"]]["years_lost_money"] = [y for y, v in ys.items() if v < 0]
        res[p["name"]]["worst_year_pct"] = min(ys.values())
    pts = TP.zigzag(A, TP.ZIG)
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
    return res, trades, years, turn, tsum, dict(first_decision=sub[0], last_day=sub[-1])


def synth_eval(pols):
    rows = []
    for trend in SC.TRENDS:
        for case in SC.LIQ_CASES:
            for seed in range(SC.SEEDS):
                O, C, liq = SC.generate(trend, case, seed)
                px = {LONG: dict(O=[float(x) for x in O], C=[float(x) for x in C])}
                cost = lambda leg, j: LI.FEE + liq[j]["cost_bp"] / 10000.0
                res = {}
                for p in pols:
                    r, _, _ = run_pol(p, px, cost, SC.WARM, len(C))
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
            out[t][p["name"]] = dict(median_total_pct=float(np.median(v("total_pct"))), median_vault_pct=float(np.median(v("vault_pct"))), worst_total_pct=float(min(v("total_pct"))), median_lowest_pct=float(np.median(v("lowest_total_pct"))), worst_lowest_pct=float(min(v("lowest_total_pct"))), lost_money=int(sum(x < 0 for x in v("total_pct"))), lost_over_5pct=int(sum(x < -5 for x in v("total_pct"))), median_minus_hold=float(np.median([r["res"][p["name"]]["total_pct"] - r["res"]["R0_hold"]["total_pct"] for r in ps])), median_mdd=float(np.median(v("mdd_total"))), paths=len(ps))
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
        print(f"{p['name']:18s} vault {r['vault_pct']:6.1f} total {r['total_pct']:7.1f} lowest {r['lowest_total_pct']:6.1f} mdd {r['mdd_total']:6.1f} worstyear {r['worst_year_pct']:6.1f} lostyears {r['years_lost_money']} sales {r['sales']} buys {r['buys']} cost {r['cost_pct']:.2f} belowfloor {r['days_below_floor']}", flush=True)
    rows = synth_eval(pols)
    summ = summarize(rows, pols)
    rec = dict(version="harvest-v6", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], policies=pols, real=dict(info=info, results=real, years=years, turning_points=turn, turning_summary=tsum), real_trades=trades, synthetic_summary=summ, synthetic_paths=rows)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    for t in list(SC.TRENDS) + ["all"]:
        for p in pols:
            s = summ[t][p["name"]]
            print(f"SYN {t:8s} {p['name']:18s} total {s['median_total_pct']:7.1f} worst {s['worst_total_pct']:7.1f} lowest med {s['median_lowest_pct']:6.1f} worst {s['worst_lowest_pct']:6.1f} lost {s['lost_money']:3d} >5% {s['lost_over_5pct']:3d}/{s['paths']} -hold {s['median_minus_hold']:7.1f} vault {s['median_vault_pct']:5.1f} mdd {s['median_mdd']:6.1f}")
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
