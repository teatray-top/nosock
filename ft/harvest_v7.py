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
import learn_shadow as SH
import harvest_v5 as H5
import harvest_v6 as H6
import turning_points_v1 as TP

CAPITAL = H5.CAPITAL
VAULT_SHARE = H5.VAULT_SHARE
LONG, INV = H5.LONG, H5.INV
BAND, SKIM_G = H6.BAND, H6.SKIM_G


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def policies():
    out = [dict(name="R0_hold", kind="skim"), dict(name="R1_skim10", kind="skim", g=SKIM_G), dict(name="R3_skim10_bearinv", kind="skim", g=SKIM_G, bear="inv")]
    for f in (0.90, 0.95):
        for sw in (None, "ribbon", "model"):
            out.append(dict(name=f"P{int(f * 100)}_m3" + (f"_inv_{sw}" if sw else ""), kind="cppi2", floor=f, m=3, g=SKIM_G, switch=sw))
    return out


def run_cppi2(pol, px, cost, start, n, want_leg, keep=False):
    cash, vault, q, B, leg = CAPITAL, 0.0, 0.0, 0.0, None
    F = pol["floor"] * CAPITAL
    m, g = pol["m"], pol["g"]
    sales, buys, switches, cost_paid, trades = 0, 0, 0, 0.0, []
    series = {start: CAPITAL}
    peak, mdd, low = CAPITAL, 0.0, CAPITAL
    breach05, strict = 0, 0
    for i in range(start, n - 1):
        j = i + 1
        want = want_leg(i)
        if leg is not None and q > 0 and want != leg:
            risky = q * px[leg]["C"][i]
            W = cash + risky + vault
            target = min(cash + risky, m * max(0.0, W - F))
            o, c = px[leg]["O"][j], cost(leg, j)
            proceeds = q * o * (1 - c)
            cost_paid += q * o * c
            profit = proceeds - B
            tv = VAULT_SHARE * profit if profit > 0 else 0.0
            vault += tv
            cash += proceeds - tv
            sales += 1
            switches += 1
            if keep:
                trades.append(dict(side="sell", leg=leg, decision_index=i, fill_index=j, price=o, qty=q, proceeds=proceeds, profit=profit, to_vault=tv, reason="switch"))
            q, B, leg = 0.0, 0.0, None
            if target > 0:
                o2, c2 = px[want]["O"][j], cost(want, j)
                spend = min(cash, target / px[want]["C"][i] * o2 * (1 + c2))
                if spend > 0:
                    bq = spend / (o2 * (1 + c2))
                    q, B, leg = bq, spend, want
                    cash -= spend
                    cost_paid += bq * o2 * c2
                    buys += 1
                    if keep:
                        trades.append(dict(side="buy", leg=want, decision_index=i, fill_index=j, price=o2, qty=bq, spend=spend, reason="switch"))
        else:
            cur = leg if (leg is not None and q > 0) else want
            c_i, o, c = px[cur]["C"][i], px[cur]["O"][j], cost(cur, j)
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
                    trades.append(dict(side="skim", leg=cur, decision_index=i, fill_index=j, price=o, qty=sq, proceeds=proceeds, to_vault=proceeds))
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
                            leg = cur
                            if keep:
                                trades.append(dict(side="buy", leg=cur, decision_index=i, fill_index=j, price=o, qty=bq, spend=spend))
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
                            trades.append(dict(side="sell", leg=cur, decision_index=i, fill_index=j, price=o, qty=sq, proceeds=proceeds, profit=profit, to_vault=tv))
                        if q <= 0:
                            q, B, leg = 0.0, 0.0, None
        tot = cash + (q * px[leg]["C"][j] if leg else 0.0) + vault
        series[j] = tot
        peak = max(peak, tot)
        mdd = min(mdd, tot / peak - 1)
        low = min(low, tot)
        breach05 += tot < F - 0.005 * CAPITAL
        strict += tot < F
    if leg:
        c = cost(leg, n - 1)
        cost_paid += q * px[leg]["C"][n - 1] * c
        eq_end = cash + q * px[leg]["C"][n - 1] * (1 - c)
    else:
        eq_end = cash
    final = eq_end + vault
    mdd = min(mdd, final / peak - 1)
    low = min(low, final)
    series[n - 1] = final
    out = dict(vault_pct=100 * vault / CAPITAL, equity_pct=100 * eq_end / CAPITAL, total_pct=100 * (final / CAPITAL - 1), mdd_total=100 * mdd, lowest_total_pct=100 * (low / CAPITAL - 1), days_below_floor_05=int(breach05), days_below_floor_strict=int(strict), terminal_breach=bool(final < F), sales=sales, buys=buys, switches=switches, cost_pct=100 * cost_paid / CAPITAL)
    return out, series, trades


def run_pol(p, px, cost, start, n, bear, pred, keep=False):
    if p["kind"] == "cppi2":
        sw = p["switch"]
        if sw is None:
            want = lambda i: LONG
        elif sw == "ribbon":
            want = lambda i: INV if bear(i) else LONG
        else:
            def want(i):
                pl, ps = pred(i)
                return INV if (pl is not None and ps > pl and ps > LI.BAND) else LONG
        return run_cppi2(p, px, cost, start, n, want, keep)
    r, s, t, _ = H5.run_skim(p, px, cost, start, n, bear, pred, keep)
    vals = [s[k] for k in sorted(s)]
    r = dict(r, lowest_total_pct=100 * (min(vals) / CAPITAL - 1), days_below_floor_05=None, days_below_floor_strict=None, terminal_breach=None, buys=None, switches=None)
    return r, s, t


def real_eval(pols):
    sub, px, cost, bull, bear, brk, pred, A = H5.load_real()
    res, series, trades, years = {}, {}, {}, {}
    for p in pols:
        r, s, t = run_pol(p, px, cost, 0, len(sub), bear, pred, keep=True)
        res[p["name"]], series[p["name"]] = r, s
        trades[p["name"]] = [dict(x, decision_day=sub[x["decision_index"]], fill_day=sub[x["fill_index"]]) for x in t]
        years[p["name"]] = H5.yearly(s, sub)
    for p in pols:
        ys = years[p["name"]]
        res[p["name"]]["years_lost_money"] = [y for y, v in ys.items() if v < 0]
        res[p["name"]]["worst_year_pct"] = min(ys.values())
    v6 = {"P90_m3": 72.13757940376377, "P95_m3": 26.48}
    parity = dict(P90_m3=res["P90_m3"]["total_pct"], P95_m3=res["P95_m3"]["total_pct"])
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
    return res, trades, years, turn, tsum, parity, dict(first_decision=sub[0], last_day=sub[-1])


def synth_eval(pols, models):
    rows = []
    for trend in SC.TRENDS:
        for case in SC.LIQ_CASES:
            for seed in range(SC.SEEDS):
                O, C, liq = SC.generate(trend, case, seed)
                IO, IC = SC.inverse_path(O, C)
                Cl = [float(x) for x in C]
                F = LI.features([None] * len(Cl), dict(Ca=Cl))
                idx = list(range(SC.WARM, len(Cl) - 1))
                X = np.asarray([F[i] for i in idx])
                pm = dict(zip(idx, zip((float(x) for x in models["long"].predict(X)), (float(x) for x in models["inv"].predict(X)))))
                e = {k: CT.ema(Cl, k) for k in (8, 21, 50, 200)}
                px = {LONG: dict(O=[float(x) for x in O], C=Cl), INV: dict(O=[float(x) for x in IO], C=[float(x) for x in IC])}
                cost = lambda leg, j: LI.FEE + liq[j]["cost_bp"] / 10000.0
                bear = lambda i: e[8][i] < e[21][i] < e[50][i] < e[200][i]
                pred = lambda i: pm.get(i, (None, None))
                res = {}
                for p in pols:
                    r, _, _ = run_pol(p, px, cost, SC.WARM, len(Cl), bear, pred)
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
            d = dict(median_total_pct=float(np.median(v("total_pct"))), median_vault_pct=float(np.median(v("vault_pct"))), worst_total_pct=float(min(v("total_pct"))), best_total_pct=float(max(v("total_pct"))), median_lowest_pct=float(np.median(v("lowest_total_pct"))), worst_lowest_pct=float(min(v("lowest_total_pct"))), made_money=int(sum(x > 0 for x in v("total_pct"))), lost_over_5pct=int(sum(x < -5 for x in v("total_pct"))), median_minus_hold=float(np.median([r["res"][p["name"]]["total_pct"] - r["res"]["R0_hold"]["total_pct"] for r in ps])), median_mdd=float(np.median(v("mdd_total"))), paths=len(ps))
            if p["kind"] == "cppi2":
                d.update(paths_strict_breach=int(sum(x > 0 for x in v("days_below_floor_strict"))), paths_breach_05=int(sum(x > 0 for x in v("days_below_floor_05"))), paths_terminal_breach=int(sum(bool(x) for x in v("terminal_breach"))), median_switches=float(np.median(v("switches"))))
            out[t][p["name"]] = d
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
    real, trades, years, turn, tsum, parity, info = real_eval(pols)
    print("REAL", info, "parity", parity, flush=True)
    for p in pols:
        r = real[p["name"]]
        print(f"{p['name']:22s} vault {r['vault_pct']:6.1f} total {r['total_pct']:7.1f} lowest {r['lowest_total_pct']:6.1f} mdd {r['mdd_total']:6.1f} worstyear {r['worst_year_pct']:6.1f} lostyears {len(r['years_lost_money'])} sales {r['sales']} switches {r['switches']} cost {r['cost_pct']:.2f} strict {r['days_below_floor_strict']} b05 {r['days_below_floor_05']} term {r['terminal_breach']}", flush=True)
    models, _ = SH.load_models()
    rows = synth_eval(pols, models)
    summ = summarize(rows, pols)
    rec = dict(version="harvest-v7", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], policies=pols, real=dict(info=info, parity=parity, results=real, years=years, turning_points=turn, turning_summary=tsum), real_trades=trades, synthetic_summary=summ, synthetic_paths=rows)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    for t in list(SC.TRENDS) + ["all"]:
        for p in pols:
            s = summ[t][p["name"]]
            extra = f" strict {s['paths_strict_breach']} b05 {s['paths_breach_05']} term {s['paths_terminal_breach']} sw {s['median_switches']:.0f}" if "paths_strict_breach" in s else ""
            print(f"SYN {t:8s} {p['name']:22s} total {s['median_total_pct']:7.1f} worst {s['worst_total_pct']:7.1f} best {s['best_total_pct']:7.1f} made {s['made_money']:3d} >5%loss {s['lost_over_5pct']:3d}/{s['paths']} -hold {s['median_minus_hold']:7.1f} vault {s['median_vault_pct']:5.1f}{extra}")
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
