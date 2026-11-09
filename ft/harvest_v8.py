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
import voltime_v1 as VT
import turning_points_v1 as TP

CAPITAL = H5.CAPITAL
LONG, INV = H5.LONG, H5.INV
SKIM_G = 0.10
ONE_WON_FROM = "2025-09-01"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def tick(day, price):
    if price < 2000 or (day >= ONE_WON_FROM and price < 5000):
        return 1.0
    return 5.0


def policies():
    out = [dict(name="R0_hold", kind="skim", sig=None), dict(name="R1_skim10", kind="skim", g=SKIM_G, sig=None)]
    for s in ("ribbon", "ribbon_highvol", "sma200_down", "sma200_down_highvol"):
        out.append(dict(name=f"I_{s}", kind="skim", g=SKIM_G, bear="inv", sig=s))
    return out


def signals(A):
    A = np.asarray(A, float)
    e = {k: np.asarray(CT.ema(list(A), k)) for k in (8, 21, 50, 200)}
    rib = (e[8] < e[21]) & (e[21] < e[50]) & (e[50] < e[200])
    vol = VT.ewma_vol(list(A))
    med = VT.expanding_median(vol)
    hv = ~np.isnan(vol) & ~np.isnan(med) & (vol > med)
    sma = np.full(len(A), np.nan)
    cs = np.cumsum(np.insert(A, 0, 0.0))
    sma[199:] = (cs[200:] - cs[:-200]) / 200.0
    down = np.zeros(len(A), bool)
    down[219:] = (A[219:] < sma[219:]) & (sma[219:] < sma[199:-20])
    return dict(ribbon=rib, ribbon_highvol=rib & hv, sma200_down=down, sma200_down_highvol=down & hv)


def run_set(pols, px, cost, start, n, sig, keep=False, days=None):
    res, series, trades = {}, {}, {}
    for p in pols:
        bear = (lambda i: False) if p["sig"] is None else (lambda i, s=sig[p["sig"]]: bool(s[i]))
        r, s, t, _ = H5.run_skim(p, px, cost, start, n, bear, lambda i: (None, None), keep)
        vals = [s[k] for k in sorted(s)]
        r = dict(r, lowest_total_pct=100 * (min(vals) / CAPITAL - 1), inverse_days=int(sum(bool(sig[p["sig"]][i]) for i in range(start, n - 1))) if p["sig"] else 0, switches=sum(1 for x in t if x["side"] == "buy" and x["leg"] == INV) if keep else None)
        res[p["name"]], series[p["name"]], trades[p["name"]] = r, s, t
    return res, series, trades


def real_eval(pols):
    raw = json.load(open(os.path.join(os.path.dirname(HERE), "data", "kr_etf", "kodex_daily_raw.json"), encoding="utf-8"))[LONG]
    adj = json.load(open(CT.ADJ, encoding="utf-8"))[LONG]
    days = [d for d in raw["days"] if d <= H5.H2.REAL_TO]
    sig_all = signals([float(x) for x in adj["C"][:len(days)]])
    sub, px, _, bull, bear, brk, pred, Ak = H5.load_real()
    off = days.index(sub[0])
    sig = {k: v[off:] for k, v in sig_all.items()}
    cost = lambda leg, j: LI.FEE + (LI.TICK if leg == LONG else tick(sub[j], px[INV]["O"][j])) / px[leg]["O"][j]
    res, series, trades = run_set(pols, px, cost, 0, len(sub), sig, keep=True)
    years = {k: H5.yearly(s, sub) for k, s in series.items()}
    for k in res:
        res[k]["years_lost_money"] = [y for y, v in years[k].items() if v < 0]
        res[k]["worst_year_pct"] = min(years[k].values())
    tr = {k: [dict(x, decision_day=sub[x["decision_index"]], fill_day=sub[x["fill_index"]]) for x in t] for k, t in trades.items()}
    pts = TP.zigzag(Ak, TP.ZIG)
    turn = []
    for kind, i in pts:
        a, b = max(1, i - TP.BEFORE), min(len(sub) - 1, i + TP.AFTER)
        row = dict(kind=kind, day=sub[i], window=[sub[a], sub[b]], market_pct=100 * (px[LONG]["C"][b] / px[LONG]["C"][a] - 1))
        for p in pols:
            row[p["name"]] = 100 * (series[p["name"]][b] / series[p["name"]][a] - 1)
        turn.append(row)
    return res, years, tr, turn, dict(first_decision=sub[0], last_day=sub[-1])


def crisis_eval(pols):
    raw = json.load(open(os.path.join(os.path.dirname(HERE), "data", "kr_etf", "kodex_daily_raw.json"), encoding="utf-8"))[LONG]
    adj = json.load(open(CT.ADJ, encoding="utf-8"))[LONG]
    days = raw["days"]
    sig_all = signals([float(x) for x in adj["C"]])
    a = days.index(next(d for d in days if d >= "2007-01-02")) - 1
    b = days.index(next(d for d in days if d >= "2012-12-28"))
    keep = list(range(a, b + 1))
    O = np.asarray([float(raw["O"][i]) for i in keep])
    C = np.asarray([float(raw["C"][i]) for i in keep])
    IO, IC = SC.inverse_path(O, C)
    px = {LONG: dict(O=list(O), C=list(C)), INV: dict(O=[float(x) for x in IO], C=[float(x) for x in IC])}
    cost = lambda leg, j: LI.FEE + LI.TICK / px[leg]["O"][j]
    sig = {k: v[a:b + 1] for k, v in sig_all.items()}
    res, series, _ = run_set(pols, px, cost, 0, len(keep), sig)
    sd = [days[i] for i in keep]
    for k in res:
        res[k]["years"] = H5.yearly(series[k], sd)
    return dict(window=[sd[0], sd[-1]], inverse="synthetic -1x daily of 069500 raw with scenario_test_v1 INV_DRAG", results=res)


def synth_eval(pols):
    rows = []
    for trend in SC.TRENDS:
        for case in SC.LIQ_CASES:
            for seed in range(SC.SEEDS):
                O, C, liq = SC.generate(trend, case, seed)
                IO, IC = SC.inverse_path(O, C)
                px = {LONG: dict(O=[float(x) for x in O], C=[float(x) for x in C]), INV: dict(O=[float(x) for x in IO], C=[float(x) for x in IC])}
                cost = lambda leg, j: LI.FEE + liq[j]["cost_bp"] / 10000.0
                sig = signals([float(x) for x in C])
                res, _, _ = run_set(pols, px, cost, SC.WARM, len(C), sig)
                rows.append(dict(trend=trend, liquidity=case, seed=seed, res=res))
    return rows


def summarize(rows, pols):
    out = {}
    for t in list(SC.TRENDS) + ["all"]:
        ps = [r for r in rows if t == "all" or r["trend"] == t]
        out[t] = {}
        for p in pols:
            v = lambda k: [r["res"][p["name"]][k] for r in ps]
            out[t][p["name"]] = dict(median_total_pct=float(np.median(v("total_pct"))), worst_total_pct=float(min(v("total_pct"))), median_lowest_pct=float(np.median(v("lowest_total_pct"))), median_mdd=float(np.median(v("mdd_total"))), made_money=int(sum(x > 0 for x in v("total_pct"))), median_minus_hold=float(np.median([r["res"][p["name"]]["total_pct"] - r["res"]["R0_hold"]["total_pct"] for r in ps])), median_minus_R1=float(np.median([r["res"][p["name"]]["total_pct"] - r["res"]["R1_skim10"]["total_pct"] for r in ps])), paths=len(ps))
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
    real, years, trades, turn, info = real_eval(pols)
    print("REAL", info, flush=True)
    for p in pols:
        r = real[p["name"]]
        print(f"{p['name']:24s} vault {r['vault_pct']:6.1f} total {r['total_pct']:7.1f} lowest {r['lowest_total_pct']:6.1f} mdd {r['mdd_total']:6.1f} worstyear {r['worst_year_pct']:6.1f} lost {len(r['years_lost_money'])} invdays {r['inverse_days']} switches {r['switches']} cost {r['cost_pct']:.2f}", flush=True)
        print("   years", {y: round(float(v), 1) for y, v in years[p["name"]].items()}, flush=True)
    cr = crisis_eval(pols)
    print("CRISIS", cr["window"], flush=True)
    for p in pols:
        r = cr["results"][p["name"]]
        print(f"  {p['name']:24s} total {r['total_pct']:7.1f} lowest {r['lowest_total_pct']:6.1f} mdd {r['mdd_total']:6.1f} invdays {r['inverse_days']} years {({y: round(float(v), 1) for y, v in r['years'].items()})}", flush=True)
    rows = synth_eval(pols)
    summ = summarize(rows, pols)
    rec = dict(version="harvest-v8", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], policies=pols, real=dict(info=info, results=real, years=years, turning_points=turn), real_trades=trades, crisis=cr, synthetic_summary=summ, synthetic_paths=rows)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, default=float)
    for t in list(SC.TRENDS) + ["all"]:
        for p in pols:
            s = summ[t][p["name"]]
            print(f"SYN {t:8s} {p['name']:24s} total {s['median_total_pct']:7.1f} worst {s['worst_total_pct']:7.1f} lowest {s['median_lowest_pct']:6.1f} made {s['made_money']}/{s['paths']} -hold {s['median_minus_hold']:6.1f} -R1 {s['median_minus_R1']:6.1f}")
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
