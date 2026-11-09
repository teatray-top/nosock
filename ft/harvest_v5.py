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
import harvest_v2 as H2
import harvest_v4 as H4
import turning_points_v1 as TP

CAPITAL = H4.CAPITAL
VAULT_SHARE = H4.VAULT_SHARE
LONG, INV = H4.LONG, H4.INV


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def policies():
    return [
        dict(name="R0_hold", kind="skim"),
        dict(name="R1_skim10", kind="skim", g=0.10),
        dict(name="R2_skim10_bearcash", kind="skim", g=0.10, bear="cash"),
        dict(name="R3_skim10_bearinv", kind="skim", g=0.10, bear="inv"),
        dict(name="R4_skim10_modelcash", kind="skim", g=0.10, model=True),
        dict(name="R2a_skim5_bearcash", kind="skim", g=0.05, bear="cash"),
        dict(name="R2b_skim20_bearcash", kind="skim", g=0.20, bear="cash"),
        dict(name="ref_A_long_X1_trail5", kind="A_long", trail=0.05),
    ]


def run_skim(pol, px, cost, start, n, bear_at, pred, keep=False):
    cash, vault = CAPITAL, 0.0
    leg, q, B = None, 0.0, 0.0
    sales, cost_paid, trades, flows = 0, 0.0, [], []
    series = {start: CAPITAL}
    peak, mdd = CAPITAL, 0.0
    g, bear, model = pol.get("g"), pol.get("bear"), pol.get("model")
    out_state = False
    for i in range(start, n - 1):
        j = i + 1
        want = LONG
        if bear and bear_at(i):
            want = INV if bear == "inv" else None
        if model:
            pl = pred(i)[0]
            if pl is not None:
                if pl < -LI.BAND:
                    out_state = True
                elif pl > LI.BAND:
                    out_state = False
            if out_state:
                want = None
        if leg != want:
            if leg is not None:
                c = cost(leg, j)
                o = px[leg]["O"][j]
                proceeds = q * o * (1 - c)
                cost_paid += q * o * c
                profit = proceeds - B
                tv = VAULT_SHARE * profit if profit > 0 else 0.0
                vault += tv
                cash += proceeds - tv
                sales += 1
                if tv > 0:
                    flows.append((j, tv))
                if keep:
                    trades.append(dict(side="sell", leg=leg, decision_index=i, fill_index=j, price=o, qty=q, proceeds=proceeds, profit=profit, to_vault=tv, reason="regime_exit"))
                leg, q, B = None, 0.0, 0.0
            if want is not None and cash > 0:
                c = cost(want, j)
                o = px[want]["O"][j]
                q = cash * (1 - c) / o
                cost_paid += cash * c
                B, cash, leg = cash, 0.0, want
                if keep:
                    trades.append(dict(side="buy", leg=leg, decision_index=i, fill_index=j, price=o, qty=q, cost_basis=B))
        elif leg is not None and g is not None:
            v = q * px[leg]["C"][i]
            p = v - B
            if p >= g * B:
                f = VAULT_SHARE * p / v
                c = cost(leg, j)
                o = px[leg]["O"][j]
                sq = f * q
                proceeds = sq * o * (1 - c)
                cost_paid += sq * o * c
                vault += proceeds
                flows.append((j, proceeds))
                sales += 1
                q -= sq
                B = q * o
                if keep:
                    trades.append(dict(side="skim", leg=leg, decision_index=i, fill_index=j, price=o, qty=sq, proceeds=proceeds, to_vault=proceeds, new_basis=B))
        tot = cash + (q * px[leg]["C"][j] if leg else 0.0) + vault
        series[j] = tot
        peak = max(peak, tot)
        mdd = min(mdd, tot / peak - 1)
    if leg:
        c = cost(leg, n - 1)
        cost_paid += q * px[leg]["C"][n - 1] * c
        eq_end = cash + q * px[leg]["C"][n - 1] * (1 - c)
    else:
        eq_end = cash
    final = eq_end + vault
    mdd = min(mdd, final / peak - 1)
    series[n - 1] = final
    out = dict(vault_pct=100 * vault / CAPITAL, equity_pct=100 * eq_end / CAPITAL, total_pct=100 * (final / CAPITAL - 1), mdd_total=100 * mdd, sales=sales, cost_pct=100 * cost_paid / CAPITAL)
    return out, series, trades, flows


def same_flow_hold(flows, px, cost, start, n):
    j0 = start + 1
    c0 = cost(LONG, j0)
    q = CAPITAL * (1 - c0) / px[LONG]["O"][j0]
    vault = 0.0
    for j, amt in flows:
        c = cost(LONG, j)
        o = px[LONG]["O"][j]
        sq = min(q, amt / (o * (1 - c)))
        q -= sq
        vault += sq * o * (1 - c)
    c = cost(LONG, n - 1)
    return 100 * ((vault + q * px[LONG]["C"][n - 1] * (1 - c)) / CAPITAL - 1)


def run_any(pol, px, cost, start, n, pred, bull, bear, brk, keep=False):
    if pol["kind"] == "skim":
        r, s, t, fl = run_skim(pol, px, cost, start, n, bear, pred, keep)
    else:
        r, s, t = H4.run(pol, px, cost, start, n, pred, bull, bear, brk, keep_trades=True)
        fl = [(x["fill_index"], x["to_vault"]) for x in t if x["side"] == "sell" and x["to_vault"] > 0]
        if not keep:
            t = []
    r = dict(r, same_flow_hold_total_pct=same_flow_hold(fl, px, cost, start, n))
    r["minus_same_flow_hold"] = r["total_pct"] - r["same_flow_hold_total_pct"]
    return r, s, t


def load_real():
    raw = json.load(open(os.path.join(os.path.dirname(HERE), "data", "kr_etf", "kodex_daily_raw.json"), encoding="utf-8"))[LONG]
    adj = json.load(open(CT.ADJ, encoding="utf-8"))[LONG]
    iv = json.load(open(H2.INV_DAILY, encoding="utf-8"))[INV]
    imap = {d: (float(o), float(c)) for d, o, c in zip(iv["days"], iv["O"], iv["C"])}
    days = [d for d in raw["days"] if d <= H2.REAL_TO]
    A = [float(x) for x in adj["C"][:len(days)]]
    e = {m: CT.ema(A, m) for m in (8, 21, 50, 200)}
    start = days.index(H2.REAL_FROM) - 1
    keep = list(range(start, len(days)))
    sub = [days[i] for i in keep]
    px = {LONG: dict(O=[float(raw["O"][i]) for i in keep], C=[float(raw["C"][i]) for i in keep]), INV: dict(O=[imap[days[i]][0] for i in keep], C=[imap[days[i]][1] for i in keep])}
    pr = json.load(open(H2.PRED_FILE, encoding="utf-8"))["preds"]
    bull = lambda i: e[8][keep[i]] > e[21][keep[i]] > e[50][keep[i]] > e[200][keep[i]]
    bear = lambda i: e[8][keep[i]] < e[21][keep[i]] < e[50][keep[i]] < e[200][keep[i]]
    brk = lambda i, leg: (e[8][keep[i]] < e[21][keep[i]]) if leg == LONG else (e[8][keep[i]] > e[21][keep[i]])
    pred = lambda i: tuple(pr[sub[i]]) if sub[i] in pr else (None, None)
    cost = lambda leg, j: LI.FEE + LI.TICK / px[leg]["O"][j]
    return sub, px, cost, bull, bear, brk, pred, [A[i] for i in keep]


def yearly(series, sub):
    out, prev = {}, 0
    years = sorted({d[:4] for d in sub[1:]})
    for y in years:
        last = max(i for i, d in enumerate(sub) if d[:4] == y)
        out[y] = 100 * (series[last] / series[prev] - 1)
        prev = last
    return out


def real_eval(pols):
    sub, px, cost, bull, bear, brk, pred, A = load_real()
    res, series, trades, years = {}, {}, {}, {}
    for p in pols:
        r, s, t = run_any(p, px, cost, 0, len(sub), pred, bull, bear, brk, keep=True)
        res[p["name"]], series[p["name"]] = r, s
        trades[p["name"]] = [dict(x, decision_day=sub[x["decision_index"]], fill_day=sub[x["fill_index"]]) for x in t]
        years[p["name"]] = yearly(s, sub)
    h4 = H4.run(dict(name="hold", kind="hold"), px, cost, 0, len(sub), pred, bull, bear, brk)[0]
    parity = dict(v4_hold_total_pct=h4["total_pct"], v5_R0_total_pct=res["R0_hold"]["total_pct"], equal=abs(h4["total_pct"] - res["R0_hold"]["total_pct"]) < 1e-9)
    for p in pols:
        ys = years[p["name"]]
        res[p["name"]]["years_lost_money"] = [y for y, v in ys.items() if v < 0]
        res[p["name"]]["years_below_hold"] = [y for y, v in ys.items() if v < years["R0_hold"][y]]
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
                e = {m: CT.ema(Cl, m) for m in (8, 21, 50, 200)}
                px = {LONG: dict(O=[float(x) for x in O], C=Cl), INV: dict(O=[float(x) for x in IO], C=[float(x) for x in IC])}
                cost = lambda leg, j: LI.FEE + liq[j]["cost_bp"] / 10000.0
                bull = lambda i: e[8][i] > e[21][i] > e[50][i] > e[200][i]
                bear = lambda i: e[8][i] < e[21][i] < e[50][i] < e[200][i]
                brk = lambda i, leg: (e[8][i] < e[21][i]) if leg == LONG else (e[8][i] > e[21][i])
                res = {}
                for p in pols:
                    r, _, _ = run_any(p, px, cost, SC.WARM, len(Cl), lambda i: pm.get(i, (None, None)), bull, bear, brk)
                    res[p["name"]] = r
                rows.append(dict(trend=trend, liquidity=case, seed=seed, res=res))
    return rows


def summarize(rows, pols):
    out = {}
    for t in list(SC.TRENDS) + ["all"]:
        ps = [r for r in rows if t == "all" or r["trend"] == t]
        m = lambda name, key: float(np.median([r["res"][name][key] for r in ps]))
        out[t] = {p["name"]: dict(median_vault_pct=m(p["name"], "vault_pct"), median_total_pct=m(p["name"], "total_pct"), median_mdd=m(p["name"], "mdd_total"), median_minus_hold=float(np.median([r["res"][p["name"]]["total_pct"] - r["res"]["R0_hold"]["total_pct"] for r in ps])), median_minus_same_flow_hold=m(p["name"], "minus_same_flow_hold"), lost_money=int(sum(r["res"][p["name"]]["total_pct"] < 0 for r in ps)), below_hold=int(sum(r["res"][p["name"]]["total_pct"] < r["res"]["R0_hold"]["total_pct"] for r in ps)), below_same_flow_hold=int(sum(r["res"][p["name"]]["minus_same_flow_hold"] < 0 for r in ps)), paths=len(ps)) for p in pols}
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
        n = p["name"]
        r = real[n]
        print(f"{n:24s} vault {r['vault_pct']:6.1f} total {r['total_pct']:7.1f} sameflow {r['same_flow_hold_total_pct']:7.1f} mdd {r['mdd_total']:6.1f} sales {r['sales']:3d} cost {r['cost_pct']:5.2f} lostyears {r['years_lost_money']} belowhold {len(r['years_below_hold'])}", flush=True)
    models, _ = SH.load_models()
    rows = synth_eval(pols, models)
    summ = summarize(rows, pols)
    rec = dict(version="harvest-v5", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], policies=pols, real=dict(info=info, parity=parity, results=real, years=years, turning_points=turn, turning_summary=tsum), real_trades=trades, synthetic_summary=summ, synthetic_paths=rows)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    for t in list(SC.TRENDS) + ["all"]:
        for p in pols:
            s = summ[t][p["name"]]
            print(f"SYN {t:8s} {p['name']:24s} total {s['median_total_pct']:7.1f} vault {s['median_vault_pct']:6.1f} -hold {s['median_minus_hold']:7.1f} -sameflow {s['median_minus_same_flow_hold']:6.1f} lost {s['lost_money']:3d}/{s['paths']} belowhold {s['below_hold']:3d} belowsameflow {s['below_same_flow_hold']:3d} mdd {s['median_mdd']:6.1f}")
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
