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
import turning_points_v1 as TP

CAPITAL = 100_000_000.0
VAULT_SHARE = 0.5
TP_GAIN = 0.10
EXITS = {"X0_sl10": dict(sl=0.10), "X1_trail5": dict(trail=0.05), "X2_trail7": dict(trail=0.07), "X3_ema_break_sl10": dict(sl=0.10, ema_break=True)}
LONG, INV = "069500", "114800"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def policies():
    out = [dict(name="hold", kind="hold"), dict(name="ribbon", kind="ribbon")]
    for fam in ("A_long", "C_with_inverse"):
        for x, cfg in EXITS.items():
            out.append(dict(name=f"{fam}_{x}", kind=fam, **cfg))
    return out


def run(pol, px, cost, start, n, pred, bull_at, bear_at, brk_at, keep_trades=False):
    cash, vault = CAPITAL, 0.0
    leg, q, basis, entry, top = None, 0.0, 0.0, None, None
    sales, cost_paid, trades = 0, 0.0, []
    series = {start: CAPITAL}
    peak, mdd = CAPITAL, 0.0
    k = pol["kind"]
    for i in range(start, n - 1):
        j = i + 1
        act, why = None, None
        if k == "hold":
            if leg is None and i == start:
                act = ("buy", LONG)
        elif k == "ribbon":
            if leg is None and bull_at(i):
                act = ("buy", LONG)
            elif leg is not None and not bull_at(i):
                act, why = ("sell", None), "ribbon_off"
        elif leg is not None:
            c_now = px[leg]["C"][i]
            top = max(top, c_now)
            if c_now >= entry * (1 + TP_GAIN):
                act, why = ("sell", None), "take_profit"
            elif pol.get("sl") and c_now <= entry * (1 - pol["sl"]):
                act, why = ("sell", None), "stop_loss"
            elif pol.get("trail") and c_now <= top * (1 - pol["trail"]):
                act, why = ("sell", None), "trailing_stop"
            elif pol.get("ema_break") and brk_at(i, leg):
                act, why = ("sell", None), "trend_break"
        else:
            pl, ps = pred(i)
            tgt = H2.choose(k, pl, ps, bull_at(i), bear_at(i))
            if tgt:
                act = ("buy", tgt)
        if act and act[0] == "sell" and leg is not None:
            c = cost(leg, j)
            o = px[leg]["O"][j]
            proceeds = q * o * (1 - c)
            cost_paid += q * o * c
            profit = proceeds - basis
            tv = VAULT_SHARE * profit if profit > 0 else 0.0
            vault += tv
            cash += proceeds - tv
            sales += 1
            if keep_trades:
                trades.append(dict(side="sell", leg=leg, decision_index=i, fill_index=j, price=o, qty=q, proceeds=proceeds, profit=profit, to_vault=tv, reason=why))
            leg, q, basis, entry, top = None, 0.0, 0.0, None, None
        elif act and act[0] == "buy" and leg is None and cash > 0:
            leg = act[1]
            c = cost(leg, j)
            o = px[leg]["O"][j]
            q = cash * (1 - c) / o
            cost_paid += cash * c
            basis, entry, top, cash = cash, cash / q, px[leg]["C"][j], 0.0
            if keep_trades:
                trades.append(dict(side="buy", leg=leg, decision_index=i, fill_index=j, price=o, qty=q, cost_basis=basis))
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
    return out, series, trades


def real_eval(pols):
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
    res, series, trades = {}, {}, {}
    for p in pols:
        r, s, t = run(p, px, cost, 0, len(sub), pred, bull, bear, brk, keep_trades=True)
        res[p["name"]], series[p["name"]] = r, s
        trades[p["name"]] = [dict(x, decision_day=sub[x["decision_index"]], fill_day=sub[x["fill_index"]]) for x in t]
    pts = TP.zigzag([A[i] for i in keep], TP.ZIG)
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
    return res, trades, turn, tsum, dict(first_decision=sub[0], last_day=sub[-1])


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
                res = {}
                for p in pols:
                    r, _, _ = run(p, px, lambda leg, j: LI.FEE + liq[j]["cost_bp"] / 10000.0, SC.WARM, len(Cl), lambda i: pm.get(i, (None, None)), lambda i: e[8][i] > e[21][i] > e[50][i] > e[200][i], lambda i: e[8][i] < e[21][i] < e[50][i] < e[200][i], lambda i, leg: (e[8][i] < e[21][i]) if leg == LONG else (e[8][i] > e[21][i]))
                    res[p["name"]] = r
                rows.append(dict(trend=trend, liquidity=case, seed=seed, res=res))
    return rows


def summarize(rows, pols):
    out = {}
    for t in list(SC.TRENDS) + ["all"]:
        ps = [r for r in rows if t == "all" or r["trend"] == t]
        out[t] = {p["name"]: dict(median_vault_pct=float(np.median([r["res"][p["name"]]["vault_pct"] for r in ps])), median_equity_pct=float(np.median([r["res"][p["name"]]["equity_pct"] for r in ps])), median_total_pct=float(np.median([r["res"][p["name"]]["total_pct"] for r in ps])), median_mdd=float(np.median([r["res"][p["name"]]["mdd_total"] for r in ps])), total_positive=int(sum(r["res"][p["name"]]["total_pct"] > 0 for r in ps)), total_beats_hold=int(sum(r["res"][p["name"]]["total_pct"] > r["res"]["hold"]["total_pct"] for r in ps)), paths=len(ps)) for p in pols}
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
    real, trades, turn, tsum, info = real_eval(pols)
    models, _ = SH.load_models()
    rows = synth_eval(pols, models)
    summ = summarize(rows, pols)
    rec = dict(version="harvest-v4", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], policies=pols, real=dict(info=info, results=real, turning_points=turn, turning_summary=tsum), real_trades=trades, synthetic_summary=summ, synthetic_paths=rows)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print("REAL", info)
    for p in pols:
        n = p["name"]
        r, s, pk, tr = real[n], summ, tsum["peak"][n], tsum["trough"][n]
        print(f"{n:28s} real vault {r['vault_pct']:6.1f}% total {r['total_pct']:7.1f}% mdd {r['mdd_total']:6.1f} sales {r['sales']:3d} | peaks lost {pk['lost']}/{pk['n']} worst {pk['worst']:6.1f} mean {pk['mean']:5.1f} | troughs lost {tr['lost']}/{tr['n']} | synth v/t bull {s['bull'][n]['median_vault_pct']:5.1f}/{s['bull'][n]['median_total_pct']:6.1f} bear {s['bear'][n]['median_vault_pct']:5.1f}/{s['bear'][n]['median_total_pct']:6.1f} side {s['sideways'][n]['median_vault_pct']:5.1f}/{s['sideways'][n]['median_total_pct']:6.1f} all >hold {s['all'][n]['total_beats_hold']}")
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
