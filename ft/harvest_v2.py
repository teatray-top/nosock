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

CAPITAL = 100_000_000.0
VAULT_SHARE = 0.5
TPS = (0.05, 0.10)
SLS = (None, 0.10)
REAL_FROM, REAL_TO = "2013-01-02", "2026-11-05"
PRED_FILE = os.path.join(HERE, "results", "g0_walkforward_predictions_v1.json")
INV_DAILY = os.path.join(os.path.dirname(HERE), "data", "kr_etf", "inverse_114800_20261107_151154", "daily_raw.json")
LONG, INV = "069500", "114800"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def policies():
    out = [dict(name="hold", kind="hold"), dict(name="ribbon", kind="ribbon")]
    for k in ("A_long", "B_long_gated", "C_with_inverse", "D_trend_switch"):
        for t in TPS:
            for s in SLS:
                out.append(dict(name=f"{k}_tp{int(t * 100)}_sl{int(s * 100) if s else 0}", kind=k, tp=t, sl=s))
    return out


def choose(kind, pl, ps, bull, bear):
    if pl is None:
        return None
    if kind == "A_long":
        return LONG if pl > LI.BAND else None
    if kind == "B_long_gated":
        return LONG if pl > LI.BAND and not bear else None
    if kind == "C_with_inverse":
        best = max(((0.0, None), (pl, LONG), (ps, INV)), key=lambda x: x[0])
        return best[1] if best[0] > LI.BAND else None
    if kind == "D_trend_switch":
        if bear:
            return INV if ps > LI.BAND else None
        return LONG if pl > LI.BAND else None
    return None


def run(pol, px, cost, year, start, n, pred, bull_at, bear_at):
    cash, vault = CAPITAL, 0.0
    leg, q, basis, entry = None, 0.0, 0.0, None
    vault_by_year, sales, cost_paid, inv_sales = {}, 0, 0.0, 0
    peak, mdd = CAPITAL, 0.0
    k = pol["kind"]
    for i in range(start, n - 1):
        j = i + 1
        act = None
        if k == "hold":
            if leg is None and i == start:
                act = ("buy", LONG)
        elif k == "ribbon":
            if leg is None and bull_at(i):
                act = ("buy", LONG)
            elif leg is not None and not bull_at(i):
                act = ("sell", None)
        else:
            if leg is not None:
                c_now = px[leg]["C"][i]
                if c_now >= entry * (1 + pol["tp"]) or (pol["sl"] and c_now <= entry * (1 - pol["sl"])):
                    act = ("sell", None)
            else:
                pl, ps = pred(i)
                tgt = choose(k, pl, ps, bull_at(i), bear_at(i))
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
            vault_by_year[year(j)] = vault_by_year.get(year(j), 0.0) + tv
            cash += proceeds - tv
            sales += 1
            inv_sales += leg == INV
            leg, q, basis, entry = None, 0.0, 0.0, None
        elif act and act[0] == "buy" and leg is None and cash > 0:
            leg = act[1]
            c = cost(leg, j)
            o = px[leg]["O"][j]
            q = cash * (1 - c) / o
            cost_paid += cash * c
            basis = cash
            entry = basis / q
            cash = 0.0
        eq = cash + (q * px[leg]["C"][j] if leg else 0.0)
        tot = eq + vault
        peak = max(peak, tot)
        mdd = min(mdd, tot / peak - 1)
    eq_end = cash + (q * px[leg]["C"][n - 1] * (1 - cost(leg, n - 1)) if leg else 0.0)
    return dict(vault_pct=100 * vault / CAPITAL, equity_pct=100 * eq_end / CAPITAL, total_pct=100 * ((vault + eq_end) / CAPITAL - 1), mdd_total=100 * mdd, sales=sales, inverse_sales=inv_sales, cost_pct=100 * cost_paid / CAPITAL, vault_by_year={y: round(v) for y, v in sorted(vault_by_year.items())})


def real_eval(pols):
    raw = json.load(open(os.path.join(os.path.dirname(HERE), "data", "kr_etf", "kodex_daily_raw.json"), encoding="utf-8"))[LONG]
    adj = json.load(open(CT.ADJ, encoding="utf-8"))[LONG]
    iv = json.load(open(INV_DAILY, encoding="utf-8"))[INV]
    imap = {d: (float(o), float(c)) for d, o, c in zip(iv["days"], iv["O"], iv["C"])}
    days = [d for d in raw["days"] if d <= REAL_TO]
    A = [float(x) for x in adj["C"][:len(days)]]
    e = {m: CT.ema(A, m) for m in (8, 21, 50, 200)}
    start = days.index(REAL_FROM) - 1
    keep = list(range(start, len(days)))
    if any(days[i] not in imap for i in keep):
        raise SystemExit("114800 missing a 069500 session in the window")
    sub = [days[i] for i in keep]
    px = {LONG: dict(O=[float(raw["O"][i]) for i in keep], C=[float(raw["C"][i]) for i in keep]), INV: dict(O=[imap[days[i]][0] for i in keep], C=[imap[days[i]][1] for i in keep])}
    pr = json.load(open(PRED_FILE, encoding="utf-8"))["preds"]
    bull = lambda i: e[8][keep[i]] > e[21][keep[i]] > e[50][keep[i]] > e[200][keep[i]]
    bear = lambda i: e[8][keep[i]] < e[21][keep[i]] < e[50][keep[i]] < e[200][keep[i]]
    pred = lambda i: tuple(pr[sub[i]]) if sub[i] in pr else (None, None)
    out = {p["name"]: run(p, px, lambda leg, j: LI.FEE + LI.TICK / px[leg]["O"][j], lambda j: sub[j][:4], 0, len(sub), pred, bull, bear) for p in pols}
    return out, dict(first_decision=sub[0], last_day=sub[-1])


def synth_eval(pols, models):
    rows = []
    for trend in SC.TRENDS:
        for case in SC.LIQ_CASES:
            for seed in range(SC.SEEDS):
                O, C, liq = SC.generate(trend, case, seed)
                IO, IC = SC.inverse_path(O, C)
                C_list = [float(x) for x in C]
                F = LI.features([None] * len(C_list), dict(Ca=C_list))
                idx = list(range(SC.WARM, len(C_list) - 1))
                X = np.asarray([F[i] for i in idx])
                pm = dict(zip(idx, zip((float(x) for x in models["long"].predict(X)), (float(x) for x in models["inv"].predict(X)))))
                e = {m: CT.ema(C_list, m) for m in (8, 21, 50, 200)}
                px = {LONG: dict(O=[float(x) for x in O], C=C_list), INV: dict(O=[float(x) for x in IO], C=[float(x) for x in IC])}
                res = {p["name"]: run(p, px, lambda leg, j: LI.FEE + liq[j]["cost_bp"] / 10000.0, lambda j: str((j - SC.WARM) // 250), SC.WARM, len(C_list), lambda i: pm.get(i, (None, None)), lambda i: e[8][i] > e[21][i] > e[50][i] > e[200][i], lambda i: e[8][i] < e[21][i] < e[50][i] < e[200][i]) for p in pols}
                rows.append(dict(trend=trend, liquidity=case, seed=seed, res=res))
    return rows


def summarize(rows, pols):
    out = {}
    for t in list(SC.TRENDS) + ["all"]:
        ps = [r for r in rows if t == "all" or r["trend"] == t]
        out[t] = {}
        for p in pols:
            n = p["name"]
            out[t][n] = dict(median_vault_pct=float(np.median([r["res"][n]["vault_pct"] for r in ps])), median_equity_pct=float(np.median([r["res"][n]["equity_pct"] for r in ps])), median_total_pct=float(np.median([r["res"][n]["total_pct"] for r in ps])), median_mdd=float(np.median([r["res"][n]["mdd_total"] for r in ps])), total_positive=int(sum(r["res"][n]["total_pct"] > 0 for r in ps)), total_beats_hold=int(sum(r["res"][n]["total_pct"] > r["res"]["hold"]["total_pct"] for r in ps)), median_sales=float(np.median([r["res"][n]["sales"] for r in ps])), paths=len(ps))
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
    real, info = real_eval(pols)
    models, _ = SH.load_models()
    rows = synth_eval(pols, models)
    summ = summarize(rows, pols)
    rec = dict(version="harvest-v2", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], policies=pols, real=dict(info=info, results=real), synthetic_summary=summ, synthetic_paths=rows)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print("REAL", info)
    for p in pols:
        n = p["name"]
        r = real[n]
        s = summ
        print(f"{n:26s} real vault {r['vault_pct']:6.1f}% total {r['total_pct']:7.1f}% mdd {r['mdd_total']:6.1f} sales {r['sales']:3d} inv {r['inverse_sales']:3d} | synth vault/total bull {s['bull'][n]['median_vault_pct']:5.1f}/{s['bull'][n]['median_total_pct']:6.1f} bear {s['bear'][n]['median_vault_pct']:5.1f}/{s['bear'][n]['median_total_pct']:6.1f} (+{s['bear'][n]['total_positive']}) side {s['sideways'][n]['median_vault_pct']:5.1f}/{s['sideways'][n]['median_total_pct']:6.1f} all {s['all'][n]['median_vault_pct']:5.1f}/{s['all'][n]['median_total_pct']:6.1f} >hold {s['all'][n]['total_beats_hold']}")
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
