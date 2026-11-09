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
TPS = (0.05, 0.10, 0.20)
DIPS = (0.03, 0.05, 0.10)
SLS = (None, 0.10)
REAL_FROM, REAL_TO = "2013-01-02", "2026-11-05"
PRED_FILE = os.path.join(HERE, "results", "g0_walkforward_predictions_v1.json")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def policies():
    out = [dict(name="hold", kind="hold"), dict(name="annual_harvest", kind="annual")]
    out += [dict(name=f"harvest_tp{int(t * 100)}", kind="rebuy", tp=t) for t in TPS]
    out += [dict(name="ribbon", kind="ribbon")]
    out += [dict(name=f"price_tp{int(t * 100)}_dip{int(d * 100)}_sl{int(s * 100) if s else 0}", kind="price", tp=t, dip=d, sl=s) for t in TPS for d in DIPS for s in SLS]
    out += [dict(name=f"model_tp{int(t * 100)}_sl{int(s * 100) if s else 0}", kind="model", tp=t, sl=s) for t in TPS for s in SLS]
    return out


def run(pol, O, C, cost, year, start, pred, ribbon):
    cash, q, basis, vault = CAPITAL, 0.0, 0.0, 0.0
    entry_px, last_sell = None, None
    vault_by_year, trips, cost_paid = {}, 0, 0.0
    peak_tot, mdd_tot, peak_eq, mdd_eq = CAPITAL, 0.0, CAPITAL, 0.0
    hw = CAPITAL
    n = len(C)

    def buy(j):
        nonlocal cash, q, basis, entry_px
        c = cost(j)
        q = cash * (1 - c) / O[j]
        basis = cash
        cost_paid_add = cash * c
        cash = 0.0
        entry_px = basis / q
        return cost_paid_add

    def sell(j, frac=1.0):
        nonlocal cash, q, basis, vault, last_sell, trips
        c = cost(j)
        qs = q * frac
        proceeds = qs * O[j] * (1 - c)
        b = basis * frac
        profit = proceeds - b
        to_vault = VAULT_SHARE * profit if profit > 0 else 0.0
        vault += to_vault
        vault_by_year[year(j)] = vault_by_year.get(year(j), 0.0) + to_vault
        cash += proceeds - to_vault
        q -= qs
        basis -= b
        last_sell = O[j]
        trips += 1
        return qs * O[j] * c

    for i in range(start, n - 1):
        j = i + 1
        act = None
        k = pol["kind"]
        if k == "hold":
            act = "buy" if q == 0 and i == start else None
        elif k == "annual":
            if q == 0 and i == start:
                act = "buy"
            elif year(i) != year(j) and q > 0:
                act = "annual"
        elif k == "rebuy":
            if q == 0:
                act = "buy"
            elif C[i] >= entry_px * (1 + pol["tp"]):
                act = "rebuy"
        elif k == "ribbon":
            sig = ribbon(i)
            if q == 0 and sig:
                act = "buy"
            elif q > 0 and not sig:
                act = "sell"
        elif k == "price":
            if q == 0:
                if last_sell is None or C[i] <= last_sell * (1 - pol["dip"]):
                    act = "buy"
            elif C[i] >= entry_px * (1 + pol["tp"]) or (pol["sl"] and C[i] <= entry_px * (1 - pol["sl"])):
                act = "sell"
        elif k == "model":
            if q == 0:
                if pred(i) is not None and pred(i) > LI.BAND:
                    act = "buy"
            elif C[i] >= entry_px * (1 + pol["tp"]) or (pol["sl"] and C[i] <= entry_px * (1 - pol["sl"])):
                act = "sell"
        if act == "buy" and cash > 0:
            cost_paid += buy(j)
        elif act == "sell":
            cost_paid += sell(j)
        elif act == "rebuy":
            cost_paid += sell(j)
            cost_paid += buy(j)
        elif act == "annual":
            eq = cash + q * O[j]
            gain = eq - hw
            if gain > 0:
                frac = min(1.0, VAULT_SHARE * gain / (q * O[j]))
                c = cost(j)
                proceeds = q * frac * O[j] * (1 - c)
                vault += proceeds
                vault_by_year[year(j)] = vault_by_year.get(year(j), 0.0) + proceeds
                cost_paid += q * frac * O[j] * c
                basis *= (1 - frac)
                q *= (1 - frac)
                trips += 1
            hw = cash + q * O[j]
        eq = cash + q * C[j]
        tot = eq + vault
        peak_eq, peak_tot = max(peak_eq, eq), max(peak_tot, tot)
        mdd_eq, mdd_tot = min(mdd_eq, eq / peak_eq - 1), min(mdd_tot, tot / peak_tot - 1)
    eq_end = cash + q * C[n - 1] * (1 - cost(n - 1))
    return dict(vault=vault, equity=eq_end, total=vault + eq_end, vault_pct=100 * vault / CAPITAL, total_pct=100 * ((vault + eq_end) / CAPITAL - 1), mdd_total=100 * mdd_tot, mdd_equity=100 * mdd_eq, trips=trips, cost_pct=100 * cost_paid / CAPITAL, vault_by_year={y: round(v) for y, v in sorted(vault_by_year.items())})


def real_eval(pols):
    raw = json.load(open(os.path.join(os.path.dirname(HERE), "data", "kr_etf", "kodex_daily_raw.json"), encoding="utf-8"))["069500"]
    adj = json.load(open(CT.ADJ, encoding="utf-8"))["069500"]
    days = [d for d in raw["days"] if d <= REAL_TO]
    O = [float(x) for x in raw["O"][:len(days)]]
    C = [float(x) for x in raw["C"][:len(days)]]
    A = [float(x) for x in adj["C"][:len(days)]]
    e = {m: CT.ema(A, m) for m in (8, 21, 50, 200)}
    pr = json.load(open(PRED_FILE, encoding="utf-8"))["preds"]
    start = days.index(REAL_FROM) - 1
    out = {}
    for p in pols:
        out[p["name"]] = run(p, O, C, lambda j: LI.FEE + LI.TICK / O[j], lambda j: days[j][:4], start, lambda i: pr[days[i]][0] if days[i] in pr else None, lambda i: e[8][i] > e[21][i] > e[50][i] > e[200][i])
    return out, dict(first_decision=days[start], last_day=days[-1])


def synth_eval(pols, models):
    rows = []
    for trend in SC.TRENDS:
        for case in SC.LIQ_CASES:
            for seed in range(SC.SEEDS):
                O, C, liq = SC.generate(trend, case, seed)
                O, C = [float(x) for x in O], [float(x) for x in C]
                F = LI.features([None] * len(C), dict(Ca=C))
                idx = list(range(SC.WARM, len(C) - 1))
                pl = models["long"].predict(np.asarray([F[i] for i in idx]))
                pm = dict(zip(idx, (float(x) for x in pl)))
                e = {m: CT.ema(C, m) for m in (8, 21, 50, 200)}
                res = {p["name"]: run(p, O, C, lambda j: LI.FEE + liq[j]["cost_bp"] / 10000.0, lambda j: str((j - SC.WARM) // 250), SC.WARM, lambda i: pm.get(i), lambda i: e[8][i] > e[21][i] > e[50][i] > e[200][i]) for p in pols}
                rows.append(dict(trend=trend, liquidity=case, seed=seed, res=res))
    return rows


def summarize(rows, pols):
    out = {}
    for t in list(SC.TRENDS) + ["all"]:
        ps = [r for r in rows if t == "all" or r["trend"] == t]
        out[t] = {}
        for p in pols:
            n = p["name"]
            v = [r["res"][n]["vault_pct"] for r in ps]
            tot = [r["res"][n]["total_pct"] for r in ps]
            out[t][n] = dict(median_vault_pct=float(np.median(v)), median_total_pct=float(np.median(tot)), median_mdd_total=float(np.median([r["res"][n]["mdd_total"] for r in ps])), median_trips=float(np.median([r["res"][n]["trips"] for r in ps])), total_beats_hold=int(sum(r["res"][n]["total_pct"] > r["res"]["hold"]["total_pct"] for r in ps)), vault_positive=int(sum(x > 0 for x in v)), paths=len(ps))
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
    real, real_info = real_eval(pols)
    models, _ = SH.load_models()
    rows = synth_eval(pols, models)
    summ = summarize(rows, pols)
    rec = dict(version="harvest-v1", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], policies=pols, real=dict(info=real_info, results=real), synthetic_summary=summ, synthetic_paths=rows)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print("REAL", real_info)
    for p in pols:
        r = real[p["name"]]
        a = summ["all"][p["name"]]
        print(f"{p['name']:24s} real vault {r['vault_pct']:7.1f}% total {r['total_pct']:7.1f}% mdd {r['mdd_total']:6.1f} trips {r['trips']:3d} | synth all vault {a['median_vault_pct']:6.1f}% total {a['median_total_pct']:6.1f}% >hold {a['total_beats_hold']:3d} | bull {summ['bull'][p['name']]['median_vault_pct']:5.1f}/{summ['bull'][p['name']]['median_total_pct']:6.1f} bear {summ['bear'][p['name']]['median_vault_pct']:5.1f}/{summ['bear'][p['name']]['median_total_pct']:6.1f} side {summ['sideways'][p['name']]['median_vault_pct']:5.1f}/{summ['sideways'][p['name']]['median_total_pct']:6.1f}")
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
