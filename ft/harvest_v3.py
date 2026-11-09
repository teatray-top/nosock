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
import lightgbm as lgb
import learn_inverse as LI
import compare_trend_v2 as CT
import scenario_test_v1 as SC
import learn_shadow as SH
import short_head_v1 as SHV
import harvest_v2 as H2

CAPITAL = 100_000_000.0
VAULT_SHARE = 0.5
ER_RANGE = 0.3
RANGE_TP, RANGE_MAX_HOLD = 0.03, 5
TREND_TP, TREND_SL = 0.10, 0.10
LONG, INV = "069500", "114800"
SHORT_PRED = os.path.join(HERE, "results", "short_walkforward_predictions_v1.json")
SHORT_MODELS = os.path.join(HERE, "results", "short_models_v1")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def policies():
    return [dict(name="hold", kind="hold"), dict(name="A_long_tp10_sl10", kind="A"), dict(name="C_with_inverse_tp10_sl10", kind="C"), dict(name="E1_range_scalp_long", kind="E1"), dict(name="E2_range_scalp_both", kind="E2"), dict(name="E3_scalp_only_long", kind="E3"), dict(name="E4_scalp_only_both", kind="E4")]


def pick(cands, band):
    best = max([(0.0, None)] + cands, key=lambda x: x[0])
    return best[1] if best[0] > band else None


def run(pol, px, cost, year, start, n, g0, sh, er):
    cash, vault = CAPITAL, 0.0
    leg, q, basis, entry, since, mode = None, 0.0, 0.0, None, None, None
    sales, inv_sales, cost_paid, range_entries = 0, 0, 0.0, 0
    peak, mdd = CAPITAL, 0.0
    k = pol["kind"]
    for i in range(start, n - 1):
        j = i + 1
        act = None
        if k == "hold":
            if leg is None and i == start:
                act = ("buy", LONG, "trend")
        elif leg is not None:
            c_now = px[leg]["C"][i]
            if mode == "range":
                if c_now >= entry * (1 + RANGE_TP) or i - since >= RANGE_MAX_HOLD:
                    act = ("sell",)
            elif c_now >= entry * (1 + TREND_TP) or c_now <= entry * (1 - TREND_SL):
                act = ("sell",)
        else:
            pl, pi = g0(i)
            sl, si = sh(i)
            band_c = max(LI.BAND, 2 * cost(LONG, i))
            rng = er(i) is not None and er(i) < ER_RANGE
            tgt, md = None, None
            if k == "A" and pl is not None:
                tgt, md = (LONG if pl > LI.BAND else None), "trend"
            elif k == "C" and pl is not None:
                tgt, md = pick([(pl, LONG), (pi, INV)], LI.BAND), "trend"
            elif k in ("E1", "E2"):
                if rng and sl is not None:
                    tgt = pick([(sl, LONG)] + ([(si, INV)] if k == "E2" else []), band_c)
                    md = "range"
                elif pl is not None:
                    tgt = pick([(pl, LONG)] + ([(pi, INV)] if k == "E2" else []), LI.BAND)
                    md = "trend"
            elif k in ("E3", "E4") and sl is not None:
                tgt = pick([(sl, LONG)] + ([(si, INV)] if k == "E4" else []), band_c)
                md = "range"
            if tgt:
                act = ("buy", tgt, md)
        if act and act[0] == "sell":
            c = cost(leg, j)
            o = px[leg]["O"][j]
            proceeds = q * o * (1 - c)
            cost_paid += q * o * c
            profit = proceeds - basis
            tv = VAULT_SHARE * profit if profit > 0 else 0.0
            vault += tv
            cash += proceeds - tv
            sales += 1
            inv_sales += leg == INV
            leg, q, basis, entry, since, mode = None, 0.0, 0.0, None, None, None
        elif act and act[0] == "buy" and cash > 0:
            leg, mode = act[1], act[2]
            c = cost(leg, j)
            o = px[leg]["O"][j]
            q = cash * (1 - c) / o
            cost_paid += cash * c
            basis, entry, since = cash, cash / q, j
            range_entries += mode == "range"
            cash = 0.0
        eq = cash + (q * px[leg]["C"][j] if leg else 0.0)
        tot = eq + vault
        peak = max(peak, tot)
        mdd = min(mdd, tot / peak - 1)
    eq_end = cash + (q * px[leg]["C"][n - 1] * (1 - cost(leg, n - 1)) if leg else 0.0)
    return dict(vault_pct=100 * vault / CAPITAL, equity_pct=100 * eq_end / CAPITAL, total_pct=100 * ((vault + eq_end) / CAPITAL - 1), mdd_total=100 * mdd, sales=sales, inverse_sales=inv_sales, range_entries=range_entries, cost_pct=100 * cost_paid / CAPITAL)


def real_eval(pols):
    raw = json.load(open(os.path.join(os.path.dirname(HERE), "data", "kr_etf", "kodex_daily_raw.json"), encoding="utf-8"))[LONG]
    iv = json.load(open(H2.INV_DAILY, encoding="utf-8"))[INV]
    imap = {d: (float(o), float(c)) for d, o, c in zip(iv["days"], iv["O"], iv["C"])}
    days = [d for d in raw["days"] if d <= H2.REAL_TO]
    start = days.index(H2.REAL_FROM) - 1
    keep = list(range(start, len(days)))
    sub = [days[i] for i in keep]
    px = {LONG: dict(O=[float(raw["O"][i]) for i in keep], C=[float(raw["C"][i]) for i in keep]), INV: dict(O=[imap[days[i]][0] for i in keep], C=[imap[days[i]][1] for i in keep])}
    g0 = json.load(open(H2.PRED_FILE, encoding="utf-8"))["preds"]
    sp = json.load(open(SHORT_PRED, encoding="utf-8"))["preds"]
    out = {p["name"]: run(p, px, lambda leg, j: LI.FEE + LI.TICK / px[leg]["O"][j], None, 0, len(sub), lambda i: tuple(g0[sub[i]]) if sub[i] in g0 else (None, None), lambda i: (sp[sub[i]][0], sp[sub[i]][1]) if sub[i] in sp else (None, None), lambda i: sp[sub[i]][4] if sub[i] in sp else None) for p in pols}
    regime = sum(1 for d in sub if d in sp and sp[d][4] < ER_RANGE) / len(sub)
    return out, dict(first_decision=sub[0], last_day=sub[-1], share_range_days=regime)


def load_short_models():
    meta = json.load(open(os.path.join(SHORT_MODELS, "meta.json"), encoding="utf-8"))
    out = {}
    for name in ("long", "inv"):
        b = open(os.path.join(SHORT_MODELS, name + ".txt"), "rb").read()
        if hashlib.sha256(b).hexdigest() != meta["files"][name]:
            raise SystemExit("short model hash differs")
        out[name] = lgb.Booster(model_str=b.decode("utf-8"))
    return out


def synth_eval(pols, g0m, shm):
    rows = []
    for trend in SC.TRENDS:
        for case in SC.LIQ_CASES:
            for seed in range(SC.SEEDS):
                O, C, liq = SC.generate(trend, case, seed)
                IO, IC = SC.inverse_path(O, C)
                Cl = [float(x) for x in C]
                F21 = SHV.features21(Cl)
                idx = list(range(SC.WARM, len(Cl) - 1))
                X21 = np.asarray([F21[i] for i in idx])
                X17 = X21[:, :17]
                gp = dict(zip(idx, zip((float(x) for x in g0m["long"].predict(X17)), (float(x) for x in g0m["inv"].predict(X17)))))
                spd = dict(zip(idx, zip((float(x) for x in shm["long"].predict(X21)), (float(x) for x in shm["inv"].predict(X21)))))
                erd = {i: F21[i][19] for i in idx}
                px = {LONG: dict(O=[float(x) for x in O], C=Cl), INV: dict(O=[float(x) for x in IO], C=[float(x) for x in IC])}
                res = {p["name"]: run(p, px, lambda leg, j: LI.FEE + liq[j]["cost_bp"] / 10000.0, None, SC.WARM, len(Cl), lambda i: gp.get(i, (None, None)), lambda i: spd.get(i, (None, None)), lambda i: erd.get(i)) for p in pols}
                rows.append(dict(trend=trend, liquidity=case, seed=seed, share_range=float(np.mean([erd[i] < ER_RANGE for i in idx])), res=res))
    return rows


def summarize(rows, pols):
    out = {}
    for t in list(SC.TRENDS) + ["all"]:
        ps = [r for r in rows if t == "all" or r["trend"] == t]
        out[t] = dict(share_range_median=float(np.median([r["share_range"] for r in ps])))
        for p in pols:
            n = p["name"]
            out[t][n] = dict(median_vault_pct=float(np.median([r["res"][n]["vault_pct"] for r in ps])), median_total_pct=float(np.median([r["res"][n]["total_pct"] for r in ps])), median_mdd=float(np.median([r["res"][n]["mdd_total"] for r in ps])), total_positive=int(sum(r["res"][n]["total_pct"] > 0 for r in ps)), total_beats_hold=int(sum(r["res"][n]["total_pct"] > r["res"]["hold"]["total_pct"] for r in ps)), median_sales=float(np.median([r["res"][n]["sales"] for r in ps])), paths=len(ps))
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
    g0m, _ = SH.load_models()
    shm = load_short_models()
    rows = synth_eval(pols, g0m, shm)
    summ = summarize(rows, pols)
    rec = dict(version="harvest-v3", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], policies=pols, params=dict(er_range=ER_RANGE, range_tp=RANGE_TP, range_max_hold=RANGE_MAX_HOLD, trend_tp=TREND_TP, trend_sl=TREND_SL), real=dict(info=info, results=real), synthetic_summary=summ, synthetic_paths=rows)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print("REAL", info)
    for p in pols:
        n = p["name"]
        r = real[n]
        s = summ
        print(f"{n:26s} real vault {r['vault_pct']:6.1f}% total {r['total_pct']:7.1f}% mdd {r['mdd_total']:6.1f} sales {r['sales']:3d} range {r['range_entries']:3d} | synth vault/total bull {s['bull'][n]['median_vault_pct']:5.1f}/{s['bull'][n]['median_total_pct']:6.1f} bear {s['bear'][n]['median_vault_pct']:5.1f}/{s['bear'][n]['median_total_pct']:6.1f} side {s['sideways'][n]['median_vault_pct']:5.1f}/{s['sideways'][n]['median_total_pct']:6.1f} (+{s['sideways'][n]['total_positive']}) all {s['all'][n]['median_vault_pct']:5.1f}/{s['all'][n]['median_total_pct']:6.1f} >hold {s['all'][n]['total_beats_hold']}")
    print("range share median", {t: s[t]["share_range_median"] for t in s})
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
