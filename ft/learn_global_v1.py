import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import bisect
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
import learn_inverse_synth_v3 as V3
import compare_inverse as CI
import compare_trend_v2 as CT
import scenario_test_v1 as SC
import learn_shadow as SH

L, S = CI.L, CI.S
GLOBAL_DIR = os.path.join(os.path.dirname(HERE), "data", "global", "yf_20261107_161526")
OTHER_FROM = "2000-01-01"
OTHER_COST = 0.0002
CLIP = 0.5
NON_TRADABLE = ("^VIX", "^TNX")
GLOBAL_FEATS = ["gspc_r1", "gspc_r5", "gspc_r20", "ixic_r5", "vix_level", "vix_chg5", "krw_r5", "krw_r20", "gold_r20", "wti_r20", "btc_r5", "btc_r20", "tnx_chg20"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_global():
    man = json.load(open(os.path.join(GLOBAL_DIR, "manifest.json"), encoding="utf-8"))
    out = {}
    for t, info in man["files"].items():
        if "file" not in info:
            continue
        p = os.path.join(GLOBAL_DIR, info["file"])
        if sha(p) != info["sha256"]:
            raise SystemExit(f"global file hash differs {t}")
        out[t] = json.load(open(p, encoding="utf-8"))
    return out


class Asof:
    def __init__(self, rec):
        self.days = rec["days"]
        self.C = rec["C"]

    def idx(self, d):
        return bisect.bisect_left(self.days, d) - 1

    def ret(self, d, n):
        i = self.idx(d)
        if i - n < 0:
            return float("nan")
        return math.log(self.C[i] / self.C[i - n])

    def level(self, d):
        i = self.idx(d)
        return self.C[i] if i >= 0 else float("nan")

    def diff(self, d, n):
        i = self.idx(d)
        if i - n < 0:
            return float("nan")
        return self.C[i] - self.C[i - n]


def global_features(g, d):
    sp, nq, vx, kr, au, oil, btc, tn = (g[t] for t in ("^GSPC", "^IXIC", "^VIX", "KRW=X", "GC=F", "CL=F", "BTC-USD", "^TNX"))
    lv = vx.level(d)
    return [sp.ret(d, 1), sp.ret(d, 5), sp.ret(d, 20), nq.ret(d, 5), lv, vx.ret(d, 5), kr.ret(d, 5), kr.ret(d, 20), au.ret(d, 20), oil.ret(d, 20), btc.ret(d, 5), btc.ret(d, 20), tn.diff(d, 20)]


def other_rows(recs, g):
    rows = []
    stats = {}
    for t, rec in recs.items():
        if t in NON_TRADABLE:
            continue
        days, O, C = rec["days"], rec["O"], rec["C"]
        F = LI.features([None] * len(C), dict(Ca=C))
        n_clip = n = 0
        for i in range(LI.FEAT_START, len(C) - LI.H - 1):
            if days[i] < OTHER_FROM or i not in F:
                continue
            o1, o2 = O[i + 1], O[i + LI.H + 1]
            if not (o1 > 0 and o2 > 0):
                continue
            yl = o2 * (1 - OTHER_COST) / (o1 * (1 + OTHER_COST)) - 1
            yi = -(o2 / o1 - 1) - 2 * OTHER_COST
            if abs(yl) > CLIP or abs(yi) > CLIP:
                n_clip += 1
            yl, yi = max(-CLIP, min(CLIP, yl)), max(-CLIP, min(CLIP, yi))
            f = F[i]
            if not all(math.isfinite(v) for v in f):
                continue
            rows.append((t, days[i], days[i + LI.H + 1], f, global_features(g, days[i]), yl, yi))
            n += 1
        stats[t] = dict(rows=n, clipped=n_clip)
    return rows, stats


def fit_predict(days, k, inv, F, G, pool_X, pool_yl, pool_R, others, use_others, use_global):
    ng = len(GLOBAL_FEATS)
    nan_g = np.full((len(pool_X), ng), np.nan)
    preds, info, models, cals = {}, [], {}, {}
    if others:
        o_X = np.asarray([r[3] for r in others])
        o_G = np.asarray([r[4] for r in others])
        o_yl = np.asarray([r[5] for r in others])
        o_yi = np.asarray([r[6] for r in others])
        o_end = np.asarray([r[2] for r in others])
    for f in LI.folds(days):
        jl = list(range(LI.FEAT_START, f["train_last"] + 1))
        js = [j for j in jl if LI.label_inv(days, inv, j) is not None]
        cal = V3.calibrate(days, k, LI.INV_DIR_DEFAULT, f["cutoff_day"])
        cals[f["year"]] = cal
        pool_yi = V3.inv_labels_from_ratio(pool_R, cal)
        XL = [np.asarray([F[j] for j in jl]), pool_X]
        GL = [np.asarray([G[j] for j in jl]), nan_g]
        yL = [np.asarray([LI.label_long(k, j) for j in jl]), pool_yl]
        XS = [np.asarray([F[j] for j in js]), pool_X]
        GS = [np.asarray([G[j] for j in js]), nan_g]
        yS = [np.asarray([LI.label_inv(days, inv, j) for j in js]), pool_yi]
        n_other = 0
        if use_others:
            m = o_end < f["cutoff_day"]
            n_other = int(m.sum())
            XL.append(o_X[m]); GL.append(o_G[m]); yL.append(o_yl[m])
            XS.append(o_X[m]); GS.append(o_G[m]); yS.append(o_yi[m])
        Xl, Xs = np.vstack(XL), np.vstack(XS)
        if use_global:
            Xl, Xs = np.hstack([Xl, np.vstack(GL)]), np.hstack([Xs, np.vstack(GS)])
        ml, ms = LI.fit(Xl, np.concatenate(yL)), LI.fit(Xs, np.concatenate(yS))
        dec = list(range(f["first_dec"], f["last_dec"] + 1))
        Xd = np.asarray([F[i] for i in dec])
        if use_global:
            Xd = np.hstack([Xd, np.asarray([G[i] for i in dec])])
        pl, ps = ml.predict(Xd), ms.predict(Xd)
        for i, a, b in zip(dec, pl, ps):
            preds[days[i]] = (float(a), float(b))
        info.append(dict(year=f["year"], cutoff_day=f["cutoff_day"], rows_long=int(len(Xl)), rows_inv=int(len(Xs)), other_rows=n_other, pred_long_mean=float(pl.mean()), pred_long_pos=float((pl > 0).mean()), pred_inv_pref=float(((ps > pl) & (ps > 0)).mean())))
        models[f["year"]] = (ml, ms)
    return preds, info, models


def scenario_eval(models, use_global):
    out = []
    for trend in SC.TRENDS:
        for case in SC.LIQ_CASES:
            for seed in range(SC.SEEDS):
                O, C, liq = SC.generate(trend, case, seed)
                IO, IC = SC.inverse_path(O, C)
                F = LI.features([None] * len(C), dict(Ca=list(C)))
                rows = list(range(SC.WARM, len(C) - 1))
                X = np.asarray([F[i] for i in rows])
                if use_global:
                    X = np.hstack([X, np.full((len(rows), len(GLOBAL_FEATS)), np.nan)])
                pl, ps = models[0].predict(X), models[1].predict(X)
                tl, ti, sl, si = {}, {}, None, None
                for i, a, b in zip(rows, pl, ps):
                    sl = SH.policy_step(sl, float(a), float(b), False)
                    si = SH.policy_step(si, float(a), float(b), True)
                    tl[i], ti[i] = sl, si
                hold = SC.ledger(O, C, IO, IC, liq, {i: L for i in rows})
                out.append(dict(trend=trend, liquidity=case, seed=seed, hold=hold["ret"], learn_long=SC.ledger(O, C, IO, IC, liq, tl), learn_inv=SC.ledger(O, C, IO, IC, liq, ti)))
    summ = {}
    for t in list(SC.TRENDS) + ["all"]:
        ps = [p for p in out if t == "all" or p["trend"] == t]
        summ[t] = {m: dict(median_ret=float(np.median([p[m]["ret"] for p in ps])), median_mdd=float(np.median([p[m]["mdd"] for p in ps])), median_trades=float(np.median([p[m]["trades"] for p in ps])), beats_hold=int(sum(p[m]["ret"] > p["hold"] for p in ps))) for m in ("learn_long", "learn_inv")}
        summ[t]["hold_median"] = float(np.median([p["hold"] for p in ps]))
    return summ


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, p in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), p)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    days, k, inv = LI.load(LI.INV_DIR_DEFAULT)
    recs = load_global()
    g = {t: Asof(r) for t, r in recs.items()}
    F = LI.features(days, k)
    G = {i: global_features(g, days[i]) for i in F}
    man = json.load(open(os.path.join(V3.PATHS, "manifest.json"), encoding="utf-8"))
    train = [V3.load_path(e) for e in man["paths"] if e["split"] == "train"]
    pool_X = np.vstack([p["X"] for p in train])
    pool_yl = np.concatenate([p["yl"] for p in train])
    pool_R = np.concatenate([p["R"] for p in train])
    others, ostats = other_rows(recs, g)
    variants = {"G0": (False, False), "G1": (True, False), "G2": (True, True)}
    built = {n: fit_predict(days, k, inv, F, G, pool_X, pool_yl, pool_R, others, *v) for n, v in variants.items()}
    p0 = built["G0"][0]["2026-11-05"]
    parity = p0 == (-0.00542095336063337, 0.025339779548631997)
    tg_rule, all_days = CI.targets()
    tg = {"hold": tg_rule["hold"], "ribbon": tg_rule["ribbon"]}
    for n in variants:
        tg[n + "_long"] = LI.policy(built[n][0], False)
        tg[n + "_inv"] = LI.policy(built[n][0], True)
    methods = list(tg)
    prev = {d: all_days[i - 1] for i, d in enumerate(all_days) if i > 0}
    cal = json.load(open(CT.CAL, encoding="utf-8"))["days"]
    msess = {L: CI.minute_sessions(CT.SNAP[L][0], cal), S: CI.minute_sessions(os.path.join(LI.INV_DIR_DEFAULT, S + ".csv"), cal)}
    res = {}

    def block(days_b, sess, fill_of):
        sims = {m: CI.simulate(days_b, sess, (lambda d, m=m: tg[m][prev[d]]), fill_of) for m in methods}
        hf = sims["hold"]["final"]
        return dict(window=[days_b[0], days_b[-1]], sessions=len(days_b), methods={m: CI.report(sims[m], hf) for m in methods})

    for name, (a, b) in CT.WINDOWS.items():
        dd = [d for d in sorted(msess[L]) if a <= d <= b and prev.get(d) in built["G0"][0]]
        res[f"minute_{name}"] = block(dd, msess, lambda leg, d, after: CI.first_fill(msess[leg][d], after))
    kr = json.load(open(CI.RAW, encoding="utf-8"))[L]
    ir = json.load(open(os.path.join(LI.INV_DIR_DEFAULT, "daily_raw.json"), encoding="utf-8"))[S]
    dsess = {L: {}, S: {}}
    for sym, src in ((L, kr), (S, ir)):
        for i, d in enumerate(src["days"]):
            if d <= LI.LAST:
                dsess[sym][d] = dict(open=float(src["O"][i]), close=float(src["C"][i]), vol=float(src["V"][i]))
    ddays = sorted(d for d in set(dsess[L]) & set(dsess[S]) if prev.get(d) in built["G0"][0])
    res["daily_open"] = block(ddays, dsess, lambda leg, d, after: (dsess[leg][d]["open"], "09:00") if dsess[leg][d]["vol"] > 0 and dsess[leg][d]["open"] > 0 else None)
    idx = {d: i for i, d in enumerate(days)}
    oos = {}
    for n in variants:
        pr = built[n][0]
        o = np.asarray([(pr[d][0], LI.label_long(k, idx[d])) for d in sorted(pr) if idx[d] + LI.H + 1 < len(days)])
        oos[n] = dict(corr=float(np.corrcoef(o[:, 0], o[:, 1])[0, 1]), sign_hit=float(np.mean((o[:, 0] > 0) == (o[:, 1] > 0))), base=float(np.mean(o[:, 1] > 0)))
    scen = {n: scenario_eval(built[n][2][LI.LAST_YEAR], variants[n][1]) for n in variants}
    rec = dict(version="learn-global-v1", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], g0_parity_with_v3=parity, other_assets=ostats, other_rows_total=len(others), global_features=GLOBAL_FEATS, folds={n: built[n][1] for n in variants}, oos=oos, scenario=scen, latest={n: dict(day=max(built[n][0]), pred=built[n][0][max(built[n][0])], long=tg[n + "_long"][max(built[n][0])], inv=tg[n + "_inv"][max(built[n][0])]) for n in variants}, results=res)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print("G0 parity with v3", parity, "other rows", len(others))
    print("oos", json.dumps(oos))
    for kk, v in res.items():
        print(f"== {kk} {v['window'][0]}..{v['window'][1]}")
        for m in methods:
            r = v["methods"][m]
            print(f"  {m:9s} return {r['return_pct']:8.2f}% maxDD {r['max_drawdown_daily_pct']:7.2f}% buys {r['buys']:3d} inv {r['inverse_buys']:3d}")
    for n in variants:
        print("scenario", n, json.dumps({t: {m: (round(v[m]['median_ret'], 1), v[m]['beats_hold']) for m in ('learn_long', 'learn_inv')} for t, v in scen[n].items()}))
    print("latest", rec["latest"])
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
