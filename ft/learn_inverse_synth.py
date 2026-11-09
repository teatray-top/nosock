import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import learn_inverse as LI
import compare_inverse as CI
import compare_trend_v2 as CT

L, S = CI.L, CI.S
PATHS = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work\scenario_paths_v1_20261106")
CAL_FROM = "2013-01-02"
CAPITAL = 100_000_000.0


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def calibrate(days, k, inv_dir):
    iv = json.load(open(os.path.join(inv_dir, "daily_raw.json"), encoding="utf-8"))[S]
    ic = {d: float(c) for d, c in zip(iv["days"], iv["C"]) if d <= LI.LAST}
    resid, closes = [], []
    for i in range(1, len(days)):
        d0, d1 = days[i - 1], days[i]
        ex_date = abs((k["Ca"][i] / k["C"][i]) / (k["Ca"][i - 1] / k["C"][i - 1]) - 1) > 1e-6
        if d1 >= CAL_FROM and d0 in ic and d1 in ic and not ex_date:
            resid.append((ic[d1] / ic[d0] - 1) + (k["C"][i] / k["C"][i - 1] - 1))
            closes.append(ic[d1])
    return float(np.mean(resid)), float(np.median(closes))


def synth_inverse(O, C, drag):
    n = len(C)
    IO, IC = np.empty(n), np.empty(n)
    IO[0] = IC[0] = 1.0
    for t in range(1, n):
        IO[t] = IC[t - 1] * (2 - O[t] / C[t - 1]) * (1 + drag)
        IC[t] = IC[t - 1] * (2 - C[t] / C[t - 1]) * (1 + drag)
    return IO, IC


def inv_label(IO, i, tr):
    return IO[i + LI.H + 1] * (1 - tr) * (1 - LI.FEE) / (IO[i + 1] * (1 + tr) * (1 + LI.FEE)) - 1


def load_path(entry, drag, tr):
    p = os.path.join(PATHS, entry["observed_file"])
    if sha(p) != entry["observed_sha256"]:
        raise SystemExit(f"path hash differs {entry['path_id']}")
    d = np.load(p, allow_pickle=False)
    O, C = [float(x) for x in d["O"]], [float(x) for x in d["C"]]
    k = dict(O=O, C=C, Oa=O, Ca=C)
    F = LI.features([None] * len(C), k)
    IO, IC = synth_inverse(O, C, drag)
    rows = range(LI.FEAT_START, len(C) - LI.H - 1)
    return dict(id=entry["path_id"], k=k, F=F, IO=IO, IC=IC, X=np.asarray([F[i] for i in rows]), yl=np.asarray([LI.label_long(k, i) for i in rows]), yi=np.asarray([inv_label(IO, i, tr) for i in rows]))


def build_mixed(days, k, inv, pool_X, pool_yl, pool_yi, ratio):
    F = LI.features(days, k)
    preds, info, models = {}, [], {}
    for f in LI.folds(days):
        jl = list(range(LI.FEAT_START, f["train_last"] + 1))
        js = [j for j in jl if LI.label_inv(days, inv, j) is not None]
        rng = np.random.default_rng([20261107, f["year"], 0 if ratio is None else ratio])
        if ratio is None:
            a = b = np.arange(len(pool_X))
        else:
            a = rng.choice(len(pool_X), size=min(len(pool_X), ratio * len(jl)), replace=False)
            b = rng.choice(len(pool_X), size=min(len(pool_X), ratio * len(js)), replace=False)
        Xl = np.vstack([np.asarray([F[j] for j in jl]), pool_X[a]])
        yl = np.concatenate([[LI.label_long(k, j) for j in jl], pool_yl[a]])
        Xs = np.vstack([np.asarray([F[j] for j in js]), pool_X[b]])
        ys = np.concatenate([[LI.label_inv(days, inv, j) for j in js], pool_yi[b]])
        ml, ms = LI.fit(Xl, yl), LI.fit(Xs, ys)
        dec = list(range(f["first_dec"], f["last_dec"] + 1))
        Xd = np.asarray([F[i] for i in dec])
        pl, ps = ml.predict(Xd), ms.predict(Xd)
        for i, x, y in zip(dec, pl, ps):
            preds[days[i]] = (float(x), float(y))
        info.append(dict(year=f["year"], real_long=len(jl), real_inv=len(js), synth_long=len(a), synth_inv=len(b), synth_idx_sha=hashlib.sha256(np.concatenate([a, b]).astype(np.int64).tobytes()).hexdigest(), pred_long_mean=float(pl.mean()), pred_long_pos=float((pl > 0).mean()), pred_inv_pref=float(((ps > pl) & (ps > 0)).mean())))
        models[f["year"]] = (ml, ms)
    return preds, info, models


def real_only_models(days, k, inv):
    F = LI.features(days, k)
    out = {}
    for f in LI.folds(days):
        if f["year"] != LI.LAST_YEAR:
            continue
        jl = list(range(LI.FEAT_START, f["train_last"] + 1))
        js = [j for j in jl if LI.label_inv(days, inv, j) is not None]
        out = (LI.fit(np.asarray([F[j] for j in jl]), np.asarray([LI.label_long(k, j) for j in jl])), LI.fit(np.asarray([F[j] for j in js]), np.asarray([LI.label_inv(days, inv, j) for j in js])))
    return out


def synth_ledger(path, targets, tr):
    O, C, IO, IC = path["k"]["O"], path["k"]["C"], path["IO"], path["IC"]
    cash, ql, qi = CAPITAL, 0.0, 0.0
    peak, mdd = CAPITAL, 0.0
    for i in range(LI.FEAT_START, len(C) - 1):
        t = targets[i]
        j = i + 1
        if ql > 0 and t != L:
            cash += ql * (O[j] - LI.TICK) * (1 - LI.FEE)
            ql = 0.0
        if qi > 0 and t != S:
            cash += qi * IO[j] * (1 - tr) * (1 - LI.FEE)
            qi = 0.0
        if t == L and ql == 0:
            ql = cash / ((O[j] + LI.TICK) * (1 + LI.FEE))
            cash = 0.0
        if t == S and qi == 0:
            qi = cash / (IO[j] * (1 + tr) * (1 + LI.FEE))
            cash = 0.0
        v = cash + ql * C[j] + qi * IC[j]
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    n = len(C) - 1
    final = cash + ql * (C[n] - LI.TICK) * (1 - LI.FEE) + qi * IC[n] * (1 - tr) * (1 - LI.FEE)
    return 100 * (final / CAPITAL - 1), 100 * mdd


def path_policy(pl, ps, allow_inv):
    out, state = {}, None
    for i in range(len(pl)):
        v = {None: 0.0, L: pl[i]}
        if allow_inv:
            v[S] = ps[i]
        best = max(v, key=lambda a: v[a])
        if state in v and v[state] >= v[best] - LI.BAND:
            best = state
        out[i + LI.FEAT_START] = best
        state = best
    return out


def main(out_path, spec_path, spec_sha, inv_dir=LI.INV_DIR_DEFAULT):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for p, key in ((os.path.join(HERE, "learn_inverse.py"), "learn_inverse_sha256"), (os.path.join(PATHS, "manifest.json"), "paths_manifest_sha256"), (CI.RAW, "kodex_daily_raw_sha256"), (LI.ADJ_FILE, "kodex_daily_adj_sha256"), (os.path.join(inv_dir, "daily_raw.json"), "inverse_daily_raw_sha256"), (os.path.join(inv_dir, S + ".csv"), "inverse_minute_sha256"), (os.path.join(HERE, "compare_inverse.py"), "compare_inverse_sha256")):
        if sha(p) != spec["inputs"][key]:
            raise SystemExit(f"input hash differs: {key}")
    days, k, inv = LI.load(inv_dir)
    drag, pref = calibrate(days, k, inv_dir)
    tr = LI.TICK / pref
    man = json.load(open(os.path.join(PATHS, "manifest.json"), encoding="utf-8"))
    train = [load_path(e, drag, tr) for e in man["paths"] if e["split"] == "train"]
    test = [load_path(e, drag, tr) for e in man["paths"] if e["split"] == "test"]
    p0 = train[0]
    rc = np.diff(np.asarray(p0["k"]["C"])) / np.asarray(p0["k"]["C"][:-1])
    ri = np.diff(p0["IC"]) / p0["IC"][:-1]
    inv_check = dict(path=p0["id"], corr_daily=float(np.corrcoef(rc, ri)[0, 1]), mean_sum_daily=float(np.mean(rc + ri)), index_total=float(p0["k"]["C"][-1] / p0["k"]["C"][0] - 1), inverse_total=float(p0["IC"][-1] / p0["IC"][0] - 1))
    print("synthetic inverse check", inv_check)
    pool_X = np.vstack([p["X"] for p in train])
    pool_yl = np.concatenate([p["yl"] for p in train])
    pool_yi = np.concatenate([p["yi"] for p in train])
    variants = {"B5": 5, "Ball": None}
    built = {name: build_mixed(days, k, inv, pool_X, pool_yl, pool_yi, r) for name, r in variants.items()}
    predsB = built["B5"][0]
    predsA, _ = LI.build(days, k, inv)
    tg_rule, all_days = CI.targets()
    tg = {"hold": tg_rule["hold"], "ribbon": tg_rule["ribbon"], "ribbon_inv": tg_rule["ribbon_inv"], "A_learn_inv": LI.policy(predsA, True)}
    for name in variants:
        tg[name + "_learn_long"] = LI.policy(built[name][0], False)
        tg[name + "_learn_inv"] = LI.policy(built[name][0], True)
    methods = list(tg)
    prev = {d: all_days[i - 1] for i, d in enumerate(all_days) if i > 0}
    cal = json.load(open(CT.CAL, encoding="utf-8"))["days"]
    msess = {L: CI.minute_sessions(CT.SNAP[L][0], cal), S: CI.minute_sessions(os.path.join(inv_dir, S + ".csv"), cal)}
    res = {}

    def block(days_b, sess, fill_of):
        sims = {m: CI.simulate(days_b, sess, (lambda d, m=m: tg[m][prev[d]]), fill_of) for m in methods}
        hf = sims["hold"]["final"]
        return dict(window=[days_b[0], days_b[-1]], sessions=len(days_b), methods={m: CI.report(sims[m], hf) for m in methods})

    for name, (a, b) in CT.WINDOWS.items():
        dd = [d for d in sorted(msess[L]) if a <= d <= b and prev.get(d) in predsB]
        res[f"minute_{name}"] = block(dd, msess, lambda leg, d, after: CI.first_fill(msess[leg][d], after))
    kr = json.load(open(CI.RAW, encoding="utf-8"))[L]
    ir = json.load(open(os.path.join(inv_dir, "daily_raw.json"), encoding="utf-8"))[S]
    dsess = {L: {}, S: {}}
    for sym, src in ((L, kr), (S, ir)):
        for i, d in enumerate(src["days"]):
            if d <= LI.LAST:
                dsess[sym][d] = dict(open=float(src["O"][i]), close=float(src["C"][i]), vol=float(src["V"][i]))
    ddays = sorted(d for d in set(dsess[L]) & set(dsess[S]) if prev.get(d) in predsB)
    res["daily_open"] = block(ddays, dsess, lambda leg, d, after: (dsess[leg][d]["open"], "09:00") if dsess[leg][d]["vol"] > 0 and dsess[leg][d]["open"] > 0 else None)
    idx = {d: i for i, d in enumerate(days)}
    oos_all = {}
    for name, pr in (("A", predsA), ("B5", built["B5"][0]), ("Ball", built["Ball"][0])):
        o = np.asarray([(pr[d][0], LI.label_long(k, idx[d])) for d in sorted(pr) if idx[d] + LI.H + 1 < len(days)])
        oos_all[name] = dict(corr=float(np.corrcoef(o[:, 0], o[:, 1])[0, 1]), sign_hit=float(np.mean((o[:, 0] > 0) == (o[:, 1] > 0))), base=float(np.mean(o[:, 1] > 0)), n=int(len(o)))
    models = {"A": real_only_models(days, k, inv), "B5": built["B5"][2][LI.LAST_YEAR], "Ball": built["Ball"][2][LI.LAST_YEAR]}
    synth = []
    for p in test:
        rows = sorted(p["F"])
        X = np.asarray([p["F"][i] for i in rows])
        out = dict(path=p["id"])
        out["hold"] = synth_ledger(p, {i: L for i in rows}, tr)
        for name, (ml, ms) in models.items():
            pl, ps = ml.predict(X), ms.predict(X)
            out[name + "_inv"] = synth_ledger(p, path_policy(pl, ps, True), tr)
            out[name + "_long"] = synth_ledger(p, path_policy(pl, ps, False), tr)
        synth.append(out)
    summ = {}
    for m in [n + s for n in models for s in ("_inv", "_long")]:
        r = np.asarray([s[m][0] for s in synth])
        h = np.asarray([s["hold"][0] for s in synth])
        summ[m] = dict(beats_hold=int((r > h).sum()), median_return=float(np.median(r)), median_hold=float(np.median(h)), median_mdd=float(np.median([s[m][1] for s in synth])), median_hold_mdd=float(np.median([s["hold"][1] for s in synth])))
    rec = dict(version="learn-inverse-synth-v1", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], calibration=dict(inverse_daily_drag=drag, inverse_ref_price=pref, inverse_tick_ratio=tr), inverse_check=inv_check, synth_pool_rows=int(len(pool_X)), variants=variants, folds={n: built[n][1] for n in variants}, oos=oos_all, synthetic_test=dict(paths=len(synth), summary=summ, per_path=synth), latest={n: dict(day=max(built[n][0]), pred=built[n][0][max(built[n][0])], learn_inv=tg[n + "_learn_inv"][max(built[n][0])]) for n in variants}, results=res)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print(f"calibration drag/day {drag:+.6f} ref price {pref:.0f} tick ratio {tr:.5f}; synth pool rows {len(pool_X)}")
    for n in variants:
        for fi in built[n][1]:
            print(f"{n} fold {fi['year']} real L {fi['real_long']} S {fi['real_inv']} + synth {fi['synth_long']}/{fi['synth_inv']} | pred L mean {100 * fi['pred_long_mean']:+.2f}% pos {100 * fi['pred_long_pos']:.0f}% inv pref {100 * fi['pred_inv_pref']:.0f}%")
    print("out-of-sample on real decisions", json.dumps(oos_all))
    for kk, v in res.items():
        print(f"== {kk} {v['window'][0]}..{v['window'][1]} sessions {v['sessions']}")
        for m in methods:
            r = v["methods"][m]
            print(f"  {m:12s} return {r['return_pct']:8.2f}% vs_hold {r['vs_hold_pct']:8.2f} maxDD {r['max_drawdown_daily_pct']:7.2f}% buys {r['buys']:3d} inv {r['inverse_buys']:3d} days L/S/C {r['days_long']}/{r['days_inverse']}/{r['days_cash']} years {r['by_year_pct']}")
    print("synthetic test paths", json.dumps(summ))
    print("latest", rec["latest"])
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
