import hashlib
import json
import os
import sys
from datetime import datetime

import numpy as np

import mh_data as md
import mh_ledger as ml

HERE = os.path.dirname(os.path.abspath(__file__))
ADJ = os.path.join(HERE, "data", "kr_etf", "kodex_daily_adj.json")
OUT_ROOT = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work\mh_synth")
CAL_START = "2019-01-01"
WARM, EVAL, BLOCK = 250, 252, 20
SEEDS = tuple(range(20))
P0 = {"069500": 100000.0, "229200": 10000.0}
TICK = 5.0
CLIP = 0.30
SCENARIOS = {"baseline": {}, "slow_decline": dict(intraday_add=-0.0012), "gap_down": dict(gap_add=-0.025, gap_days=(30, 40)), "high_vol": dict(scale=2.5, center=True), "crash_rebound": dict(crash=(-0.021, 20, 40), rebound=(0.005, 40, 120))}
MODEL_FOLD = "L2020"
SYMS = md.SYMS


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def train_mask_days(tables, fold=MODEL_FOLD):
    import mh_train as mt
    sessions = mt.Sessions(json.load(open(mt.RAW, encoding="utf-8"))["069500"]["days"])
    S, L, _ = mt.load_tables(tables)
    spec = next(s for s in mt.l_fold_specs(sessions) if s[0] == fold)
    f = sessions.resolve(spec)
    tr = mt.phase_mask(L["day"], L["label_end"], f["train"][0], f["train"][1], f["val"][0], mt.EMBARGO["L"], sessions, True) & L["labeled"]
    per_sym = {s: sorted(set(L["day"][tr & (L["sym"] == s)])) for s in SYMS}
    joint = sorted(set.intersection(*[set(v) for v in per_sym.values()]))
    return dict(fold=fold, dates=f, per_sym_counts={s: len(v) for s, v in per_sym.items()}, joint=joint)


def load_base(tables):
    tm = train_mask_days(tables)
    days = tm["joint"]
    adj = json.load(open(ADJ, encoding="utf-8"))
    comp = {}
    for s in SYMS:
        idx = {d: i for i, d in enumerate(adj[s]["days"])}
        O, H, L, C, V = (np.array(adj[s][k], float) for k in ("O", "H", "L", "C", "V"))
        rows = []
        for d in days:
            i = idx[d]
            rows.append((O[i] / C[i - 1] - 1, C[i] / O[i] - 1, H[i] / max(O[i], C[i]) - 1, 1 - L[i] / min(O[i], C[i]), V[i]))
        comp[s] = np.array(rows, float)
    return dict(days=days, comp=comp, n=len(days), mask=dict(fold=tm["fold"], train_dates=tm["dates"]["train"], val_start=tm["dates"]["val"][0], per_sym_counts=tm["per_sym_counts"], joint_count=len(days), joint_span=[days[0], days[-1]] if days else None))


def generate(base, scenario, seed, warm=WARM, evaln=EVAL, block=BLOCK):
    rng = np.random.default_rng(seed)
    n, total = base["n"], warm + evaln
    starts = rng.integers(0, n - block + 1, size=total // block + 1)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:total]
    cfg = SCENARIOS[scenario]
    out = {}
    for s in SYMS:
        comp = base["comp"][s][idx].copy()
        gap, intra = comp[:, 0], comp[:, 1]
        ev = slice(warm, total)
        if cfg.get("scale"):
            gap[ev] *= cfg["scale"]
            intra[ev] *= cfg["scale"]
            if cfg.get("center"):
                gap[ev] -= gap[ev].mean()
                intra[ev] -= intra[ev].mean()
        if cfg.get("intraday_add"):
            intra[ev] += cfg["intraday_add"]
        if cfg.get("gap_add"):
            a, b = cfg["gap_days"]
            gap[warm + a:warm + b] += cfg["gap_add"]
        if cfg.get("crash"):
            r, a, b = cfg["crash"]
            intra[warm + a:warm + b] += r
            r2, a2, b2 = cfg["rebound"]
            intra[warm + a2:warm + b2] += r2
        gap, intra = np.clip(gap, -CLIP, CLIP), np.clip(intra, -CLIP, CLIP)
        O, H, L, C = np.empty(total), np.empty(total), np.empty(total), np.empty(total)
        prev = P0[s]
        for t in range(total):
            o = max(TICK, round(prev * (1 + gap[t]) / TICK) * TICK)
            c = max(TICK, round(o * (1 + intra[t]) / TICK) * TICK)
            h = max(o, c) * (1 + max(0.0, comp[t, 2]))
            l = min(o, c) * (1 - min(0.99, max(0.0, comp[t, 3])))
            O[t], H[t], L[t], C[t] = o, round(h / TICK) * TICK, max(TICK, round(l / TICK) * TICK), c
            prev = c
        out[s] = dict(O=O, H=H, L=L, C=C, V=comp[:, 4].copy(), gap=gap, intra=intra)
    days = [str(d) for d in np.busday_offset(CAL_START, np.arange(total), roll="forward").astype("datetime64[D]")]
    return dict(days=days, syms=out, warm=warm, evaln=evaln, scenario=scenario, seed=seed, block_starts=starts.tolist())


def quality(base, gen):
    def stats(r):
        r = np.asarray(r, float)
        m, sd = r.mean(), r.std(ddof=1)
        z = (r - m) / sd if sd > 0 else r * 0
        ac = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if len(r) > 2 else float("nan")
        return dict(mean=float(m), std=float(sd), skew=float((z ** 3).mean()), kurt=float((z ** 4).mean() - 3), ac1=ac)
    out = {"note": "generated_eval uses pre-rounding gap/intra components; generated_eval_ohlc uses close-to-close returns of the tick-rounded, floored OHLC actually fed to the ledgers; high_vol centering zeroes the arithmetic mean of components, not the cumulative return"}
    for s in SYMS:
        b = base["comp"][s]
        base_ret = (1 + b[:, 0]) * (1 + b[:, 1]) - 1
        g = gen["syms"][s]
        ev = slice(gen["warm"], gen["warm"] + gen["evaln"])
        gen_ret = (1 + g["gap"][ev]) * (1 + g["intra"][ev]) - 1
        C = g["C"]
        ohlc_ret = C[gen["warm"]:gen["warm"] + gen["evaln"]] / C[gen["warm"] - 1:gen["warm"] + gen["evaln"] - 1] - 1
        out[s] = dict(base=stats(base_ret), generated_eval=stats(gen_ret), generated_eval_ohlc=stats(ohlc_ret), cumulative_eval_return_ohlc=float(C[gen["warm"] + gen["evaln"] - 1] / C[gen["warm"] - 1] - 1), volume_mean_ratio=float(g["V"][ev].mean() / b[:, 4].mean()) if b[:, 4].mean() > 0 else None, min_price=float(min(g["L"].min(), g["C"].min())), tick_rounding_abs_mean=float(np.mean(np.abs(ohlc_ret - gen_ret))))
    b0, b1 = base["comp"][SYMS[0]], base["comp"][SYMS[1]]
    out["cross_corr_base"] = float(np.corrcoef((1 + b0[:, 0]) * (1 + b0[:, 1]) - 1, (1 + b1[:, 0]) * (1 + b1[:, 1]) - 1)[0, 1])
    g0, g1 = gen["syms"][SYMS[0]], gen["syms"][SYMS[1]]
    ev = slice(gen["warm"], gen["warm"] + gen["evaln"])
    out["cross_corr_generated_eval"] = float(np.corrcoef((1 + g0["gap"][ev]) * (1 + g0["intra"][ev]) - 1, (1 + g1["gap"][ev]) * (1 + g1["intra"][ev]) - 1)[0, 1])
    w, e = gen["warm"], gen["warm"] + gen["evaln"]
    out["cross_corr_generated_eval_ohlc"] = float(np.corrcoef(g0["C"][w:e] / g0["C"][w - 1:e - 1] - 1, g1["C"][w:e] / g1["C"][w - 1:e - 1] - 1)[0, 1])
    return out


def daily_dict(gen):
    return {s: dict(days=gen["days"], Ca=gen["syms"][s]["C"], Oa=gen["syms"][s]["O"], Va=gen["syms"][s]["V"], Or=gen["syms"][s]["O"], Cr=gen["syms"][s]["C"]) for s in SYMS}


def ledger_rows(gen, predict_fn, tau):
    rows = md.l_table(daily_dict(gen))
    X = np.array([r["feats"] for r in rows], float)
    pred = predict_fn(X) if len(rows) else np.array([])
    out = []
    for k, r in enumerate(rows):
        p = float(pred[k])
        out.append(dict(i=k, sym=r["sym"], day=r["day"], pb=r["pb"], ps=r["ps"], entry_day=r["entry_day"], exit_day=r["exit_day"], pending=r["pending"], pred=p, signal=bool(np.isfinite(p) and tau is not None and p >= tau and p > 0), tau=tau, dip_first=r["dip_first"], ma120=float(r["feats"][md.L_FEATS.index("ma120")]), r1=float(r["feats"][0])))
    return out


def evaluate(gen, rows):
    days = gen["days"]
    closes = {s: dict(zip(days, map(float, gen["syms"][s]["C"]))) for s in SYMS}
    ev = days[gen["warm"]:gen["warm"] + gen["evaln"]]
    res = {}
    for name, lk in (("long_only_model", "model"), ("long_only_D1", "L-D1"), ("long_only_D2", "L-D2"), ("cash", None)):
        led = ml.simulate(name, ml.SHARES["long_only"] if lk else ml.SHARES["cash"], [], rows, {}, closes, ev, None, lk, all_sessions=days)
        m = ml.period_metrics(led, ev[0], ev[-1])
        m["n_trades"] = len(led.trades)
        res[name] = m
    led = ml.simulate_buyhold("long_only_buyhold", rows, closes, ev)
    m = ml.period_metrics(led, ev[0], ev[-1])
    m["n_trades"] = 0
    res["long_only_buyhold"] = m
    rows_pp = [dict(r, signal=bool(np.isfinite(r["pred"]) and r["pred"] > 0), tau=0.0) for r in rows]
    led = ml.simulate("long_only_model_predpos_exploratory", ml.SHARES["long_only"], [], rows_pp, {}, closes, ev, None, "model", all_sessions=days)
    m = ml.period_metrics(led, ev[0], ev[-1])
    m["n_trades"] = len(led.trades)
    m["note"] = "exploratory diagnostic decided after seeing L2020 tau=None; not pre-registered as a primary rule"
    res["long_only_model_predpos_exploratory"] = m
    return res


def booster_predictor(train_dir, fold=MODEL_FOLD):
    import lightgbm as lgb
    res = json.load(open(os.path.join(train_dir, "results.json"), encoding="utf-8"))
    rec = next(r for r in res["L"] if r["fold"] == fold)
    if rec.get("status") != "fitted" or not rec.get("model_files"):
        raise SystemExit(f"{fold} not fitted or models not saved")
    boosters = [lgb.Booster(model_file=os.path.join(train_dir, "models", f)) for f in rec["model_files"]]
    tau = rec["tau"]

    def predict(X):
        okx = np.isfinite(X).all(1)
        p = np.full(len(X), np.nan)
        if okx.any():
            p[okx] = np.mean([b.predict(X[okx], num_threads=2) for b in boosters], axis=0)
        return p
    return predict, tau, rec


def main(tables, train_dir):
    predict, tau, rec = booster_predictor(train_dir)
    base = load_base(tables)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=False)
    meta = dict(script_sha256=sha(os.path.abspath(__file__)), tables=os.path.relpath(tables, HERE), tables_sha256=sha(tables), train_dir=os.path.relpath(train_dir, HERE), model_fold=MODEL_FOLD, model_files=rec["model_files"], model_sha256={f: sha(os.path.join(train_dir, "models", f)) for f in rec["model_files"]}, tau=tau, adj_sha256=sha(ADJ), base=base["mask"], warm=WARM, evaln=EVAL, block=BLOCK, calendar_start=CAL_START, seeds=list(SEEDS), scenarios=SCENARIOS, clip=CLIP, p0=P0, started=datetime.now().isoformat(timespec="seconds"))
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    results, arrays = [], {}
    for scen in SCENARIOS:
        for seed in SEEDS:
            gen = generate(base, scen, seed)
            q = quality(base, gen)
            rows = ledger_rows(gen, predict, tau)
            ev = evaluate(gen, rows)
            results.append(dict(scenario=scen, seed=seed, quality=q, ledgers=ev, n_signals=int(sum(r["signal"] for r in rows)), block_starts=gen["block_starts"]))
            for s in SYMS:
                for k in ("O", "H", "L", "C", "V"):
                    arrays[f"{scen}_s{seed}_{s}_{k}"] = gen["syms"][s][k]
            arrays[f"{scen}_s{seed}_days"] = np.array(gen["days"])
    np.savez_compressed(os.path.join(out_dir, "generated.npz"), **arrays)
    summary = {}
    for scen in SCENARIOS:
        rs = [r for r in results if r["scenario"] == scen]
        summary[scen] = {}
        for name in ("long_only_model", "long_only_D1", "long_only_D2", "long_only_buyhold", "cash", "long_only_model_predpos_exploratory"):
            pn = np.array([r["ledgers"][name]["pnl"] if r["ledgers"][name]["pnl"] is not None else np.nan for r in rs], float)
            dd = np.array([r["ledgers"][name]["max_dd_won"] if r["ledgers"][name]["max_dd_won"] is not None else np.nan for r in rs], float)
            summary[scen][name] = dict(pnl_median=float(np.nanmedian(pn)), pnl_q1=float(np.nanpercentile(pn, 25)), pnl_q3=float(np.nanpercentile(pn, 75)), dd_median=float(np.nanmedian(dd)), n_paths=int(np.isfinite(pn).sum()))
        mp = np.array([r["ledgers"]["long_only_model"]["pnl"] for r in rs], float)
        summary[scen]["model_beats_cash_paths"] = int((mp > 0).sum())
        summary[scen]["model_beats_buyhold_paths"] = int((mp > np.array([r["ledgers"]["long_only_buyhold"]["pnl"] for r in rs], float)).sum())
        summary[scen]["note"] = "path counts are not real-market probabilities; long_only_model_predpos_exploratory is an exploratory diagnostic (pred>0 without tau), reported separately from the pre-registered primary (tau rule; L2020 tau=None means cash)"
    json.dump(dict(meta=meta, summary=summary, results=results), open(os.path.join(out_dir, "synth_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    for scen, s in summary.items():
        print(scen, " | ".join(f"{n} 중앙 {v['pnl_median']:+,.0f} [{v['pnl_q1']:+,.0f}, {v['pnl_q3']:+,.0f}] 낙폭 {v['dd_median']:+,.0f}" for n, v in s.items() if isinstance(v, dict)), "| 모델>현금", s["model_beats_cash_paths"], "모델>보유", s["model_beats_buyhold_paths"])
    print("saved", out_dir)


def selftest():
    rng = np.random.default_rng(123)
    n = 800
    comp = {}
    for s in SYMS:
        gap = rng.normal(0.0002, 0.006, n)
        intra = rng.normal(0.0001, 0.009, n)
        comp[s] = np.column_stack([gap, intra, np.abs(rng.normal(0, 0.004, n)), np.abs(rng.normal(0, 0.004, n)), rng.integers(1000, 5000, n).astype(float)])
    base = dict(days=[f"B{i}" for i in range(n)], comp=comp, n=n)
    g0 = generate(base, "baseline", 0)
    g0b = generate(base, "baseline", 0)
    assert all(np.array_equal(g0["syms"][s]["C"], g0b["syms"][s]["C"]) for s in SYMS) and len(g0["days"]) == WARM + EVAL and g0["days"][0] == "2019-01-01"
    for s in SYMS:
        y = g0["syms"][s]
        assert (y["C"] > 0).all() and (y["L"] > 0).all() and (y["H"] >= np.maximum(y["O"], y["C"]) - 1e-9).all() and (y["L"] <= np.minimum(y["O"], y["C"]) + 1e-9).all()
        assert np.allclose(y["C"] % TICK, 0) and np.allclose(y["O"] % TICK, 0)
    assert g0["block_starts"] == generate(base, "slow_decline", 0)["block_starts"]
    ev = slice(WARM, WARM + EVAL)
    gd = generate(base, "slow_decline", 0)
    assert abs((gd["syms"]["069500"]["intra"][ev] - g0["syms"]["069500"]["intra"][ev]).mean() + 0.0012) < 1e-9
    gg = generate(base, "gap_down", 0)
    diff = gg["syms"]["229200"]["gap"] - g0["syms"]["229200"]["gap"]
    assert np.allclose(diff[WARM + 30:WARM + 40], -0.025) and np.allclose(diff[:WARM + 30], 0) and np.allclose(diff[WARM + 40:], 0)
    gv = generate(base, "high_vol", 0)
    r0 = g0["syms"]["069500"]["intra"][ev]
    rv = gv["syms"]["069500"]["intra"][ev]
    assert abs(rv.mean()) < 1e-9 and abs(rv.std() / r0.std() - 2.5) < 0.05, (rv.mean(), rv.std() / r0.std())
    gc = generate(base, "crash_rebound", 0)
    dc = gc["syms"]["069500"]["intra"] - g0["syms"]["069500"]["intra"]
    assert np.allclose(dc[WARM + 20:WARM + 40], -0.021) and np.allclose(dc[WARM + 40:WARM + 120], 0.005) and np.allclose(dc[WARM + 120:], 0)
    crash_ret = np.prod(1 + gc["syms"]["069500"]["intra"][WARM + 20:WARM + 40]) / np.prod(1 + g0["syms"]["069500"]["intra"][WARM + 20:WARM + 40]) - 1
    assert -0.40 < crash_ret < -0.30, crash_ret
    q = quality(base, g0)
    assert set(q) == set(SYMS) | {"note", "cross_corr_base", "cross_corr_generated_eval", "cross_corr_generated_eval_ohlc"} and all(k in q["069500"]["base"] for k in ("mean", "std", "skew", "kurt", "ac1")) and q["069500"]["min_price"] > 0
    assert "generated_eval_ohlc" in q["069500"] and abs(q["069500"]["generated_eval_ohlc"]["mean"] - q["069500"]["generated_eval"]["mean"]) < 0.001 and q["069500"]["tick_rounding_abs_mean"] < 0.001, q["069500"]
    cum = np.prod(1 + (1 + g0["syms"]["069500"]["gap"][ev]) * (1 + g0["syms"]["069500"]["intra"][ev]) - 1) - 1
    assert abs(q["069500"]["cumulative_eval_return_ohlc"] - cum) < 0.02, (q["069500"]["cumulative_eval_return_ohlc"], cum)
    rows = ledger_rows(g0, lambda X: -X[:, 0], 0.0)
    assert len(rows) == 2 * (WARM + EVAL - 1 - md.L_WARM) and all(r["day"] >= g0["days"][md.L_WARM] for r in rows)
    assert any(r["signal"] for r in rows) and all((r["pred"] >= 0.0 and r["pred"] > 0) == r["signal"] for r in rows if np.isfinite(r["pred"]))
    res = evaluate(g0, rows)
    assert set(res) == {"long_only_model", "long_only_D1", "long_only_D2", "long_only_buyhold", "cash", "long_only_model_predpos_exploratory"} and res["cash"]["pnl"] == 0.0 and res["cash"]["deposits"] == EVAL * ml.DEPOSIT and res["long_only_buyhold"]["exposure_avg"] > 0.5, {k: (v["pnl"], v["deposits"]) for k, v in res.items()}
    assert all(v["sessions"] == EVAL for v in res.values())
    rows_none = ledger_rows(g0, lambda X: -X[:, 0], None)
    assert not any(r["signal"] for r in rows_none)
    res_none = evaluate(g0, rows_none)
    assert res_none["long_only_model"]["pnl"] == 0.0 and res_none["long_only_model"]["n_trades"] == 0 and res_none["long_only_model_predpos_exploratory"]["n_trades"] > 0 and "exploratory" in res_none["long_only_model_predpos_exploratory"]["note"], (res_none["long_only_model"], res_none["long_only_model_predpos_exploratory"]["n_trades"])
    import tempfile
    import mh_train as mt
    sess = mt.Sessions(json.load(open(mt.RAW, encoding="utf-8"))["069500"]["days"])
    days_all = [d for d in sess.days if "2015-10-01" <= d <= "2020-12-30"]
    ld, ls, le, lel = [], [], [], []
    for s_i, s in enumerate(SYMS):
        for d in days_all:
            if s == "229200" and d < "2016-10-10":
                continue
            ld.append(d)
            ls.append(s)
            k = sess.idx[d] + 61
            le.append(sess.days[k] if k < len(sess.days) else None)
            lel.append(k < len(sess.days))
    nrow = len(ld)
    tmp = os.path.join(tempfile.mkdtemp(), "t.npz")
    np.savez(tmp, s_eligible=np.zeros(0, bool), l_eligible=np.array(lel, bool), s_sym=np.array([], dtype=object), s_day=np.array([], dtype=object), s_t=np.array([], dtype=object), s_X=np.zeros((0, len(md.S_FEATS))), s_y=np.zeros(0), s_flag=np.zeros(0, np.int8), s_pb=np.zeros(0), s_ps=np.zeros(0), s_entry_tod=np.array([], dtype=object), s_exit_tod=np.array([], dtype=object), s_entry_ts=np.array([], dtype=object), s_exit_ts=np.array([], dtype=object), s_exit_reason=np.array([], dtype=object), s_day_cls=np.array([], dtype=object), s_day_incomplete=np.zeros(0, bool), s_day_conflicts=np.zeros(0, np.int32), s_day_dups=np.zeros(0, np.int32), s_brk60=np.zeros(0), l_dip_first=np.zeros(nrow, bool), l_sym=np.array(ls, dtype=object), l_day=np.array(ld, dtype=object), l_X=np.zeros((nrow, len(md.L_FEATS))), l_y=np.zeros(nrow), l_pb=np.ones(nrow), l_ps=np.ones(nrow), l_entry_day=np.array(ld, dtype=object), l_exit_day=np.array(le, dtype=object), l_pending=~np.array(lel, bool), s_feats=np.array(md.S_FEATS), l_feats=np.array(md.L_FEATS), flags=np.array(json.dumps(md.FLAG)), meta=np.array(json.dumps({})))
    tm = train_mask_days(tmp)
    assert tm["joint"][0] >= "2016-10-10" and tm["joint"][-1] < "2019-01-02" and tm["dates"]["val"][0] == "2019-01-02", (tm["joint"][0], tm["joint"][-1])
    last_ok = sess.shift(tm["dates"]["train"][1], -mt.EMBARGO["L"])
    assert tm["joint"][-1] <= last_ok and sess.days[sess.idx[tm["joint"][-1]] + 61] < "2019-01-02", (tm["joint"][-1], last_ok)
    assert tm["per_sym_counts"]["069500"] > tm["per_sym_counts"]["229200"] == len(tm["joint"])
    b2 = load_base(tmp)
    assert b2["n"] == len(tm["joint"]) and b2["mask"]["joint_span"][0] == tm["joint"][0] and all(b2["comp"][s].shape == (b2["n"], 5) for s in SYMS)
    print("selftest ok", dict(sha=sha(os.path.abspath(__file__)), scenarios=list(SCENARIOS), seeds=len(SEEDS), base_joint_fake=b2["n"]))


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("usage: mh_synth.py <tables.npz> <train_dir> | selftest")
