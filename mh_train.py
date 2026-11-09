import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import warnings
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")


class Deadline:
    def __init__(self, seconds):
        self.seconds = seconds
        self.t_end = time.monotonic() + seconds

    def left(self):
        return self.t_end - time.monotonic()

    def check(self, where):
        if self.left() <= 0:
            raise SystemExit(f"wall-clock cap {self.seconds}s reached at {where}")

    def callback(self):
        def cb(env):
            if self.left() <= 0:
                raise SystemExit(f"wall-clock cap {self.seconds}s reached during fit (iteration {env.iteration})")
        cb.order = 10
        return cb


def start_watchdog(deadline, grace=0.0):
    def run():
        time.sleep(max(0.0, deadline.left()) + grace)
        sys.stderr.write("watchdog: wall-clock cap exceeded, exiting 3\n")
        sys.stderr.flush()
        os._exit(3)
    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "kr_etf", "kodex_daily_raw.json")
OUT_ROOT = os.path.join(HERE, "runs", "mh")
PARAMS = dict(objective="regression", num_leaves=31, learning_rate=0.05, n_estimators=400, min_child_samples=200, colsample_bytree=0.8, subsample=0.8, subsample_freq=1, reg_lambda=1.0, num_threads=2, verbose=-1)
L_PARAMS = dict(PARAMS, min_child_samples=50)
SEEDS = (0, 1, 2, 3, 4)
S_FOLDS = [("F1", "2022-11-23", "2023-12-29", "2024-01-02", "2024-03-29", "2024-04-01", "2024-12-30"), ("F2", "2022-11-23", "2024-09-30", "2024-10-01", "2024-12-30", "2025-01-02", "2025-12-30"), ("F3", "2022-11-23", "2025-09-30", "2025-10-01", "2025-12-30", "2026-01-02", "2026-11-05")]
L_TEST_YEARS = list(range(2015, 2027))
L_TRAIN_START = "2002-10-14"
L_LAST = "2026-11-05"
Q = dict(S=(0.99, 0.98, 0.95), L=(0.90, 0.80, 0.70))
MIN_EVENTS = dict(S=100, L=30)
EMBARGO = dict(S=1, L=5)
MAX_FITS, WALL_MAX_S = 75, 7200
OK_FLAGS = (0, 2)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


class Sessions:
    def __init__(self, days):
        self.days = sorted(days)
        self.idx = {d: i for i, d in enumerate(self.days)}

    def first_on_or_after(self, d):
        for x in self.days:
            if x >= d:
                return x
        return None

    def last_on_or_before(self, d):
        out = None
        for x in self.days:
            if x <= d:
                out = x
            else:
                break
        return out

    def shift(self, d, n):
        i = self.idx[d] + n
        return self.days[i] if 0 <= i < len(self.days) else None

    def resolve(self, spec):
        name, ts, te, vs, ve, xs, xe = spec
        r = dict(name=name, train=(self.first_on_or_after(ts), self.last_on_or_before(te)), val=(self.first_on_or_after(vs), self.last_on_or_before(ve)), test=(self.first_on_or_after(xs), self.last_on_or_before(xe)))
        for k in ("train", "val", "test"):
            if None in r[k] or r[k][0] > r[k][1]:
                raise SystemExit(f"fold {name} phase {k} unresolved: {r[k]}")
        if not (r["train"][1] < r["val"][0] <= r["val"][1] < r["test"][0]):
            raise SystemExit(f"fold {name} phases overlap: {r}")
        return r


def l_fold_specs(sessions):
    out = []
    for y in L_TEST_YEARS:
        xe = L_LAST if y == 2026 else f"{y}-12-31"
        out.append((f"L{y}", L_TRAIN_START, f"{y - 2}-12-31", f"{y - 1}-01-01", f"{y - 1}-12-31", f"{y}-01-01", xe))
    return out


def phase_mask(day, label_end, start, end, next_start, embargo, sessions, purge):
    m = (day >= start) & (day <= end)
    if embargo:
        cut = sessions.shift(end, -embargo)
        m &= day <= cut
    if purge:
        le = np.where(label_end == None, "9999-99-99", label_end).astype(str)
        m &= le < next_start
    return m


def select_threshold(pred, y, head):
    cands = []
    for q in Q[head]:
        tau = float(np.quantile(pred, q, method="linear"))
        sel = (pred >= tau) & (pred > 0)
        n = int(sel.sum())
        mean = float(y[sel].mean()) if n else float("nan")
        cands.append(dict(q=q, tau=tau, n=n, mean=mean, ok=bool(n >= MIN_EVENTS[head] and mean > 0)))
    ok = [c for c in cands if c["ok"]]
    chosen = max(ok, key=lambda c: (c["mean"], c["tau"])) if ok else None
    return chosen, cands


def fit_predict(Xtr, ytr, Xs, params, budget, save_dir=None, tag=None):
    import lightgbm as lgb
    preds = [np.full(len(x), np.nan) for x in Xs]
    dl = budget["deadline"]
    for seed in SEEDS:
        if budget["fits"] >= MAX_FITS:
            raise SystemExit("fit cap reached")
        dl.check("before fit")
        m = lgb.LGBMRegressor(**params, random_state=seed)
        m.fit(Xtr, ytr, callbacks=[dl.callback()])
        budget["fits"] += 1
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            m.booster_.save_model(os.path.join(save_dir, f"{tag}_s{seed}.txt"))
        dl.check("before predict")
        for i, x in enumerate(Xs):
            p = np.full(len(x), np.nan)
            okx = np.isfinite(x).all(1)
            if okx.any():
                p[okx] = m.predict(x[okx])
            preds[i] = p if seed == SEEDS[0] else preds[i] + p
    return [p / len(SEEDS) for p in preds]


def stats(y):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if not len(y):
        return dict(n=0)
    return dict(n=int(len(y)), mean=float(y.mean()), median=float(np.median(y)), hit=float((y > 0).mean()), p05=float(np.quantile(y, 0.05)), min=float(y.min()), max=float(y.max()))


def run_head(head, T, sessions, specs, params, budget, out, model_dir=None):
    X, y, day, sym = T["X"], T["y"], T["day"], T["sym"]
    label_end, labeled = T["label_end"], T["labeled"]
    results, pred_rows = [], []
    for spec in specs:
        f = sessions.resolve(spec)
        tr = phase_mask(day, label_end, f["train"][0], f["train"][1], f["val"][0], EMBARGO[head], sessions, True) & labeled
        va_all = (day >= f["val"][0]) & (day <= f["val"][1])
        va = phase_mask(day, label_end, f["val"][0], f["val"][1], f["test"][0], EMBARGO[head], sessions, True) & labeled
        te = (day >= f["test"][0]) & (day <= f["test"][1])
        by = lambda m: {str(s): int(c) for s, c in zip(*np.unique(sym[m], return_counts=True))} if m.any() else {}
        n_tr_sym, n_te_sym = by(tr), by(te)
        rec = dict(fold=f["name"], dates=f, n_train=int(tr.sum()), n_val_mature=int(va.sum()), n_val_all=int(va_all.sum()), n_test_rows=int(te.sum()), n_test_labeled=int((te & labeled).sum()), n_train_by_sym=n_tr_sym, n_test_by_sym=n_te_sym, transfer_syms=sorted(s for s in n_te_sym if n_tr_sym.get(s, 0) == 0), params=params, seeds=list(SEEDS))
        if tr.sum() < 500 or va.sum() < MIN_EVENTS[head]:
            rec.update(status="skipped_insufficient_rows", tau=None)
            results.append(rec)
            continue
        p_va, p_te = fit_predict(X[tr], y[tr], [X[va], X[te]], params, budget, save_dir=model_dir, tag=f"{head}_{f['name']}")
        if model_dir:
            rec["model_files"] = [f"{head}_{f['name']}_s{s}.txt" for s in SEEDS]
        chosen, cands = select_threshold(p_va, y[va], head)
        rec.update(status="fitted", candidates=cands, chosen=chosen, tau=chosen["tau"] if chosen else None)
        pos = p_va > 0
        rec["val_pred_pos_rule"] = dict(n=int(pos.sum()), mean=float(y[va][pos].mean()) if pos.any() else None)
        rec["val_population"] = stats(y[va])
        yt, pt, el = y[te], p_te, labeled[te]
        if chosen:
            sig = (pt >= chosen["tau"]) & (pt > 0)
        else:
            sig = np.zeros(len(pt), bool)
        rec["test"] = dict(population_eligible=stats(yt[el]), n_test_ineligible=int((~el).sum()), signals=int(sig.sum()), signals_eligible=int((sig & el).sum()), signals_ineligible=int((sig & ~el).sum()), signals_eligible_stats=stats(yt[sig & el]), pred_pos_rule=dict(signals=int((pt > 0).sum()), signals_eligible=int(((pt > 0) & el).sum()), **{"eligible_" + k: v for k, v in stats(yt[(pt > 0) & el]).items()}), unpredicted=int(np.isnan(pt).sum()))
        if "flag" in T:
            fl = T["flag"][te]
            rec["test"]["signal_flags"] = {str(k): int(v) for k, v in zip(*np.unique(fl[sig], return_counts=True))} if sig.any() else {}
        results.append(rec)
        for i in np.flatnonzero(te):
            pass
        idx = np.flatnonzero(te)
        pred_rows.append(dict(fold=f["name"], idx=idx, pred=pt, signal=sig))
    out[head] = results
    return pred_rows


def load_tables(path):
    z = np.load(path, allow_pickle=True)
    S = dict(X=z["s_X"], y=z["s_y"], day=z["s_day"].astype(str), sym=z["s_sym"].astype(str), t=z["s_t"].astype(str), flag=z["s_flag"], label_end=np.array([e[:10] if e else None for e in z["s_exit_ts"]], dtype=object))
    S["labeled"] = z["s_eligible"].astype(bool)
    L = dict(X=z["l_X"], y=z["l_y"], day=z["l_day"].astype(str), sym=z["l_sym"].astype(str), label_end=np.array([e if e else None for e in z["l_exit_day"]], dtype=object))
    L["labeled"] = z["l_eligible"].astype(bool)
    return S, L, json.loads(str(z["meta"]))


def main(tables, heads=("S", "L")):
    sessions = Sessions(json.load(open(RAW, encoding="utf-8"))["069500"]["days"])
    S, L, tmeta = load_tables(tables)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=False)
    deadline = Deadline(WALL_MAX_S)
    start_watchdog(deadline)
    budget = dict(fits=0, t0=time.time(), deadline=deadline)
    meta = dict(script_sha256=sha(os.path.abspath(__file__)), heads=list(heads), tables=os.path.relpath(tables, HERE), tables_sha256=sha(tables), tables_meta=tmeta, raw_sha256=sha(RAW), params=PARAMS, l_params=L_PARAMS, seeds=list(SEEDS), s_folds=S_FOLDS, l_test_years=L_TEST_YEARS, q=Q, min_events=MIN_EVENTS, embargo=EMBARGO, max_fits=MAX_FITS, wall_max_s=WALL_MAX_S, started=datetime.now().isoformat(timespec="seconds"))
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("META", json.dumps({k: meta[k] for k in ("script_sha256", "heads", "tables_sha256", "raw_sha256", "started")}))
    out = {"S": [], "L": []}
    model_dir = os.path.join(out_dir, "models")
    ps = run_head("S", S, sessions, S_FOLDS, PARAMS, budget, out, model_dir) if "S" in heads and len(S["y"]) else []
    pl = run_head("L", L, sessions, l_fold_specs(sessions), L_PARAMS, budget, out, model_dir) if "L" in heads else []
    np.savez(os.path.join(out_dir, "predictions.npz"), **{f"S_{r['fold']}_idx": r["idx"] for r in ps}, **{f"S_{r['fold']}_pred": r["pred"] for r in ps}, **{f"S_{r['fold']}_signal": r["signal"] for r in ps}, **{f"L_{r['fold']}_idx": r["idx"] for r in pl}, **{f"L_{r['fold']}_pred": r["pred"] for r in pl}, **{f"L_{r['fold']}_signal": r["signal"] for r in pl})
    out["budget"] = dict(fits=budget["fits"], wall_s=round(time.time() - budget["t0"], 1))
    json.dump(out, open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    for head in ("S", "L"):
        for r in out[head]:
            t = r.get("test", {})
            ch = r.get("chosen")
            print(head, r["fold"], r["status"], "train", r["n_train"], "val", r["n_val_mature"], "tau", f"{ch['tau']:.5f} q{ch['q']}" if ch else None, "| test signals", t.get("signals"), "eligible", t.get("signals_eligible"), stats_line(t.get("signals_eligible_stats")), "| pop(eligible)", stats_line(t.get("population_eligible")), "ineligible", t.get("n_test_ineligible"))
    print("budget", out["budget"], "saved", out_dir)


def stats_line(s):
    if not s or not s.get("n"):
        return "n0"
    return f"n{s['n']} mean {s['mean']*100:+.3f}% med {s['median']*100:+.3f}% hit {s['hit']*100:.0f}%"


def selftest():
    sess = Sessions(json.load(open(RAW, encoding="utf-8"))["069500"]["days"])
    f = sess.resolve(S_FOLDS[0])
    assert f["train"] == ("2022-11-23", "2023-12-28") or f["train"][1] <= "2023-12-29", f
    assert f["val"][0] >= "2024-01-02" and f["test"][0] >= "2024-04-01" and f["test"][1] <= "2024-12-30", f
    days = np.array(sess.days[sess.idx[f["train"][0]]:sess.idx[f["test"][1]] + 1])
    le = np.array([d for d in days], dtype=object)
    m = phase_mask(days, le, f["train"][0], f["train"][1], f["val"][0], 1, sess, True)
    assert max(days[m]) == sess.shift(f["train"][1], -1) and f["train"][1] not in set(days[m]), max(days[m])
    le2 = le.copy()
    le2[days == f["val"][1]] = f["test"][0]
    mv = phase_mask(days, le2, f["val"][0], f["val"][1], f["test"][0], 0, sess, True)
    assert f["val"][1] not in set(days[mv]) and sess.shift(f["val"][1], -1) in set(days[mv])
    lf = sess.resolve(l_fold_specs(sess)[0])
    assert lf["val"][0] >= "2014-01-01" and lf["val"][1] <= "2014-12-31" and lf["test"][0] >= "2015-01-01" and lf["train"][1] <= "2013-12-31", lf
    rng = np.random.default_rng(0)
    pred = rng.normal(0, 1, 1000)
    y_neg = -np.abs(rng.normal(0, 1, 1000))
    ch, c = select_threshold(pred, y_neg, "S")
    assert ch is None and all(not x["ok"] for x in c)
    y_pos = pred * 0.01 + rng.normal(0, 0.001, 1000)
    ch, c = select_threshold(pred, y_pos, "L")
    assert ch is not None and ch["n"] >= 30 and ch["mean"] > 0
    pred2 = np.concatenate([np.full(500, 1.0), np.full(500, 2.0)])
    ch2, c2 = select_threshold(pred2, np.full(1000, 0.01), "L")
    assert ch2["tau"] == max(x["tau"] for x in c2 if x["ok"]), ch2
    n = 6000
    X = rng.normal(0, 1, (n, 3))
    yv = 0.01 * X[:, 0] + rng.normal(0, 0.005, n)
    Xs = X.copy()
    params = dict(PARAMS, n_estimators=20, min_child_samples=20)
    budget = dict(fits=0, t0=time.time(), deadline=Deadline(600))
    (p1, p2) = fit_predict(X[:4000], yv[:4000], [X[4000:5000], Xs[5000:]], params, budget)
    assert budget["fits"] == 5 and np.isfinite(p1).all() and np.corrcoef(p1, yv[4000:5000])[0, 1] > 0.5
    X[5000, 0] = np.nan
    (p3,) = fit_predict(X[:4000], yv[:4000], [X[5000:5002]], params, budget)
    assert np.isnan(p3[0]) and np.isfinite(p3[1])
    t0 = time.monotonic()
    try:
        fit_predict(X[:4000], yv[:4000], [X[4000:4100]], dict(PARAMS, n_estimators=5000, min_child_samples=20), dict(fits=0, t0=time.time(), deadline=Deadline(0.3)))
        raise AssertionError("deadline did not abort the fit")
    except SystemExit as e:
        assert "during fit" in str(e) or "before" in str(e), str(e)
    assert time.monotonic() - t0 < 5, time.monotonic() - t0
    t1 = time.monotonic()
    r = subprocess.run([sys.executable, "-X", "utf8", os.path.abspath(__file__), "watchdog-test"], capture_output=True, text=True, timeout=30)
    assert r.returncode == 3 and "watchdog" in r.stderr and time.monotonic() - t1 < 10, (r.returncode, r.stderr[-200:], time.monotonic() - t1)
    days_all = [d for d in sess.days if "2022-11-23" <= d <= "2024-12-30"]
    per_day = 60
    nrow = len(days_all) * per_day
    rng2 = np.random.default_rng(7)
    Xf = rng2.normal(0, 1, (nrow, 3))
    yf = 0.01 * Xf[:, 0] + rng2.normal(0, 0.003, nrow)
    dayf = np.repeat(np.array(days_all), per_day)
    labeled = np.ones(nrow, bool)
    test_mask = (dayf >= "2024-04-01")
    bad_idx = np.flatnonzero(test_mask)[:37]
    labeled[bad_idx] = False
    yf_bad = yf.copy()
    yf_bad[bad_idx] = 9.0
    T = dict(X=Xf, y=yf_bad, day=dayf, sym=np.array(["069500"] * nrow), label_end=dayf.astype(object), labeled=labeled, flag=np.zeros(nrow, np.int8))
    out = {}
    budget2 = dict(fits=0, t0=time.time(), deadline=Deadline(600))
    import tempfile
    mdir = tempfile.mkdtemp()
    run_head("S", T, sess, [S_FOLDS[0]], dict(PARAMS, n_estimators=20, min_child_samples=20), budget2, out, mdir)
    rec = out["S"][0]
    assert rec["status"] == "fitted" and budget2["fits"] == 5, (rec["status"], budget2)
    assert sorted(os.listdir(mdir)) == [f"S_F1_s{s}.txt" for s in SEEDS] and rec["model_files"] == [f"S_F1_s{s}.txt" for s in SEEDS], os.listdir(mdir)
    import lightgbm as lgb
    bst = lgb.Booster(model_file=os.path.join(mdir, "S_F1_s0.txt"))
    assert bst.num_trees() == 20 and np.isfinite(bst.predict(Xf[:5])).all()
    tt = rec["test"]
    assert rec["n_train_by_sym"].get("069500", 0) > 0 and rec["transfer_syms"] == [], (rec["n_train_by_sym"], rec["transfer_syms"])
    n_test = int(test_mask.sum())
    assert tt["n_test_ineligible"] == 37 and tt["population_eligible"]["n"] == n_test - 37 and abs(tt["population_eligible"]["mean"]) < 0.002, tt
    assert rec["chosen"] is not None and tt["signals_eligible_stats"]["n"] <= tt["signals"] and tt["signals_eligible"] + tt["signals_ineligible"] == tt["signals"], tt
    assert tt["signals_eligible_stats"]["mean"] > tt["population_eligible"]["mean"], tt
    assert all(r["day"] <= "2023-12-28" for r in [dict(day=d) for d in dayf[(dayf >= "2022-11-23") & (dayf <= "2023-12-28")]])
    print("selftest ok", dict(sha=sha(os.path.abspath(__file__)), s_f1=f, l_2015=lf))


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    elif sys.argv[1:] == ["watchdog-test"]:
        start_watchdog(Deadline(0.5), grace=0.5)
        time.sleep(20)
        print("watchdog failed")
        sys.exit(0)
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 4 and sys.argv[2] == "--heads":
        main(sys.argv[1], tuple(h for h in sys.argv[3].split(",") if h in ("S", "L")))
    else:
        print("usage: mh_train.py <tables.npz> [--heads S,L|L] | selftest")
