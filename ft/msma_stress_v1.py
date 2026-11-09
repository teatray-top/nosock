import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
import math
import sys
import warnings

sys.dont_write_bytecode = True
warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import learn_global_v1 as LG
import msma_v1 as M1
import msma_v2 as M2
import msma_v3 as M3

STRATS = {"W1_safe": (1.0, 0.30), "W5_balanced": (0.0, 0.50)}
ORIG_TW = M2.target_weights
ORIG_FEE = M2.LI.FEE
ORIG_TICK = M2.H8.tick
PLACEBO_SEEDS = 50
PROXY = ["^KS11", "^GSPC", "^IXIC", "^RUT", "^N225", "^HSI", "^STOXX50E", "GC=F", "CL=F", "KRW=X", "BOND10"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def eq_score(S, sigs=None):
    sigs = sigs or M1.EQ_SIGNALS
    return np.nanmean(np.tanh(np.stack([S[:, :, M1.SIG.index(s)] for s in sigs], axis=2)), axis=2)


def run_strat(name, days, codes, O, C, sd60, r, score, start, band=0.02):
    if name in STRATS:
        p, cap = STRATS[name]
        M2.target_weights = M3.make_weights(codes, p, cap, ())
        out = M2.run(days, codes, O, C, sd60, r, score, start, "score", 1.0, band, False)
    else:
        M2.target_weights = ORIG_TW
        out = M2.run(days, codes, O, C, sd60, r, score, start, name, 1.0, band, False)
    M2.target_weights = ORIG_TW
    return out


def summary(x):
    return dict(total_pct=x["total_pct"], cagr_pct=x["cagr_pct"], sharpe=x["sharpe"], mdd_pct=x["mdd_pct"], losing_years=len(x["losing_years"]), cost_pct=x["cost_pct"])


def block_permute(score, start, seed, block=63):
    rng = np.random.default_rng(seed)
    out = score.copy()
    n, m = score.shape
    idx = np.arange(start, n)
    nb = int(math.ceil(len(idx) / block))
    for j in range(m):
        blocks = [idx[k * block:(k + 1) * block] for k in range(nb)]
        order = rng.permutation(nb)
        seq = np.concatenate([blocks[k] for k in order])[:len(idx)]
        out[idx, j] = score[seq, j]
    return out


def window_years(x, a, b):
    ys = [v for y, v in x["years"].items() if a <= y <= b]
    return 100 * (np.prod([1 + v / 100 for v in ys]) - 1)


def load_proxy():
    g = LG.load_global()
    base = [d for d in g["^GSPC"]["days"] if "2000-01-01" <= d <= M1.END]
    pos = {d: i for i, d in enumerate(base)}
    n, m = len(base), len(PROXY)
    C = np.full((n, m), np.nan)
    for j, t in enumerate(PROXY):
        if t == "BOND10":
            rec = g["^TNX"]
            y = {d: c for d, c in zip(rec["days"], rec["C"])}
            lvl, prev = 100.0, None
            for i, d in enumerate(base):
                if d in y:
                    if prev is not None:
                        lvl *= math.exp(-8.0 * (y[d] - prev) / 100.0 + prev / 100.0 / 252.0)
                    prev = y[d]
                    C[i, j] = lvl
            continue
        rec = g[t]
        for d, c in zip(rec["days"], rec["C"]):
            if d in pos and c > 0:
                C[pos[d], j] = c
    for j in range(m):
        last = np.nan
        for i in range(n):
            if np.isnan(C[i, j]) and not np.isnan(last):
                C[i, j] = last
            if not np.isnan(C[i, j]):
                last = C[i, j]
    return np.array(base), PROXY, C.copy(), C


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, pth in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), pth)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    res = {}
    days, codes, O, C = M1.load()
    S, sd60, r = M1.signals(days, C)
    eq = eq_score(S)
    start = int(np.argmax(days >= M1.TEST_FROM)) - 1
    base = {k: run_strat(k, days, codes, O, C, sd60, r, eq, start) for k in ("equal_weight_all", "W1_safe", "W5_balanced")}
    res["base"] = {k: summary(v) for k, v in base.items()}
    res["subperiods"] = {k: {w: window_years(v, a, b) for w, (a, b) in {"2013-2017": ("2013", "2017"), "2018-2021": ("2018", "2021"), "2022-2026": ("2022", "2026")}.items()} for k, v in base.items()}
    print("BASE", json.dumps(res["base"]), flush=True)
    print("SUBPERIODS", json.dumps(res["subperiods"]), flush=True)
    rob = []

    def add(tag, sc=eq, Cx=C, Ox=O, st=start, sdx=sd60, rx=r):
        ew = run_strat("equal_weight_all", days, codes, Ox, Cx, sdx, rx, sc, st)
        row = dict(test=tag, equal_weight=summary(ew))
        for k in STRATS:
            x = run_strat(k, days, codes, Ox, Cx, sdx, rx, sc, st)
            row[k] = summary(x)
            row[k]["excess_cagr_vs_ew"] = x["cagr_pct"] - ew["cagr_pct"]
        rob.append(row)
        print("ROB", tag, {k: (round(row[k]["cagr_pct"], 2), round(row[k]["excess_cagr_vs_ew"], 2), round(row[k]["mdd_pct"], 1)) for k in STRATS}, "ew", round(ew["cagr_pct"], 2), flush=True)

    for k in range(1, 5):
        add(f"rebalance_offset_{k}", st=start + k)
    for h in (10, 20):
        M1.HOLD = h
        add(f"hold_{h}")
    M1.HOLD = 5
    for f in (2.0, 3.0):
        M2.LI.FEE = ORIG_FEE * f
        M2.H8.tick = lambda d, p, f=f: f * ORIG_TICK(d, p)
        add(f"cost_x{int(f)}")
    M2.LI.FEE, M2.H8.tick = ORIG_FEE, ORIG_TICK
    for s in M1.EQ_SIGNALS:
        add(f"drop_signal_{s}", sc=eq_score(S, [x for x in M1.EQ_SIGNALS if x != s]))
    for j, c in enumerate(codes):
        Cx, Ox = C.copy(), O.copy()
        Cx[:, j], Ox[:, j] = np.nan, np.nan
        sdx = sd60.copy()
        sdx[:, j] = np.nan
        add(f"drop_asset_{c}", Cx=Cx, Ox=Ox, sdx=sdx)
    lag = np.vstack([np.full((1, eq.shape[1]), np.nan), eq[:-1]])
    add("signal_lag_plus1", sc=lag)
    res["robustness"] = rob
    real_ex = {k: base[k]["cagr_pct"] - base["equal_weight_all"]["cagr_pct"] for k in STRATS}
    plac = {k: [] for k in STRATS}
    for sd in range(PLACEBO_SEEDS):
        ps = block_permute(eq, start, 4000 + sd)
        for k in STRATS:
            plac[k].append(run_strat(k, days, codes, O, C, sd60, r, ps, start)["cagr_pct"] - base["equal_weight_all"]["cagr_pct"])
    res["placebo"] = {k: dict(real_excess_cagr=real_ex[k], placebo_excess_cagr=plac[k], placebo_mean=float(np.mean(plac[k])), placebo_p95=float(np.percentile(plac[k], 95)), share_placebo_ge_real=float(np.mean(np.array(plac[k]) >= real_ex[k]))) for k in STRATS}
    print("PLACEBO", {k: (round(v["real_excess_cagr"], 2), round(v["placebo_mean"], 2), round(v["placebo_p95"], 2), v["share_placebo_ge_real"]) for k, v in res["placebo"].items()}, flush=True)
    pdays, pcodes, PO, PC = load_proxy()
    PS, psd, pr = M1.signals(pdays, PC)
    peq = eq_score(PS)
    M2.LI.FEE = 0.0005
    M2.H8.tick = lambda d, p: 0.0
    PO = np.vstack([PC[:1], PC[1:]])
    prox = {}
    for wn, (a, b) in {"2001-09..2012-12 (out of period)": ("2001-09-01", "2012-12-31"), "2013-01..2026-11 (proxy replication)": ("2013-01-01", M1.END)}.items():
        st = int(np.argmax(pdays >= a)) - 1
        en = int(np.argmax(pdays > b)) if (pdays > b).any() else len(pdays)
        sl = slice(0, en)
        row = {}
        for k in ("equal_weight_all", "W1_safe", "W5_balanced"):
            row[k] = summary(run_strat(k, pdays[sl], pcodes, PO[sl], PC[sl], psd[sl], pr[sl], peq[sl], st))
        kospi = PC[en - 1, 0] / PC[st + 1, 0] - 1
        row["hold_KOSPI_index_pct"] = 100 * kospi
        pl = {k: [] for k in STRATS}
        for sd in range(20):
            ps = block_permute(peq[sl], st, 6000 + sd)
            for k in STRATS:
                pl[k].append(run_strat(k, pdays[sl], pcodes, PO[sl], PC[sl], psd[sl], pr[sl], ps, st)["cagr_pct"] - row["equal_weight_all"]["cagr_pct"])
        row["placebo"] = {k: dict(real_excess_cagr=row[k]["cagr_pct"] - row["equal_weight_all"]["cagr_pct"], placebo_mean=float(np.mean(pl[k])), placebo_p95=float(np.percentile(pl[k], 95)), share_placebo_ge_real=float(np.mean(np.array(pl[k]) >= row[k]["cagr_pct"] - row["equal_weight_all"]["cagr_pct"]))) for k in STRATS}
        prox[wn] = row
        print("PROXY", wn, json.dumps({k: (round(v["cagr_pct"], 2), round(v["mdd_pct"], 1), round(v["sharpe"], 2)) if isinstance(v, dict) and "cagr_pct" in v else v for k, v in row.items() if k != "placebo"}), "placebo", {k: (round(v["real_excess_cagr"], 2), round(v["placebo_mean"], 2), v["share_placebo_ge_real"]) for k, v in row["placebo"].items()}, flush=True)
    M2.LI.FEE, M2.H8.tick = ORIG_FEE, ORIG_TICK
    res["proxy"] = prox
    rec = dict(version="msma-stress-v1", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], strategies=STRATS, results=res)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
