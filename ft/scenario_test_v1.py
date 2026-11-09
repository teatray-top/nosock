import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
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
import compare_trend_v2 as CT
import learn_shadow as SH

L, S = "069500", "114800"
WARM, EVAL = 250, 750
SEEDS = 10
START_PRICE = 30000.0
BASE_VOL = 0.014
GARCH_A, GARCH_B = 0.06, 0.92
DF = 5
WARM_DRIFT = 0.0003
TRENDS = {"bull": math.log(1.20) / 250, "bear": math.log(0.80) / 250, "sideways": 0.0}
SIDEWAYS_KAPPA = math.log(2) / 60
LIQ = {"high": dict(cost_bp=2.0, vol=0.8, gap=0.25), "mid": dict(cost_bp=8.0, vol=1.0, gap=0.30), "low": dict(cost_bp=30.0, vol=1.5, gap=0.45)}
CRUNCH = dict(cost_bp=60.0, vol=2.5, gap=0.55)
LIQ_CASES = ("const_high", "const_mid", "const_low", "drying", "recovering", "crunch", "cyclic")
INV_DRAG = 0.000103
FEE = LI.FEE
CAPITAL = 100_000_000.0
METHODS = ("hold", "ribbon", "learn_long", "learn_inv")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def liquidity_path(case):
    n = WARM + EVAL
    out = []
    for t in range(n):
        if t < WARM:
            out.append(dict(LIQ["mid"]))
            continue
        u = (t - WARM) / (EVAL - 1)
        if case.startswith("const_"):
            out.append(dict(LIQ[case[6:]]))
            continue
        if case == "drying":
            w = u
        elif case == "recovering":
            w = 1 - u
        elif case == "cyclic":
            w = 0.5 - 0.5 * math.cos(2 * math.pi * (t - WARM) / 250)
        else:
            w = None
        if case == "crunch":
            k = t - WARM
            out.append(dict(CRUNCH) if 300 <= k < 360 else dict(LIQ["mid"]))
            continue
        hi, lo = LIQ["high"], LIQ["low"]
        out.append({key: hi[key] + (lo[key] - hi[key]) * w for key in hi})
    return out


def generate(trend, case, seed):
    rng = np.random.default_rng([20261107, list(TRENDS).index(trend), LIQ_CASES.index(case), seed])
    liq = liquidity_path(case)
    n = WARM + EVAL
    O, C = np.empty(n), np.empty(n)
    h = 1.0
    logp = math.log(START_PRICE)
    anchor = None
    O[0] = C[0] = START_PRICE
    for t in range(1, n):
        z1 = rng.standard_t(DF) / math.sqrt(DF / (DF - 2))
        z2 = rng.standard_t(DF) / math.sqrt(DF / (DF - 2))
        sd = BASE_VOL * liq[t]["vol"] * math.sqrt(h)
        if t < WARM:
            mu = WARM_DRIFT
        elif trend == "sideways":
            if anchor is None:
                anchor = logp
            mu = -SIDEWAYS_KAPPA * (logp - anchor)
        else:
            mu = TRENDS[trend]
        g = liq[t]["gap"]
        z = math.sqrt(g) * z1 + math.sqrt(1 - g) * z2
        prev = C[t - 1]
        O[t] = prev * math.exp(mu * g - 0.5 * g * sd * sd + sd * math.sqrt(g) * z1)
        C[t] = O[t] * math.exp(mu * (1 - g) - 0.5 * (1 - g) * sd * sd + sd * math.sqrt(1 - g) * z2)
        h = (1 - GARCH_A - GARCH_B) + GARCH_A * z * z + GARCH_B * h
        logp = math.log(C[t])
    return O, C, liq


def inverse_path(O, C):
    n = len(C)
    IO, IC = np.empty(n), np.empty(n)
    IO[0] = IC[0] = 10000.0
    for t in range(1, n):
        IO[t] = IC[t - 1] * (2 - O[t] / C[t - 1]) * (1 + INV_DRAG)
        IC[t] = IC[t - 1] * (2 - C[t] / C[t - 1]) * (1 + INV_DRAG)
    return IO, IC


def targets(O, C, models):
    F = LI.features([None] * len(C), dict(Ca=list(C)))
    rows = list(range(WARM, len(C) - 1))
    X = np.asarray([F[i] for i in rows])
    pl, ps = models["long"].predict(X), models["inv"].predict(X)
    e = {m: CT.ema(list(C), m) for m in (8, 21, 50, 200)}
    out = {m: {} for m in METHODS}
    sl = si = None
    for i, a, b in zip(rows, pl, ps):
        out["hold"][i] = L
        out["ribbon"][i] = L if e[8][i] > e[21][i] > e[50][i] > e[200][i] else None
        sl = SH.policy_step(sl, float(a), float(b), False)
        si = SH.policy_step(si, float(a), float(b), True)
        out["learn_long"][i] = sl
        out["learn_inv"][i] = si
    return out, dict(pred_long_mean=float(pl.mean()), pred_long_pos=float((pl > 0).mean()), pred_inv_pref=float(((ps > pl) & (ps > 0)).mean()))


def ledger(O, C, IO, IC, liq, tg):
    cash, ql, qi = CAPITAL, 0.0, 0.0
    peak, mdd, trades, cost_paid = CAPITAL, 0.0, 0, 0.0
    days_l = days_s = 0
    for i in range(WARM, len(C) - 1):
        t = tg[i]
        j = i + 1
        c = FEE + liq[j]["cost_bp"] / 10000.0
        if ql > 0 and t != L:
            v = ql * O[j]
            cash += v * (1 - c)
            cost_paid += v * c
            ql = 0.0
            trades += 1
        if qi > 0 and t != S:
            v = qi * IO[j]
            cash += v * (1 - c)
            cost_paid += v * c
            qi = 0.0
            trades += 1
        if t == L and ql == 0 and cash > 0:
            ql = cash * (1 - c) / O[j]
            cost_paid += cash * c
            cash = 0.0
            trades += 1
        if t == S and qi == 0 and cash > 0:
            qi = cash * (1 - c) / IO[j]
            cost_paid += cash * c
            cash = 0.0
            trades += 1
        days_l += ql > 0
        days_s += qi > 0
        v = cash + ql * C[j] + qi * IC[j]
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    n = len(C) - 1
    c = FEE + liq[n]["cost_bp"] / 10000.0
    final = cash + (ql * C[n] + qi * IC[n]) * (1 - c)
    return dict(ret=float(100 * (final / CAPITAL - 1)), mdd=float(100 * mdd), trades=int(trades), cost_pct=float(100 * cost_paid / CAPITAL), days_long=int(days_l), days_inv=int(days_s))


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    if spec["inputs"]["learn_shadow_sha256"] != sha(SH.__file__):
        raise SystemExit("learn_shadow hash differs")
    models, meta = SH.load_models()
    paths, cells = [], {}
    for trend in TRENDS:
        for case in LIQ_CASES:
            for seed in range(SEEDS):
                O, C, liq = generate(trend, case, seed)
                IO, IC = inverse_path(O, C)
                tg, diag = targets(O, C, models)
                r = {m: ledger(O, C, IO, IC, liq, tg[m]) for m in METHODS}
                market = float(100 * (C[-1] / C[WARM] - 1))
                paths.append(dict(trend=trend, liquidity=case, seed=seed, market_eval_pct=market, diag=diag, results=r))
                cells.setdefault((trend, case), []).append(paths[-1])
    summary = []
    for (trend, case), ps in cells.items():
        row = dict(trend=trend, liquidity=case, paths=len(ps), market_median=float(np.median([p["market_eval_pct"] for p in ps])))
        for m in METHODS:
            row[m] = dict(median_ret=float(np.median([p["results"][m]["ret"] for p in ps])), median_mdd=float(np.median([p["results"][m]["mdd"] for p in ps])), median_trades=float(np.median([p["results"][m]["trades"] for p in ps])), median_cost_pct=float(np.median([p["results"][m]["cost_pct"] for p in ps])))
        for m in ("ribbon", "learn_long", "learn_inv"):
            row[m]["beats_hold"] = int(sum(p["results"][m]["ret"] > p["results"]["hold"]["ret"] for p in ps))
        for m in ("learn_long", "learn_inv"):
            row[m]["beats_ribbon"] = int(sum(p["results"][m]["ret"] > p["results"]["ribbon"]["ret"] for p in ps))
        summary.append(row)
    totals = {}
    for m in ("ribbon", "learn_long", "learn_inv"):
        totals[m] = dict(beats_hold=int(sum(p["results"][m]["ret"] > p["results"]["hold"]["ret"] for p in paths)), paths=len(paths))
    for m in ("learn_long", "learn_inv"):
        totals[m]["beats_ribbon"] = int(sum(p["results"][m]["ret"] > p["results"]["ribbon"]["ret"] for p in paths))
    rec = dict(version="scenario-test-v1", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], models_meta_sha256=SH.META_SHA, summary=summary, totals=totals, paths=paths)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    for row in summary:
        print(f"{row['trend']:8s} {row['liquidity']:11s} market {row['market_median']:+7.1f}% | " + " | ".join(f"{m} {row[m]['median_ret']:+7.1f}% dd {row[m]['median_mdd']:6.1f} tr {row[m]['median_trades']:4.0f}" + (f" >hold {row[m]['beats_hold']}/10" if m != 'hold' else "") for m in METHODS))
    print("totals", json.dumps(totals))
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
