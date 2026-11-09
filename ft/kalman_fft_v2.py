import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
import sys
from collections import deque

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import learn_inverse as LI
import scalp_etf_v1 as S1
import harvest_v8 as H8
import harvest_v5 as H5
import kalman_fft_v1 as K1

SCALES = (5, 15, 30, 60, "1d")
WINDOWS = (16, 32, 64, 128)
HARMONICS = (1, 3)
LAMBDAS = (0.01, 0.1)
RULES = ("d1", "phase")
MODES = ("long_only", "long_inverse")
SEEDS = 5
ONLINE_N = 500
CAPITAL = 100_000_000.0
VAULT_SHARE = 0.5
MIN_TEST_FROM = S1.TEST_FROM
DAILY_TEST_FROM = "2013-01-02"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def minute_bars(B):
    d, _ = S1.load()
    ok = (d["mod"] <= d["ce"]) & (d["volume"] > 0) & (d["iv"] > 0)
    d = d[ok].copy()
    d["bar"] = ((d["mod"] - d["start"] - 1) // B).astype(int)
    g = d.groupby(["day", "bar"], sort=True)
    a = g.agg(O=("open", "first"), C=("close", "last"), IO=("io", "first"), IC=("ic", "last")).reset_index()
    return a["day"].to_numpy(), a["O"].to_numpy(float), a["C"].to_numpy(float), a["IO"].to_numpy(float), a["IC"].to_numpy(float), MIN_TEST_FROM


def daily_bars():
    raw = json.load(open(os.path.join(os.path.dirname(HERE), "data", "kr_etf", "kodex_daily_raw.json"), encoding="utf-8"))[H5.LONG]
    iv = json.load(open(H5.H2.INV_DAILY, encoding="utf-8"))[H5.INV]
    im = {d: (float(o), float(c)) for d, o, c in zip(iv["days"], iv["O"], iv["C"])}
    rows = [(d, float(o), float(c), im[d][0], im[d][1]) for d, o, c in zip(raw["days"], raw["O"], raw["C"]) if d in im and d <= H5.H2.REAL_TO]
    day = np.array([r[0] for r in rows])
    return day, np.array([r[1] for r in rows]), np.array([r[2] for r in rows]), np.array([r[3] for r in rows]), np.array([r[4] for r in rows]), DAILY_TEST_FROM


def kalman_all(y, lam):
    return K1.kalman(y, np.zeros(len(y), dtype=int), lam)


def run_bars(target, ok, day, O, IO, zero_cost=False):
    n = len(target)
    cash, vault, leg, q, basis = CAPITAL, 0.0, None, 0.0, 0.0
    nets = []

    def cost(lg, e):
        if zero_cost:
            return 0.0
        return LI.FEE + (LI.TICK if lg == "L" else H8.tick(day[e], IO[e])) / (O[e] if lg == "L" else IO[e])

    def close(e):
        nonlocal cash, vault, leg, q, basis
        P = O if leg == "L" else IO
        proceeds = q * P[e] * (1 - cost(leg, e))
        profit = proceeds - basis
        tv = VAULT_SHARE * profit if profit > 0 else 0.0
        vault += tv
        cash += proceeds - tv
        nets.append((leg, proceeds / basis - 1))
        leg, q, basis = None, 0.0, 0.0

    for t in range(n - 1):
        want = target[t] if ok[t] else None
        if want == leg:
            continue
        if leg is not None:
            close(t + 1)
        if want is not None:
            P = O if want == "L" else IO
            q = cash * (1 - cost(want, t + 1)) / P[t + 1]
            basis, cash, leg = cash, 0.0, want
    if leg is not None:
        close(n - 1)
    r = [x[1] for x in nets]
    return dict(total_pct=100 * ((cash + vault) / CAPITAL - 1), vault_pct=100 * vault / CAPITAL, trades=len(nets), mean_net_bp=1e4 * float(np.mean(r)) if r else None, win=float(np.mean([x > 0 for x in r])) if r else None, long_share=float(np.mean([x[0] == "L" for x in nets])) if nets else None)


def targets(f, rule, mode):
    tg = K1.state_targets(f, rule)
    if mode == "long_only":
        tg = np.array([x if x == "L" else None for x in tg], dtype=object)
    return tg


def evaluate(tg, ok, day, O, IO):
    r = run_bars(tg, ok, day, O, IO)
    z = run_bars(tg, ok, day, O, IO, zero_cost=True)
    idx = np.nonzero(ok)[0]
    rnd, rz = [], []
    for sd in range(SEEDS):
        rng = np.random.default_rng(500 + sd)
        tr = tg.copy()
        if len(idx) > 10:
            sub = tg[idx]
            off = int(rng.integers(len(idx) // 10, 9 * len(idx) // 10))
            tr[idx] = np.roll(sub, off)
        rnd.append(run_bars(tr, ok, day, O, IO)["total_pct"])
        zz = run_bars(tr, ok, day, O, IO, zero_cost=True)["mean_net_bp"]
        rz.append(zz if zz is not None else 0.0)
    return dict(costed=r, zero_cost_diagnostic=z, random_total_pct_mean=float(np.mean(rnd)), random_zero_cost_bp_mean=float(np.mean(rz)))


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, p in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), p)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    out = {}
    for B in SCALES:
        day, O, C, IO, IC = None, None, None, None, None
        day, O, C, IO, IC, test_from = daily_bars() if B == "1d" else minute_bars(B)
        y = np.log(C)
        pos = np.arange(len(C))
        test = day >= test_from
        hold = 100 * (C[-1] / O[np.argmax(test)] - 1)
        days_test = len(set(day[test]))
        rows = []
        feats = {}
        for lam in LAMBDAS:
            lev, _ = kalman_all(y, lam)
            for W in WINDOWS:
                for Kh in HARMONICS:
                    f = K1.fft_features(lev, pos, W, Kh)
                    feats[(W, Kh, lam)] = f
                    ok = test & ~np.isnan(f[:, 5])
                    for rule in RULES:
                        for mode in MODES:
                            tg = targets(f, rule, mode)
                            ev = evaluate(tg, ok, day, O, IO)
                            row = dict(scale=str(B), W=W, K=Kh, lam=lam, rule=rule, mode=mode, **ev)
                            rows.append(row)
                            c, z = ev["costed"], ev["zero_cost_diagnostic"]
                            print(f"{str(B):>3} {rule:5s} {mode:12s} W{W:<3d} K{Kh} L{lam:<4} trades {c['trades']:6d} ({c['trades'] / days_test:5.2f}/day) total {c['total_pct']:9.2f}% random {ev['random_total_pct_mean']:9.2f}% | gross {z['mean_net_bp'] if z['mean_net_bp'] is None else round(z['mean_net_bp'], 2)}bp random-gross {ev['random_zero_cost_bp_mean']:.2f}bp costed {c['mean_net_bp'] if c['mean_net_bp'] is None else round(c['mean_net_bp'], 2)}bp", flush=True)
        cfgs = list(feats)
        nxt_ret = np.r_[np.diff(np.log(O))[1:], [np.nan, np.nan]][:len(O)]
        sel_rows = []
        for rule in RULES:
            for mode in MODES:
                dq = [deque(maxlen=ONLINE_N) for _ in cfgs]
                sums = np.zeros(len(cfgs))
                tg = np.array([None] * len(C), dtype=object)
                chosen = np.full(len(C), -1)
                tgs = [targets(feats[c], rule, mode) for c in cfgs]
                for t in range(len(C)):
                    if t >= 2:
                        s = t - 2
                        for ci, c in enumerate(cfgs):
                            v = feats[c][s, 5]
                            if np.isnan(v) or np.isnan(nxt_ret[s]):
                                continue
                            h = 1.0 if np.sign(v) == np.sign(nxt_ret[s]) else 0.0
                            if len(dq[ci]) == ONLINE_N:
                                sums[ci] -= dq[ci][0]
                            dq[ci].append(h)
                            sums[ci] += h
                    sc = np.array([sums[ci] / len(dq[ci]) if len(dq[ci]) >= 100 else -1 for ci in range(len(cfgs))])
                    b = int(sc.argmax())
                    if sc[b] < 0:
                        continue
                    chosen[t] = b
                    tg[t] = tgs[b][t]
                ok = test & (chosen >= 0)
                ev = evaluate(tg, ok, day, O, IO)
                share = {f"W{c[0]}_K{c[1]}_L{c[2]}": float(np.mean(chosen[ok] == ci)) for ci, c in enumerate(cfgs)}
                row = dict(scale=str(B), rule=rule, mode=mode, online=True, chosen_share=share, **ev)
                sel_rows.append(row)
                c, z = ev["costed"], ev["zero_cost_diagnostic"]
                print(f"{str(B):>3} ONLINE {rule:5s} {mode:12s} trades {c['trades']:6d} total {c['total_pct']:9.2f}% random {ev['random_total_pct_mean']:9.2f}% | gross {z['mean_net_bp'] if z['mean_net_bp'] is None else round(z['mean_net_bp'], 2)}bp random-gross {ev['random_zero_cost_bp_mean']:.2f}bp top {sorted(share.items(), key=lambda q: -q[1])[:2]}", flush=True)
        out[str(B)] = dict(bars=int(len(C)), test_from=test_from, test_days=days_test, hold_pct=hold, fixed=rows, online=sel_rows)
        print(f"{str(B):>3} HOLD 069500 over test {hold:.2f}%", flush=True)
    rec = dict(version="kalman-fft-v2", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], results=out)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
