import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "4"
import hashlib
import json
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import torch
import learn_inverse as LI
import compare_trend_v2 as CT
import learn_global_v1 as LG
import harvest_v5 as H5
import harvest_v8 as H8
import kalman_fft_v1 as K1
import kalman_fft_v2 as K2
import turning_learn_v1 as TL

L, H = 128, 20
PERIODS = [4.0 * (2 ** (k / 2)) for k in range(13)]
HID = 256
BLOCKS = 3
EPOCHS = 6
BATCH = 1024
LR = 1e-3
BACK_W = 0.1
STEP_OTHER = 2
TEST_YEARS = list(range(2008, 2027))
RUN_FROM, RUN_TO = "2008-01-02", "2026-11-05"
SEEDS = 5
CAPITAL = H5.CAPITAL
LONG, INV = H5.LONG, H5.INV


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def basis():
    tb = np.arange(-L + 1, 1, dtype=float)
    tf = np.arange(1, H + 1, dtype=float)

    def mk(t):
        cols = [np.ones_like(t), t / L, (t / L) ** 2]
        for P in PERIODS:
            cols += [np.cos(2 * np.pi * t / P), np.sin(2 * np.pi * t / P)]
        return np.stack(cols, axis=1)

    return torch.tensor(mk(tb), dtype=torch.float32), torch.tensor(mk(tf), dtype=torch.float32)


class Block(torch.nn.Module):
    def __init__(self, nb):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(L, HID), torch.nn.ReLU(), torch.nn.Linear(HID, HID), torch.nn.ReLU(), torch.nn.Linear(HID, nb))

    def forward(self, x):
        return self.net(x)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        Bb, Bf = basis()
        self.register_buffer("Bb", Bb)
        self.register_buffer("Bf", Bf)
        self.blocks = torch.nn.ModuleList([Block(Bb.shape[1]) for _ in range(BLOCKS)])
        self.unc = torch.nn.Sequential(torch.nn.Linear(L, HID), torch.nn.ReLU(), torch.nn.Linear(HID, H))

    def forward(self, x):
        res = x
        fc = torch.zeros(x.shape[0], H)
        for b in self.blocks:
            th = b(res)
            res = res - th @ self.Bb.T
            fc = fc + th @ self.Bf.T
        return fc, res, self.unc(x)


def samples(logp, days, step):
    X, Y, S, E, T = [], [], [], [], []
    r = np.diff(logp)
    for t in range(L - 1, len(logp) - H - 1 + 1, step):
        w = logp[t - L + 1:t + 1]
        sd = float(np.std(r[t - L + 1:t]))
        if not np.isfinite(sd) or sd < 1e-4:
            continue
        X.append((w - logp[t]) / sd)
        Y.append((logp[t + 1:t + H + 1] - logp[t]) / sd)
        S.append(sd)
        E.append(days[t + H])
        T.append(t)
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), np.asarray(S), np.asarray(E), np.asarray(T)


def build_pool():
    adj = json.load(open(CT.ADJ, encoding="utf-8"))[LONG]
    days = [d for d in adj["days"] if d <= RUN_TO]
    lp = np.log(np.asarray(adj["C"][:len(days)], float))
    Xk, Yk, Sk, Ek, Tk = samples(lp, days, 1)
    pool = [(Xk, Yk, Ek, "069500")]
    g = LG.load_global()
    for t, rec in g.items():
        C = np.asarray(rec["C"], float)
        dd = rec["days"]
        m = np.isfinite(C) & (C > 0)
        C, dd = C[m], [d for d, k in zip(dd, m) if k]
        if len(C) < L + H + 200:
            continue
        X, Y, S, E, T = samples(np.log(C), dd, STEP_OTHER)
        if len(X):
            pool.append((X, Y, E, t))
    return days, lp, (Xk, Yk, Sk, Ek, Tk), pool


def train(pool, cutoff, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xs, Ys = [], []
    for X, Y, E, _ in pool:
        m = E < cutoff
        Xs.append(X[m])
        Ys.append(Y[m])
    X = torch.tensor(np.vstack(Xs))
    Y = torch.tensor(np.vstack(Ys))
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    n = len(X)
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        for k in range(0, n, BATCH):
            idx = perm[k:k + BATCH]
            fc, res, lv = net(X[idx])
            lv = torch.clamp(lv, -8.0, 8.0)
            loss = torch.mean((fc - Y[idx]) ** 2 * torch.exp(-lv) + lv) + BACK_W * torch.mean(res ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net, n


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, p in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), p)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    days, lp, (Xk, Yk, Sk, Ek, Tk), pool = build_pool()
    print("pool series", len(pool), "samples", sum(len(p[0]) for p in pool), flush=True)
    pred5 = np.full(len(days), np.nan)
    pred20 = np.full(len(days), np.nan)
    sd5 = np.full(len(days), np.nan)
    sd20 = np.full(len(days), np.nan)
    curves = {}
    folds = []
    tpos = {int(t): k for k, t in enumerate(Tk)}
    for y in TEST_YEARS:
        cutoff = f"{y}-01-01"
        net, ntr = train(pool, cutoff, 1000 + y)
        idx = [t for t in range(L - 1, len(days)) if days[t][:4] == str(y)]
        xs, sds = [], []
        r = np.diff(lp)
        for t in idx:
            w = lp[t - L + 1:t + 1]
            sd = max(float(np.std(r[t - L + 1:t])), 1e-4)
            xs.append((w - lp[t]) / sd)
            sds.append(sd)
        with torch.no_grad():
            fc, _, lv = net(torch.tensor(np.asarray(xs, np.float32)))
        sc = np.asarray(sds)[:, None]
        fc = fc.numpy() * sc
        sdv = np.exp(0.5 * np.clip(lv.numpy(), -8, 8)) * sc
        pred5[idx] = fc[:, 4]
        pred20[idx] = fc[:, 19]
        sd5[idx] = sdv[:, 4]
        sd20[idx] = sdv[:, 19]
        for t, c in zip(idx, fc):
            curves[t] = c
        real20 = np.array([lp[t + 20] - lp[t] if t + 20 < len(lp) else np.nan for t in idx])
        m = ~np.isnan(real20)
        folds.append(dict(year=y, train_samples=int(ntr), corr20=float(np.corrcoef(fc[m, 19], real20[m])[0, 1]) if m.sum() > 10 else None, hit20=float(np.mean(np.sign(fc[m, 19]) == np.sign(real20[m]))) if m.sum() else None, up20=float(np.mean(real20[m] > 0)) if m.sum() else None))
        print("fold", folds[-1], flush=True)
    oos = np.array([i for i, d in enumerate(days) if d >= RUN_FROM and i + 20 < len(lp)])
    r5 = np.array([lp[i + 5] - lp[i] for i in oos])
    r20 = np.array([lp[i + 20] - lp[i] for i in oos])
    diag = dict(corr5=float(np.corrcoef(pred5[oos], r5)[0, 1]), corr20=float(np.corrcoef(pred20[oos], r20)[0, 1]), hit5=float(np.mean(np.sign(pred5[oos]) == np.sign(r5))), hit20=float(np.mean(np.sign(pred20[oos]) == np.sign(r20))), up5=float(np.mean(r5 > 0)), up20=float(np.mean(r20 > 0)))
    from math import erf, sqrt
    phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
    p5 = np.array([phi(abs(pred5[i]) / sd5[i]) for i in oos])
    p20 = np.array([phi(abs(pred20[i]) / sd20[i]) for i in oos])
    shape = []
    for i in oos:
        real = lp[i + 1:i + 21] - lp[i]
        c = curves[i]
        shape.append(float(np.corrcoef(c, real)[0, 1]) if np.std(c) > 0 and np.std(real) > 0 else np.nan)
    shape = np.asarray(shape)
    cal = []
    for lo, hi in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)):
        for hz, pp, pr, rr in (("5d", p5, pred5[oos], r5), ("20d", p20, pred20[oos], r20)):
            m = (pp >= lo) & (pp < hi)
            cal.append(dict(horizon=hz, band=[lo, hi], n=int(m.sum()), said=float(pp[m].mean()) if m.sum() else None, direction_followed=float(np.mean(np.sign(pr[m]) == np.sign(rr[m]))) if m.sum() else None, shape_corr_median=float(np.nanmedian(shape[m])) if m.sum() else None, shape_corr_ge_0_7=float(np.nanmean(shape[m] >= 0.7)) if m.sum() else None))
            c = cal[-1]
            print(f"CAL {hz:3s} model said {lo:.1f}-{min(hi, 1):.1f}: n {c['n']:5d} mean said {c['said'] if c['said'] is None else round(c['said'], 3)} direction followed {c['direction_followed'] if c['direction_followed'] is None else round(c['direction_followed'], 3)} shape corr median {c['shape_corr_median'] if c['shape_corr_median'] is None else round(c['shape_corr_median'], 3)} share shape>=0.7 {c['shape_corr_ge_0_7'] if c['shape_corr_ge_0_7'] is None else round(c['shape_corr_ge_0_7'], 3)}", flush=True)
    diag["calibration"] = cal
    diag["share_days_p5_ge_0_7"] = float(np.mean(p5 >= 0.7))
    print("OOS", {k: v for k, v in diag.items() if k != "calibration"}, flush=True)
    raw = json.load(open(os.path.join(os.path.dirname(HERE), "data", "kr_etf", "kodex_daily_raw.json"), encoding="utf-8"))[LONG]
    k = dict(O=[float(x) for x in raw["O"][:len(days)]], C=[float(x) for x in raw["C"][:len(days)]])
    a = days.index(next(d for d in days if d >= RUN_FROM)) - 1
    b = days.index(RUN_TO)
    sub = days[a:b + 1]
    IO, IC, splice = TL.spliced_inverse(days, k, a, b)
    O = np.asarray(k["O"][a:b + 1])
    C = np.asarray(k["C"][a:b + 1])
    IO, IC = np.asarray(IO), np.asarray(IC)
    cost = lambda leg, j: LI.FEE + (LI.TICK if leg == "L" else H8.tick(sub[j], IO[j])) / (O[j] if leg == "L" else IO[j])
    import kalman_fft_v3 as K3
    rawC = np.log(np.asarray(k["C"], float))
    lev, _ = K2.kalman_all(rawC, 0.01)
    fk = K1.fft_features(lev, np.arange(len(rawC)), 128, 3)
    tg = {}
    tg["nb_long_only"] = np.array(["L" if (not np.isnan(pred5[a + j]) and pred5[a + j] > 0) else None for j in range(len(sub))], dtype=object)
    tg["nb_long_inverse"] = np.array([("L" if pred5[a + j] > 0 else "I") if not np.isnan(pred5[a + j]) else None for j in range(len(sub))], dtype=object)
    from math import erf as _erf, sqrt as _sqrt
    conf = np.array([0.5 * (1 + _erf(abs(pred5[a + j]) / sd5[a + j] / _sqrt(2))) if not np.isnan(pred5[a + j]) else np.nan for j in range(len(sub))])
    tg["nb_conf70_long_only"] = np.array(["L" if (not np.isnan(conf[j]) and conf[j] >= 0.7 and pred5[a + j] > 0) else None for j in range(len(sub))], dtype=object)
    tg["nb_conf70_long_inverse"] = np.array([(("L" if pred5[a + j] > 0 else "I") if (not np.isnan(conf[j]) and conf[j] >= 0.7) else None) for j in range(len(sub))], dtype=object)
    tg["kfft_W128_K3_long_only"] = np.array(["L" if (not np.isnan(fk[a + j, 5]) and fk[a + j, 5] > 0) else None for j in range(len(sub))], dtype=object)
    tg["hold"] = np.array(["L"] * len(sub), dtype=object)
    windows = {"2008-2026": (0, len(sub) - 1), "2008-2012": (0, max(j for j, d in enumerate(sub) if d <= "2012-12-28")), "2013-2026": (min(j for j, d in enumerate(sub) if d >= "2013-01-02") - 1, len(sub) - 1)}
    res = {}
    for wn, (s0, s1) in windows.items():
        res[wn] = {}
        sl = slice(s0, s1 + 1)
        for name, t in tg.items():
            ok = np.ones(s1 - s0 + 1, bool)
            r = K3.run_marked(t[sl], ok, O[sl], C[sl], IO[sl], IC[sl], lambda leg, j, s0=s0: cost(leg, j + s0), 0)
            if name != "hold":
                rnd = [K3.run_marked(K3.shifted(t[sl], ok, 700 + sd), ok, O[sl], C[sl], IO[sl], IC[sl], lambda leg, j, s0=s0: cost(leg, j + s0), 0)["total_pct"] for sd in range(SEEDS)]
                r["random_shift_mean_total_pct"] = float(np.mean(rnd))
            res[wn][name] = r
            print(f"{wn} {name:24s} total {r['total_pct']:8.1f}% random {r.get('random_shift_mean_total_pct', float('nan')):8.1f}% lowest {r['lowest_pct']:6.1f} mdd {r['mdd_pct']:6.1f} trades {r['trades']}", flush=True)
    rec = dict(version="nbeats-fourier-v1", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], periods=PERIODS, folds=folds, oos=diag, splice_day=splice, results=res, predictions={days[i]: [float(pred5[i]), float(pred20[i])] for i in range(len(days)) if not np.isnan(pred5[i])})
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
