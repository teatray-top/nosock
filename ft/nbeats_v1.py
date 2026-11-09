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
import harvest_v5 as H5
import harvest_v8 as H8
import kalman_fft_v1 as K1
import kalman_fft_v2 as K2
import kalman_fft_v3 as K3
import turning_learn_v1 as TL
import nbeats_fourier_v1 as NF

L, H = NF.L, NF.H
TREND_DEGREE = 3
TREND_UNITS = 256
SEAS_UNITS = 512
LAYERS = 4
BLOCKS_PER_STACK = 3
EPOCHS = 6
BATCH = 1024
LR = 1e-3
TEST_YEARS = list(range(2008, 2027))
RUN_FROM, RUN_TO = NF.RUN_FROM, NF.RUN_TO
SEEDS = 5
LONG = H5.LONG


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def trend_basis(n):
    t = np.arange(n, dtype=float) / n
    return torch.tensor(np.stack([t ** i for i in range(TREND_DEGREE + 1)]), dtype=torch.float32)


def seas_basis(n):
    t = np.arange(n, dtype=float) / n
    k = n // 2 - 1
    rows = [np.cos(2 * np.pi * i * t) for i in range(k)] + [np.sin(2 * np.pi * i * t) for i in range(k)]
    return torch.tensor(np.stack(rows), dtype=torch.float32)


class Block(torch.nn.Module):
    def __init__(self, units, Bb, Bf):
        super().__init__()
        layers = [torch.nn.Linear(L, units), torch.nn.ReLU()]
        for _ in range(LAYERS - 1):
            layers += [torch.nn.Linear(units, units), torch.nn.ReLU()]
        self.fc = torch.nn.Sequential(*layers)
        self.register_buffer("Bb", Bb)
        self.register_buffer("Bf", Bf)
        self.tb = torch.nn.Linear(units, Bb.shape[0], bias=False)
        self.tf = torch.nn.Linear(units, Bf.shape[0], bias=False)

    def forward(self, x):
        h = self.fc(x)
        return self.tb(h) @ self.Bb, self.tf(h) @ self.Bf


class NBeats(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trend = Block(TREND_UNITS, trend_basis(L), trend_basis(H))
        self.seas = Block(SEAS_UNITS, seas_basis(L), seas_basis(H))

    def forward(self, x):
        res = x
        fc = torch.zeros(x.shape[0], H)
        parts = {}
        for name, blk in (("trend", self.trend), ("seasonality", self.seas)):
            part = torch.zeros(x.shape[0], H)
            for _ in range(BLOCKS_PER_STACK):
                b, f = blk(res)
                res = res - b
                part = part + f
            parts[name] = part
            fc = fc + part
        return fc, parts


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
    net = NBeats()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    n = len(X)
    last = None
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        tot = 0.0
        for k in range(0, n, BATCH):
            idx = perm[k:k + BATCH]
            fc, _ = net(X[idx])
            loss = torch.mean((fc - Y[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        last = tot / n
    return net, n, last


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
    days, lp, (Xk, Yk, Sk, Ek, Tk), pool = NF.build_pool()
    print("pool series", len(pool), "samples", sum(len(p[0]) for p in pool), flush=True)
    pred = {h: np.full(len(days), np.nan) for h in (1, 5, 20)}
    seas_amp = np.full(len(days), np.nan)
    folds = []
    r = np.diff(lp)
    for y in TEST_YEARS:
        net, ntr, loss = train(pool, f"{y}-01-01", 2000 + y)
        idx = [t for t in range(L - 1, len(days)) if days[t][:4] == str(y)]
        xs, sds = [], []
        for t in idx:
            sd = max(float(np.std(r[t - L + 1:t])), 1e-4)
            xs.append((lp[t - L + 1:t + 1] - lp[t]) / sd)
            sds.append(sd)
        with torch.no_grad():
            fc, parts = net(torch.tensor(np.asarray(xs, np.float32)))
        sc = np.asarray(sds)[:, None]
        fc = fc.numpy() * sc
        for h in (1, 5, 20):
            pred[h][idx] = fc[:, h - 1]
        seas_amp[idx] = (parts["seasonality"].numpy() * sc).std(axis=1)
        real20 = np.array([lp[t + 20] - lp[t] if t + 20 < len(lp) else np.nan for t in idx])
        real5 = np.array([lp[t + 5] - lp[t] if t + 5 < len(lp) else np.nan for t in idx])
        m = ~np.isnan(real20)
        m5 = ~np.isnan(real5)
        folds.append(dict(year=y, train_samples=int(ntr), final_train_mse=loss, corr5=float(np.corrcoef(fc[m5, 4], real5[m5])[0, 1]) if m5.sum() > 10 else None, hit5=float(np.mean(np.sign(fc[m5, 4]) == np.sign(real5[m5]))) if m5.sum() else None, corr20=float(np.corrcoef(fc[m, 19], real20[m])[0, 1]) if m.sum() > 10 else None, hit20=float(np.mean(np.sign(fc[m, 19]) == np.sign(real20[m]))) if m.sum() else None, up5=float(np.mean(real5[m5] > 0)) if m5.sum() else None))
        print("fold", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in folds[-1].items()}, flush=True)
    oos = np.array([i for i, d in enumerate(days) if d >= RUN_FROM and i + 20 < len(lp)])
    diag = {}
    for h in (1, 5, 20):
        rr = np.array([lp[i + h] - lp[i] for i in oos])
        diag[f"corr{h}"] = float(np.corrcoef(pred[h][oos], rr)[0, 1])
        diag[f"hit{h}"] = float(np.mean(np.sign(pred[h][oos]) == np.sign(rr)))
        diag[f"up{h}"] = float(np.mean(rr > 0))
    print("OOS", diag, flush=True)
    raw = json.load(open(os.path.join(os.path.dirname(HERE), "data", "kr_etf", "kodex_daily_raw.json"), encoding="utf-8"))[LONG]
    k = dict(O=[float(x) for x in raw["O"][:len(days)]], C=[float(x) for x in raw["C"][:len(days)]])
    a = days.index(next(d for d in days if d >= RUN_FROM)) - 1
    b = days.index(RUN_TO)
    sub = days[a:b + 1]
    IO, IC, splice = TL.spliced_inverse(days, k, a, b)
    O, C = np.asarray(k["O"][a:b + 1]), np.asarray(k["C"][a:b + 1])
    IO, IC = np.asarray(IO), np.asarray(IC)
    cost = lambda leg, j: LI.FEE + (LI.TICK if leg == "L" else H8.tick(sub[j], IO[j])) / (O[j] if leg == "L" else IO[j])
    lev, _ = K2.kalman_all(np.log(np.asarray(k["C"], float)), 0.01)
    fk = K1.fft_features(lev, np.arange(len(k["C"])), 128, 3)
    p5 = pred[5]
    tg = dict(hold=np.array(["L"] * len(sub), dtype=object), nbeats_long_only=np.array(["L" if (not np.isnan(p5[a + j]) and p5[a + j] > 0) else None for j in range(len(sub))], dtype=object), nbeats_long_inverse=np.array([("L" if p5[a + j] > 0 else "I") if not np.isnan(p5[a + j]) else None for j in range(len(sub))], dtype=object), kfft_W128_K3_long_only=np.array(["L" if (not np.isnan(fk[a + j, 5]) and fk[a + j, 5] > 0) else None for j in range(len(sub))], dtype=object))
    windows = {"2008-2026": (0, len(sub) - 1), "2008-2012": (0, max(j for j, d in enumerate(sub) if d <= "2012-12-28")), "2013-2026": (min(j for j, d in enumerate(sub) if d >= "2013-01-02") - 1, len(sub) - 1)}
    res = {}
    for wn, (s0, s1) in windows.items():
        res[wn] = {}
        sl = slice(s0, s1 + 1)
        ok = np.ones(s1 - s0 + 1, bool)
        cw = lambda leg, j, s0=s0: cost(leg, j + s0)
        for name, t in tg.items():
            rr = K3.run_marked(t[sl], ok, O[sl], C[sl], IO[sl], IC[sl], cw, 0)
            if name != "hold":
                rr["random_shift_mean_total_pct"] = float(np.mean([K3.run_marked(K3.shifted(t[sl], ok, 800 + sd), ok, O[sl], C[sl], IO[sl], IC[sl], cw, 0)["total_pct"] for sd in range(SEEDS)]))
            res[wn][name] = rr
            print(f"{wn} {name:24s} total {rr['total_pct']:8.1f}% random {rr.get('random_shift_mean_total_pct', float('nan')):8.1f}% lowest {rr['lowest_pct']:6.1f} mdd {rr['mdd_pct']:6.1f} trades {rr['trades']}", flush=True)
    rec = dict(version="nbeats-v1", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], folds=folds, oos=diag, splice_day=splice, results=res, predictions={days[i]: [float(pred[1][i]), float(pred[5][i]), float(pred[20][i]), float(seas_amp[i])] for i in range(len(days)) if not np.isnan(pred[5][i])})
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
