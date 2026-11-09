import sys

import numpy as np
import torch

from sim import Data, run_days


def check(X, ret, h, stop_pct, k=300, log=print):
    fails = []
    C, O = X.C, X.O
    last = C[:, -1]
    both = (X.oc > 0) & (last > 0)
    gap = float(torch.median(torch.abs(torch.log(X.oc[both] / last[both]))))
    log(f"check: official close vs last bar median |log| {gap:.5f}")
    if gap > 0.02:
        fails.append(f"daily close misaligned ({gap:.4f})")
    if not bool((X.pc[1:] == X.oc[:-1]).all()):
        fails.append("pc != previous official close")
    d = slice(1, X.D)
    o1 = torch.where(O[d, 1] > 0, O[d, 1], C[d, 0]).clamp_min(1e-4)
    fut = torch.log(last[d].clamp_min(1e-4) / o1)
    ok = X.buyable[d, 0] & (last[d] > 0) & (O[d, 1] > 0)
    F0 = X.F[d, 0]
    worst = 0.0
    for i in range(F0.shape[-1]):
        a, b = F0[..., i][ok].double(), fut[ok]
        if a.std() > 0 and len(a) > 100:
            c = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
            worst = max(worst, abs(c))
            if abs(c) > 0.3:
                fails.append(f"feature {i} at bar 0 correlates {c:+.2f} with the day's future return")
    log(f"check: max |corr(feature@bar0, future day return)| {worst:.3f}")
    N = X.N
    v = torch.nonzero(~torch.isnan(ret) & X.buyable)
    g = torch.Generator().manual_seed(0)
    idx = v[torch.randperm(len(v), generator=g)[:k].to(v.device)]
    diffs = []
    for d0, b0, n0 in idx.tolist():
        st = {}

        def act(env, per, g_, mk):
            b, pos = int(env.b[0]), int(env.pos[0])
            if pos < N:
                c = float(X.C[d0, b, pos])
                if b - st["b"] >= h or c <= float(env.entry[0]) * (1 - stop_pct) or (X.us and X.tp > 0 and c >= float(env.tpp[0])):
                    return torch.tensor([N], device=X.dev)
                return env.pos.clone()
            if b == b0 and "b" not in st and bool(mk[0, n0]):
                st["b"] = b
                st["inv"] = None
                return torch.tensor([n0], device=X.dev)
            return torch.tensor([N], device=X.dev)

        rows = run_days(X, [d0], act, flat=True)
        if "b" not in st or rows[0]["trades"] == 0:
            continue
        exp = X.deposit * (np.exp(float(ret[d0, b0, n0])) - 1)
        diffs.append((rows[0]["pnl"] - exp) / X.deposit)
    diffs = np.abs(np.array(diffs))
    log(f"check: label vs env pnl, n {len(diffs)}, median |diff| {np.median(diffs):.5f} of deposit, p95 {np.percentile(diffs, 95):.5f}, max {diffs.max():.5f}")
    if np.median(diffs) > 0.003 or np.percentile(diffs, 95) > 0.02:
        fails.append("label/env fill mismatch")
    for f in fails:
        log("CHECK FAIL: " + f)
    return fails


if __name__ == "__main__":
    import reader as R
    iv = sys.argv[1]
    h = int(sys.argv[2]) if len(sys.argv) > 2 else (36 if iv.endswith("5m") else 2)
    X = Data(R.DEV, iv)
    ret, stop, hor = R.labels(X, h)
    sys.exit(1 if check(X, ret, h, R.STOP) else 0)
