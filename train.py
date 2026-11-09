import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sim
from sim import Data, Env, run_days, summary

DEV = "cuda" if torch.cuda.is_available() else "cpu"
IV = os.environ.get("PENNY_IV", "5m")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


class Policy(nn.Module):
    def __init__(self, fp, fg, h=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(fp + fg, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
        self.cash = nn.Sequential(nn.Linear(fg, h), nn.GELU(), nn.Linear(h, 1))
        self.v = nn.Sequential(nn.Linear(fg + 2 * fp, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, per, g, m):
        x = torch.cat([per, g[:, None].expand(-1, per.shape[1], -1)], -1)
        lo = torch.cat([self.enc(x).squeeze(-1), self.cash(g)], -1)
        lo = lo.masked_fill(~m, -1e9)
        mm = m[:, :-1, None].float()
        pool = (per * mm).sum(1) / mm.sum(1).clamp_min(1)
        hp = (per * per[..., -2:-1]).sum(1)
        return lo, self.v(torch.cat([g, pool, hp], -1)).squeeze(-1)


def splits(data):
    days = list(range(1, data.D))
    n_test = max(10, len(days) // 3)
    n_val = max(6, len(days) // 6)
    return days[:-(n_test + n_val)], days[-(n_test + n_val):-n_test], days[-n_test:]


def greedy(pol):
    def f(env, per, g, m):
        with torch.no_grad():
            lo, _ = pol(per, g, m)
        return lo.argmax(-1)
    return f


def baselines(data, days):
    N = data.N
    dev = data.dev
    cashk = lambda: torch.tensor([N], device=dev)

    def cash(env, per, g, m):
        return cashk()

    def chaser(rand):
        rng = np.random.default_rng(0)

        def f(env, per, g, m):
            b = int(env.b[0])
            pos = int(env.pos[0])
            if pos < N:
                u = float(data.C[env.d[0], b, pos]) / float(env.entry[0]) - 1
                if u <= -0.03 or u >= 0.05:
                    return cashk()
                return env.pos.clone()
            if b < 6 or b > 60 or int(env.trades[0]) >= 3:
                return cashk()
            rday = per[0, :, 4] / 10
            cand = m[0, :-1] & (rday > 0.05) & (rday < 0.20) & (per[0, :, 6] > 1.0)
            idx = torch.nonzero(cand).flatten()
            if len(idx) == 0:
                return cashk()
            pick = idx[int(rng.integers(len(idx)))] if rand else idx[rday[idx].argmax()]
            return pick[None]
        return f

    def rnd():
        rng = np.random.default_rng(1)

        def f(env, per, g, m):
            b = int(env.b[0])
            if int(env.pos[0]) < N:
                return env.pos.clone() if rng.random() > 1 / 12 else cashk()
            idx = torch.nonzero(m[0, :-1]).flatten()
            if len(idx) == 0 or b > 60 or rng.random() > 1 / 12:
                return cashk()
            return idx[int(rng.integers(len(idx)))][None]
        return f

    out = {}
    for k, f in (("cash", cash), ("momentum_chaser", chaser(False)), ("random_chaser", chaser(True)), ("random", rnd())):
        out[k] = summary(run_days(data, days, f))
    return out


def train(tag, iters, n_env=128, lr=3e-4, seed=0):
    torch.manual_seed(seed)
    data = Data(DEV, IV)
    tr, va, te = splits(data)
    print("days train", len(tr), "val", len(va), "test", len(te), "stocks", data.N, flush=True)
    env = Env(data, n_env, tr, seed)
    per, g, m = env.reset()
    pol = Policy(per.shape[-1], g.shape[-1]).to(DEV)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    os.makedirs(OUT, exist_ok=True)
    log = open(os.path.join(OUT, f"{tag}.log"), "a", encoding="utf-8")
    best = -float("inf")
    lam, clip = 0.95, 0.2
    for it in range(iters):
        t0 = time.time()
        per, g, m = env.reset()
        B = dict(per=[], g=[], m=[], a=[], lp=[], v=[], r=[])
        with torch.no_grad():
            while True:
                lo, v = pol(per, g, m)
                dist = torch.distributions.Categorical(logits=lo)
                a = dist.sample()
                B["per"].append(per); B["g"].append(g); B["m"].append(m); B["a"].append(a)
                B["lp"].append(dist.log_prob(a)); B["v"].append(v)
                r, done = env.step(a)
                B["r"].append(r * 100)
                if bool(done.all()):
                    break
                per, g, m = env.obs()
        V = torch.stack(B["v"]); R = torch.stack(B["r"])
        T = len(R)
        adv = torch.zeros_like(R)
        gg = torch.zeros(n_env, device=DEV)
        for k in reversed(range(T)):
            nv = V[k + 1] if k + 1 < T else torch.zeros_like(gg)
            gg = R[k] + nv - V[k] + lam * gg
            adv[k] = gg
        ret = adv + V
        fl = lambda x: torch.stack(x).flatten(0, 1)
        P_, G_, M_, A_, LP_ = fl(B["per"]), fl(B["g"]), fl(B["m"]), fl(B["a"]), fl(B["lp"])
        ADV, RET = adv.flatten(), ret.flatten()
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        n = len(ADV)
        for _ in range(3):
            perm = torch.randperm(n, device=DEV)
            for s in range(0, n, 1024):
                i = perm[s:s + 1024]
                lo, v = pol(P_[i], G_[i], M_[i])
                dist = torch.distributions.Categorical(logits=lo)
                ratio = torch.exp(dist.log_prob(A_[i]) - LP_[i])
                pl = -torch.min(ratio * ADV[i], ratio.clamp(1 - clip, 1 + clip) * ADV[i]).mean()
                loss = pl + 0.5 * F.mse_loss(v, RET[i]) - 0.003 * dist.entropy().mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(pol.parameters(), 0.5)
                opt.step()
        msg = f"it {it} day_ret {float(R.sum(0).mean()):.3f}% trades {float(env.trades.mean()):.1f} {time.time()-t0:.1f}s"
        if it % 20 == 0 or it == iters - 1:
            s = summary(run_days(data, va, greedy(pol)))
            msg += f" | val pnl {s['total_pnl']:.0f} win {s['win_days']:.2f} trades {s['trades_per_day']:.1f}"
            if s["total_pnl"] > best:
                best = s["total_pnl"]
                torch.save(pol.state_dict(), os.path.join(OUT, f"{tag}_best.pt"))
                json.dump(dict(it=it, val=s), open(os.path.join(OUT, f"{tag}_best.json"), "w"), indent=1)
                msg += " *"
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()
    torch.save(pol.state_dict(), os.path.join(OUT, f"{tag}_last.pt"))


def test(tag):
    data = Data(DEV, IV)
    tr, va, te = splits(data)
    env = Env(data, 1, te)
    per, g, m = env.reset()
    res = {}
    for k in ("best", "last"):
        f = os.path.join(OUT, f"{tag}_{k}.pt")
        if os.path.exists(f):
            pol = Policy(per.shape[-1], g.shape[-1]).to(DEV)
            pol.load_state_dict(torch.load(f, map_location=DEV))
            rows = run_days(data, te, greedy(pol))
            res[f"dobby_{k}"] = summary(rows)
            json.dump(rows, open(os.path.join(OUT, f"{tag}_{k}_days.json"), "w"), indent=1)
    res.update(baselines(data, te))
    for k, v in res.items():
        print(f"{k:16s} " + " ".join(f"{a}={b:,.1f}" if isinstance(b, float) else f"{a}={b}" for a, b in v.items()))
    json.dump(res, open(os.path.join(OUT, f"{tag}_test.json"), "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[-2] == "--slip":
        sim.SLIP = float(sys.argv[-1])
        sys.argv = sys.argv[:-2]
    if sys.argv[1] == "train":
        train(sys.argv[2], int(sys.argv[3]), seed=int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    elif sys.argv[1] == "test":
        test(sys.argv[2])
    elif sys.argv[1] == "base":
        data = Data(DEV, IV)
        tr, va, te = splits(data)
        print("slip ticks", sim.SLIP, "stocks", data.N, "days", len(tr), len(va), len(te), flush=True)
        for name, ds in (("train", tr), ("val", va), ("test", te)):
            for k, v in baselines(data, ds).items():
                print(name, f"{k:16s} " + " ".join(f"{a}={b:,.1f}" for a, b in v.items()), flush=True)
