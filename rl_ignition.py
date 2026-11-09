import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("SCALP_TP", "0")
os.environ.setdefault("READER_STOP", "0.03")
import sim
from sim import Data, Env, run_days, summary
from train import Policy

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IV = os.environ.get("BT_IV", "usm5m")
TEST, VAL = int(os.environ.get("RL_TEST", 42)), int(os.environ.get("RL_VAL", 21))
VR, RR, MINPX = float(os.environ.get("RL_VR", 2)), float(os.environ.get("RL_RR", 0.01)), float(os.environ.get("RL_MINPX", 5))
STOP = float(os.environ.get("READER_STOP", 0.03))
HOLD_MAX = int(os.environ.get("RL_HOLD_MAX", 36))


def to_dev(X, dev):
    for k, v in list(vars(X).items()):
        if torch.is_tensor(v):
            setattr(X, k, v.to(dev))
    X.dev = dev
    return X


def signals(X):
    C = X.C.double()
    pc = X.pc.double()[:, None, :].expand_as(C)
    Cs = torch.where(C > 0, C, pc).clamp_min(1e-4)
    lg = torch.log(Cs)
    r1 = lg - torch.cat([torch.log(pc[:, :1]), lg[:, :-1]], 1)
    cnt = torch.arange(1, X.B + 1, dtype=torch.float64, device=C.device)[None, :, None]
    vr = X.V.double() / (torch.cumsum(X.V.double(), 1) / cnt).clamp_min(1)
    bar = torch.arange(X.B, device=C.device)[None, :, None].expand_as(Cs)
    return (bar >= 2) & (vr >= VR) & (r1 >= RR) & (Cs >= MINPX), r1.float()


class SigEnv(Env):
    def mask(self):
        m = super().mask()
        X = self.X
        keep = torch.zeros_like(m)
        keep[:, :-1] = FIRE[self.d, self.b]
        keep[:, X.N] = True
        keep[torch.arange(self.E, device=X.dev), self.pos] = True
        m = m & keep
        forced = self.forced_exit()
        m[forced] = False
        m[forced, X.N] = True
        return m

    def forced_exit(self):
        X = self.X
        inpos = self.pos < X.N
        j = self.pos.clamp_max(X.N - 1)
        c = X.C[self.d, self.b, j]
        stop = inpos & (c > 0) & (c <= self.entry * (1 - STOP))
        too_long = inpos & (self.b - self.held_b >= HOLD_MAX)
        return stop | too_long

    def reset(self, day=None, cap=None):
        self.held_b = torch.zeros(self.E, dtype=torch.long, device=self.X.dev)
        return super().reset(day, cap)

    def step(self, k):
        X = self.X
        was = self.pos.clone()
        r, done = super().step(k)
        self.held_b = torch.where((self.pos < X.N) & (self.pos != was), self.b.clone(), self.held_b)
        return r, done


sim.Env = SigEnv


def act_fn(pol, stochastic=False):
    def f(env, per, g, m):
        with torch.no_grad():
            lo, _ = pol(per, g, m)
        return torch.distributions.Categorical(logits=lo).sample() if stochastic else lo.argmax(-1)
    return f


def rule_fn(X, vr, rr, h):
    held = {}

    def f(env, per, g, m):
        d, b, pos = int(env.d[0]), int(env.b[0]), int(env.pos[0])
        if pos < X.N:
            if b - held["b"] >= h or bool(env.forced_exit()[0]):
                return torch.tensor([X.N], device=X.dev)
            return env.pos.clone()
        sc = torch.where(FIRE_RULE[d, b], R1[d, b], torch.zeros_like(R1[d, b])).masked_fill(~m[0, :-1], 0)
        j = int(sc.argmax())
        if float(sc[j]) > 0:
            held["b"] = b
            return torch.tensor([j], device=X.dev)
        return torch.tensor([X.N], device=X.dev)
    return f


def evaluate(X, days, f, seed=0):
    rows = run_days(X, days, f, flat=True)
    s = summary(rows)
    p = np.array([r["pnl"] for r in rows])
    return s, p, rows


if __name__ == "__main__":
    tag = sys.argv[1]
    iters = int(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    torch.manual_seed(seed)
    t0 = time.time()
    X = to_dev(Data("cpu", IV), DEV)
    FIRE, R1 = signals(X)
    FIRE_RULE = FIRE & (R1 >= 0.02)
    cnt = torch.arange(1, X.B + 1, dtype=torch.float64, device=DEV)[None, :, None]
    vr3 = X.V.double() / (torch.cumsum(X.V.double(), 1) / cnt).clamp_min(1) >= 3
    FIRE_RULE = FIRE_RULE & vr3
    days = list(range(1, X.D))
    te, va, tr = days[-TEST:], days[-TEST - VAL:-TEST], days[:-TEST - VAL]
    print(f"{IV} {X.D} days, train {len(tr)} val {len(va)} test {len(te)}; signals/day broad {float(FIRE[days].sum()) / len(days):.1f} rule(3x,+2%) {float(FIRE_RULE[days].sum()) / len(days):.1f}; stop {STOP} hold_max {HOLD_MAX}; dev {DEV}; {time.time() - t0:.0f}s", flush=True)
    out = {}
    for nm, ds in (("val", va), ("test", te)):
        s, p, _ = evaluate(X, ds, rule_fn(X, 3, 0.02, 12))
        out[f"rule_h12_{nm}"] = s
        print(f"baseline rule 3x/+2% clock60 stop3% on {nm}: pnl {s['total_pnl']:+.3f} ({s['total_pnl'] / X.deposit / len(ds) * 100:+.2f}%/day) trades/day {s['trades_per_day']:.2f} win {s['win_days']:.2f} ex-top3 {s['ex_top3']:+.3f}", flush=True)
    env = SigEnv(X, int(os.environ.get("RL_NENV", 128)), tr, seed)
    per, g, m = env.reset()
    pol = Policy(per.shape[-1], g.shape[-1]).to(DEV)
    opt = torch.optim.Adam(pol.parameters(), lr=3e-4)
    log = open(os.path.join(HERE, "runs", f"{tag}.log"), "a", encoding="utf-8")
    best = -float("inf")
    lam, clip = 0.95, 0.2
    for it in range(iters):
        t1 = time.time()
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
        V = torch.stack(B["v"]); Rw = torch.stack(B["r"])
        T = len(Rw)
        adv = torch.zeros_like(Rw)
        gg = torch.zeros(env.E, device=DEV)
        for k in reversed(range(T)):
            nv = V[k + 1] if k + 1 < T else torch.zeros_like(gg)
            gg = Rw[k] + nv - V[k] + lam * gg
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
        msg = f"it {it} day_ret {float(Rw.sum(0).mean()):+.3f}% trades {float(env.trades.mean()):.2f} {time.time() - t1:.1f}s"
        if it % 10 == 0 or it == iters - 1:
            s, p, _ = evaluate(X, va, act_fn(pol))
            msg += f" | val pnl {s['total_pnl']:+.3f} ({s['total_pnl'] / X.deposit / len(va) * 100:+.2f}%/d) trades/d {s['trades_per_day']:.2f} win {s['win_days']:.2f}"
            if s["total_pnl"] > best:
                best = s["total_pnl"]
                torch.save(pol.state_dict(), os.path.join(HERE, "runs", f"{tag}_best.pt"))
                msg += " *"
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()
    torch.save(pol.state_dict(), os.path.join(HERE, "runs", f"{tag}_last.pt"))
    for k in ("best", "last"):
        pol.load_state_dict(torch.load(os.path.join(HERE, "runs", f"{tag}_{k}.pt"), map_location=DEV))
        s, p, rows = evaluate(X, te, act_fn(pol))
        out[f"rl_{k}_test"] = s
        srt = np.sort(p)[::-1]
        print(f"RL {k} on test {X.days[te[0]]}..{X.days[te[-1]]}: pnl {s['total_pnl']:+.3f} ({s['total_pnl'] / X.deposit / len(te) * 100:+.2f}%/day) trades/day {s['trades_per_day']:.2f} win days {s['win_days']:.2f} zero {s['zero_days']} ex-top3 {s['total_pnl'] - srt[:3].sum():+.3f} median day {np.median(p):+.4f}", flush=True)
    json.dump(out, open(os.path.join(HERE, "runs", f"{tag}_result.json"), "w"), indent=1)
    print(f"done {time.time() - t0:.0f}s", flush=True)
