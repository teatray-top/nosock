import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train import Policy

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PANEL = os.environ.get("DAILY_PANEL", os.path.join(HERE, "data", "panel_daily_v1.npz"))
DEPOSIT = float(os.environ.get("RL_DEPOSIT", 7.0))
FEE, SEC, MIN_FEE, TICK = 0.001, 2.06e-5, 0.01, 0.01
EPISODE = int(os.environ.get("RL_EPISODE", 63))
TEST_FRAC, VAL_FRAC = float(os.environ.get("RL_TEST_FRAC", 0.2)), float(os.environ.get("RL_VAL_FRAC", 0.1))


def ema(x, span):
    a = 2.0 / (span + 1)
    out = np.empty_like(x)
    s = np.nan_to_num(x[0])
    for t in range(len(x)):
        v = x[t]
        s = np.where(np.isnan(v), s, a * v + (1 - a) * s)
        out[t] = s
    return out


def roll(x, k, fn):
    out = np.full_like(x, np.nan)
    for t in range(k, len(x)):
        out[t] = fn(x[t - k:t], 0)
    return out


def features(C, V, adj=None, listed=None, dv_raw=None):
    D, N = C.shape
    adj = np.ones_like(C) if adj is None else adj
    listed = ~np.isnan(C) if listed is None else listed
    lc = np.log(C)
    r1_raw = np.diff(lc, axis=0, prepend=lc[:1])
    r1 = np.nan_to_num(r1_raw, nan=0.0)
    vol = np.sqrt(ema(r1 ** 2, 60)) + 1e-6
    f = []
    for k in (5, 21, 63, 126, 252):
        rk = lc - np.vstack([lc[:1].repeat(k, 0), lc[:-k]])
        f.append(np.clip(np.nan_to_num(rk) / (vol * np.sqrt(k)), -5, 5))
    price = np.nan_to_num(C, nan=1.0)
    std63 = roll(price, 63, np.nanstd) + 1e-6
    for s, l in ((8, 24), (16, 48), (32, 96)):
        q = (ema(price, s) - ema(price, l)) / std63
        f.append(np.clip(np.nan_to_num(q / (roll(q, 252, np.nanstd) + 1e-6)), -5, 5))
    gain, loss = np.clip(r1, 0, None), np.clip(-r1, 0, None)
    rsi = 100 - 100 / (1 + ema(gain, 30) / (ema(loss, 30) + 1e-9))
    f.append(rsi / 50 - 1)
    f.append(np.clip(np.log(vol * np.sqrt(252)) + 1.5, -5, 5))
    dv = np.log1p(np.nan_to_num(dv_raw if dv_raw is not None else C / adj * V))
    f.append(np.clip((dv - ema(dv, 60)) / 2, -5, 5))
    f.append(np.clip((dv - 17) / 3, -5, 5))
    mkt = np.nanmedian(np.where(listed, r1_raw, np.nan), axis=1, keepdims=True)
    mkt = np.nan_to_num(mkt)
    for k in (1, 5, 21):
        mk = np.cumsum(mkt, 0)
        mk = mk - np.vstack([np.zeros((k, 1)), mk[:-k]])
        f.append(np.repeat(np.clip(mk * 20 / np.sqrt(k), -5, 5), N, 1))
    f.append(np.clip(np.nan_to_num(r1 - mkt) * 20, -5, 5))
    F_ = np.stack(f, -1).astype(np.float32)
    rank_src = F_[..., 2]
    order = np.argsort(np.argsort(np.where(listed, rank_src, -1e9), 1), 1).astype(np.float64)
    n_listed = listed.sum(1, keepdims=True)
    order = order - (N - n_listed)
    rank = np.where(listed, order / np.maximum(n_listed - 1, 1) - 0.5, 0.0)
    F_ = np.concatenate([F_, rank[..., None].astype(np.float32)], -1)
    F_ = np.where(listed[..., None], F_, 0.0)
    return np.nan_to_num(F_)


HAIRCUT = float(os.environ.get("RL_DELIST_HAIRCUT", 0.10))


class DailyEnv:
    def __init__(self, O, C, elig, F_, days, n_env, seed=0, adj=None, listed=None):
        self.O, self.C, self.elig, self.F, self.days = O, C, elig, F_, days
        self.adj = adj if adj is not None else torch.ones_like(C)
        self.listed = listed if listed is not None else torch.ones_like(C, dtype=torch.bool)
        self.forced = torch.zeros(n_env, device=DEV)
        self.E, self.N = n_env, C.shape[1]
        self.g = torch.Generator(device="cpu").manual_seed(seed)
        self.d = torch.zeros(n_env, dtype=torch.long, device=DEV)
        self.end = torch.zeros(n_env, dtype=torch.long, device=DEV)
        self.pos = torch.full((n_env,), self.N, dtype=torch.long, device=DEV)
        self.sh = torch.zeros(n_env, dtype=torch.float64, device=DEV)
        self.cash = torch.full((n_env,), DEPOSIT, dtype=torch.float64, device=DEV)
        self.entry = torch.zeros(n_env, dtype=torch.float64, device=DEV)
        self.held = torch.zeros(n_env, dtype=torch.long, device=DEV)
        self.trades = torch.zeros(n_env, device=DEV)

    def reset(self, start=None, length=None):
        length = EPISODE if length is None else length
        if start is None:
            start = self.days[torch.randint(len(self.days) - length, (self.E,), generator=self.g)].to(DEV)
        self.d[:] = start
        self.end[:] = start + length
        self.start = self.d.clone()
        self.length = max(int(length), 1)
        self.pos[:] = self.N
        self.sh[:] = 0
        self.cash[:] = DEPOSIT
        self.entry[:] = 0
        self.held[:] = 0
        self.trades[:] = 0
        self.forced[:] = 0
        return self.obs()

    def equity(self, px):
        j = self.pos.clamp_max(self.N - 1)
        p = px[self.d, j]
        amt = self.sh * (p - TICK * self.adj[self.d, j]).clamp_min(0.01)
        v = amt - torch.floor(amt * FEE * 100) / 100 - torch.maximum(amt * SEC, torch.full_like(amt, MIN_FEE))
        return self.cash + torch.where(self.pos < self.N, v, torch.zeros_like(v))

    def mask(self):
        ok = self.elig[self.d] & (self.C[self.d] / self.adj[self.d] >= 5.0)
        m = torch.cat([ok, torch.ones(self.E, 1, dtype=torch.bool, device=DEV)], 1)
        m[torch.arange(self.E, device=DEV), self.pos] = True
        return m

    def obs(self):
        per = self.F[self.d]
        held = torch.zeros(self.E, self.N, device=DEV)
        inpos = self.pos < self.N
        e = torch.arange(self.E, device=DEV)
        held[e[inpos], self.pos[inpos]] = 1
        j = self.pos.clamp_max(self.N - 1)
        upnl = torch.where(inpos, torch.log(self.C[self.d, j].clamp_min(1e-4) / self.entry.clamp_min(1e-4)), torch.zeros_like(self.entry)).float()
        per = torch.cat([per, held[..., None], (held * upnl[:, None] * 10)[..., None]], -1)
        eq = self.equity(self.C)
        g = torch.stack([((self.d - self.start).float() / self.length), torch.log(eq / DEPOSIT).float() * 5, inpos.float(), (self.held.float() / 20).clamp_max(2), self.trades / 10], -1)
        return per, g, self.mask()

    def step(self, k):
        d1 = self.d + 1
        eq0 = self.equity(self.C)
        inpos = self.pos < self.N
        j = self.pos.clamp_max(self.N - 1)
        gone = inpos & ~self.listed[d1, j]
        k = torch.where(gone, torch.full_like(k, self.N), k)
        switch = k != self.pos
        o = self.O[d1, j]
        o = torch.where(gone, self.C[self.d, j] * (1 - HAIRCUT), o)
        sell = switch & inpos & ((torch.isfinite(o) & (o > 0)) | gone)
        amt = torch.where(gone, self.sh * o.clamp_min(0.0), self.sh * (o - TICK * self.adj[d1, j]).clamp_min(0.01))
        proceeds = (amt - torch.floor(amt * FEE * 100) / 100 - torch.maximum(amt * SEC, torch.full_like(amt, MIN_FEE))).clamp_min(0.0)
        self.cash = torch.where(sell, self.cash + proceeds, self.cash)
        self.sh = torch.where(sell, torch.zeros_like(self.sh), self.sh)
        self.pos = torch.where(sell, torch.full_like(self.pos, self.N), self.pos)
        self.forced += (sell & gone).float()
        kk = k.clamp_max(self.N - 1)
        ob = self.O[d1, kk]
        want = switch & (k < self.N) & (self.pos == self.N) & torch.isfinite(ob) & (ob > 0) & (self.cash >= 1.0)
        bp = ob + TICK * self.adj[d1, kk]
        amt = self.cash / (1 + FEE)
        n = amt / bp
        self.cash = torch.where(want, self.cash - amt - torch.floor(amt * FEE * 100) / 100, self.cash)
        self.sh = torch.where(want, n, self.sh)
        self.pos = torch.where(want, kk, self.pos)
        self.entry = torch.where(want, bp, self.entry)
        self.held = torch.where(want, torch.zeros_like(self.held), self.held + 1)
        self.trades += want.float()
        self.d = d1
        eq1 = self.equity(self.C)
        r = torch.log(eq1.clamp_min(1e-6) / eq0.clamp_min(1e-6)).float()
        return r, self.d >= self.end


def run_policy(env, days, f):
    rows = []
    for start in range(days[0], days[-1] - 1, EPISODE):
        length = min(EPISODE, days[-1] - start)
        per, g, m = env.reset(torch.full((env.E,), start, dtype=torch.long, device=DEV), length)
        eq0 = float(env.equity(env.C)[0])
        first = True
        while True:
            a = f(env, per, g, m)
            r, done = env.step(a)
            rows.append(dict(day=str(env.days_str[int(env.d[0])]), ret=float(r[0]), pos=int(env.pos[0]), trades=int(env.trades[0]), forced=int(env.forced[0]), ep_start=first))
            first = False
            if bool(done[0]):
                break
            per, g, m = env.obs()
    return rows


def summarize(name, rows):
    r = np.array([x["ret"] for x in rows])
    tot = float(np.exp(r.sum()) - 1)
    sharpe = float(r.mean() / (r.std() + 1e-9) * np.sqrt(252))
    eq = np.exp(np.cumsum(r))
    mdd = float((1 - eq / np.maximum.accumulate(eq)).max())
    pos = [x["pos"] for x in rows]
    buys = sum(1 for i in range(len(pos)) if x_is_stock(pos[i], rows) and (rows[i].get("ep_start") or pos[i] != pos[i - 1]))
    days_in = sum(1 for x in rows if x_is_stock(x["pos"], rows))
    eps = sum(1 for x in rows if x.get("ep_start"))
    forced = sum(x.get("forced", 0) for i, x in enumerate(rows) if (i + 1 == len(rows)) or rows[i + 1].get("ep_start"))
    hit = [x.get("forced", 0) > (0 if x.get("ep_start") else rows[i - 1].get("forced", 0)) for i, x in enumerate(rows)]
    r_ex = np.array([0.0 if h else x["ret"] for x, h in zip(rows, hit)])
    tot_ex = float(np.exp(r_ex.sum()) - 1)
    excluded = [x["day"] for x, h in zip(rows, hit) if h]
    if forced:
        print(f"   sensitivity: excluding {int(sum(hit))} assumed-settlement days (treated as unobservable, ret=0): total {tot_ex * 100:+.2f}% vs {tot * 100:+.2f}%", flush=True)
    print(f"== {name:28s} days {len(r)} total {tot * 100:+.2f}% ({float(r.mean()) * 100:+.3f}%/day) sharpe {sharpe:+.2f} maxDD {mdd * 100:.1f}% buys {buys} days-in-market {days_in} episodes {eps} assumed-settlement-exits {forced} (each restarts at ${DEPOSIT:.0f}; totals chain log-returns; settlement at last close x{1 - HAIRCUT:.2f} is an assumption)", flush=True)
    return dict(days=len(r), scenario_index_total=tot, total=None if forced else tot, per_day=float(r.mean()), sharpe=sharpe, mdd=mdd, buys=buys, days_in_market=days_in, episodes=eps, assumed_settlement_exits=forced, excluded_days=excluded, observable_rows_total=tot_ex, pnl_status="undetermined: unsettled exposure, scenario index only" if forced else "no assumed settlement occurred (not a verification of settlement)", settlement_assumption=f"last close x {1 - HAIRCUT:.2f}", settlement_verified=False)


def x_is_stock(p, rows):
    return p < N_SYMS


if __name__ == "__main__":
    tag, iters = sys.argv[1], int(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    torch.manual_seed(seed)
    t0 = time.time()
    import hashlib
    print(f"run {tag} seed {seed}: script sha256 {hashlib.sha256(open(__file__, 'rb').read()).hexdigest()[:16]} panel {os.path.basename(PANEL)} sha256 {hashlib.sha256(open(PANEL, 'rb').read()).hexdigest()[:16]} deposit {DEPOSIT} episode {EPISODE} test_frac {TEST_FRAC} val_frac {VAL_FRAC}", flush=True)
    zd = dict(np.load(PANEL, allow_pickle=True))
    if os.environ.get("RL_END_DAY"):
        nd = len(zd["days"])
        cut = int(np.searchsorted(zd["days"].astype(str), os.environ["RL_END_DAY"], side="right"))
        zd = {k: (v[:cut] if hasattr(v, "shape") and v.shape and v.shape[0] == nd and k != "syms" else v) for k, v in zd.items()}
        print(f"panel truncated at {os.environ['RL_END_DAY']}: {cut} days", flush=True)

    class Z:
        files = list(zd.keys())

        def __getitem__(self, k):
            return zd[k]
    z = Z()
    C, O, V, elig = z["C"], z["O"], z["V"], z["elig"]
    Cf = pd_ffill = None
    import pandas as pd
    Cf = pd.DataFrame(C).ffill().values
    adj_np = np.nan_to_num(z["adj"], nan=1.0) if "adj" in z.files else np.ones_like(C)
    cs_mask = (~np.isnan(C)) & elig.astype(bool)
    F_ = features(Cf, V, adj_np, cs_mask, z["dv"] if "dv" in z.files else None)
    print(f"cross-section mask = listed & point-in-time eligible (mean {cs_mask.sum(1).mean():.0f} names/day)", flush=True)
    if "dv" not in z.files:
        if os.environ.get("RL_ALLOW_LEGACY_PANEL") != "1":
            sys.exit("panel has no raw dollar-volume array (dv); rebuild it (panel_daily.py / fetch_etf_daily.py) or set RL_ALLOW_LEGACY_PANEL=1 for a reference-only run")
        print("WARNING legacy panel: dollar-volume feature = C/adj*V with split-adjusted V (units mixed); reference-only run", flush=True)
    else:
        print("dollar-volume feature source: raw dv array", flush=True)
    D, N = C.shape
    N_SYMS = N
    print(f"panel {D} days x {N} syms, features {F_.shape[-1]}; {time.time() - t0:.0f}s", flush=True)
    Ct = torch.tensor(np.nan_to_num(Cf, nan=1.0), dtype=torch.float64, device=DEV)
    Ot = torch.tensor(O, dtype=torch.float64, device=DEV)
    Ft = torch.tensor(F_, device=DEV)
    Et = torch.tensor(elig, device=DEV)
    At = torch.tensor(np.nan_to_num(z["adj"], nan=1.0) if "adj" in z.files else np.ones_like(C), dtype=torch.float64, device=DEV)
    print(f"accounting: raw price = adjusted / adj (adj present: {'adj' in z.files}); ticks and $5 floor on raw", flush=True)
    days = list(range(260, D - 1))
    n_test, n_val = int(len(days) * TEST_FRAC), int(len(days) * VAL_FRAC)
    tr, va, te = days[:-(n_test + n_val)], days[-(n_test + n_val):-n_test], days[-n_test:]
    print(f"train {z['days'][tr[0]]}..{z['days'][tr[-1]]} ({len(tr)}) val {z['days'][va[0]]}..{z['days'][va[-1]]} test {z['days'][te[0]]}..{z['days'][te[-1]]} ({len(te)})", flush=True)
    Lt = torch.tensor(~np.isnan(C), device=DEV)
    print(f"missing-data rule (ASSUMED settlement, not a verified bound): held name with no raw close next day is closed at last close x (1-{HAIRCUT}); a one-day data gap is treated the same as a delisting", flush=True)
    env = DailyEnv(Ot, Ct, Et, Ft, torch.tensor(tr, device=DEV), int(os.environ.get("RL_NENV", 64)), seed, At, Lt)
    env.days_str = z["days"]
    eval_env = DailyEnv(Ot, Ct, Et, Ft, torch.tensor(te, device=DEV), 1, seed, At, Lt)
    eval_env.days_str = z["days"]

    def cash_fn(env, per, g, m):
        return torch.full((env.E,), env.N, dtype=torch.long, device=DEV)

    def mom_fn(env, per, g, m):
        sc = per[..., 2].masked_fill(~m[:, :-1], -1e9)
        best = sc.argmax(-1)
        keep = (env.pos < env.N) & (env.held < 21)
        return torch.where(keep, env.pos, best)

    def rand_fn(env, per, g, m):
        keep = (env.pos < env.N) & (env.held < 5)
        idx = m[:, :-1].float().multinomial(1).squeeze(1)
        return torch.where(keep, env.pos, idx)

    syms_list = [str(s) for s in z["syms"]]
    base_list = [("cash", cash_fn), ("momentum top-63d, hold 21d", mom_fn), ("random stock, hold 5d", rand_fn)]
    if "SPY" in syms_list:
        spy = syms_list.index("SPY")

        def spy_fn(env, per, g, m):
            return torch.full((env.E,), spy, dtype=torch.long, device=DEV)
        base_list.append(("SPY buy-and-hold", spy_fn))
    out = {}
    for nm, fn in base_list:
        for split, ds in (("val", va), ("test", te)):
            if nm.startswith("random"):
                tots = []
                for rs in range(5):
                    torch.manual_seed(1000 + rs)
                    rows = run_policy(eval_env, ds, fn)
                    tots.append(summarize(f"{nm} seed{rs} [{split}]", rows)["scenario_index_total"])
                torch.manual_seed(seed)
                print(f"== random 5-seed [{split}]: mean total {np.mean(tots) * 100:+.2f}% sd {np.std(tots) * 100:.2f}% min {np.min(tots) * 100:+.2f}% max {np.max(tots) * 100:+.2f}%", flush=True)
                out[f"{nm}|{split}"] = dict(mean_total=float(np.mean(tots)), sd_total=float(np.std(tots)), totals=[float(t) for t in tots])
                continue
            rows = run_policy(eval_env, ds, fn)
            out[f"{nm}|{split}"] = summarize(f"{nm} [{split}]", rows)
    per, g, m = env.reset()
    pol = Policy(per.shape[-1], g.shape[-1]).to(DEV)
    opt = torch.optim.Adam(pol.parameters(), lr=3e-4)
    best = -float("inf")
    lam, clip = 0.95, 0.2
    log = open(os.path.join(HERE, "runs", f"{tag}.log"), "a", encoding="utf-8")
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
        Vv = torch.stack(B["v"]); Rw = torch.stack(B["r"])
        T = len(Rw)
        adv = torch.zeros_like(Rw)
        gg = torch.zeros(env.E, device=DEV)
        for k in reversed(range(T)):
            nv = Vv[k + 1] if k + 1 < T else torch.zeros_like(gg)
            gg = Rw[k] + nv - Vv[k] + lam * gg
            adv[k] = gg
        ret = adv + Vv
        fl = lambda x: torch.stack(x).flatten(0, 1)
        P_, G_, M_, A_, LP_ = fl(B["per"]), fl(B["g"]), fl(B["m"]), fl(B["a"]), fl(B["lp"])
        ADV, RET = adv.flatten(), ret.flatten()
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        n = len(ADV)
        for _ in range(3):
            perm = torch.randperm(n, device=DEV)
            for s in range(0, n, 512):
                i = perm[s:s + 512]
                lo, v = pol(P_[i], G_[i], M_[i])
                dist = torch.distributions.Categorical(logits=lo)
                ratio = torch.exp(dist.log_prob(A_[i]) - LP_[i])
                pl = -torch.min(ratio * ADV[i], ratio.clamp(1 - clip, 1 + clip) * ADV[i]).mean()
                loss = pl + 0.5 * F.mse_loss(v, RET[i]) - 0.003 * dist.entropy().mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(pol.parameters(), 0.5)
                opt.step()
        msg = f"it {it} ep_ret {float(Rw.sum(0).mean()):+.2f}% trades {float(env.trades.mean()):.1f} {time.time() - t1:.1f}s"
        if it % 10 == 0 or it == iters - 1:
            def greedy(env_, per_, g_, m_):
                with torch.no_grad():
                    lo_, _ = pol(per_, g_, m_)
                return lo_.argmax(-1)
            rows = run_policy(eval_env, va, greedy)
            r = np.array([x["ret"] for x in rows])
            sh = float(r.mean() / (r.std() + 1e-9) * np.sqrt(252))
            msg += f" | val total {float(np.exp(r.sum()) - 1) * 100:+.2f}% sharpe {sh:+.2f} trades {rows[-1]['trades']}"
            if sh > best:
                best = sh
                torch.save(pol.state_dict(), os.path.join(HERE, "runs", f"{tag}_best.pt"))
                msg += " *"
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()
    torch.save(pol.state_dict(), os.path.join(HERE, "runs", f"{tag}_last.pt"))
    for k in ("best", "last"):
        pol.load_state_dict(torch.load(os.path.join(HERE, "runs", f"{tag}_{k}.pt"), map_location=DEV))
        def greedy(env_, per_, g_, m_):
            with torch.no_grad():
                lo_, _ = pol(per_, g_, m_)
            return lo_.argmax(-1)
        for split, ds in (("val", va), ("test", te)):
            rows = run_policy(eval_env, ds, greedy)
            out[f"rl_{k}|{split}"] = summarize(f"RL {k} [{split}]", rows)
            json.dump(rows, open(os.path.join(HERE, "runs", f"{tag}_{k}_{split}_days.json"), "w"))
    json.dump(out, open(os.path.join(HERE, "runs", f"{tag}_result.json"), "w"), indent=1)
    print(f"done {time.time() - t0:.0f}s", flush=True)
