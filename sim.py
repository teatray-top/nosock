import os

import numpy as np
import torch

import panel

FEE = float(os.environ.get("DOBBY_FEE", 0.00015))
TAX = float(os.environ.get("DOBBY_TAX", 0.0020))
US_FEE = float(os.environ.get("DOBBY_US_FEE", 0.001))
US_SEC = float(os.environ.get("DOBBY_US_SEC", 2.06e-5))
US_MIN_FEE = float(os.environ.get("DOBBY_US_MIN_FEE", 0.01))
DEPOSIT = float(os.environ.get("DOBBY_DEPOSIT", 10000))
DEPOSIT_US = float(os.environ.get("DOBBY_DEPOSIT_US", 7.0))
VAULT_SHARE = float(os.environ.get("DOBBY_VAULT_SHARE", 0.5))
MAX_REFILL = float(os.environ.get("DOBBY_MAX_REFILL", 20))
MIN_PRICE = float(os.environ.get("DOBBY_MIN_PRICE", 1000))
MIN_SHARES = 5
MIN_TURN = 3e8
VOL_CAP = 0.2
SLIP = float(os.environ.get("DOBBY_SLIP", 1.0))
STOP = float(os.environ.get("READER_STOP", 0.06))
TP = float(os.environ.get("SCALP_TP", 0))
US_EXIT = int(os.environ.get("DOBBY_US_EXIT", 66))
SESSION = {("kr", "5m"): (66, 71), ("kr", "60m"): (5, 6), ("us", "5m"): (US_EXIT - 6, US_EXIT), ("us", "60m"): (4, 5)}
_KR_B = [2000.0, 5000.0, 20000.0, 50000.0, 200000.0, 500000.0]
_KR_T = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]


def tick_us(p):
    return torch.where(p < 1, 0.0001, 0.01).to(p.dtype)


def tick(p):
    b = torch.tensor(_KR_B, dtype=p.dtype, device=p.device)
    t = torch.tensor(_KR_T, dtype=p.dtype, device=p.device)
    return t[torch.bucketize(p, b, right=True)]


class Data:
    def __init__(self, dev, iv="5m", arrays=None):
        p = arrays if arrays is not None else panel.load(iv)
        self.iv = iv
        sub, base = panel.split(iv)
        self.us = sub.startswith("us")
        self.tick = tick_us if self.us else tick
        self.fee_buy = US_FEE if self.us else FEE
        self.fee_sell = US_FEE if self.us else FEE + TAX
        self.sec = US_SEC if self.us else 0.0
        self.min_fee = US_MIN_FEE if self.us else 0.0
        self.deposit = DEPOSIT_US if self.us else DEPOSIT
        self.min_turn = 1e6 if self.us else MIN_TURN
        self.stop, self.tp = STOP, TP
        self.entry_last, self.exit_bar = SESSION[("us" if self.us else "kr", base)]
        self.lags = (1, 3, 6, 12) if base == "5m" else (1, 2, 3, 5)
        self.dev = dev
        t = lambda a: torch.tensor(np.nan_to_num(a, nan=0.0), dtype=torch.float64, device=dev)
        self.O, self.H, self.L, self.C, self.V = t(p["O"]), t(p["H"]), t(p["L"]), t(p["C"]), t(p["V"])
        self.has = torch.tensor(p["has"], device=dev)
        self.pc, self.oc, self.turn = t(p["pc"]), t(p["oc"]), t(p["turn"])
        self.days, self.syms = p["days"], p["syms"]
        self.D, self.B, self.N = self.C.shape
        pt = torch.zeros_like(self.turn)
        pt[1:] = self.turn[:-1]
        self.prev_turn = pt
        self.elig_day = (self.pc > 0) & (pt >= self.min_turn)
        self.F = self.features()
        self.buyable = self.has & self.elig_day[:, None, :] & self.buy_ok(self.C, self.pc[:, None, :], self.deposit, 1.28)

    def buy_ok(self, price, pc, cap, lim=1.29):
        ok = price > 0
        cap = torch.as_tensor(cap, dtype=price.dtype, device=price.device)
        if self.us:
            return ok & (cap >= 0.5)
        return ok & (price < pc * lim) & (price >= MIN_PRICE) & (price * MIN_SHARES <= cap)

    def fill_buy(self, o):
        return o + SLIP * self.tick(o)

    def fill_sell(self, o):
        return (o - SLIP * self.tick(o)).clamp_min(1e-4)

    def tp_price(self, bp):
        x = bp * (1 + self.tp)
        t = self.tick(x)
        return torch.ceil(x / t - 1e-9) * t

    def cost_buy(self, amt):
        c = amt * self.fee_buy
        return torch.floor(c * 100 + 1e-9) / 100 if self.us else c

    def cost_sell(self, amt):
        c = amt * self.fee_sell
        if self.us:
            c = torch.floor(c * 100 + 1e-9) / 100 + torch.maximum(amt * self.sec, torch.full_like(amt, self.min_fee))
        return c

    def eff_rates(self, notional):
        if not self.us:
            return self.fee_buy, self.fee_sell
        r = lambda x: int(x * 100 + 1e-9) / 100 / notional
        return r(notional * self.fee_buy), r(notional * self.fee_sell) + max(self.sec, self.min_fee / notional)

    def features(self):
        C, O, V, H, L = self.C, self.O, self.V, self.H, self.L
        pc = self.pc[:, None, :].expand_as(C)
        Cs = torch.where(C > 0, C, pc).clamp_min(1e-4)
        o0 = torch.where(O[:, :1] > 0, O[:, :1], pc[:, :1]).clamp_min(1e-4)
        f = []
        for k in self.lags:
            prev = torch.cat([pc[:, :k], Cs[:, :-k]], 1).clamp_min(1e-4)
            f.append(torch.log(Cs / prev) * 20)
        f.append(torch.log(Cs / o0) * 10)
        f.append(torch.log(o0 / pc.clamp_min(1e-4)).expand_as(C) * 10)
        cv = torch.cumsum(V, 1)
        cnt = torch.arange(1, self.B + 1, device=self.dev, dtype=C.dtype)[None, :, None]
        f.append(torch.log1p(V / (cv / cnt).clamp_min(1)))
        f.append((torch.log10(torch.cumsum(Cs * V, 1) + 1) - 8) / 2)
        hh = torch.cummax(torch.where(H > 0, H, Cs), 1).values
        ll = torch.cummin(torch.where(L > 0, L, Cs), 1).values
        f.append(torch.log(Cs / hh) * 10)
        f.append(torch.log(Cs / ll) * 10)
        f.append(self.tick(Cs) / Cs * 100)
        f.append(((torch.log10(self.prev_turn + 1) - 9))[:, None, :].expand_as(C))
        F = torch.stack(f, -1).float().clamp(-5, 5)
        return torch.nan_to_num(F)


class Env:
    def __init__(self, data, n_env, days, seed=0):
        self.X = data
        self.E = n_env
        self.days = torch.tensor(days, device=data.dev)
        self.g = torch.Generator(device="cpu").manual_seed(seed)
        d = data.dev
        z = lambda dt=torch.float64: torch.zeros(self.E, dtype=dt, device=d)
        self.d, self.b, self.pos = z(torch.long), z(torch.long), z(torch.long)
        self.sh, self.cash, self.cap, self.entry = z(), z(), z(), z()
        self.trades, self.fills, self.tpp = z(), z(), z()

    def reset(self, day=None, cap=None):
        X = self.X
        if day is None:
            day = self.days[torch.randint(len(self.days), (self.E,), generator=self.g).to(X.dev)]
        if cap is None:
            cap = X.deposit * (1 + torch.rand(self.E, generator=self.g, dtype=torch.float64).to(X.dev) ** 2 * 2)
        self.d[:] = day
        self.b[:] = 0
        self.pos[:] = X.N
        self.sh[:] = 0
        self.cash[:] = cap
        self.cap[:] = cap
        self.entry[:] = 0
        self.trades[:] = 0
        self.fills[:] = 0
        self.tpp[:] = 0
        return self.obs()

    def mark(self, b):
        X = self.X
        j = self.pos.clamp_max(X.N - 1)
        bb = b.clamp_max(X.B - 1)
        c = torch.where(b >= X.B, X.oc[self.d, j], X.C[self.d, bb, j])
        amt = self.sh * X.fill_sell(c)
        v = amt - X.cost_sell(amt)
        return self.cash + torch.where(self.pos < X.N, v, torch.zeros_like(v))

    def mask(self):
        X = self.X
        ok = X.has[self.d, self.b] & X.elig_day[self.d] & (self.b[:, None] + 1 <= X.entry_last)
        ok = ok & X.buy_ok(X.C[self.d, self.b], X.pc[self.d], self.cap[:, None], 1.28)
        m = torch.cat([ok, torch.ones(self.E, 1, dtype=torch.bool, device=X.dev)], 1)
        m[torch.arange(self.E, device=X.dev), self.pos] = True
        last = self.b + 1 >= X.exit_bar
        m[last] = False
        m[last, X.N] = True
        return m

    def obs(self):
        X = self.X
        F = X.F[self.d, self.b]
        held = torch.zeros(self.E, X.N, device=X.dev)
        e = torch.arange(self.E, device=X.dev)
        inpos = self.pos < X.N
        held[e[inpos], self.pos[inpos]] = 1
        j = self.pos.clamp_max(X.N - 1)
        upnl = torch.where(inpos, torch.log(X.C[self.d, self.b, j].clamp_min(1e-4) / self.entry.clamp_min(1e-4)), torch.zeros_like(self.entry)).float()
        per = torch.cat([F, held[..., None], (held * upnl[:, None] * 20)[..., None]], -1)
        g = torch.stack([self.b.float() / X.B, torch.log(self.cap / X.deposit).float(), inpos.float(), (torch.log(self.mark(self.b) / self.cap) * 20).float(), self.trades.float() / 5], -1)
        return per, g, self.mask()

    def step(self, k):
        X = self.X
        v0 = self.mark(self.b)
        nb = self.b + 1
        at_exit = nb >= X.exit_bar
        k = torch.where(at_exit, torch.full_like(k, X.N), k)
        switch = k != self.pos
        inpos = self.pos < X.N
        j = self.pos.clamp_max(X.N - 1)
        nbc = nb.clamp_max(X.B - 1)
        o = X.O[self.d, nbc, j]
        live = (nb < X.B) & X.has[self.d, nbc, j] & (o > 0)
        o = torch.where(live, o, X.oc[self.d, j])
        can_sell = switch & inpos & (live | at_exit) & (o > 0)
        if not X.us:
            can_sell = can_sell & ~((o <= X.pc[self.d, j] * 0.705) & ~at_exit)
        amt = self.sh * X.fill_sell(o)
        proceeds = amt - X.cost_sell(amt)
        self.cash = torch.where(can_sell, self.cash + proceeds, self.cash)
        self.sh = torch.where(can_sell, torch.zeros_like(self.sh), self.sh)
        self.pos = torch.where(can_sell, torch.full_like(self.pos, X.N), self.pos)
        self.tpp = torch.where(can_sell, torch.zeros_like(self.tpp), self.tpp)
        self.fills += can_sell.double()
        want = switch & (k < X.N) & (self.pos == X.N) & (nb < X.B)
        kk = k.clamp_max(X.N - 1)
        ob = X.O[self.d, nbc, kk]
        ok = want & X.has[self.d, nbc, kk] & X.buy_ok(ob, X.pc[self.d, kk], self.cash, 1.29)
        bp = X.fill_buy(ob)
        n = (self.cash - X.cost_buy(self.cash)) / bp.clamp_min(1e-4)
        vref = X.V[self.d, self.b, kk]
        vref = torch.where(vref > 0, vref, X.prev_turn[self.d, kk] / X.pc[self.d, kk].clamp_min(1e-4) / X.B)
        n = torch.minimum(n, vref * VOL_CAP)
        if not X.us:
            n = torch.floor(n)
        ok = ok & ((n * bp >= 0.5) if X.us else (n >= 1))
        amt = n * bp
        self.cash = torch.where(ok, self.cash - amt - X.cost_buy(amt), self.cash)
        self.sh = torch.where(ok, n, self.sh)
        self.pos = torch.where(ok, kk, self.pos)
        self.entry = torch.where(ok, bp, self.entry)
        if X.tp > 0:
            self.tpp = torch.where(ok, X.tp_price(bp), self.tpp)
        self.trades += ok.double()
        self.fills += ok.double()
        if X.tp > 0 and not X.us:
            inpos = self.pos < X.N
            j = self.pos.clamp_max(X.N - 1)
            hh, cc = X.H[self.d, nbc, j], X.C[self.d, nbc, j]
            fill = inpos & (nb < X.B) & X.has[self.d, nbc, j] & (hh >= self.tpp + X.tick(self.tpp)) & (cc > self.entry * (1 - X.stop))
            amt = self.sh * self.tpp
            proceeds = amt - X.cost_sell(amt)
            self.cash = torch.where(fill, self.cash + proceeds, self.cash)
            self.sh = torch.where(fill, torch.zeros_like(self.sh), self.sh)
            self.pos = torch.where(fill, torch.full_like(self.pos, X.N), self.pos)
            self.tpp = torch.where(fill, torch.zeros_like(self.tpp), self.tpp)
            self.fills += fill.double()
        self.b = nb
        v1 = self.mark(self.b)
        r = (v1 - v0) / self.cap
        done = self.b >= X.exit_bar
        return r.float(), done


def run_days(data, days, act_fn, seed=0, deposit=None, vault_share=None, flat=False, max_refill=None):
    deposit = data.deposit if deposit is None else deposit
    vault_share = VAULT_SHARE if vault_share is None else vault_share
    max_refill = MAX_REFILL * deposit if max_refill is None else max_refill
    out = []
    R, vault, refill = 0.0, 0.0, 0.0
    env = Env(data, 1, days, seed)
    halted = False
    for d in days:
        cap = deposit if flat else deposit + R
        per, g, m = env.reset(torch.tensor([d], device=data.dev), torch.tensor([cap], dtype=torch.float64, device=data.dev))
        if halted:
            out.append(dict(day=str(data.days[d]), cap=cap, pnl=0.0, trades=0, fills=0, R=R, vault=vault, refill=refill, halted=True))
            continue
        while True:
            k = act_fn(env, per, g, m)
            _, done = env.step(k)
            if bool(done[0]):
                break
            per, g, m = env.obs()
        assert int(env.pos[0]) == data.N
        pi = float(env.cash[0]) - cap
        if pi > 0:
            vault += vault_share * pi
            R += (1 - vault_share) * pi
        else:
            loss = -pi
            use = min(R, loss)
            R -= use
            refill += loss - use
        halted = refill > max_refill
        out.append(dict(day=str(data.days[d]), cap=cap, pnl=pi, trades=int(env.trades[0]), fills=int(env.fills[0]), R=R, vault=vault, refill=refill, halted=halted))
    return out


def summary(rows):
    p = np.array([r["pnl"] for r in rows])
    last = rows[-1]
    srt = np.sort(p)[::-1]
    return dict(days=len(rows), total_pnl=float(p.sum()), win_days=float((p > 0).mean()), zero_days=int((p == 0).sum()), mean_day=float(p.mean()), median_day=float(np.median(p)), worst_day=float(p.min()), best_day=float(p.max()), ex_top3=float(p.sum() - srt[:3].sum()), trades_per_day=float(np.mean([r["trades"] for r in rows])), vault=last["vault"], R=last["R"], refill=last["refill"], net=last["vault"] + last["R"] - last["refill"], halted=bool(last.get("halted", False)))
