import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sim
from sim import Data

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
STOP = sim.STOP
RANK = bool(int(os.environ.get("READER_RANK", "1")))
HEADS = ("gain", "tp" if sim.TP > 0 else "stop", "gain_cs")


def next_live_open(X):
    O, D, B, N = X.O, X.D, X.B, X.N
    nxt = torch.empty((D, B + 1, N), dtype=O.dtype, device=X.dev)
    nxt[:, B] = X.oc
    for t in reversed(range(B)):
        live = X.has[:, t] & (O[:, t] > 0)
        nxt[:, t] = torch.where(live, O[:, t], nxt[:, t + 1])
    xb = X.exit_bar
    if xb < B:
        live = X.has[:, xb] & (O[:, xb] > 0)
        nxt[:, xb:] = torch.where(live, O[:, xb], X.oc)[:, None, :]
    return nxt


def horizon(X, h):
    hor = torch.zeros((X.D, X.B, X.N), dtype=torch.float32, device=X.dev)
    for b in range(X.B):
        e = b + 1
        if e > X.entry_last:
            break
        hor[:, b] = (min(e + h, X.exit_bar) - e) / h
    return hor


def labels(X, h):
    O, C, H = X.O, X.C, X.H
    D, B, N = C.shape
    nxt = next_live_open(X)
    fb, fs = X.eff_rates(X.deposit)
    cost = float(np.log(1 - fs) - np.log(1 + fb))
    ret = torch.full((D, B, N), float("nan"), dtype=torch.float64, device=X.dev)
    flag = torch.zeros((D, B, N), dtype=torch.bool, device=X.dev)
    hor = torch.zeros((D, B, N), dtype=torch.float32, device=X.dev)
    for b in range(B):
        e = b + 1
        if e > X.entry_last:
            break
        x = min(e + h, X.exit_bar)
        o = O[:, e]
        ok = X.has[:, e] & X.buy_ok(o, X.pc, X.deposit, 1.29)
        bp = X.fill_buy(o)
        sp_x = X.fill_sell(nxt[:, x])
        Cw, Hw, hw = C[:, e:x], H[:, e:x], X.has[:, e:x]
        stop_hit = (Cw > 0) & (Cw <= (bp * (1 - X.stop))[:, None, :])
        if X.tp > 0:
            tpp = X.tp_price(bp)
            if X.us:
                tp_hit = hw & (Cw >= tpp[:, None, :])
            else:
                tp_hit = hw & (Hw >= (tpp + X.tick(tpp))[:, None, :]) & ~stop_hit
        else:
            tp_hit = torch.zeros_like(stop_hit)
        ev = stop_hit | tp_hit
        any_ = ev.any(1)
        first = ev.float().argmax(1)
        is_tp = tp_hit.gather(1, first[:, None, :]).squeeze(1) & any_
        tsell = (e + first + 1).clamp_max(x)
        sp_m = X.fill_sell(nxt.gather(1, tsell[:, None, :]).squeeze(1))
        if X.tp > 0 and not X.us:
            sp_m = torch.where(is_tp, tpp, sp_m)
        r = torch.where(any_, torch.log(sp_m / bp), torch.log(sp_x / bp)) + cost
        ret[:, b] = torch.where(ok, r, torch.full_like(r, float("nan")))
        flag[:, b] = (is_tp if X.tp > 0 else any_) & ok
        hor[:, b] = (x - e) / h
    return ret, flag, hor


def cross(ret):
    med = torch.nanmedian(ret, dim=2, keepdim=True).values
    return ret - torch.nan_to_num(med)


def context(F_, mask):
    m = mask.float()[..., None]
    return (F_ * m).sum(-2) / m.sum(-2).clamp_min(1)


class Reader(nn.Module):
    def __init__(self, f, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(f, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, len(HEADS)))

    def forward(self, x):
        return self.net(x)


def feats(X, days, hor):
    F_ = X.F[days]
    tradable = X.has[days] & X.elig_day[days][:, None, :]
    ctx = context(F_, tradable)[:, :, None].expand_as(F_)
    tb = (torch.arange(X.B, device=X.dev).float() / X.B)[None, :, None, None].expand(*F_.shape[:3], 1)
    parts = [F_]
    if RANK:
        big = torch.where(tradable[..., None], F_, torch.full_like(F_, float("inf")))
        rk = big.argsort(2).argsort(2).float()
        cnt = tradable.sum(2, keepdim=True)[..., None].float().clamp_min(1)
        parts.append(torch.where(tradable[..., None], rk / cnt - 0.5, torch.zeros_like(rk)))
    return torch.cat(parts + [ctx, tb, hor[days][..., None]], -1)


def build(X, days, ret, stop, hor):
    feat = feats(X, days, hor)
    r = ret[days]
    valid = ~torch.isnan(r)
    y = torch.stack([r > 0, stop[days], cross(r) > 0], -1).float()
    return feat, r, y, valid


def auc(score, y):
    o = torch.argsort(score)
    rank = torch.empty_like(o, dtype=torch.float64)
    rank[o] = torch.arange(1, len(score) + 1, device=score.device, dtype=torch.float64)
    p = y.sum()
    n = len(y) - p
    return float((rank[y].sum() - p * (p + 1) / 2) / (p * n).clamp_min(1))


def fit(X, days, ret, stop, hor, epochs=8, seed=0, log=print):
    torch.manual_seed(seed)
    f, r, y, v = build(X, days, ret, stop, hor)
    x, y = f[v], y[v]
    log(f"fit {X.iv} days {len(days)} samples {len(x)} base " + " ".join(f"{k} {float(y[:, i].mean()):.3f}" for i, k in enumerate(HEADS)))
    m = Reader(x.shape[-1]).to(X.dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(x)
    for ep in range(epochs):
        perm = torch.randperm(n, device=X.dev)
        for s in range(0, n, 8192):
            i = perm[s:s + 8192]
            loss = F.binary_cross_entropy_with_logits(m(x[i]), y[i])
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    return m


def fit_or_load(X, days, ret, stop, hor, path, **kw):
    f = build(X, days[:1], ret, stop, hor)[0].shape[-1]
    if os.path.exists(path):
        m = Reader(f).to(X.dev)
        m.load_state_dict(torch.load(path, map_location=X.dev))
        return m.eval()
    m = fit(X, days, ret, stop, hor, **kw)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(m.state_dict(), path)
    return m


def predict(m, X, days, ret, stop, hor):
    f = build(X, days, ret, stop, hor)[0]
    with torch.no_grad():
        return torch.sigmoid(torch.stack([m(f[d]) for d in range(len(days))]))


def run(iv, h, back=0):
    X = Data(DEV, iv)
    ret, stop, hor = labels(X, h)
    days = list(range(1, X.D))
    n_test = len(days) // 4
    end = len(days) - back
    tr, te = days[:end - n_test], days[end - n_test:end]
    m = fit(X, tr, ret, stop, hor)
    f, r, y, v = build(X, te, ret, stop, hor)
    p = torch.sigmoid(torch.stack([m(f[d]) for d in range(len(te))]).detach())
    res = dict(iv=iv, h=h, stop=STOP, train_days=len(tr), test_days=len(te), n=int(v.sum()), mean_ret=float(r[v].mean()))
    for i, k in enumerate(HEADS):
        res[f"auc_{k}"] = auc(p[..., i][v], y[..., i][v] > 0.5)
        res[f"base_{k}"] = float(y[..., i][v].mean())
    print(json.dumps(res, indent=1), flush=True)
    os.makedirs(OUT, exist_ok=True)
    tag = f"reader_{iv}_h{h}_s{STOP}" + (f"_back{back}" if back else "")
    torch.save(m.state_dict(), os.path.join(OUT, tag + ".pt"))
    json.dump(res, open(os.path.join(OUT, tag + ".json"), "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 3:
        STOP = float(sys.argv[3])
    run(sys.argv[1], int(sys.argv[2]), int(os.environ.get("READER_BACK", "0")))
