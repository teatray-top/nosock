import datetime as dt
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.environ.get("SCALP_PANEL", os.path.join(HERE, "data", "panelusm1m_w1_v4.npz"))
OUT = os.environ.get("SCALP_OUT", os.path.join(HERE, "runs", "scalp_1m_w1_2026-11-06.json"))
FEE, SEC, MIN_FEE, TICK = 0.001, 2.06e-5, 0.01, 0.01
SIZES = (7.0, 50.0)
SEEDS = 20
NBOOT = 10000
MIN_PX = 5.0
ENTRY_FIRST, ENTRY_LAST, FORCED_EXIT = 10, 300, 325
CONFIGS = {"tp0.6_sl0.4_h15": (0.006, 0.004, 15), "tp1.0_sl0.6_h30": (0.010, 0.006, 30)}
DAYS_LIMIT = int(os.environ.get("SCALP_DAYS", 0))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()[:16]


def seed_of(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


z = np.load(PANEL, allow_pickle=True)
DAYS = [str(d) for d in z["days"]]
SYMS = [str(s) for s in z["syms"]]
O, H, C, V, HAS = z["O"], z["H"], z["C"], z["V"], z["has"]
D, B, N = C.shape
if DAYS_LIMIT:
    O, H, C, V, HAS, DAYS = O[-DAYS_LIMIT:], H[-DAYS_LIMIT:], C[-DAYS_LIMIT:], V[-DAYS_LIMIT:], HAS[-DAYS_LIMIT:], DAYS[-DAYS_LIMIT:]
    D = len(DAYS)
print(f"panel {os.path.basename(PANEL)} {D} days x {B} bars x {N} syms ({DAYS[0]}..{DAYS[-1]})", flush=True)


def signals(d):
    c, v, h = C[d], V[d], H[d]
    with np.errstate(invalid="ignore", divide="ignore"):
        r1 = c[1:] / c[:-1] - 1
        r1 = np.vstack([np.full((1, N), np.nan), r1])
        r5 = np.full((B, N), np.nan)
        r5[5:] = c[5:] / c[:-5] - 1
        vm = np.full((B, N), np.nan)
        cs = np.nancumsum(np.nan_to_num(v), axis=0)
        vm[20:] = (cs[20:] - cs[:-20]) / 20.0
        hmax = np.maximum.accumulate(np.nan_to_num(h, nan=-np.inf), axis=0)
        prev_hmax = np.vstack([np.full((1, N), -np.inf), hmax[:-1]])
        base = HAS[d] & np.isfinite(c) & (c >= MIN_PX)
        s1 = base & (r1 >= 0.005) & (v >= 3 * vm) & (r5 > 0)
        s2 = base & (r5 <= -0.015)
        s3 = base & (c > prev_hmax) & (v >= 1.5 * vm)
        s3[:30] = False
    out = {}
    for name, s, strength in (("S1_ignition", s1, r1), ("S2_reversal", s2, -r5), ("S3_breakout", s3, v / np.where(vm > 0, vm, np.nan))):
        s = s.copy()
        s[:ENTRY_FIRST] = False
        s[ENTRY_LAST + 1:] = False
        t, j = np.where(s)
        out[name] = (t, j, strength[t, j])
    return out


def settle_paths(d, t, j, A, tp, sl, hmax):
    n = len(t)
    if n == 0:
        return np.zeros(0), np.zeros(0, dtype=int), np.zeros(0, dtype=bool), np.zeros(0, dtype=int)
    c = C[d]
    e = t + 1
    po = O[d][e, j] + TICK
    filled = np.isfinite(po) & (po > 0)
    k = np.arange(1, hmax + 1)
    idx = np.minimum(e[:, None] + k[None, :], B - 1)
    path = c[idx, j[:, None]]
    up = path >= po[:, None] * (1 + tp)
    dn = path <= po[:, None] * (1 - sl)
    hit = up | dn
    first = np.where(hit.any(1), hit.argmax(1), hmax - 1)
    exit_bar = e + 1 + first
    forced = exit_bar > FORCED_EXIT
    exit_bar = np.minimum(exit_bar, FORCED_EXIT)
    px = c[exit_bar, j]
    bad = ~np.isfinite(px)
    for i in np.where(bad & filled)[0]:
        w = np.where(np.isfinite(c[e[i]:exit_bar[i] + 1, j[i]]))[0]
        px[i] = c[e[i] + w[-1], j[i]] if len(w) else po[i]
    px = px - TICK
    sh = np.where(filled, A / po, 0.0)
    comm_b = np.floor(A * FEE * 100) / 100
    amt = sh * np.maximum(np.nan_to_num(px, nan=0.0), 0.0)
    comm_s = np.floor(amt * FEE * 100) / 100
    sec = np.where(amt > 0, np.maximum(amt * SEC, MIN_FEE), 0.0)
    net = np.where(filled, amt - comm_s - sec - A - comm_b, 0.0)
    reason = np.where(~filled, 0, np.where(forced, 4, np.where(hit.any(1), np.where(up[np.arange(n), first], 1, 2), 3)))
    return net / A, exit_bar, filled, reason


def boot_days(vals, days):
    rng = np.random.default_rng(0)
    vals, days = np.asarray(vals, dtype=float), np.asarray(days)
    ud = np.unique(days)
    parts = [vals[days == u] for u in ud]
    bs = np.array([np.concatenate([parts[i] for i in rng.integers(len(ud), size=len(ud))]).mean() for _ in range(NBOOT)])
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], float((bs <= 0).mean())


res = dict(meta=dict(panel=os.path.relpath(PANEL, HERE), panel_sha256=sha(PANEL) if os.environ.get("SCALP_HASH", "1") == "1" else None, script_sha256=sha(__file__), days=[DAYS[0], DAYS[-1]], n_days=D, bars=B, syms=N, fee=FEE, sec=SEC, min_fee=MIN_FEE, tick=TICK, sizes=list(SIZES), seeds=SEEDS, min_px=MIN_PX, entry_first=ENTRY_FIRST, entry_last=ENTRY_LAST, forced_exit=FORCED_EXIT, configs=CONFIGS, started=dt.datetime.now().isoformat(timespec="seconds")))
print("META " + json.dumps(res["meta"]), flush=True)
sig_all = {}
for d in range(D):
    for name, (t, j, st) in signals(d).items():
        sig_all.setdefault(name, []).append((d, t, j, st))
    if (d + 1) % 100 == 0:
        print(f"signals {d + 1}/{D}", flush=True)
for name, lst in sig_all.items():
    print(f"{name}: {sum(len(x[1]) for x in lst)} signals over {D} days", flush=True)

for name, lst in sig_all.items():
    res[name] = {}
    acc = {(cname, A): dict(rets=[], rand=[], days=[], reasons=[], seq={}) for cname in CONFIGS for A in SIZES}
    for d, t, j, st in lst:
        if len(t) == 0:
            continue
        elig = HAS[d] & np.isfinite(C[d]) & (C[d] >= MIN_PX)
        cols = np.argsort(~elig, axis=1, kind="stable")
        cnt = elig.sum(1)
        jr_seeds = []
        for sd in range(SEEDS):
            g = np.random.default_rng(seed_of(name, d, sd))
            u = g.random(len(t))
            jr_seeds.append(cols[t, np.minimum((u * cnt[t]).astype(int), np.maximum(cnt[t] - 1, 0))])
        for cname, (tp, sl, hmax) in CONFIGS.items():
            for A in SIZES:
                r, xb, filled, reason = settle_paths(d, t, j, A, tp, sl, hmax)
                rr = np.mean([settle_paths(d, t, jr, A, tp, sl, hmax)[0] for jr in jr_seeds], axis=0)
                a = acc[(cname, A)]
                a["rets"].append(r)
                a["rand"].append(rr)
                a["days"].append(np.full(len(t), d))
                a["reasons"].append(reason)
                order = np.lexsort((-st, t))
                pos_free, pnl, ntr = 0, 0.0, 0
                for i in order:
                    if t[i] + 1 >= pos_free and filled[i]:
                        pnl += r[i] * A
                        ntr += 1
                        pos_free = xb[i] + 1
                a["seq"][d] = (pnl, ntr)
                a.setdefault("daily", {})[d] = dict(day=DAYS[d], n=int(len(t)), filled=int(filled.sum()), sum_ret=float(r.sum()), sum_rand=float(rr.sum()), seq_pnl=float(pnl), seq_trades=int(ntr))
    for (cname, A), a in acc.items():
        rets, rand_rets, days_of, reasons = (np.concatenate(a[k]) for k in ("rets", "rand", "days", "reasons"))
        seq_day = a["seq"]
        diff = rets - rand_rets
        ci, p = boot_days(diff, days_of)
        sd_pnl = np.array([seq_day[d][0] for d in sorted(seq_day)])
        sd_n = np.array([seq_day[d][1] for d in sorted(seq_day)])
        sci, sp = boot_days(sd_pnl, np.arange(len(sd_pnl)))
        yrs = {}
        for y in sorted(set(DAYS[d][:4] for d in seq_day)):
            ds = [d for d in seq_day if DAYS[d][:4] == y]
            yrs[y] = dict(days=len(ds), seq_total=float(sum(seq_day[d][0] for d in ds)), trades=int(sum(seq_day[d][1] for d in ds)))
        s = dict(n_signals=int(len(rets)), mean_ret=float(rets.mean()), median_ret=float(np.median(rets)), win_rate=float((rets > 0).mean()), rand_mean_ret=float(rand_rets.mean()), diff_mean=float(diff.mean()), diff_ci95=ci, diff_p_le0=p, exit_mix=dict(unfilled=int((reasons == 0).sum()), tp=int((reasons == 1).sum()), sl=int((reasons == 2).sum()), time=int((reasons == 3).sum()), forced=int((reasons == 4).sum())), seq_days=int(len(sd_pnl)), seq_total=float(sd_pnl.sum()), seq_per_day=float(sd_pnl.mean()), seq_ci95=sci, seq_p_le0=sp, seq_trades_per_day=float(sd_n.mean()), seq_ex_top3=float(sd_pnl.sum() - np.sort(sd_pnl)[-3:].sum()), by_year=yrs)
        s["daily"] = [a["daily"][d] for d in sorted(a["daily"])]
        res[name][f"{cname}_${A:.0f}"] = s
        print(f"== {name} {cname} ${A:.0f}: signals {s['n_signals']} mean {s['mean_ret'] * 100:+.3f}% median {s['median_ret'] * 100:+.3f}% win {s['win_rate']:.3f} rand {s['rand_mean_ret'] * 100:+.3f}% diff {s['diff_mean'] * 100:+.3f}% CI[{ci[0] * 100:+.3f}%, {ci[1] * 100:+.3f}%] P {p:.3f} exits {s['exit_mix']} | sequential 1-pos: {s['seq_days']} days total ${s['seq_total']:+.2f} ({s['seq_per_day']:+.4f}/day, CI [{sci[0]:+.4f}, {sci[1]:+.4f}], P {sp:.3f}) trades/day {s['seq_trades_per_day']:.1f} ex-top3 ${s['seq_ex_top3']:+.2f}", flush=True)
        for y, v in yrs.items():
            print(f"   {y}: days {v['days']} seq ${v['seq_total']:+.2f} trades {v['trades']}", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print(f"saved {OUT}", flush=True)
