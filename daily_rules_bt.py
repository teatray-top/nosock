import bisect
import datetime as dt
import glob
import hashlib
import json
import os
import sys
from zoneinfo import ZoneInfo

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(HERE, "data", "panel_daily_v2.npz")
ETF = os.path.join(HERE, "data", "panel_etf_v3.npz")
EDGAR = os.environ.get("EDGAR_DIR", os.path.join(HERE, "data", "edgar", "w1"))
OUT = os.environ.get("RULES_OUT", os.path.join(HERE, "runs", "daily_rules_v2_2026-11-05.json"))
ONLY = sys.argv[2] if len(sys.argv) > 2 else ""
FEE, SEC, MIN_FEE, TICK = 0.001, 2.06e-5, 0.01, 0.01
HAIRCUTS = (0.0, 0.10, 1.0)
HC = 0.10
SIZES = (7.0, 50.0)
SEEDS = 20
Q = 0.10
TOPK = 10
MIN_UNIVERSE = 50
NBOOT = 10000


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def sha_dir(d):
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        h.update(os.path.basename(p).encode())
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def seed_of(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


z = np.load(PANEL, allow_pickle=True)
O, C, ADJ, ELIG = z["O"], z["C"], z["adj"], z["elig"]
DAYS = [str(d) for d in z["days"]]
SYMS = [str(s) for s in z["syms"]]
SIDX = {s: i for i, s in enumerate(SYMS)}
D, N = C.shape
LISTED = np.isfinite(C) & (C > 0)
OPEN_OK = np.isfinite(O) & (O > 0)
YEAR = np.array([int(d[:4]) for d in DAYS])

RAND_SYMS = os.environ.get("RAND_SYMS")
RMASK = np.ones(N, dtype=bool)
EDGAR_TICKERS = set()
for _p in glob.glob(os.path.join(EDGAR, "*.jsonl")):
    with open(_p, encoding="utf-8") as _f:
        EDGAR_TICKERS.add(json.loads(_f.readline()).get("ticker"))
if RAND_SYMS == "edgar":
    RMASK = np.array([s in EDGAR_TICKERS for s in SYMS])
elif RAND_SYMS:
    _rs = set(str(s) for s in np.load(RAND_SYMS, allow_pickle=True)["syms"])
    RMASK = np.array([s in _rs for s in SYMS])

e = np.load(ETF, allow_pickle=True)
EDAYS = [str(d) for d in e["days"]]
SPY = e["O"][:, list(e["syms"]).index("SPY")]
SPYO = np.array([SPY[EDAYS.index(d)] if d in EDAYS else np.nan for d in DAYS])


def settle(j, db, ds, A, hc):
    j, db, ds = (np.asarray(x) for x in (j, db, ds))
    with np.errstate(invalid="ignore", divide="ignore"):
        filled = OPEN_OK[db, j]
        po = np.where(filled, O[db, j] + TICK * ADJ[db, j], np.nan)
        sh = np.where(filled, A / po, 0.0)
        comm_b = np.floor(A * FEE * 100) / 100
        ok = OPEN_OK[ds, j]
        px = np.where(ok, O[ds, j] - TICK * ADJ[ds, j], np.nan)
        forced = filled & ~ok
        for i in np.where(forced)[0]:
            w = np.where(LISTED[db[i] - 1:ds[i], j[i]])[0]
            px[i] = C[db[i] - 1 + w[-1], j[i]] * (1 - hc)
        amt = sh * np.maximum(np.nan_to_num(px, nan=0.0), 0.0)
        comm_s = np.floor(amt * FEE * 100) / 100
        sec = np.where(amt > 0, np.maximum(amt * SEC, MIN_FEE), 0.0)
        net = np.where(filled, amt - comm_s - sec - A - comm_b, 0.0)
    return net, forced, filled


def evaluate(pick, db, ds, A, idx, key):
    out = {}
    alloc = len(pick) * A
    for hc in HAIRCUTS:
        net, forced, filled = settle(pick, db, ds, A, hc)
        inv = filled.sum() * A
        out[f"net_h{hc}"] = float(net.sum())
        out[f"ret_h{hc}"] = float(net.sum() / alloc)
        if hc == HC:
            obs = filled & ~forced
            out.update(forced=int(forced.sum()), unfilled=int((~filled).sum()), filled=int(filled.sum()), invested=float(inv), allocated=float(alloc), fill_ret=float(net.sum() / inv) if inv > 0 else np.nan, obs_net=float(net[obs].sum()), obs_ret=float(net[obs].sum() / (obs.sum() * A)) if obs.any() else np.nan)
    rn = {hc: [] for hc in HAIRCUTS}
    rf, ro, rfr = [], [], []
    for sd in range(SEEDS):
        g = np.random.default_rng(seed_of(*key, sd))
        rp = g.choice(idx, len(pick), replace=False)
        for hc in HAIRCUTS:
            net, forced, filled = settle(rp, db, ds, A, hc)
            inv = filled.sum() * A
            rn[hc].append(net.sum() / alloc)
            if hc == HC:
                obs = filled & ~forced
                rf.append(int(forced.sum()))
                ro.append(net[obs].sum() / (obs.sum() * A) if obs.any() else np.nan)
                rfr.append(net.sum() / inv if inv > 0 else np.nan)
    for hc in HAIRCUTS:
        out[f"rand_ret_h{hc}"] = float(np.mean(rn[hc]))
    out["rand_forced"] = float(np.mean(rf))
    out["rand_obs_ret"] = float(np.nanmean(ro))
    out["rand_fill_ret"] = float(np.nanmean(rfr))
    return out


def boot(diff, groups=None):
    rng = np.random.default_rng(0)
    diff = np.asarray(diff, dtype=float)
    if len(diff) == 0:
        return [np.nan, np.nan], np.nan
    if groups is None:
        bs = np.array([rng.choice(diff, len(diff)).mean() for _ in range(NBOOT)])
    else:
        g = np.asarray(groups)
        ug = np.unique(g)
        parts = [diff[g == u] for u in ug]
        bs = np.array([np.concatenate([parts[i] for i in rng.integers(len(ug), size=len(ug))]).mean() for _ in range(NBOOT)])
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], float((bs <= 0).mean())


def paired(rows, a, b, group_key):
    sa = np.array([r[a] for r in rows], dtype=float)
    sb = np.array([r[b] for r in rows], dtype=float)
    m = np.isfinite(sa) & np.isfinite(sb)
    ci, p = boot(sa[m] - sb[m], [r[group_key] for r, k in zip(rows, m) if k] if group_key else None)
    return dict(n=int(m.sum()), ret_mean=float(sa[m].mean()), ret_median=float(np.median(sa[m])), rand_ret_mean=float(sb[m].mean()), diff_mean=float((sa[m] - sb[m]).mean()), ci95=ci, p_le0=p)


def agg(name, rows, group_key=None, capital=None):
    forced = int(sum(r["forced"] for r in rows))
    res = dict(n=len(rows), span=[rows[0]["day"], rows[-1]["day"]], forced=forced, unfilled=int(sum(r["unfilled"] for r in rows)), rand_forced_total=float(sum(r["rand_forced"] for r in rows)), invested_mean=float(np.mean([r["invested"] for r in rows])), capital_required=capital, spy_ret_mean=float(np.nanmean([r["spy_ret"] for r in rows])))
    for hc in HAIRCUTS:
        s = paired(rows, f"ret_h{hc}", f"rand_ret_h{hc}", group_key)
        s["total_net"] = float(sum(r[f"net_h{hc}"] for r in rows))
        res[f"scenario_h{hc}"] = s
    sc = res[f"scenario_h{HC}"]
    nets = np.array([r[f"net_h{HC}"] for r in rows])
    res["ex_top3_net"] = float(nets.sum() - np.sort(nets)[-3:].sum()) if len(nets) >= 3 else None
    res["total_net"] = sc["total_net"] if forced == 0 and res["rand_forced_total"] == 0 else None
    res["pnl_status"] = "no assumed settlement on either side" if res["total_net"] is not None else f"undetermined: strategy {forced} / random {res['rand_forced_total']:.1f} assumed settlements; scenario total range h0..h1 = [{res['scenario_h1.0']['total_net']:+.2f}, {res['scenario_h0.0']['total_net']:+.2f}]"
    ob = paired(rows, "obs_ret", "rand_obs_ret", group_key)
    ob["total_net"] = float(sum(r["obs_net"] for r in rows))
    ob["note"] = "forced positions excluded on both sides; bias direction depends on actual recovery value"
    res["observable"] = ob
    fc = paired(rows, "fill_ret", "rand_fill_ret", group_key)
    fc["note"] = "conditional on fill (unfilled rows dropped); primary stats above are on allocated capital with unfilled = 0 return"
    res["fill_conditional"] = fc
    res["returns_basis"] = "allocated capital per period (k x A); unfilled positions count as 0 return"
    res["scenario_return_on_max_principal"] = float(sc["total_net"] / capital) if capital else None
    res["principal_note"] = f"max principal deployed at once (${capital:.0f}); not an initial-capital requirement and not an account return (fee drag, cumulative losses and settlement delay not modeled)" + ("; scenario value, assumed settlements present" if res["total_net"] is None else "")
    yrs = {}
    for y in sorted(set(r["year"] for r in rows)):
        rs = [r for r in rows if r["year"] == y]
        yrs[str(y)] = dict(n=len(rs), net=float(sum(r[f"net_h{HC}"] for r in rs)), ret_mean=float(np.nanmean([r[f"ret_h{HC}"] for r in rs])), rand_ret_mean=float(np.nanmean([r[f"rand_ret_h{HC}"] for r in rs])), forced=int(sum(r["forced"] for r in rs)), unfilled=int(sum(r["unfilled"] for r in rs)))
    res["by_year"] = yrs
    print(f"== {name}: n {res['n']} {res['span'][0]}..{res['span'][1]} | h0.1 total ${sc['total_net']:+.2f} (range h1..h0 [{res['scenario_h1.0']['total_net']:+.2f}, {res['scenario_h0.0']['total_net']:+.2f}]) ret {sc['ret_mean'] * 100:+.3f}% rand {sc['rand_ret_mean'] * 100:+.3f}% diff {sc['diff_mean'] * 100:+.3f}% CI95 [{sc['ci95'][0] * 100:+.3f}%, {sc['ci95'][1] * 100:+.3f}%] P(<=0) {sc['p_le0']:.3f} | forced {forced} rand_forced {res['rand_forced_total']:.1f} unfilled {res['unfilled']} | observable diff {ob['diff_mean'] * 100:+.3f}% CI [{ob['ci95'][0] * 100:+.3f}%, {ob['ci95'][1] * 100:+.3f}%] | fill-cond diff {fc['diff_mean'] * 100:+.3f}% CI [{fc['ci95'][0] * 100:+.3f}%, {fc['ci95'][1] * 100:+.3f}%] | ex-top3 ${res['ex_top3_net']:+.2f} max-principal ${capital:.0f} scenario-net/max-principal {res['scenario_return_on_max_principal'] * 100:+.2f}% spy {res['spy_ret_mean'] * 100:+.3f}% | {res['pnl_status']}", flush=True)
    for y, v in yrs.items():
        print(f"   {y}: n {v['n']} net ${v['net']:+.2f} ret {v['ret_mean'] * 100:+.3f}% rand {v['rand_ret_mean'] * 100:+.3f}% forced {v['forced']} unfilled {v['unfilled']}", flush=True)
    return res


def spy_ret(db, ds):
    a, b = SPYO[db], SPYO[ds]
    return float(b / a - 1) if np.isfinite(a) and np.isfinite(b) else np.nan


def periodic(name, signal, H, step, side, first):
    res = {}
    for mode in ("decile", "top10"):
        for A in SIZES:
            rows = []
            for r in range(first, D - 1 - H, step):
                s = signal(r)
                m = ELIG[r] & LISTED[r] & np.isfinite(s)
                idx = np.where(m)[0]
                if len(idx) < MIN_UNIVERSE:
                    continue
                k = max(1, int(len(idx) * Q)) if mode == "decile" else min(TOPK, len(idx))
                order = np.argsort(s[idx], kind="stable")
                pick = idx[order[-k:]] if side == "top" else idx[order[:k]]
                db = np.full(k, r + 1)
                ds = np.full(k, r + 1 + H)
                row = evaluate(pick, db, ds, A, idx, (name, mode, A, r))
                row.update(day=DAYS[r], year=int(YEAR[r]), k=int(k), universe=int(len(idx)), spy_ret=spy_ret(r + 1, r + 1 + H), picks=[SYMS[j] for j in pick] if mode == "top10" else None)
                rows.append(row)
            cap = float(max(r["invested"] for r in rows))
            res[f"{mode}_${A:.0f}"] = dict(summary=agg(f"{name} {mode} ${A:.0f}", rows, capital=cap), rows=rows)
    return res


def mom_signal(r):
    s = C[r - 21] / C[r - 252] - 1
    s[~(LISTED[r - 21] & LISTED[r - 252])] = np.nan
    return s


def rev_signal(r):
    s = C[r] / C[r - 5] - 1
    s[~(LISTED[r] & LISTED[r - 5])] = np.nan
    return s


def load_events():
    ET = ZoneInfo("America/New_York")
    ev = set()
    stats = dict(files=0, tickers_in_panel=0, raw_202=0, mapped=0)
    for p in sorted(glob.glob(os.path.join(EDGAR, "*.jsonl"))):
        stats["files"] += 1
        lines = open(p, encoding="utf-8").read().splitlines()
        if not lines:
            continue
        head = json.loads(lines[0])
        t = head.get("ticker")
        if t not in SIDX:
            continue
        stats["tickers_in_panel"] += 1
        for l in lines[1:]:
            d = json.loads(l)
            if d.get("form") != "8-K" or "2.02" not in (d.get("items") or "").split(","):
                continue
            acc = d.get("accepted")
            if not acc:
                continue
            stats["raw_202"] += 1
            ts = dt.datetime.fromisoformat(acc.replace("Z", "+00:00")).astimezone(ET)
            day = ts.date() + (dt.timedelta(days=1) if ts.time() >= dt.time(16, 0) else dt.timedelta(0))
            i = bisect.bisect_left(DAYS, day.isoformat())
            if i >= D or i < 1:
                continue
            ev.add((SIDX[t], i))
            stats["mapped"] += 1
    return sorted(ev, key=lambda x: (x[1], x[0])), stats


def ead(H=20):
    events, stats = load_events()
    print(f"EAD events: {stats} unique {len(events)}", flush=True)
    res = dict(stats=stats, unique_events=len(events))
    rec = [(j, i, float(C[i, j] / C[i - 1, j] - 1)) for j, i in events if i + 1 + H < D and LISTED[i, j] and LISTED[i - 1, j] and ELIG[i, j]]
    print(f"EAD usable events {len(rec)}", flush=True)
    variants = dict(ar_ge_5=lambda a: a >= 0.05, ar_ge_10=lambda a: a >= 0.10, ar_le_m5=lambda a: a <= -0.05, all=lambda a: True)
    for vn, cond in variants.items():
        sel = [(j, i, a) for j, i, a in rec if cond(a)]
        for A in SIZES:
            rows = []
            for j, i, a in sel:
                idx = np.where(ELIG[i] & LISTED[i] & RMASK)[0]
                row = evaluate(np.array([j]), np.array([i + 1]), np.array([i + 1 + H]), A, idx, ("ead", vn, A, i, j))
                row.update(day=DAYS[i], year=int(YEAR[i]), month=DAYS[i][:7], sym=SYMS[j], ar=a, spy_ret=spy_ret(i + 1, i + 1 + H))
                rows.append(row)
            if not rows:
                continue
            conc = np.zeros(D, dtype=int)
            for r in rows:
                if r["filled"]:
                    i0 = DAYS.index(r["day"]) + 1
                    conc[i0:i0 + H] += 1
            s = agg(f"EAD {vn} H{H} ${A:.0f}", rows, group_key="month", capital=float(conc.max() * A))
            s["max_concurrent"] = int(conc.max())
            res[f"{vn}_${A:.0f}"] = dict(summary=s, rows=rows)
    return res


if __name__ == "__main__":
    meta = dict(version=2, script_sha256=sha(__file__), panel=os.path.relpath(PANEL, HERE), panel_sha256=sha(PANEL), etf_panel_sha256=sha(ETF), edgar_dir=os.path.relpath(EDGAR, HERE), edgar_dir_sha256=sha_dir(EDGAR), only=ONLY, rand_syms=RAND_SYMS, rand_universe_syms=int(RMASK.sum()), rand_universe_sha256=hashlib.sha256(",".join(sorted(s for s, m in zip(SYMS, RMASK) if m)).encode()).hexdigest()[:16], edgar_files=len(EDGAR_TICKERS), edgar_tickers_in_panel=int(sum(t in SIDX for t in EDGAR_TICKERS)), days=[DAYS[0], DAYS[-1]], n_days=D, n_syms=N, fee=FEE, sec=SEC, min_fee=MIN_FEE, tick=TICK, haircuts=list(HAIRCUTS), primary_haircut=HC, sizes=list(SIZES), seeds=SEEDS, decile=Q, topk=TOPK, nboot=NBOOT, started=dt.datetime.now().isoformat(timespec="seconds"))
    print("META " + json.dumps(meta), flush=True)
    if len(sys.argv) > 1 and sys.argv[1] == "meta":
        sys.exit(0)
    out = dict(meta=meta)
    if ONLY in ("", "MOM"):
        out["MOM"] = periodic("MOM 12-1 top", mom_signal, 21, 21, "top", 252)
    if ONLY in ("", "REV"):
        out["REV"] = periodic("REV 5d bottom", rev_signal, 5, 5, "bottom", 20)
    if ONLY in ("", "EAD"):
        out["EAD"] = ead(20)
    spy_days = [i for i in range(D) if np.isfinite(SPYO[i])]
    out["SPY_reference"] = dict(first=DAYS[spy_days[0]], last=DAYS[spy_days[-1]], open_to_open=float(SPYO[spy_days[-1]] / SPYO[spy_days[0]] - 1), by_year={str(y): float(SPYO[np.where(YEAR == y)[0][-1]] / SPYO[np.where(YEAR == y)[0][0]] - 1) for y in sorted(set(YEAR))})
    print("SPY reference (no costs): " + json.dumps(out["SPY_reference"]), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"saved {OUT}", flush=True)
