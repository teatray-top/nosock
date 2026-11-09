import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
import math
import sys
import warnings
import bisect
from datetime import date

sys.dont_write_bytecode = True
warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import lightgbm as lgb
import learn_inverse as LI
import harvest_v8 as H8
import msma_v1 as M1
import ms_role_v2 as R
import ms_alloc_v3 as A3
import ms_alloc_v6 as A6

US_DIR = os.path.join(os.path.dirname(HERE), "data", "us_etf", "yf_us_etfs_20261108_112233")
END = M1.END
F_GRID = A6.F_GRID
SEEDS = 20
DEPOSIT = R.DEPOSIT
VAULT_SHARE = R.VAULT_SHARE
US_TICK = 0.01
MIN_US_ORDER = 1000.0


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_us():
    man = json.load(open(os.path.join(US_DIR, "manifest.json"), encoding="utf-8"))
    out = {}
    for key, info in man["files"].items():
        p = os.path.join(US_DIR, info["file"])
        if sha(p) != info["sha256"]:
            raise SystemExit(f"us hash differs {key}")
        out[key] = json.load(open(p, encoding="utf-8"))
    return man, out


def instrument_x(rec, code, cls, classes, market, costf_maker, fx=None):
    keep = [i for i, d in enumerate(rec["days"]) if d <= END and (fx is None or d in fx)]
    days = [rec["days"][i] for i in keep]
    n = len(days)
    m = np.array([fx[d] for d in days]) if fx is not None else np.ones(n)
    O, H, L, C = (np.asarray([rec[k][i] for i in keep], float) * m for k in ("O", "H", "L", "C"))
    r = np.r_[np.nan, np.diff(np.log(C))]
    rs = lambda w: np.array([np.std(r[max(1, i - w + 1):i + 1]) if i >= w else np.nan for i in range(n)])
    v20, v60 = rs(20), rs(60)
    S, _, _ = M1.signals(np.array(days), C[:, None])
    eq = np.nanmean(np.tanh(np.stack([S[:, :, M1.SIG.index(s)] for s in M1.EQ_SIGNALS], axis=2)), axis=2)[:, 0]
    hi = np.array([C[max(0, i - 249):i + 1].max() for i in range(n)])
    lc = np.log(C)
    lag = lambda k: np.r_[[np.nan] * k, lc[k:] - lc[:-k]]
    gaps = np.array([(date.fromisoformat(days[i + 1]) - date.fromisoformat(days[i])).days if i + 1 < n else 1 for i in range(n)], float)
    dow = np.array([date.fromisoformat(x).weekday() for x in days], float)
    onehot = np.zeros((n, len(classes)))
    onehot[:, classes.index(cls)] = 1.0
    F = np.column_stack([lag(1), lag(5), lag(20), lag(60), v20, v60, v20 / v60, C / hi - 1, eq, dow, gaps, onehot])
    ins = dict(code=code, market=market, days=days, O=O, H=H, L=L, C=C, F=F, v20=v20, fxv=m)
    ins["cost"] = costf_maker(ins)
    return ins


def kr_cost(ins):
    return lambda k, p: LI.FEE + H8.tick(ins["days"][k], p) / p


def us_cost_maker(fee, spread):
    return lambda ins: (lambda k, p: fee + spread + US_TICK * ins["fxv"][k] / p)


def outcome_x(ins, t, act):
    d, tp, sl, hz = act
    O, H, L, C = ins["O"], ins["H"], ins["L"], ins["C"]
    n = len(C)
    e = t + 1
    if e >= n:
        return np.nan, t
    if d == 0:
        pin, k = O[e], e
    else:
        lvl = C[t] * (1 - d)
        if L[e] > lvl:
            return 0.0, e
        pin, k = (min(lvl, O[e]) if O[e] < lvl else lvl), e
    cost = lambda p: ins["cost"](min(k, n - 1), p)
    cin = cost(pin)
    stop = min(k + int(hz) - 1, n - 1)
    up = pin * (1 + tp + 2 * cin) if tp > 0 else np.inf
    dn = pin * (1 - sl) if sl > 0 else -np.inf
    for j in range(k, stop + 1):
        first = j > k or d == 0
        if L[j] <= dn:
            px = min(dn, O[j]) if (j > k and O[j] < dn) else dn
            return math.log(px * (1 - cost(px)) / (pin * (1 + cin))), j
        if H[j] >= up and first:
            px = max(up, O[j]) if (j > k and O[j] > up) else up
            return math.log(px * (1 - cost(px)) / (pin * (1 + cin))), j
    px = C[stop]
    return math.log(px * (1 - cost(px)) / (pin * (1 + cin))), stop


def samples(inst):
    rng = np.random.default_rng(31)
    hold_idx = np.nonzero(R.GRID[:, 1] == 0)[0]
    samp = {}
    for c, ins in inst.items():
        n = len(ins["C"])
        rows_ = []
        for t in range(260, n - 1):
            if np.isnan(ins["F"][t]).any():
                continue
            idx = np.r_[hold_idx, rng.choice(len(R.GRID) - len(hold_idx), size=R.SAMPLES - len(hold_idx), replace=False)]
            out = [outcome_x(ins, t, R.GRID[a]) for a in idx]
            rows_.append((t, idx, np.array([o[0] for o in out]), np.array([o[1] for o in out])))
        samp[c] = rows_
        print("samples", c, len(rows_), flush=True)
    return samp


def decisions(inst, samp):
    rng = np.random.default_rng(77)
    fsamp = {c: [rng.choice(F_GRID, size=len(idx)) for _, idx, _, _ in samp[c]] for c in inst}
    dec = {}
    for y in R.TEST_YEARS:
        cut = f"{y}-01-01"
        X, Y = [], []
        for c, ins in inst.items():
            for (t, idx, Rr, ex), fs in zip(samp[c], fsamp[c]):
                if ins["days"][t] < cut and ex.max() < len(ins["days"]) and ins["days"][int(ex.max())] < cut and not np.isnan(Rr).any():
                    X.append(A6.rows_f(ins["F"][t], R.GRID[idx], ins["v20"][t], fs).astype(np.float32))
                    Y.append(np.log1p(fs * np.expm1(Rr)))
        m = lgb.train(LI.PARAMS, lgb.Dataset(np.vstack(X), label=np.concatenate(Y)), num_boost_round=LI.ROUNDS)
        del X
        for c, ins in inst.items():
            n = len(ins["C"])
            for t, dday in enumerate(ins["days"]):
                if dday[:4] != str(y) or t < 260 or t + 1 >= n or np.isnan(ins["F"][t]).any():
                    continue
                Xa, ai, fi = A6.full_rows(ins["F"][t], ins["v20"][t])
                p = m.predict(Xa)
                k = int(np.argmax(p))
                a = int(ai[k])
                rec = dict(t=t, a=a, f=float(fi[k]), v=float(p[k]), market=ins["market"])
                Rr, ex = outcome_x(ins, t, R.GRID[a])
                if not np.isnan(Rr):
                    rec.update(R=float(Rr), exit_day=ins["days"][int(ex)], hz=int(R.GRID[a][3]), hold=bool(R.GRID[a][1] == 0))
                dec.setdefault(dday, {})[c] = rec
        print("year", y, "model fitted, sample-days", len(Y), flush=True)
    return dec


def attach(dec, inst, raw_kr):
    kr_dec = {D: {c: r for c, r in row.items() if r["market"] == "KR"} for D, row in dec.items()}
    A3.attach_prices(kr_dec, inst, raw_kr)
    for D, row in dec.items():
        for c, r in row.items():
            if r["market"] != "US" or "R" not in r:
                continue
            ins = inst[c]
            t = r["t"]
            d = float(R.GRID[r["a"]][0])
            r["ref_won"] = 1.0
            r["frac"] = True
            r["filled"] = not (d > 0 and ins["L"][t + 1] > ins["C"][t] * (1 - d))


def engine(dec, days, kr_days, pick=None, fixed_f=None):
    pool, vault, dep = 0.0, 0.0, 0.0
    cum, hwm = 0.0, 0.0
    opn = {}
    events, losses = [], []
    use = {}
    trades, skipped_price = 0, 0
    vault_year = {}
    npos, expo, us_share = [], [], []

    def release(c):
        nonlocal pool, vault, cum, hwm
        size, Rr, exday, mk = opn.pop(c)
        prof = size * math.expm1(Rr)
        cum += prof
        v = VAULT_SHARE * (cum - hwm) if cum > hwm else 0.0
        hwm = max(hwm, cum)
        if v > 0:
            vault += v
            vault_year[exday[:4]] = vault_year.get(exday[:4], 0.0) + v
            events.append((exday, v))
        pool += size + prof - v
        if prof < 0:
            losses.append(prof)

    def phase(D, mk):
        nonlocal pool, trades, skipped_price
        row = {c: r for c, r in dec.get(D, {}).items() if r["market"] == mk}
        free = [(c, r) for c, r in row.items() if c not in opn and "R" in r and r.get("ref_won")]
        cands = pick(free) if pick else sorted([(c, r) for c, r in free if r["v"] > 0], key=lambda z: -z[1]["v"])
        equity = pool + sum(x[0] for x in opn.values())
        for c, r in cands:
            f = fixed_f if fixed_f is not None else r["f"]
            budget = min(pool, f * equity)
            if r.get("frac"):
                if budget < MIN_US_ORDER:
                    skipped_price += 1
                    continue
                size = budget
                Rr = r["R"]
            else:
                shares = math.floor(budget / r["ref_won"])
                if shares < 1:
                    skipped_price += 1
                    continue
                px = r["entry_won"] if r["filled"] else r["ref_won"]
                size = shares * px
                while size > pool and shares > 0:
                    shares -= 1
                    size = shares * px
                if shares < 1:
                    skipped_price += 1
                    continue
                Rr = r["R"]
            opn[c] = (size, Rr, r["exit_day"], mk)
            pool -= size
            trades += 1
            u = use.setdefault(c, dict(money=0.0, trades=0, f_sum=0.0, cycles={}))
            u["money"] += size
            u["trades"] += 1
            u["f_sum"] += f
            key = f"hold{r['hz']}" if r["hold"] else str(r["hz"])
            u["cycles"][key] = u["cycles"].get(key, 0) + 1

    for D in days:
        for c in [c for c, x in opn.items() if (x[3] == "KR" and x[2] <= D) or (x[3] == "US" and x[2] < D)]:
            release(c)
        if D in kr_days:
            pool += DEPOSIT
            dep += DEPOSIT
        phase(D, "KR")
        for c in [c for c, x in opn.items() if x[2] <= D]:
            release(c)
        phase(D, "US")
        npos.append(len(opn))
        inv = sum(x[0] for x in opn.values())
        eq = pool + inv
        expo.append(inv / eq if eq > 0 else 0.0)
        us_share.append(sum(x[0] for x in opn.values() if x[3] == "US") / inv if inv > 0 else 0.0)
    for c in list(opn):
        release(c)
    cash_days = sorted(set(e[0] for e in events))
    gaps = [(date.fromisoformat(b) - date.fromisoformat(a)).days for a, b in zip(cash_days[:-1], cash_days[1:])]
    months = len(kr_days) / 21.0
    return dict(net_realized_profit_won=cum, deposits_won=dep, vault_won=vault, acct_won=pool, total_won=pool + vault, gain_over_deposits_pct=100 * ((pool + vault) / dep - 1), trades=trades, skipped_for_share_price=skipped_price, losing_trades=len(losses), loss_sum_won=float(sum(losses)), worst_loss_won=float(min(losses)) if losses else 0.0, open_positions_mean=float(np.mean(npos)), open_positions_max=int(max(npos)), invested_share_mean=float(np.mean(expo)), us_share_of_invested_mean=float(np.mean(us_share)), cash_out_days=len(cash_days), median_days_between_cash_outs=float(np.median(gaps)) if gaps else None, vault_per_month_avg_won=vault / months, vault_last_12_months_won=sum(v for d, v in events if d >= "2025-11-06"), vault_by_year_won=vault_year, use=use)


def hold_same_deposits(ins, kr_days, us):
    units, dep = 0.0, 0.0
    days = ins["days"]
    for D in kr_days:
        j = bisect.bisect_left(days, D) if us else bisect.bisect_right(days, D)
        if j >= len(days):
            continue
        p = ins["O"][j]
        units += DEPOSIT / (p * (1 + ins["cost"](j, p)))
        dep += DEPOSIT
    return dict(deposits_won=dep, total_won=units * ins["C"][-1])


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, pth in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), pth)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    us_fee, us_spread = float(spec["us_fee"]), float(spec["us_fx_spread"])
    man_us, us = load_us()
    fxrec = us["KRW_X_adj"]
    fx = {d: c for d, c in zip(fxrec["days"], fxrec["C"]) if c > 500}
    fx_days = sorted(fx)
    us_days_all = sorted(set(d for k, v in us.items() if k.endswith("_adj") and not k.startswith("KRW") for d in v["days"]))
    fx_ff, last = {}, None
    j = 0
    for d in us_days_all:
        while j < len(fx_days) and fx_days[j] <= d:
            last = fx[fx_days[j]]
            j += 1
        if last is not None:
            fx_ff[d] = last
    classes = sorted(set(M1.CLASSES) | set(man_us["classes"].values()))
    recs = R.load()
    inst = {}
    for c in M1.CLASS:
        inst[c] = instrument_x(recs[c], c, M1.CLASS[c], classes, "KR", kr_cost)
    for c, cls in man_us["classes"].items():
        inst[c] = instrument_x(us[f"{c}_adj"], c, cls, classes, "US", us_cost_maker(us_fee, us_spread), fx_ff)
    chk = R.instrument(recs["069500"], "069500")
    t0 = 400
    same = all(abs(outcome_x(inst["069500"], t0, R.GRID[a])[0] - R.outcome(chk, t0, R.GRID[a])[0]) < 1e-12 for a in range(0, len(R.GRID), 37))
    print("KR outcome equals ms_role_v2 outcome:", same, flush=True)
    if not same:
        raise SystemExit("KR outcome mismatch")
    raw_kr = A3.load_raw()
    samp = samples(inst)
    dec = decisions(inst, samp)
    attach(dec, inst, raw_kr)
    kr_days = set(d for d in dec if any(r["market"] == "KR" for r in dec[d].values()))
    days = sorted(dec)
    names = dict(json.load(open(os.path.join(R.ETF_DIR, "manifest.json"), encoding="utf-8"))["names"], **man_us["names"])
    ai = engine(dec, days, kr_days)
    rnd = []
    for s in range(SEEDS):
        g = np.random.default_rng(3500 + s)

        def pick(free, g=g):
            k = sum(1 for _, r in free if r["v"] > 0)
            if k == 0:
                return []
            idx = g.choice(len(free), size=k, replace=False)
            return [free[i] for i in idx]

        rnd.append(engine(dec, days, kr_days, pick))
    fs = [r["f"] for row in dec.values() for r in row.values() if r["v"] > 0]
    f_mean = float(np.mean(fs)) if fs else 0.2
    flat = engine(dec, days, kr_days, fixed_f=f_mean)
    first = min(kr_days)
    kd = sorted(kr_days)
    holds = {c: hold_same_deposits(inst[c], kd, inst[c]["market"] == "US") for c in ("069500", "SPY", "QQQ")}
    ctl = dict(random_instruments_same_count=dict(total_won_mean=float(np.mean([x["total_won"] for x in rnd])), total_won_range=[float(min(x["total_won"] for x in rnd)), float(max(x["total_won"] for x in rnd))], ai_beats=int(sum(ai["total_won"] > x["total_won"] for x in rnd))), same_selection_flat_size=dict(f=f_mean, total_won=flat["total_won"], vault_won=flat["vault_won"]))
    print(f"KR+US AI select+size (US fee {100 * us_fee:.2f}% + fx {100 * us_spread:.2f}%): deposits {ai['deposits_won'] / 1e4:.0f}만 vault {ai['vault_won'] / 1e4:.1f}만 acct {ai['acct_won'] / 1e4:.1f}만 total {ai['total_won'] / 1e4:.1f}만 ({ai['gain_over_deposits_pct']:+.1f}%) net realized {ai['net_realized_profit_won'] / 1e4:.1f}만 trades {ai['trades']} skipped {ai['skipped_for_share_price']} positions mean {ai['open_positions_mean']:.1f} invested {100 * ai['invested_share_mean']:.0f}% US share of invested {100 * ai['us_share_of_invested_mean']:.0f}% losing {ai['losing_trades']} worst {ai['worst_loss_won'] / 1e4:.1f}만 cash-out days {ai['cash_out_days']} median gap {ai['median_days_between_cash_outs']} d, vault last 12m {ai['vault_last_12_months_won'] / 1e4:.1f}만", flush=True)
    print(f"  control random instruments: total {ctl['random_instruments_same_count']['total_won_mean'] / 1e4:.1f}만 range {[round(x / 1e4) for x in ctl['random_instruments_same_count']['total_won_range']]}, AI beats {ctl['random_instruments_same_count']['ai_beats']}/{SEEDS}; flat size f={f_mean:.3f}: total {flat['total_won'] / 1e4:.1f}만", flush=True)
    print("  same deposits held:", {c: round(h["total_won"] / 1e4, 1) for c, h in holds.items()}, "만", flush=True)
    tot = sum(u["money"] for u in ai["use"].values())
    for c, u in sorted(ai["use"].items(), key=lambda z: -z[1]["money"]):
        print(f"    {c:6s} {names[c][:16]:16s} money share {100 * u['money'] / tot:5.1f}% trades {u['trades']:4d} mean size {100 * u['f_sum'] / u['trades']:4.1f}% daily vol {100 * float(np.nanmean(inst[c]['v20'])):.2f}% cycles {u['cycles']}", flush=True)
    print("    vault by year", {y: round(v / 1e4) for y, v in sorted(ai["vault_by_year_won"].items())}, flush=True)
    res = dict(AI=ai, controls=ctl, holds_same_deposits=holds, us_fee=us_fee, us_fx_spread=us_spread, classes=classes, first_day=first)
    rec = dict(version="ms-alloc-v7", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], results=res)
    with open(out_path, "x", encoding="utf-8") as fo:
        json.dump(rec, fo, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
