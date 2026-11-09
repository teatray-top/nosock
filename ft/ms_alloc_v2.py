import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
import math
import sys
import warnings
from datetime import date

sys.dont_write_bytecode = True
warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import learn_inverse as LI
import msma_v1 as M1
import ms_role_v2 as R
import ms_alloc_v1 as A1

SLOTS = A1.SLOTS
SEEDS = A1.SEEDS
BUFFER = 0.02
DEPOSIT = R.DEPOSIT
VAULT_SHARE = R.VAULT_SHARE


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_raw():
    man = json.load(open(os.path.join(R.ETF_DIR, "manifest.json"), encoding="utf-8"))
    out = {}
    for code in M1.CLASS:
        info = man["files"][f"{code}_raw"]
        p = os.path.join(R.ETF_DIR, info["file"])
        if sha(p) != info["sha256"]:
            raise SystemExit(f"raw hash differs {code}")
        rec = json.load(open(p, encoding="utf-8"))
        out[code] = dict(zip(rec["days"], rec["C"]))
    return out


def attach_prices(dec, inst, raw):
    for D, row in dec.items():
        for c, r in row.items():
            if "R" not in r:
                continue
            ins = inst[c]
            t = r["t"]
            rc = raw[c].get(D)
            if rc is None or not rc > 0:
                r["ref_won"] = None
                continue
            f = rc / ins["C"][t]
            d = float(R.GRID[r["a"]][0])
            e = t + 1
            if d == 0:
                entry_adj = ins["O"][e]
                ref = rc * (1 + BUFFER)
            else:
                lvl = ins["C"][t] * (1 - d)
                entry_adj = min(lvl, ins["O"][e]) if ins["O"][e] < lvl else lvl
                ref = rc * (1 - d)
            r["ref_won"] = ref
            r["entry_won"] = entry_adj * f
            r["filled"] = not (d > 0 and ins["L"][e] > ins["C"][t] * (1 - d))


def engine(dec, days, slots, order):
    pool, vault, dep = 0.0, 0.0, 0.0
    opn = {}
    events, losses = [], []
    use = {}
    trades, skipped_price = 0, 0
    vault_year = {}

    def release(c):
        nonlocal pool, vault
        size, Rr, exday = opn.pop(c)
        prof = size * math.expm1(Rr)
        if prof > 0:
            v = VAULT_SHARE * prof
            vault += v
            vault_year[exday[:4]] = vault_year.get(exday[:4], 0.0) + v
            events.append((exday, v))
            pool += size + prof - v
        else:
            pool += size + prof
            if prof < 0:
                losses.append(prof)

    for D in days:
        for c in [c for c, x in opn.items() if x[2] <= D]:
            release(c)
        pool += DEPOSIT
        dep += DEPOSIT
        cands = [(c, r) for c, r in dec[D].items() if c not in opn and r["v"] > 0 and "R" in r and r.get("ref_won")]
        cands = order(cands)
        equity = pool + sum(x[0] for x in opn.values())
        for c, r in cands:
            if len(opn) >= slots:
                break
            budget = min(pool, equity / slots)
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
            opn[c] = (size, r["R"], r["exit_day"])
            pool -= size
            trades += 1
            u = use.setdefault(c, dict(money=0.0, trades=0, cycles={}))
            u["money"] += size
            u["trades"] += 1
            key = f"hold{r['hz']}" if r["hold"] else str(r["hz"])
            u["cycles"][key] = u["cycles"].get(key, 0) + 1
    for c in list(opn):
        release(c)
    cash_days = sorted(set(e[0] for e in events))
    gaps = [(date.fromisoformat(b) - date.fromisoformat(a)).days for a, b in zip(cash_days[:-1], cash_days[1:])]
    months = len(days) / 21.0
    first_cash = cash_days[0] if cash_days else None
    return dict(deposits_won=dep, vault_won=vault, acct_won=pool, total_won=pool + vault, gain_over_deposits_pct=100 * ((pool + vault) / dep - 1), trades=trades, skipped_for_share_price=skipped_price, losing_trades=len(losses), loss_sum_won=float(sum(losses)), worst_loss_won=float(min(losses)) if losses else 0.0, first_cash_out_day=first_cash, cash_out_days=len(cash_days), median_days_between_cash_outs=float(np.median(gaps)) if gaps else None, vault_per_month_avg_won=vault / months, vault_last_12_months_won=sum(v for d, v in events if d >= "2025-11-06"), vault_by_year_won=vault_year, use=use)


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, pth in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), pth)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    recs = R.load()
    inst = {c: R.instrument(recs[c], c) for c in M1.CLASS}
    raw = load_raw()
    samp = A1.samples(inst)
    print("samples done", flush=True)
    dec = A1.decisions(inst, samp)
    attach_prices(dec, inst, raw)
    days = sorted(dec)
    names = json.load(open(os.path.join(R.ETF_DIR, "manifest.json"), encoding="utf-8"))["names"]
    res = {}
    by_value = lambda cs: sorted(cs, key=lambda z: -z[1]["v"])
    for k in SLOTS:
        ai = engine(dec, days, k, by_value)
        rnd = []
        for s in range(SEEDS):
            g = np.random.default_rng(900 + 10 * k + s)
            rnd.append(engine(dec, days, k, lambda cs, g=g: [cs[i] for i in g.permutation(len(cs))]))
        ctl = dict(total_won_mean=float(np.mean([x["total_won"] for x in rnd])), vault_won_mean=float(np.mean([x["vault_won"] for x in rnd])), total_won_range=[float(min(x["total_won"] for x in rnd)), float(max(x["total_won"] for x in rnd))], ai_rank_beats=int(sum(ai["total_won"] > x["total_won"] for x in rnd)))
        res[f"slots{k}"] = dict(AI_ranked=ai, random_order_among_positive=ctl)
        print(f"slots {k}: deposits {ai['deposits_won'] / 1e4:.0f}만 vault {ai['vault_won'] / 1e4:.1f}만 acct {ai['acct_won'] / 1e4:.1f}만 total {ai['total_won'] / 1e4:.1f}만 ({ai['gain_over_deposits_pct']:+.1f}%) trades {ai['trades']} skipped-for-price {ai['skipped_for_share_price']} losing {ai['losing_trades']} worst {ai['worst_loss_won'] / 1e4:.1f}만 first cash-out {ai['first_cash_out_day']} cash-out days {ai['cash_out_days']} median gap {ai['median_days_between_cash_outs']} d, vault last 12m {ai['vault_last_12_months_won'] / 1e4:.1f}만 | random order: total {ctl['total_won_mean'] / 1e4:.1f}만, AI beats {ctl['ai_rank_beats']}/{SEEDS}", flush=True)
        tot = sum(u["money"] for u in ai["use"].values())
        for c, u in sorted(ai["use"].items(), key=lambda z: -z[1]["money"]):
            print(f"    {c} {names[c][:16]:16s} money share {100 * u['money'] / tot:5.1f}% trades {u['trades']:4d} cycles {u['cycles']}", flush=True)
        print("    vault by year", {y: round(v / 1e4) for y, v in sorted(ai["vault_by_year_won"].items())}, flush=True)
    rec = dict(version="ms-alloc-v2", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], results=res)
    with open(out_path, "x", encoding="utf-8") as fo:
        json.dump(rec, fo, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
