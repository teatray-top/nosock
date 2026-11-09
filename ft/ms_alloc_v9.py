import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
import sys
import warnings

sys.dont_write_bytecode = True
warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import msma_v1 as M1
import ms_role_v2 as R
import ms_alloc_v3 as A3
import ms_alloc_v6 as A6
import ms_alloc_v7 as A7

KRUS_DIR = os.path.join(os.path.dirname(HERE), "data", "kr_etf", "yf_kr_us_20261108_114455")
SEEDS = 20


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_krus():
    man = json.load(open(os.path.join(KRUS_DIR, "manifest.json"), encoding="utf-8"))
    out = {}
    for key, info in man["files"].items():
        p = os.path.join(KRUS_DIR, info["file"])
        if sha(p) != info["sha256"]:
            raise SystemExit(f"kr-us hash differs {key}")
        out[key] = json.load(open(p, encoding="utf-8"))
    return man, out


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, pth in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), pth)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    man, krus = load_krus()
    classes = sorted(set(M1.CLASSES) | set(man["classes"].values()))
    recs = R.load()
    inst = {c: A7.instrument_x(recs[c], c, M1.CLASS[c], classes, "KR", A7.kr_cost) for c in M1.CLASS}
    for c, cls in man["classes"].items():
        inst[c] = A7.instrument_x(krus[f"{c}_adj"], c, cls, classes, "KR", A7.kr_cost)
    raw = A3.load_raw()
    for c in man["classes"]:
        r = krus[f"{c}_raw"]
        raw[c] = dict(zip(r["days"], r["C"]))
    samp = A7.samples(inst)
    dec = A7.decisions(inst, samp)
    A3.attach_prices(dec, inst, raw)
    days = sorted(dec)
    names = dict(json.load(open(os.path.join(R.ETF_DIR, "manifest.json"), encoding="utf-8"))["names"], **man["names"])
    ai = A6.engine(dec, days)
    rnd = []
    for s in range(SEEDS):
        g = np.random.default_rng(5500 + s)

        def pick(free, g=g):
            k = sum(1 for _, r in free if r["v"] > 0)
            if k == 0:
                return []
            idx = g.choice(len(free), size=k, replace=False)
            return [free[i] for i in idx]

        rnd.append(A6.engine(dec, days, pick))
    fs = [r["f"] for row in dec.values() for r in row.values() if r["v"] > 0]
    f_mean = float(np.mean(fs)) if fs else 0.2
    flat = A6.engine(dec, days, fixed_f=f_mean)
    ctl = dict(random_instruments_same_count=dict(total_won_mean=float(np.mean([x["total_won"] for x in rnd])), total_won_range=[float(min(x["total_won"] for x in rnd)), float(max(x["total_won"] for x in rnd))], ai_beats=int(sum(ai["total_won"] > x["total_won"] for x in rnd))), same_selection_flat_size=dict(f=f_mean, total_won=flat["total_won"], vault_won=flat["vault_won"]))
    print(f"KR 15 + KR-listed US 13 (won, no FX), model selects and sizes: deposits {ai['deposits_won'] / 1e4:.0f}만 vault {ai['vault_won'] / 1e4:.1f}만 acct {ai['acct_won'] / 1e4:.1f}만 total {ai['total_won'] / 1e4:.1f}만 ({ai['gain_over_deposits_pct']:+.1f}%) net realized {ai['net_realized_profit_won'] / 1e4:.1f}만 trades {ai['trades']} skipped {ai['skipped_for_share_price']} positions mean {ai['open_positions_mean']:.1f} invested {100 * ai['invested_share_mean']:.0f}% losing {ai['losing_trades']} worst {ai['worst_loss_won'] / 1e4:.1f}만 cash-out days {ai['cash_out_days']} median gap {ai['median_days_between_cash_outs']} d, vault last 12m {ai['vault_last_12_months_won'] / 1e4:.1f}만", flush=True)
    print(f"  control random instruments: total {ctl['random_instruments_same_count']['total_won_mean'] / 1e4:.1f}만 range {[round(x / 1e4) for x in ctl['random_instruments_same_count']['total_won_range']]}, AI beats {ctl['random_instruments_same_count']['ai_beats']}/{SEEDS}; flat size f={f_mean:.3f}: total {flat['total_won'] / 1e4:.1f}만", flush=True)
    tot = sum(u["money"] for u in ai["use"].values())
    for c, u in sorted(ai["use"].items(), key=lambda z: -z[1]["money"]):
        print(f"    {c} {names[c][:18]:18s} money share {100 * u['money'] / tot:5.1f}% trades {u['trades']:4d} mean size {100 * u['f_sum'] / u['trades']:4.1f}% daily vol {100 * float(np.nanmean(inst[c]['v20'])):.2f}% cycles {u['cycles']}", flush=True)
    print("    vault by year", {y: round(v / 1e4) for y, v in sorted(ai["vault_by_year_won"].items())}, flush=True)
    res = dict(AI=ai, controls=ctl, classes=classes)
    rec = dict(version="ms-alloc-v9", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], results=res)
    with open(out_path, "x", encoding="utf-8") as fo:
        json.dump(rec, fo, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
