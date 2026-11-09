import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
import math
import sys
import warnings

sys.dont_write_bytecode = True
warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import pandas as pd
import learn_inverse as LI
import scalp_etf_v1 as S1

CAPITAL = 100_000_000.0
STEP = 0.25
START_W = 0.5
SPACINGS = [("g0.25pct", 0.0025, None), ("g0.5pct", 0.005, None), ("g1.0pct", 0.01, None), ("g0.5sd", None, 0.5), ("g1.0sd", None, 1.0)]
TEST_FROM = "2023-01-01"
EXIT_HM = "15:19"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def day_table(d):
    day = d["day"].to_numpy()
    lc = np.log(d["close"].to_numpy(float))
    same = np.r_[False, day[1:] == day[:-1]]
    r1 = np.where(same, np.r_[np.nan, np.diff(lc)], np.nan)
    dv = pd.Series(r1).groupby(day).std()
    ud = list(dv.index)
    v = dv.to_numpy()
    s3 = np.array([np.nanmean(v[max(0, k - 3):k]) if k >= 3 else np.nan for k in range(len(ud))])
    med = np.array([np.nanmedian(v[max(0, k - 250):k]) if k >= 60 else np.nan for k in range(len(ud))])
    return {u: dict(turb=(bool(s3[k] > 1.5 * med[k]) if not np.isnan(med[k]) else None), sd_day=s3[k] * math.sqrt(380) if not np.isnan(s3[k]) else np.nan) for k, u in enumerate(ud)}


def grid_day(o, h, l, c, hm, g, tick_pen):
    fee = LI.FEE
    px0 = o[0] + tick_pen
    q = START_W / (px0 * (1 + fee))
    cash = 1.0 - START_W
    unit = STEP / (px0 * (1 + fee))
    last = o[0]
    fills = 1
    exit_k = int(np.searchsorted(hm, EXIT_HM, side="right")) - 1
    for k in range(1, max(exit_k, 1)):
        if l[k] <= last * (1 - g):
            while l[k] <= last * (1 - g):
                lvl = last * (1 - g)
                p = lvl + tick_pen
                if cash < unit * p * (1 + fee) - 1e-12:
                    break
                q += unit
                cash -= unit * p * (1 + fee)
                last = lvl
                fills += 1
        elif h[k] >= last * (1 + g):
            while h[k] >= last * (1 + g):
                lvl = last * (1 + g)
                if q < unit - 1e-12:
                    break
                p = lvl - tick_pen
                q -= unit
                cash += unit * p * (1 - fee)
                last = lvl
                fills += 1
    pe = o[min(exit_k + 1, len(o) - 1)] - tick_pen
    cash += q * pe * (1 - fee)
    fills += 1
    return cash - 1.0, fills


def hold_day(o, c, hm, tick_pen):
    fee = LI.FEE
    exit_k = int(np.searchsorted(hm, EXIT_HM, side="right")) - 1
    pe = o[min(exit_k + 1, len(o) - 1)] - tick_pen
    p0 = o[0] + tick_pen
    return (pe * (1 - fee)) / (p0 * (1 + fee)) - 1.0


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, pth in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), pth)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    d, _ = S1.load()
    d = d[(d["mod"] <= d["ce"]) & (d["volume"] > 0)].reset_index(drop=True)
    tab = day_table(d)
    res = {}
    groups = {k: g for k, g in d.groupby("day")}
    for pen_name, pen_ticks in (("limit_at_level", 0.0), ("one_tick_worse", 1.0)):
        rows = []
        for u, g in groups.items():
            if u < TEST_FROM or tab[u]["turb"] is None:
                continue
            o, h, l, c = (g[x].to_numpy(float) for x in ("open", "high", "low", "close"))
            hm = g["hm"].to_numpy()
            if len(o) < 60:
                continue
            tp = pen_ticks * LI.TICK
            row = dict(day=u, turb=tab[u]["turb"], hold_intraday=hold_day(o, c, hm, tp))
            for nm, gp, gs in SPACINGS:
                gg = gp if gp is not None else gs * tab[u]["sd_day"]
                if not np.isfinite(gg) or gg <= 0:
                    row[nm], row[nm + "_fills"] = np.nan, 0
                    continue
                pnl, nf = grid_day(o, h, l, c, hm, gg, tp)
                row[nm], row[nm + "_fills"] = pnl, nf
            rows.append(row)
        R = pd.DataFrame(rows)
        out = {}
        for reg_name, m in (("turbulent", R["turb"]), ("stable", ~R["turb"]), ("turbulent_2026", R["turb"] & (R["day"] >= "2026-01-01"))):
            sub = R[m]
            o2 = dict(days=int(len(sub)), hold_intraday_mean_bp=1e4 * float(sub["hold_intraday"].mean()), hold_intraday_compound_pct=100 * float(np.prod(1 + sub["hold_intraday"]) - 1))
            for nm, _, _ in SPACINGS:
                x = sub[nm].dropna()
                diff = (sub[nm] - sub["hold_intraday"]).dropna()
                o2[nm] = dict(mean_bp=1e4 * float(x.mean()), compound_pct=100 * float(np.prod(1 + x) - 1), win_days=float((x > 0).mean()), mean_fills=float(sub[nm + "_fills"].mean()), minus_hold_mean_bp=1e4 * float(diff.mean()), minus_hold_t=float(diff.mean() / (diff.std() / math.sqrt(len(diff)))) if len(diff) > 2 else None, beat_hold_days=float((diff > 0).mean()))
            out[reg_name] = o2
            print(pen_name, reg_name, "days", o2["days"], "hold intraday mean", round(o2["hold_intraday_mean_bp"], 1), "bp compound", round(o2["hold_intraday_compound_pct"], 1), "%", flush=True)
            for nm, _, _ in SPACINGS:
                z = o2[nm]
                print(f"    {nm:9s} mean {z['mean_bp']:+7.1f}bp/day compound {z['compound_pct']:+7.1f}% win {z['win_days']:.2f} fills/day {z['mean_fills']:5.1f} vs hold {z['minus_hold_mean_bp']:+6.1f}bp/day (t {z['minus_hold_t']:+.2f}) beat-hold days {z['beat_hold_days']:.2f}", flush=True)
        res[pen_name] = out
    rec = dict(version="grid-v1", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], results=res)
    with open(out_path, "x", encoding="utf-8") as fo:
        json.dump(rec, fo, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
