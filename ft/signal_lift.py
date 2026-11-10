import json
import os
import sys
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import numpy as np
import regime_compare as RC

OUT = os.path.join(os.path.dirname(HERE), "ft", "results", "signal_lift_v1.json")


def fwd_dd(px, k, n):
    seg = px[k + 1:k + 1 + n]
    if len(seg) < n:
        return None
    return float(seg.min() / px[k] - 1)


def main():
    spx = RC.yf("IDX_GSPC")
    ks = RC.yf("IDX_KS11")
    vix = RC.Series(RC.yf("IDX_VIX"))
    vix3 = RC.Series(RC.fred("VXVCLS"))
    baa = RC.Series(RC.fred("BAA10Y"))
    curve = RC.Series(RC.fred("T10Y2Y"))
    krw = RC.Series(RC.yf("KRW_X"))
    days = sorted(d for d in spx if d >= "1990-01-02")
    px = np.array([spx[d] for d in days])
    ks_s = RC.Series(ks)
    elect = RC.election_days(1990, 2030)
    sig = {k: np.zeros(len(days), bool) for k in ("vts_onset", "vts_state", "vix_jump25", "vix_calm13", "credit_accel", "credit_rise60", "curve_uninvert", "krw_spike2", "trend50_break", "trend200_break", "near_high_ks_weak", "pre_election", "dd_from_high5")}
    prev_ratio, prev_above50, prev_above200 = None, None, None
    for k, D in enumerate(days):
        _, v = vix.upto(D)
        _, v_prev = vix.upto(D, lag=1)
        _, v3 = vix3.upto(D, lag=1)
        ratio = v / v3 if (v and v3) else None
        if ratio is not None:
            sig["vts_state"][k] = ratio > 1.0
            sig["vts_onset"][k] = ratio > 1.0 and (prev_ratio is not None and prev_ratio <= 1.0)
            prev_ratio = ratio
        if v and v_prev:
            sig["vix_jump25"][k] = v / v_prev - 1 > 0.25
            sig["vix_calm13"][k] = v < 13
        b = baa.window(D, 60, lag=1)
        if len(b) >= 11:
            sig["credit_accel"][k] = b[-1] - b[-11] > 0.15
            sig["credit_rise60"][k] = b[-1] - min(b) > 0.30
        cw = curve.window(D, 120, lag=1)
        if len(cw) >= 60:
            sig["curve_uninvert"][k] = cw[-1] > 0 and min(cw[-60:]) < 0
        kw = krw.window(D, 6)
        if len(kw) >= 6:
            sig["krw_spike2"][k] = kw[-1] / kw[0] - 1 > 0.02
        if k >= 200:
            ma50, ma200 = px[k - 49:k + 1].mean(), px[k - 199:k + 1].mean()
            a50, a200 = bool(px[k] > ma50), bool(px[k] > ma200)
            sig["trend50_break"][k] = (prev_above50 is True) and not a50
            sig["trend200_break"][k] = (prev_above200 is True) and not a200
            prev_above50, prev_above200 = a50, a200
            hi = px[max(0, k - 251):k + 1].max()
            sig["dd_from_high5"][k] = px[k] / hi - 1 < -0.05
            kw2 = ks_s.window(D, 50)
            if len(kw2) >= 50:
                sig["near_high_ks_weak"][k] = px[k] / hi - 1 > -0.03 and kw2[-1] < sum(kw2) / 50
        for e in elect:
            if (date.fromisoformat(e) - timedelta(days=14)).isoformat() <= D < e:
                sig["pre_election"][k] = True
    out = {}
    for n, thr in ((10, -0.05), (20, -0.05), (20, -0.10)):
        y = np.array([(lambda x: np.nan if x is None else float(x <= thr))(fwd_dd(px, k, n)) for k in range(len(days))])
        fr = np.array([np.nan if k + n >= len(px) else px[k + n] / px[k] - 1 for k in range(len(days))])
        key = f"fwd{n}_dd{int(-100 * thr)}"
        out[key] = {}
        for part, lo, hi in (("train 1990-2016", "1990-01-01", "2016-12-31"), ("test 2017-2026", "2017-01-01", "2026-12-31")):
            msk = np.array([lo <= d <= hi for d in days]) & np.isfinite(y)
            base = float(np.nanmean(y[msk]))
            out[key][part] = dict(base=base, signals={})
            for s, arr in sig.items():
                mm = msk & arr
                cnt = int(mm.sum())
                if cnt < 5:
                    out[key][part]["signals"][s] = dict(days=cnt)
                    continue
                p = float(y[mm].mean())
                out[key][part]["signals"][s] = dict(days=cnt, p=p, lift=p / base if base else None, fwd_ret=float(np.nanmean(fr[mm])), base_fwd_ret=float(np.nanmean(fr[msk])))
    hz = (1, 2, 3, 5, 10, 20)
    lr = np.r_[np.diff(np.log(px)), 0.0]
    ev = {}
    for part, lo, hi in (("train", "1990-01-01", "2016-12-31"), ("test", "2017-01-01", "2026-12-31")):
        msk = np.array([lo <= d <= hi for d in days])
        idx = np.where(msk)[0]
        base = {h: float(np.mean([lr[k + 1:k + 1 + h].sum() for k in idx if k + h < len(px)])) for h in hz}
        ev[part] = dict(base=base)
        for s_, arr in sig.items():
            ks_ = [k for k in idx if arr[k] and k + 20 < len(px)]
            if len(ks_) < 5:
                continue
            ev[part][s_] = {h: float(np.mean([lr[k + 1:k + 1 + h].sum() for k in ks_])) for h in hz}
            ev[part][s_]["n"] = len(ks_)
    out["event_study"] = ev
    print("== cumulative log return after signal, by day (train | test), base first")
    for part in ("train", "test"):
        e = ev[part]
        print(f" [{part}] base " + " ".join(f"d{h}:{100 * e['base'][h]:+.2f}" for h in hz))
        for s_ in sig:
            if s_ in e:
                print(f"   {s_:18s} n{e[s_]['n']:5d} " + " ".join(f"d{h}:{100 * (e[s_][h] - e['base'][h]):+.2f}" for h in hz))
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for key, parts in out.items():
        if key == "event_study":
            continue
        print("==", key)
        tr, te = parts["train 1990-2016"], parts["test 2017-2026"]
        print(f"   base: train {100 * tr['base']:.1f}%  test {100 * te['base']:.1f}%")
        for s in sig:
            a, b = tr["signals"][s], te["signals"][s]
            fa = f"{a['days']:5d}d p {100 * a['p']:5.1f}% lift {a['lift']:4.2f} fwd {100 * a['fwd_ret']:+5.2f}%" if "p" in a else f"{a['days']:5d}d"
            fb = f"{b['days']:5d}d p {100 * b['p']:5.1f}% lift {b['lift']:4.2f} fwd {100 * b['fwd_ret']:+5.2f}%" if "p" in b else f"{b['days']:5d}d"
            print(f"   {s:18s} train {fa} | test {fb}")


if __name__ == "__main__":
    main()
