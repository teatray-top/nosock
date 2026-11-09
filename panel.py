import glob
import os

import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STEP = {"5m": 5, "60m": 60}
NBARS = {"kr": {"5m": 72, "60m": 6}, "us": {"5m": 78, "60m": 7}}
TZ = {"kr": ("Asia/Seoul", 540), "us": ("America/New_York", 570)}
DAILY = {"": "1d", "all": os.path.join("all", "1d"), "archive": os.path.join("all", "1d"), "us": os.path.join("us", "1d"), "usall": os.path.join("us", "1d"), "usm": os.path.join("us", "1d")}
VER = 4


def split(iv):
    for pre in ("archive", "all", "usall", "usm", "us"):
        if iv.startswith(pre):
            return pre, iv[len(pre):]
    return "", iv


def daily(sub, syms, days, tz):
    root = os.path.join(D, DAILY[sub])
    dix = {d: i for i, d in enumerate(days)}
    oc = np.full((len(days), len(syms)), np.nan)
    dv = np.full((len(days), len(syms)), np.nan)
    n = 0
    for j, s in enumerate(syms):
        f = os.path.join(root, s + ".csv")
        if not os.path.exists(f):
            continue
        x = pd.read_csv(f, index_col=0)
        idx = pd.to_datetime(x.index)
        if idx.tz is not None:
            idx = idx.tz_convert(tz)
        idx = idx.tz_localize(None).normalize()
        x = x[~idx.duplicated()]
        idx = idx[~idx.duplicated()]
        di = np.array([dix.get(d.date(), -1) for d in idx])
        ok = di >= 0
        oc[di[ok], j] = x["Close"].values[ok]
        dv[di[ok], j] = (x["Close"].values * x["Volume"].values)[ok]
        n += 1
    return oc, dv, n


def build(iv="5m"):
    sub, base = split(iv)
    mk = "us" if sub.startswith("us") else "kr"
    BARS, step = NBARS[mk][base], STEP[base]
    tz, off0 = TZ[mk]
    files = sorted(glob.glob(os.path.join(D, sub, base, "*.csv")))
    frames, syms = [], []
    for f in files:
        x = pd.read_csv(f, index_col=0)
        x.index = pd.to_datetime(x.index, utc=True).tz_convert(tz)
        x = x[~x.index.duplicated()]
        if len(x) < 5 * BARS:
            continue
        frames.append(x)
        syms.append(os.path.basename(f)[:-4])
    days = sorted(set().union(*[set(x.index.tz_localize(None).date) for x in frames]))
    Dn, N = len(days), len(frames)
    dix = {d: i for i, d in enumerate(days)}
    first = {}
    for x in frames:
        m = pd.Series((x.index.hour * 60 + x.index.minute).values, index=x.index.tz_localize(None).date)
        for d, v in m.groupby(level=0).min().items():
            first.setdefault(d, []).append(v)
    off = np.array([600 if mk == "kr" and np.median(first[d]) >= 600 else off0 for d in days])
    O, H, L, C, V = [np.full((Dn, BARS, N), np.nan) for _ in range(5)]
    for j, x in enumerate(frames):
        di = np.array([dix[d] for d in x.index.tz_localize(None).date])
        m = (x.index.hour * 60 + x.index.minute).values - off[di]
        b = m // step
        ok = (b >= 0) & (b < BARS)
        for arr, col in ((O, "Open"), (H, "High"), (L, "Low"), (C, "Close"), (V, "Volume")):
            arr[di[ok], b[ok], j] = x[col].values[ok]
    V = np.nan_to_num(V)
    bar0 = np.arange(BARS)[None, :, None] == 0
    has = ~np.isnan(C) & ((V > 0) | (bar0 & ~np.isnan(O)))
    Cf = C.copy()
    for d in range(Dn):
        Cf[d] = pd.DataFrame(C[d]).ffill().values
    last = Cf[:, -1, :]
    oc1, dv1, nd = daily(sub, syms, days, tz)
    with np.errstate(divide="ignore", invalid="ignore"):
        bad = np.isnan(oc1) | (last <= 0) | (np.abs(np.log(oc1 / last)) > 0.15)
    oc = np.where(bad, last, oc1)
    turn = np.where(bad | np.isnan(dv1), np.nansum(np.nan_to_num(C) * V, 1), dv1)
    nbad = int((~np.isnan(oc1) & bad).sum())
    pc = np.full((Dn, N), np.nan)
    pc[1:] = oc[:-1]
    both = ~np.isnan(oc1) & (last > 0)
    gap = np.nanmedian(np.abs(np.log(oc1[both] / last[both]))) if both.any() else float("nan")
    print(f"panel {iv}: {Dn} days x {BARS} bars x {N} syms, daily files {nd}/{N}, late-open days {int((off != off0).sum())}, median |log(official close / last bar close)| {gap:.5f}, daily closes rejected (>15% off intraday) {nbad}", flush=True)
    return dict(O=O, H=H, L=L, C=Cf, V=V, has=has, pc=pc, oc=oc, turn=turn, off=off, days=np.array([str(d) for d in days]), syms=np.array(syms))


def load(iv="5m"):
    sub, base = split(iv)
    if os.environ.get("PANEL_PATH"):
        z = np.load(os.environ["PANEL_PATH"], allow_pickle=True)
        return {k: z[k] for k in z.files}
    cache = os.path.join(D, f"panel{iv}_v{VER}.npz")
    srcs = glob.glob(os.path.join(D, sub, base, "*.csv")) + glob.glob(os.path.join(D, DAILY[sub], "*.csv"))
    newest = max((os.path.getmtime(f) for f in srcs), default=0)
    if os.path.exists(cache) and os.path.getmtime(cache) > newest:
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    p = build(iv)
    np.savez(cache, **p)
    return p


if __name__ == "__main__":
    import sys
    p = load(sys.argv[1] if len(sys.argv) > 1 else "5m")
    print(p["C"].shape, p["days"][0], p["days"][-1])
    print("bars with trades per day-stock", p["has"].sum(1).mean())
    print("has by bar", np.round(p["has"].mean((0, 2)), 2))
    print("median daily turnover", np.nanmedian(p["turn"][p["turn"] > 0]))
