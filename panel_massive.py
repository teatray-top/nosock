import glob
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.csv as pcsv

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "massive", "flat")
ET = "America/New_York"
OFF = 570
BARS = {"5m": 78, "1m": 390}
STEP = {"5m": 5, "1m": 1}
PLAIN = re.compile(r"^[A-Z]{1,5}$")


def read(path, cols):
    return pcsv.read_csv(path, convert_options=pcsv.ConvertOptions(include_columns=cols)).to_pandas()


def day_files(d0, d1):
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "day", "*.csv.gz"))):
        d = os.path.basename(f)[:10]
        if d0 <= d <= d1:
            x = read(f, ["ticker", "close", "volume", "high", "low"])
            out[d] = x.drop_duplicates("ticker").set_index("ticker")
    return out


def cs_tickers():
    path = os.path.join(os.path.dirname(ROOT), "tickers_cs.json")
    if os.path.exists(path):
        return set(json.load(open(path)))
    import massive
    c = massive.Client(rpm=100)
    out = set()
    for active in ("true", "false"):
        for j in c.pages("/v3/reference/tickers", dict(market="stocks", type="CS", active=active, limit=1000)):
            out |= {r["ticker"] for r in j.get("results", [])}
    json.dump(sorted(out), open(path, "w"))
    return out


def universe(dayf, pre, n, min_dv, min_px, max_px, min_rng):
    cs = cs_tickers()
    dv = pd.concat([dayf[d]["close"] * dayf[d]["volume"] for d in pre], axis=1)
    px = pd.concat([dayf[d]["close"] for d in pre], axis=1)
    rng = pd.concat([(dayf[d]["high"] - dayf[d]["low"]) / dayf[d]["close"] for d in pre], axis=1)
    ok = (dv.notna().sum(1) >= 0.8 * len(pre)) & (dv.median(1) >= min_dv) & (px.median(1) >= min_px) & (px.median(1) <= max_px) & (rng.median(1) >= min_rng)
    ok &= pd.Series([bool(PLAIN.match(t)) and t in cs for t in dv.index], index=dv.index)
    r = dv[ok].median(1).sort_values(ascending=False)
    return list(r.index[:n]), dict(common_stocks=len(cs), candidates=int(ok.sum()), dv_cut=float(r.iloc[min(n, len(r)) - 1]) if len(r) else None)


def build(start, end, out, base="5m", n=1000, pre_n=60, min_dv=2e6, min_px=1.0, max_px=500.0, min_rng=0.025):
    t0 = time.time()
    mfiles = {os.path.basename(f)[:10]: f for f in glob.glob(os.path.join(ROOT, "minute", "*.csv.gz"))}
    dayf = day_files("1900-01-01", end)
    alld = sorted(dayf)
    pre = [d for d in alld if d < start][-pre_n:]
    if len(pre) < pre_n:
        sys.exit(f"only {len(pre)} pre-window days before {start}, need {pre_n}")
    syms, info = universe(dayf, pre, n, min_dv, min_px, max_px, min_rng)
    jx = {s: j for j, s in enumerate(syms)}
    days = [d for d in alld if start <= d <= end and d in mfiles]
    B, step, N = BARS[base], STEP[base], len(syms)
    O, H, L, C, V = [np.full((len(days), B, N), np.nan) for _ in range(5)]
    keep = []
    for i, d in enumerate(days):
        x = read(mfiles[d], ["ticker", "volume", "open", "close", "high", "low", "window_start"])
        x = x[x["ticker"].isin(jx)]
        ts = pd.to_datetime(x["window_start"].values, utc=True).tz_convert(ET)
        m = (ts.hour * 60 + ts.minute).values - OFF
        b = m // step
        x = x.assign(b=b)[(b >= 0) & (b < B)].sort_values(["ticker", "window_start"])
        g = x.groupby(["ticker", "b"], sort=False).agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"), v=("volume", "sum")).reset_index()
        jj = g["ticker"].map(jx).values
        bb = g["b"].values
        O[i, bb, jj], H[i, bb, jj], L[i, bb, jj], C[i, bb, jj], V[i, bb, jj] = g["o"].values, g["h"].values, g["l"].values, g["c"].values, g["v"].values
        late = np.mean(~np.isnan(C[i, B - 1]))
        keep.append(late >= 0.3)
        if not keep[-1]:
            print(f"{d}: only {late:.0%} of universe has a last bar, dropped (early close)", flush=True)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(days)} {d} {time.time() - t0:.0f}s", flush=True)
    keep = np.array(keep)
    days = [d for d, k in zip(days, keep) if k]
    O, H, L, C, V = O[keep], H[keep], L[keep], C[keep], V[keep]
    Dn = len(days)
    V = np.nan_to_num(V)
    bar0 = np.arange(B)[None, :, None] == 0
    has = ~np.isnan(C) & ((V > 0) | (bar0 & ~np.isnan(O)))
    Cf = C.copy()
    for d in range(Dn):
        Cf[d] = pd.DataFrame(C[d]).ffill().values
    last = Cf[:, -1, :]
    oc1 = np.full((Dn, N), np.nan)
    dv1 = np.full((Dn, N), np.nan)
    for i, d in enumerate(days):
        f = dayf[d].reindex(syms)
        oc1[i] = f["close"].values
        dv1[i] = (f["close"] * f["volume"]).values
    with np.errstate(divide="ignore", invalid="ignore"):
        bad = np.isnan(oc1) | (last <= 0) | (np.abs(np.log(oc1 / last)) > 0.15)
    oc = np.where(bad, last, oc1)
    turn = np.where(bad | np.isnan(dv1), np.nansum(np.nan_to_num(C) * V, 1), dv1)
    pc = np.full((Dn, N), np.nan)
    pc[1:] = oc[:-1]
    pc[0] = dayf[pre[-1]].reindex(syms)["close"].values
    both = ~np.isnan(oc1) & (last > 0)
    gap = float(np.nanmedian(np.abs(np.log(oc1[both] / last[both]))))
    print(f"panel {out}: {Dn} days x {B} bars x {N} syms ({days[0]}..{days[-1]}), universe from {pre[0]}..{pre[-1]} common stocks {info['common_stocks']} candidates {info['candidates']} dollar-volume cut {info['dv_cut']:.0f}, median |log(official close / last bar close)| {gap:.5f}, daily closes rejected {int((~np.isnan(oc1) & bad).sum())}, {time.time() - t0:.0f}s", flush=True)
    np.savez(out, O=O, H=H, L=L, C=Cf, V=V, has=has, pc=pc, oc=oc, turn=turn, off=np.full(Dn, OFF), days=np.array(days), syms=np.array(syms))
    json.dump(dict(start=start, end=end, base=base, n=n, pre=[pre[0], pre[-1]], min_dv=min_dv, min_px=min_px, max_px=max_px, min_rng=min_rng, syms=syms, days=days, info=info), open(out[:-4] + ".json", "w"), indent=0)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], sys.argv[3], base=sys.argv[4] if len(sys.argv) > 4 else "5m", n=int(sys.argv[5]) if len(sys.argv) > 5 else 1000)
