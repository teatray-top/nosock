import bisect
import csv
import json
import math
import os
import sys
import time
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import numpy as np
import ms_alloc_v3 as A3
from nosock import engine as E
from nosock import run_backtest as RB

DEFENSIVE = {"114800", "148070", "114260", "153130", "TLT", "IEF", "SHY", "GLD", "SH", "UUP"}
GLOBAL = os.path.join(ROOT, "data", "global", "yf_20261107_161526")
FRED = os.path.join(ROOT, "data", "macro", "fred_20261109")
OUT = os.path.join(ROOT, "ft", "results", "regime_compare_v1.json")

WINDOWS = [
    ("서브프라임", "2007-10-09", "2009-03-09"),
    ("미국 신용등급 강등", "2011-07-22", "2011-10-03"),
    ("중국 위안화 쇼크", "2015-08-10", "2016-02-11"),
    ("2018 중간선거 뒤 급락", "2018-11-06", "2018-12-24"),
    ("2018 4분기", "2018-09-20", "2018-12-24"),
    ("코로나", "2020-02-19", "2020-03-23"),
    ("우크라이나 침공", "2022-02-09", "2022-03-08"),
    ("2022 하락장", "2022-01-03", "2022-10-12"),
    ("엔캐리 청산", "2024-07-16", "2024-08-05"),
    ("계엄", "2024-12-02", "2024-12-09"),
    ("관세 쇼크", "2025-04-02", "2025-04-08"),
]


def yf(name):
    d = json.load(open(os.path.join(GLOBAL, name + ".json"), encoding="utf-8"))
    return {k: float(v) for k, v in zip(d["days"], d["C"]) if v is not None}


def fred(name):
    out = {}
    with open(os.path.join(FRED, name + ".csv"), encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0][:1] in "12":
                try:
                    out[row[0]] = float(row[1])
                except ValueError:
                    pass
    return out


class Series:
    def __init__(self, d):
        self.days = sorted(d)
        self.v = [d[k] for k in self.days]

    def upto(self, D, lag=0):
        i = bisect.bisect_right(self.days, D) - 1 - lag
        return (self.days[i], self.v[i]) if i >= 0 else (None, None)

    def window(self, D, n, lag=0):
        i = bisect.bisect_right(self.days, D) - lag
        return self.v[max(0, i - n):i]


def election_days(y0, y1):
    out = []
    for y in range(y0, y1 + 1):
        if y % 2:
            continue
        d = date(y, 11, 1)
        while d.weekday() != 0:
            d += timedelta(days=1)
        out.append((d + timedelta(days=1)).isoformat())
    return out


def build_dials(days):
    spx, vix, vix3, baa = Series(yf("IDX_GSPC")), Series(yf("IDX_VIX")), Series(fred("VXVCLS")), Series(fred("BAA10Y"))
    elect = election_days(2000, 2030)
    out = {k: {} for k in ("trend", "vts", "credit", "event", "combo", "pre")}
    st = dict(trend=1.0, vts=1.0, credit=1.0)
    for D in days:
        w = spx.window(D, 200)
        if len(w) >= 200:
            ma = sum(w) / 200
            px = w[-1]
            if st["trend"] == 1.0 and px < ma * 0.98:
                st["trend"] = 0.0
            elif st["trend"] == 0.0 and px > ma * 1.01:
                st["trend"] = 1.0
        _, v = vix.upto(D)
        _, v3 = vix3.upto(D, lag=1)
        if v and v3:
            r = v / v3
            if r > 1.10:
                st["vts"] = 0.0
            elif r > 1.00 and st["vts"] == 1.0:
                st["vts"] = 0.5
            elif r < 0.92:
                st["vts"] = 1.0
        b = baa.window(D, 60, lag=1)
        if len(b) >= 20:
            rise = b[-1] - min(b)
            if rise > 0.45:
                st["credit"] = 0.0
            elif rise > 0.30 and st["credit"] == 1.0:
                st["credit"] = 0.5
            elif rise < 0.15:
                st["credit"] = 1.0
        ev = 1.0
        for e in elect:
            lo = (date.fromisoformat(e) - timedelta(days=14)).isoformat()
            hi = (date.fromisoformat(e) + timedelta(days=21)).isoformat()
            if lo <= D <= hi:
                ev = 0.5
        out["trend"][D] = st["trend"]
        out["vts"][D] = st["vts"]
        out["credit"][D] = st["credit"]
        out["event"][D] = ev
        out["pre"][D] = min(st["vts"], st["credit"], ev)
        out["combo"][D] = min(st["trend"], st["vts"], st["credit"], ev)
    return out


def dial_weights(tw, s, safe):
    if s >= 1.0:
        return dict(tw)
    out, freed = {}, 0.0
    for c, w in tw.items():
        if c in DEFENSIVE:
            out[c] = out.get(c, 0.0) + w
        else:
            out[c] = out.get(c, 0.0) + w * s
            freed += w * (1 - s)
    if safe and freed > 0:
        for c in safe:
            out[c] = out.get(c, 0.0) + freed / len(safe)
    return out


def run_x(m, strategy, start, end, decide="h20", band=None, dial=None, safe=None, deposit=E.DEPOSIT, intraday=None, xform=None):
    cash, vault, dep = 0.0, 0.0, 0.0
    units, shares, basis = {}, {}, {}
    pending, marks, n_trades, cost_paid = [], [], 0, 0.0
    span = sorted(d for d in m.days if start <= d <= end)
    krs = sorted(m.kr_days)
    cal = set()
    for d in span:
        if d in m.kr_days:
            cal.add(d)
            continue
        j = bisect.bisect_left(krs, d)
        near = [krs[k] for k in (j - 1, j) if 0 <= k < len(krs)]
        if not any(abs((date.fromisoformat(x) - date.fromisoformat(d)).days) <= 10 for x in near):
            cal.add(d)
    kdays = sorted(cal)
    dset = set(kdays[::E.H]) if decide == "h20" else set(kdays)
    tw_last = None
    for D in sorted(d for d in m.days if start <= d <= end):
        todo = sorted((p for p in pending if p[2] == D), key=lambda p: p[0])
        pending = [p for p in pending if p[2] != D]
        for phase in ("sell", "buy"):
            for (c, val, _) in todo:
                ins = m.inst[c]
                i = m.ipos[c][D]
                o = ins["O"][i]
                cur = units.get(c, 0.0)
                if ins["market"] == "KR":
                    px = o * m.won_ratio(c, i)
                    n_cur = shares.get(c, 0)
                    n_tgt = int(math.floor(val / px + 0.5)) if val > 0 else 0
                    dn = n_tgt - n_cur
                    if dn == 0 or (dn < 0) != (phase == "sell"):
                        continue
                    rate = ins["cost"](i, px)
                    if dn > 0:
                        dn = min(dn, int(math.floor(max(cash, 0.0) / (px * (1 + rate)))))
                        if dn <= 0:
                            continue
                        pay = dn * px * (1 + rate)
                        cash -= pay
                        units[c] = cur + dn * px / o
                        shares[c] = n_cur + dn
                        basis[c] = basis.get(c, 0.0) + pay
                    else:
                        q = -dn
                        frac = q / n_cur
                        cash += q * px * (1 - rate)
                        units[c] = cur * (1 - frac)
                        shares[c] = n_cur - q
                        basis[c] = basis.get(c, 0.0) * (1 - frac)
                    n_trades += 1
                    cost_paid += abs(dn) * px * rate
                else:
                    nu = val / o
                    du = nu - cur
                    if abs(du) * o < 1.0 * m.inst[c]["fxv"][i] and nu > 1e-12:
                        continue
                    if abs(du) < 1e-12 or (du < 0) != (phase == "sell"):
                        continue
                    rate = ins["cost"](i, o)
                    if du > 0:
                        amt = min(du * o * (1 + rate), max(cash, 0.0))
                        if amt < 1.0 * m.inst[c]["fxv"][i]:
                            continue
                        du = amt / (o * (1 + rate))
                        cash -= amt
                        units[c] = cur + du
                        basis[c] = basis.get(c, 0.0) + amt
                    else:
                        q = -du
                        cash += q * o * (1 - rate)
                        units[c] = cur - q
                        basis[c] = basis.get(c, 0.0) * (1 - q / cur)
                    n_trades += 1
                    cost_paid += abs(du) * o * rate
        cool = False
        if intraday is not None and D in intraday["pos"]:
            MD, thr = intraday, intraday["thr"]
            di = MD["pos"][D]
            k_spy = MD["tk"]["SPY"]
            prev = MD["prev"].get(D)
            path = MD["C"][di, k_spy]
            hit = np.where(np.isfinite(path) & (path <= prev * (1 - thr)))[0] if prev else []
            if len(hit) and hit[0] + 1 < 390:
                t = int(hit[0]) + 1
                for c in [c for c, u in units.items() if u > 1e-12 and m.inst[c]["market"] == "US" and c not in DEFENSIVE]:
                    k = MD["tk"].get(c)
                    i = m.ipos[c].get(D)
                    if k is None or i is None:
                        continue
                    pu = MD["O"][di, k, t]
                    if not np.isfinite(pu):
                        pu = MD["C"][di, k, t - 1]
                    row = MD["O"][di, k]
                    f0 = row[np.isfinite(row)]
                    if not np.isfinite(pu) or not len(f0):
                        continue
                    px = m.inst[c]["O"][i] * float(pu) / float(f0[0])
                    rate = m.inst[c]["cost"](i, px)
                    cash += units[c] * px * (1 - rate)
                    cost_paid += units[c] * px * rate
                    n_trades += 1
                    units[c] = 0.0
                pending = [p for p in pending if not (m.inst[p[0]]["market"] == "US" and p[0] not in DEFENSIVE)]
                cool = True
                intraday["hits"].append(D)
        if D in cal:
            cash += deposit
            dep += deposit
            val = {c: u * m.close_at(c, D)[0] for c, u in units.items() if u}
            eq = cash + sum(val.values())
            if D in dset or tw_last is None:
                tw_last, _ = strategy.decide(D, 1)
            s = dial[D] if dial is not None else 1.0
            te = dial_weights(tw_last, s, safe)
            if xform is not None:
                te = xform(D, te)
            if cool or (intraday is not None and intraday.get("cool_kr_day") == D):
                te = {c: (w if (m.inst[c]["market"] != "US" or c in DEFENSIVE) else 0.0) for c, w in te.items()}
            if not pending:
                full = D in dset and band is None
                for c in sorted(set(units) | set(te)):
                    o, i, d_exec = m.next_open(c, D)
                    cd = m.close_at(c, D)[0]
                    if o is None or cd is None or d_exec > end or (date.fromisoformat(d_exec) - date.fromisoformat(D)).days > 10:
                        continue
                    tgt = te.get(c, 0.0) * eq
                    cur = val.get(c, 0.0)
                    if full or (band is not None and (cur - tgt > band * eq or tgt > cur)):
                        pending.append((c, tgt * (o / cd), d_exec))
        eqd = cash + sum(u * m.close_at(c, D)[0] for c, u in units.items() if u)
        marks.append((D, eqd + vault, dep))
    return marks, dict(trades=n_trades, cost=cost_paid, deposits=dep, total=marks[-1][1] if marks else 0.0)


def twr(marks):
    idx, acc = [], 1.0
    for k, (D, tot, dep) in enumerate(marks):
        if k:
            flow = dep - marks[k - 1][2]
            base = marks[k - 1][1] + flow
            if base > 0:
                acc *= tot / base
        idx.append((D, acc))
    return idx


def stats(marks, info):
    ix = twr(marks)
    days = [d for d, _ in ix]
    v = np.array([x for _, x in ix])
    peak = np.maximum.accumulate(v)
    ddv = v / peak - 1
    mdd = float(ddv.min())
    k_low = int(ddv.argmin())
    k_peak = int(np.argmax(v[:k_low + 1])) if k_low > 0 else 0
    yrs = max(1e-9, (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25)
    cagr = float(v[-1] ** (1 / yrs) - 1)
    win = {}
    for name, a, b in WINDOWS:
        i0 = bisect.bisect_left(days, a)
        i1 = bisect.bisect_right(days, b) - 1
        if i0 >= len(days) or i1 <= i0:
            continue
        seg = v[i0:i1 + 1]
        win[name] = dict(ret=float(seg[-1] / seg[0] - 1), mdd=float((seg / np.maximum.accumulate(seg) - 1).min()))
    w20 = float(min(v[k + 20] / v[k] - 1 for k in range(len(v) - 20))) if len(v) > 21 else None
    return dict(cagr=cagr, mdd=mdd, mdd_from=days[k_peak], mdd_to=days[k_low], worst20=w20, total_twr=float(v[-1] - 1), profit=info["total"] - info["deposits"], trades=info["trades"], cost=info["cost"], windows=win)


def load_minute(thr):
    z = np.load(os.path.join(ROOT, "data", "massive", "etf_minute_v1.npz"))
    days = [str(d) for d in z["days"]]
    tk = {str(t): k for k, t in enumerate(z["tickers"])}
    C, O = z["C"], z["O"]
    prev, last, k = {}, None, tk["SPY"]
    for di, d in enumerate(days):
        if last is not None:
            prev[d] = last
        f = C[di, k][np.isfinite(C[di, k])]
        if len(f):
            last = float(f[-1])
    return dict(pos={d: i for i, d in enumerate(days)}, tk=tk, C=C, O=O, prev=prev, thr=thr, hits=[])


def main():
    t0 = time.time()
    inst, groups, names, classes = RB.load_market()
    codes = groups["U29"]
    m = E.Market(inst, codes, names)
    m.raw = A3.load_raw()
    model = RB.model_strategy(inst, codes, m)
    equal = RB.strategy_for(inst, codes, "equal", m)
    start, end = "2007-02-01", max(m.days)
    dials = build_dials(m.days)
    safe_kr_us = ["153130", "SHY"]
    runs = [
        ("E0 균등, 20일", equal, dict(decide="h20")),
        ("E3 균등 + 사전 신호(변동성 구조·신용·선거)", equal, dict(decide="daily", band=0.02, dial=dials["pre"], safe=safe_kr_us)),
        ("E3c 균등 + 사전 + 추세", equal, dict(decide="daily", band=0.02, dial=dials["combo"], safe=safe_kr_us)),
        ("R0 모델, 20일", model, dict(decide="h20")),
        ("R1 모델 20일 + 2%p 넘으면 매일 매도", model, dict(decide="h20", band=0.02)),
        ("R2 모델 매일 재계산 + 2%p", model, dict(decide="daily", band=0.02)),
        ("R3a R2 + 추세(200일선)", model, dict(decide="daily", band=0.02, dial=dials["trend"], safe=safe_kr_us)),
        ("R3b R2 + 변동성 구조(VIX/VIX3M)", model, dict(decide="daily", band=0.02, dial=dials["vts"], safe=safe_kr_us)),
        ("R3c R2 + 신용 스프레드", model, dict(decide="daily", band=0.02, dial=dials["credit"], safe=safe_kr_us)),
        ("R3d R2 + 선거 전후", model, dict(decide="daily", band=0.02, dial=dials["event"], safe=safe_kr_us)),
        ("R3e R2 + 사전 신호 합(b·c·d)", model, dict(decide="daily", band=0.02, dial=dials["pre"], safe=safe_kr_us)),
        ("R3f R2 + 전부(추세 포함)", model, dict(decide="daily", band=0.02, dial=dials["combo"], safe=safe_kr_us)),
    ]
    only = set(sys.argv[1:])
    if "B" in only:
        b0 = "2021-11-05"
        runs = [
            ("S1 모델 20일 + 2%p 매도", model, dict(decide="h20", band=0.02), None),
            ("S3e 모델 매일 + 사전 신호", model, dict(decide="daily", band=0.02, dial=dials["pre"], safe=safe_kr_us), None),
            ("S3f 모델 매일 + 사전 + 추세", model, dict(decide="daily", band=0.02, dial=dials["combo"], safe=safe_kr_us), None),
            ("S4a S1 + 장중 SPY −2%면 즉시 매도", model, dict(decide="h20", band=0.02), 0.02),
            ("S4b S1 + 장중 SPY −3%면 즉시 매도", model, dict(decide="h20", band=0.02), 0.03),
            ("S4c S3e + 장중 SPY −2%", model, dict(decide="daily", band=0.02, dial=dials["pre"], safe=safe_kr_us), 0.02),
            ("S4d S3f + 장중 SPY −2%", model, dict(decide="daily", band=0.02, dial=dials["combo"], safe=safe_kr_us), 0.02),
        ]
        res = {}
        for name, strat, kw, thr in runs:
            t1 = time.time()
            intr = load_minute(thr) if thr else None
            marks, info = run_x(m, strat, b0, end, intraday=intr, **kw)
            res[name] = stats(marks, info)
            r = res[name]
            if intr:
                r["intraday_hits"] = intr["hits"]
            print(f"{name}: CAGR {100 * r['cagr']:+.2f}% MDD {100 * r['mdd']:.1f}% ({r['mdd_from']}~{r['mdd_to']}) worst20 {100 * r['worst20']:.1f}% trades {r['trades']} cost {r['cost']:,.0f} hits {len(intr['hits']) if intr else 0} ({time.time() - t1:.0f}s)", flush=True)
        json.dump(dict(start=b0, end=end, windows=WINDOWS, runs=res), open(OUT.replace("_v1", "_intraday_v1"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("saved B")
        return
    res = {}
    for name, strat, kw in runs:
        if only and name.split()[0] not in only:
            continue
        t1 = time.time()
        marks, info = run_x(m, strat, start, end, **kw)
        res[name] = stats(marks, info)
        r = res[name]
        print(f"{name}: CAGR {100 * r['cagr']:+.2f}% MDD {100 * r['mdd']:.1f}% ({r['mdd_from']}~{r['mdd_to']}) worst20 {100 * r['worst20']:.1f}% trades {r['trades']} cost {r['cost']:,.0f} ({time.time() - t1:.0f}s)", flush=True)
    off = {k: {D: s for D, s in v.items() if s < 1.0} for k, v in dials.items()}
    share = {k: round(len(v) / max(1, len(dials[k])), 3) for k, v in off.items()}
    json.dump(dict(start=start, end=end, windows=WINDOWS, runs=res, dial_off_share=share, secs=round(time.time() - t0)), open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", OUT, "dial off share", share)


if __name__ == "__main__":
    main()
