import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ETF = os.path.join(HERE, "data", "panel_etf_v3.npz")
KR = os.path.join(HERE, "data", "kr_etf", "kodex_daily_adj.json")
KR_RAW = os.path.join(HERE, "data", "kr_etf", "kodex_daily_raw.json")
OUT = os.environ.get("DIP_OUT", os.path.join(HERE, "runs", "dip_entry_2026-11-06.json"))
KR_LAST_COMPLETE = "2026-11-05"
H_LIST = (20, 60, 120, 250)
GAP = 20
LOOK = 60
DCA_N = 20
NBOOT = 10000
ERA = "2015-01-01"
US = dict(fee=0.001, sec=2.06e-5, min_fee=0.01, tick=0.01, sizes=(50.0, 7.0), frac=True, unit="$")
KRC = dict(fee=0.00015, sec=0.0, min_fee=0.0, tick=5.0, sizes=(1,), frac=False, unit="주")
SIGS = dict(S1=("1d", -0.04), S2=("5d", -0.08), S3=("dd60", -0.10))
PRIMARY = dict(sig="S1", H=60, us_size=50.0, kr_size=1)
SERIES = (("SPY", "US"), ("QQQ", "US"), ("069500", "KR"), ("229200", "KR"))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def load(sym, market):
    if market == "US":
        z = np.load(ETF, allow_pickle=True)
        syms = [str(s) for s in z["syms"]]
        j = syms.index(sym)
        days = np.array([str(d) for d in z["days"]])
        O, C, A = z["O"][:, j].astype(float), z["C"][:, j].astype(float), z["adj"][:, j].astype(float)
    else:
        k = json.load(open(KR, encoding="utf-8"))[sym]
        raw = json.load(open(KR_RAW, encoding="utf-8"))[sym]
        if k["days"] != raw["days"]:
            raise SystemExit(f"{sym}: adjusted/raw day lists differ")
        days, O, C = np.array(k["days"]), np.array(k["O"], float), np.array(k["C"], float)
        A = C / np.array(raw["C"], float)
        keep = days <= KR_LAST_COMPLETE
        days, O, C, A = days[keep], O[keep], C[keep], A[keep]
    ok = np.isfinite(O) & np.isfinite(C) & np.isfinite(A) & (O > 0) & (C > 0) & (A > 0)
    return days[ok], O[ok], C[ok], A[ok]


def signals(C):
    n = len(C)
    s1 = np.zeros(n, bool)
    s1[1:] = C[1:] / C[:-1] - 1 <= SIGS["S1"][1]
    s2 = np.zeros(n, bool)
    s2[5:] = C[5:] / C[:-5] - 1 <= SIGS["S2"][1]
    s3 = np.zeros(n, bool)
    for i in range(LOOK - 1, n):
        s3[i] = C[i] / C[i - LOOK + 1:i + 1].max() - 1 <= SIGS["S3"][1]
    return dict(S1=s1, S2=s2, S3=s3)


def episodes(sig):
    out = []
    for i in np.flatnonzero(sig):
        if not sig[max(0, i - GAP):i].any():
            out.append(int(i))
    return np.array(out, int)


def fee_amt(x, cfg):
    if cfg["frac"]:
        return np.maximum(np.floor(x * cfg["fee"] * 100) / 100, cfg["min_fee"])
    return np.floor(x * cfg["fee"])


def sec_amt(proceeds, cfg):
    if not cfg["sec"]:
        return 0.0
    return np.maximum(np.floor(proceeds * cfg["sec"] * 100) / 100, 0.01)


def trade(o_e, o_x, t_e, t_x, a_e, a_x, cfg, size):
    pb, ps = o_e + t_e, o_x - t_x
    if cfg["frac"]:
        amt = np.full_like(pb, float(size))
        qty = amt / pb
    else:
        qty = float(size) / a_e
        amt = qty * pb
    fb = fee_amt(amt, cfg)
    proceeds = qty * ps
    fs = fee_amt(proceeds, cfg) + sec_amt(proceeds, cfg)
    return (proceeds - fs - amt - fb) / (amt + fb)


def dca(O, T, A, e, x, cfg, size):
    buys = O[e:e + DCA_N] + T[e:e + DCA_N]
    if cfg["frac"]:
        amt = np.full(DCA_N, float(size) / DCA_N)
        qty = amt / buys
    else:
        qty = (float(size) / DCA_N) / A[e:e + DCA_N]
        amt = qty * buys
    fb = fee_amt(amt, cfg).sum()
    proceeds = qty.sum() * (O[x] - T[x])
    fs = float(fee_amt(np.array([proceeds]), cfg)[0])
    if cfg["sec"]:
        fs += float(sec_amt(np.array([proceeds]), cfg)[0])
    inv = amt.sum() + fb
    return (proceeds - fs - inv) / inv


def stats(ret):
    ret = np.asarray(ret, float)
    if len(ret) == 0:
        return dict(n=0)
    return dict(n=int(len(ret)), mean=float(ret.mean()), median=float(np.median(ret)), hit=float((ret > 0).mean()), p05=float(np.quantile(ret, 0.05)), p10=float(np.quantile(ret, 0.10)), min=float(ret.min()), max=float(ret.max()))


def boot(ret, base_mean, rng):
    if len(ret) < 2:
        return None, None
    idx = rng.integers(0, len(ret), (NBOOT, len(ret)))
    m = ret[idx].mean(1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))], float((m <= base_mean).mean())


def seed_of(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def run_series(sym, market, lines):
    cfg = US if market == "US" else KRC
    days, O, C, A = load(sym, market)
    T = cfg["tick"] * A
    n = len(C)
    sig = signals(C)
    res = dict(sym=sym, market=market, n_days=n, span=[str(days[0]), str(days[-1])], adj_ratio_first_last=[float(A[0]), float(A[-1])], signal_days={k: int(v.sum()) for k, v in sig.items()}, H={})
    lines.append(f"== {sym} ({market}) {days[0]}~{days[-1]} {n}일, 신호일 " + ", ".join(f"{k} {int(v.sum())}" for k, v in sig.items()))
    for H in H_LIST:
        hr = {}
        r_all = np.arange(LOOK - 1, n - 1 - H)
        e_all, x_all = r_all + 1, r_all + 1 + H
        for size in cfg["sizes"]:
            key = f"size_{size}"
            base = trade(O[e_all], O[x_all], T[e_all], T[x_all], A[e_all], A[x_all], cfg, size)
            pre = days[r_all] < ERA
            entry = dict(all_days=stats(base), all_days_era=dict(pre=stats(base[pre]), post=stats(base[~pre])), signals={})
            lines.append(f"-- H={H} {cfg['unit']}{size}: 아무 날 진입 n={len(base)} 평균 {base.mean()*100:+.2f}% 중앙 {np.median(base)*100:+.2f}% 승률 {(base>0).mean()*100:.0f}% 하위5% {np.quantile(base, 0.05)*100:+.2f}%")
            for sname, sarr in sig.items():
                ep_all = episodes(sarr)
                ep_all = ep_all[ep_all >= LOOK - 1]
                ep = ep_all[ep_all <= n - 2 - H]
                pending = ep_all[ep_all > n - 2 - H]
                alls = np.flatnonzero(sarr)
                alls = alls[(alls >= LOOK - 1) & (alls <= n - 2 - H)]
                rec = dict(episodes=int(len(ep)), episode_dates=[str(days[i]) for i in ep], pending_episodes=[str(days[i]) for i in pending], all_signal_days=int(len(alls)))
                if len(ep):
                    ret = trade(O[ep + 1], O[ep + 1 + H], T[ep + 1], T[ep + 1 + H], A[ep + 1], A[ep + 1 + H], cfg, size)
                    rng = np.random.default_rng(seed_of(sym, H, size, sname))
                    ci, frac_le = boot(ret, base.mean(), rng)
                    fdd = np.array([min(C[i + 1:i + 1 + H].min(), O[i + 1 + H] - T[i + 1 + H]) / (O[i + 1] + T[i + 1]) - 1 for i in ep])
                    pre_e = days[ep] < ERA
                    rec.update(stats=stats(ret), diff_mean=float(ret.mean() - base.mean()), ci95_iid_reference=ci, boot_mean_le_base_frac=frac_le, low_close_vs_entry=dict(mean=float(fdd.mean()), median=float(np.median(fdd)), p10=float(np.quantile(fdd, 0.10)), worst=float(fdd.min())), era=dict(pre=stats(ret[pre_e]), post=stats(ret[~pre_e])), returns=[round(float(v), 5) for v in ret])
                    if len(alls):
                        rec["all_signal_days_stats"] = stats(trade(O[alls + 1], O[alls + 1 + H], T[alls + 1], T[alls + 1 + H], A[alls + 1], A[alls + 1 + H], cfg, size))
                    cis = f"iid참고CI[{ci[0]*100:+.2f}%, {ci[1]*100:+.2f}%] 재표집평균≤기준 비율 {frac_le:.3f}" if ci else "CI n/a"
                    lines.append(f"   {sname} 에피소드 {len(ep)}(미완료 {len(pending)}): 평균 {ret.mean()*100:+.2f}% 중앙 {np.median(ret)*100:+.2f}% 승률 {(ret>0).mean()*100:.0f}% 하위5% {np.quantile(ret, 0.05)*100:+.2f}% 차이 {(ret.mean()-base.mean())*100:+.2f}%p {cis} | 진입가 대비 최저종가 평균 {fdd.mean()*100:.1f}% 중앙 {np.median(fdd)*100:.1f}% 최악 {fdd.min()*100:.1f}% | {ERA[:4]}전 {len(ret[pre_e])}건 {ret[pre_e].mean()*100 if pre_e.any() else float('nan'):+.2f}% 후 {len(ret[~pre_e])}건 {ret[~pre_e].mean()*100 if (~pre_e).any() else float('nan'):+.2f}%")
                else:
                    lines.append(f"   {sname} 에피소드 0(미완료 {len(pending)})")
                entry["signals"][sname] = rec
            if H in (60, 120):
                ok_all = x_all - e_all >= DCA_N
                sz = size * (DCA_N if not cfg["frac"] else 1)
                dca_all = np.array([dca(O, T, A, e, x, cfg, sz) for e, x in zip(e_all[ok_all], x_all[ok_all])])
                lump_all2 = trade(O[e_all[ok_all]], O[x_all[ok_all]], T[e_all[ok_all]], T[x_all[ok_all]], A[e_all[ok_all]], A[x_all[ok_all]], cfg, sz)
                d = dict(basis="fixed quantity/amount per buy; return on invested cash incl. fees", all_days=dict(n=int(ok_all.sum()), lump_mean=float(lump_all2.mean()), dca_mean=float(dca_all.mean()), diff_mean=float((dca_all - lump_all2).mean()), dca_hit=float((dca_all > 0).mean())), signals={})
                lines.append(f"   분할 vs 일시(아무 날, H={H}): 일시 {lump_all2.mean()*100:+.2f}% 분할20일 {dca_all.mean()*100:+.2f}% 차이 {(dca_all-lump_all2).mean()*100:+.2f}%p")
                for sname, sarr in sig.items():
                    ep = episodes(sarr)
                    ep = ep[(ep >= LOOK - 1) & (ep <= n - 2 - H)]
                    if len(ep):
                        dd = np.array([dca(O, T, A, i + 1, i + 1 + H, cfg, sz) for i in ep])
                        ll = trade(O[ep + 1], O[ep + 1 + H], T[ep + 1], T[ep + 1 + H], A[ep + 1], A[ep + 1 + H], cfg, sz)
                        d["signals"][sname] = dict(n=int(len(ep)), lump_mean=float(ll.mean()), dca_mean=float(dd.mean()), diff_mean=float((dd - ll).mean()), dca_hit=float((dd > 0).mean()))
                        lines.append(f"   분할 vs 일시({sname}, H={H}): 일시 {ll.mean()*100:+.2f}% 분할20일 {dd.mean()*100:+.2f}% 차이 {(dd-ll).mean()*100:+.2f}%p")
                entry["dca"] = d
            hr[key] = entry
        res["H"][str(H)] = hr
    return res


def main():
    meta = dict(version=4, script_sha256=sha(os.path.abspath(__file__)), etf_panel=os.path.relpath(ETF, HERE), etf_panel_sha256=sha(ETF), kr_file=os.path.relpath(KR, HERE), kr_sha256=sha(KR), kr_raw_file=os.path.relpath(KR_RAW, HERE), kr_raw_sha256=sha(KR_RAW), kr_last_complete=KR_LAST_COMPLETE, tick_basis="raw tick x (adjusted/raw) per day", kr_quantity_basis="real shares; adjusted units = shares / A[buy day]; amounts therefore in real won incl. distributions via A[x]/A[d]", kr_fee_basis="floor(real won amount x rate)", H_list=list(H_LIST), gap=GAP, look=LOOK, dca_n=DCA_N, nboot=NBOOT, era=ERA, us=US, kr=KRC, signals=SIGS, primary=PRIMARY, inference="descriptive only; iid bootstrap CI is reference, not a test")
    if sys.argv[1:] == ["meta"]:
        print("META", json.dumps(meta, ensure_ascii=False))
        return
    lines = []
    out = dict(meta=meta, series={})
    for sym, market in SERIES:
        out["series"][sym] = run_series(sym, market, lines)
    p = out["series"]
    summary = {}
    for sym, size in (("SPY", PRIMARY["us_size"]), ("QQQ", PRIMARY["us_size"]), ("069500", PRIMARY["kr_size"]), ("229200", PRIMARY["kr_size"])):
        e = p[sym]["H"][str(PRIMARY["H"])][f"size_{size}"]
        s = e["signals"][PRIMARY["sig"]]
        base = e["all_days"]["mean"]
        st = s.get("stats", {})
        summary[sym] = dict(signal_n=st.get("n", 0), signal_mean=st.get("mean"), signal_median=st.get("median"), signal_hit=st.get("hit"), all_days_mean=base, diff_mean=s.get("diff_mean"), ci95_iid_reference=s.get("ci95_iid_reference"), positive_after_cost=bool(st.get("mean", -1) > 0))
        lines.append(f"요약 {sym} {PRIMARY['sig']} H={PRIMARY['H']}: 신호 {st.get('n', 0)}건 평균 {st.get('mean', float('nan'))*100:+.2f}% (아무 날 {base*100:+.2f}%, 차이 {s.get('diff_mean', float('nan'))*100:+.2f}%p), 기술통계만, 판정 없음")
    out["summary_primary"] = summary
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n".join(lines))
    print("saved", OUT)


if __name__ == "__main__":
    main()
