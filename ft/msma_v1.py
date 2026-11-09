import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
import math
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import lightgbm as lgb
import learn_inverse as LI
import compare_trend_v2 as CT
import learn_global_v1 as LG
import harvest_v8 as H8
import kalman_fft_v1 as K1
import kalman_fft_v2 as K2

ETF_DIR = os.path.join(os.path.dirname(HERE), "data", "kr_etf", "yf_etfs_20261107_224606")
CLASS = {"069500": "kr_eq", "229200": "kr_eq", "114800": "kr_inv", "091160": "kr_sector", "091170": "kr_sector", "133690": "us_eq", "143850": "us_eq", "360750": "us_eq", "192090": "cn_eq", "132030": "gold", "130680": "oil", "261240": "usd", "148070": "bond", "114260": "bond", "153130": "cash_like"}
CLASSES = sorted(set(CLASS.values()))
SIG = ["mom5", "mom20", "mom60", "mom120", "mom250", "macd8_24", "macd16_48", "macd32_96", "volratio", "dd250", "kfft128", "kfft64"]
GLOB = ["gspc_r1", "gspc_r5", "ixic_r1", "vix_level", "vix_chg5", "krw_r5", "tnx_d20"]
EQ_SIGNALS = ["mom20", "mom60", "mom120", "mom250", "macd8_24", "macd16_48", "macd32_96", "kfft128"]
HOLD = 5
TARGET_VOL = 0.12
CAP = 0.30
TEST_FROM = "2013-01-02"
END = "2026-11-05"
CAPITAL = 100_000_000.0


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load():
    man = json.load(open(os.path.join(ETF_DIR, "manifest.json"), encoding="utf-8"))
    recs = {}
    for code in CLASS:
        info = man["files"][f"{code}_adj"]
        p = os.path.join(ETF_DIR, info["file"])
        if sha(p) != info["sha256"]:
            raise SystemExit(f"etf hash differs {code}")
        recs[code] = json.load(open(p, encoding="utf-8"))
    days = [d for d in recs["069500"]["days"] if d <= END]
    codes = list(CLASS)
    n, m = len(days), len(codes)
    O = np.full((n, m), np.nan)
    C = np.full((n, m), np.nan)
    pos = {d: i for i, d in enumerate(days)}
    for j, c in enumerate(codes):
        for d, o, cl in zip(recs[c]["days"], recs[c]["O"], recs[c]["C"]):
            if d in pos:
                O[pos[d], j], C[pos[d], j] = o, cl
    for j in range(m):
        last_o = last_c = np.nan
        for i in range(n):
            if np.isnan(C[i, j]) and not np.isnan(last_c):
                O[i, j], C[i, j] = last_c, last_c
            if not np.isnan(C[i, j]):
                last_c = C[i, j]
    return np.array(days), codes, O, C


def ema(x, n):
    out = np.full(len(x), np.nan)
    a = 2.0 / (n + 1)
    v = np.nan
    for i, z in enumerate(x):
        if np.isnan(z):
            continue
        v = z if np.isnan(v) else a * z + (1 - a) * v
        out[i] = v
    return out


def signals(days, C):
    n, m = C.shape
    lc = np.log(C)
    r = np.vstack([np.full(m, np.nan), np.diff(lc, axis=0)])
    S = np.full((n, m, len(SIG)), np.nan)
    sd60 = np.full((n, m), np.nan)
    for j in range(m):
        rj = r[:, j]
        s60 = np.array([np.nanstd(rj[max(0, i - 59):i + 1]) if np.sum(~np.isnan(rj[max(0, i - 59):i + 1])) >= 40 else np.nan for i in range(n)])
        s20 = np.array([np.nanstd(rj[max(0, i - 19):i + 1]) if np.sum(~np.isnan(rj[max(0, i - 19):i + 1])) >= 15 else np.nan for i in range(n)])
        s250 = np.array([np.nanstd(rj[max(0, i - 249):i + 1]) if np.sum(~np.isnan(rj[max(0, i - 249):i + 1])) >= 200 else np.nan for i in range(n)])
        sd60[:, j] = s60
        x = lc[:, j]
        for q, k in enumerate((5, 20, 60, 120, 250)):
            S[k:, j, q] = (x[k:] - x[:-k]) / (s60[k:] * math.sqrt(k))
        for q, (a, b) in enumerate(((8, 24), (16, 48), (32, 96))):
            S[:, j, 5 + q] = (ema(C[:, j], a) - ema(C[:, j], b)) / (C[:, j] * s60 * math.sqrt(b))
        S[:, j, 8] = s20 / s250
        hi = np.array([np.nanmax(C[max(0, i - 249):i + 1, j]) if not np.isnan(C[i, j]) else np.nan for i in range(n)])
        S[:, j, 9] = C[:, j] / hi - 1
        valid = ~np.isnan(x)
        if valid.sum() > 300:
            idx = np.nonzero(valid)[0]
            lev, _ = K1.kalman(x[idx], np.zeros(len(idx), dtype=int), 0.01)
            for q, (W, Kh) in enumerate(((128, 3), (64, 1))):
                f = K1.fft_features(lev, np.arange(len(idx)), W, Kh)
                S[idx, j, 10 + q] = f[:, 5] / s60[idx]
    return S, sd60, r


def global_feats(days):
    g = LG.load_global()
    a = {t: LG.Asof(g[t]) for t in ("^GSPC", "^IXIC", "^VIX", "KRW=X", "^TNX")}
    out = np.full((len(days), len(GLOB)), np.nan)
    for i, d in enumerate(days):
        out[i] = [a["^GSPC"].ret(d, 1), a["^GSPC"].ret(d, 5), a["^IXIC"].ret(d, 1), a["^VIX"].level(d), a["^VIX"].ret(d, 5), a["KRW=X"].ret(d, 5), a["^TNX"].diff(d, 20)]
    return out


def labels(O, sd60):
    n, m = O.shape
    y = np.full((n, m), np.nan)
    for i in range(n - HOLD - 1):
        y[i] = np.log(O[i + 1 + HOLD] / O[i + 1]) / (sd60[i] * math.sqrt(HOLD))
    return y


def portfolio(days, codes, O, C, sd60, r, score, start, rule):
    n, m = C.shape
    cost_side = lambda j, i: LI.FEE + H8.tick(days[i], O[i, j]) / O[i, j]
    cash, q = CAPITAL, np.zeros(m)
    vals, peak, mdd, low, turn, costs = [], CAPITAL, 0.0, CAPITAL, 0.0, 0.0
    for i in range(start, n - 1):
        if (i - start) % HOLD == 0:
            avail = ~np.isnan(C[i]) & ~np.isnan(sd60[i]) & (np.arange(n)[i] >= 0)
            w = np.zeros(m)
            if rule == "hold_069500":
                w[codes.index("069500")] = 1.0
            elif rule == "equal_weight_all":
                k = avail.sum()
                w[avail] = 1.0 / k
            else:
                sc = np.where(avail, score[i] if rule != "inv_vol_all" else 1.0, np.nan)
                pos_ = np.nan_to_num(sc) > 0
                if pos_.any():
                    raw = np.where(pos_, np.nan_to_num(sc) if rule != "inv_vol_all" else 1.0, 0.0) / np.where(pos_, sd60[i], 1.0)
                    w = raw / raw.sum()
                    for _ in range(5):
                        over = w > CAP
                        if not over.any():
                            break
                        ex = (w[over] - CAP).sum()
                        w[over] = CAP
                        under = (~over) & (w > 0)
                        if under.any():
                            w[under] += ex * w[under] / w[under].sum()
                    lo = max(0, i - 59)
                    R = np.nan_to_num(r[lo + 1:i + 1])
                    cov = np.cov(R.T) * 252 if R.shape[0] > 20 else np.eye(m) * 0.04
                    pv = math.sqrt(max(w @ cov @ w, 1e-12))
                    w = w * min(1.0, TARGET_VOL / pv)
            px = C[i]
            val = cash + np.nansum(q * np.nan_to_num(px))
            tgt = w * val / np.where(np.isnan(px), 1.0, px)
            for j in range(m):
                dq = tgt[j] - q[j]
                if abs(dq * np.nan_to_num(px[j])) < 1e-6 * val:
                    continue
                o = O[i + 1, j]
                if np.isnan(o):
                    continue
                c = cost_side(j, i + 1)
                if dq > 0:
                    spend = min(cash, dq * o * (1 + c))
                    bq = spend / (o * (1 + c))
                    q[j] += bq
                    cash -= spend
                    costs += bq * o * c
                    turn += spend
                else:
                    sq = min(q[j], -dq)
                    cash += sq * o * (1 - c)
                    q[j] -= sq
                    costs += sq * o * c
                    turn += sq * o
        tot = cash + np.nansum(q * np.nan_to_num(C[i + 1]))
        vals.append(tot)
        peak = max(peak, tot)
        mdd = min(mdd, tot / peak - 1)
        low = min(low, tot)
    v = np.asarray(vals)
    dr = np.diff(np.log(np.r_[CAPITAL, v]))
    yrs = {}
    dd = days[start + 1:n]
    prev = CAPITAL
    for y in sorted(set(x[:4] for x in dd)):
        last = max(k for k, x in enumerate(dd) if x[:4] == y)
        yrs[y] = 100 * (v[last] / prev - 1)
        prev = v[last]
    years = len(v) / 252
    return dict(total_pct=100 * (v[-1] / CAPITAL - 1), cagr_pct=100 * ((v[-1] / CAPITAL) ** (1 / years) - 1), sharpe=float(dr.mean() / dr.std() * math.sqrt(252)) if dr.std() > 0 else None, mdd_pct=100 * mdd, lowest_pct=100 * (low / CAPITAL - 1), cost_pct=100 * costs / CAPITAL, turnover_x=turn / CAPITAL, years=yrs, losing_years=[y for y, x in yrs.items() if x < 0])


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, p in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), p)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    days, codes, O, C = load()
    n, m = C.shape
    S, sd60, r = signals(days, C)
    G = global_feats(days)
    y = labels(O, sd60)
    print("days", n, days[0], days[-1], "assets", m, flush=True)
    ic = {}
    test_rows = days >= TEST_FROM
    for q, s in enumerate(SIG):
        a, b = S[test_rows, :, q].ravel(), y[test_rows].ravel()
        ok = ~np.isnan(a) & ~np.isnan(b)
        ic[s] = float(np.corrcoef(a[ok], b[ok])[0, 1])
    eq = np.nanmean(np.tanh(np.stack([S[:, :, SIG.index(s)] for s in EQ_SIGNALS], axis=2)), axis=2)
    a, b = eq[test_rows].ravel(), y[test_rows].ravel()
    ok = ~np.isnan(a) & ~np.isnan(b)
    ic["EQ_combined"] = float(np.corrcoef(a[ok], b[ok])[0, 1])
    onehot = np.zeros((m, len(CLASSES)))
    for j, c in enumerate(codes):
        onehot[j, CLASSES.index(CLASS[c])] = 1
    X = np.concatenate([S, np.repeat(G[:, None, :], m, axis=1), np.repeat(onehot[None, :, :], n, axis=0)], axis=2)
    lg = np.full((n, m), np.nan)
    params = dict(LI.PARAMS)
    for yr in range(2013, 2027):
        cut = np.argmax(days >= f"{yr}-01-01")
        nxt = np.argmax(days >= f"{yr + 1}-01-01") if (days >= f"{yr + 1}-01-01").any() else n
        tr_i = np.arange(0, max(0, cut - HOLD - 2))
        Xt = X[tr_i].reshape(-1, X.shape[2])
        yt = y[tr_i].ravel()
        ok = ~np.isnan(yt) & (np.sum(~np.isnan(Xt[:, :len(SIG)]), axis=1) >= 8)
        mdl = lgb.train(params, lgb.Dataset(Xt[ok], label=np.clip(yt[ok], -5, 5)), num_boost_round=LI.ROUNDS)
        te = np.arange(cut, nxt)
        Xe = X[te].reshape(-1, X.shape[2])
        lg[te] = mdl.predict(Xe).reshape(len(te), m)
        print("fold", yr, "train rows", int(ok.sum()), flush=True)
    a, b = lg[test_rows].ravel(), y[test_rows].ravel()
    ok = ~np.isnan(a) & ~np.isnan(b)
    ic["LGBM_combined"] = float(np.corrcoef(a[ok], b[ok])[0, 1])
    print("IC (pooled, test 2013+, next-5-day vol-scaled return):", {k: round(v, 4) for k, v in ic.items()}, flush=True)
    start = int(np.argmax(days >= TEST_FROM)) - 1
    res = {}
    mom250 = S[:, :, SIG.index("mom250")]
    kf = S[:, :, SIG.index("kfft128")]
    for rule, sc in (("hold_069500", None), ("equal_weight_all", None), ("inv_vol_all", None), ("single_mom250", mom250), ("single_kfft128", kf), ("EQ_multi_signal", eq), ("LGBM_multi_signal", lg)):
        res[rule] = portfolio(days, codes, O, C, sd60, r, sc if sc is not None else np.zeros((n, m)), start, rule if rule in ("hold_069500", "equal_weight_all", "inv_vol_all") else "score")
        x = res[rule]
        print(f"{rule:20s} total {x['total_pct']:8.1f}% cagr {x['cagr_pct']:5.2f}% sharpe {x['sharpe']:.2f} mdd {x['mdd_pct']:6.1f} lowest {x['lowest_pct']:6.1f} cost {x['cost_pct']:.2f}% turnover {x['turnover_x']:.1f}x losing years {len(x['losing_years'])}", flush=True)
        print("    years", {k: round(v, 1) for k, v in x["years"].items()}, flush=True)
    rec = dict(version="msma-v1", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], codes=codes, signals=SIG, global_features=GLOB, ic=ic, results=res)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
