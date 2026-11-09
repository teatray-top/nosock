import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import bisect
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
import scenario_test_v1 as SC
import learn_global_v1 as LG
import harvest_v5 as H5
import harvest_v8 as H8
import voltime_v1 as VT
import turning_points_v1 as TP

LONG, INV = H5.LONG, H5.INV
CAPITAL = H5.CAPITAL
HORIZON = 40
BARRIER = 0.08
TEST_YEARS = list(range(2008, 2027))
RUN_FROM, RUN_TO = "2008-01-02", "2026-11-05"
TICKERS = ["^GSPC", "^IXIC", "^RUT", "^N225", "^HSI", "000001.SS", "^TWII", "^KQ11", "^VIX", "KRW=X", "JPY=X", "DX-Y.NYB", "^TNX", "GC=F", "CL=F", "HG=F", "BTC-USD"]
THRESHOLDS = (0.5, 0.4)
PARAMS = dict(LI.PARAMS, objective="binary")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


class Asof(LG.Asof):
    def dd(self, d, n):
        i = self.idx(d)
        if i - n + 1 < 0:
            return float("nan")
        return self.C[i] / max(self.C[i - n + 1:i + 1]) - 1


def gfeat_names():
    out = []
    for t in TICKERS:
        out += [f"{t}_r1", f"{t}_r5", f"{t}_r20", f"{t}_dd60"]
    return out + ["vix_level", "tnx_level", "tnx_d20"]


def gfeat(g, d):
    out = []
    for t in TICKERS:
        a = g[t]
        out += [a.ret(d, 1), a.ret(d, 5), a.ret(d, 20), a.dd(d, 60)]
    return out + [g["^VIX"].level(d), g["^TNX"].level(d), g["^TNX"].diff(d, 20)]


def barrier_labels(Oa, Ca):
    n = len(Ca)
    up = np.full(n, np.nan)
    dn = np.full(n, np.nan)
    for i in range(n - HORIZON - 1):
        o = Oa[i + 1]
        u = d = 0.0
        for j in range(i + 1, i + 1 + HORIZON):
            if Ca[j] >= o * (1 + BARRIER):
                u = 1.0
                break
            if Ca[j] <= o * (1 - BARRIER):
                d = 1.0
                break
        up[i], dn[i] = u, d
    return up, dn


def build():
    days, k, _ = LI.load(LI.INV_DIR_DEFAULT)
    F = LI.features(days, k)
    raw = LG.load_global()
    g = {t: Asof(raw[t]) for t in TICKERS}
    vol = VT.ewma_vol(k["Ca"])
    med = VT.expanding_median(vol)
    names = LI.FEATS + ["ewma_vol", "vol_vs_median"] + gfeat_names()
    X = np.full((len(days), len(names)), np.nan)
    for i in range(LI.FEAT_START, len(days)):
        X[i, :len(LI.FEATS)] = F[i]
        X[i, len(LI.FEATS)] = vol[i]
        X[i, len(LI.FEATS) + 1] = vol[i] / med[i] if not np.isnan(med[i]) else np.nan
        X[i, len(LI.FEATS) + 2:] = gfeat(g, days[i])
    up, dn = barrier_labels(k["Oa"], k["Ca"])
    return days, k, X, names, up, dn


def walk_forward(days, X, up, dn):
    pu = np.full(len(days), np.nan)
    pd_ = np.full(len(days), np.nan)
    folds, imp = [], {}
    for y in TEST_YEARS:
        a = next(j for j, d in enumerate(days) if d >= f"{y}-01-01")
        b = next((j for j, d in enumerate(days) if d >= f"{y + 1}-01-01"), len(days))
        last_train = a - HORIZON - 2
        tr = [i for i in range(LI.FEAT_START, last_train + 1) if not np.isnan(up[i])]
        te = list(range(a, b))
        mu = lgb.train(PARAMS, lgb.Dataset(X[tr], label=up[tr]), num_boost_round=LI.ROUNDS)
        md = lgb.train(PARAMS, lgb.Dataset(X[tr], label=dn[tr]), num_boost_round=LI.ROUNDS)
        pu[te] = mu.predict(X[te])
        pd_[te] = md.predict(X[te])
        imp[y] = md.feature_importance("gain").tolist()
        folds.append(dict(year=y, train_rows=len(tr), train_last=days[last_train], test=[days[a], days[b - 1]], train_down_rate=float(np.mean(dn[tr])), train_up_rate=float(np.mean(up[tr]))))
    return pu, pd_, folds, imp


def auc(p, y):
    m = ~np.isnan(p) & ~np.isnan(y)
    p, y = p[m], y[m]
    if y.sum() == 0 or y.sum() == len(y):
        return None
    order = np.argsort(p)
    r = np.empty(len(p))
    r[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def spliced_inverse(days, k, a, b):
    iv = json.load(open(H5.H2.INV_DAILY, encoding="utf-8"))[INV]
    imap = {d: (float(o), float(c)) for d, o, c in zip(iv["days"], iv["O"], iv["C"])}
    O = np.asarray(k["O"][a:b + 1], float)
    C = np.asarray(k["C"][a:b + 1], float)
    SO, SCc = SC.inverse_path(O, C)
    sub = days[a:b + 1]
    first = next(j for j, d in enumerate(sub) if d in imap)
    scale = imap[sub[first]][1] / SCc[first]
    IO = [SO[j] * scale if j < first else imap[sub[j]][0] for j in range(len(sub))]
    IC = [SCc[j] * scale if j < first else imap[sub[j]][1] for j in range(len(sub))]
    return IO, IC, sub[first]


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, p in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), p)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    days, k, X, names, up, dn = build()
    pu, pdn, folds, imp = walk_forward(days, X, up, dn)
    print("folds", len(folds), "rows", int(np.sum(~np.isnan(up))), "base down/up", float(np.nanmean(dn)), float(np.nanmean(up)), flush=True)
    per_year = []
    for f in folds:
        y = f["year"]
        idx = np.array([i for i, d in enumerate(days) if d[:4] == str(y)])
        per_year.append(dict(year=y, auc_down=auc(pdn[idx], dn[idx]), auc_up=auc(pu[idx], up[idx]), down_rate=float(np.nanmean(dn[idx])) if np.any(~np.isnan(dn[idx])) else None, mean_p_down=float(np.mean(pdn[idx])), max_p_down=float(np.max(pdn[idx]))))
        r = per_year[-1]
        print(f"  {y} aucD {r['auc_down']} aucU {r['auc_up']} downrate {r['down_rate']} meanPd {r['mean_p_down']:.3f} maxPd {r['max_p_down']:.3f}", flush=True)
    oos = np.array([i for i, d in enumerate(days) if d >= "2008-01-01"])
    allauc = dict(down=auc(pdn[oos], dn[oos]), up=auc(pu[oos], up[oos]))
    print("OOS AUC 2008-2026", allauc, flush=True)
    a = days.index(next(d for d in days if d >= RUN_FROM)) - 1
    b = days.index(RUN_TO)
    sub = days[a:b + 1]
    IO, IC, splice = spliced_inverse(days, k, a, b)
    px = {LONG: dict(O=k["O"][a:b + 1], C=k["C"][a:b + 1]), INV: dict(O=IO, C=IC)}
    cost = lambda leg, j: LI.FEE + (LI.TICK if leg == LONG else H8.tick(sub[j], px[INV]["O"][j])) / px[leg]["O"][j]
    sig8 = {kk: v[a:b + 1] for kk, v in H8.signals(k["Ca"]).items()}
    pols = [dict(name="R0_hold", kind="skim", sig=None), dict(name="R1_skim10", kind="skim", g=0.10, sig=None), dict(name="I_sma200_down", kind="skim", g=0.10, bear="inv", sig="sma200_down")]
    for th in THRESHOLDS:
        nm = f"L_turn_p{int(th * 100)}"
        s = np.zeros(len(sub), bool)
        for j in range(len(sub)):
            p, q = pdn[a + j], pu[a + j]
            s[j] = (not np.isnan(p)) and p > th and p > q
        sig8[nm] = s
        pols.append(dict(name=nm, kind="skim", g=0.10, bear="inv", sig=nm))
    res, series, trades = H8.run_set(pols, px, cost, 0, len(sub), sig8, keep=True)
    years = {kk: H5.yearly(s, sub) for kk, s in series.items()}
    for kk in res:
        res[kk]["years"] = years[kk]
        res[kk]["years_lost_money"] = [y for y, v in years[kk].items() if v < 0]
        print(f"{kk:18s} vault {res[kk]['vault_pct']:6.1f} total {res[kk]['total_pct']:7.1f} lowest {res[kk]['lowest_total_pct']:6.1f} mdd {res[kk]['mdd_total']:6.1f} invdays {res[kk]['inverse_days']} switches {res[kk]['switches']} cost {res[kk]['cost_pct']:.2f} lostyears {len(res[kk]['years_lost_money'])}", flush=True)
        print("    years", {y: round(float(v), 1) for y, v in years[kk].items()}, flush=True)
    A = k["Ca"][a:b + 1]
    pts = TP.zigzag(A, TP.ZIG)
    events = []
    for kind, i in pts:
        row = dict(kind=kind, day=sub[i], market_window=[sub[max(0, i - 40)], sub[min(len(sub) - 1, i + 60)]])
        for nm in ["sma200_down"] + [f"L_turn_p{int(th * 100)}" for th in THRESHOLDS]:
            s = sig8[nm]
            on = [j for j in range(max(0, i - 40), min(len(sub), i + 61)) if s[j] and (j == 0 or not s[j - 1])]
            row[nm] = dict(first_on_offset=(on[0] - i) if on else None, on_days_window=int(sum(s[max(0, i - 40):min(len(sub), i + 61)])))
        row["p_down_at_turn"] = float(pdn[a + i])
        row["p_down_max_before"] = float(np.nanmax(pdn[a + max(0, i - 40):a + i + 1]))
        events.append(row)
        print("  turn", kind, sub[i], {nm: row[nm]["first_on_offset"] for nm in ["sma200_down"] + [f"L_turn_p{int(th * 100)}" for th in THRESHOLDS]}, "pD at", round(row["p_down_at_turn"], 3), "max before", round(row["p_down_max_before"], 3), flush=True)
    top = {}
    for y, v in imp.items():
        for nm, x in zip(names, v):
            top[nm] = top.get(nm, 0.0) + x
    top = sorted(top.items(), key=lambda z: -z[1])[:15]
    print("top features (down model gain)", [(nm, round(x)) for nm, x in top], flush=True)
    rec = dict(version="turning-learn-v1", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], features=names, folds=folds, per_year=per_year, oos_auc=allauc, splice_day=splice, policies=pols, results=res, trades={kk: [dict(x, decision_day=sub[x["decision_index"]], fill_day=sub[x["fill_index"]]) for x in t] for kk, t in trades.items()}, turning_events=events, top_features=top, predictions={days[i]: [float(pdn[i]), float(pu[i])] for i in range(len(days)) if not np.isnan(pdn[i])})
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
