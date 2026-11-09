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
import learn_inverse as LI
import harvest_v8 as H8
import msma_v1 as M1

CAPITAL = M1.CAPITAL
VARIANTS = [dict(name=f"EQ_vol{int(v * 100)}_band{int(b * 100)}", vol=v, band=b, vault=False) for v in (0.12, 0.18, 0.25) for b in (0.0, 0.02)] + [dict(name="EQ_vol18_band2_vault", vol=0.18, band=0.02, vault=True), dict(name="EQ_vol25_band2_vault", vol=0.25, band=0.02, vault=True)]
SKIM_G, VAULT_SHARE = 0.10, 0.5


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def target_weights(i, avail, score, sd60, r, m, vol, rule):
    w = np.zeros(m)
    if rule == "hold_069500":
        w[0] = 1.0
        return w
    if rule == "equal_weight_all":
        w[avail] = 1.0 / avail.sum()
        return w
    sc = np.where(avail, score[i], np.nan)
    pos = np.nan_to_num(sc) > 0
    if not pos.any():
        return w
    raw = np.where(pos, np.nan_to_num(sc), 0.0) / np.where(pos, sd60[i], 1.0)
    w = raw / raw.sum()
    for _ in range(5):
        over = w > M1.CAP
        if not over.any():
            break
        ex = (w[over] - M1.CAP).sum()
        w[over] = M1.CAP
        under = (~over) & (w > 0)
        if under.any():
            w[under] += ex * w[under] / w[under].sum()
    lo = max(0, i - 59)
    R = np.nan_to_num(r[lo + 1:i + 1])
    cov = np.cov(R.T) * 252 if R.shape[0] > 20 else np.eye(m) * 0.04
    pv = math.sqrt(max(w @ cov @ w, 1e-12))
    return w * min(1.0, vol / pv)


def run(days, codes, O, C, sd60, r, score, start, rule, vol=0.12, band=0.0, vault_on=False):
    n, m = C.shape
    cost = lambda j, i: LI.FEE + H8.tick(days[i], O[i, j]) / O[i, j]
    cash, vault, q = CAPITAL, 0.0, np.zeros(m)
    basis = CAPITAL
    vals, peak, mdd, low, costs, expo = [], CAPITAL, 0.0, CAPITAL, 0.0, []
    for i in range(start, n - 1):
        px = np.nan_to_num(C[i])
        acc = cash + float(np.sum(q * px))
        if vault_on and acc >= basis * (1 + SKIM_G):
            take = VAULT_SHARE * (acc - basis)
            frac = take / acc
            got = cash * frac
            cash -= got
            for j in range(m):
                if q[j] > 0 and not np.isnan(O[i + 1, j]):
                    sq = q[j] * frac
                    c = cost(j, i + 1)
                    got += sq * O[i + 1, j] * (1 - c)
                    costs += sq * O[i + 1, j] * c
                    q[j] -= sq
            vault += got
            basis = cash + float(np.sum(q * px))
        elif (i - start) % M1.HOLD == 0:
            avail = ~np.isnan(C[i]) & ~np.isnan(sd60[i])
            w = target_weights(i, avail, score, sd60, r, m, vol, rule)
            cur = q * px / acc if acc > 0 else np.zeros(m)
            for j in range(m):
                if abs(w[j] - cur[j]) <= band and not (w[j] == 0 and q[j] > 0):
                    continue
                dq = w[j] * acc / px[j] - q[j] if px[j] > 0 else -q[j]
                o = O[i + 1, j]
                if np.isnan(o) or abs(dq * o) < 1e-6 * acc:
                    continue
                c = cost(j, i + 1)
                if dq > 0:
                    spend = min(cash, dq * o * (1 + c))
                    bq = spend / (o * (1 + c))
                    q[j] += bq
                    cash -= spend
                    costs += bq * o * c
                else:
                    sq = min(q[j], -dq)
                    cash += sq * o * (1 - c)
                    q[j] -= sq
                    costs += sq * o * c
        accn = cash + float(np.sum(q * np.nan_to_num(C[i + 1])))
        tot = accn + vault
        expo.append(float(np.sum(q * np.nan_to_num(C[i + 1]))) / accn if accn > 0 else 0.0)
        vals.append(tot)
        peak = max(peak, tot)
        mdd = min(mdd, tot / peak - 1)
        low = min(low, tot)
    v = np.asarray(vals)
    dr = np.diff(np.log(np.r_[CAPITAL, v]))
    dd = days[start + 1:n]
    yrs, prev = {}, CAPITAL
    for y in sorted(set(x[:4] for x in dd)):
        last = max(k for k, x in enumerate(dd) if x[:4] == y)
        yrs[y] = 100 * (v[last] / prev - 1)
        prev = v[last]
    years = len(v) / 252
    return dict(total_pct=100 * (v[-1] / CAPITAL - 1), cagr_pct=100 * ((v[-1] / CAPITAL) ** (1 / years) - 1), sharpe=float(dr.mean() / dr.std() * math.sqrt(252)), mdd_pct=100 * mdd, lowest_pct=100 * (low / CAPITAL - 1), cost_pct=100 * costs / CAPITAL, vault_pct=100 * vault / CAPITAL, mean_exposure=float(np.mean(expo)), years=yrs, losing_years=[y for y, x in yrs.items() if x < 0])


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, p in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), p)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    days, codes, O, C = M1.load()
    n, m = C.shape
    S, sd60, r = M1.signals(days, C)
    eq = np.nanmean(np.tanh(np.stack([S[:, :, M1.SIG.index(s)] for s in M1.EQ_SIGNALS], axis=2)), axis=2)
    start = int(np.argmax(days >= M1.TEST_FROM)) - 1
    res = {}
    res["hold_069500"] = run(days, codes, O, C, sd60, r, eq, start, "hold_069500")
    res["equal_weight_all"] = run(days, codes, O, C, sd60, r, eq, start, "equal_weight_all")
    for vrt in VARIANTS:
        res[vrt["name"]] = run(days, codes, O, C, sd60, r, eq, start, "score", vrt["vol"], vrt["band"], vrt["vault"])
    for k, x in res.items():
        print(f"{k:24s} total {x['total_pct']:8.1f}% cagr {x['cagr_pct']:5.2f}% sharpe {x['sharpe']:.2f} mdd {x['mdd_pct']:6.1f} lowest {x['lowest_pct']:6.1f} cost {x['cost_pct']:.2f}% vault {x['vault_pct']:.1f}% exposure {x['mean_exposure']:.2f} losing {len(x['losing_years'])}", flush=True)
        print("    years", {y: round(float(v), 1) for y, v in x["years"].items()}, flush=True)
    rec = dict(version="msma-v2", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], variants=VARIANTS, results=res)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
