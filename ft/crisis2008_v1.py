import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import learn_inverse as LI
import learn_inverse_synth_v3 as V3
import compare_trend_v2 as CT
import harvest_v2 as H2

YEARS = (2008, 2009, 2010, 2011, 2012)
TEST_FROM, TEST_TO = "2008-01-02", "2012-12-28"
LONG, INV = "069500", "114800"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def folds(days):
    out = []
    n = len(days)
    for y in YEARS:
        e = next(j for j, d in enumerate(days) if d >= f"{y}-01-01")
        nxt = next((j for j, d in enumerate(days) if d >= f"{y + 1}-01-01"), None)
        c = e - 1
        out.append(dict(year=y, cutoff=c, cutoff_day=days[c], first_dec=c, last_dec=nxt - 2, train_last=c - LI.H - 1))
    return out


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, p in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), p)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    days, k, inv = LI.load(LI.INV_DIR_DEFAULT)
    F = LI.features(days, k)
    man = json.load(open(os.path.join(V3.PATHS, "manifest.json"), encoding="utf-8"))
    train = [V3.load_path(e) for e in man["paths"] if e["split"] == "train"]
    pool_X = np.vstack([p["X"] for p in train])
    pool_yl = np.concatenate([p["yl"] for p in train])
    pool_R = np.concatenate([p["R"] for p in train])
    cal = V3.calibrate(days, k, LI.INV_DIR_DEFAULT, "2012-12-28")
    preds, info = {}, []
    for f in folds(days):
        jl = list(range(LI.FEAT_START, f["train_last"] + 1))
        js = [j for j in jl if LI.label_inv(days, inv, j) is not None]
        Xl = np.vstack([np.asarray([F[j] for j in jl]), pool_X])
        yl = np.concatenate([[LI.label_long(k, j) for j in jl], pool_yl])
        pyi = V3.inv_labels_from_ratio(pool_R, cal)
        Xs = np.vstack([np.asarray([F[j] for j in js]).reshape(-1, 17), pool_X])
        ys = np.concatenate([[LI.label_inv(days, inv, j) for j in js], pyi])
        ml, ms = LI.fit(Xl, yl), LI.fit(Xs, ys)
        dec = list(range(f["first_dec"], f["last_dec"] + 1))
        Xd = np.asarray([F[i] for i in dec])
        pl, ps = ml.predict(Xd), ms.predict(Xd)
        for i, a, b in zip(dec, pl, ps):
            preds[days[i]] = (float(a), float(b))
        info.append(dict(year=f["year"], cutoff_day=f["cutoff_day"], real_long_rows=len(jl), real_inv_rows=len(js), real_first=days[jl[0]], real_last_label_day=days[f["train_last"] + LI.H + 1], pred_long_mean=float(pl.mean()), pred_long_pos=float((pl > 0).mean()), pred_inv_pref=float(((ps > pl) & (ps > 0)).mean())))
    raw = json.load(open(H2.os.path.join(os.path.dirname(HERE), "data", "kr_etf", "kodex_daily_raw.json"), encoding="utf-8"))[LONG]
    adj = json.load(open(CT.ADJ, encoding="utf-8"))[LONG]
    iv = json.load(open(H2.INV_DAILY, encoding="utf-8"))[INV]
    imap = {d: (float(o), float(c)) for d, o, c in zip(iv["days"], iv["O"], iv["C"])}
    alld = [d for d in raw["days"] if d <= TEST_TO]
    start = alld.index(TEST_FROM) - 1
    keep = list(range(start, len(alld)))
    sub = [alld[i] for i in keep]
    LO = [float(raw["O"][i]) for i in keep]
    LC = [float(raw["C"][i]) for i in keep]
    AO = [float(adj["O"][i]) for i in keep]
    AC = [float(adj["C"][i]) for i in keep]
    IO, IC = [10000.0], [10000.0]
    for t in range(1, len(keep)):
        IO.append(IC[-1] * (2 - AO[t] / AC[t - 1]) * (1 + cal["drag"]))
        IC.append(IC[-1] * (2 - AC[t] / AC[t - 1]) * (1 + cal["drag"]))
    A = [float(x) for x in adj["C"][:len(alld)]]
    e = {m: CT.ema(A, m) for m in (8, 21, 50, 200)}
    px = {LONG: dict(O=LO, C=LC), INV: dict(O=IO, C=IC)}
    bull = lambda i: e[8][keep[i]] > e[21][keep[i]] > e[50][keep[i]] > e[200][keep[i]]
    bear = lambda i: e[8][keep[i]] < e[21][keep[i]] < e[50][keep[i]] < e[200][keep[i]]
    pred = lambda i: preds.get(sub[i], (None, None))
    tr = LI.TICK / cal["pref"]
    cost = lambda leg, j: LI.FEE + (LI.TICK / px[leg]["O"][j] if leg == LONG else tr)
    pols = H2.policies()
    res = {p["name"]: H2.run(p, px, cost, lambda j: sub[j][:4], 0, len(sub), pred, bull, bear) for p in pols}
    idx = {d: i for i, d in enumerate(days)}
    o = np.asarray([(preds[d][0], LI.label_long(k, idx[d])) for d in sorted(preds) if idx[d] + LI.H + 1 < len(days)])
    oos = dict(corr=float(np.corrcoef(o[:, 0], o[:, 1])[0, 1]), sign_hit=float(np.mean((o[:, 0] > 0) == (o[:, 1] > 0))), base=float(np.mean(o[:, 1] > 0)), n=int(len(o)))
    rec = dict(version="crisis2008-v1", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], calibration=cal, folds=info, oos=oos, window=[sub[0], sub[-1]], market_069500_pct=100 * (LC[-1] / LC[0] - 1), results=res)
    with open(out_path, "x", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1)
    for fi in info:
        print(fi)
    print("oos", oos, "window", sub[0], sub[-1], "069500 price change", round(100 * (LC[-1] / LC[0] - 1), 1), "%")
    for p in pols:
        r = res[p["name"]]
        print(f"{p['name']:26s} vault {r['vault_pct']:6.1f}% total {r['total_pct']:7.1f}% mdd {r['mdd_total']:6.1f} sales {r['sales']:3d} inv {r['inverse_sales']:3d} by year {r['vault_by_year']}")
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
