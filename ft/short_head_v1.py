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
import numpy as np
import learn_inverse as LI
import learn_inverse_synth_v3 as V3

HS = 5
OUT_PRED = os.path.join(HERE, "results", "short_walkforward_predictions_v1.json")
OUT_MODELS = os.path.join(HERE, "results", "short_models_v1")
RANGE_FEATS = ["pos20", "z20", "er20", "r3"]


def range_features(C, i):
    w = C[i - 19:i + 1]
    lo, hi = min(w), max(w)
    pos = (C[i] - lo) / (hi - lo) if hi > lo else 0.5
    m = sum(w) / 20.0
    sd = math.sqrt(sum((x - m) ** 2 for x in w) / 20.0)
    z = (C[i] - m) / sd if sd > 0 else 0.0
    path = sum(abs(C[t] - C[t - 1]) for t in range(i - 19, i + 1))
    er = abs(C[i] - C[i - 20]) / path if path > 0 else 0.0
    return [pos, z, er, math.log(C[i] / C[i - 3])]


def features21(C):
    F = LI.features([None] * len(C), dict(Ca=C))
    return {i: f + range_features(C, i) for i, f in F.items()}


def label_long5(k, i):
    b, s = k["O"][i + 1] + LI.TICK, k["O"][i + HS + 1] - LI.TICK
    fac = (k["Oa"][i + HS + 1] / k["O"][i + HS + 1]) / (k["Oa"][i + 1] / k["O"][i + 1])
    return s * (1 - LI.FEE) / (b * (1 + LI.FEE)) * fac - 1


def label_inv5(days, inv, i):
    d1, d2 = days[i + 1], days[i + HS + 1]
    if d1 not in inv or d2 not in inv:
        return None
    return (inv[d2] - LI.TICK) * (1 - LI.FEE) / ((inv[d1] + LI.TICK) * (1 + LI.FEE)) - 1


def synth_pool():
    man = json.load(open(os.path.join(V3.PATHS, "manifest.json"), encoding="utf-8"))
    X, yl, R = [], [], []
    for e in man["paths"]:
        if e["split"] != "train":
            continue
        p = V3.load_path(e)
        O, C = p["k"]["O"], p["k"]["C"]
        F = features21(C)
        for i in range(LI.FEAT_START, len(C) - HS - 1):
            X.append(F[i])
            yl.append((O[i + HS + 1] - LI.TICK) * (1 - LI.FEE) / ((O[i + 1] + LI.TICK) * (1 + LI.FEE)) - 1)
            R.append(p["IO0"][i + HS + 1] / p["IO0"][i + 1])
    return np.asarray(X), np.asarray(yl), np.asarray(R)


def inv5_from_ratio(R, cal):
    tr = LI.TICK / cal["pref"]
    return R * (1 + cal["drag"]) ** HS * (1 - tr) * (1 - LI.FEE) / ((1 + tr) * (1 + LI.FEE)) - 1


def main():
    days, k, inv = LI.load(LI.INV_DIR_DEFAULT)
    F = features21(k["Ca"])
    pX, pyl, pR = synth_pool()
    preds, info = {}, []
    last_models = None
    for f in LI.folds(days):
        c = f["cutoff"]
        jl = [j for j in range(LI.FEAT_START, c - HS) if j + HS + 1 <= c]
        js = [j for j in jl if label_inv5(days, inv, j) is not None]
        cal = V3.calibrate(days, k, LI.INV_DIR_DEFAULT, f["cutoff_day"])
        ml = LI.fit(np.vstack([np.asarray([F[j] for j in jl]), pX]), np.concatenate([[label_long5(k, j) for j in jl], pyl]))
        ms = LI.fit(np.vstack([np.asarray([F[j] for j in js]), pX]), np.concatenate([[label_inv5(days, inv, j) for j in js], inv5_from_ratio(pR, cal)]))
        dec = list(range(f["first_dec"], f["last_dec"] + 1))
        Xd = np.asarray([F[i] for i in dec])
        pl, ps = ml.predict(Xd), ms.predict(Xd)
        for i, a, b in zip(dec, pl, ps):
            preds[days[i]] = [float(a), float(b)] + [float(x) for x in F[i][-4:]]
        info.append(dict(year=f["year"], cutoff_day=f["cutoff_day"], real_long=len(jl), real_inv=len(js), synth=len(pX), real_last_label_day=days[jl[-1] + HS + 1], pred_long_mean=float(pl.mean()), pred_long_pos=float((pl > 0).mean())))
        last_models = (ml, ms, f, cal)
    idx = {d: i for i, d in enumerate(days)}
    o = np.asarray([(preds[d][0], label_long5(k, idx[d])) for d in sorted(preds) if idx[d] + HS + 1 < len(days)])
    oos = dict(corr=float(np.corrcoef(o[:, 0], o[:, 1])[0, 1]), sign_hit=float(np.mean((o[:, 0] > 0) == (o[:, 1] > 0))), base=float(np.mean(o[:, 1] > 0)), n=int(len(o)))
    os.makedirs(OUT_MODELS, exist_ok=False)
    files = {}
    for name, m in (("long", last_models[0]), ("inv", last_models[1])):
        b = m.model_to_string().encode("utf-8")
        with open(os.path.join(OUT_MODELS, name + ".txt"), "xb") as fh:
            fh.write(b)
        files[name] = hashlib.sha256(b).hexdigest()
    code = {n: hashlib.sha256(open(os.path.join(HERE, n), "rb").read()).hexdigest() for n in ("short_head_v1.py", "learn_inverse.py", "learn_inverse_synth_v3.py")}
    meta = dict(version="short-models-v1", horizon=HS, features=LI.FEATS + RANGE_FEATS, fold_year=last_models[2]["year"], fit_cutoff_day=last_models[2]["cutoff_day"], calibration=last_models[3], files=files, code_sha256=code)
    with open(os.path.join(OUT_MODELS, "meta.json"), "x", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    rec = dict(version="short-walkforward-predictions-v1", horizon=HS, features=LI.FEATS + RANGE_FEATS, columns=["pred_long5", "pred_inv5"] + RANGE_FEATS, code_sha256=code, folds=info, oos_long5=oos, preds={d: v for d, v in sorted(preds.items())})
    data = json.dumps(rec).encode("utf-8")
    with open(OUT_PRED, "xb") as fh:
        fh.write(data)
    print("oos 5-session", oos)
    for fi in info:
        print(fi["year"], fi["cutoff_day"], "real", fi["real_long"], fi["real_inv"], "last label", fi["real_last_label_day"], "pred mean", round(100 * fi["pred_long_mean"], 3), "pos", round(fi["pred_long_pos"], 2))
    print("saved", OUT_PRED, hashlib.sha256(data).hexdigest(), "models", files)


if __name__ == "__main__":
    main()
