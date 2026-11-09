import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import learn_inverse as LI
import learn_inverse_synth_v3 as V3

RESULT = os.path.join(HERE, "results", "learn_inverse_synth_v3.json")
OUT = os.path.join(HERE, "results", "learn_shadow_models_v1")


def sha_b(b):
    return hashlib.sha256(b).hexdigest()


def main():
    res = json.load(open(RESULT, encoding="utf-8"))
    if res["code_sha256"] != sha_b(open(V3.__file__, "rb").read()):
        raise SystemExit("v3 code changed since the run")
    days, k, inv = LI.load(LI.INV_DIR_DEFAULT)
    man = json.load(open(os.path.join(V3.PATHS, "manifest.json"), encoding="utf-8"))
    train = [V3.load_path(e) for e in man["paths"] if e["split"] == "train"]
    pool_X = np.vstack([p["X"] for p in train])
    pool_yl = np.concatenate([p["yl"] for p in train])
    pool_R = np.concatenate([p["R"] for p in train])
    F = LI.features(days, k)
    f = [x for x in LI.folds(days) if x["year"] == LI.LAST_YEAR][0]
    jl = list(range(LI.FEAT_START, f["train_last"] + 1))
    js = [j for j in jl if LI.label_inv(days, inv, j) is not None]
    cal = V3.calibrate(days, k, LI.INV_DIR_DEFAULT, f["cutoff_day"])
    if cal != res["calibration_per_fold"][str(LI.LAST_YEAR)]:
        raise SystemExit("calibration differs from the v3 run")
    pool_yi = V3.inv_labels_from_ratio(pool_R, cal)
    a = np.arange(len(pool_X))
    ml = LI.fit(np.vstack([np.asarray([F[j] for j in jl]), pool_X[a]]), np.concatenate([[LI.label_long(k, j) for j in jl], pool_yl[a]]))
    ms = LI.fit(np.vstack([np.asarray([F[j] for j in js]), pool_X[a]]), np.concatenate([[LI.label_inv(days, inv, j) for j in js], pool_yi[a]]))
    last = len(days) - 1
    pl, ps = float(ml.predict(np.asarray([F[last]]))[0]), float(ms.predict(np.asarray([F[last]]))[0])
    want = res["latest"]["Ball"]
    if want["day"] != days[last] or pl != want["pred"][0] or ps != want["pred"][1]:
        raise SystemExit(f"prediction mismatch: {pl} {ps} vs {want}")
    os.makedirs(OUT, exist_ok=False)
    meta = dict(version="learn-shadow-models-v1", source_run=dict(file="ft/results/learn_inverse_synth_v3.json", sha256=sha_b(open(RESULT, "rb").read()), code_sha256=res["code_sha256"]), variant="Ball (real rows + all 340,020 synthetic training rows)", fold_year=LI.LAST_YEAR, fit_cutoff_day=f["cutoff_day"], real_rows_long=len(jl), real_rows_inv=len(js), calibration=cal, features=LI.FEATS, horizon=LI.H, band=LI.BAND, check=dict(day=days[last], pred_long=pl, pred_inv=ps, equals_v3_latest=True), files={})
    for name, m in (("long", ml), ("inv", ms)):
        b = m.model_to_string().encode("utf-8")
        with open(os.path.join(OUT, name + ".txt"), "xb") as fh:
            fh.write(b)
        meta["files"][name] = dict(file=name + ".txt", sha256=sha_b(b))
    data = json.dumps(meta, ensure_ascii=False, indent=1).encode("utf-8")
    with open(os.path.join(OUT, "meta.json"), "xb") as fh:
        fh.write(data)
    print("saved", OUT, "meta", sha_b(data), meta["files"], meta["check"])


if __name__ == "__main__":
    main()
