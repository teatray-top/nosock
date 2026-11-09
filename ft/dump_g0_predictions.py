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
import learn_global_v1 as GL

OUT = os.path.join(HERE, "results", "g0_walkforward_predictions_v1.json")


def main():
    days, k, inv = LI.load(LI.INV_DIR_DEFAULT)
    F = LI.features(days, k)
    G = {i: [float("nan")] * len(GL.GLOBAL_FEATS) for i in F}
    man = json.load(open(os.path.join(V3.PATHS, "manifest.json"), encoding="utf-8"))
    train = [V3.load_path(e) for e in man["paths"] if e["split"] == "train"]
    pool_X = np.vstack([p["X"] for p in train])
    pool_yl = np.concatenate([p["yl"] for p in train])
    pool_R = np.concatenate([p["R"] for p in train])
    preds, info, _ = GL.fit_predict(days, k, inv, F, G, pool_X, pool_yl, pool_R, [], False, False)
    if preds["2026-11-05"] != (-0.00542095336063337, 0.025339779548631997):
        raise SystemExit("G0 parity failed")
    stored = json.load(open(os.path.join(HERE, "results", "learn_global_v1.json"), encoding="utf-8"))["folds"]["G0"]
    if [dict(f, rows_long=f["rows_long"]) for f in info] != stored:
        raise SystemExit("fold summaries differ from learn_global_v1 G0")
    rec = dict(version="g0-walkforward-predictions-v1", source="ft/learn_global_v1.py fit_predict G0 (= v3 Ball), yearly folds 2013-2026", code_sha256={n: hashlib.sha256(open(os.path.join(HERE, n), "rb").read()).hexdigest() for n in ("learn_global_v1.py", "learn_inverse_synth_v3.py", "learn_inverse.py")}, parity_2026_11_05=True, folds_match_learn_global_v1=True, preds={d: list(v) for d, v in sorted(preds.items())})
    data = json.dumps(rec).encode("utf-8")
    with open(OUT, "xb") as f:
        f.write(data)
    print(OUT, hashlib.sha256(data).hexdigest(), len(preds))


if __name__ == "__main__":
    main()
