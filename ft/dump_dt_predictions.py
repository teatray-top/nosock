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
import dt_scale_v1 as D

OUT_DIR = os.path.join(HERE, "results", "dt_scale_v1b_artifacts")


def main():
    res = json.load(open(os.path.join(HERE, "results", "dt_scale_v1b.json"), encoding="utf-8"))
    code = hashlib.sha256(open(D.__file__, "rb").read()).hexdigest()
    if res["code_sha256"] != code:
        raise SystemExit("dt_scale_v1.py changed since the v1b run")
    days, k, inv = D.LI.load(D.LI.INV_DIR_DEFAULT)
    sX, sy = D.synth_training()
    cal = json.load(open(D.CT.CAL, encoding="utf-8"))["days"]
    bars, _ = D.build_bars(D.load_minutes(D.CT.SNAP[D.LONG][0], cal), cal)
    ibars, _ = D.build_bars(D.load_minutes(D.INV_MIN, cal), cal)
    key = lambda b: (b["day"], b["bar"])
    imap = {key(b): b for b in ibars}
    bars = [b for b in bars if key(b) in imap]
    ib = [imap[key(b)] for b in bars]
    bc, ic = [b["close"] for b in bars], [b["close"] for b in ib]
    BX, Bs, Bok = D.norm_features(bc)
    fl, fi = D.next_fill(bars), D.next_fill(ib)
    dX, ds, dok = D.norm_features(k["Ca"])
    os.makedirs(OUT_DIR, exist_ok=False)
    pred_bar, pred_day, models = {}, {}, {}
    for y in D.TEST_YEARS:
        e = next(j for j, d in enumerate(days) if d >= f"{y}-01-01")
        X, yv = D.daily_training(days, k, e - 1)
        m = D.LI.lgb.train(D.LI.PARAMS, D.LI.lgb.Dataset(np.vstack([X, sX]), label=np.concatenate([yv, sy]), free_raw_data=False), num_boost_round=D.LI.ROUNDS)
        b = m.model_to_string().encode("utf-8")
        with open(os.path.join(OUT_DIR, f"model_{y}.txt"), "xb") as fh:
            fh.write(b)
        models[y] = hashlib.sha256(b).hexdigest()
        bi = [i for i, bb in enumerate(bars) if bb["day"][:4] == str(y) and Bok[i]]
        for i, v in zip(bi, m.predict(BX[bi])):
            pred_bar[i] = float(v)
        di = [i for i, d in enumerate(days) if d[:4] == str(y) and dok[i] and d <= D.EVAL_TO]
        for i, v in zip(di, m.predict(dX[di])):
            pred_day[i] = float(v)
    start_bar = next(i for i, bb in enumerate(bars) if bb["day"] >= D.EVAL_FROM)
    thr = lambda i, leg: 2 * (D.LI.FEE + D.LI.TICK / (bc[i] if leg == D.LONG else ic[i])) + D.LI.BAND
    pr = lambda i: pred_bar[i] * Bs[i] * math.sqrt(D.H) if i in pred_bar else None
    check = {}
    for name, kind, ex in (("bar30_A_X0", "A", "X0"), ("bar30_C_X1", "C", "X1")):
        r, _, _ = D.run_harvest(kind, ex, bc, lambda i: Bs[i], pr, fl, fi, ic, thr, len(bars), start_bar)
        check[name] = abs(r["total_pct"] - res["results"][name]["total_pct"]) < 1e-9 and abs(r["vault_pct"] - res["results"][name]["vault_pct"]) < 1e-9
    rows = [dict(bar_index=i, day=bars[i]["day"], bar=bars[i]["bar"], end=bars[i]["end"], close_069500=bc[i], close_114800=ic[i], s=float(Bs[i]), pred_norm=pred_bar.get(i), pred_return=(pred_bar[i] * Bs[i] * math.sqrt(D.H)) if i in pred_bar else None, threshold_long=thr(i, D.LONG), threshold_inv=thr(i, D.INV), next_fill_bar=fl[i][0] if fl[i] else None, next_fill_price=fl[i][1] if fl[i] else None, next_fill_minute=(bars[fl[i][0]]["day"] + " " + bars[fl[i][0]]["fill"][0]) if fl[i] else None) for i in range(len(bars))]
    drows = [dict(day=days[i], s=float(ds[i]), pred_norm=pred_day[i]) for i in sorted(pred_day)]
    rec = dict(version="dt-scale-v1b-artifacts", note="deterministic recomputation after the v1b run (same code sha, LightGBM deterministic, 1 thread); trades in dt_scale_v1b.json carry the execution bar; the decision bar is the bar whose next_fill_bar equals it", code_sha256=code, run_result_sha256=hashlib.sha256(open(os.path.join(HERE, "results", "dt_scale_v1b.json"), "rb").read()).hexdigest(), reproduces=check, models=models, bars=rows, daily=drows)
    data = json.dumps(rec, ensure_ascii=False).encode("utf-8")
    with open(os.path.join(OUT_DIR, "predictions.json"), "xb") as fh:
        fh.write(data)
    print("reproduces", check, "models", models)
    print(os.path.join(OUT_DIR, "predictions.json"), hashlib.sha256(data).hexdigest(), len(rows), len(drows))


if __name__ == "__main__":
    main()
