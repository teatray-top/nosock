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
import msma_v1 as M1
import msma_v2 as M2

VARIANTS = [
    dict(name="W1_score_over_sd_cap30", p=1.0, cap=0.30, exclude=()),
    dict(name="W2_score_over_sd_cap30_no_cash_bond3y", p=1.0, cap=0.30, exclude=("153130", "114260")),
    dict(name="W3_score_over_sqrt_sd_cap30", p=0.5, cap=0.30, exclude=()),
    dict(name="W4_score_only_cap30", p=0.0, cap=0.30, exclude=()),
    dict(name="W5_score_only_cap50", p=0.0, cap=0.50, exclude=()),
    dict(name="W6_score_only_cap30_no_cash_bond3y", p=0.0, cap=0.30, exclude=("153130", "114260")),
]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def make_weights(codes, p, cap, exclude):
    ex = np.array([c in exclude for c in codes])

    def tw(i, avail, score, sd60, r, m, vol, rule):
        w = np.zeros(m)
        if rule != "score":
            return M2.target_weights.__wrapped__(i, avail, score, sd60, r, m, vol, rule) if hasattr(M2.target_weights, "__wrapped__") else w
        sc = np.where(avail & ~ex, score[i], np.nan)
        pos = np.nan_to_num(sc) > 0
        if not pos.any():
            return w
        raw = np.where(pos, np.nan_to_num(sc), 0.0) / np.where(pos, sd60[i] ** p, 1.0)
        w = raw / raw.sum()
        for _ in range(8):
            over = w > cap
            if not over.any():
                break
            exs = (w[over] - cap).sum()
            w[over] = cap
            under = (~over) & (w > 0)
            if under.any():
                w[under] += exs * w[under] / w[under].sum()
        return w / max(1.0, w.sum())
    return tw


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, pth in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), pth)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    days, codes, O, C = M1.load()
    S, sd60, r = M1.signals(days, C)
    eq = np.nanmean(np.tanh(np.stack([S[:, :, M1.SIG.index(s)] for s in M1.EQ_SIGNALS], axis=2)), axis=2)
    start = int(np.argmax(days >= M1.TEST_FROM)) - 1
    res = {}
    orig = M2.target_weights
    for v in VARIANTS:
        for vault in (False, True):
            M2.target_weights = make_weights(codes, v["p"], v["cap"], v["exclude"])
            name = v["name"] + ("_vault" if vault else "")
            res[name] = M2.run(days, codes, O, C, sd60, r, eq, start, "score", 1.0, 0.02, vault)
            x = res[name]
            print(f"{name:44s} total {x['total_pct']:8.1f}% cagr {x['cagr_pct']:5.2f}% sharpe {x['sharpe']:.2f} mdd {x['mdd_pct']:6.1f} lowest {x['lowest_pct']:6.1f} cost {x['cost_pct']:.2f}% vault {x['vault_pct']:.1f}% exposure {x['mean_exposure']:.2f} losing {x['losing_years']}", flush=True)
            print("    years", {y: round(float(z), 1) for y, z in x["years"].items()}, flush=True)
    M2.target_weights = orig
    rec = dict(version="msma-v3", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], variants=VARIANTS, results=res)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
