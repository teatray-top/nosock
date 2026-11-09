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
import sel_diag_v1 as D1

H = 20
START = "2016-01-01"
N_RANDOM = 50
ESTIMATORS = [("ewma", 0.94, 0.0), ("ewma", 0.94, 0.3), ("ewma", 0.97, 0.0), ("ewma", 0.97, 0.3), ("sample60", None, 0.0), ("sample60", None, 0.3)]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def returns_matrix(inst, codes):
    days = sorted(set(d for c in codes for d in inst[c]["days"]))
    pos = {d: i for i, d in enumerate(days)}
    Rm = np.zeros((len(days), len(codes)))
    live = np.zeros((len(days), len(codes)), dtype=bool)
    for j, c in enumerate(codes):
        ins = inst[c]
        lc = np.log(ins["C"])
        for i in range(1, len(ins["days"])):
            k = pos[ins["days"][i]]
            Rm[k, j] = lc[i] - lc[i - 1]
            live[k, j] = True
    return days, Rm, live


def cov_at(Rm, t, kind, lam, shrink):
    if kind == "ewma":
        n = min(t + 1, 500)
        X = Rm[t + 1 - n:t + 1]
        w = lam ** np.arange(n - 1, -1, -1)
        w = w / w.sum()
        S = (X * w[:, None]).T @ X
    else:
        X = Rm[t - 59:t + 1]
        S = np.cov(X, rowvar=False, bias=False)
    d = np.diag(np.diag(S))
    return (1 - shrink) * S + shrink * d


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, pth in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), pth)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    inst, groups = D1.build_all()
    out = {}
    for g, codes in groups.items():
        days, Rm, live = returns_matrix(inst, codes)
        rng = np.random.default_rng(7)
        t0 = next(i for i, d in enumerate(days) if d >= START)
        pts = list(range(t0, len(days) - H, H))
        res = {}
        for kind, lam, shrink in ESTIMATORS:
            name = f"{kind}{'' if lam is None else lam}_shrink{shrink}"
            z_all, ql_all, z_eq, z_iv = [], [], [], []
            for t in pts:
                avail = live[max(0, t - 260):t + 1].sum(axis=0) >= 250
                idx = np.nonzero(avail)[0]
                if len(idx) < 3:
                    continue
                S = cov_at(Rm[:, idx], t, kind, lam, shrink) * H
                fut = Rm[t + 1:t + 1 + H][:, idx].sum(axis=0)
                ports = [np.full(len(idx), 1.0 / len(idx))]
                iv = 1.0 / np.sqrt(np.maximum(np.diag(S), 1e-12))
                ports.append(iv / iv.sum())
                for _ in range(N_RANDOM):
                    ports.append(rng.dirichlet(np.ones(len(idx))))
                for k, w in enumerate(ports):
                    var = float(w @ S @ w)
                    if var <= 0:
                        continue
                    r = float(w @ fut)
                    z = r / math.sqrt(var)
                    ql = math.log(var) + r * r / var
                    z_all.append(z)
                    ql_all.append(ql)
                    if k == 0:
                        z_eq.append(z)
                    elif k == 1:
                        z_iv.append(z)
            res[name] = dict(bias_all=float(np.std(z_all)), bias_equal_weight=float(np.std(z_eq)), bias_inverse_vol=float(np.std(z_iv)), qlike=float(np.mean(ql_all)), points=len(pts))
        out[g] = res
        best = min(res, key=lambda k: res[k]["qlike"])
        print(g, "instruments", len(codes), "best by QLIKE:", best, flush=True)
        for k, v in res.items():
            print(f"   {k:22s} bias(all) {v['bias_all']:.3f} equal-weight {v['bias_equal_weight']:.3f} inverse-vol {v['bias_inverse_vol']:.3f} QLIKE {v['qlike']:+.4f}", flush=True)
    rank = {}
    for g in out:
        for i, k in enumerate(sorted(out[g], key=lambda k: out[g][k]["qlike"])):
            rank[k] = rank.get(k, 0) + i
    chosen = min(rank, key=rank.get)
    print("chosen (lowest summed QLIKE rank over U15/U28/U36):", chosen, flush=True)
    rec = dict(version="risk-diag-v1", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], results=out, chosen=chosen)
    with open(out_path, "x", encoding="utf-8") as fo:
        json.dump(rec, fo, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
