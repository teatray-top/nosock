import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
import sys
import warnings

sys.dont_write_bytecode = True
warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import sel_diag_v1 as D1
import port_c_v2 as C2


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, pth in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), pth)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    root = os.path.dirname(HERE)
    inst, groups = D1.build_all()
    simple = {c: D1.simple_scores(ins) for c, ins in inst.items()}
    ks = json.load(open(os.path.join(root, spec["inputs"]["files"]["kappa_scores"]), encoding="utf-8"))["scores"]
    out, wins = {}, []
    for g in ("U15", "U28", "U36"):
        codes = groups[g]
        prior = json.load(open(os.path.join(root, spec["inputs"]["files"][f"port_c_v2_{g}"]), encoding="utf-8"))["results"][g]
        kap = {}
        for c in codes:
            assert ks[g][c]["days"] == inst[c]["days"], c
            kap[c] = np.array([np.nan if x is None else x for x in ks[g][c]["q_hold20"]], float)
        absmom = {c: simple[c]["mom252_21"] for c in codes}
        w = C2.World2(inst, codes, ("ewma", 0.94, 0.0), dict(mom=kap), absmom)
        res = {}
        for mode in ("topk", "dual"):
            kby = {int(y): k for y, k in prior[mode]["k_by_year"].items()}
            r = w.run("2016-01-01", "2026-11-05", lambda D: kby[int(D[:4])], mode)
            vp = C2.by_period(r["vault_by_year_won"])
            m = prior[mode]
            mp = C2.by_period(m["vault_by_year_won"])
            pre_k = vp["2016-2020"] + vp["2021-2025"]
            pre_m = mp["2016-2020"] + mp["2021-2025"]
            win = r["total_won"] > m["total_won"] and pre_k > pre_m
            wins.append(win)
            res[mode] = dict(kappa=r, kappa_vault_by_period_won=vp, momentum_total_won=m["total_won"], momentum_vault_to_2025_won=pre_m, kappa_vault_to_2025_won=pre_k, kappa_wins=win)
            print(f"{g} {mode}: kappa total {r['total_won'] / 1e4:.1f}만 vs momentum {m['total_won'] / 1e4:.1f}만 | vault to 2025 {pre_k / 1e4:.0f}만 vs {pre_m / 1e4:.0f}만 | kappa drawdown {r['worst_profit_drawdown_won'] / 1e4:.1f}만 vs {m['worst_profit_drawdown_won'] / 1e4:.1f}만 | payouts {r['payouts']} vs {m['payouts']} -> {'kappa wins' if win else 'momentum stays'}", flush=True)
        out[g] = res
    verdict = "replace momentum with kappa" if all(wins) else "keep momentum"
    print("VERDICT:", verdict, f"({sum(wins)}/{len(wins)} universe-mode pairs won by kappa)", flush=True)
    rec = dict(version="port-c-v4", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], results=out, verdict=verdict)
    with open(out_path, "x", encoding="utf-8") as fo:
        json.dump(rec, fo, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
