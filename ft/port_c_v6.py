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
import ms_alloc_v3 as A3
import sel_diag_v1 as D1
import sel_diag_v2 as D2
import port_c_v1 as C1
import port_c_v2 as C2
import port_c_v5 as C5

KR_DOMESTIC = ["069500", "229200", "091160", "091170", "114800", "148070", "114260", "153130"]


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
    inst, groups = D1.build_all()
    us = [c for c in groups["U36"] if inst[c]["market"] == "US"]
    codes = KR_DOMESTIC + us
    simple = {c: D1.simple_scores(inst[c]) for c in codes}
    dates = D2.by_date(inst, codes)
    raw = A3.load_raw()
    w = C5.World3(inst, codes, ("ewma", 0.94, 0.0), dict(mom=D2.composite(inst, codes, simple, dates)), {c: simple[c]["mom252_21"] for c in codes}, raw=raw)
    res = {}
    for mode in ("equal", "topk", "dual"):
        kby = {y: 1 for y in C1.TEST_YEARS} if mode == "equal" else C2.choose_k(w, mode)
        frac = w.run("2016-01-01", "2026-11-05", lambda D: kby[int(D[:4])], mode)
        real = w.run3("2016-01-01", "2026-11-05", lambda D: kby[int(D[:4])], mode, whole=True, tax=True)
        res[mode] = dict(k_by_year=kby, fractional=frac, realistic=real)
        for name, r in (("fractional", frac), ("realistic", real)):
            vp = C2.by_period(r["vault_by_year_won"])
            print(f"U29 {mode:5s} {name:10s} total {r['total_won'] / 1e4:7.1f}만 vault {r['vault_won'] / 1e4:7.1f}만 (2016-20 {vp['2016-2020'] / 1e4:.0f} | 2021-25 {vp['2021-2025'] / 1e4:.0f} | 2026 {vp['2026'] / 1e4:.0f}) drawdown {r['worst_profit_drawdown_won'] / 1e4:.1f}만 payouts {r['payouts']}" + (f" taxes {r['taxes_won'] / 1e4:.1f}만" if name == "realistic" else "") + f" K {[kby[y] for y in C1.TEST_YEARS]}", flush=True)
    rec = dict(version="port-c-v6", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], universe=codes, results=res)
    with open(out_path, "x", encoding="utf-8") as fo:
        json.dump(rec, fo, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
