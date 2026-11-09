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
import sel_diag_v2 as D2
import port_c_v1 as C1

KS = (2, 3, 5, 8)
PERIODS = (("2016-2020", "2016", "2020"), ("2021-2025", "2021", "2025"), ("2026", "2026", "2026"))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


class World2(C1.World):
    def __init__(self, inst, codes, est, scores, absmom):
        super().__init__(inst, codes, est, scores)
        self.absmom = absmom

    def targets(self, D, k, mode):
        if mode == "equal":
            return super().targets(D, 1.0, "equal")
        t = self.pos[D]
        hist = self.live[max(0, t - 260):t + 1].sum(axis=0)
        cs = [self.codes[j] for j in range(len(self.codes)) if hist[j] >= C1.MIN_HIST]
        sc = []
        for c in cs:
            _, i = self.close_at(c, D)
            v = self.scores["mom"][c][i] if i is not None else np.nan
            a = self.absmom[c][i] if i is not None else np.nan
            sc.append((c, v, a))
        sc = [x for x in sc if not np.isnan(x[1])]
        if not sc:
            return {}
        sc.sort(key=lambda x: -x[1])
        k = int(k)
        top = sc[:k]
        if mode == "dual":
            return {c: 1.0 / k for c, _, a in top if not np.isnan(a) and a > 0}
        return {c: 1.0 / k for c, _, _ in top}


def choose_k(world, mode):
    out = {}
    for y in C1.TEST_YEARS:
        best, best_v = None, -1e18
        for k in KS:
            r = world.run(f"{y - C1.LOOKBACK_YEARS}-01-01", f"{y - 1}-12-31", lambda D, k=k: k, mode)
            if r["total_won"] > best_v + 1e-6:
                best, best_v = k, r["total_won"]
        out[y] = best
    return out


def by_period(vy):
    return {name: sum(v for y, v in vy.items() if a <= y <= b) for name, a, b in PERIODS}


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
    simple = {c: D1.simple_scores(ins) for c, ins in inst.items()}
    out = {}
    for g in spec["universes"]:
        codes = groups[g]
        dates = D2.by_date(inst, codes)
        scores = dict(mom=D2.composite(inst, codes, simple, dates))
        absmom = {c: simple[c]["mom252_21"] for c in codes}
        world = World2(inst, codes, ("ewma", 0.94, 0.0), scores, absmom)
        res = {}
        for mode in ("equal", "topk", "dual"):
            kby = {y: 1 for y in C1.TEST_YEARS} if mode == "equal" else choose_k(world, mode)
            r = world.run("2016-01-01", "2026-11-05", lambda D: kby[int(D[:4])], mode)
            r["k_by_year"] = kby
            r["vault_by_period_won"] = by_period(r["vault_by_year_won"])
            res[mode] = r
            vp = r["vault_by_period_won"]
            print(f"{g} {mode:6s} total {r['total_won'] / 1e4:.1f}만 vault {r['vault_won'] / 1e4:.1f}만 (2016-20 {vp['2016-2020'] / 1e4:.0f} | 2021-25 {vp['2021-2025'] / 1e4:.0f} | 2026 {vp['2026'] / 1e4:.0f}) acct {r['acct_won'] / 1e4:.1f}만 worst profit drawdown {r['worst_profit_drawdown_won'] / 1e4:.1f}만 payouts {r['payouts']} longest gap {r['longest_sessions_between_payouts']} sessions invested {100 * (r['invested_share_mean'] or 0):.0f}% K {[kby[y] for y in C1.TEST_YEARS]}", flush=True)
        out[g] = res
    rec = dict(version="port-c-v2", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], results=out)
    with open(out_path, "x", encoding="utf-8") as fo:
        json.dump(rec, fo, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
