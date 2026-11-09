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
import ms_alloc_v7 as A7
import sel_diag_v1 as D1
import sel_diag_v2 as D2
import port_c_v1 as C1
import port_c_v2 as C2

SEEDS = 20


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


class Placebo(C2.World2):
    def __init__(self, *a, seed=0, **kw):
        super().__init__(*a, **kw)
        self.rng = np.random.default_rng(seed)

    def targets(self, D, k, mode):
        t = self.pos[D]
        hist = self.live[max(0, t - 260):t + 1].sum(axis=0)
        cs = [self.codes[j] for j in range(len(self.codes)) if hist[j] >= C1.MIN_HIST]
        k = int(k)
        if not cs:
            return {}
        pick = [cs[i] for i in self.rng.choice(len(cs), size=min(k, len(cs)), replace=False)]
        if mode == "dual":
            out = {}
            for c in pick:
                _, i = self.close_at(c, D)
                a = self.absmom[c][i] if i is not None else np.nan
                if not np.isnan(a) and a > 0:
                    out[c] = 1.0 / k
            return out
        return {c: 1.0 / k for c in pick}


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
        prior = json.load(open(os.path.join(os.path.dirname(HERE), spec["inputs"]["files"][f"port_c_v2_{g}"]), encoding="utf-8"))["results"][g]
        dates = D2.by_date(inst, codes)
        scores = dict(mom=D2.composite(inst, codes, simple, dates))
        absmom = {c: simple[c]["mom252_21"] for c in codes}
        res = {}
        for mode in ("topk", "dual"):
            kby = {int(y): k for y, k in prior[mode]["k_by_year"].items()}
            ai = prior[mode]
            tots, pre = [], []
            for s in range(SEEDS):
                w = Placebo(inst, codes, ("ewma", 0.94, 0.0), scores, absmom, seed=8000 + s)
                r = w.run("2016-01-01", "2026-11-05", lambda D: kby[int(D[:4])], mode)
                tots.append(r["total_won"])
                pre.append(sum(v for y, v in r["vault_by_year_won"].items() if y < "2026"))
            ai_pre = sum(v for y, v in ai["vault_by_year_won"].items() if y < "2026")
            res[mode] = dict(momentum_total_won=ai["total_won"], placebo_total_mean_won=float(np.mean(tots)), placebo_total_range_won=[float(min(tots)), float(max(tots))], momentum_beats_total=int(sum(ai["total_won"] > x for x in tots)), momentum_vault_to_2025_won=ai_pre, placebo_vault_to_2025_mean_won=float(np.mean(pre)), momentum_beats_vault_to_2025=int(sum(ai_pre > x for x in pre)))
            x = res[mode]
            print(f"{g} {mode}: momentum total {x['momentum_total_won'] / 1e4:.0f}만 vs random-K {x['placebo_total_mean_won'] / 1e4:.0f}만 (range {[round(v / 1e4) for v in x['placebo_total_range_won']]}), beats {x['momentum_beats_total']}/{SEEDS}; vault to 2025 {x['momentum_vault_to_2025_won'] / 1e4:.0f}만 vs {x['placebo_vault_to_2025_mean_won'] / 1e4:.0f}만, beats {x['momentum_beats_vault_to_2025']}/{SEEDS}", flush=True)
        out[g] = res
    if spec.get("kis_cost_rerun"):
        g = "U36"
        codes = groups[g]
        for c in codes:
            if inst[c]["market"] == "US":
                inst[c]["cost"] = A7.us_cost_maker(spec["kis_cost_rerun"]["fee"], spec["kis_cost_rerun"]["fx"])(inst[c])
        dates = D2.by_date(inst, codes)
        scores = dict(mom=D2.composite(inst, codes, simple, dates))
        absmom = {c: simple[c]["mom252_21"] for c in codes}
        world = C2.World2(inst, codes, ("ewma", 0.94, 0.0), scores, absmom)
        kis = {}
        for mode in ("equal", "topk", "dual"):
            kby = {y: 1 for y in C1.TEST_YEARS} if mode == "equal" else C2.choose_k(world, mode)
            r = world.run("2016-01-01", "2026-11-05", lambda D: kby[int(D[:4])], mode)
            r["k_by_year"] = kby
            r["vault_by_period_won"] = C2.by_period(r["vault_by_year_won"])
            kis[mode] = r
            vp = r["vault_by_period_won"]
            print(f"U36 KIS costs {mode:6s} total {r['total_won'] / 1e4:.1f}만 vault {r['vault_won'] / 1e4:.1f}만 (2016-20 {vp['2016-2020'] / 1e4:.0f} | 2021-25 {vp['2021-2025'] / 1e4:.0f} | 2026 {vp['2026'] / 1e4:.0f}) worst profit drawdown {r['worst_profit_drawdown_won'] / 1e4:.1f}만 payouts {r['payouts']}", flush=True)
        out["U36_kis_costs"] = kis
    rec = dict(version="port-c-v3", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], results=out)
    with open(out_path, "x", encoding="utf-8") as fo:
        json.dump(rec, fo, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
