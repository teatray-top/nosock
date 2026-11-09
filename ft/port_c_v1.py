import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import bisect
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
import sel_diag_v2 as D2
import risk_diag_v1 as RB

H = 20
DEPOSIT = 10_000.0
VAULT_SHARE = 0.5
S0 = 0.3 * math.sqrt(H / 252)
IC = 0.05
LAMBDAS = (1.0, 2.0, 3.0, 5.0, 10.0, 20.0)
LOOKBACK_YEARS = 3
MIN_HIST = 250
TEST_YEARS = list(range(2016, 2027))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def proj(v):
    w = np.maximum(v, 0.0)
    if w.sum() <= 1.0:
        return w
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    k = np.nonzero(u - (css - 1.0) / (np.arange(len(u)) + 1) > 0)[0][-1]
    tau = (css[k] - 1.0) / (k + 1)
    return np.maximum(v - tau, 0.0)


def solve(mu, S, lam, iters=3000):
    L = lam * max(np.linalg.eigvalsh(S).max(), 1e-12)
    eta = 1.0 / L
    w = np.zeros(len(mu))
    for _ in range(iters):
        w2 = proj(w + eta * (mu - lam * S @ w))
        if np.abs(w2 - w).max() < 1e-10:
            w = w2
            break
        w = w2
    return w


class World:
    def __init__(self, inst, codes, est, scores):
        self.inst, self.codes = inst, codes
        self.days, self.Rm, self.live = RB.returns_matrix(inst, codes)
        self.pos = {d: i for i, d in enumerate(self.days)}
        self.est = est
        self.scores = scores
        self.kr_days = set(inst["069500"]["days"])
        self.ipos = {c: {d: i for i, d in enumerate(inst[c]["days"])} for c in codes}

    def close_at(self, c, D):
        ins = self.inst[c]
        i = bisect.bisect_right(ins["days"], D) - 1
        return (ins["C"][i], i) if i >= 0 else (None, None)

    def next_open(self, c, D):
        ins = self.inst[c]
        i = bisect.bisect_right(ins["days"], D)
        return (ins["O"][i], i, ins["days"][i]) if i < len(ins["days"]) else (None, None, None)

    def targets(self, D, lam, mode):
        t = self.pos[D]
        hist = self.live[max(0, t - 260):t + 1].sum(axis=0)
        idx = [j for j in range(len(self.codes)) if hist[j] >= MIN_HIST]
        if len(idx) < 2:
            return {}
        kind, l2, shrink = self.est
        S = RB.cov_at(self.Rm[:, idx], t, kind, l2, shrink) * H
        sig = np.sqrt(np.maximum(np.diag(S), 1e-12))
        cs = [self.codes[j] for j in idx]
        if mode == "equal":
            return {c: 1.0 / len(cs) for c in cs}
        z = np.zeros(len(cs))
        if mode not in ("s0",):
            raw = []
            for c in cs:
                _, i = self.close_at(c, D)
                v = self.scores[mode][c][i] if i is not None else np.nan
                raw.append(np.nan if v is None else v)
            raw = np.array(raw, float)
            ok = ~np.isnan(raw)
            if ok.sum() >= 3:
                z[ok] = np.clip((raw[ok] - raw[ok].mean()) / (raw[ok].std() + 1e-12), -3, 3)
        if mode.startswith("top2_"):
            o = np.argsort(-z)[:2]
            return {cs[k]: 0.5 for k in o}
        mu = sig * (S0 + IC * z)
        w = solve(mu, S, lam)
        return {c: float(x) for c, x in zip(cs, w) if x > 1e-6}

    def run(self, start, end, lam_of, mode):
        cash, vault, dep, hwm = 0.0, 0.0, 0.0, 0.0
        units = {}
        pending = []
        kdays = sorted(d for d in self.days if start <= d <= end and d in self.kr_days)
        reb = set(kdays[::H])
        vault_year, pay_days, wealth = {}, [], []
        invested = []
        for D in sorted(d for d in self.days if start <= d <= end):
            keep = []
            for (c, val, d_exec) in pending:
                if d_exec != D:
                    keep.append((c, val, d_exec))
                    continue
                ins = self.inst[c]
                i = self.ipos[c][D]
                o = ins["O"][i]
                nu = val / o
                du = nu - units.get(c, 0.0)
                cost = abs(du) * o * ins["cost"](i, o)
                cash -= du * o + cost
                units[c] = nu
            pending = keep
            if D in self.kr_days:
                cash += DEPOSIT
                dep += DEPOSIT
            if D in reb:
                eq = cash + sum(u * self.close_at(c, D)[0] for c, u in units.items() if u)
                P = eq + vault - dep
                if P > hwm:
                    pay = VAULT_SHARE * (P - hwm)
                    hwm = P
                    vault += pay
                    cash -= pay
                    eq -= pay
                    vault_year[D[:4]] = vault_year.get(D[:4], 0.0) + pay
                    pay_days.append(D)
                tw = self.targets(D, lam_of(D), mode)
                for c in set(units) | set(tw):
                    o, i, d_exec = self.next_open(c, D)
                    if o is None or d_exec > end:
                        continue
                    pending.append((c, tw.get(c, 0.0) * eq * (o / self.close_at(c, D)[0]), d_exec))
                invested.append(sum(tw.values()))
            eqd = cash + sum(u * self.close_at(c, D)[0] for c, u in units.items() if u)
            wealth.append((D, eqd + vault - dep))
        eq = cash + sum(u * self.close_at(c, end)[0] for c, u in units.items() if u)
        P = np.array([w for _, w in wealth])
        dd = float((P - np.maximum.accumulate(P)).min()) if len(P) else 0.0
        gaps = []
        for a, b in zip(pay_days[:-1], pay_days[1:]):
            gaps.append(sum(1 for d in kdays if a < d <= b))
        return dict(deposits_won=dep, vault_won=vault, acct_won=eq, total_won=eq + vault, gain_over_deposits_pct=100 * ((eq + vault) / dep - 1) if dep else None, worst_profit_drawdown_won=dd, payouts=len(pay_days), longest_sessions_between_payouts=max(gaps) if gaps else None, invested_share_mean=float(np.mean(invested)) if invested else None, vault_by_year_won=vault_year)


def choose_lams(world, mode):
    lam_by_year = {}
    for y in TEST_YEARS:
        best, best_v = None, -1e18
        for lam in LAMBDAS:
            r = world.run(f"{y - LOOKBACK_YEARS}-01-01", f"{y - 1}-12-31", lambda D, lam=lam: lam, mode)
            v = r["total_won"]
            if v > best_v + 1e-6 or (abs(v - best_v) <= 1e-6 and best is not None and lam > best):
                best, best_v = lam, v
        lam_by_year[y] = best
    return lam_by_year


def load_kappa(path, codes, inst):
    d = json.load(open(path, encoding="utf-8"))["scores"]
    out = {}
    for g in d.values():
        for c, v in g.items():
            if c in codes and c not in out:
                out[c] = np.array([np.nan if x is None else x for x in v["q_hold20"]], float)
    return out


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, pth in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), pth)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    est = tuple(spec["risk_estimator"])
    modes = spec["modes"]
    inst, groups = D1.build_all()
    simple = {c: D1.simple_scores(ins) for c, ins in inst.items()}
    out = {}
    for g in spec["universes"]:
        codes = groups[g]
        dates = D2.by_date(inst, codes)
        scores = dict(mom=D2.composite(inst, codes, simple, dates))
        scores["top2_mom"] = scores["mom"]
        if "kappa" in modes or "top2_kappa" in modes:
            kp = os.path.join(os.path.dirname(HERE), spec["inputs"]["files"]["kappa_scores"])
            per_g = json.load(open(kp, encoding="utf-8"))["scores"][g]
            scores["kappa"] = {c: np.array([np.nan if x is None else x for x in per_g[c]["q_hold20"]], float) for c in codes}
            scores["top2_kappa"] = scores["kappa"]
        world = World(inst, codes, est, scores)
        res = {}
        for mode in modes:
            if mode in ("equal",) or mode.startswith("top2_"):
                lam_by_year = {y: 1.0 for y in TEST_YEARS}
            else:
                lam_by_year = choose_lams(world, mode)
            r = world.run("2016-01-01", "2026-11-05", lambda D: lam_by_year[int(D[:4])], mode)
            r["lambda_by_year"] = lam_by_year
            res[mode] = r
            print(f"{g} {mode:12s} deposits {r['deposits_won'] / 1e4:.0f}만 vault {r['vault_won'] / 1e4:.1f}만 acct {r['acct_won'] / 1e4:.1f}만 total {r['total_won'] / 1e4:.1f}만 ({r['gain_over_deposits_pct']:+.1f}%) worst profit drawdown {r['worst_profit_drawdown_won'] / 1e4:.1f}만 payouts {r['payouts']} longest gap {r['longest_sessions_between_payouts']} sessions invested {100 * (r['invested_share_mean'] or 0):.0f}% lambda {sorted(set(lam_by_year.values()))}", flush=True)
            print("      vault by year", {y: round(v / 1e4) for y, v in sorted(r["vault_by_year_won"].items())}, flush=True)
        out[g] = res
    rec = dict(version="port-c-v1", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], results=out)
    with open(out_path, "x", encoding="utf-8") as fo:
        json.dump(rec, fo, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
