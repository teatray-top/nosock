import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import bisect
import hashlib
import json
import math
import sys
import warnings
from datetime import date

sys.dont_write_bytecode = True
warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import ms_alloc_v3 as A3
import ms_alloc_v9 as A9
import sel_diag_v1 as D1
import sel_diag_v2 as D2
import port_c_v1 as C1
import port_c_v2 as C2

KR_TAX_FREE = {"069500", "229200", "091160", "091170"}
KR_TAX = 0.154
US_TAX, US_DEDUCTION = 0.22, 2_500_000.0
BOOKS = (("U28", "dual"), ("U28", "equal"), ("U36", "topk"), ("U36", "equal"))
VARIANTS = ("fractional", "whole_shares", "whole_shares_tax")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


class World3(C2.World2):
    def __init__(self, *a, raw=None, **kw):
        super().__init__(*a, **kw)
        self.raw = raw

    def ratio(self, c, i):
        ins = self.inst[c]
        rc = self.raw.get(c, {}).get(ins["days"][i - 1]) if i > 0 else None
        ac = ins["C"][i - 1] if i > 0 else None
        return rc / ac if rc and ac else 1.0

    def run3(self, start, end, k_of, mode, whole, tax):
        cash, vault, dep, hwm = 0.0, 0.0, 0.0, 0.0
        units, shares, basis = {}, {}, {}
        us_gain = {}
        taxes = 0.0
        pending = []
        kdays = sorted(d for d in self.days if start <= d <= end and d in self.kr_days)
        reb = set(kdays[::C1.H])
        vault_year, pay_days, wealth, idle = {}, [], [], []
        last_tax_year = None
        for D in sorted(d for d in self.days if start <= d <= end):
            if tax and last_tax_year is not None and D[:4] != last_tax_year:
                g = us_gain.get(last_tax_year, 0.0)
                t = US_TAX * max(0.0, g - US_DEDUCTION)
                cash -= t
                taxes += t
            last_tax_year = D[:4]
            todo = [p for p in pending if p[2] == D]
            pending = [p for p in pending if p[2] != D]
            for phase in ("sell", "buy"):
                for (c, val, _) in todo:
                    ins = self.inst[c]
                    i = self.ipos[c][D]
                    o = ins["O"][i]
                    cur = units.get(c, 0.0)
                    if ins["market"] == "KR" and whole:
                        px = o * self.ratio(c, i)
                        n_cur = shares.get(c, 0)
                        n_tgt = int(math.floor(val / px)) if val > 0 else 0
                        dn = n_tgt - n_cur
                        if (dn < 0) != (phase == "sell") or dn == 0:
                            continue
                        if dn > 0:
                            rate = ins["cost"](i, px)
                            while dn > 0 and dn * px * (1 + rate) > cash:
                                dn -= 1
                            if dn <= 0:
                                continue
                            pay = dn * px * (1 + rate)
                            cash -= pay
                            units[c] = cur + dn * px / o
                            shares[c] = n_cur + dn
                            basis[c] = basis.get(c, 0.0) + pay
                        else:
                            q = -dn
                            frac = q / n_cur
                            proceeds = q * px * (1 - ins["cost"](i, px))
                            b = basis.get(c, 0.0) * frac
                            gain = proceeds - b
                            if tax and c not in KR_TAX_FREE and gain > 0:
                                t = KR_TAX * gain
                                proceeds -= t
                                taxes += t
                            cash += proceeds
                            units[c] = cur * (1 - frac)
                            shares[c] = n_cur - q
                            basis[c] = basis.get(c, 0.0) - b
                    else:
                        nu = val / o
                        du = nu - cur
                        if (du < 0) != (phase == "sell") or abs(du) < 1e-12:
                            continue
                        rate = ins["cost"](i, o)
                        if du > 0:
                            amt = min(du * o * (1 + rate), max(cash, 0.0)) if whole else du * o * (1 + rate)
                            if amt <= 0:
                                continue
                            du = amt / (o * (1 + rate))
                            cash -= amt
                            units[c] = cur + du
                            basis[c] = basis.get(c, 0.0) + amt
                        else:
                            q = -du
                            frac = q / cur
                            proceeds = q * o * (1 - rate)
                            b = basis.get(c, 0.0) * frac
                            gain = proceeds - b
                            if tax:
                                if ins["market"] == "US":
                                    us_gain[D[:4]] = us_gain.get(D[:4], 0.0) + gain
                                elif c not in KR_TAX_FREE and gain > 0:
                                    t = KR_TAX * gain
                                    proceeds -= t
                                    taxes += t
                            cash += proceeds
                            units[c] = cur - q
                            basis[c] = basis.get(c, 0.0) - b
            if D in self.kr_days:
                cash += C1.DEPOSIT
                dep += C1.DEPOSIT
            if D in reb:
                eq = cash + sum(u * self.close_at(c, D)[0] for c, u in units.items() if u)
                P = eq + vault - dep
                if P > hwm:
                    pay = C1.VAULT_SHARE * (P - hwm)
                    hwm = P
                    vault += pay
                    cash -= pay
                    eq -= pay
                    vault_year[D[:4]] = vault_year.get(D[:4], 0.0) + pay
                    pay_days.append(D)
                tw = self.targets(D, k_of(D), mode)
                for c in set(units) | set(tw):
                    o, i, d_exec = self.next_open(c, D)
                    if o is None or d_exec > end:
                        continue
                    pending.append((c, tw.get(c, 0.0) * eq * (o / self.close_at(c, D)[0]), d_exec))
            eqd = cash + sum(u * self.close_at(c, D)[0] for c, u in units.items() if u)
            wealth.append(eqd + vault - dep)
            idle.append(max(cash, 0.0) / eqd if eqd > 0 else 1.0)
        eq = cash + sum(u * self.close_at(c, end)[0] for c, u in units.items() if u)
        P = np.array(wealth)
        dd = float((P - np.maximum.accumulate(P)).min()) if len(P) else 0.0
        return dict(deposits_won=dep, vault_won=vault, acct_won=eq, total_won=eq + vault, taxes_won=taxes, worst_profit_drawdown_won=dd, payouts=len(pay_days), idle_cash_share_mean=float(np.mean(idle)), vault_by_year_won=vault_year, vault_by_period_won=C2.by_period(vault_year))


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
    raw = A3.load_raw()
    man_kus, kus = A9.load_krus()
    for c in man_kus["classes"]:
        r = kus[f"{c}_raw"]
        raw[c] = dict(zip(r["days"], r["C"]))
    out = {}
    for g, mode in BOOKS:
        codes = groups[g]
        prior = json.load(open(os.path.join(root, spec["inputs"]["files"][f"port_c_v2_{g}"]), encoding="utf-8"))["results"][g][mode]
        kby = {int(y): k for y, k in prior["k_by_year"].items()}
        dates = D2.by_date(inst, codes)
        w = World3(inst, codes, ("ewma", 0.94, 0.0), dict(mom=D2.composite(inst, codes, simple, dates)), {c: simple[c]["mom252_21"] for c in codes}, raw=raw)
        res = {}
        for v in VARIANTS:
            r = w.run3("2016-01-01", "2026-11-05", lambda D: kby[int(D[:4])], mode, whole=v != "fractional", tax=v == "whole_shares_tax")
            res[v] = r
            vp = r["vault_by_period_won"]
            print(f"{g} {mode:5s} {v:17s} total {r['total_won'] / 1e4:7.1f}만 vault {r['vault_won'] / 1e4:7.1f}만 (2016-20 {vp['2016-2020'] / 1e4:.0f} | 2021-25 {vp['2021-2025'] / 1e4:.0f} | 2026 {vp['2026'] / 1e4:.0f}) taxes {r['taxes_won'] / 1e4:.1f}만 idle cash {100 * r['idle_cash_share_mean']:.1f}% drawdown {r['worst_profit_drawdown_won'] / 1e4:.1f}만 payouts {r['payouts']}" + (f" | port_c_v2 {prior['total_won'] / 1e4:.1f}만" if v == "fractional" else ""), flush=True)
        out[f"{g}_{mode}"] = res
    rec = dict(version="port-c-v5", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], results=out)
    with open(out_path, "x", encoding="utf-8") as fo:
        json.dump(rec, fo, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
