import bisect
import math
from datetime import date, timedelta
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ft"))
import numpy as np

H = 20
DEPOSIT = 10_000.0
VAULT_SHARE = 0.5
MIN_HIST = 250
KR_TAX_FREE = {"069500", "229200", "091160", "091170"}
KR_TAX = 0.154
US_TAX, US_DEDUCTION = 0.22, 2_500_000.0
KS = (2, 3, 5, 8)
LOOKBACK_YEARS = 3


class Market:
    def __init__(self, inst, codes, names=None, kr_calendar="069500"):
        self.inst = inst
        self.codes = list(codes)
        self.names = names or {}
        self.days = sorted(set(d for c in self.codes for d in inst[c]["days"]))
        self.kr_days = set(inst[kr_calendar]["days"])
        self.ipos = {c: {d: i for i, d in enumerate(inst[c]["days"])} for c in self.codes}
        self.pos = {d: i for i, d in enumerate(self.days)}
        live = np.zeros((len(self.days), len(self.codes)), dtype=bool)
        for j, c in enumerate(self.codes):
            for d in inst[c]["days"][1:]:
                live[self.pos[d], j] = True
        self.live = live
        self.raw = {}

    def close_at(self, c, D):
        ins = self.inst[c]
        i = bisect.bisect_right(ins["days"], D) - 1
        return (ins["C"][i], i) if i >= 0 else (None, None)

    def next_open(self, c, D):
        ins = self.inst[c]
        i = bisect.bisect_right(ins["days"], D)
        return (ins["O"][i], i, ins["days"][i]) if i < len(ins["days"]) else (None, None, None)

    def eligible(self, D):
        out = []
        lim = (date.fromisoformat(D) - timedelta(days=7)).isoformat()
        for c in self.codes:
            days = self.inst[c]["days"]
            i = bisect.bisect_right(days, D) - 1
            if i >= MIN_HIST and days[i] >= lim:
                out.append(c)
        return out

    def won_ratio(self, c, i):
        ins = self.inst[c]
        if i <= 0:
            return 1.0
        rc = self.raw.get(c, {}).get(ins["days"][i - 1])
        ac = ins["C"][i - 1]
        return rc / ac if rc and ac else 1.0


class Momentum:
    def __init__(self, market, score, absmom, mom126, mode):
        self.m = market
        self.score = score
        self.absmom = absmom
        self.mom126 = mom126
        self.mode = mode

    def decide(self, D, k):
        m = self.m
        if self.mode == "equal":
            cs = m.eligible(D)
            return ({c: 1.0 / len(cs) for c in cs} if cs else {}), dict(rule="equal", k=len(cs), candidates=[dict(code=c, name=m.names.get(c, c), selected=True) for c in cs])
        rows = []
        for c in m.eligible(D):
            _, i = m.close_at(c, D)
            v = self.score[c][i] if i is not None else np.nan
            a = self.absmom[c][i] if i is not None else np.nan
            r6 = self.mom126[c][i] if i is not None else np.nan
            rows.append((c, v, a, r6))
        rows = [x for x in rows if not np.isnan(x[1])]
        rows.sort(key=lambda x: -x[1])
        k = int(k)
        w, cand = {}, []
        for rank, (c, v, a, r6) in enumerate(rows, 1):
            top = rank <= k
            passed = not (self.mode == "dual") or (not np.isnan(a) and a > 0)
            pick = top and passed
            if pick:
                w[c] = 1.0 / k
            why = "상위 K 안" if top else "순위 밖"
            if top and not passed:
                why = "상위 K지만 12-1개월 수익 ≤ 0 → 현금"
            cand.append(dict(code=c, name=m.names.get(c, c), market=m.inst[c]["market"], rank=rank, z=float(v), mom126_pct=None if np.isnan(r6) else 100 * float(math.expm1(r6)), mom12_1_pct=None if np.isnan(a) else 100 * float(math.expm1(a)), selected=pick, reason=why))
        return w, dict(rule=self.mode, k=k, candidates=cand, cash_weight=max(0.0, 1.0 - sum(w.values())))


def run(market, strategy, start, end, k_of, whole=True, tax=True, record=True, deposit=DEPOSIT, overlay=None, vault_rule="mark"):
    m = market
    cash, vault, dep, hwm = 0.0, 0.0, 0.0, 0.0
    units, shares, basis = {}, {}, {}
    us_gain = {}
    taxes = 0.0
    pending = []
    rN, rM, us_acc = 0.0, 0.0, {}

    def realize_(D, g):
        nonlocal rN, rM, vault, cash
        rN += g
        if rN > rM:
            pay = VAULT_SHARE * (rN - rM)
            payouts.append(dict(day=D, amount=pay, profit=rN, previous_high=rM))
            rM = rN
            vault += pay
            cash -= pay
    kdays = sorted(d for d in m.days if start <= d <= end and d in m.kr_days)
    reb = set(kdays[::H])
    marks, decisions, fills, payouts, exits = [], [], [], [], []
    last_year = None
    for D in sorted(d for d in m.days if start <= d <= end):
        if tax and last_year is not None and D[:4] != last_year:
            g = us_gain.get(last_year, 0.0)
            t = US_TAX * max(0.0, g - US_DEDUCTION)
            if t > 0:
                cash -= t
                taxes += t
                if record:
                    fills.append(dict(day=D, code="US-TAX", side="tax", amount=t, note=f"{last_year} 미국 양도세"))
        last_year = D[:4]
        todo = sorted((p for p in pending if p[2] == D), key=lambda p: p[0])
        pending = [p for p in pending if p[2] != D]
        for phase in ("sell", "buy"):
            for (c, val, _) in todo:
                ins = m.inst[c]
                i = m.ipos[c][D]
                o = ins["O"][i]
                cur = units.get(c, 0.0)
                if ins["market"] == "KR" and whole:
                    px = o * m.won_ratio(c, i)
                    n_cur = shares.get(c, 0)
                    n_tgt = int(math.floor(val / px)) if val > 0 else 0
                    dn = n_tgt - n_cur
                    if dn == 0 or (dn < 0) != (phase == "sell"):
                        continue
                    rate = ins["cost"](i, px)
                    if dn > 0:
                        while dn > 0 and dn * px * (1 + rate) > cash:
                            dn -= 1
                        if dn <= 0:
                            continue
                        pay = dn * px * (1 + rate)
                        cash -= pay
                        units[c] = cur + dn * px / o
                        shares[c] = n_cur + dn
                        basis[c] = basis.get(c, 0.0) + pay
                        if record:
                            fills.append(dict(day=D, code=c, side="buy", qty=dn, price=px, amount=pay, cost=dn * px * rate))
                    else:
                        q = -dn
                        frac = q / n_cur
                        proceeds = q * px * (1 - rate)
                        b = basis.get(c, 0.0) * frac
                        gain = proceeds - b
                        t = KR_TAX * gain if (tax and c not in KR_TAX_FREE and gain > 0) else 0.0
                        if vault_rule == "realized":
                            realize_(D, gain - t)
                        proceeds -= t
                        taxes += t
                        cash += proceeds
                        units[c] = cur * (1 - frac)
                        shares[c] = n_cur - q
                        basis[c] = basis.get(c, 0.0) - b
                        if record:
                            fills.append(dict(day=D, code=c, side="sell", qty=q, price=px, amount=proceeds, gain=gain, tax=t, cost=q * px * rate))
                else:
                    nu = val / o
                    du = nu - cur
                    if abs(du) < 1e-12 or (du < 0) != (phase == "sell"):
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
                        if record:
                            fills.append(dict(day=D, code=c, side="buy", qty=du, price=o, amount=amt, cost=du * o * rate))
                    else:
                        q = -du
                        frac = q / cur
                        proceeds = q * o * (1 - rate)
                        b = basis.get(c, 0.0) * frac
                        gain = proceeds - b
                        t = 0.0
                        if tax:
                            if ins["market"] == "US":
                                us_gain[D[:4]] = us_gain.get(D[:4], 0.0) + gain
                            elif c not in KR_TAX_FREE and gain > 0:
                                t = KR_TAX * gain
                        if vault_rule == "realized":
                            d_tax = 0.0
                            if tax and ins["market"] == "US":
                                acc = US_TAX * max(0.0, us_gain.get(D[:4], 0.0) - US_DEDUCTION)
                                d_tax = acc - us_acc.get(D[:4], 0.0)
                                us_acc[D[:4]] = acc
                            realize_(D, gain - t - d_tax)
                        proceeds -= t
                        taxes += t
                        cash += proceeds
                        units[c] = cur - q
                        basis[c] = basis.get(c, 0.0) - b
                        if record:
                            fills.append(dict(day=D, code=c, side="sell", qty=q, price=o, amount=proceeds, gain=gain, tax=t, cost=q * o * rate))
        if D in m.kr_days:
            cash += deposit
            dep += deposit
        if D in reb:
            eq = cash + sum(u * m.close_at(c, D)[0] for c, u in units.items() if u)
            P = eq + vault - dep
            pay = 0.0
            if vault_rule == "mark" and P > hwm:
                pay = VAULT_SHARE * (P - hwm)
                payouts.append(dict(day=D, amount=pay, profit=P, previous_high=hwm))
                hwm = P
                vault += pay
                cash -= pay
                eq -= pay
            tw, why = strategy.decide(D, k_of(D))
            if overlay is not None:
                tw, why = overlay.targets(D, tw, why)
            for c in sorted(set(units) | set(tw)):
                o, i, d_exec = m.next_open(c, D)
                if o is None or d_exec > end:
                    continue
                pending.append((c, tw.get(c, 0.0) * eq * (o / m.close_at(c, D)[0]), d_exec))
            if record:
                hold = {c: dict(value=u * m.close_at(c, D)[0]) for c, u in units.items() if u > 1e-12}
                decisions.append(dict(day=D, equity=eq, vault_paid=pay, profit=P, targets=tw, holdings_before=hold, **why))
        elif overlay is not None:
            busy = {p[0] for p in pending}
            held = [c for c, u in units.items() if u > 1e-12 and c not in busy]
            for c, why_x in sorted(overlay.exits(D, held).items()):
                o, i, d_exec = m.next_open(c, D)
                if o is None or d_exec > end:
                    continue
                pending.append((c, 0.0, d_exec))
                exits.append(dict(day=D, code=c, exec_day=d_exec, reason=why_x))
        eqd = cash + sum(u * m.close_at(c, D)[0] for c, u in units.items() if u)
        if record:
            marks.append(dict(day=D, total=eqd + vault, vault=vault, deposits=dep, cash=cash))
    eq = cash + sum(u * m.close_at(c, end)[0] for c, u in units.items() if u)
    P = np.array([x["total"] - x["deposits"] for x in marks]) if marks else np.array([0.0])
    dd = float((P - np.maximum.accumulate(P)).min())
    hold = {c: dict(units=u, value=u * m.close_at(c, end)[0], market=m.inst[c]["market"], shares=shares.get(c)) for c, u in units.items() if u > 1e-12}
    return dict(deposits=dep, vault=vault, account=eq, total=eq + vault, cash=cash, taxes=taxes, worst_profit_drawdown=dd, payouts=payouts, decisions=decisions, fills=fills, marks=marks, holdings=hold, exits=exits)


def choose_k(market, strategy, years, ks=KS):
    out = {}
    for y in years:
        best, best_v, table = None, -1e18, {}
        for k in ks:
            r = run(market, strategy, f"{y - LOOKBACK_YEARS}-01-01", f"{y - 1}-12-31", lambda D, k=k: k, whole=False, tax=False, record=False)
            table[k] = r["total"]
            if r["total"] > best_v + 1e-6:
                best, best_v = k, r["total"]
        out[y] = dict(k=best, past_totals=table)
    return out
