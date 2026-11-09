import bisect
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import numpy as np
import ms_alloc_v3 as A3
import ms_alloc_v7 as A7
import mom_shadow as MS
from nosock import engine as E
from nosock import run_backtest as RB

INV_DIR = os.path.join(ROOT, "data", "us_etf", "yf_us_inv1x_20261108_170458")
INV_MANIFEST_SHA = "0d48741423ccbec233196ff9e77f09856af6f3a9096a05e698bbefada9475fec"
OUT = os.path.join(HERE, "results", "dd_study_v1.json")
PAIRS = {"SPY": "SH", "QQQ": "PSQ", "DIA": "DOG", "IWM": "RWM", "XLF": "SEF", "EFA": "EFZ", "EEM": "EUM", "TLT": "TBF", "069500": "114800"}
EPISODES = [("2008 금융위기", "2007-10-01", "2009-06-30"), ("2011 유럽위기", "2011-05-01", "2011-12-31"), ("2015-16 중국", "2015-06-01", "2016-03-31"), ("2018 4분기", "2018-09-01", "2019-03-31"), ("2020 코로나", "2020-01-15", "2020-06-30"), ("2022 긴축", "2022-01-01", "2022-12-31"), ("2026", "2026-01-01", "2026-11-05")]
LONG_START, SHORT_START = "2007-01-02", "2016-01-01"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load_inv():
    mp = os.path.join(INV_DIR, "manifest.json")
    if hashlib.sha256(open(mp, "rb").read()).hexdigest() != INV_MANIFEST_SHA:
        raise SystemExit("inverse manifest hash differs")
    man = json.load(open(mp, encoding="utf-8"))
    out = {}
    for key, info in man["files"].items():
        if not key.endswith("_adj"):
            continue
        p = os.path.join(INV_DIR, info["file"])
        if hashlib.sha256(open(p, "rb").read()).hexdigest() != info["sha256"]:
            raise SystemExit("inverse file hash differs " + key)
        out[key[:-4]] = json.load(open(p, encoding="utf-8"))
    return man, out


def market_all():
    kr, kus, us, fx = MS.pinned()
    man_inv, inv = load_inv()
    us_all = dict(us)
    us_all.update(inv)
    cut = lambda d: {c: MS.cut(v, MS.ANCHOR) for c, v in d.items()}
    inst, _ = MS.build(cut(kr), cut(kus), cut(us_all), {d: v for d, v in fx.items() if d <= MS.ANCHOR}, end=MS.ANCHOR)
    for c, ins in inst.items():
        ins["cost_maker"] = A7.kr_cost if ins["market"] == "KR" else A7.us_cost_maker(RB.US_FEE, RB.US_FX)
    u29 = MS.KR_DOMESTIC + list(us)
    groups = dict(U29=u29, U38=u29 + list(inv))
    return inst, groups


def sma(C, w):
    out = np.full(len(C), np.nan)
    cs = np.cumsum(np.r_[0.0, C])
    out[w - 1:] = (cs[w:] - cs[:-w]) / w
    return out


class Overlay:
    name = "기본"

    def __init__(self, m):
        self.m = m

    def ci(self, c, D):
        return self.m.close_at(c, D)

    def targets(self, D, tw, why):
        return tw, why

    def exits(self, D, held):
        return {}


def drop_to_cash(tw, why, drops, reason):
    tw = dict(tw)
    cands = []
    for x in why.get("candidates", []):
        if x.get("code") in drops and x.get("selected"):
            x = dict(x, selected=False, reason=reason)
        cands.append(x)
    for c in drops:
        tw[c] = 0.0
    return tw, dict(why, candidates=cands, cash_weight=max(0.0, 1.0 - sum(tw.values())))


class WeeklyAbs(Overlay):
    name = "주간 12-1 점검"

    def __init__(self, m, absmom):
        super().__init__(m)
        self.absmom = absmom
        self.week = None

    def exits(self, D, held):
        wk = date.fromisoformat(D).isocalendar()[:2]
        if wk == self.week:
            return {}
        self.week = wk
        out = {}
        for c in held:
            _, i = self.ci(c, D)
            a = self.absmom[c][i] if i is not None else np.nan
            if not np.isnan(a) and a <= 0:
                out[c] = "12-1개월 수익 0 이하 (주간 점검)"
        return out


class Trend(Overlay):
    def __init__(self, m, w=100, inverse=False):
        super().__init__(m)
        self.w = w
        self.inverse = inverse
        self.name = f"{w}일선 아래면 {'짝 인버스' if inverse else '현금'}"
        self.s = {c: sma(np.asarray(m.inst[c]["C"]), w) for c in m.codes}

    def below(self, c, D):
        px, i = self.ci(c, D)
        if i is None:
            return False
        s = self.s[c][i]
        return not np.isnan(s) and px < s

    def targets(self, D, tw, why):
        drops = [c for c, w in tw.items() if w > 0 and self.below(c, D)]
        if not drops:
            return tw, why
        tw2, why2 = drop_to_cash(tw, why, drops, f"{self.w}일선 아래 → {'인버스' if self.inverse else '현금'}")
        if self.inverse:
            elig = set(self.m.eligible(D))
            for c in drops:
                inv = PAIRS.get(c)
                if inv and inv in elig and not self.below(inv, D):
                    tw2[inv] = tw2.get(inv, 0.0) + tw[c]
            why2["cash_weight"] = max(0.0, 1.0 - sum(tw2.values()))
        return tw2, why2

    def exits(self, D, held):
        return {c: f"{self.w}일선 아래" for c in held if self.below(c, D)}


class Trail(Overlay):
    def __init__(self, m, dd=0.10):
        super().__init__(m)
        self.dd = dd
        self.name = f"고점 대비 −{int(dd * 100)}%면 현금"
        self.peak = {}

    def exits(self, D, held):
        hs = set(held)
        for c in list(self.peak):
            if c not in hs:
                del self.peak[c]
        out = {}
        for c in held:
            px, i = self.ci(c, D)
            if i is None:
                continue
            p = max(self.peak.get(c, px), px)
            self.peak[c] = p
            if px < (1 - self.dd) * p:
                out[c] = f"고점 대비 −{int(self.dd * 100)}%"
                del self.peak[c]
        return out


class VolTarget(Overlay):
    def __init__(self, m, target=0.15, look=60):
        super().__init__(m)
        self.target = target
        self.look = look
        self.name = f"변동성 연 {int(target * 100)}%에 맞춰 비중 조절"

    def targets(self, D, tw, why):
        codes = [c for c, w in tw.items() if w > 0]
        if not codes:
            return tw, why
        j = bisect.bisect_right(self.m.days, D)
        days = self.m.days[max(0, j - self.look - 1):j]
        P = np.array([[self.ci(c, d)[0] or np.nan for c in codes] for d in days], float)
        R = np.diff(np.log(P), axis=0)
        R = R[~np.isnan(R).any(axis=1)]
        if len(R) < 20:
            return tw, why
        w = np.array([tw[c] for c in codes])
        vol = float(math.sqrt(max(0.0, w @ (np.cov(R.T) * 252 if len(codes) > 1 else np.array([[np.var(R) * 252]])) @ w)))
        s = min(1.0, self.target / vol) if vol > 0 else 1.0
        tw2 = {c: v * s for c, v in tw.items()}
        return tw2, dict(why, cash_weight=max(0.0, 1.0 - sum(tw2.values())), vol_scale=s, port_vol=vol)


class RegimeInverse(Overlay):
    name = "S&P500이 200일선 아래면 현금 몫을 SH로"

    def __init__(self, m):
        super().__init__(m)
        self.s = sma(np.asarray(m.inst["SPY"]["C"]), 200)

    def targets(self, D, tw, why):
        px, i = self.ci("SPY", D)
        if i is None or np.isnan(self.s[i]) or px >= self.s[i] or "SH" not in self.m.eligible(D):
            return tw, why
        cw = 1.0 - sum(tw.values())
        if cw <= 1e-9:
            return tw, why
        tw2 = dict(tw)
        tw2["SH"] = tw2.get("SH", 0.0) + cw
        return tw2, dict(why, cash_weight=0.0, regime="SPY<200d")


def twr_index(marks):
    idx = np.ones(len(marks))
    for t in range(1, len(marks)):
        prev = marks[t - 1]["total"]
        flow = marks[t]["deposits"] - marks[t - 1]["deposits"]
        r = (marks[t]["total"] - flow) / prev - 1 if prev > 0 else 0.0
        idx[t] = idx[t - 1] * (1 + r)
    return idx


def metrics(r):
    marks = r["marks"]
    days = [x["day"] for x in marks]
    I = twr_index(marks)
    run_max = np.maximum.accumulate(I)
    out = dict(total=r["total"], vault=r["vault"], deposits=r["deposits"], taxes=r["taxes"], costs=float(sum(f.get("cost", 0.0) for f in r["fills"])), fills=len(r["fills"]), exits=len(r.get("exits", [])), worst_profit_drawdown=r["worst_profit_drawdown"], twr_max_dd=float((I / run_max - 1).min()))
    yrs = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25
    out["twr_cagr"] = float(I[-1] ** (1 / yrs) - 1) if yrs > 0 else None
    ep = {}
    for name, a, b in EPISODES:
        lo, hi = bisect.bisect_left(days, a), bisect.bisect_right(days, b)
        if hi - lo < 20:
            continue
        seg = I[lo:hi]
        ep[name] = float((seg / np.maximum.accumulate(seg) - 1).min())
    out["episodes"] = ep
    return out


def load_engine_copy(path):
    spec = importlib.util.spec_from_file_location("engine_before", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    t0 = time.time()
    inst, groups = market_all()
    names = {}
    log(f"market built: {len(inst)} instruments")
    m29 = E.Market(inst, groups["U29"], names)
    m29.raw = A3.load_raw()
    m38 = E.Market(inst, groups["U38"], names)
    m38.raw = m29.raw
    dual = RB.strategy_for(inst, groups["U29"], "dual", m29)
    years = list(range(int(LONG_START[:4]), int(MS.ANCHOR[:4]) + 1))
    kinfo = E.choose_k(m29, dual, years)
    kpath = {y: v["k"] for y, v in kinfo.items()}
    k_of = lambda D: kpath[int(D[:4])]
    log(f"K path {kpath}")

    report = dict(anchor=MS.ANCHOR, long_start=LONG_START, short_start=SHORT_START, k_path=kpath, note="모든 숫자는 현실 조건(1주 단위, 수수료, 호가, 세금) 후. K는 모든 단계에서 기본 듀얼과 같은 연도별 K를 씀(장치 효과만 분리).", rungs=[])

    old = sys.argv[1] if len(sys.argv) > 1 else None
    if old:
        EO = load_engine_copy(old)
        a = E.run(m29, dual, SHORT_START, MS.ANCHOR, k_of, record=False)
        b = EO.run(m29, dual, SHORT_START, MS.ANCHOR, k_of, record=False)
        report["parity"] = dict(new=a["total"], old=b["total"], diff=a["total"] - b["total"])
        log(f"parity new {a['total']:.2f} old {b['total']:.2f}")

    absmom29 = dual.absmom
    rungs = [
        ("R0", "듀얼 모멘텀 그대로", m29, "dual", None, k_of),
        ("T0", "상위 K 그대로", m29, "topk", None, k_of),
        ("Q0", "같은 후보 균등 분배", m29, "equal", None, lambda D: 1),
        ("R1", "+ 매주 12-1 수익 점검", m29, "dual", lambda m, s: WeeklyAbs(m, s.absmom), k_of),
        ("R2", "+ 100일선 아래면 현금", m29, "dual", lambda m, s: Trend(m, 100), k_of),
        ("R3", "+ 고점 대비 −10%면 현금", m29, "dual", lambda m, s: Trail(m, 0.10), k_of),
        ("R4", "+ 변동성 연 15%에 맞춰 비중", m29, "dual", lambda m, s: VolTarget(m, 0.15), k_of),
        ("R5", "+ 최소 3종목", m29, "dual", None, lambda D: max(3, k_of(D))),
        ("R6", "+ S&P500 200일선 아래면 현금 몫을 SH로", m29, "dual", lambda m, s: RegimeInverse(m), k_of),
        ("R7", "+ 1배 인버스 9종을 후보에 추가", m38, "dual", None, k_of),
        ("R8", "+ 100일선 아래면 짝 1배 인버스로", m38, "dual", lambda m, s: Trend(m, 100, inverse=True), k_of),
    ]
    for code, label, m, mode, mk, kf in rungs:
        t1 = time.time()
        strat = RB.strategy_for(inst, m.codes, mode, m) if not (m is m29 and mode == "dual") else dual
        row = dict(code=code, label=label)
        for tag, start in (("long", LONG_START), ("short", SHORT_START)):
            ov = mk(m, strat) if mk else None
            r = E.run(m, strat, start, MS.ANCHOR, kf, whole=True, tax=True, record=True, overlay=ov)
            row[tag] = metrics(r)
        report["rungs"].append(row)
        L, Sh = row["long"], row["short"]
        log(f"{code} {label}: 2016~ 합계 {Sh['total'] / 1e4:,.0f}만 금고 {Sh['vault'] / 1e4:,.0f}만 순이익낙폭 {Sh['worst_profit_drawdown'] / 1e4:,.0f}만 | 2007~ 합계 {L['total'] / 1e4:,.0f}만 TWR연 {100 * L['twr_cagr']:.1f}% 최대낙폭 {100 * L['twr_max_dd']:.1f}% | " + " ".join(f"{k} {100 * v:.0f}%" for k, v in L["episodes"].items()) + f" ({time.time() - t1:.0f}s)")
        tmp = OUT + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1, default=float)
        os.replace(tmp, OUT)
    log(f"done {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
