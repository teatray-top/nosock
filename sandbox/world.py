import json
import math
import os

import numpy as np

FACTORS = ("G", "T", "E", "R", "S", "L", "D", "F")
FVOL = dict(G=0.009, T=0.010, E=0.018, R=0.005, S=0.008, L=0.008, D=0.007, F=0.004)
FDRIFT = dict(G=0.0003, T=0.0003, E=0.0, R=0.0001, S=0.0001, L=0.0002, D=0.0002, F=0.0)
INST = {
    "HB200": dict(name="한반 대형주200 ETF", load=dict(L=1.0, G=0.5, T=0.3), idio=0.004),
    "HBSEMI": dict(name="한반 반도체 ETF", load=dict(L=0.9, T=1.4, G=0.4), idio=0.008),
    "HBINV": dict(name="한반 대형주200 인버스 1배", inverse="HB200"),
    "HBBOND3": dict(name="한반 국채3년 ETF", load=dict(R=0.25), idio=0.0008),
    "HBCASH": dict(name="한반 단기채 ETF", load={}, idio=0.0002, drift=0.00012),
    "DSIDX": dict(name="대서 대표지수500 ETF", load=dict(G=1.0, T=0.3), idio=0.003),
    "DSTECH": dict(name="대서 기술주 ETF", load=dict(G=1.0, T=1.0), idio=0.005),
    "DSSEMI": dict(name="대서 반도체 ETF", load=dict(G=0.9, T=1.6), idio=0.008),
    "DSINV": dict(name="대서 대표지수500 인버스 1배", inverse="DSIDX"),
    "OIL": dict(name="원유 선물 ETF", load=dict(E=1.0), idio=0.006),
    "ENRG": dict(name="대서 에너지주 ETF", load=dict(E=0.6, G=0.6), idio=0.005),
    "GOLD": dict(name="금 ETF", load=dict(S=1.0, F=-0.3), idio=0.003),
    "DEFN": dict(name="대서 방산주 ETF", load=dict(G=0.6, D=1.0), idio=0.005),
    "TBOND": dict(name="대서 장기국채 ETF", load=dict(R=1.0), idio=0.002),
    "USDX": dict(name="대서 달러 지수 ETF", load=dict(F=1.0), idio=0.001),
}
DIVS = {"HB200": (0.004, 3), "HBBOND3": (0.0025, 1), "HBCASH": (0.003, 1), "DSIDX": (0.0035, 3), "ENRG": (0.009, 3), "TBOND": (0.003, 1)}
DIV_TAX = 0.154
OUTLETS = ("한반일보", "대서타임스", "마켓데일리", "글로벌경제신문", "카라드통신", "세계정세리포트", "투자자뉴스", "한반경제TV")


def factor_impact(imp):
    out = np.zeros(len(FACTORS))
    for k, v in imp.items():
        out[FACTORS.index(k)] += v
    return out


def event_catalog(rng, days):
    ev = []
    for i, d in enumerate((int(rng.integers(18, 26)), int(rng.integers(45, 52)))):
        if d >= days - 3:
            continue
        p = rng.dirichlet([2.0, 5.0, 2.0])
        ev.append(dict(id=f"dscb{i + 1}", kind="dscb", title=f"대서 중앙은행 통화정책회의 ({i + 1}차)", announce=0, resolve=d, scheduled=True,
                       branches=[dict(key="hike", label="금리 인상", p=p[0], jump=dict(R=-0.015, G=-0.012, T=-0.018, F=0.008, L=-0.006)),
                                 dict(key="hold", label="동결", p=p[1], jump={}),
                                 dict(key="cut", label="금리 인하", p=p[2], jump=dict(R=0.012, G=0.010, T=0.015, F=-0.006, L=0.005))],
                       signal="poll", noise=0.10))
    d = int(rng.integers(30, 40))
    if d < days - 15:
        q = float(rng.uniform(0.35, 0.80))
        child_p = float(rng.uniform(0.35, 0.85))
        ev.append(dict(id="midterm", kind="election", title="대서 연방 중간선거", announce=0, resolve=d, scheduled=True,
                       branches=[dict(key="opp", label="야당(진보연합) 승리", p=q, jump=dict(T=-0.02, G=-0.005), child="ai_bill"),
                                 dict(key="gov", label="여당(자유공화당) 승리", p=1 - q, jump=dict(T=0.015, G=0.004))],
                       signal="poll", noise=0.06, child_p=child_p))
    if rng.random() < 0.85:
        a = int(rng.integers(8, max(9, days - 25)))
        p = rng.dirichlet([5.0, 3.0, 2.0])
        ev.append(dict(id="karad", kind="geo", title="대서-카라드 군사 충돌", announce=a, resolve=a + int(rng.integers(8, 15)), scheduled=False,
                       shock=dict(E=0.06, G=-0.02, D=0.03, S=0.02, L=-0.01),
                       branches=[dict(key="calm", label="단발 공습 후 진정", p=p[0], jump=dict(E=-0.05, G=0.015, D=-0.01, S=-0.01)),
                                 dict(key="escalate", label="해협 봉쇄와 확전", p=p[1], jump=dict(E=0.12, G=-0.05, D=0.05, S=0.04, L=-0.04, F=0.01), drift=dict(E=0.05, G=-0.02), drift_days=8),
                                 dict(key="talks", label="곧바로 협상 국면", p=p[2], jump=dict(E=-0.07, G=0.03, D=-0.02, S=-0.015))],
                       signal="dev", noise=0.0))
    if rng.random() < 0.6:
        a = int(rng.integers(5, max(6, days - 20)))
        p = float(rng.uniform(0.3, 0.7))
        ev.append(dict(id="chipctl", kind="policy", title="대서 반도체 대동방 수출 규제 검토", announce=a, resolve=a + int(rng.integers(10, 18)), scheduled=False,
                       branches=[dict(key="hard", label="강경 규제 시행", p=p, jump=dict(T=-0.05), drift=dict(T=-0.03), drift_days=6),
                                 dict(key="soft", label="완화된 규제", p=1 - p, jump=dict(T=0.03))],
                       signal="poll", noise=0.12))
    for i, (co, tick, load) in enumerate((("네오칩", "DSSEMI", dict(T=1.0)), ("한반전자", "HBSEMI", dict(T=0.6, L=0.6)))):
        d = int(rng.integers(12, max(13, days - 8)))
        p = rng.dirichlet([3.0, 3.0, 2.0])
        ev.append(dict(id=f"earn{i + 1}", kind="earnings", title=f"{co} 분기 실적 발표", announce=0, resolve=d, scheduled=True, company=co, ticker=tick,
                       branches=[dict(key="beat", label="예상 상회와 가이던스 상향", p=p[0], jump={k: 0.05 * v for k, v in load.items()}),
                                 dict(key="inline", label="예상 부합", p=p[1], jump={k: -0.005 * v for k, v in load.items()}),
                                 dict(key="miss", label="예상 하회", p=p[2], jump={k: -0.06 * v for k, v in load.items()})],
                       signal="poll", noise=0.12))
    return ev


def dividend_schedule(rng, days):
    out = {}
    for c, (amt, every) in DIVS.items():
        n = max(1, int(round(days / (21 * every))))
        first = int(rng.integers(5, max(6, min(days - 2, 21 * every))))
        out[c] = [dict(ex=d, amount=amt, announce=max(0, d - 10)) for d in range(first, days - 1, 21 * every)][:n + 1]
    return out


def norm(p):
    p = np.maximum(np.asarray(p, float), 1e-6)
    return p / p.sum()


def make_world(seed, days=60):
    rng = np.random.default_rng(seed)
    events = event_catalog(rng, days)
    for e in events:
        ps = norm([b["p"] for b in e["branches"]])
        for b, x in zip(e["branches"], ps):
            b["p"] = float(x)
        e["outcome"] = int(rng.choice(len(ps), p=ps))
    child = None
    for e in events:
        if e["id"] == "midterm" and "child" in e["branches"][e["outcome"]]:
            cp = e["child_p"]
            child = dict(id="ai_bill", kind="policy", title="대서 AI 규제법(알고리즘 책임법) 표결", announce=e["resolve"] + 1, resolve=min(days - 2, e["resolve"] + int(rng.integers(10, 16))), scheduled=False,
                         branches=[dict(key="pass", label="AI 규제법 통과", p=cp, jump=dict(T=-0.07, G=-0.01), drift=dict(T=-0.04), drift_days=10),
                                   dict(key="fail", label="AI 규제법 부결", p=1 - cp, jump=dict(T=0.035))],
                         signal="poll", noise=0.10)
            child["outcome"] = int(rng.choice(2, p=norm([cp, 1 - cp])))
    if child and child["announce"] < days - 3:
        events.append(child)
    F = np.zeros((days, len(FACTORS)))
    vol = np.array([FVOL[k] for k in FACTORS])
    drift = np.array([FDRIFT[k] for k in FACTORS])
    stress = np.ones(days)
    for e in events:
        if e["kind"] == "geo":
            stress[e["announce"]:e["resolve"] + 5] = 1.8
    for t in range(days):
        F[t] = drift + vol * rng.standard_normal(len(FACTORS)) * stress[t]
    news = []
    consensus = {}
    alpha = 0.6
    for e in events:
        k = len(e["branches"])
        ptrue = np.array([b["p"] for b in e["branches"]])
        J = np.array([factor_impact(b["jump"]) for b in e["branches"]])
        q = norm(ptrue + rng.normal(0, 0.12, k))
        if e["signal"] == "dev":
            q = norm(np.ones(k) / k + rng.normal(0, 0.05, k))
        prev = 0.0
        qs = {}
        if e.get("shock") and e["announce"] < days:
            F[e["announce"]] += factor_impact(e["shock"])
        for t in range(e["announce"], min(e["resolve"], days)):
            if e["signal"] == "dev":
                lr = np.ones(k)
                lr[e["outcome"]] = 1.35
                q = norm(q * np.where(rng.random(k) < 0.7, lr, 1 / lr) ** 0.5)
            else:
                q = norm(0.85 * q + 0.15 * norm(ptrue + rng.normal(0, e["noise"], k)))
            exp = alpha * (q @ J)
            F[t] += exp - prev
            prev = exp
            qs[t] = q.tolist()
        if e["resolve"] < days:
            b = e["branches"][e["outcome"]]
            F[e["resolve"]] += J[e["outcome"]] - prev
            if b.get("drift"):
                n = b.get("drift_days", 5)
                for t in range(e["resolve"] + 1, min(days, e["resolve"] + 1 + n)):
                    F[t] += factor_impact(b["drift"]) / n
        consensus[e["id"]] = qs
    R = {}
    for c, spec in INST.items():
        if "inverse" in spec:
            continue
        L = np.array([spec["load"].get(k, 0.0) for k in FACTORS])
        R[c] = F @ L + spec.get("drift", 0.0) + spec["idio"] * rng.standard_normal(days)
    for c, spec in INST.items():
        if "inverse" in spec:
            R[c] = -R[spec["inverse"]] - 0.00004
    divs = dividend_schedule(rng, days)
    for c, lst in divs.items():
        for d in lst:
            R[c][d["ex"]] -= d["amount"]
    P = {c: (100.0 * np.exp(np.cumsum(np.log1p(r)))).tolist() for c, r in R.items()}
    news = write_news(rng, events, consensus, P, days, divs)
    return dict(seed=seed, days=days, start="2031-03-03", events=events, consensus=consensus, prices=P, returns={c: list(map(float, r)) for c, r in R.items()}, dividends=divs, div_tax=DIV_TAX, news=news)


def day_label(start, t):
    from datetime import date, timedelta
    d = date.fromisoformat(start)
    n = 0
    while n < t:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return d.isoformat()


def write_news(rng, events, consensus, P, days, divs=None):
    out = []
    nid = [0]

    def add(t, title, body, outlet=None):
        nid[0] += 1
        out.append(dict(id=f"n{nid[0]:04d}", day=t, outlet=outlet or str(rng.choice(OUTLETS)), title=title, body=body))

    names = {c: s["name"] for c, s in INST.items()}
    for t in range(days):
        if t > 0:
            moves = sorted(((c, P[c][t] / P[c][t - 1] - 1) for c in P), key=lambda x: x[1])
            lo, hi = moves[:3], moves[-3:]
            add(t, "오늘의 시장 마감", "상승: " + ", ".join(f"{names[c]} {100 * r:+.1f}%" for c, r in reversed(hi)) + ". 하락: " + ", ".join(f"{names[c]} {100 * r:+.1f}%" for c, r in lo) + ".", "마켓데일리")
        if rng.random() < 0.5:
            add(t, str(rng.choice(["한반 소비심리 소폭 개선", "대서 고용지표 예상 부합", "동방 경기 둔화 우려 지속", "한반 수출 증가세 유지", "대서 주택시장 혼조"])), "시장 영향은 제한적이라는 평가가 많다.")
    for c, lst in (divs or {}).items():
        for d in lst:
            add(d["announce"], f"[일정] {INST[c]['name']} 분배금 공시", f"주당 분배금은 가격의 약 {100 * d['amount']:.2f}%, 배당락일은 {d['ex']}거래일차(시작일 기준). 배당락일에 가격이 분배금만큼 떨어지고, 분배금에는 배당소득세 15.4%가 붙는다.", "한반경제TV")
    for e in events:
        if e["kind"] == "earnings":
            for t in sorted(int(x) for x in rng.choice(np.arange(max(1, e["resolve"] - 12), e["resolve"]), size=3, replace=False)):
                b = e["branches"]
                ptrue = np.array([x["p"] for x in b])
                obs = norm(ptrue + rng.normal(0, 0.15, 3))
                kind = str(rng.choice(["애널리스트 실적 전망", "공급망 점검 보고서", "시장 소문(위스퍼 넘버)"]))
                add(t, f"{e['company']} 실적 앞두고 {kind}", f"{kind} 기준 결과 가능성: " + ", ".join(f"{x['label']} {100 * v:.0f}%" for x, v in zip(b, obs)) + ".")
    for e in events:
        k = len(e["branches"])
        labels = [b["label"] for b in e["branches"]]
        if e["scheduled"]:
            add(0, f"[일정] {e['title']} 예정", f"{e['title']}이(가) 시장 개장일 기준 약 {e['resolve']}거래일 뒤에 열린다. 가능한 결과: {', '.join(labels)}.")
        else:
            if e["kind"] == "geo":
                add(e["announce"], "[속보] 대서, 카라드 군사시설 공습", "대서 연방군이 카라드 해안 미사일 기지를 공습했다. 카라드는 보복을 예고했고 해협 통행 차단 가능성이 거론된다. 원유 가격이 급등했다.", "카라드통신")
            else:
                add(e["announce"], f"[속보] {e['title']}", f"관련 결정은 약 {e['resolve'] - e['announce']}거래일 뒤 나올 전망. 가능한 결과: {', '.join(labels)}.")
        if e["id"] == "midterm":
            add(int(rng.integers(0, 6)), "야당 진보연합, AI 알고리즘 책임법 공약", "야당은 선거에서 이기면 대형 AI 기업과 반도체 공급망에 책임을 묻는 법안을 첫 과제로 추진하겠다고 밝혔다. 기술주 부담 요인이라는 분석이 나온다.")
        qs = consensus.get(e["id"], {})
        ptrue = np.array([b["p"] for b in e["branches"]])
        for t in sorted(qs):
            if e["signal"] == "poll" and (t - e["announce"]) % 3 == 0:
                obs = norm(ptrue + rng.normal(0, e["noise"], k))
                bias = rng.random() < 0.2
                if bias:
                    obs = norm(obs + np.eye(k)[int(rng.integers(k))] * 0.15)
                kind = "여론조사" if e["kind"] == "election" else "전문가 설문"
                add(t, f"{e['title']} {kind}", ", ".join(f"{l} {100 * x:.0f}%" for l, x in zip(labels, obs)) + (" (응답 표본이 작아 오차가 크다는 지적)" if bias else ""), "글로벌경제신문" if not bias else str(rng.choice(OUTLETS)))
            if (t - e["announce"]) % 4 == 2:
                q = qs[t]
                add(t, f"예측시장: {e['title']}", "예측시장 배당 기준 확률 " + ", ".join(f"{l} {100 * x:.0f}%" for l, x in zip(labels, q)) + ".", "투자자뉴스")
            if e["signal"] == "dev" and t > e["announce"]:
                true = e["outcome"]
                hint = true if rng.random() < 0.65 else int(rng.integers(k))
                text = {"calm": ["카라드 정부, 추가 대응 자제 시사", "해협 선박 통행 정상화 조짐"], "escalate": ["카라드 해군, 해협 순찰 강화", "해협 인근 유조선 피격 보도"], "talks": ["중재국, 대서-카라드 비공식 접촉 확인", "양측 대표단 제3국 회동설"]}
                add(t, str(rng.choice(text[e["branches"][hint]["key"]])), "현지 소식통 인용 보도. 사실 여부는 엇갈린다.", "세계정세리포트")
        if e["resolve"] < days:
            b = e["branches"][e["outcome"]]
            add(e["resolve"], f"[결과] {e['title']}: {b['label']}", f"{e['title']} 결과가 '{b['label']}'(으)로 확정됐다.")
    out.sort(key=lambda x: (x["day"], x["id"]))
    return out


if __name__ == "__main__":
    import sys
    w = make_world(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    print(len(w["events"]), [(e["id"], e["announce"], e["resolve"], e["branches"][e["outcome"]]["key"], [round(b["p"], 2) for b in e["branches"]]) for e in w["events"]])
    print(len(w["news"]), "articles")
    print({c: round(w["prices"][c][-1], 1) for c in w["prices"]})
