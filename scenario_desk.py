import csv
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ft"))
sys.dont_write_bytecode = True
import numpy as np

KST = ZoneInfo("Asia/Seoul")
RUNS = os.environ.get("NOSOCK_RUNS", os.path.join(HERE, "runs"))
LOG = os.environ.get("NOSOCK_SCENARIO_LOG", os.path.join(RUNS, "scenario_desk.jsonl"))
WORK = os.environ.get("NOSOCK_SCENARIO_WORK", os.path.join(RUNS, "scenario_work"))
CAL = os.path.join(HERE, "data", "events", "calendar.json")
MACRO_LIVE = os.environ.get("NOSOCK_MACRO_LIVE", os.path.join(HERE, "data", "macro", "live"))
CAP = float(os.environ.get("NOSOCK_SCENARIO_CAP", 0.30))
MODEL = os.environ.get("NOSOCK_SCENARIO_MODEL", "gpt-6.1-sol")
DESK_AT = tuple(int(x) for x in os.environ.get("NOSOCK_DESK_AT", "08:15").split(":"))
RMAX = 40.0
DAILY = ("DGS10", "DGS2", "DGS3MO", "T10Y2Y", "VIXCLS", "DFF")
MONTHLY = ("IRLTLT01KRM156N", "IR3TIB01KRM156N", "IRLTLT01USM156N", "IR3TIB01USM156N", "IRLTLT01JPM156N", "IRLTLT01DEM156N", "IR3TIB01CNM156N")
EVENT_LABEL = {"fomc": "미국 연준 FOMC 결정", "bok": "한국은행 금통위 결정", "kr_expiry": "한국 선물옵션 동시만기", "us_quad_witching": "미국 분기 파생 만기", "kr_president": "한국 대선", "kr_general": "한국 총선", "kr_local": "한국 지방선거", "us_president": "미국 대선", "us_midterm": "미국 중간선거"}


def now():
    return datetime.now(KST)


def log(kind, **kw):
    row = dict(ts=now().isoformat(timespec="seconds"), kind=kind, **kw)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
    return row


def codex_exe():
    if os.environ.get("NOSOCK_CODEX"):
        return os.environ["NOSOCK_CODEX"]
    hits = sorted(glob.glob(r"C:\Program Files\WindowsApps\OpenAI.Codex_*_x64__*\app\resources\codex.exe"))
    if not hits:
        raise FileNotFoundError("codex.exe not found")
    return hits[-1]


def codex(prompt, timeout=1800):
    os.makedirs(WORK, exist_ok=True)
    last = os.path.join(WORK, f"last_{os.getpid()}.txt")
    args = [codex_exe(), "--search", "-m", MODEL, "exec", "--skip-git-repo-check", "--ephemeral", "--json", "-s", "workspace-write", "-c", "sandbox_workspace_write.network_access=true", "-C", WORK, "-o", last, "-"]
    t = time.time()
    r = subprocess.run(args, input=prompt, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=WORK, timeout=timeout, **({"creationflags": 0x40} if os.name == "nt" else {}))
    usage = {}
    for line in r.stdout.splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "turn.completed":
            usage = e.get("usage", {})
    text = open(last, encoding="utf-8").read() if os.path.exists(last) else ""
    if os.path.exists(last):
        os.remove(last)
    if r.returncode != 0 or not text:
        raise RuntimeError(f"codex exit {r.returncode}")
    return text, dict(secs=round(time.time() - t, 1), input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"))


def parse_json(text):
    s = text.strip()
    if "```" in s:
        parts = s.split("```")
        s = max(parts[1::2], key=len)
        s = s[4:] if s.startswith("json") else s
    a, b = s.find("{"), s.rfind("}")
    return json.loads(s[a:b + 1])


def fetch_fred(day):
    d = os.path.join(MACRO_LIVE, day)
    os.makedirs(d, exist_ok=True)
    for sid in DAILY + MONTHLY:
        p = os.path.join(d, sid + ".csv")
        if os.path.exists(p) and os.path.getsize(p) > 100:
            continue
        got = None
        for k in range(3):
            try:
                r = subprocess.run(["curl", "-s", "-m", "90", "-f", f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=2024-01-01"], capture_output=True, timeout=120)
                if r.returncode == 0 and len(r.stdout) > 100:
                    got = r.stdout
                    break
            except subprocess.TimeoutExpired:
                pass
            time.sleep(5 * (k + 1))
        if got is None:
            old = sorted(glob.glob(os.path.join(HERE, "data", "macro", "*", sid + ".csv")))
            old = [x for x in old if os.path.dirname(x) != d]
            if not old:
                raise RuntimeError(f"FRED {sid} unavailable")
            got = open(old[-1], "rb").read()
            log("macro_fallback", series=sid, used=os.path.relpath(old[-1], HERE))
        open(p, "wb").write(got)
    return d


def last_values(d, sid, n=1):
    rows = []
    with open(os.path.join(d, sid + ".csv"), encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0][:1] == "2":
                try:
                    rows.append((row[0], float(row[1])))
                except ValueError:
                    pass
    return rows[-n:]


def macro_snapshot(day):
    d = fetch_fred(day)
    out = {}
    for sid in DAILY:
        v = last_values(d, sid, 61)
        if v:
            out[sid] = dict(date=v[-1][0], value=v[-1][1], chg_20d=round(v[-1][1] - v[-21][1], 3) if len(v) > 21 else None, chg_60d=round(v[-1][1] - v[0][1], 3) if len(v) > 60 else None)
    for sid in MONTHLY:
        v = last_values(d, sid, 1)
        if v:
            out[sid] = dict(month=v[-1][0][:7], value=v[-1][1])
    return out


def upcoming(asof, ahead=35, back=7):
    cal = json.load(open(CAL, encoding="utf-8"))
    lo = (date.fromisoformat(asof) - timedelta(days=back)).isoformat()
    hi = (date.fromisoformat(asof) + timedelta(days=ahead)).isoformat()
    ev = []
    for k, lab in EVENT_LABEL.items():
        for d in cal.get(k, []):
            if lo <= d <= hi:
                ev.append(dict(date=d, event=lab, passed=d < asof))
    corp = os.path.join(HERE, "data", "events", "corporate.json")
    if os.path.exists(corp):
        cj = json.load(open(corp, encoding="utf-8"))
        for t, v in cj.get("earnings", {}).items():
            for d in v.get("dates", []):
                if lo <= d <= hi:
                    ev.append(dict(date=d, event=f"{v['name']}({t}) 실적 발표", passed=d < asof))
        for c, v in cj.get("exdiv", {}).items():
            nx = v.get("next")
            if not nx or not nx.get("period_days"):
                continue
            d = date.fromisoformat(nx["date"])
            while d.isoformat() < asof:
                d += timedelta(days=nx["period_days"])
            if d.isoformat() <= hi:
                ev.append(dict(date=d.isoformat(), event=f"{c} 배당락(추정, 직전 분배금 {nx['last_amount']})", passed=False, estimated=True))
    return sorted(ev, key=lambda x: x["date"])


def names_map():
    import ms_alloc_v7 as A7
    import ms_role_v2 as R
    names = json.load(open(os.path.join(R.ETF_DIR, "manifest.json"), encoding="utf-8"))["names"]
    man_us, _ = A7.load_us()
    return dict(names, **man_us["names"])


def snapshot(inst, codes, D):
    out = []
    for c in codes:
        ins = inst.get(c)
        if ins is None:
            continue
        import bisect
        i = bisect.bisect_right(ins["days"], D) - 1
        if i < 260:
            continue
        C = np.asarray(ins["C"], float)
        out.append(dict(code=c, market=ins["market"], last=ins["days"][i], r20=round(100 * (C[i] / C[i - 20] - 1), 1), r126=round(100 * (C[i] / C[i - 126] - 1), 1), dd252=round(100 * (C[i] / C[i - 251:i + 1].max() - 1), 1)))
    return out


PROMPT = """너는 작은 자동매매 계좌(레버리지 없음, 공매도 없음, 국내·미국 ETF와 일부 개별주만 매수)의 시나리오 데스크다.
오늘은 {today} (한국 시각 {now})이고, 마지막 종가 기준일은 {asof}이다.

할 일:
1. 웹 검색과 작업 폴더 안의 도구(명령 실행, 인터넷 조회)를 자유롭게 써서 지금 시점의 최신 뉴스와 일정(정치, 지정학, 통화정책, 규제, 실적, 선거 결과 등)을 확인해라. 사실은 반드시 검색으로 확인하고 출처 URL을 붙여라. 확인 못 한 것은 쓰지 마라.
2. 앞으로 20거래일 동안 아래 후보 종목 가격을 크게 움직일 수 있는 일을 정리해라. 이미 정해진 일정(아래 목록)과 속보·진행 중인 사건 모두.
3. 앞으로 20거래일에 대한 '함께 일어나는 큰 그림' 시나리오를 3~6개 써라. 서로 겹치지 않고 합치면 전부가 되게, 확률 합이 1이 되게.
   각 시나리오마다: 이름(한국어), 확률, 인과 사슬(예: 중간선거 결과 → 의회 구성 변화 → AI 규제 법안 가능성 → 반도체 하락), 각 후보 종목의 20거래일 예상 수익률(%), 나중에 맞았는지 채점할 수 있는 확인 지점(날짜·관찰 대상·기준값).
   확률은 근거(여론조사, 예측시장, 선물 가격, 전문가 전망 등)가 있으면 그걸 쓰고 출처를 붙여라.
4. 예상 수익률은 후보 목록의 코드만 쓰고, 의견이 없는 종목은 빼라(0으로 처리됨). 범위는 -40 ~ +40.
5. 진행 중인 충격 사건 장부(event_book)를 갱신해라. 시장을 지금 흔들고 있거나 곧 흔들 사건마다 하나씩. 아래 '이전 사건 장부'가 있으면 같은 사건은 같은 id를 유지하고, 끝난 사건은 state를 resolved로 남긴다.
   - type: war_geopolitics(전쟁·지정학) | credit(신용·금융회사 부실) | tightening(긴축·금리 충격) | trade_policy(관세·정책 충격) | pandemic | election | other
   - signal: 이전 장부 이후 새로 나온 결정적 사실이 악화면 "W", 해소면 "R", 없으면 "none".
     악화(W)는 충격을 키우는 공식 조치나 확인된 사실이다: 개전·휴전 붕괴·봉쇄, 금융회사 파산·환매 중단, 관세 발효·보복, 긴급 긴축 등.
     해소(R)는 충격의 원인을 직접 다루는 공식 조치다: 휴전·협상 타결, 중앙은행 지원·금리 인하, 구제금융, 관세 유예·합의, 계엄 해제 등.
     소문, 전망, 가격 움직임 자체는 신호가 아니다. 검색으로 확인한 사실만 쓰고 출처를 붙여라.
   - persistence: "persistent"(해소 수단이 마땅치 않아 계속 악화될 사건: 신용 연쇄 부실, 봉쇄를 동반한 전쟁, 유가 공급 충격) 또는 "transient"(정책 한 번으로 해소될 수 있는 사건)
   - mode: 이 사건만 놓고 본 권고. 악화가 켜진 사건이면 "cash", 그중 persistent이고 이미 하락이 진행 중이면 "inverse", 해소되면 "risk_on".
   - p_escalation_20d, p_resolution_20d: 앞으로 20거래일 안에 추가 악화(W)가 나올 확률, 해소(R)가 나올 확률.

이전 사건 장부 (갱신해라):
{event_book}

후보 종목 (code, 시장, 마지막 종가일, 20일 수익%, 6개월 수익%, 1년 고점 대비%, 이름):
{cands}

모델의 기본 비율 (가격 추세 모델, 이걸 기준으로 시나리오가 조정한다):
{base}

금리·시장 지표 (FRED, 최신값과 변화):
{macro}

이미 정해진 일정 (지난 7일 ~ 앞으로 35일, 실적 발표와 배당락 포함. 배당락일엔 가격이 분배금만큼 떨어지고 분배금엔 세금 15.4%가 붙는다):
{events}

출력은 JSON 하나만. 형식:
{{
 "asof": "{today}",
 "news": [{{"date": "YYYY-MM-DD", "headline": "한국어 요약", "source_url": "https://...", "why_it_matters": "한국어"}}],
 "events": [{{"date": "YYYY-MM-DD", "event": "한국어", "known_in_advance": true, "note": "한국어"}}],
 "scenarios": [
  {{"name": "한국어", "probability": 0.0, "chain": ["한국어 단계", "..."], "returns_pct": {{"CODE": 0.0}}, "checkpoints": [{{"by": "YYYY-MM-DD", "observable": "한국어", "threshold": "한국어"}}], "sources": ["https://..."]}}
 ],
 "event_book": [
  {{"id": "lower_snake_case", "name": "한국어", "type": "war_geopolitics", "markets": ["KR", "US"], "state": "escalating|stable|resolving|resolved", "signal": "W|R|none", "signal_date": "YYYY-MM-DD", "evidence": [{{"date": "YYYY-MM-DD", "headline": "한국어", "source_url": "https://..."}}], "persistence": "persistent|transient", "mode": "risk_on|cash|inverse", "p_escalation_20d": 0.0, "p_resolution_20d": 0.0, "note": "한국어"}}
 ],
 "notes": "한국어, 가장 불확실한 부분"
}}
"""

EVENT_TYPES = {"war_geopolitics", "credit", "tightening", "trade_policy", "pandemic", "election", "other"}
EVENT_CALLS = os.path.join(RUNS, "event_calls.jsonl")


def clean_event_book(book):
    out = []
    for e in book if isinstance(book, list) else []:
        if not isinstance(e, dict) or not isinstance(e.get("id"), str):
            continue
        eid = __import__("re").sub(r"[^a-z0-9]+", "_", e["id"].lower()).strip("_")[:60]
        if not eid:
            continue
        pick = lambda k, ok, d: e.get(k) if e.get(k) in ok else d
        prob = lambda k: min(1.0, max(0.0, float(e.get(k)))) if isinstance(e.get(k), (int, float)) else None
        ev = [x for x in (e.get("evidence") or []) if isinstance(x, dict) and str(x.get("source_url", "")).startswith("http")][:8]
        out.append(dict(id=eid, name=str(e.get("name", ""))[:200], type=pick("type", EVENT_TYPES, "other"),
                        markets=[m for m in (e.get("markets") or []) if m in ("KR", "US")] or ["KR", "US"],
                        state=pick("state", {"escalating", "stable", "resolving", "resolved"}, "stable"), signal=pick("signal", {"W", "R", "none"}, "none"),
                        signal_date=str(e.get("signal_date", ""))[:10], evidence=ev, persistence=pick("persistence", {"persistent", "transient"}, "transient"),
                        mode=pick("mode", {"risk_on", "cash", "inverse"}, "risk_on"), p_escalation_20d=prob("p_escalation_20d"), p_resolution_20d=prob("p_resolution_20d"),
                        note=str(e.get("note", ""))[:1000]))
    return out


def last_event_book():
    if not os.path.exists(LOG):
        return []
    book = []
    for line in open(LOG, encoding="utf-8").read().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("kind") == "scenario" and isinstance(r.get("event_book"), list):
            book = r["event_book"]
    return book


def record_event_calls(today, asof, trigger, book):
    with open(EVENT_CALLS, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(ts=now().isoformat(timespec="seconds"), day=today, asof=asof, trigger=trigger, event_book=book), ensure_ascii=False) + "\n")


INVERSE_PAIRS = {"114800": "069500", "SH": "SPY"}
INV_DRAG_20D = 0.0075
SIDE_COST = {"KR": 0.0015, "US": 0.002}


def validate(js, codes, horizon=20):
    sc = [s for s in js.get("scenarios", []) if isinstance(s, dict)]
    if not sc:
        raise ValueError("no scenarios")
    p = np.array([max(0.0, float(s.get("probability", 0))) for s in sc])
    if p.sum() <= 0:
        raise ValueError("zero probability")
    p = p / p.sum()
    R = np.zeros((len(sc), len(codes)))
    for k, s in enumerate(sc):
        for c, v in (s.get("returns_pct") or {}).items():
            if c in codes:
                R[k, codes.index(c)] = max(-RMAX, min(RMAX, float(v))) / 100
    for inv, und in INVERSE_PAIRS.items():
        if inv in codes and und in codes:
            R[:, codes.index(inv)] = -R[:, codes.index(und)] - INV_DRAG_20D * horizon / 20
    return p, R


def net_inverse(w, codes):
    w = w.copy()
    for inv, und in INVERSE_PAIRS.items():
        if inv in codes and und in codes:
            i, u = codes.index(inv), codes.index(und)
            net = w[u] - w[i]
            w[u], w[i] = max(net, 0.0), max(-net, 0.0)
    return w


PER_NAME = float(os.environ.get("NOSOCK_SCENARIO_PER_NAME", 0.15))


def current_cap():
    try:
        return max(0.0, min(0.5, float(open(os.path.join(RUNS, "scenario_cap.txt"), encoding="utf-8").read().strip())))
    except (OSError, ValueError):
        return CAP


SIGMA_REF = float(os.environ.get("NOSOCK_SCENARIO_SIGMA_REF", 0.20))


def realized_vols(inst, codes, D, days=60):
    import bisect
    out = {}
    for c in codes:
        ins = inst.get(c)
        if ins is None:
            continue
        i = bisect.bisect_right(ins["days"], D) - 1
        if i < days + 1:
            continue
        C = np.log(np.asarray(ins["C"][i - days:i + 1], float))
        out[c] = round(float(np.diff(C).std() * np.sqrt(252)), 4)
    return out


def optimize(w0, p, R, cap=CAP, per_name=PER_NAME, codes=None, vols=None):
    from scipy.optimize import minimize
    n = len(w0)
    free = 1 - w0.sum()
    cost = np.array([SIDE_COST["KR" if c[:1].isdigit() else "US"] for c in codes]) if codes else np.full(n, SIDE_COST["US"])

    def wof(x):
        return w0 + x[:n] - x[n:]

    def obj(x):
        w = wof(x)
        port = 1 + R @ w
        return -float(p @ np.log(np.maximum(port, 1e-6))) + float(cost @ (x[:n] + x[n:]))

    cons = [dict(type="ineq", fun=lambda x: wof(x)), dict(type="ineq", fun=lambda x: free - (x[:n].sum() - x[n:].sum())), dict(type="ineq", fun=lambda x: cap - x[:n].sum()), dict(type="ineq", fun=lambda x: cap - x[n:].sum())]
    if codes and vols and all(c in vols for c in codes):
        v = np.array([vols[c] for c in codes])
        cons.append(dict(type="ineq", fun=lambda x: cap * SIGMA_REF - float(v @ x[:n])))
        up_bound = cap
    else:
        up_bound = per_name
    r = minimize(obj, np.zeros(2 * n), method="SLSQP", bounds=[(0, up_bound)] * n + [(0, 1)] * n, constraints=cons, options=dict(maxiter=500, ftol=1e-12))
    w = np.maximum(wof(r.x), 0)
    if codes:
        w = net_inverse(w, codes)
    return w, r.success


def base_weights(D, inst, groups, extra):
    st = {}
    try:
        st = json.load(open(os.path.join(RUNS, "live_nosock_state.json"), encoding="utf-8"))
    except (OSError, ValueError):
        pass
    cyc = st.get("current")
    if cyc and cyc in st.get("cycles", {}):
        return st["cycles"][cyc]["weights"], f"live cycle {cyc}"
    import live_nosock as L
    w, why, markets, m = L.decide(inst, groups, D, extra)
    return w, "decide"


def run(trigger="morning"):
    import alloc_shadow
    import live_nosock as L
    t0 = time.time()
    t = alloc_shadow.read_only_client()
    extra = sorted(set(L.listfile("adopt.txt")) | set(L.listfile("candidates_extra.txt")))
    inst, groups, fx_now, fetched = L.build_market(t, extra)
    today = now().strftime("%Y-%m-%d")
    D = max(d for d in inst["069500"]["days"] if d < today)
    codes = [c for c in list(groups["U29"]) + extra if c in inst]
    names = names_map()
    w0d, base_src = base_weights(D, inst, groups, extra)
    snap = snapshot(inst, codes, D)
    codes = [x["code"] for x in snap]
    json.dump(codes, open(CODES, "w", encoding="utf-8"))
    macro = macro_snapshot(today)
    events = upcoming(today)
    cands = "\n".join(f"{x['code']}, {x['market']}, {x['last']}, {x['r20']:+.1f}, {x['r126']:+.1f}, {x['dd252']:+.1f}, {names.get(x['code'], x['code'])}" for x in snap)
    base = "\n".join(f"{c} {100 * w:.1f}%" for c, w in sorted(w0d.items(), key=lambda kv: -kv[1])) + f"\n현금 {100 * max(0.0, 1 - sum(w0d.values())):.1f}%"
    prev_book = last_event_book()
    prompt = PROMPT.format(today=today, now=now().strftime("%H:%M"), asof=D, cands=cands, base=base, macro=json.dumps(macro, ensure_ascii=False), events=json.dumps(events, ensure_ascii=False),
                           event_book=json.dumps(prev_book, ensure_ascii=False) if prev_book else "(없음, 처음 작성)")
    issues_text, issue_ids = recent_issues()
    if issues_text:
        prompt += "\n\n서버 Codex가 모아 둔 최근 이슈 심층 분석 (최근 36시간). 이걸 1차 재료로 쓰되, 중요한 사실은 웹으로 다시 확인해라:\n" + issues_text
    text, usage = codex(prompt)
    try:
        js = parse_json(text)
        p, R = validate(js, codes)
    except Exception as e:
        log("scenario_error", day=today, error=type(e).__name__, raw=text[:20000], usage=usage)
        raise
    w0 = np.array([w0d.get(c, 0.0) for c in codes])
    vols = realized_vols(inst, codes, D)
    for inv, und in INVERSE_PAIRS.items():
        if und in vols:
            vols[inv] = vols[und]
    w, ok = optimize(w0, p, R, current_cap(), PER_NAME, codes, vols)
    eb, et = float(p @ np.log(1 + R @ w0)), float(p @ np.log(1 + R @ w))
    worst_b, worst_t = float((R @ w0).min()), float((R @ w).min())
    tilt = {c: round(float(a - b), 4) for c, a, b in zip(codes, w, w0) if abs(a - b) > 0.002}
    book = clean_event_book(js.get("event_book"))
    record_event_calls(today, D, trigger, book)
    row = log("scenario", day=today, asof=D, issue_ids=issue_ids, trigger=trigger, fetched_at=fetched, base_source=base_src, cap=current_cap(), model=MODEL, usage=usage, prompt_sha=hashlib.sha256(prompt.encode()).hexdigest(), scenarios=js.get("scenarios"), news=js.get("news"), events=js.get("events"), event_book=book, notes=js.get("notes"),
              probabilities=[float(x) for x in p], base_weights={c: float(x) for c, x in zip(codes, w0) if x > 0}, tilted_weights={c: float(x) for c, x in zip(codes, w) if x > 1e-4}, tilt=tilt, moved=float(np.clip(w - w0, 0, None).sum()), opt_ok=bool(ok),
              exp_log_base=eb, exp_log_tilt=et, worst_base=worst_b, worst_tilt=worst_t, vols=vols, sigma_ref=SIGMA_REF, paper=True, secs=round(time.time() - t0, 1), raw=text)
    print(f"{today} 기준 {D}, 시나리오 {len(p)}개, 걸린 시간 {row['secs']}s")
    for s, q in zip(js["scenarios"], p):
        print(f"  {100 * q:4.0f}% {s.get('name')}: " + " → ".join(s.get("chain", [])[:5]))
    print(f"  기대 로그성장 기본 {eb:+.4f} → 조정 {et:+.4f}, 최악 시나리오 {100 * worst_b:+.1f}% → {100 * worst_t:+.1f}%, 움직인 비율 {100 * row['moved']:.0f}%")
    for c, d in sorted(tilt.items(), key=lambda kv: kv[1]):
        print(f"    {c:7s} {names.get(c, c)[:20]:20s} {100 * d:+.1f}%p")
    return row


SSH = [r"C:\Windows\System32\OpenSSH\ssh.exe", "-o", "BatchMode=yes", "-p", os.environ.get("NOSOCK_SSH_PORT", "22"), "-i", os.path.expanduser(os.environ.get("NOSOCK_SSH_KEY", r"~\.ssh\id_ed25519")), os.environ.get("NOSOCK_SSH_DEST", "user@host")]
PULL = os.path.join(RUNS, "scenario_pull.json")
ISSUES = os.path.join(RUNS, "issues")
URGENT = os.path.join(RUNS, "scenario_urgent.flag")
CODES = os.path.join(RUNS, "scenario_codes.json")
STOP = os.path.join(RUNS, "live_nosock.stop")


def ssh(cmd, timeout=60):
    if os.environ.get("NOSOCK_LOCAL_INBOX"):
        r = subprocess.run(["sh", "-c", cmd.replace("sudo -n ", "")], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, stdin=subprocess.DEVNULL)
        if r.returncode != 0:
            raise RuntimeError(f"local exit {r.returncode}")
        return r.stdout
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        raise RuntimeError(f"ssh exit {r.returncode}")
    return r.stdout


def light_base():
    try:
        st = json.load(open(os.path.join(RUNS, "live_nosock_state.json"), encoding="utf-8"))
        cyc = st.get("current")
        if cyc:
            return st["cycles"][cyc]["weights"], f"live cycle {cyc}"
    except (OSError, ValueError, KeyError):
        pass
    if os.path.exists(LOG):
        for line in reversed(open(LOG, encoding="utf-8").read().splitlines()):
            r = json.loads(line)
            if r.get("kind") == "scenario" and r.get("base_weights"):
                return r["base_weights"], f"last desk {r['day']}"
    return {}, "none"


def process_external(js, source, sid):
    base, src = light_base()
    allowed = set(json.load(open(CODES, encoding="utf-8"))) if os.path.exists(CODES) else set(base)
    codes = sorted(allowed | set(base))
    p, R = validate(js, codes)
    w0 = np.array([base.get(c, 0.0) for c in codes])
    w, ok = optimize(w0, p, R, current_cap(), PER_NAME, codes)
    tilt = {c: round(float(a - b), 4) for c, a, b in zip(codes, w, w0) if abs(a - b) > 0.002}
    unknown = sorted({c for s in js.get("scenarios", []) for c in (s.get("returns_pct") or {})} - set(codes))
    return log("scenario", day=now().strftime("%Y-%m-%d"), source=source, external_id=sid, base_source=src, cap=current_cap(), trigger=js.get("trigger"), scenarios=js.get("scenarios"), news=js.get("news"), notes=js.get("notes"), probabilities=[float(x) for x in p],
               base_weights={c: float(x) for c, x in zip(codes, w0) if x > 0}, tilted_weights={c: float(x) for c, x in zip(codes, w) if x > 1e-4}, tilt=tilt, moved=float(np.clip(w - w0, 0, None).sum()), opt_ok=bool(ok),
               exp_log_base=float(p @ np.log(1 + R @ w0)), exp_log_tilt=float(p @ np.log(1 + R @ w)), worst_base=float((R @ w0).min()), worst_tilt=float((R @ w).min()), unknown_codes=unknown, paper=True)


def pull():
    seen = set(json.load(open(PULL, encoding="utf-8"))) if os.path.exists(PULL) else set()
    total = 0
    for kind in ("scenarios", "issues", "requests"):
        names = [x.strip() for x in ssh(f"sudo -n ls /var/lib/nosock/web/{kind}/ 2>/dev/null || true").split() if x.strip().endswith(".json")]
        new = [n for n in sorted(names) if f"{kind}/{n}" not in seen]
        for n in new:
            try:
                rec = json.loads(ssh(f"sudo -n cat /var/lib/nosock/web/{kind}/{n}"))
                if kind == "requests":
                    log("desk_requested", external_id=rec.get("id", n), reason=rec.get("reason"))
                    open(URGENT, "w").write("dot_request:" + rec.get("id", n))
                elif kind == "scenarios":
                    process_external(rec["payload"], rec.get("source", "dot"), rec.get("id", n))
                else:
                    os.makedirs(ISSUES, exist_ok=True)
                    json.dump(rec, open(os.path.join(ISSUES, n), "w", encoding="utf-8"), ensure_ascii=False)
                    iss = rec["payload"].get("issues", [])
                    urgent = any(str(x.get("urgency", "")).lower() in ("high", "긴급") for x in iss)
                    log("issues_received", external_id=rec.get("id", n), source=rec.get("source"), n=len(iss), urgent=urgent, titles=[x.get("title") for x in iss][:20])
                    if urgent:
                        open(URGENT, "w").write(rec.get("id", n))
            except Exception as e:
                log("scenario_error", source=kind, external_id=n, error=type(e).__name__)
            seen.add(f"{kind}/{n}")
            json.dump(sorted(seen), open(PULL, "w", encoding="utf-8"))
        total += len(new)
    return total


def recent_issues(hours=36, limit_chars=45000):
    if not os.path.isdir(ISSUES):
        return "", []
    cut = time.time() - hours * 3600
    files = sorted((f for f in os.listdir(ISSUES) if f.endswith(".json")), reverse=True)
    parts, ids, used = [], [], 0
    for f in files:
        fp = os.path.join(ISSUES, f)
        if os.path.getmtime(fp) < cut:
            continue
        rec = json.load(open(fp, encoding="utf-8"))
        for x in rec["payload"].get("issues", []):
            aff = "; ".join(f"{c} {v.get('direction')}" + (f" {v.get('magnitude_pct')}%" if v.get("magnitude_pct") is not None else "") + (f" ({v.get('why')})" if v.get("why") else "") for c, v in (x.get("affected") or {}).items())
            tag = f", 사건 {x.get('event_id')} {x.get('event_type', '')} 신호 {x.get('signal', '-')} {x.get('signal_date', '')}" if x.get("event_id") else ""
            block = f"### {x.get('title')} [{x.get('category')}, {x.get('status')}, 긴급도 {x.get('urgency', '-')}{tag}]\n요약: {x.get('summary')}\n분석: {x.get('analysis', '')[:4000]}\n영향 종목: {aff}\n주요 날짜: {x.get('key_dates', '')}\n지켜볼 것: {x.get('watch', '')}\n출처: {' '.join(x.get('sources', [])[:6])}\n"
            if used + len(block) > limit_chars:
                break
            parts.append(block)
            used += len(block)
        ids.append(rec.get("id", f))
    return "\n".join(parts), ids


def desk_done(today):
    if not os.path.exists(LOG):
        return False
    for line in open(LOG, encoding="utf-8").read().splitlines():
        r = json.loads(line)
        if r.get("kind") == "scenario" and r.get("day") == today and not r.get("source") and r.get("trigger", "morning") == "morning":
            return True
    return False


DROPBOX = os.environ.get("NOSOCK_DROPBOX", "")
CODE_OK = __import__("re").compile(r"^[A-Z0-9.]{1,12}$")


def check_issue_pack(js):
    if not isinstance(js, dict):
        return "body must be an object"
    iss = js.get("issues")
    if not isinstance(iss, list) or not 1 <= len(iss) <= 40:
        return "issues: 1-40 items"
    for x in iss:
        if not isinstance(x, dict):
            return "issue must be an object"
        for k, lim in (("title", 300), ("category", 60), ("status", 60), ("summary", 3000), ("analysis", 20000)):
            if not isinstance(x.get(k), str) or len(x[k]) > lim:
                return f"issue.{k}"
        src = x.get("sources", [])
        if not isinstance(src, list) or not src or len(src) > 30 or any(not isinstance(u, str) or not u.startswith("http") for u in src):
            return "issue.sources (1-30 urls)"
        aff = x.get("affected", {})
        if not isinstance(aff, dict) or len(aff) > 60:
            return "issue.affected"
        for c, v in aff.items():
            if not CODE_OK.match(str(c)) or not isinstance(v, dict) or v.get("direction") not in ("up", "down", "mixed", "unclear"):
                return "issue.affected entry"
    return None


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=float)
    os.chmod(tmp, 0o660)
    os.replace(tmp, path)


def dropbox_context():
    base, src = light_base()
    rows = [json.loads(x) for x in open(LOG, encoding="utf-8")] if os.path.exists(LOG) else []
    desk = next((r for r in reversed(rows) if r.get("kind") == "scenario"), None)
    titles = [dict(ts=r["ts"], titles=r.get("titles"), urgent=r.get("urgent")) for r in rows if r.get("kind") == "issues_received"][-15:]
    cap = current_cap()
    cands = json.load(open(CODES, encoding="utf-8")) if os.path.exists(CODES) else sorted(base)
    nm = names_map()
    return dict(asof=now().isoformat(timespec="seconds"), candidates=cands, candidate_names={c: nm.get(c, c) for c in cands}, model_weights=base, model_source=src, live_scenario_cap=cap,
                upcoming_events=upcoming(now().strftime("%Y-%m-%d")), recent_issue_batches=titles,
                latest_scenario=dict(ts=desk["ts"], trigger=desk.get("trigger"), probabilities=desk.get("probabilities"), names=[s.get("name") for s in desk.get("scenarios", [])], notes=desk.get("notes"),
                                     event_book=[{k: e.get(k) for k in ("id", "name", "type", "state", "signal", "signal_date", "persistence", "mode", "p_escalation_20d", "p_resolution_20d")} for e in desk.get("event_book") or []]) if desk else None,
                how_to=dict(submit_issues="write JSON to in/issues_<anything>.json", request_desk="write {\"reason\": \"...\"} to in/desk_<anything>.json", submit_scenarios="write JSON to in/scenario_<anything>.json", results="see done/<same name>.result.json"))


def dropbox_cycle(state):
    if not DROPBOX or not os.path.isdir(DROPBOX):
        return
    inbox, done = os.path.join(DROPBOX, "in"), os.path.join(DROPBOX, "done")
    os.makedirs(inbox, exist_ok=True)
    os.makedirs(done, exist_ok=True)
    for name in sorted(os.listdir(inbox)):
        p = os.path.join(inbox, name)
        if not name.endswith(".json") or not os.path.isfile(p) or time.time() - os.path.getmtime(p) < 5:
            continue
        res = dict(file=name, processed_at=now().isoformat(timespec="seconds"))
        try:
            if os.path.getsize(p) > 1_100_000:
                raise ValueError("file larger than 1.1 MB")
            js = json.load(open(p, encoding="utf-8"))
            rid = "dropbox-" + now().strftime("%Y%m%d-%H%M%S") + "-" + os.path.splitext(name)[0][:40]
            if name.startswith("issues"):
                err = check_issue_pack(js)
                if err:
                    raise ValueError(err)
                os.makedirs(ISSUES, exist_ok=True)
                json.dump(dict(id=rid, kind="issues", received_at=res["processed_at"], source=str(js.get("author", "dot-dropbox"))[:40], payload=js), open(os.path.join(ISSUES, rid + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
                iss = js["issues"]
                urgent = any(str(x.get("urgency", "")).lower() in ("high", "긴급") for x in iss)
                log("issues_received", external_id=rid, source="dropbox", n=len(iss), urgent=urgent, titles=[x.get("title") for x in iss][:20])
                if urgent:
                    open(URGENT, "w").write(rid)
                res.update(ok=True, id=rid, urgent=urgent)
            elif name.startswith("desk"):
                if time.time() - state.get("last_desk_req", 0) < 3600:
                    raise ValueError("a desk run was requested within the last hour")
                state["last_desk_req"] = time.time()
                log("desk_requested", external_id=rid, source="dropbox", reason=str(js.get("reason", ""))[:500])
                open(URGENT, "w").write("dropbox_request:" + rid)
                res.update(ok=True, id=rid, note="desk run starts within about a minute and takes 5-10 minutes; then read latest_scenario.json")
            elif name.startswith("scenario"):
                process_external(js, "dropbox", rid)
                res.update(ok=True, id=rid)
            else:
                raise ValueError("file name must start with issues, desk or scenario")
        except Exception as e:
            res.update(ok=False, error=str(e)[:300] if isinstance(e, ValueError) else type(e).__name__)
        write_json(os.path.join(done, name.replace(".json", "") + ".result.json"), res)
        os.replace(p, os.path.join(done, name))
    if time.time() - state.get("last_ctx", 0) > 300:
        ctx = dropbox_context()
        write_json(os.path.join(DROPBOX, "context.json"), ctx)
        ref = os.path.join(DROPBOX, "ref")
        os.makedirs(ref, exist_ok=True)
        os.chmod(ref, 0o2770)
        write_json(os.path.join(ref, "event_book.json"), dict(asof=ctx["asof"], event_book=last_event_book()))
        for fn in ("calendar.json", "corporate.json"):
            src = os.path.join(HERE, "data", "events", fn)
            if os.path.exists(src):
                write_json(os.path.join(ref, fn), json.load(open(src, encoding="utf-8")))
        md = "# 후보 종목 (영향 종목은 이 코드만 쓴다)\n" + "".join(f"- {c}: {ctx['candidate_names'].get(c, c)}\n" for c in ctx["candidates"])
        tmp = os.path.join(ref, "candidates.md.tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(md)
        os.chmod(tmp, 0o660)
        os.replace(tmp, os.path.join(ref, "candidates.md"))
        rows = [json.loads(x) for x in open(LOG, encoding="utf-8")] if os.path.exists(LOG) else []
        desk = next((r for r in reversed(rows) if r.get("kind") == "scenario"), None)
        if desk:
            desk = {k: v for k, v in desk.items() if k != "raw"}
            write_json(os.path.join(DROPBOX, "latest_scenario.json"), desk)
        state["last_ctx"] = time.time()


def watch():
    log("scenario_watch_start", cap=CAP, model=MODEL, dropbox=DROPBOX or None)
    last_pull, fail_until, last_urgent = 0.0, 0.0, 0.0
    dstate = {}
    while not os.path.exists(STOP):
        n = now()
        today = n.strftime("%Y-%m-%d")
        try:
            dropbox_cycle(dstate)
        except Exception as e:
            log("scenario_error", where="dropbox", error=type(e).__name__)
        if time.time() - last_pull > 300:
            try:
                pull()
            except Exception as e:
                log("scenario_error", where="pull", error=type(e).__name__)
            last_pull = time.time()
        if n.weekday() < 5 and DESK_AT <= (n.hour, n.minute) < (15, 0) and time.time() > fail_until and not desk_done(today):
            try:
                run()
            except Exception as e:
                log("scenario_error", where="run", day=today, error=type(e).__name__)
                fail_until = time.time() + 1800
        if os.path.exists(URGENT) and time.time() > fail_until and time.time() - last_urgent > 3600:
            src = open(URGENT).read().strip()
            os.remove(URGENT)
            last_urgent = time.time()
            try:
                run(trigger=f"urgent:{src}")
            except Exception as e:
                log("scenario_error", where="urgent_run", error=type(e).__name__)
        time.sleep(60)
    log("scenario_watch_stop")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        run()
    elif cmd == "pull":
        print(pull(), "new")
    elif cmd == "watch":
        watch()
