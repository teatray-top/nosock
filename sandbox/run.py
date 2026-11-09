import itertools
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
import numpy as np

from sandbox import world as W

OUT = os.environ.get("NOSOCK_SANDBOX_OUT", os.path.join(ROOT, "runs", "sandbox"))
TOOLS = os.path.join(HERE, "tools")
CAP = float(os.environ.get("NOSOCK_SANDBOX_CAP", 0.175))
COST = 0.001
ALPHA = 0.6
BASE = ("HB200", "HBSEMI", "HBBOND3", "DSIDX", "DSTECH", "DSSEMI", "OIL", "ENRG", "GOLD", "DEFN", "TBOND", "USDX")
MODEL = os.environ.get("NOSOCK_SCENARIO_MODEL", "gpt-6.1-sol")


def codes():
    return list(W.INST)


def open_events(w, t):
    return [e for e in w["events"] if e["announce"] <= t < e["resolve"]]


def inst_return(f):
    out = {}
    for c, spec in W.INST.items():
        if "inverse" in spec:
            continue
        out[c] = sum(spec["load"].get(k, 0.0) * f[i] for i, k in enumerate(W.FACTORS))
    for c, spec in W.INST.items():
        if "inverse" in spec:
            out[c] = -out[spec["inverse"]]
    return out


def model_agent(w, t, use_true):
    evs = open_events(w, t)[:3]
    if not evs:
        return None
    per = []
    for e in evs:
        q = np.array(w["consensus"][e["id"]].get(str(t), w["consensus"][e["id"]].get(t, [1 / len(e["branches"])] * len(e["branches"]))))
        J = np.array([W.factor_impact(b["jump"]) for b in e["branches"]])
        priced = ALPHA * (q @ J)
        p = np.array([b["p"] for b in e["branches"]]) if use_true else q
        per.append([(e["id"], b["key"], float(pk), J[k] - priced) for k, (b, pk) in enumerate(zip(e["branches"], p))])
    scen, probs = [], []
    for combo in itertools.product(*per):
        pr = float(np.prod([x[2] for x in combo]))
        f = sum(x[3] for x in combo)
        r = inst_return(f)
        scen.append(dict(name=" / ".join(f"{x[0]}:{x[1]}" for x in combo), probability=pr, chain=[], returns_pct={c: 100 * v for c, v in r.items()}))
        probs.append(pr)
    ev_probs = {e["id"]: {b["key"]: float(x[2]) for b, x in zip(e["branches"], pe)} for e, pe in zip(evs, per)}
    return dict(scenarios=scen, event_probs=ev_probs)


def write_workspace(w, t, wd):
    if os.path.exists(wd):
        shutil.rmtree(wd)
    os.makedirs(wd)
    with open(os.path.join(wd, "news.jsonl"), "w", encoding="utf-8") as f:
        for n in w["news"]:
            if n["day"] <= t:
                f.write(json.dumps(dict(n, date=W.day_label(w["start"], n["day"])), ensure_ascii=False) + "\n")
    with open(os.path.join(wd, "prices.csv"), "w", encoding="utf-8") as f:
        cs = codes()
        f.write("date," + ",".join(cs) + "\n")
        for d in range(t + 1):
            f.write(W.day_label(w["start"], d) + "," + ",".join(f"{w['prices'][c][d]:.3f}" for c in cs) + "\n")
    for fn in os.listdir(TOOLS):
        shutil.copy(os.path.join(TOOLS, fn), wd)


PROMPT = """이것은 완전히 가상의 세계다. 등장하는 나라(한반, 대서, 카라드, 동방), 회사, 사람, 종목은 모두 지어낸 것이다. 현실 세계의 지식은 쓰지 마라. 인터넷은 없다.
오늘은 가상 날짜 {today} (시작일부터 {t}거래일차)이고 장 마감 직후다. 작업 폴더의 도구로만 정보를 찾아라:
- python search.py 키워드 [키워드 ...]  : 오늘까지 나온 기사 검색 (최신순)
- python search.py --day {today}        : 특정 날짜 기사 전부
- python prices.py 종목코드 [일수]       : 종가 기록
- news.jsonl, prices.csv 파일을 직접 읽어도 된다.

너는 레버리지 없이 매수만 하는 계좌의 시나리오 데스크다. 기본 비율은 아래 12종목 균등이고, 너의 시나리오에 따라 최대 30%까지 비율을 옮긴다.
시장은 공개된 예측시장 확률만큼 사건을 미리 가격에 반영해 두었다. 시장보다 확률을 더 잘 읽어야 돈을 번다.
배당락일에는 가격이 분배금만큼 떨어지고 분배금에는 세금 15.4%가 붙는다. 실적 발표, 선거, 중앙은행 회의, 돌발 사건 모두 고려해라.

종목 (코드: 이름):
{cands}

지금 열려 있는 사건 (id: 가능한 결과 키):
{events}

할 일:
1. 기사를 검색해 각 열린 사건의 결과 확률을 추정해라 (여론조사·설문·예측시장·소문의 신뢰도를 따져서).
2. 다음 5거래일 동안의 '함께 일어나는 큰 그림' 시나리오를 2~6개 써라. 확률 합 1. 인과 사슬과 각 종목의 5거래일 예상 수익률(%)을 써라. 의견 없는 종목은 빼라.

JSON 하나만 출력:
{{"event_probs": {{"사건id": {{"결과키": 0.0}}}}, "scenarios": [{{"name": "한국어", "probability": 0.0, "chain": ["단계"], "returns_pct": {{"코드": 0.0}}}}], "notes": "한국어"}}
"""


def codex_agent(w, t, wd):
    from scenario_desk import codex_exe, parse_json
    write_workspace(w, t, wd)
    evs = open_events(w, t)
    cands = "\n".join(f"{c}: {s['name']}" for c, s in W.INST.items())
    ev = "\n".join(f"{e['id']}: " + ", ".join(f"{b['key']}({b['label']})" for b in e["branches"]) for e in evs) or "(없음)"
    prompt = PROMPT.format(today=W.day_label(w["start"], t), t=t, cands=cands, events=ev)
    last = os.path.join(wd, "_last.txt")
    args = [codex_exe(), "-m", MODEL, "-c", 'web_search="disabled"', "-c", "sandbox_workspace_write.network_access=false", "exec", "--skip-git-repo-check", "--ephemeral", "--json", "-s", "workspace-write", "-C", wd, "-o", last, "-"]
    t0 = time.time()
    r = subprocess.run(args, input=prompt, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=wd, timeout=2400, **({"creationflags": 0x40} if os.name == "nt" else {}))
    text = open(last, encoding="utf-8").read() if os.path.exists(last) else ""
    if r.returncode != 0 or not text:
        raise RuntimeError(f"codex exit {r.returncode}")
    js = parse_json(text)
    js["_web_calls"] = sum(1 for line in r.stdout.splitlines() if '"web_search' in line)
    js["_secs"] = round(time.time() - t0, 1)
    js["_raw"] = text
    return js


def world_vols(w, t, min_days=10):
    if w is None or t is None:
        return None
    hi = max(t, min_days)
    return {c: float(np.std(w["returns"][c][1:hi + 1]) * np.sqrt(252)) for c in codes()}


def tilt(js, w0, w=None, t=None):
    import scenario_desk as SD
    SD.INVERSE_PAIRS = {"HBINV": "HB200", "DSINV": "DSIDX"}
    cs = codes()
    if not js or not js.get("scenarios"):
        return w0.copy(), None
    p, R = SD.validate(js, cs, horizon=5)
    w, ok = SD.optimize(w0, p, R, CAP, SD.PER_NAME, cs, world_vols(w, t))
    return w, dict(p=p.tolist(), ok=bool(ok))


def simulate(w, decisions):
    cs = codes()
    days = w["days"]
    R = np.array([w["returns"][c] for c in cs]).T
    div = np.zeros_like(R)
    for c, lst in w["dividends"].items():
        for d in lst:
            div[d["ex"], cs.index(c)] = d["amount"] * (1 - w["div_tax"])
    w0 = np.array([1 / len(BASE) if c in BASE else 0.0 for c in cs])
    out = {}
    for name in ("base", "tilt"):
        nav, cur, hist = 1.0, w0.copy(), [1.0]
        for t in range(days - 1):
            if name == "tilt" and t in decisions:
                tgt = decisions[t]
                nav *= 1 - COST * np.abs(tgt - cur).sum()
                cur = tgt.copy()
            g = cur @ (R[t + 1] + div[t + 1]) + (1 - cur.sum()) * 0.0
            nav *= 1 + g
            cur = cur * (1 + R[t + 1] + div[t + 1]) / (1 + g)
            hist.append(nav)
        h = np.array(hist)
        out[name] = dict(total=float(h[-1] - 1), maxdd=float((h / np.maximum.accumulate(h) - 1).min()), path=h.tolist())
    return out


def brier(w, logs):
    rows = []
    for t, js in logs.items():
        for e in open_events(w, t):
            ep = (js or {}).get("event_probs", {}).get(e["id"])
            k = len(e["branches"])
            y = np.eye(k)[e["outcome"]]
            q = np.array(w["consensus"][e["id"]].get(str(t), w["consensus"][e["id"]].get(t, [1 / k] * k)))
            pt = np.array([b["p"] for b in e["branches"]])
            a = None
            if isinstance(ep, dict):
                a = W.norm([float(ep.get(b["key"], 0.0)) for b in e["branches"]])
            rows.append(dict(t=t, event=e["id"], agent=None if a is None else float(((a - y) ** 2).sum()), market=float(((q - y) ** 2).sum()), true=float(((pt - y) ** 2).sum())))
    have = [r for r in rows if r["agent"] is not None]
    m = lambda k, rs: float(np.mean([r[k] for r in rs])) if rs else None
    return dict(n=len(rows), n_agent=len(have), agent=m("agent", have), market=m("market", have), true=m("true", have), market_all=m("market", rows))


def decision_days(w, every=5):
    ds = set(range(1, w["days"] - 1, every))
    for e in w["events"]:
        for d in (e["announce"], e["resolve"] - 1, e["resolve"] - 3):
            if 0 < d < w["days"] - 1:
                ds.add(d)
    return sorted(ds)


def run_world(seed, agent, every=5, days=60):
    w = W.make_world(seed, days)
    cs = codes()
    w0 = np.array([1 / len(BASE) if c in BASE else 0.0 for c in cs])
    wid = f"w{seed:03d}_{agent}_{time.strftime('%Y%m%d_%H%M%S')}"
    d = os.path.join(OUT, wid)
    os.makedirs(d, exist_ok=True)
    logs, decisions = {}, {}
    for t in decision_days(w, every):
        try:
            if agent == "codex":
                js = codex_agent(w, t, os.path.join(d, "work"))
            elif agent == "oracle":
                js = model_agent(w, t, True)
            elif agent == "market":
                js = model_agent(w, t, False)
            else:
                js = None
        except Exception as e:
            js = dict(error=type(e).__name__)
        tw, info = tilt(js, w0, w, t) if js and not js.get("error") else (w0.copy(), None)
        decisions[t] = tw
        logs[t] = js
        with open(os.path.join(d, "decisions.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(t=t, date=W.day_label(w["start"], t), weights={c: round(float(x), 4) for c, x in zip(cs, tw) if x > 1e-4}, info=info, agent=js), ensure_ascii=False, default=float) + "\n")
    sim = simulate(w, decisions)
    sc = brier(w, logs)
    rep = dict(seed=seed, agent=agent, days=days, every=every, decisions=len(decisions), base=dict(total=sim["base"]["total"], maxdd=sim["base"]["maxdd"]), tilt=dict(total=sim["tilt"]["total"], maxdd=sim["tilt"]["maxdd"]), brier=sc,
               events=[dict(id=e["id"], announce=e["announce"], resolve=e["resolve"], outcome=e["branches"][e["outcome"]]["key"], p_true=[round(b["p"], 3) for b in e["branches"]]) for e in w["events"]])
    json.dump(dict(rep, nav_base=sim["base"]["path"], nav_tilt=sim["tilt"]["path"]), open(os.path.join(d, "report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return rep


if __name__ == "__main__":
    agent = sys.argv[1] if len(sys.argv) > 1 else "oracle"
    seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1]
    every = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    for s in seeds:
        r = run_world(s, agent, every)
        b = r["brier"]
        print(f"seed {s} {agent}: tilt {100 * r['tilt']['total']:+.2f}% (dd {100 * r['tilt']['maxdd']:.1f}%) vs base {100 * r['base']['total']:+.2f}% (dd {100 * r['base']['maxdd']:.1f}%), brier agent {b['agent']} market {b['market']} true {b['true']}, decisions {r['decisions']}", flush=True)
