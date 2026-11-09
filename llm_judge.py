import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from toss import ApiError, Toss
import jev_client as J

HERE = os.path.dirname(os.path.abspath(__file__))
ET, KST = ZoneInfo("America/New_York"), ZoneInfo("Asia/Seoul")
OPENCODE = os.environ.get("OPENCODE_EXE", r"G:\AI\opencode-llm-mcp\node_modules\opencode-ai\bin\opencode.exe")
MODEL = os.environ.get("LLM_MODEL", "opencode/glm-5.3-flash-free")
BACKEND = os.environ.get("LLM_BACKEND", "opencode")
SRC_LOG = os.environ.get("LLM_SRC_LOG", os.path.join(HERE, "runs", "live_us_20261105.jsonl"))
DAY = os.environ.get("LLM_DAY", "20261105")
TAG = MODEL.split("/")[-1]
OUT = os.path.join(HERE, "runs", f"llm_judge_{TAG}_{DAY}.jsonl")
A = float(os.environ.get("LLM_SIZE", 7.0))
FEE, SEC, MIN_FEE = 0.001, 2.06e-5, 0.01
EXIT_MIN_ET = 14 * 60 + 55
ONCE = os.environ.get("LLM_ONCE") == "1"
TIMEOUT = int(os.environ.get("LLM_TIMEOUT", 150))
NCAND = int(os.environ.get("LLM_NCAND", 3))


def log(kind, **kw):
    row = dict(ts=datetime.now(KST).isoformat(timespec="seconds"), kind=kind, **kw)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False)[:300], flush=True)


def latest_scan():
    rows = [json.loads(l) for l in open(SRC_LOG, encoding="utf-8")]
    sc = [r for r in rows if r["kind"] in ("scan", "hold")]
    return sc[-1] if sc else None


def bars5(t, sym, n1m=60):
    try:
        c = t.candles(sym, "1m", count=n1m)["candles"]
    except ApiError as e:
        return None, f"api {e.code}"
    now = datetime.now(KST)
    agg = {}
    for x in sorted(c, key=lambda x: x["timestamp"]):
        ts = datetime.fromisoformat(x["timestamp"].replace("Z", "+00:00"))
        if ts > now:
            continue
        et = ts.astimezone(ET)
        m = et.hour * 60 + et.minute - 570 - 1
        if m < 0:
            continue
        b = m // 5
        o, h, l, cl, v = (float(x[k]) for k in ("openPrice", "highPrice", "lowPrice", "closePrice", "volume"))
        if b not in agg:
            agg[b] = [o, h, l, cl, v]
        else:
            a = agg[b]
            a[1], a[2], a[3], a[4] = max(a[1], h), min(a[2], l), cl, a[4] + v
    out = [(b, [round(a[0], 2), round(a[1], 2), round(a[2], 2), round(a[3], 2), int(a[4])]) for b, a in sorted(agg.items())]
    return out[-8:], None


def prev_close(t, sym):
    try:
        c = t.candles(sym, "1d", count=3)["candles"]
        today = datetime.now(ET).date()
        prev = [x for x in c if datetime.fromisoformat(x["timestamp"].replace("Z", "+00:00")).astimezone(ET).date() < today]
        return float(prev[0]["closePrice"]) if prev else None
    except ApiError:
        return None


def book(t, sym):
    try:
        ob = t.orderbook(sym)
        a, b = (ob.get("asks") or [{}])[0], (ob.get("bids") or [{}])[0]
        return dict(ask=float(a.get("price") or 0), ask_qty=float(a.get("volume") or 0), bid=float(b.get("price") or 0), bid_qty=float(b.get("volume") or 0), ob_ts=ob.get("timestamp"))
    except ApiError as e:
        return dict(error=f"api {e.code}")


def ask_llm(prompt):
    t0 = time.time()
    try:
        r = subprocess.run([OPENCODE, "run", "--pure", "--model", MODEL, prompt], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=TIMEOUT, cwd=HERE, stdin=subprocess.DEVNULL)
        raw = (r.stdout or "") + (("\nSTDERR: " + r.stderr) if r.stderr and r.returncode else "")
        status = "ok" if r.returncode == 0 else f"rc {r.returncode}"
    except subprocess.TimeoutExpired:
        raw, status = "", "timeout"
    return raw, status, round(time.time() - t0, 2)


def parse(raw):
    s = raw
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    for cand in (s[i:j + 1],):
        try:
            d = json.loads(cand)
            act = str(d.get("action", "")).lower()
            if act in ("buy", "sell", "hold"):
                return dict(action=act, symbol=(d.get("symbol") or "").upper() or None, confidence=d.get("confidence"), reason=str(d.get("reason", ""))[:300])
        except Exception:
            pass
    return None


def net_sell(sh, px):
    amt = sh * px
    return amt - np.floor(amt * FEE * 100) / 100 - max(amt * SEC, MIN_FEE)


class Paper:
    def __init__(self):
        self.pos = None
        for l in open(OUT, encoding="utf-8") if os.path.exists(OUT) else []:
            r = json.loads(l)
            if r["kind"] == "paper_buy":
                self.pos = dict(symbol=r["symbol"], entry=r["price"], sh=r["sh"], since=r["ts"], cost_in=r["cost"])
            elif r["kind"] in ("paper_sell",):
                self.pos = None

    def buy(self, sym, ask, reason, ask_row):
        if not ask or ask <= 0:
            log("paper_skip", symbol=sym, reason="no ask")
            return
        comm = np.floor(A * FEE * 100) / 100
        sh = (A - comm) / ask
        self.pos = dict(symbol=sym, entry=ask, sh=sh, since=datetime.now(KST).isoformat(timespec="seconds"), cost_in=comm)
        log("paper_buy", symbol=sym, price=ask, sh=round(sh, 6), amount=A, cost=comm, llm_reason=reason, fill_basis="post-answer ask", **{k: v for k, v in ask_row.items() if k in ("ask_qty", "bid", "bid_qty", "ob_ts", "pre_ask", "pre_ob_ts", "error")})

    def sell(self, bid, reason, why):
        p = self.pos
        if not p:
            return
        proceeds = net_sell(p["sh"], bid) if bid and bid > 0 else 0.0
        pnl = proceeds - A
        log("paper_sell", symbol=p["symbol"], entry=p["entry"], price=bid, sh=round(p["sh"], 6), proceeds=round(float(proceeds), 4), pnl=round(float(pnl), 4), pnl_pct=round(float(pnl / A * 100), 3), why=why, llm_reason=reason)
        self.pos = None


def build_prompt(now_et, cands, pos, pos_quote):
    lines = [
        "You are a US-stock intraday trading judge. Decide ONE action for the next 5 minutes.",
        f"Time: {now_et.strftime('%Y-%m-%d %H:%M')} ET (regular session 09:30-16:00; all positions must be sold by 14:55 ET).",
        f"Account: one position max, ${A:.0f} per position, round-trip cost about 0.3% (fees + spread). Only buy if you expect more than that.",
        "Position: none" if not pos else f"Position: {pos['symbol']} bought at {pos['entry']:.2f} since {pos['since'][11:16]} KST; current bid {pos_quote.get('bid')} ask {pos_quote.get('ask')}; unrealized {((pos_quote.get('bid') or pos['entry']) / pos['entry'] - 1) * 100:+.2f}%.",
        "Candidates (from a 5-minute pattern model; score is NOT a reliable probability):",
    ]
    for c in cands:
        b = c.get("bars") or []
        bars_txt = "; ".join(f"b{bb}:O{o} H{h} L{l} C{cl} V{v}" for bb, (o, h, l, cl, v) in b)
        pc = c.get("prev_close")
        chg = f"{(b[-1][1][3] / pc - 1) * 100:+.2f}% vs prev close {pc}" if pc and b else "n/a"
        q = c.get("quote", {})
        lines.append(f"- {c['symbol']} score {c['score']}: today {chg}; last 5m bars {bars_txt}; bid {q.get('bid')} x{q.get('bid_qty')} / ask {q.get('ask')} x{q.get('ask_qty')}")
    lines += [
        "Rules: if you hold a position you may 'sell' it or 'hold'; if you hold none you may 'buy' exactly one candidate symbol or 'hold' (stay in cash).",
        'Answer with JSON only, no prose: {"action":"buy|sell|hold","symbol":"TICKER or null","confidence":0.0-1.0,"reason":"<=200 chars"}',
    ]
    return "\n".join(lines)


def main():
    t = Toss()
    t.account = t.accounts()[0]["accountSeq"]
    paper = Paper()
    log("start", model=MODEL, backend=BACKEND, opencode=OPENCODE, src_log=SRC_LOG, size=A, pid=os.getpid(), once=ONCE)
    last_bar = None
    while True:
        now_et = datetime.now(ET)
        m = now_et.hour * 60 + now_et.minute
        if m >= EXIT_MIN_ET:
            if paper.pos:
                q = book(t, paper.pos["symbol"])
                paper.sell(q.get("bid"), "", "forced_exit_1455")
            log("end", reason="session_exit")
            return
        sc = latest_scan()
        bar = sc.get("bar") if sc else None
        if sc is None or bar == last_bar:
            if ONCE and sc is None:
                log("end", reason="no scan row")
                return
            time.sleep(15)
            continue
        last_bar = bar
        top = sc.get("top") or []
        cands = []
        for sym, p in top[:NCAND]:
            b, err = bars5(t, sym)
            cands.append(dict(symbol=sym, score=p, bars=b, bars_err=err, prev_close=prev_close(t, sym), quote=book(t, sym)))
        pos_quote = book(t, paper.pos["symbol"]) if paper.pos else {}
        prompt = build_prompt(now_et, cands, paper.pos, pos_quote)
        ph = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        log("ask", bar=bar, src_ts=sc["ts"], prompt_sha256=ph, prompt=prompt, candidates=[c["symbol"] for c in cands])
        if BACKEND == "jev":
            state = prompt.split("Answer with JSON only")[0]
            cs = [dict(symbol=c["symbol"], summary=f"model score {c['score']}, prev close {c.get('prev_close')}, bid {c['quote'].get('bid')} / ask {c['quote'].get('ask')}") for c in cands]
            jd = J.decide(state, cs, paper.pos is not None, model=MODEL)
            log("answer", bar=bar, prompt_sha256=ph, status=jd.get("status"), latency_s=jd.get("latency_s"), raw=jd.get("raw"), parsed={k: v for k, v in jd.items() if k != "raw"})
            status = jd.get("status") or "unknown"
            d = dict(action=jd["action"], symbol=jd.get("symbol"), confidence=jd.get("confidence"), reason=f"{jd.get('reason')} | p={jd.get('action_probabilities')} which={jd.get('which_probabilities')} risk={jd.get('risk_flag')}") if status == "ok" else None
        else:
            raw, status, lat = ask_llm(prompt)
            d = parse(raw)
            log("answer", bar=bar, prompt_sha256=ph, status=status, latency_s=lat, raw=raw[-1500:], parsed=d)
        if d is None:
            log("no_action", bar=bar, reason="unparsable or " + status)
        elif d["action"] == "buy" and not paper.pos and d["symbol"] in [c["symbol"] for c in cands]:
            q_pre = next(c["quote"] for c in cands if c["symbol"] == d["symbol"])
            q = book(t, d["symbol"])
            q["pre_ask"], q["pre_ob_ts"] = q_pre.get("ask"), q_pre.get("ob_ts")
            paper.buy(d["symbol"], q.get("ask"), d["reason"], q)
        elif d["action"] == "sell" and paper.pos:
            q = book(t, paper.pos["symbol"])
            paper.sell(q.get("bid"), d["reason"], "llm_sell")
        else:
            log("hold", bar=bar, action=d["action"], symbol=d.get("symbol"), position=paper.pos["symbol"] if paper.pos else None)
        if ONCE:
            log("end", reason="once")
            return
        nxt = (datetime.now(ET) + timedelta(minutes=5 - (datetime.now(ET).minute % 5))).replace(second=0, microsecond=0) + timedelta(seconds=170)
        time.sleep(max(5, (nxt - datetime.now(ET)).total_seconds()))


if __name__ == "__main__":
    main()
