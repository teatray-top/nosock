import base64
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

import chart_png as CP
import jev_client as J
from toss import ApiError, Toss

HERE = os.path.dirname(os.path.abspath(__file__))
ET, KST = ZoneInfo("America/New_York"), ZoneInfo("Asia/Seoul")
LIVE = os.environ.get("JEV_LIVE") == "1"
ONCE = os.environ.get("JEV_ONCE") == "1"
MODEL = os.environ.get("JEV_MODEL", "jev-1.13-free")
SRC_LOG = os.environ.get("JEV_SRC_LOG", os.path.join(HERE, "runs", "live_us_20261105.jsonl"))
DAY = os.environ.get("JEV_DAY", "20261105")
OUT = os.path.join(HERE, "runs", f"jev_scalp_{DAY}.jsonl")
CHART_DIR = os.path.join(HERE, "runs", "jev_charts")
A = float(os.environ.get("JEV_SIZE", 7.0))
TP, SL, MAX_HOLD_MIN = float(os.environ.get("JEV_TP", 0.006)), float(os.environ.get("JEV_SL", 0.004)), int(os.environ.get("JEV_MAX_HOLD", 15))
EXIT_MIN_ET, LAST_ENTRY_MIN_ET = 14 * 60 + 55, 14 * 60 + 35
MAX_TRADES, MAX_LOSS = int(os.environ.get("JEV_MAX_TRADES", 12)), float(os.environ.get("JEV_MAX_LOSS", 1.0))
NCAND = int(os.environ.get("JEV_NCAND", 3))
FEE, SEC, MIN_FEE = 0.001, 2.06e-5, 0.01
INPUT_DIR = os.path.join(HERE, "runs", "jev_inputs", DAY)
TIMEOUT = 90
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)


def file_sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


CODE_SHA = {n: file_sha(os.path.join(HERE, n)) for n in ("jev_scalp.py", "jev_client.py", "chart_png.py")}


def kst_ms():
    return datetime.now(KST).isoformat(timespec="milliseconds")


def write_new(path, data):
    raw = bytes(data) if isinstance(data, (bytes, bytearray)) else json.dumps(data, ensure_ascii=False, indent=1).encode("utf-8")
    with open(path, "xb") as f:
        f.write(raw)
    return hashlib.sha256(raw).hexdigest()


class Bundle:
    def __init__(self):
        self.id = f"{DAY}_{datetime.now(KST).strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.dir = os.path.join(INPUT_DIR, self.id)
        os.makedirs(self.dir, exist_ok=False)
        self.files = {}
        self.failed = None

    def put(self, name, data):
        self.files[name] = write_new(os.path.join(self.dir, name), data)
        return self.files[name]

    def rel(self, name=""):
        return os.path.relpath(os.path.join(self.dir, name), HERE)

    def drop_if_empty(self):
        if not self.files and not os.listdir(self.dir):
            os.rmdir(self.dir)


def log(kind, **kw):
    row = dict(ts=datetime.now(KST).isoformat(timespec="seconds"), kind=kind, **kw)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False)[:240], flush=True)


def latest_top():
    try:
        rows = [json.loads(l) for l in open(SRC_LOG, encoding="utf-8")]
    except OSError:
        return []
    sc = [r for r in rows if r["kind"] in ("scan", "hold") and r.get("top")]
    return [s for s, p in sc[-1]["top"]] if sc else []


def bars1m(t, sym, n=60):
    req_ts = kst_ms()
    raw = t.candles(sym, "1m", count=n)
    recv_ts = kst_ms()
    now = datetime.now(KST)
    out, dropped = [], []
    for x in sorted(raw["candles"], key=lambda x: x["timestamp"]):
        ts = datetime.fromisoformat(x["timestamp"].replace("Z", "+00:00"))
        et = ts.astimezone(ET)
        if ts > now:
            dropped.append((x["timestamp"], "in-progress bar (end-stamped after now)"))
            continue
        if et.hour * 60 + et.minute < 570 + 1 or et.date() != datetime.now(ET).date():
            dropped.append((x["timestamp"], "outside today's regular session"))
            continue
        out.append((et.strftime("%H:%M"), [float(x["openPrice"]), float(x["highPrice"]), float(x["lowPrice"]), float(x["closePrice"]), float(x["volume"])]))
    return dict(bars=out, raw=raw, req_ts=req_ts, recv_ts=recv_ts, reference_now=now.isoformat(timespec="milliseconds"), dropped=dropped, interval="1m", count=n)


def prev_close(t, sym):
    req_ts = kst_ms()
    try:
        raw = t.candles(sym, "1d", count=3)
    except ApiError as e:
        return dict(value=None, raw=None, req_ts=req_ts, recv_ts=kst_ms(), error=f"api {e.code}")
    today = datetime.now(ET).date()
    prev = [x for x in raw["candles"] if datetime.fromisoformat(x["timestamp"].replace("Z", "+00:00")).astimezone(ET).date() < today]
    return dict(value=float(prev[0]["closePrice"]) if prev else None, raw=raw, req_ts=req_ts, recv_ts=kst_ms(), source_ts=prev[0]["timestamp"] if prev else None)


def book(t, sym):
    try:
        ob = t.orderbook(sym)
        a, b = (ob.get("asks") or [{}])[0], (ob.get("bids") or [{}])[0]
        ask, bid = float(a.get("price") or 0), float(b.get("price") or 0)
        return dict(ask=ask, ask_qty=float(a.get("volume") or 0), bid=bid, bid_qty=float(b.get("volume") or 0), spread_pct=round((ask - bid) / ask * 100, 3) if ask and bid else None, ob_ts=ob.get("timestamp"))
    except ApiError as e:
        return dict(error=f"api {e.code}")


def valid_px(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and x not in (float('inf'), float('-inf')) and x > 0


def net_sell(sh, px):
    amt = sh * px
    return amt - np.floor(amt * FEE * 100) / 100 - max(amt * SEC, MIN_FEE)


class Book:
    def __init__(self, t):
        self.t = t
        self.pos = None
        self.trades = 0
        self.realized = 0.0
        if os.path.exists(OUT):
            for l in open(OUT, encoding="utf-8"):
                r = json.loads(l)
                if r["kind"] == "buy" and r.get("live") == LIVE:
                    self.pos = dict(symbol=r["symbol"], entry=r["price"], sh=r["sh"], since=r["ts"], cost_in=r.get("cost", 0.0))
                elif r["kind"] == "sell" and r.get("live") == LIVE:
                    self.pos = None
                    self.trades += 1
                    self.realized += r.get("pnl", 0.0)

    def buy(self, sym, q, reason):
        ask = q.get("ask")
        if not valid_px(ask):
            log("skip_buy", symbol=sym, reason="no valid ask", ask=ask, live=LIVE)
            return
        if LIVE:
            usd = float(self.t.buying_power("USD")["cashBuyingPower"])
            amt = max(0.0, np.floor(usd * 100 - 1) / 100)
            if amt < 2.0:
                log("skip_buy", symbol=sym, reason="usd too small", usd=usd, live=LIVE)
                return
            r = self.t.place(sym, "BUY", "MARKET", amount=f"{amt:.2f}", client_order_id="jev-" + uuid.uuid4().hex[:12])
            o = wait(self.t, r["orderId"])
            ex = o["execution"]
            qty, px = float(ex["filledQuantity"] or 0), float(ex["averageFilledPrice"] or 0)
            log("buy", symbol=sym, live=True, status=o["status"], sh=qty, price=px, amount=amt, cost=float(ex["commission"] or 0), ask_at_decision=ask, ob_ts=q.get("ob_ts"), llm_reason=reason)
            if qty > 0:
                self.pos = dict(symbol=sym, entry=px, sh=qty, since=datetime.now(KST).isoformat(timespec="seconds"), cost_in=float(ex["commission"] or 0))
        else:
            comm = np.floor(A * FEE * 100) / 100
            sh = (A - comm) / ask
            self.pos = dict(symbol=sym, entry=ask, sh=sh, since=datetime.now(KST).isoformat(timespec="seconds"), cost_in=comm)
            log("buy", symbol=sym, live=False, price=ask, sh=round(sh, 6), amount=A, cost=comm, fill_basis="post-answer ask", **{k: q.get(k) for k in ("ask_qty", "bid", "bid_qty", "spread_pct", "ob_ts")}, llm_reason=reason)

    def sell(self, q, why, reason=""):
        p = self.pos
        if not p:
            return
        if LIVE:
            sq = self.t.get("/api/v1/sellable-quantity", account=True, symbol=p["symbol"])["sellableQuantity"]
            if float(sq) <= 0:
                log("sell_skip", symbol=p["symbol"], reason="no sellable qty", live=True)
                self.pos = None
                return
            r = self.t.place(p["symbol"], "SELL", "MARKET", quantity=sq, client_order_id="jev-" + uuid.uuid4().hex[:12])
            o = wait(self.t, r["orderId"])
            ex = o["execution"]
            proceeds = float(ex["filledAmount"] or 0) - float(ex["commission"] or 0) - float(ex["tax"] or 0)
            pnl = proceeds - p["entry"] * p["sh"] - p.get("cost_in", 0.0)
            log("sell", symbol=p["symbol"], live=True, status=o["status"], why=why, entry=p["entry"], price=float(ex["averageFilledPrice"] or 0), sh=float(ex["filledQuantity"] or 0), proceeds=round(proceeds, 4), pnl=round(pnl, 4), pnl_pct=round(pnl / max(1e-9, p["entry"] * p["sh"]) * 100, 3), bid_at_decision=q.get("bid"), llm_reason=reason)
        else:
            bid = q.get("bid")
            if not valid_px(bid):
                log("sell_skip", symbol=p["symbol"], reason="no valid bid; position kept", why=why, bid=bid, live=False)
                return
            proceeds = net_sell(p["sh"], bid)
            pnl = proceeds - A
            log("sell", symbol=p["symbol"], live=False, why=why, entry=p["entry"], price=bid, sh=round(p["sh"], 6), proceeds=round(float(proceeds), 4), pnl=round(float(pnl), 4), pnl_pct=round(float(pnl / A * 100), 3), fill_basis="post-decision bid", ob_ts=q.get("ob_ts"), llm_reason=reason)
        self.trades += 1
        self.realized += float(pnl)
        self.pos = None


def wait(t, oid, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        o = t.order(oid)
        if o["status"] in ("FILLED", "CANCELED", "REJECTED"):
            return o
        time.sleep(2)
    return t.order(oid)


def facts(now_et, cands, pos, pos_q, held_min):
    lines = [f"Time {now_et.strftime('%H:%M')} ET. Regular session; all positions must be closed by 14:55 ET ({max(0, EXIT_MIN_ET - (now_et.hour * 60 + now_et.minute))} min left). Scalping: target +{TP * 100:.1f}%, stop -{SL * 100:.1f}%, max hold {MAX_HOLD_MIN} min; round-trip cost about 0.3%. One position max, ${A:.0f} each."]
    if pos:
        up = (pos_q.get("bid") or pos["entry"]) / pos["entry"] - 1
        lines.append(f"Position: {pos['symbol']} entry {pos['entry']:.2f}, held {held_min} min, bid now {pos_q.get('bid')} ask {pos_q.get('ask')}, unrealized {up * 100:+.2f}%. Chart `chart_{pos['symbol']}`.")
    else:
        lines.append("Position: none (cash).")
    for c in cands:
        q, pc, last = c["quote"], c.get("prev_close"), c.get("last")
        chg = f"{(last / pc - 1) * 100:+.2f}% vs prev close {pc:.2f}" if (pc and last) else "n/a"
        tag = f" (5m-model score {c['score']})" if c.get("score") is not None else " (pre-selected by a 5-minute pattern model; score not provided)"
        lines.append(f"Candidate {c['symbol']}{tag}: last {last}, {chg}, bid {q.get('bid')} x{q.get('bid_qty')} / ask {q.get('ask')} x{q.get('ask_qty')}, spread {q.get('spread_pct')}%. Chart `chart_{c['symbol']}` = last 30 one-minute candles (red up, blue down) of today's session, orange MA20 of 1-minute closes, purple VWAP cumulated over the input bars shown in the chart header, black prev close, grey volume.")
    return "\n".join(lines)


def main():
    t = Toss()
    t.account = t.accounts()[0]["accountSeq"]
    bk = Book(t)
    log("start", live=LIVE, model=MODEL, timeout=TIMEOUT, size=A, tp=TP, sl=SL, max_hold_min=MAX_HOLD_MIN, max_trades=MAX_TRADES, max_loss=MAX_LOSS, ncand=NCAND, src_log=SRC_LOG, input_dir=os.path.relpath(INPUT_DIR, HERE), code_sha256=CODE_SHA, pid=os.getpid(), once=ONCE, resumed_pos=bk.pos["symbol"] if bk.pos else None, trades_so_far=bk.trades, realized_so_far=round(bk.realized, 4))
    while True:
        now_et = datetime.now(ET)
        m = now_et.hour * 60 + now_et.minute
        if m >= EXIT_MIN_ET or os.path.exists(os.path.join(HERE, "runs", "JEV_STOP")):
            if bk.pos:
                bk.sell(book(t, bk.pos["symbol"]), "forced_exit" if m >= EXIT_MIN_ET else "stop_file")
            log("end", reason="session_exit" if m >= EXIT_MIN_ET else "stop_file", trades=bk.trades, realized=round(bk.realized, 4), live=LIVE)
            return
        if m < 570 + 2:
            time.sleep(20)
            continue
        if bk.pos:
            q = book(t, bk.pos["symbol"])
            held_min = int((datetime.now(KST) - datetime.fromisoformat(bk.pos["since"])).total_seconds() // 60)
            bid = q.get("bid")
            if valid_px(bid):
                r = bid / bk.pos["entry"] - 1
                why = "tp" if r >= TP else ("sl" if r <= -SL else ("time" if held_min >= MAX_HOLD_MIN else None))
                if why:
                    bk.sell(q, why)
        syms = [s for s in latest_top() if s != (bk.pos or {}).get("symbol")][:NCAND]
        cands, assets, charts, pos_q, inputs = [], {}, {}, {}, {}
        try:
            bundle = Bundle()
        except OSError as e:
            bundle = None
            log("bundle_error", error=f"{type(e).__name__}: {str(e)[:120]}")
        for s in (([bk.pos["symbol"]] if bk.pos else []) + syms) if bundle else []:
            try:
                bb = bars1m(t, s)
            except ApiError as e:
                log("cand_error", symbol=s, error=f"api {e.code}")
                continue
            b = bb["bars"]
            if len(b) < 5:
                inputs[s] = dict(candles=bb, skipped="fewer than 5 session bars")
                continue
            pcd = prev_close(t, s)
            pc = pcd["value"]
            img, info = CP.render(b, prev_close=pc, show_last=30, ma_n=20, title=s, interval="1m", tz_label="ET")
            b64, nbytes = CP.to_base64_png(img)
            png = base64.b64decode(b64)
            ref = f"chart_{s}"
            assets[ref] = {"type": "image", "format": "png", "data_base64": b64}
            try:
                png_sha = bundle.put(f"chart_{s}.png", png)
            except OSError as e:
                log("bundle_error", symbol=s, request_id=bundle.id, error=f"chart save failed: {type(e).__name__}: {str(e)[:120]}")
                bundle.failed = f"chart_{s}.png save failed"
                break
            charts[s] = dict(file=bundle.rel(f"chart_{s}.png"), png_bytes=len(png), png_sha256=png_sha, **info)
            q = book(t, s)
            inputs[s] = dict(candles=bb, bars_used=b, prev_close=pcd, quote=q, quote_recv_ts=kst_ms(), chart=charts[s])
            sc = None
            if not bk.pos or s != bk.pos["symbol"]:
                cands.append(dict(symbol=s, score=sc, quote=q, prev_close=pc, last=b[-1][1][3], summary=f"last {b[-1][1][3]}, bid {q.get('bid')} / ask {q.get('ask')}, spread {q.get('spread_pct')}%"))
            else:
                pos_q = q
        if bundle is None or (not cands and not bk.pos):
            if bundle:
                bundle.drop_if_empty()
            log("idle", reason="no candidates" if bundle else "bundle dir unavailable", request_id=bundle.id if bundle else None)
        elif bundle.failed:
            log("ask_skipped", request_id=bundle.id, transport_attempted=False, reason="input save failed, whole round blocked: " + bundle.failed)
        else:
            if bk.pos and not pos_q:
                pos_q = book(t, bk.pos["symbol"])
            held_min = int((datetime.now(KST) - datetime.fromisoformat(bk.pos["since"])).total_seconds() // 60) if bk.pos else 0
            state = {"facts": facts(now_et, cands, bk.pos, pos_q if bk.pos else {}, held_min)}
            for s in list(charts):
                state[f"chart_{s}"] = f"chart_{s}"
            can_buy = (not bk.pos) and m < LAST_ENTRY_MIN_ET and bk.trades < MAX_TRADES and bk.realized > -MAX_LOSS
            qs = J.trade_questions(cands if can_buy else [], bk.pos is not None)
            if not can_buy:
                qs["action"]["criteria"].pop("buy", None)
                qs.pop("which", None)
            ph = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:16]
            meta = dict(request_id=bundle.id, day=DAY, ts_kst=kst_ms(), now_et=now_et.isoformat(timespec="milliseconds"), model=MODEL, timeout=TIMEOUT, endpoint=J.URL, code_sha256=CODE_SHA, params=dict(size=A, tp=TP, sl=SL, max_hold_min=MAX_HOLD_MIN, max_trades=MAX_TRADES, max_loss=MAX_LOSS, ncand=NCAND, last_entry_min_et=LAST_ENTRY_MIN_ET, exit_min_et=EXIT_MIN_ET), can_buy=can_buy, candidates=[c["symbol"] for c in cands], position=bk.pos, position_quote=pos_q if bk.pos else None, held_min=held_min, trades_so_far=bk.trades, realized_so_far=bk.realized, live=LIVE)
            sent = dict(prepared=False, attempted=False, received=None)

            def save(body, hdrs):
                sent["request_sha256"] = bundle.put("request.json", body)
                sent["request_bytes"] = len(body)
                bundle.put("headers.json", hdrs)
                bundle.put("manifest.json", dict(meta, files=dict(bundle.files)))
                sent["prepared"] = True

            def before_send():
                sent["attempted"] = True
                sent["attempt_ts"] = kst_ms()

            def save_response(raw, st, lt):
                sent["received"] = dict(status=st, latency_s=lt, received_ts=kst_ms(), bytes=len(raw))
                sent["response_sha256"] = bundle.put("response.raw", raw)

            status, out, lat = None, None, None
            try:
                bundle.put("inputs.json", inputs)
                bundle.put("state.json", state)
                bundle.put("questions.json", qs)
                bundle.put("meta.json", meta)
                log("ask", request_id=bundle.id, bundle=bundle.rel(), state=state["facts"], charts=charts, prompt_sha256=ph, can_buy=can_buy, candidates=[c["symbol"] for c in cands], position=bk.pos["symbol"] if bk.pos else None)
                out, status, lat = J.ask(state, qs, model=MODEL, assets=assets, timeout=TIMEOUT, save=save, save_response=save_response, before_send=before_send)
            except OSError as e:
                if not sent["prepared"]:
                    why = "bundle save failed before send; transport not attempted"
                elif not sent["attempted"]:
                    why = "request saved but transport not attempted (error while preparing the request or credentials)"
                elif sent["received"] is None:
                    why = "transport attempted; no response received or error before receipt"
                else:
                    why = f"response received (http {sent['received']['status']}) but response save failed"
                log("ask_skipped", request_id=bundle.id, prepared=sent["prepared"], transport_attempted=sent["attempted"], attempt_ts=sent.get("attempt_ts"), received=sent["received"], reason=f"{why}: {type(e).__name__}: {str(e)[:120]}")
            if status is not None:
                try:
                    bundle.put("response.json", dict(status=status, latency_s=lat, received=sent["received"], request_sha256=sent.get("request_sha256"), response_raw_sha256=sent.get("response_sha256"), parsed=out))
                except OSError as e:
                    log("bundle_error", request_id=bundle.id, error=f"response.json save failed: {type(e).__name__}: {str(e)[:120]}")
            if status is None:
                pass
            elif status != 200:
                log("answer", request_id=bundle.id, request_sha256=sent.get("request_sha256"), prompt_sha256=ph, status=f"api_error {status}", latency_s=lat, raw=out)
            else:
                ans = J.answers(out)
                act, act_p, act_c = J.read_choice(ans, "action")
                which, which_p, _ = J.read_choice(ans, "which") if "which" in qs else (None, None, None)
                conf, _, _ = J.read_score(ans, "confidence", len(J.CONF_LEVELS))
                risk = J.read_noul(ans, "risk_flag")
                usage = out.get("usage") if isinstance(out.get("usage"), dict) else {}
                log("answer", request_id=bundle.id, request_sha256=sent.get("request_sha256"), prompt_sha256=ph, status="ok" if "error" not in out else "invalid_response_body", latency_s=lat, action=act, action_p=act_p, which=which, which_p=which_p, confidence=conf, risk_flag=risk, input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"), cost=usage.get("cost"), raw=out)
                invalid = [n for n, v in (("action", act), ("confidence", conf), ("risk_flag", risk)) if v is None] + (["which"] if can_buy and which is None else [])
                if invalid:
                    log("no_action", reason="missing/abnormal answers: " + ",".join(invalid), action=act, which=which, position=bk.pos["symbol"] if bk.pos else None)
                elif act == "buy" and can_buy and which in [c["symbol"] for c in cands] and risk < 0.5:
                    q2 = book(t, which)
                    q2["ask_at_ask_time"] = next(c["quote"].get("ask") for c in cands if c["symbol"] == which)
                    bk.buy(which, q2, f"jev buy p={act_p} which_p={which_p} conf={conf} risk={risk}")
                elif act == "sell" and bk.pos:
                    bk.sell(book(t, bk.pos["symbol"]), "jev_sell", f"jev sell p={act_p} conf={conf}")
                else:
                    log("hold", action=act, which=which, position=bk.pos["symbol"] if bk.pos else None, can_buy=can_buy, risk_flag=risk)
        if ONCE:
            log("end", reason="once", live=LIVE)
            return
        nxt = (datetime.now(ET) + timedelta(minutes=1)).replace(second=20, microsecond=0)
        time.sleep(max(3, (nxt - datetime.now(ET)).total_seconds()))


if __name__ == "__main__":
    main()
