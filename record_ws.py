import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import websockets

import sim
from collect_toss import Pacer
from toss import ApiError, Toss

KST = ZoneInfo("Asia/Seoul")
HERE = os.path.dirname(os.path.abspath(__file__))
WS = "wss://openapi-ws.tossinvest.com/ws/v1"
PER_CONN = 50
MAX_CONN = 2
MK = sys.argv[1] if len(sys.argv) > 1 else "kr"
SESSION = {"kr": ("09:00", "15:30", "15:35"), "us": ("23:30", "06:00", "06:05")}[MK]
_end_hhmm = os.environ.get("WS_END", SESSION[2])
_now = datetime.now(KST)
END_AT = _now.replace(hour=int(_end_hhmm[:2]), minute=int(_end_hhmm[3:]), second=0, microsecond=0)
if END_AT <= _now:
    END_AT += timedelta(days=1)


def running():
    return datetime.now(KST) < END_AT


def in_session():
    hhmm = datetime.now(KST).strftime("%H:%M")
    a, b = SESSION[0], SESSION[1]
    return (a <= hhmm < b) if a < b else (hhmm >= a or hhmm < b)


def universe_kr(t, cap=10000.0):
    syms = [s["symbol"] for s in json.load(open(os.path.join(HERE, "data", "toss", "universe_kr.json"), encoding="utf-8"))]
    today = datetime.now(KST).date()
    pacer = Pacer(float(os.environ.get("TOSS_RPS", 10)))
    rows = []
    for s in syms:
        pacer.wait()
        try:
            c = t.candles(s, "1d", count=3)["candles"]
        except ApiError:
            continue
        prev = [x for x in c if x["timestamp"][:10] < str(today)]
        if not prev:
            continue
        pc, vol = float(prev[0]["closePrice"]), float(prev[0]["volume"])
        if pc >= sim.MIN_PRICE and pc * sim.MIN_SHARES <= cap and pc * vol >= sim.MIN_TURN:
            rows.append((pc * vol, s, pc))
    rows.sort(reverse=True)
    return [(s, pc) for _, s, pc in rows]


async def conn(tok, codes, out_dir, idx):
    last_book = {}
    ft = open(os.path.join(out_dir, f"{MK}_trade_{idx}.jsonl"), "a", encoding="utf-8")
    fb = open(os.path.join(out_dir, f"{MK}_book_{idx}.jsonl"), "a", encoding="utf-8")
    n = {"trade": 0, "book": 0, "drop": 0}
    while running():
        try:
            async with websockets.connect(WS, additional_headers={"Authorization": f"Bearer {tok()}"}, max_size=2**22, ping_interval=None) as ws:
                await ws.send(json.dumps([{"id": f"c{idx}"}, {"type": f"trade:{MK}", "codes": codes}, {"type": f"orderbook:{MK}", "codes": codes}]))
                last_ping = last_msg = last_flush = last_beat = time.time()
                while running():
                    now = time.time()
                    if now - last_ping > 60:
                        await ws.send("PING")
                        last_ping = now
                    if now - last_flush > 30:
                        ft.flush(); fb.flush()
                        last_flush = now
                    if now - last_beat > 300:
                        print(f"conn {idx}: {datetime.now(KST).strftime('%H:%M')} {n}", flush=True)
                        last_beat = now
                    if now - last_msg > 90 and in_session():
                        print(f"conn {idx}: no frames for 90s at {datetime.now(KST).strftime('%H:%M')}, resubscribing", flush=True)
                        break
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=5)
                    except asyncio.TimeoutError:
                        continue
                    m = json.loads(raw)
                    k = m.get("type")
                    if k == "message":
                        last_msg = time.time()
                        topic = m["topic"]
                        rec = time.time()
                        if topic.startswith("trade:"):
                            ft.write(json.dumps(dict(r=round(rec, 3), s=topic.split(":")[-1], **m["data"]), ensure_ascii=False) + "\n")
                            n["trade"] += 1
                        elif topic.startswith("orderbook:"):
                            s = topic.split(":")[-1]
                            if rec - last_book.get(s, 0) < 1.0:
                                n["drop"] += 1
                                continue
                            last_book[s] = rec
                            d = m["data"]
                            fb.write(json.dumps(dict(r=round(rec, 3), s=s, t=d.get("timestamp"), a=[[x["price"], x["volume"]] for x in d["asks"][:10]], b=[[x["price"], x["volume"]] for x in d["bids"][:10]]), ensure_ascii=False) + "\n")
                            n["book"] += 1
                        if (n["trade"] + n["book"]) % 2000 == 0:
                            ft.flush(); fb.flush()
                    elif k == "subscriptions":
                        print(f"conn {idx}: subscribed {len(m.get('subscribed', []))} rejected {m.get('rejected')}", flush=True)
                    elif k == "error":
                        print(f"conn {idx}: error {m.get('error')}", flush=True)
                        if m.get("error", {}).get("code") in ("auth-revoked", "server-shutdown"):
                            break
        except Exception as e:
            print(f"conn {idx}: reconnect after {type(e).__name__}: {str(e)[:120]}", flush=True)
            await asyncio.sleep(5)
    ft.close(); fb.close()
    print(f"conn {idx}: done {n}", flush=True)


async def main():
    t = Toss()
    out_dir = os.path.join(HERE, "data", "toss", "ws", datetime.now(KST).strftime("%Y%m%d") + ("" if MK == "kr" else "_" + MK))
    os.makedirs(out_dir, exist_ok=True)
    up = os.path.join(out_dir, "universe.json")
    if os.path.exists(up):
        u = json.load(open(up))
    elif MK == "kr":
        u = universe_kr(t)
    else:
        u = json.load(open(os.environ.get("WS_UNIVERSE", os.path.join(HERE, "data", "toss", "ws", f"universe_{MK}.json")), encoding="utf-8"))
    json.dump(u, open(up, "w"))
    extra = [x for x in os.environ.get("WS_EXTRA", "").split(",") if x]
    codes = extra + [s for s, _ in u if s not in extra][:PER_CONN * MAX_CONN - len(extra)]
    print(f"{datetime.now(KST).isoformat(timespec='seconds')} {MK} universe {len(u)}, recording {len(codes)} until {END_AT:%m-%d %H:%M}", flush=True)

    def tok():
        t._load_token()
        if not t.tok or t.exp < time.time() + 60:
            t._issue()
        return t.tok

    await asyncio.gather(*[conn(tok, codes[i:i + PER_CONN], out_dir, i // PER_CONN) for i in range(0, len(codes), PER_CONN)])


if __name__ == "__main__":
    asyncio.run(main())
