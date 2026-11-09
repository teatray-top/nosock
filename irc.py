import asyncio
import base64
import json
import os
import sys
import time
import urllib.request
from datetime import datetime
from http.cookiejar import CookieJar
from zoneinfo import ZoneInfo

import websockets

import cred

KST = ZoneInfo("Asia/Seoul")
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("IRC_BASE", "https://teatray.top")
WS = BASE.replace("https://", "wss://").replace("http://", "ws://") + "/irc/ws"
CHANNEL = os.environ.get("IRC_CHANNEL", "#bigmoney")
OUTBOX = os.path.join(HERE, "runs", "irc_outbox.txt")
INLOG = os.path.join(HERE, "runs", "irc_in.jsonl")
UA = "dobby-bot/0.1"


def log(*a):
    print(datetime.now(KST).strftime("%H:%M:%S"), *a, flush=True)


def saved_session():
    try:
        _, blob = cred.load("TeatrayIRCSession")
        s = json.loads(blob)
    except Exception:
        return None
    if s.get("expires_at", "") <= datetime.utcnow().isoformat():
        log("saved token expired; trying the device session refresh")
    jar = CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    req = urllib.request.Request(BASE + "/irc/api/auth/session", headers={"Cookie": s["cookie"], "User-Agent": UA})
    try:
        with opener.open(req, timeout=30) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        log(f"saved session check -> HTTP {e.code}")
        return None
    except urllib.error.URLError as e:
        log(f"saved session check failed: {type(e).__name__}")
        return None
    token = d.get("irc_token") or s["token"]
    exp = (d.get("expires_at") or s.get("expires_at") or "").replace("Z", "")
    if not exp or exp <= datetime.utcnow().isoformat():
        log("session endpoint gave no live token")
        return None
    parts = dict(p.strip().split("=", 1) for p in s["cookie"].split(";") if "=" in p)
    for c in jar:
        parts[c.name] = c.value
    cookie = "; ".join(f"{k}={v}" for k, v in parts.items())
    if token != s["token"] or cookie != s["cookie"] or exp != s.get("expires_at"):
        try:
            cred.store("TeatrayIRCSession", s["account"], json.dumps(dict(account=s["account"], token=token, cookie=cookie, expires_at=exp)))
            log(f"session refreshed: token changed {token != s['token']}, cookie changed {cookie != s['cookie']}, expires {exp}")
        except Exception as e:
            log(f"session persist failed: {type(e).__name__}")
    log(f"saved session ok: account {d.get('account')} expires {d.get('expires_at')}")
    return s["account"], d.get("nickname") or s["account"], token, cookie


def login():
    s = saved_session()
    if s:
        return s
    account, key = cred.load("TeatrayIRC")
    jar = CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    for path, body in (("login", {"id": account, "password": key}), ("redeem", {"account": account, "code": key})):
        req = urllib.request.Request(BASE + "/irc/api/auth/" + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "User-Agent": UA}, method="POST")
        try:
            with opener.open(req, timeout=30) as r:
                d = json.loads(r.read())
            cookie = "; ".join(f"{c.name}={c.value}" for c in jar)
            log(f"auth via {path}: account {d.get('account')} nick {d.get('nickname')} admin {d.get('admin')} expires {d.get('expires_at')}")
            try:
                cred.store("TeatrayIRCSession", d["account"], json.dumps(dict(account=d["account"], token=d["irc_token"], cookie=cookie, expires_at=(d.get("expires_at") or "9999").replace("Z", ""))))
            except Exception as e:
                log(f"session persist failed: {type(e).__name__}")
            return d["account"], d.get("nickname") or d["account"], d["irc_token"], cookie
        except urllib.error.HTTPError as e:
            log(f"auth {path} -> HTTP {e.code}")
    raise SystemExit("auth failed")


def chunks(text, limit):
    out, cur = [], ""
    for word in text.split(" "):
        cand = (cur + " " + word) if cur else word
        if len(cand.encode("utf-8")) > limit and cur:
            out.append(cur)
            cur = word
        else:
            cur = cand
        while len(cur.encode("utf-8")) > limit:
            cut = cur
            while len(cut.encode("utf-8")) > limit:
                cut = cut[:-1]
            out.append(cut)
            cur = cur[len(cut):]
    if cur:
        out.append(cur)
    return out


def parse(line):
    tags = {}
    if line.startswith("@"):
        t, line = line[1:].split(" ", 1)
        tags = dict((kv.split("=", 1) + [""])[:2] for kv in t.split(";"))
    prefix = ""
    if line.startswith(":"):
        prefix, line = line[1:].split(" ", 1)
    if " :" in line:
        head, trailing = line.split(" :", 1)
        parts = head.split() + [trailing]
    else:
        parts = line.split()
    return tags, prefix, parts[0].upper() if parts else "", parts[1:]


async def session():
    account, nick, token, cookie = login()
    outbox_pos = os.path.getsize(OUTBOX) if os.path.exists(OUTBOX) else 0
    async with websockets.connect(WS, subprotocols=["text.ircv3.net"], additional_headers={"Cookie": cookie, "User-Agent": UA, "Origin": BASE}, ping_interval=None, max_size=2**20) as ws:
        async def send(line):
            await ws.send(line)

        await send("CAP LS 302")
        registered = False
        joined = False
        last_ping = time.time()
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=2)
            except asyncio.TimeoutError:
                raw = None
            if raw is not None:
                for line in str(raw).split("\r\n"):
                    line = line.strip("\r\n")
                    if not line:
                        continue
                    tags, prefix, cmd, p = parse(line)
                    last = p[-1] if p else ""
                    if cmd == "PING":
                        await send("PONG :" + last)
                    elif cmd == "CAP":
                        sub = (p[1] if len(p) > 1 else "").upper()
                        if sub == "LS":
                            caps = {c.split("=")[0] for c in last.split()}
                            if len(p) > 2 and p[2] == "*":
                                continue
                            want = [c for c in ("server-time", "message-tags", "sasl", "echo-message") if c in caps]
                            await send("CAP REQ :" + " ".join(want))
                        elif sub == "ACK":
                            await send("AUTHENTICATE PLAIN")
                        elif sub == "NAK":
                            raise RuntimeError("CAP NAK " + last)
                    elif cmd == "AUTHENTICATE" and last == "+":
                        await send("AUTHENTICATE " + base64.b64encode(f"{account}\0{account}\0{token}".encode()).decode())
                    elif cmd in ("903", "900"):
                        if cmd == "903":
                            await send("NICK " + nick)
                            await send("USER web 0 * :dobby")
                            await send("CAP END")
                    elif cmd in ("904", "905", "906", "907"):
                        raise RuntimeError("SASL failed " + line[:120])
                    elif cmd == "001":
                        registered = True
                        log("registered as", nick)
                        await send("JOIN " + CHANNEL)
                    elif cmd == "433":
                        nick = nick + "_"
                        await send("NICK " + nick)
                    elif cmd == "JOIN" and prefix.split("!")[0].lower() == nick.lower():
                        joined = True
                        log("joined", p[0] if p else CHANNEL)
                    elif cmd in ("PRIVMSG", "NOTICE"):
                        who = prefix.split("!")[0]
                        rec = dict(ts=datetime.now(KST).isoformat(timespec="seconds"), server_time=tags.get("time"), kind=cmd, **{"from": who}, to=p[0] if p else "", text=last)
                        with open(INLOG, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        if who.lower() != nick.lower():
                            log(f"<{who}> {last[:200]}")
                    elif cmd in ("ERROR",):
                        raise RuntimeError("server ERROR " + last[:120])
            if joined and os.path.exists(OUTBOX):
                size = os.path.getsize(OUTBOX)
                if size > outbox_pos:
                    with open(OUTBOX, "r", encoding="utf-8") as f:
                        f.seek(outbox_pos)
                        chunk = f.read()
                    outbox_pos = size
                    for msg in chunk.splitlines():
                        msg = msg.strip()
                        if not msg:
                            continue
                        target = CHANNEL
                        if msg.startswith("/join "):
                            await send("JOIN " + msg[6:].strip()); log("join " + msg[6:].strip()); continue
                        if msg.startswith("/part "):
                            await send("PART " + msg[6:].strip()); log("part " + msg[6:].strip()); continue
                        if msg.startswith("/raw "):
                            await send(msg[5:].strip()); log("raw " + msg[5:60]); continue
                        if msg.startswith("/msg "):
                            target, _, msg = msg[5:].partition(" ")
                            msg = msg.strip()
                        for part in chunks(msg, 340):
                            await send(f"PRIVMSG {target} :{part}")
                            log(f"sent[{target}]: {part[:80]}")
                            await asyncio.sleep(0.3)
            if registered and time.time() - last_ping > 60:
                await send("PING :dobby-" + str(int(time.time())))
                last_ping = time.time()


async def main():
    delay = 2
    while True:
        try:
            await session()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            log(f"disconnected: {type(e).__name__}: {str(e)[:160]}; reconnect in {delay}s")
            await asyncio.sleep(delay)
            delay = min(60, delay * 2)
        else:
            delay = 2


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "say":
        os.makedirs(os.path.dirname(OUTBOX), exist_ok=True)
        with open(OUTBOX, "a", encoding="utf-8") as f:
            f.write(" ".join(sys.argv[2:]).replace("\n", " ") + "\n")
    else:
        asyncio.run(main())
