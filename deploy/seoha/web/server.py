import base64
import hashlib
import hmac
import re
import urllib.error
import urllib.request
import json
import os
import secrets
import subprocess
import uuid
import threading
import time
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

INBOX = os.environ.get("NOSOCK_INBOX", "/srv/nosock-inbox")
STATE = os.environ.get("NOSOCK_STATE", "/var/lib/nosock/web")
PORT = int(os.environ.get("NOSOCK_WEB_PORT", "8795"))
BASE = "/nosock"
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
CRED = os.environ.get("CREDENTIALS_DIRECTORY", "")
COOKIE = "ns"
SESSION_DAYS = 365
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".json": "application/json", ".png": "image/png", ".svg": "image/svg+xml", ".webmanifest": "application/manifest+json"}
LOCK = threading.Lock()
AUTH_VERIFY = os.environ.get("NOSOCK_AUTH_VERIFY", "http://127.0.0.1:8069/verify")
WEB_PUSH = os.environ.get("NOSOCK_WEB_PUSH", "0") == "1"
ATTEMPTS = {}
APP_DIR = os.environ.get("NOSOCK_APP", "/opt/NoSock/app")
PY = os.environ.get("NOSOCK_PY", "/opt/NoSock/venv/bin/python")
BT_DIR = os.path.join(STATE, "bt")
BT_QUEUE = []
BT_LOCK = threading.Lock()
BT_FIELDS = {"data": ("historical", "bootstrap", "bear", "sideways", "volatile"), "mode": ("model", "topk", "dual", "equal")}


def bt_clean(p):
    q = {}
    for k, allowed in BT_FIELDS.items():
        v = str(p.get(k, allowed[0]))
        if v not in allowed:
            raise ValueError(k)
        q[k] = v
    k = str(p.get("k", "auto"))
    if k not in ("auto", "2", "3", "5", "8"):
        raise ValueError("k")
    q["k"] = k
    q["realistic"] = bool(p.get("realistic", True))
    dep = int(p.get("deposit", 10000))
    if not 1000 <= dep <= 10_000_000:
        raise ValueError("deposit")
    q["deposit"] = dep
    if q["data"] == "historical":
        for f, d in (("start", "2016-01-01"), ("end", "2026-11-05")):
            v = str(p.get(f, d))[:10]
            time.strptime(v, "%Y-%m-%d")
            q[f] = v
        if q["start"] >= q["end"]:
            raise ValueError("period")
    else:
        q["paths"] = max(1, min(200, int(p.get("paths", 20))))
        q["years"] = max(1, min(30, int(p.get("years", 10))))
        q["seed"] = max(0, min(10**6, int(p.get("seed", 1))))
        q["regime"] = q["data"] if q["data"] != "bootstrap" else "bootstrap"
    return q


def bt_meta(i):
    return load(os.path.join("bt", i + ".meta.json"), None)


def bt_worker():
    while True:
        job = None
        with BT_LOCK:
            if BT_QUEUE:
                job = BT_QUEUE.pop(0)
        if not job:
            time.sleep(2)
            continue
        meta = bt_meta(job) or {}
        meta.update(status="running", started=time.strftime("%Y-%m-%d %H:%M:%S"))
        save(os.path.join("bt", job + ".meta.json"), meta)
        out = os.path.join(BT_DIR, job + ".json")
        try:
            r = subprocess.run([PY, "-X", "utf8", "-m", "nosock.run_backtest", os.path.join(BT_DIR, job + ".params.json"), out], cwd=APP_DIR, capture_output=True, timeout=3600)
            ok = r.returncode == 0 and os.path.exists(out)
            meta.update(status="done" if ok else "failed", finished=time.strftime("%Y-%m-%d %H:%M:%S"), error=None if ok else r.stderr.decode("utf-8", "ignore")[-400:])
            if ok:
                res = json.load(open(out, encoding="utf-8"))
                meta["summary"] = res.get("summary") or res.get("distribution")
                meta["elapsed_s"] = res.get("elapsed_s")
        except Exception as e:
            meta.update(status="failed", error=type(e).__name__)
        save(os.path.join("bt", job + ".meta.json"), meta)


def verify_account(identity, password):
    tok = base64.b64encode(f"{identity}:{password}".encode("utf-8")).decode()
    req = urllib.request.Request(AUTH_VERIFY, headers={"Authorization": "Basic " + tok})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            d = json.loads(r.read() or b"{}")
            return d if r.status == 200 else None
    except urllib.error.HTTPError:
        return None
    except (OSError, ValueError):
        return None


def secret_text(name):
    return open(os.path.join(CRED, name), encoding="utf-8").read().strip()


def load(name, default):
    p = os.path.join(STATE, name)
    try:
        return json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        return default


def save(name, obj):
    os.makedirs(STATE, exist_ok=True)
    p = os.path.join(STATE, name)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, p)


def rows(name, kind):
    p = os.path.join(INBOX, name)
    out = []
    try:
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("kind") == kind:
                out.append(r)
    except OSError:
        pass
    return out


def vapid_public():
    from cryptography.hazmat.primitives import serialization
    pem = open(os.path.join(CRED, "NosockVapid"), "rb").read()
    key = serialization.load_pem_private_key(pem, password=None)
    raw = key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


LIVE_LOG = os.environ.get("NOSOCK_LIVE_LOG", "/var/lib/nosock/runs/live/live_nosock.jsonl")


def live_rows():
    p = LIVE_LOG if os.path.exists(LIVE_LOG) else os.path.join(INBOX, "live_nosock.jsonl")
    out = []
    try:
        for line in open(p, encoding="utf-8"):
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    except OSError:
        pass
    return out


def fx_requests(lr=None):
    done = load("fx_done.json", {})
    out = []
    for r in (lr if lr is not None else live_rows()):
        if r.get("kind") == "fx_request":
            out.append(dict(id=r["id"], book="live", day=r["ts"][:10], direction=r["direction"], amount_won=r["amount_won"], amount_usd=r["amount_usd"], fx_mid=r["fx_mid"], body=r.get("body"), live=r.get("live"), done=done.get(r["id"])))
    seen, uniq = set(), []
    for f in reversed(out):
        if f["id"] not in seen:
            seen.add(f["id"])
            uniq.append(f)
    return list(reversed(uniq))


def live_state(lr):
    plans = [r for r in lr if r.get("kind") == "plan"]
    marks = [dict(ts=r["ts"], equity=r["equity"], krw=r.get("krw"), usd=r.get("usd"), fx=r.get("fx"), phase=r.get("phase"), vault=r.get("vault"), tax_reserve=r.get("tax_reserve"), contributed=r.get("contributed")) for r in lr if r.get("kind") == "mark"]
    last_mark = next((r for r in reversed(lr) if r.get("kind") == "mark"), None)
    orders = [r for r in lr if r.get("kind") in ("order", "dry_order", "order_error", "skip", "scale_us", "refuse", "fx_missing")][-200:]
    errors = [r for r in lr if r.get("kind") == "error"][-5:]
    starts = [r for r in lr if r.get("kind") in ("start", "stop")]
    vault = next((r for r in reversed(lr) if r.get("kind") == "vault"), None)
    mode_row = next((r for r in reversed(lr) if r.get("kind") in ("start", "mark", "plan") and "live" in r), None)
    return dict(plan=plans[-1] if plans else None, plans=[dict(ts=p["ts"], cycle=p["cycle"], decided_on=p["decided_on"], weights=p["weights"]) for p in plans[-30:]], marks=marks[-2000:], last_mark=last_mark, orders=orders, errors=errors, last_start=starts[-1] if starts else None, vault=vault, live=bool(mode_row and mode_row.get("live")))


DOT_POSTS = []
DESK_LOG = os.environ.get("NOSOCK_DESK_LOG", "/var/lib/nosock/runs/live/scenario_desk.jsonl")
LIVE_STATE = os.environ.get("NOSOCK_LIVE_STATE", "/var/lib/nosock/runs/live/live_nosock_state.json")
EVENTS_DIR = os.path.join(APP_DIR, "data", "events")


DOT_ACTIONS = {"context": "상황 조회 (GET /dot/context)", "latest": "최신 시나리오 조회", "issues": "이슈 제출 (JSON)", "desk": "시나리오 재계산 요청 (사유)", "scenarios": "시나리오 직접 제출 (JSON)"}


def store_inbox(kind, js):
    size = len(json.dumps(js, ensure_ascii=False).encode("utf-8"))
    if size > (1048576 if kind == "issues" else 262144):
        return {"error": "size"}
    err = issues_check(js) if kind == "issues" else scenario_check(js)
    if err:
        return {"error": err}
    now = time.time()
    with LOCK:
        DOT_POSTS[:] = [t for t in DOT_POSTS if t > now - 86400]
        if len(DOT_POSTS) >= 96:
            return {"error": "daily limit"}
        DOT_POSTS.append(now)
    sid = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
    d = os.path.join(STATE, kind)
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, sid + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(id=sid, kind=kind, received_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), source=str(js.get("author", "dot"))[:40], payload=js), f, ensure_ascii=False)
    os.replace(tmp, os.path.join(d, sid + ".json"))
    return {"ok": True, "id": sid}


def request_desk(reason):
    d = os.path.join(STATE, "requests")
    os.makedirs(d, exist_ok=True)
    if any(f.endswith(".json") and time.time() - os.path.getmtime(os.path.join(d, f)) < 3600 for f in os.listdir(d)):
        return {"error": "a desk run was requested within the last hour"}
    rid = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
    with open(os.path.join(d, rid + ".json"), "w", encoding="utf-8") as f:
        json.dump(dict(id=rid, kind="desk", reason=reason[:500], received_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")), f, ensure_ascii=False)
    return {"ok": True, "id": rid, "note": "the scenario desk picks this up within about 5 minutes; check the latest scenario after about 10-15 minutes"}


def last_row(path, kind):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            n = f.tell()
            f.seek(max(0, n - 4_000_000))
            lines = f.read().decode("utf-8", "ignore").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("kind") == kind:
            r.pop("raw", None)
            return r
    return None


def dot_context():
    from datetime import date, timedelta
    today = time.strftime("%Y-%m-%d")
    hi = (date.today() + timedelta(days=35)).isoformat()
    lo = (date.today() - timedelta(days=7)).isoformat()
    ev = []
    try:
        cal = json.load(open(os.path.join(EVENTS_DIR, "calendar.json"), encoding="utf-8"))
        for k in ("fomc", "bok", "kr_expiry", "us_quad_witching", "kr_president", "kr_general", "kr_local", "us_president", "us_midterm"):
            ev += [dict(date=d, event=k) for d in cal.get(k, []) if lo <= d <= hi]
        cj = json.load(open(os.path.join(EVENTS_DIR, "corporate.json"), encoding="utf-8"))
        for t, v in cj.get("earnings", {}).items():
            ev += [dict(date=d, event=f"earnings {t} {v.get('name')}") for d in v.get("dates", []) if lo <= d <= hi]
        for c, v in cj.get("exdiv", {}).items():
            nx = v.get("next") or {}
            if nx.get("date") and nx.get("period_days"):
                d = date.fromisoformat(nx["date"])
                while d.isoformat() < today:
                    d += timedelta(days=nx["period_days"])
                if d.isoformat() <= hi:
                    ev.append(dict(date=d.isoformat(), event=f"ex-dividend {c} (estimated, last {nx.get('last_amount')})"))
    except (OSError, ValueError):
        pass
    weights, cycle = {}, None
    try:
        st = json.load(open(LIVE_STATE, encoding="utf-8"))
        cycle = st.get("current")
        weights = st["cycles"][cycle]["weights"] if cycle else {}
    except (OSError, ValueError, KeyError):
        pass
    if not weights:
        prev = last_row(LIVE_LOG, "plan_preview") or {}
        weights = prev.get("weights", {})
    desk = last_row(DESK_LOG, "scenario") or {}
    issues = []
    d = os.path.join(STATE, "issues")
    if os.path.isdir(d):
        for f in sorted(os.listdir(d), reverse=True)[:10]:
            try:
                rec = json.load(open(os.path.join(d, f), encoding="utf-8"))
                issues += [dict(received=rec.get("received_at"), title=x.get("title"), urgency=x.get("urgency"), category=x.get("category")) for x in rec.get("payload", {}).get("issues", [])]
            except (OSError, ValueError):
                continue
    return dict(asof=time.strftime("%Y-%m-%dT%H:%M:%S%z"), candidates=sorted(set(weights) | set((desk.get("base_weights") or {}))), model_cycle=cycle, model_weights=weights,
                upcoming_events=sorted(ev, key=lambda x: x["date"]), recent_issues=issues[:40],
                latest_scenario=dict(ts=desk.get("ts"), trigger=desk.get("trigger"), probabilities=desk.get("probabilities"), names=[x.get("name") for x in (desk.get("scenarios") or [])], tilt=desk.get("tilt"), notes=desk.get("notes")) if desk else None,
                api=dict(post_issues="/nosock/api/issues", request_desk="POST /nosock/api/dot/desk {reason}", latest_scenario="/nosock/api/dot/scenario/latest"))
CODE_RE = re.compile(r"^[A-Z0-9.]{1,12}$")


def scenario_check(js):
    if not isinstance(js, dict):
        return "body must be an object"
    sc = js.get("scenarios")
    if not isinstance(sc, list) or not 1 <= len(sc) <= 8:
        return "scenarios: 1-8 items"
    tot = 0.0
    for s in sc:
        if not isinstance(s, dict) or not isinstance(s.get("name"), str) or len(s["name"]) > 200:
            return "scenario.name"
        p = s.get("probability")
        if not isinstance(p, (int, float)) or not 0 <= p <= 1:
            return "scenario.probability"
        tot += p
        r = s.get("returns_pct")
        if not isinstance(r, dict) or len(r) > 80:
            return "scenario.returns_pct"
        for k, v in r.items():
            if not CODE_RE.match(str(k)) or not isinstance(v, (int, float)) or not -100 <= v <= 100:
                return "scenario.returns_pct entry"
        ch = s.get("chain", [])
        if not isinstance(ch, list) or len(ch) > 20 or any(not isinstance(x, str) or len(x) > 300 for x in ch):
            return "scenario.chain"
    if not 0.5 <= tot <= 1.5:
        return "probabilities must sum to about 1"
    return None


def issues_check(js):
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
            if not CODE_RE.match(str(c)) or not isinstance(v, dict) or v.get("direction") not in ("up", "down", "mixed", "unclear"):
                return "issue.affected entry"
    return None


def alerts(lr):
    out = []
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    for r in lr[-400:]:
        try:
            if (now - datetime.fromisoformat(r["ts"])).total_seconds() > 12 * 3600:
                continue
        except (KeyError, ValueError, TypeError):
            continue
        k = r.get("kind")
        if k == "order" and r.get("status") in ("FILLED", "PARTIAL_FILLED"):
            side = "매수" if r["side"] == "BUY" else "매도"
            out.append(dict(id="ord-" + r.get("client_order_id", r["ts"]), title=f"{side} 체결: {r['code']}", body=f"{r.get('filled_qty')}주, 평균 {r.get('avg_price')} ({r['ts'][11:16]})"))
        elif k in ("order_error", "refuse", "error"):
            out.append(dict(id="err-" + r["ts"] + "-" + str(r.get("code", "")), title="NoSock 주문 문제", body=f"{r.get('code', '')} {r.get('error') or r.get('reason') or ''} {r.get('code_detail') or ''}".strip()))
        elif k == "deposit_short":
            out.append(dict(id="dep-" + str(r.get("day")), title="NoSock 원화 부족", body=f"계좌에 꺼낼 원화가 {int(r.get('amount') or 0):,}원뿐이라 오늘은 그만큼만 넣었습니다. 도비는 있는 돈으로만 굴립니다."))
        elif k == "fx_missing":
            out.append(dict(id="fxm-" + r["ts"], title="NoSock 환전 안 됨", body=f"요청한 ${r.get('requested')}가 계좌에 없어 오늘 밤 미국 매수를 건너뛰었습니다."))
        elif k == "plan":
            n = len(r.get("weights", {}))
            out.append(dict(id="plan-" + r["cycle"], title="NoSock 리밸런싱 계획", body=f"{r['decided_on']} 종가 기준으로 {n}종목 비율을 정했습니다. 국내는 지금, 미국은 미국장이 열리자마자 주문합니다."))
    return out[-30:]


def state():
    mom = rows("mom_shadow_ledger.jsonl", "mom_day")
    alloc = rows("alloc_shadow_ledger.jsonl", "alloc_day")
    skips = rows("mom_shadow_ledger.jsonl", "mom_skip")[-3:]
    hist = [dict(day=r["day"], **{b: dict(total=x["total_won"], vault=x["vault_won"], deposits=x["deposits_won"], rebalance=x.get("decision")) for b, x in r["books"].items()}) for r in mom]
    last = mom[-1] if mom else None
    a = alloc[-1] if alloc else None
    lr = live_rows()
    lv = live_state(lr)
    return dict(updated=time.strftime("%Y-%m-%d %H:%M:%S"), mom_last=last, mom_history=hist, alloc_last=None if not a else dict(day=a["day"], total=a["total"], vault=a["vault"], deposits=a["deposits"], account=a["account_marked"]), fx=fx_requests(lr), alerts=alerts(lr), live=lv, skips=skips, paper=not lv["live"], ping=load("ping.json", None), app=load(os.path.join("app", "version.json"), None))


def send_push(payload):
    if not WEB_PUSH:
        return 0
    from pywebpush import WebPushException, webpush
    subs = load("subscriptions.json", {})
    vapid = os.path.join(CRED, "NosockVapid")
    sent, dead = 0, []
    for sid, sub in subs.items():
        try:
            webpush(subscription_info=sub, data=json.dumps(payload, ensure_ascii=False), vapid_private_key=vapid, vapid_claims={"sub": "mailto:nosock@teatray.top"}, ttl=86400, headers={"Urgency": "high"})
            sent += 1
        except WebPushException as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            print("push failed", code, flush=True)
            if code in (404, 410):
                dead.append(sid)
        except Exception as e:
            print("push error", type(e).__name__, flush=True)
    if dead:
        with LOCK:
            subs = load("subscriptions.json", {})
            for sid in dead:
                subs.pop(sid, None)
            save("subscriptions.json", subs)
    return sent


def watcher():
    while True:
        try:
            pushed = load("pushed.json", {})
            new = [f for f in fx_requests() if f["id"] not in pushed and not f.get("done")]
            for f in new:
                arrow = "원화 → 달러" if f["direction"] == "KRW->USD" else "달러 → 원화"
                body = f.get("body") or f"{arrow} {f['amount_won']:,}원 (약 ${f['amount_usd']:,.2f})."
                n = send_push(dict(id=f["id"], title="환전 요청", body=body, url=f"{BASE}/#fx"))
                pushed[f["id"]] = dict(at=time.strftime("%Y-%m-%d %H:%M:%S"), sent=n)
                save("pushed.json", pushed)
        except Exception:
            pass
        time.sleep(60)


class H(BaseHTTPRequestHandler):
    server_version = "nosock"

    def log_message(self, fmt, *args):
        return

    def session_entry(self):
        c = cookies.SimpleCookie(self.headers.get("Cookie", ""))
        tok = c.get(COOKIE)
        if not tok:
            return None, None
        v = load("sessions.json", {}).get(tok.value)
        if isinstance(v, (int, float)):
            v = {"exp": v, "acct": None}
        if not v or v.get("exp", 0) <= time.time():
            return tok.value, None
        return tok.value, v

    def session_ok(self):
        return self.session_entry()[1] is not None

    def send(self, code, body, ctype="application/json", extra=None):
        b = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(b)

    def dot_log(self, path, code):
        try:
            ip = self.headers.get("CF-Connecting-IP") or self.headers.get("X-Forwarded-For", "").split(",")[0].strip() or self.client_address[0]
            with open(os.path.join(STATE, "dot_access.log"), "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"), ip=ip, ua=(self.headers.get("User-Agent") or "")[:160], path=path, code=code), ensure_ascii=False) + chr(10))
        except OSError:
            pass

    def dot_ok(self, tok=None):
        if tok is None:
            auth = self.headers.get("Authorization", "")
            tok = auth[7:].strip() if auth.startswith("Bearer ") else ""
        want = load("dot_token.json", {}).get("sha256", "")
        return bool(want and tok and hmac.compare_digest(hashlib.sha256(tok.encode()).hexdigest(), want))

    def dot_get(self, p):
        ok = self.dot_ok()
        self.dot_log(p, 200 if ok else 401)
        if not ok:
            return self.send(401, {"error": "token"})
        if p == "/api/dot/context":
            return self.send(200, dot_context())
        if p == "/api/dot/scenario/latest":
            row = last_row(DESK_LOG, "scenario")
            return self.send(200, row or {"error": "no scenario yet"})
        return self.send(404, {"error": "not found"})

    def dot_desk(self):
        if not self.dot_ok():
            return self.send(401, {"error": "token"})
        out = request_desk(str(self.body_json().get("reason", "")))
        return self.send(200 if out.get("ok") else 429, out)

    def dot_page(self, result=None, action="context", body=""):
        import html as H
        opts = "".join(f'<option value="{k}"{" selected" if k == action else ""}>{v}</option>' for k, v in DOT_ACTIONS.items())
        res = f'<h2>결과</h2><pre id="result">{H.escape(result)}</pre>' if result is not None else ""
        page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex"><title>NoSock Dot 제출</title>
<style>body{{font-family:system-ui,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#111;background:#fff}}label{{display:block;margin:14px 0 6px;font-weight:600}}input,select,textarea{{width:100%;box-sizing:border-box;font:14px ui-monospace,monospace;padding:8px;border:1px solid #888;border-radius:6px}}textarea{{height:320px}}button{{margin-top:14px;padding:10px 18px;font-size:16px;border:0;border-radius:6px;background:#0A9B55;color:#fff}}pre{{white-space:pre-wrap;word-break:break-word;background:#f3f3f3;padding:12px;border-radius:6px;font-size:13px}}</style></head>
<body><h1>NoSock Dot 제출</h1><p>토큰과 동작을 고르고, 제출할 JSON이 있으면 아래 칸에 붙여 넣은 뒤 실행을 누른다. 조회 동작은 JSON 칸을 비워 둔다.</p>
<form method="post" action="/nosock/dot/form"><label for="token">토큰</label><input id="token" name="token" type="password" autocomplete="off" required>
<label for="action">동작</label><select id="action" name="action">{opts}</select>
<label for="body">JSON (이슈 제출, 시나리오 제출, 재계산 요청 사유)</label><textarea id="body" name="body" spellcheck="false">{H.escape(body)}</textarea>
<button type="submit">실행</button></form>{res}</body></html>"""
        b = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Robots-Tag", "noindex")
        self.end_headers()
        self.wfile.write(b)

    def dot_form(self):
        from urllib.parse import parse_qs as PQ
        n = int(self.headers.get("Content-Length", "0") or 0)
        if n <= 0 or n > 1100000:
            return self.dot_page(json.dumps({"error": "size"}))
        f = PQ(self.rfile.read(n).decode("utf-8", "replace"), keep_blank_values=True)
        tok, action, body = f.get("token", [""])[0].strip(), f.get("action", ["context"])[0], f.get("body", [""])[0]
        self.dot_log(f"/dot/form:{action}", 200 if self.dot_ok(tok) else 401)
        if not self.dot_ok(tok):
            return self.dot_page(json.dumps({"error": "token"}), action, body)
        if action == "context":
            out = dot_context()
        elif action == "latest":
            out = last_row(DESK_LOG, "scenario") or {"error": "no scenario yet"}
        elif action in ("issues", "scenarios"):
            try:
                js = json.loads(body)
            except ValueError:
                return self.dot_page(json.dumps({"error": "json"}), action, body)
            out = store_inbox(action, js)
        elif action == "desk":
            try:
                reason = json.loads(body).get("reason", "") if body.strip().startswith("{") else body
            except ValueError:
                reason = body
            out = request_desk(str(reason))
        else:
            out = {"error": "action"}
        return self.dot_page(json.dumps(out, ensure_ascii=False, indent=1, default=str), action, "" if out.get("ok") else body)

    def scenario_ingest(self, kind="scenarios"):
        self.dot_log(f"/api/{kind}", 200 if self.dot_ok() else 401)
        if not self.dot_ok():
            return self.send(401, {"error": "token"})
        n = int(self.headers.get("Content-Length", "0") or 0)
        if n <= 0 or n > (1048576 if kind == "issues" else 262144):
            return self.send(413, {"error": "size"})
        try:
            js = json.loads(self.rfile.read(n))
        except ValueError:
            return self.send(400, {"error": "json"})
        out = store_inbox(kind, js)
        return self.send(200 if out.get("ok") else (429 if out.get("error") == "daily limit" else 400), out)

    def body_json(self):
        n = int(self.headers.get("Content-Length", "0") or 0)
        if n <= 0 or n > 65536:
            return {}
        try:
            return json.loads(self.rfile.read(n))
        except ValueError:
            return {}

    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        if p == "/dot":
            self.dot_log("/dot", 200)
            return self.dot_page()
        if p == "/login":
            k = parse_qs(u.query).get("k", [""])[0]
            if not hmac.compare_digest(k.encode(), secret_text("NosockWebKey").encode()):
                return self.send(403, {"error": "bad key"})
            return self.send(302, b"", "text/plain", {"Set-Cookie": self.new_session("key-link"), "Location": f"{BASE}/"})
        if p.startswith("/api/dot/"):
            return self.dot_get(p)
        if p.startswith("/api/"):
            if not self.session_ok():
                return self.send(401, {"error": "login required"})
            if p == "/api/state":
                acct = (self.session_entry()[1] or {}).get("acct")
                if parse_qs(u.query).get("app", [""])[0] == "apk":
                    f = os.path.join(STATE, "app", "nosock.apk")
                    if not os.path.isfile(f):
                        return self.send(404, {"error": "no apk"})
                    return self.send(200, open(f, "rb").read(), "application/vnd.android.package-archive", {"Content-Disposition": "attachment; filename=nosock.apk"})
                ack = parse_qs(u.query).get("ack", [""])[0][:80]
                if ack:
                    with LOCK:
                        acks = load("acks.json", {})
                        acks[ack] = time.strftime("%Y-%m-%d %H:%M:%S")
                        save("acks.json", acks)
                return self.send(200, dict(state(), account=acct))
            if p == "/api/bt/list":
                os.makedirs(BT_DIR, exist_ok=True)
                metas = []
                for f in sorted(os.listdir(BT_DIR), reverse=True):
                    if f.endswith(".meta.json"):
                        m = bt_meta(f[:-10])
                        if m:
                            metas.append(m)
                return self.send(200, {"runs": metas[:50], "queue": len(BT_QUEUE)})
            if p == "/api/bt/get":
                i = parse_qs(u.query).get("id", [""])[0]
                if not i or not all(ch.isalnum() or ch == "-" for ch in i):
                    return self.send(400, {"error": "id"})
                f = os.path.join(BT_DIR, i + ".json")
                if not os.path.isfile(f):
                    return self.send(404, {"error": "not ready", "meta": bt_meta(i)})
                return self.send(200, open(f, "rb").read())
            if p == "/api/push/key":
                return self.send(200, {"key": vapid_public()})
            return self.send(404, {"error": "not found"})
        name = "index.html" if p in ("", "/") else p.lstrip("/")
        parts = name.split("/")
        if len(parts) > 2 or (len(parts) == 2 and parts[0] != "vendor") or any(not x or x.startswith(".") for x in parts):
            return self.send(404, {"error": "not found"})
        f = os.path.join(STATIC, name)
        if not os.path.isfile(f):
            return self.send(404, {"error": "not found"})
        ext = os.path.splitext(name)[1]
        extra = {"Service-Worker-Allowed": f"{BASE}/"} if name == "sw.js" else None
        return self.send(200, open(f, "rb").read(), TYPES.get(ext, "application/octet-stream"), extra)

    def new_session(self, acct=None):
        tok = secrets.token_urlsafe(32)
        with LOCK:
            sess = load("sessions.json", {})
            sess = {t: e for t, e in sess.items() if (e.get("exp", 0) if isinstance(e, dict) else e) > time.time()}
            sess[tok] = {"exp": time.time() + SESSION_DAYS * 86400, "acct": acct}
            save("sessions.json", sess)
        return f"{COOKIE}={tok}; Path={BASE}; Max-Age={SESSION_DAYS * 86400}; HttpOnly; Secure; SameSite=Strict"

    def do_POST(self):
        p = urlparse(self.path).path
        if p == "/api/scenario":
            return self.scenario_ingest()
        if p == "/api/issues":
            return self.scenario_ingest("issues")
        if p == "/api/dot/desk":
            return self.dot_desk()
        if p == "/dot/form":
            return self.dot_form()
        if p == "/login":
            if self.headers.get("Origin") not in (None, "https://teatray.top", "https://www.teatray.top"):
                return self.send(403, {"error": "origin"})
            ip = self.headers.get("CF-Connecting-IP") or self.headers.get("X-Forwarded-For", "").split(",")[0].strip() or self.client_address[0]
            now = time.time()
            hist = [t for t in ATTEMPTS.get(ip, []) if t > now - 300]
            if len(hist) >= 8:
                return self.send(429, {"error": "too many attempts, wait 5 minutes"})
            hist.append(now)
            ATTEMPTS[ip] = hist
            b = self.body_json()
            if b.get("logout"):
                tok, _ = self.session_entry()
                if tok:
                    with LOCK:
                        sess = load("sessions.json", {})
                        sess.pop(tok, None)
                        save("sessions.json", sess)
                return self.send(200, {"ok": True}, extra={"Set-Cookie": f"{COOKIE}=; Path={BASE}; Max-Age=0; HttpOnly; Secure; SameSite=Strict"})
            identity, password = str(b.get("id", ""))[:64], str(b.get("password", ""))[:256]
            info = verify_account(identity, password) if identity and password else None
            if not info:
                return self.send(401, {"error": "아이디 또는 비밀번호가 맞지 않습니다."})
            if not info.get("admin"):
                return self.send(403, {"error": "관리자 계정만 들어올 수 있습니다."})
            return self.send(200, {"ok": True, "account": info.get("account")}, extra={"Set-Cookie": self.new_session(info.get("account"))})
        if not self.session_ok():
            return self.send(401, {"error": "login required"})
        if self.headers.get("Origin") not in (None, "https://teatray.top", "https://www.teatray.top"):
            return self.send(403, {"error": "origin"})
        b = self.body_json()
        if p == "/api/push/subscribe":
            sub = b.get("subscription")
            if not (isinstance(sub, dict) and str(sub.get("endpoint", "")).startswith("https://")):
                return self.send(400, {"error": "bad subscription"})
            with LOCK:
                subs = load("subscriptions.json", {})
                subs[str(abs(hash(sub["endpoint"])))] = sub
                save("subscriptions.json", subs)
            return self.send(200, {"ok": True})
        if p == "/api/push/test":
            with LOCK:
                save("ping.json", {"id": time.strftime("%Y%m%d%H%M%S"), "at": time.strftime("%Y-%m-%d %H:%M:%S")})
            n = send_push(dict(id="test-" + time.strftime("%H%M%S"), title="NoSock 알림 테스트", body="알림이 정상적으로 도착했습니다.", url=f"{BASE}/"))
            return self.send(200, {"sent": n})
        if p == "/api/bt/run":
            try:
                q = bt_clean(b)
            except (ValueError, TypeError) as e:
                return self.send(400, {"error": "bad parameter: " + str(e)})
            os.makedirs(BT_DIR, exist_ok=True)
            i = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            save(os.path.join("bt", i + ".params.json"), q)
            save(os.path.join("bt", i + ".meta.json"), dict(id=i, params=q, status="queued", created=time.strftime("%Y-%m-%d %H:%M:%S")))
            with BT_LOCK:
                BT_QUEUE.append(i)
            return self.send(200, {"id": i})
        if p == "/api/fx/done":
            fid = str(b.get("id", ""))[:80]
            if not fid:
                return self.send(400, {"error": "id"})
            with LOCK:
                done = load("fx_done.json", {})
                done[fid] = time.strftime("%Y-%m-%d %H:%M:%S")
                save("fx_done.json", done)
            return self.send(200, {"ok": True})
        return self.send(404, {"error": "not found"})


if __name__ == "__main__":
    os.makedirs(STATE, exist_ok=True)
    threading.Thread(target=watcher, daemon=True).start()
    threading.Thread(target=bt_worker, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
