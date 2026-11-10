import gzip
import io
import json
import os
import sys
import time
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import toss

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  {detail}"))


t = toss.Toss.__new__(toss.Toss)
t.tok, t.exp, t.account = "x", time.time() + 3600, 1


def err_for(body, status=422):
    def boom(req, timeout=20):
        raise urllib.error.HTTPError(req.full_url, status, "x", {}, io.BytesIO(body))
    toss.urllib.request.urlopen = boom
    try:
        t._req("POST", "/api/v1/orders", body={}, account=True)
    except toss.ApiError as e:
        return e
    return None


e = err_for(b'{"error": {"code": "prerequisite-required", "message": "m1", "requestId": "r1"}}')
check("documented error body -> code, message, request id kept", e.status == 422 and e.code == "prerequisite-required" and e.message == "m1" and e.request_id == "r1", vars(e))
e = err_for(b'{"code": "fractional-not-allowed", "message": "top level"}')
check("code/message at the top level -> still read", e.code == "fractional-not-allowed" and e.message == "top level", vars(e))
e = err_for(b'{"status": 422, "error": "Unprocessable Entity", "message": "UUP cannot be bought by amount"}')
check("no error object (the 11-10 UUP case) -> message kept instead of empty", e.code == "http" and e.message == "UUP cannot be bought by amount", vars(e))
e = err_for(b'<html>bad gateway</html>', 502)
check("non-JSON body -> raw text kept", e.status == 502 and e.code == "http" and "bad gateway" in e.message, vars(e))
class Ok:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self.body


def gz(obj):
    return gzip.compress(json.dumps(obj, ensure_ascii=False).encode("utf-8"))


def clean(text):
    return all(ch >= " " for ch in text)


doc = {"error": {"requestId": "r9", "code": "fractional-not-allowed", "message": "소수점 주문을 할 수 없는 종목입니다"}}
e = err_for(gz(doc))
check("gzip error body (the 11-10 UUP reply) -> code, message, request id read", e.code == "fractional-not-allowed" and e.message == doc["error"]["message"] and e.request_id == "r9" and clean(e.message), vars(e))
lossy = gz(doc).decode("utf-8", "ignore")
check("the unreadable text logged on 11-10 is what an undecoded gzip body looks like", lossy[:2] == chr(0x1F) + chr(8) and not clean(lossy), repr(lossy[:20]))
e = err_for(gz({"error": {"code": "request-in-progress", "message": "m"}}), 409)
check("gzip 409 -> request-in-progress still recognised", e.status == 409 and e.code == "request-in-progress", vars(e))
e = err_for(gz({"error": {"code": "insufficient-buying-power", "message": "m"}}))
check("gzip 422 -> account-wide code still recognised", e.code == "insufficient-buying-power", vars(e))
e = err_for(bytes((0x1F, 0x8B)) + b"not really gzip", 502)
check("broken gzip body -> no crash, status kept", e is not None and e.status == 502 and e.code == "http", vars(e) if e else None)
e = err_for(gzip.compress(b"<html>bad gateway</html>"), 502)
check("gzip non-JSON body -> text kept", e.code == "http" and "bad gateway" in e.message, vars(e))

toss.urllib.request.urlopen = lambda req, timeout=20: Ok(gz({"result": [{"accountSeq": 7}]}))
check("gzip success body -> parsed", t.accounts() == [{"accountSeq": 7}])
toss.urllib.request.urlopen = lambda req, timeout=20: Ok(b'{"result": [{"accountSeq": 8}]}')
check("plain success body -> parsed as before", t.accounts() == [{"accountSeq": 8}])

calls, issued = [], []


def expired_once(req, timeout=20):
    calls.append(1)
    if len(calls) == 1:
        raise urllib.error.HTTPError(req.full_url, 401, "x", {}, io.BytesIO(gz({"error": {"code": "expired-token", "message": "m"}})))
    return Ok(b'{"result": [{"accountSeq": 9}]}')


toss.urllib.request.urlopen = expired_once
t._load_token = lambda: None
t._issue = lambda: issued.append(1)
check("gzip 401 expired-token -> token renewed and the call repeated", t.accounts() == [{"accountSeq": 9}] and len(calls) == 2 and len(issued) == 1, (calls, issued))
print(f"{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
