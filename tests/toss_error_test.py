import io
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
print(f"{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
