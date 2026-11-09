import json
import os
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import toss

tmp = tempfile.mkdtemp(prefix="toss_lock_test_")
toss.TOKEN_FILE = os.path.join(tmp, "toss_token.json")
toss.cred.load = lambda target: ("cid", "secret")
calls = []
gate = threading.Lock()


class Resp:
    def __init__(self, body):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fake_urlopen(req, timeout=20):
    with gate:
        calls.append(time.time())
        n = len(calls)
    time.sleep(0.3)
    return Resp(json.dumps(dict(access_token=f"tok{n}", expires_in=3600)).encode())


toss.urllib.request.urlopen = fake_urlopen
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  {detail}"))


clients = [toss.Toss() for _ in range(8)]
ts = [threading.Thread(target=c._issue) for c in clients]
[t.start() for t in ts]
[t.join() for t in ts]
check("8 concurrent clients -> 1 token issued", len(calls) == 1, len(calls))
check("all clients share the issued token", len({c.tok for c in clients}) == 1 and clients[0].tok == "tok1", [c.tok for c in clients])
check("cache written atomically, no tmp left", os.path.exists(toss.TOKEN_FILE) and not os.path.exists(toss.TOKEN_FILE + ".tmp"))
a, b = clients[0], clients[1]
a._issue()
check("revoked token (cache still holds it) -> reissue", len(calls) == 2 and a.tok == "tok2", (len(calls), a.tok))
b._issue()
check("second client with old token adopts newer cache token without issuing", len(calls) == 2 and b.tok == "tok2", (len(calls), b.tok))
fresh = toss.Toss()
check("new client loads cached token, no issue", fresh.tok == "tok2" and len(calls) == 2)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
