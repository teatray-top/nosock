import base64
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.error
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

PENNY = r"G:\NoSock\stock\penny"
sys.path.insert(0, PENNY)
os.chdir(PENNY)
DAY = "offlinetest"
os.environ.update(JEV_DAY=DAY, JEV_ONCE="1", OPENCODE_API_KEY="dummy-never-sent")
tmp = tempfile.mkdtemp()
src = os.path.join(tmp, "scan.jsonl")
open(src, "w").write(json.dumps(dict(kind="scan", top=[["AAA", 0.9], ["BBB", 0.8], ["CCC", 0.7], ["DDD", 0.6]])) + "\n")
os.environ["JEV_SRC_LOG"] = src
out_log = os.path.join(PENNY, "runs", f"jev_scalp_{DAY}.jsonl")
in_dir = os.path.join(PENNY, "runs", "jev_inputs", DAY)
if os.path.exists(out_log):
    os.remove(out_log)
if os.path.isdir(in_dir):
    shutil.rmtree(in_dir)

import jev_scalp as S

ET, KST = ZoneInfo("America/New_York"), ZoneInfo("Asia/Seoul")
FIXED = datetime(2026, 11, 5, 10, 30, 20, tzinfo=ET)


class FakeDT(datetime):
    @classmethod
    def now(cls, tz=None):
        return FIXED.astimezone(tz) if tz else FIXED.replace(tzinfo=None)


S.datetime = FakeDT


class FakeToss:
    calls = []

    def accounts(self):
        return [{"accountSeq": "fake"}]

    def candles(self, sym, interval, count=60):
        self.calls.append(("candles", sym, interval, count))
        if interval == "1m":
            base = datetime(2026, 11, 5, 9, 31, tzinfo=ET)
            px = {"AAA": 10.0, "BBB": 20.0, "CCC": 30.0, "DDD": 40.0}[sym]
            rows = []
            for i in range(count):
                ts = base + timedelta(minutes=i)
                o = px + i * 0.0123456789
                rows.append(dict(timestamp=ts.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%S.000Z"), openPrice=f"{o:.6f}", highPrice=f"{o + 0.05:.6f}", lowPrice=f"{o - 0.05:.6f}", closePrice=f"{o + 0.01:.6f}", volume=str(1000 + i)))
            rows.append(dict(timestamp="2026-11-05T15:31:00.000Z", openPrice="1", highPrice="1", lowPrice="1", closePrice="1", volume="1"))
            return dict(candles=list(reversed(rows)), symbol=sym, interval="1m")
        return dict(candles=[dict(timestamp="2026-11-05T05:00:00.000Z", closePrice="99"), dict(timestamp="2026-11-04T05:00:00.000Z", closePrice="9.876543"), dict(timestamp="2026-11-03T05:00:00.000Z", closePrice="9.5")], symbol=sym, interval="1d")

    def orderbook(self, sym):
        self.calls.append(("orderbook", sym))
        return dict(timestamp="2026-11-05T10:30:19.000-05:00", asks=[dict(price="10.25", volume="300")], bids=[dict(price="10.24", volume="250")])


S.Toss = FakeToss
sent_at = {}
RESP = {"answers": {"action": {"choice": "hold", "probabilities": {"buy": 0.2, "hold": 0.8}, "confidence": 0.7}, "which": {"choice": "none", "probabilities": {"AAA": 0.2, "BBB": 0.2, "CCC": 0.2, "none": 0.4}}, "confidence": {"score": 1.0, "probabilities": [0.3, 0.4, 0.2, 0.05, 0.05]}, "risk_flag": {"noul": 0.4}}, "usage": {"input_tokens": 1234, "output_tokens": 0, "cost": 0}}
RESPONSE_BODY = {"body": json.dumps(RESP).encode(), "status": 200}


class FakeResp:
    def __init__(self, body, status):
        self.body, self.status = body, status

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fake_urlopen(req, timeout=None):
    bundles = sorted(os.listdir(in_dir))
    bdir = os.path.join(in_dir, bundles[-1])
    sent_at["request_json_exists"] = os.path.exists(os.path.join(bdir, "request.json"))
    sent_at["headers_json_exists"] = os.path.exists(os.path.join(bdir, "headers.json"))
    sent_at["manifest_exists"] = os.path.exists(os.path.join(bdir, "manifest.json"))
    sent_at["body_sha"] = hashlib.sha256(req.data).hexdigest()
    sent_at["auth_header_present"] = req.get_header("Authorization") is not None
    sent_at["auth_value_is_dummy"] = req.get_header("Authorization") == "Bearer dummy-never-sent"
    sent_at["images_header"] = req.get_header("Typesafe-danger-zone-images")
    sent_at["timeout"] = timeout
    sent_at["calls"] = sent_at.get("calls", 0) + 1
    if RESPONSE_BODY["status"] != 200:
        raise urllib.error.HTTPError(S.J.URL, RESPONSE_BODY["status"], "err", {"cf-ray": "ray-1"}, __import__("io").BytesIO(RESPONSE_BODY["body"]))
    return FakeResp(RESPONSE_BODY["body"], 200)


S.J.urllib.request.urlopen = fake_urlopen


def run(label):
    if os.path.exists(out_log):
        os.remove(out_log)
    sent_at.clear()
    S.main()
    rows = [json.loads(l) for l in open(out_log, encoding="utf-8")]
    print(f"[{label}] kinds: {[r['kind'] for r in rows]} | urlopen calls: {sent_at.get('calls', 0)}")
    return rows


def manifest_check(bdir):
    man = json.load(open(os.path.join(bdir, "manifest.json")))
    bad = [n for n, h in man["files"].items() if hashlib.sha256(open(os.path.join(bdir, n), "rb").read()).hexdigest() != h]
    return sorted(man["files"]), bad


rows = run("normal")
ask = next(r for r in rows if r["kind"] == "ask")
ans = next(r for r in rows if r["kind"] == "answer")
bdir = os.path.join(PENNY, ask["bundle"])
files = sorted(os.listdir(bdir))
print("bundle:", ask["request_id"], files)
req = open(os.path.join(bdir, "request.json"), "rb").read()
print("request sha == sent body sha:", hashlib.sha256(req).hexdigest() == sent_at["body_sha"], "| == answer.request_sha256:", ans["request_sha256"] == sent_at["body_sha"])
print("saved before send:", {k: sent_at[k] for k in ("request_json_exists", "headers_json_exists", "manifest_exists", "auth_header_present", "auth_value_is_dummy", "images_header", "timeout")})
body = json.loads(req)
print("request keys:", sorted(body), "| 'Authorization' in request body:", "Authorization" in req.decode())
hdr = json.load(open(os.path.join(bdir, "headers.json")))
print("headers.json has no auth:", "Authorization" not in hdr)
for ref, a in body["assets"].items():
    png = base64.b64decode(a["data_base64"])
    sym = ref.split("_", 1)[1]
    disk = open(os.path.join(bdir, f"chart_{sym}.png"), "rb").read()
    print(f"  {ref}: sent png == saved file: {png == disk}, sha matches meta: {hashlib.sha256(disk).hexdigest() == ask['charts'][sym]['png_sha256']}")
inp = json.load(open(os.path.join(bdir, "inputs.json"), encoding="utf-8"))
s0 = inp["AAA"]
print("inputs AAA keys:", sorted(s0), "| raw candles:", len(s0["candles"]["raw"]["candles"]), "bars_used:", len(s0["bars_used"]), "dropped:", s0["candles"]["dropped"], "| prev_close:", s0["prev_close"]["value"], s0["prev_close"]["source_ts"])
print("bars_used == float(raw strings):", all(s0["bars_used"][i][1] == [float(r[k]) for k in ("openPrice", "highPrice", "lowPrice", "closePrice", "volume")] for i, r in enumerate(sorted(s0["candles"]["raw"]["candles"], key=lambda r: r["timestamp"])[:60])))
print("questions.json == request.questions:", json.load(open(os.path.join(bdir, "questions.json"), encoding="utf-8")) == body["questions"], "| state.json == request.state:", json.load(open(os.path.join(bdir, "state.json"), encoding="utf-8")) == body["state"])
mfiles, bad = manifest_check(bdir)
print("manifest files:", mfiles, "| hash mismatches (CRLF check):", bad, "| any CR in json files:", any(b"\r" in open(os.path.join(bdir, n), "rb").read() for n in files if n.endswith(".json")))
resp = json.load(open(os.path.join(bdir, "response.json"), encoding="utf-8"))
rawr = open(os.path.join(bdir, "response.raw"), "rb").read()
print("response.raw == fake body:", rawr == RESPONSE_BODY["body"], "| response.json links:", resp["request_sha256"] == sent_at["body_sha"], resp["response_raw_sha256"] == hashlib.sha256(rawr).hexdigest(), resp["status"], resp["received"]["bytes"] == len(rawr))
try:
    S.write_new(os.path.join(bdir, "request.json"), b"overwrite attempt")
    print("overwrite refused: NO (bad)")
except FileExistsError:
    print("overwrite refused: yes; unchanged:", open(os.path.join(bdir, "request.json"), "rb").read() == req)

orig_put = S.Bundle.put


def failing(names):
    def put(self, name, data):
        if name in names:
            raise OSError(f"disk full (simulated) on {name}")
        return orig_put(self, name, data)
    return put


print("--- A: request.json save fails -> no send")
S.Bundle.put = failing({"request.json"})
rows = run("A")
sk = next(r for r in rows if r["kind"] == "ask_skipped")
print("   transport_attempted:", sk["transport_attempted"], "| reason:", sk["reason"][:75])
print("--- B: headers.json save fails -> no send, not marked as transport")
S.Bundle.put = failing({"headers.json"})
rows = run("B")
sk = next(r for r in rows if r["kind"] == "ask_skipped")
print("   transport_attempted:", sk["transport_attempted"], "| reason:", sk["reason"][:75])
print("--- C: manifest.json save fails -> no send")
S.Bundle.put = failing({"manifest.json"})
rows = run("C")
sk = next(r for r in rows if r["kind"] == "ask_skipped")
print("   transport_attempted:", sk["transport_attempted"], "| reason:", sk["reason"][:75])
print("--- D: 2nd chart (BBB) save fails -> whole round blocked")
S.Bundle.put = failing({"chart_BBB.png"})
rows = run("D")
sk = next(r for r in rows if r["kind"] == "ask_skipped")
print("   transport_attempted:", sk["transport_attempted"], "| reason:", sk["reason"][:80], "| ask row present:", any(r["kind"] == "ask" for r in rows))
print("--- E: response.raw save fails after receipt")
S.Bundle.put = failing({"response.raw"})
rows = run("E")
sk = next(r for r in rows if r["kind"] == "ask_skipped")
print("   transport_attempted:", sk["transport_attempted"], "| received:", sk["received"] and sk["received"]["status"], "| reason:", sk["reason"][:70])
S.Bundle.put = orig_put
print("--- F: HTTP 500 with 900-char body -> full body kept")
RESPONSE_BODY.update(body=b"E" * 900, status=500)
rows = run("F")
ans = next(r for r in rows if r["kind"] == "answer")
bdir = os.path.join(PENNY, next(r for r in rows if r["kind"] == "ask")["bundle"])
print("   answer status:", ans["status"], "| error len in log:", len(ans["raw"]["error"]), "| response.raw len:", len(open(os.path.join(bdir, "response.raw"), "rb").read()), "| response.json parsed.error len:", len(json.load(open(os.path.join(bdir, "response.json"), encoding="utf-8"))["parsed"]["error"]))
print("--- G: HTTP 200 with invalid JSON -> raw kept, no crash, loop ends normally")
RESPONSE_BODY.update(body=b"<html>not json</html>", status=200)
rows = run("G")
ans = next(r for r in rows if r["kind"] == "answer")
bdir = os.path.join(PENNY, next(r for r in rows if r["kind"] == "ask")["bundle"])
print("   answer status:", ans["status"], "| raw.error:", ans["raw"].get("error", "")[:30], "| raw_text kept:", ans["raw"].get("raw_text") == "<html>not json</html>", "| response.raw ok:", open(os.path.join(bdir, "response.raw"), "rb").read() == b"<html>not json</html>", "| followed by:", [r["kind"] for r in rows][-2:])
RESPONSE_BODY.update(body=json.dumps(RESP).encode(), status=200)
print("--- H: transport error (URLError) after save -> attempted, not received")
def boom(req, timeout=None):
    sent_at["calls"] = sent_at.get("calls", 0) + 1
    raise urllib.error.URLError("connection reset (simulated)")
S.J.urllib.request.urlopen = boom
rows = run("H")
sk = next(r for r in rows if r["kind"] == "ask_skipped")
print("   transport_attempted:", sk["transport_attempted"], "| received:", sk["received"], "| reason:", sk["reason"][:80])
S.J.urllib.request.urlopen = fake_urlopen
print("--- J: api_key() raises OSError after save -> prepared, not attempted, urlopen 0")
orig_key = S.J.api_key
def bad_key():
    raise OSError("credential store unavailable (simulated)")
S.J.api_key = bad_key
rows = run("J")
sk = next(r for r in rows if r["kind"] == "ask_skipped")
print("   prepared:", sk["prepared"], "| transport_attempted:", sk["transport_attempted"], "| reason:", sk["reason"][:80])
S.J.api_key = orig_key
for label, body in (("K", b"null"), ("L", b"[]"), ("M", b'"just a string"'), ("N", b"42")):
    print(f"--- {label}: HTTP 200 body {body!r} -> no crash, invalid_response_body, no_action, normal end")
    RESPONSE_BODY.update(body=body, status=200)
    rows = run(label)
    ans = next(r for r in rows if r["kind"] == "answer")
    print("   answer status:", ans["status"], "| error:", ans["raw"].get("error", "")[:45], "| raw_text kept:", ans["raw"].get("raw_text") == body.decode(), "| tail:", [r["kind"] for r in rows][-2:])
print("--- O: usage=null / usage=[] in an otherwise valid object -> hold, no crash")
for body in (json.dumps(dict(RESP, usage=None)).encode(), json.dumps(dict(RESP, usage=[])).encode()):
    RESPONSE_BODY.update(body=body, status=200)
    rows = run("O")
    ans = next(r for r in rows if r["kind"] == "answer")
    print("   answer status:", ans["status"], "| input_tokens:", ans["input_tokens"], "| tail:", [r["kind"] for r in rows][-2:])
RESPONSE_BODY.update(body=json.dumps(RESP).encode(), status=200)
print("--- I: idle path (no candidates): bundle dropped if empty")
open(src, "w").write(json.dumps(dict(kind="scan", top=[])) + "\n")
n_before = len(os.listdir(in_dir))
rows = run("I")
print("   bundle dirs before/after:", n_before, len(os.listdir(in_dir)))
