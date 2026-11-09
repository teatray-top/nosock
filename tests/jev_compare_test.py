import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error

PENNY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PENNY)
os.chdir(PENNY)
STAGE = sys.argv[1]
EXPECT = sys.argv[2]
tmp = tempfile.mkdtemp()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def copy_stage(name):
    d = os.path.join(tmp, name)
    shutil.copytree(STAGE, d)
    m = json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
    for a in m["attempt_order"]:
        for k in ("request_path", "headers_path"):
            a[k] = os.path.join(d, os.path.relpath(a[k], STAGE))
    return d, m


def save(d, m):
    p = os.path.join(d, "manifest.json")
    raw = json.dumps(m, ensure_ascii=False, indent=1).encode()
    open(p, "wb").write(raw)
    return p, sha(raw)


def run(manifest, expect, extra=()):
    r = subprocess.run([sys.executable, "-X", "utf8", "jev_compare_run.py", manifest, expect, *extra], capture_output=True, text=True, encoding="utf-8", timeout=120)
    tail = [l for l in r.stdout.strip().splitlines() if l.startswith("ABORT") or l.startswith("RUN")]
    return r.returncode, tail[-1] if tail else r.stdout[-200:] + r.stderr[-300:]


import jev_compare_run as X
import jev_client as J

print("1 wrong expected sha ->", run(os.path.join(STAGE, "manifest.json"), "0" * 64))
d, m = copy_stage("tampered_body")
p, s = save(d, m)
open(m["attempt_order"][4]["request_path"], "ab").write(b" ")
print("2 tampered request body (attempt 5) ->", run(p, s))
d, m = copy_stage("auth_header")
h = json.load(open(m["attempt_order"][0]["headers_path"], encoding="utf-8")); h["Authorization"] = "Bearer leaked"
raw = json.dumps(h).encode(); open(m["attempt_order"][0]["headers_path"], "wb").write(raw); m["attempt_order"][0]["headers_sha256"] = sha(raw)
p, s = save(d, m)
print("3 headers file with Authorization ->", run(p, s))
d, m = copy_stage("out_of_order")
m["attempt_order"][1]["attempt_index"] = 7
p, s = save(d, m)
print("4 non-contiguous attempt_index ->", run(p, s))
d, m = copy_stage("over_cap")
m["max_attempts"] = 23
p, s = save(d, m)
print("5 24 attempts with max_attempts=23 ->", run(p, s))
d, m = copy_stage("retry1")
m["retry"] = 1
p, s = save(d, m)
print("6 retry=1 manifest ->", run(p, s))
d, m = copy_stage("dup")
m["attempt_order"][1]["case_id"] = m["attempt_order"][0]["case_id"]; m["attempt_order"][1]["arm"] = "A"
p, s = save(d, m)
print("7 duplicate case/arm ->", run(p, s))
d, m = copy_stage("img_hdr")
a = m["attempt_order"][1]
h = json.load(open(a["headers_path"], encoding="utf-8")); h.pop("TypeSafe-Danger-Zone-Images", None)
raw = json.dumps(h).encode(); open(a["headers_path"], "wb").write(raw); a["headers_sha256"] = sha(raw)
p, s = save(d, m)
print("8 image body without images header ->", run(p, s))

print("--- in-process: HTTP 500 at attempt 3 -> stop after 3, no retry")
d, m = copy_stage("http500")
p, s = save(d, m)
calls = []
orig_mock = X.mock_opener


def failing_mock(answer):
    def opener(req, timeout=None):
        calls.append(answer["attempt_index"])
        if answer["attempt_index"] == 3:
            raise urllib.error.HTTPError(J.URL, 500, "boom", {"cf-ray": "r"}, io.BytesIO(b"E" * 700))
        return X._MockResp(json.dumps(answer).encode())
    return opener


X.mock_opener = failing_mock
sys.argv = ["x", p, s]
rc = X.main()
X.mock_opener = orig_mock
rdir = X.LAST_RUN_DIR
summ = json.load(open(os.path.join(rdir, "summary.json"), encoding="utf-8"))
rows = [json.loads(l) for l in open(os.path.join(rdir, "log.jsonl"), encoding="utf-8")]
print("   rc", rc, "| opener calls", calls, "| attempts_done", summ["attempts_done"], "valid_schema", summ["valid_schema"], "| stopped:", (summ["stopped"] or "")[:70])
print("   response.raw of attempt 3 kept:", len(open(os.path.join(rdir, "03_" + m["attempt_order"][2]["case_id"] + "_" + m["attempt_order"][2]["arm"] + ".response.raw"), "rb").read()), "bytes | log kinds:", [r["kind"] for r in rows])

print("--- in-process: transport error (URLError) at attempt 2 -> abort row with attempted=true, no further attempts")
calls.clear()


def urlerr_mock(answer):
    def opener(req, timeout=None):
        calls.append(answer["attempt_index"])
        if answer["attempt_index"] == 2:
            raise urllib.error.URLError("reset (simulated)")
        return X._MockResp(json.dumps(answer).encode())
    return opener


X.mock_opener = urlerr_mock
rc = X.main()
X.mock_opener = orig_mock
rdir = X.LAST_RUN_DIR
rows = [json.loads(l) for l in open(os.path.join(rdir, "log.jsonl"), encoding="utf-8")]
ab = next(r for r in rows if r["kind"] == "abort")
print("   rc", rc, "| opener calls", calls, "| abort attempt", ab["attempt_index"], "attempted", ab["transport_attempted"], "received", ab["received"], "| kinds", [r["kind"] for r in rows])

print("--- S1: schema validity: {} response, action=purchase, which off-list -> invalid, not counted as pair matches")
d, m = copy_stage("schema")
p, s = save(d, m)


def schema_mock(answer):
    i = answer["attempt_index"]
    if i == 1:
        body = {}
    elif i == 2:
        body = dict(answer, answers=dict(answer["answers"], action={"choice": "purchase"}))
    elif i == 3:
        body = dict(answer, answers=dict(answer["answers"], which={"choice": "ZZZZ"}))
    else:
        body = answer

    def opener(req, timeout=None):
        return X._MockResp(json.dumps(body).encode())
    return opener


X.mock_opener = schema_mock
sys.argv = ["x", p, s]
rc = X.main()
X.mock_opener = orig_mock
summ = json.load(open(os.path.join(X.LAST_RUN_DIR, "summary.json"), encoding="utf-8"))
print("   rc", rc, "| done", summ["attempts_done"], "valid_schema", summ["valid_schema"], "| invalid:", [(x["attempt_index"], x["errors"][0][:40]) for x in summ["invalid_schema"]])
print("   pair AAL_1030 (A={} , B=purchase):", summ["pairs"]["20261105_AAL_1030"], "| pairs_both_valid", summ["pairs_both_valid"], "pairs_same_action", summ["pairs_same_action"])

print("--- S2: stop on non-zero cost at attempt 1; stop on model mismatch; continue when fields absent (flagged)")
for label, patch in (("cost 0.25", lambda a: dict(a, usage={"cost": 0.25, "input_tokens": 5})), ("model other", lambda a: dict(a, model="jev-2.0-paid")), ("no usage/model", lambda a: {k: v for k, v in a.items() if k not in ("usage",)})):
    def mk(answer, patch=patch):
        body = patch(answer) if answer["attempt_index"] == 1 else answer
        def opener(req, timeout=None):
            return X._MockResp(json.dumps(body).encode())
        return opener
    X.mock_opener = mk
    rc = X.main()
    X.mock_opener = orig_mock
    summ = json.load(open(os.path.join(X.LAST_RUN_DIR, "summary.json"), encoding="utf-8"))
    print(f"   {label}: rc {rc} done {summ['attempts_done']} stopped: {(summ['stopped'] or 'None')[:80]} | cost_unknown {summ['cost_unknown']} model_unknown {summ['model_unknown']}")

print("--- S3: header tampered after verification (attempt 2) -> abort before send")
d, m = copy_stage("hdr_after_verify")
p, s = save(d, m)
orig_verify = X.verify


def tampering_verify(mp, es):
    r = orig_verify(mp, es)
    a2 = r[0]["attempt_order"][1]
    h = json.loads(open(a2["headers_path"], "rb").read()); h.pop("TypeSafe-Danger-Zone-Images", None)
    open(a2["headers_path"], "wb").write(json.dumps(h).encode())
    return r


X.verify = tampering_verify
calls.clear()
X.mock_opener = lambda answer: (lambda req, timeout=None: (calls.append(answer["attempt_index"]), X._MockResp(json.dumps(answer).encode()))[1])
rc = X.main()
X.mock_opener = orig_mock
X.verify = orig_verify
rows = [json.loads(l) for l in open(os.path.join(X.LAST_RUN_DIR, "log.jsonl"), encoding="utf-8")]
ab = next(r for r in rows if r["kind"] == "abort")
print("   rc", rc, "| opener calls", calls, "| abort at", ab["attempt_index"], ":", ab["reason"][:70])

print("--- S4: exclusive real execution + durable ledger (simulated transport, dummy key, isolated OUT dir)")
d, m = copy_stage("real_sim")
m["max_attempts"] = 4
p, s = save(d, m)
outdir = os.path.join(tmp, "out_real_sim")
env = dict(os.environ, JEV_COMPARE_OUT=outdir)
sim = os.path.join(PENNY, "tests", "_jev_compare_real_sim.py")
m2 = json.load(open(p, encoding="utf-8")); m2["attempt_order"] = m2["attempt_order"][:2]; p2, s2 = save(d, m2)
procs = [subprocess.Popen([sys.executable, "-X", "utf8", sim, p2, s2, "0.5"], env=env, stdout=subprocess.PIPE, text=True, encoding="utf-8") for _ in range(2)]
outs = [pr.communicate(timeout=120)[0].strip().splitlines()[-1] for pr in procs]
print("   two concurrent real-sim runs (2 attempts each, cap 4):", outs)
led = [json.loads(l) for l in open(os.path.join(outdir, f"attempts_{s2[:16]}.jsonl"), encoding="utf-8")]
print("   ledger reserved:", sum(r["kind"] == "reserved" for r in led), "done:", sum(r["kind"] == "done" for r in led), "| lock removed:", not os.path.exists(os.path.join(outdir, f"lock_{s2[:16]}")))
r3 = subprocess.run([sys.executable, "-X", "utf8", sim, p2, s2, "0"], env=env, capture_output=True, text=True, encoding="utf-8", timeout=120).stdout.strip().splitlines()[-1]
print("   third sequential run (2 more, total would be 4 <= cap 4):", r3)
r4 = subprocess.run([sys.executable, "-X", "utf8", sim, p2, s2, "0"], env=env, capture_output=True, text=True, encoding="utf-8", timeout=120).stdout.strip().splitlines()[-1]
print("   fourth run (would exceed cap):", r4)
os.remove(os.path.join(outdir, f"attempts_{s2[:16]}.jsonl"))
r5 = subprocess.run([sys.executable, "-X", "utf8", sim, p2, s2, "0"], env=env, capture_output=True, text=True, encoding="utf-8", timeout=120).stdout.strip().splitlines()[-1]
print("   ledger deleted but real run dirs exist:", r5)
open(os.path.join(outdir, f"attempts_{s2[:16]}.jsonl"), "w").write("not json\n")
r6 = subprocess.run([sys.executable, "-X", "utf8", sim, p2, s2, "0"], env=env, capture_output=True, text=True, encoding="utf-8", timeout=120).stdout.strip().splitlines()[-1]
print("   corrupt ledger:", r6)
open(os.path.join(outdir, f"lock_{s2[:16]}"), "w").write("{\"run_id\": \"stale\"}")
r7 = subprocess.run([sys.executable, "-X", "utf8", sim, p2, s2, "0"], env=env, capture_output=True, text=True, encoding="utf-8", timeout=120).stdout.strip().splitlines()[-1]
print("   stale lock present:", r7)
