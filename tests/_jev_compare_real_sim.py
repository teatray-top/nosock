import json
import os
import sys
import time
import urllib.request

PENNY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PENNY)
os.chdir(PENNY)
import jev_compare_run as X
import jev_client as J

manifest, expect, delay = sys.argv[1], sys.argv[2], float(sys.argv[3])
J.api_key = lambda: "dummy-sim-key"
calls = []


class R:
    status = 200

    def __init__(self, b):
        self.b = b

    def read(self):
        return self.b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fake(req, timeout=None):
    calls.append(1)
    time.sleep(delay)
    body = json.loads(req.data)
    syms = [k for k in body["questions"].get("which", {}).get("criteria", {}) if k != "none"]
    return R(json.dumps({"answers": {"action": {"choice": "hold", "probabilities": {"buy": 0.3, "hold": 0.7}}, "which": {"choice": "none", "probabilities": {s: 0.1 for s in syms} | {"none": 0.9}}, "confidence": {"score": 1.0}, "risk_flag": {"noul": 0.3}}, "usage": {"cost": 0, "input_tokens": 1}}).encode())


urllib.request.urlopen = fake
sys.argv = ["x", manifest, expect, "--real"]
try:
    rc = X.main()
    print("SIM rc", rc, "calls", len(calls), "run", os.path.basename(X.LAST_RUN_DIR or ""))
except X.Abort as e:
    print("SIM ABORT", str(e)[:120], "calls", len(calls))
