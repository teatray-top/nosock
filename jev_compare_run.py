import glob
import hashlib
import json
import os
import sys
import time
import urllib.error
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

import jev_client as J

HERE = os.path.dirname(os.path.abspath(__file__))
KST = ZoneInfo("Asia/Seoul")
OUT_ROOT = os.environ.get("JEV_COMPARE_OUT") or os.path.join(HERE, "runs", "jev_compare")
LAST_RUN_DIR = None
REQUIRED_ATTEMPT_KEYS = ("attempt_index", "case_id", "arm", "request_path", "request_sha256", "request_bytes", "headers_path", "headers_sha256")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def kst_ms():
    return datetime.now(KST).isoformat(timespec="milliseconds")


def write_new(path, data):
    raw = bytes(data) if isinstance(data, (bytes, bytearray)) else json.dumps(data, ensure_ascii=False, indent=1).encode("utf-8")
    with open(path, "xb") as f:
        f.write(raw)
    return sha(raw)


class Abort(Exception):
    pass


def num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def verify(manifest_path, expect_sha):
    raw = open(manifest_path, "rb").read()
    if sha(raw) != expect_sha:
        raise Abort(f"manifest sha {sha(raw)[:16]} != expected {expect_sha[:16]}")
    m = json.loads(raw)
    if m.get("retry") != 0:
        raise Abort(f"manifest retry={m.get('retry')!r}, executor only runs retry=0 protocols")
    order = m.get("attempt_order") or []
    if not order or len(order) > int(m.get("max_attempts", 0)):
        raise Abort(f"{len(order)} attempts listed but max_attempts={m.get('max_attempts')}")
    if [a.get("attempt_index") for a in order] != list(range(1, len(order) + 1)):
        raise Abort("attempt_index is not a contiguous 1..N sequence")
    if m.get("endpoint") != J.URL:
        raise Abort(f"manifest endpoint {m.get('endpoint')!r} != client URL {J.URL!r}")
    seen = set()
    fixed = []
    for a in order:
        missing = [k for k in REQUIRED_ATTEMPT_KEYS if k not in a]
        if missing:
            raise Abort(f"attempt {a.get('attempt_index')} lacks {missing}")
        key = (a["case_id"], a["arm"])
        if key in seen:
            raise Abort(f"duplicate case/arm {key}")
        seen.add(key)
        body = open(a["request_path"], "rb").read()
        if sha(body) != a["request_sha256"] or len(body) != a["request_bytes"]:
            raise Abort(f"attempt {a['attempt_index']} request bytes/sha mismatch")
        hraw = open(a["headers_path"], "rb").read()
        if sha(hraw) != a["headers_sha256"]:
            raise Abort(f"attempt {a['attempt_index']} headers sha mismatch")
        hdrs = json.loads(hraw)
        if not isinstance(hdrs, dict) or any(k.lower() == "authorization" for k in hdrs):
            raise Abort(f"attempt {a['attempt_index']} headers file is not a plain object or carries an Authorization header")
        b = json.loads(body)
        if b.get("model") != m.get("model"):
            raise Abort(f"attempt {a['attempt_index']} body model {b.get('model')!r} != manifest {m.get('model')!r}")
        has_assets = bool(b.get("assets"))
        has_img_hdr = any(k.lower() == "typesafe-danger-zone-images" for k in hdrs)
        if has_assets != has_img_hdr:
            raise Abort(f"attempt {a['attempt_index']} assets={has_assets} but images header={has_img_hdr}")
        qs = b.get("questions") or {}
        if not isinstance(qs, dict) or "action" not in qs:
            raise Abort(f"attempt {a['attempt_index']} body lacks an action question")
        allowed = dict(action=sorted((qs["action"].get("criteria") or {}).keys()), which=sorted((qs.get("which", {}).get("criteria") or {}).keys()) if "which" in qs else None, confidence="confidence" in qs, risk_flag="risk_flag" in qs)
        fixed.append(dict(a=a, body=body, hdrs=hdrs, allowed=allowed, check=dict(attempt_index=a["attempt_index"], case_id=a["case_id"], arm=a["arm"], request_sha256=a["request_sha256"], headers_sha256=a["headers_sha256"], request_bytes=len(body), assets=sorted(b.get("assets", {}).keys()), questions=sorted(qs.keys()), allowed=allowed)))
    return m, fixed


def ledger_path(msha):
    return os.path.join(OUT_ROOT, f"attempts_{msha[:16]}.jsonl")


def lock_path(msha):
    return os.path.join(OUT_ROOT, f"lock_{msha[:16]}")


def read_ledger(msha):
    p = ledger_path(msha)
    prior_real = sorted(d for d in glob.glob(os.path.join(OUT_ROOT, f"*_real_{msha[:8]}_*")) if os.path.isdir(d))
    if not os.path.exists(p):
        if prior_real:
            raise Abort(f"ledger {os.path.basename(p)} is missing but {len(prior_real)} prior real run dir(s) exist for this manifest; not resuming")
        return 0, []
    rows = []
    for i, line in enumerate(open(p, "rb").read().splitlines(), 1):
        try:
            r = json.loads(line)
            if r.get("kind") not in ("reserved", "done", "abort", "run_start", "run_end") or (r["kind"] == "reserved" and not isinstance(r.get("attempt_index"), int)):
                raise ValueError("bad row")
        except (ValueError, KeyError, TypeError):
            raise Abort(f"ledger {os.path.basename(p)} line {i} is corrupt; not resuming (manual review required)")
        rows.append(r)
    return sum(r["kind"] == "reserved" for r in rows), rows


def ledger_append(msha, row):
    with open(ledger_path(msha), "ab") as f:
        f.write((json.dumps(dict(ts=kst_ms(), **row), ensure_ascii=False) + "\n").encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


def acquire_lock(msha, run_id):
    try:
        fd = os.open(lock_path(msha), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            holder = open(lock_path(msha), encoding="utf-8").read()[:200]
        except OSError:
            holder = "?"
        raise Abort(f"another run holds {os.path.basename(lock_path(msha))} ({holder}); exclusive execution refused. If stale, remove it manually after confirming no process is running")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(dict(run_id=run_id, pid=os.getpid(), ts=kst_ms())))


def mock_opener(answer):
    def opener(req, timeout=None):
        if any(k.lower() == "authorization" for k, v in req.header_items()):
            raise Abort("mock transport received an Authorization header")
        return _MockResp(json.dumps(answer).encode())
    return opener


class _MockResp:
    status = 200

    def __init__(self, body):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def mock_answer(a, body):
    syms = [k for k in body["questions"].get("which", {}).get("criteria", {}) if k != "none"]
    return {"answers": {"action": {"choice": "hold", "probabilities": {"buy": 0.25, "hold": 0.75}, "confidence": 0.6}, "which": {"choice": syms[0] if syms else "none", "probabilities": {s: 0.5 for s in syms} | {"none": 0.5}}, "confidence": {"score": 1.0, "probabilities": [0.4, 0.3, 0.2, 0.05, 0.05]}, "risk_flag": {"noul": 0.45}}, "usage": {"input_tokens": 0, "output_tokens": 0, "cost": 0}, "mock": True, "attempt_index": a["attempt_index"]}


def interpret(out, status, allowed):
    errs = []
    if status != 200:
        errs.append(f"http {status}")
    if not isinstance(out, dict) or "error" in out:
        errs.append("response error/unparsed")
    ans = J.answers(out) if isinstance(out, dict) else {}
    act, act_p, act_c = J.read_choice(ans, "action")
    which, which_p, _ = J.read_choice(ans, "which")
    conf, _, _ = J.read_score(ans, "confidence", len(J.CONF_LEVELS))
    risk = J.read_noul(ans, "risk_flag")
    if act not in allowed["action"]:
        errs.append(f"action {act!r} not in {allowed['action']}")
    if allowed["which"] is not None and which not in allowed["which"]:
        errs.append(f"which {which!r} not in allowed list")
    if allowed["confidence"] and conf is None:
        errs.append("confidence missing/out of range")
    if allowed["risk_flag"] and risk is None:
        errs.append("risk_flag missing/out of range")
    usage = out.get("usage") if isinstance(out, dict) and isinstance(out.get("usage"), dict) else {}
    cost = num(usage.get("cost"))
    rmodel = out.get("model") if isinstance(out, dict) else None
    return dict(action=act, action_p=act_p, which=which, which_p=which_p, confidence=conf, risk_flag=risk, input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"), cost=cost, cost_known=cost is not None, response_model=rmodel if isinstance(rmodel, str) else None, model_known=isinstance(rmodel, str), schema_errors=errs, valid=not errs)


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: jev_compare_run.py <manifest.json> <expected manifest sha256> [--real]")
    manifest_path, expect = sys.argv[1], sys.argv[2]
    real = "--real" in sys.argv[3:]
    mode = "real" if real else "mock"
    m, fixed = verify(manifest_path, expect)
    msha = sha(open(manifest_path, "rb").read())
    os.makedirs(OUT_ROOT, exist_ok=True)
    n = len(fixed)
    run_id = f"{datetime.now(KST).strftime('%Y%m%d_%H%M%S')}_{mode}_{msha[:8]}_{uuid.uuid4().hex[:4]}"
    used = 0
    if real:
        acquire_lock(msha, run_id)
    try:
        if real:
            used, _ = read_ledger(msha)
            if used + n > int(m["max_attempts"]):
                raise Abort(f"ledger shows {used} reserved attempt(s); {used} + {n} would exceed max_attempts {m['max_attempts']}")
        rdir = os.path.join(OUT_ROOT, run_id)
        os.makedirs(rdir, exist_ok=False)
        global LAST_RUN_DIR
        LAST_RUN_DIR = rdir
        code_sha = {f: sha(open(os.path.join(HERE, f), "rb").read()) for f in ("jev_compare_run.py", "jev_client.py")}
        write_new(os.path.join(rdir, "run_meta.json"), dict(run_id=run_id, mode=mode, manifest_path=manifest_path, manifest_sha256=msha, model=m["model"], endpoint=m["endpoint"], timeout_s=m["timeout_s"], max_attempts=m["max_attempts"], retry=m["retry"], attempts_planned=n, ledger_reserved_before=used, verified=[f["check"] for f in fixed], code_sha256=code_sha, started=kst_ms()))
        log_path = os.path.join(rdir, "log.jsonl")

        def log(**kw):
            row = dict(ts=kst_ms(), **kw)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False)[:220], flush=True)

        log(kind="start", run_id=run_id, mode=mode, attempts_planned=n, manifest_sha256=msha, ledger_reserved_before=used)
        if real:
            ledger_append(msha, dict(kind="run_start", run_id=run_id, planned=n))
        results = []
        stop = None
        auth = ("Bearer " + J.api_key()) if real else None
        for fx in fixed:
            a, body, hdrs = fx["a"], fx["body"], fx["hdrs"]
            tag = f"{a['attempt_index']:02d}_{a['case_id']}_{a['arm']}"
            if sha(body) != a["request_sha256"] or sha(open(a["request_path"], "rb").read()) != a["request_sha256"] or sha(open(a["headers_path"], "rb").read()) != a["headers_sha256"]:
                stop = f"{tag}: request/headers bytes changed between verification and send"
                log(kind="abort", attempt_index=a["attempt_index"], reason=stop)
                break
            state = dict(attempted=False, received=None)
            try:
                write_new(os.path.join(rdir, tag + ".request.json"), body)
                write_new(os.path.join(rdir, tag + ".headers.json"), hdrs)
            except OSError as e:
                stop = f"{tag}: request copy save failed: {e}"
                log(kind="abort", attempt_index=a["attempt_index"], reason=stop)
                break
            if real:
                used_now, _ = read_ledger(msha)
                if used_now + 1 > int(m["max_attempts"]):
                    stop = f"{tag}: ledger shows {used_now} reserved attempts; cap {m['max_attempts']} reached"
                    log(kind="abort", attempt_index=a["attempt_index"], reason=stop)
                    break
                ledger_append(msha, dict(kind="reserved", run_id=run_id, attempt_index=a["attempt_index"], request_sha256=a["request_sha256"]))

            def before_send():
                state["attempted"] = True
                state["attempt_ts"] = kst_ms()

            def save_response(raw, st, lt):
                state["received"] = dict(status=st, latency_s=lt, received_ts=kst_ms(), bytes=len(raw))
                state["response_sha256"] = write_new(os.path.join(rdir, tag + ".response.raw"), raw)

            send_headers = {**hdrs, "Authorization": auth} if real else dict(hdrs)
            try:
                out, status, lat = J.post(body, send_headers, timeout=int(m["timeout_s"]), before_send=before_send, save_response=save_response, opener=None if real else mock_opener(mock_answer(a, json.loads(body))))
            except (OSError, Abort) as e:
                stop = f"{tag}: {'transport' if state['attempted'] else 'pre-send'} failure: {type(e).__name__}: {str(e)[:160]}"
                log(kind="abort", attempt_index=a["attempt_index"], case_id=a["case_id"], arm=a["arm"], transport_attempted=state["attempted"], received=state["received"], reason=stop)
                if real:
                    ledger_append(msha, dict(kind="abort", run_id=run_id, attempt_index=a["attempt_index"], attempted=state["attempted"], reason=stop[:200]))
                break
            it = interpret(out, status, fx["allowed"])
            row = dict(kind="attempt", attempt_index=a["attempt_index"], case_id=a["case_id"], arm=a["arm"], request_sha256=a["request_sha256"], transport_attempted=state["attempted"], attempt_ts=state.get("attempt_ts"), received=state["received"], response_sha256=state.get("response_sha256"), status=status, latency_s=lat, **it, error=out.get("error") if isinstance(out, dict) else None)
            try:
                write_new(os.path.join(rdir, tag + ".response.json"), dict(status=status, latency_s=lat, received=state["received"], request_sha256=a["request_sha256"], response_raw_sha256=state.get("response_sha256"), parsed=out, interpretation=it))
            except OSError as e:
                stop = f"{tag}: response.json save failed: {e}"
                log(kind="abort", attempt_index=a["attempt_index"], reason=stop, **{k: row[k] for k in ("status", "received", "response_sha256")})
                break
            log(**row)
            results.append(row)
            if real:
                ledger_append(msha, dict(kind="done", run_id=run_id, attempt_index=a["attempt_index"], status=status, valid=it["valid"], cost=it["cost"], response_model=it["response_model"]))
            if status != 200 or (isinstance(out, dict) and "error" in out):
                stop = f"{tag}: http {status} / {str(out.get('error') if isinstance(out, dict) else out)[:120]}; no retry, run stopped"
            elif it["cost_known"] and it["cost"] != 0:
                stop = f"{tag}: response reports usage.cost={it['cost']} (non-zero billing); stopped before the next call"
            elif it["model_known"] and it["response_model"] != m["model"]:
                stop = f"{tag}: response model {it['response_model']!r} != manifest model {m['model']!r}; stopped before the next call"
            if stop:
                log(kind="abort", attempt_index=a["attempt_index"], reason=stop)
                break
        cases = {}
        for r in results:
            cases.setdefault(r["case_id"], {})[r["arm"]] = {k: r[k] for k in ("action", "which", "confidence", "risk_flag", "input_tokens", "latency_s", "valid", "schema_errors", "cost_known", "model_known")}
        pairs = {}
        for c, v in cases.items():
            both_valid = "A" in v and "B" in v and v["A"]["valid"] and v["B"]["valid"]
            pairs[c] = dict(both_present="A" in v and "B" in v, both_valid=both_valid, same_action=(v["A"]["action"] == v["B"]["action"]) if both_valid else None, same_which=(v["A"]["which"] == v["B"]["which"]) if both_valid else None)
        arms = sorted({r["arm"] for r in results})
        summary = dict(run_id=run_id, mode=mode, attempts_planned=n, attempts_done=len(results), valid_schema=sum(r["valid"] for r in results), invalid_schema=[dict(attempt_index=r["attempt_index"], errors=r["schema_errors"]) for r in results if not r["valid"]], cost_unknown=sum(not r["cost_known"] for r in results), model_unknown=sum(not r["model_known"] for r in results), stopped=stop, finished=kst_ms(), cases=cases, pairs=pairs, pairs_both_valid=sum(p["both_valid"] for p in pairs.values()), pairs_same_action=sum(bool(p["same_action"]) for p in pairs.values()), action_counts={arm: {k: sum(1 for r in results if r["arm"] == arm and r["valid"] and r["action"] == k) for k in ("buy", "hold", "sell")} for arm in arms}, tokens={arm: [r["input_tokens"] for r in results if r["arm"] == arm] for arm in arms}, latency={arm: [r["latency_s"] for r in results if r["arm"] == arm] for arm in arms})
        write_new(os.path.join(rdir, "summary.json"), summary)
        log(kind="end", attempts_done=len(results), valid_schema=summary["valid_schema"], stopped=stop)
        if real:
            ledger_append(msha, dict(kind="run_end", run_id=run_id, done=len(results), stopped=stop))
        print("RUN", run_id, "done", len(results), "/", n, "valid_schema", summary["valid_schema"], "stopped:", stop)
        return 0 if stop is None else 2
    finally:
        if real:
            try:
                os.remove(lock_path(msha))
            except OSError:
                pass


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Abort as e:
        print("ABORT:", e)
        sys.exit(3)
