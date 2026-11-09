import json
import os
import sys
import time
import urllib.error
import urllib.request

import cred

URL = "https://opencode.ai/inference/systemone/v1/systemone"
MODEL = os.environ.get("JEV_MODEL", "jev-1.13-free")
CONF_LEVELS = ["No reason to expect an edge; the action is a coin flip or worse", "Weak hints only (one indicator, no confirmation)", "Moderate: several indicators agree but with notable counter-signals", "Strong: price, volume and order book agree and risk is contained", "Near-certain: clear, confirmed setup with minimal downside"]


def api_key():
    k = os.environ.get("OPENCODE_API_KEY")
    if k:
        return k
    v = cred.load("OpenCodeAPI")
    if isinstance(v, (tuple, list)):
        v = v[-1]
    if not v:
        raise SystemExit("no OpenCode Console API key: set OPENCODE_API_KEY or store Credential Manager target OpenCodeAPI")
    return v


UA = "NoSock-dobby/1.0 (python-urllib; personal trading research)"


def payload(state, questions, model=MODEL, assets=None):
    p = dict(model=model, state=state, questions=questions)
    if assets:
        p["assets"] = assets
    return p


def headers(images=False):
    h = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": UA}
    if images:
        h["TypeSafe-Danger-Zone-Images"] = "true"
    return h


def ask(state, questions, model=MODEL, timeout=60, assets=None, save=None, save_response=None, before_send=None):
    body = json.dumps(payload(state, questions, model, assets)).encode()
    h = headers(bool(assets))
    if save is not None:
        save(body, dict(h))
    return post(body, {**h, "Authorization": "Bearer " + api_key()}, timeout=timeout, before_send=before_send, save_response=save_response)


def post(body, hdrs, timeout=60, before_send=None, save_response=None, opener=None):
    req = urllib.request.Request(URL, data=body, headers=hdrs, method="POST")
    if before_send is not None:
        before_send()
    t0 = time.time()
    ray = None
    try:
        with (opener or urllib.request.urlopen)(req, timeout=timeout) as r:
            raw, status = r.read(), r.status
    except urllib.error.HTTPError as e:
        raw, status, ray = e.read(), e.code, e.headers.get("cf-ray")
    lat = round(time.time() - t0, 2)
    if save_response is not None:
        save_response(raw, status, lat)
    text = raw.decode("utf-8", "replace")
    if status != 200:
        return dict(error=text, ray_id=ray), status, lat
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        return dict(error=f"invalid JSON: {e}", raw_text=text), status, lat
    if not isinstance(parsed, dict):
        return dict(error=f"response is not a JSON object ({type(parsed).__name__})", raw_text=text, parsed=parsed), status, lat
    return parsed, status, lat


def trade_questions(candidates, has_position):
    which = {c["symbol"]: f"{c['symbol']}: {c.get('summary', 'candidate')}" for c in candidates}
    which["none"] = "No candidate is worth buying now"
    action = {"buy": "Open a new long position in one candidate now", "hold": "Do nothing: stay in cash or keep the current position"}
    if has_position:
        action["sell"] = "Close the current position now"
    return {
        "action": {"type": "choice", "instructions": "Given the state, what should the trader do for the next 5 minutes? Round-trip cost is about 0.3%; only buy if the expected move clearly exceeds it.", "criteria": action},
        "which": {"type": "choice", "instructions": "If a buy were made now, which single candidate is the best choice? Choose 'none' if no candidate is worth buying.", "criteria": which},
        "confidence": {"type": "score", "instructions": "How confident are you that the chosen action beats staying in cash after costs over the next 5-60 minutes?", "criteria": CONF_LEVELS},
        "risk_flag": {"type": "noul", "instructions": "Is the best candidate in a severe multi-day downtrend, gapping down on news, or otherwise a falling-knife situation that should not be bought?", "criteria": {"true": "Yes: buying it now is catching a falling knife", "false": "No: the decline is contained or the setup is not a falling knife"}},
    }


def answers(out):
    if not isinstance(out, dict):
        return {}
    a = out.get("answers")
    return a if isinstance(a, dict) else out


def read_choice(ans, qid):
    a = ans.get(qid)
    if isinstance(a, dict) and isinstance(a.get("choice"), str):
        return a["choice"], a.get("probabilities"), a.get("confidence")
    return None, None, None


def num(x, lo, hi):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    if x != x or x in (float("inf"), float("-inf")) or x < lo or x > hi:
        return None
    return x


def read_score(ans, qid, n_levels):
    a = ans.get(qid)
    if isinstance(a, dict):
        s = num(a.get("score"), 0, n_levels - 1)
        if s is not None:
            return s / max(1, n_levels - 1), a.get("probabilities"), a.get("confidence")
    return None, None, None


def read_noul(ans, qid):
    a = ans.get(qid)
    if isinstance(a, dict):
        return num(a.get("noul"), 0, 1)
    return num(a, 0, 1)


def interpret(out, candidates, has_position):
    ans = answers(out)
    action, action_p, action_conf = read_choice(ans, "action")
    which, which_p, _ = read_choice(ans, "which")
    conf, _, _ = read_score(ans, "confidence", len(CONF_LEVELS))
    risk = read_noul(ans, "risk_flag")
    missing = [q for q, v in (("action", action), ("which", which), ("confidence", conf), ("risk_flag", risk)) if v is None]
    if missing:
        return dict(status="invalid", action="hold", reason="missing/abnormal answers: " + ",".join(missing), consistent=None, raw=out)
    syms = {c["symbol"] for c in candidates}
    consistent, final, reason = True, action, action
    if action == "buy":
        if has_position:
            consistent, final, reason = False, "hold", "buy while holding"
        elif which not in syms:
            consistent, final, reason = False, "hold", f"buy but which={which!r}"
        elif risk >= 0.5:
            consistent, final, reason = False, "hold", f"buy but risk_flag {risk:.2f}"
        else:
            reason = f"buy {which}"
    elif action == "sell":
        if not has_position:
            consistent, final, reason = False, "hold", "sell without position"
    elif action != "hold":
        return dict(status="invalid", action="hold", reason=f"unknown action {action!r}", consistent=None, raw=out)
    return dict(status="ok", action=final, symbol=which if final == "buy" else None, llm_action=action, which=which, confidence=conf, risk_flag=risk, action_probabilities=action_p, which_probabilities=which_p, action_model_confidence=action_conf, reason=reason, consistent=consistent, raw=out)


def decide(state, candidates, has_position, model=MODEL):
    out, status, lat = ask(state, trade_questions(candidates, has_position), model=model)
    if status != 200:
        return dict(status=f"api_error {status}", action="hold", reason=f"api {status}", consistent=None, raw=out, latency_s=lat)
    d = interpret(out, candidates, has_position)
    d["latency_s"] = lat
    return d


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        cands = [dict(symbol="AMAT", summary="down 19% on the week, bouncing"), dict(symbol="NVDA", summary="flat")]
        ok = {"answers": {"action": {"choice": "buy", "probabilities": {"buy": 0.7, "hold": 0.3}, "confidence": 0.8}, "which": {"choice": "AMAT", "probabilities": {"AMAT": 0.6, "NVDA": 0.3, "none": 0.1}, "confidence": 0.7}, "confidence": {"score": 3.0, "probabilities": [0.05, 0.1, 0.2, 0.5, 0.15], "confidence": 0.6}, "risk_flag": {"noul": 0.2}}}
        print("buy ok ->", {k: interpret(ok, cands, False)[k] for k in ("status", "action", "symbol", "confidence", "risk_flag", "consistent")})
        bad = json.loads(json.dumps(ok)); bad["answers"]["which"]["choice"] = "TXG"
        print("which off-list ->", {k: interpret(bad, cands, False)[k] for k in ("status", "action", "reason", "consistent")})
        risky = json.loads(json.dumps(ok)); risky["answers"]["risk_flag"]["noul"] = 0.9
        print("risk ->", {k: interpret(risky, cands, False)[k] for k in ("status", "action", "reason", "consistent")})
        print("holding+buy ->", {k: interpret(ok, cands, True)[k] for k in ("status", "action", "reason", "consistent")})
        missing = {"answers": {"action": {"choice": "buy"}}}
        print("missing ->", {k: interpret(missing, cands, False)[k] for k in ("status", "action", "reason")})
        for label, path, val in (("risk NaN", ("risk_flag", "noul"), float("nan")), ("risk -0.2", ("risk_flag", "noul"), -0.2), ("risk True(bool)", ("risk_flag", "noul"), True), ("score 5", ("confidence", "score"), 5), ("score -1", ("confidence", "score"), -1), ("score inf", ("confidence", "score"), float("inf"))):
            b = json.loads(json.dumps(ok)); b["answers"][path[0]][path[1]] = val
            r = interpret(b, cands, False)
            print(f"{label} ->", {k: r[k] for k in ("status", "action", "reason")})
        sys.exit(0)
    state = sys.argv[1] if len(sys.argv) > 1 else "Test state: SPY flat on the day, no position, no candidates."
    out, status, lat = ask(state, {"is_test_ok": {"type": "noul", "instructions": "Is this a well-formed test state?"}})
    print(status, lat, json.dumps(out, ensure_ascii=False)[:800])
