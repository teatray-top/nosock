import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime

WORK = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work")
CORE_PATH = os.path.join(WORK, "dobby_kr_core_v1", "dobby_kr_core_v1_final_d65ce644.py")
CORE_SHA = "d65ce644297726202a02b221fe3c1b0e0ed534fdab68b706daa3e128be926902"
VERSION = "dobby-kr-rule-core-v1"
RULE = "ema_ribbon_8_21_50_200"
LENGTHS = (8, 21, 50, 200)
MIN_ROWS = 400
ORIGIN_DAY = "2002-10-14"
TICK = 5
LIMITATIONS = [
    "rule evidence: one retrospective ledger 2014-12-30..2026-11-05 on latest-vintage adjusted history (work/reference_strategies_v1, gates: parameter plateau 81 neighbours, 1-session delay, 2x tick); not a forward guarantee",
    "15:31 KST completion is a regular-session assumption; holidays, special sessions and provider finality are not verified here",
    "the computed quantity is a file-level calculation under idealized fill assumptions; not an order; no order is placed by this tool",
    "inputs are files supplied by the caller; no data collection, API call or account access",
    "the EMA state is a recursion over the whole stored series from the origin session 2002-10-14; a trailing window of closes does not reproduce it (ribbon_origin_sensitivity.json), and a re-adjusted history (dividend) changes earlier closes and therefore the state",
]


def sha_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_core():
    h = sha_file(CORE_PATH)
    if h != CORE_SHA:
        raise SystemExit(f"approved core hash changed: {h}")
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("dobby_kr_core_frozen", CORE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ema_last(values, n):
    a = 2.0 / (n + 1)
    e = float(values[0])
    for x in values[1:]:
        e = a * float(x) + (1.0 - a) * e
    return e


def ribbon_state(closes, days):
    if len(days) != len(closes):
        raise ValueError(f"days and closes differ in length: {len(days)} vs {len(closes)}")
    if not days or days[0] != ORIGIN_DAY:
        raise ValueError(f"series must start at the origin session {ORIGIN_DAY} that seeds the EMAs; got {days[0] if days else None}")
    if any(days[i] >= days[i + 1] for i in range(len(days) - 1)):
        raise ValueError("days must be strictly increasing")
    if len(closes) < MIN_ROWS:
        raise ValueError(f"need at least {MIN_ROWS} closes for the ribbon state, have {len(closes)}")
    e = {n: ema_last(closes, n) for n in LENGTHS}
    long = e[8] > e[21] > e[50] > e[200]
    bearish = e[8] < e[21] < e[50] < e[200]
    return dict(emas={str(k): v for k, v in e.items()}, weight=1 if long else 0, trend="up" if long else ("down" if bearish else "mixed"), rule=RULE, origin_day=days[0], rows=len(closes), last_day=days[-1], definition="weight 1 iff EMA8 > EMA21 > EMA50 > EMA200 on adjusted closes, each EMA seeded at the adjusted close of the origin session 2002-10-14 (first session of the stored series and of the frozen backtest) with alpha 2/(n+1); otherwise 0; trend label down iff the reverse order holds, else mixed; a trailing window of closes does not reproduce these values")


def decide_rule_from_files(raw_path, adj_path, cash, qty, next_open=None, fetched_at=None, decision_session=None, observed_at=None, now_kst=None):
    core = load_core()
    cash, qty = core.validate_account(cash, qty)
    raw_full = json.load(open(raw_path, encoding="utf-8"))
    adj_full = json.load(open(adj_path, encoding="utf-8"))
    raw = raw_full.get(core.SYM, raw_full)
    adj = adj_full.get(core.SYM, adj_full)
    sources = core.fetch_sources(raw, adj, fetched_at)
    now_kst = now_kst or datetime.now(core.KST)
    if datetime.fromisoformat(sources["effective"]) > now_kst:
        raise ValueError("fetch time is in the future relative to now")
    days_all = core.validate_series(raw, adj, min_rows=MIN_ROWS)
    comp = core.completion(days_all, sources["effective"])
    comp["fetched_at_sources"] = sources["all"]
    comp["effective_source"] = sources["effective_source"]
    comp["per_file"] = sources["per_file"]
    raw, adj, dropped = core.truncate_incomplete(raw, adj, comp)
    days = core.validate_series(raw, adj, min_rows=MIN_ROWS)
    if decision_session is not None and decision_session != days[-1]:
        raise ValueError(f"requested decision session {decision_session} differs from the last completed session {days[-1]}")
    state = ribbon_state([float(c) for c in adj["C"]], list(days))
    rec = dict(version=VERSION, code_sha256=sha_file(os.path.abspath(__file__)), approved_core_sha256=CORE_SHA, symbol=core.SYM, decision_session=days[-1], close=float(raw["C"][-1]), open=float(raw["O"][-1]), rows=len(days), input_source=dict(raw_path=os.path.abspath(raw_path), adj_path=os.path.abspath(adj_path), raw_sha256=sha_file(raw_path), adj_sha256=sha_file(adj_path), fetched_at=comp["fetched_at"], rows_in_file=len(days_all)), bar_completion=comp, dropped_incomplete_rows=dropped, account=dict(cash=cash, qty=qty, held=1 if qty > 0 else 0), rule=state, weight=state["weight"], action="long" if state["weight"] == 1 else "flat", limitations=LIMITATIONS, created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if next_open is not None:
        O = core.validate_price(next_open, "next open")
        obs = core.validate_observed_at(observed_at, days[-1], now_kst)
        q = core.quantity(cash, qty, state["weight"], O + TICK, O - TICK)
        rec["computed_quantity"] = dict(label="computed quantity under idealized fill assumptions; not an order, not a fill", next_open_input=O, price_basis=obs["price_basis"], next_open_observed_at=obs["observed_at"], observed_session=obs["observed_session"], buy_price_assumed=O + TICK, sell_price_assumed=O - TICK, **q)
    return rec


def fixture_days(n):
    from datetime import date, timedelta
    out, d = [], date(2002, 10, 14)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def selftest():
    import math
    import random
    core = load_core()
    rng = random.Random(3)
    closes = [100.0]
    for _ in range(599):
        closes.append(closes[-1] * math.exp(rng.gauss(0.0004, 0.01)))
    D = fixture_days(600)
    st = ribbon_state(closes, D)
    e = {n: ema_last(closes, n) for n in LENGTHS}
    assert st["emas"] == {str(k): v for k, v in e.items()} and st["weight"] in (0, 1) and st["origin_day"] == ORIGIN_DAY and st["rows"] == 600
    up = [100.0 * (1.0 + 0.002) ** t for t in range(500)]
    assert ribbon_state(up, D[:500])["weight"] == 1 and ribbon_state(up, D[:500])["trend"] == "up"
    down = [100.0 * (1.0 - 0.002) ** t for t in range(500)]
    assert ribbon_state(down, D[:500])["weight"] == 0 and ribbon_state(down, D[:500])["trend"] == "down"
    flat = [100.0] * 500
    assert ribbon_state(flat, D[:500])["weight"] == 0 and ribbon_state(flat, D[:500])["trend"] == "mixed"
    for bad_closes, bad_days, why in ((closes[:300], D[:300], "short series"), (closes[-400:], D[-400:], "trailing window not starting at the origin"), (closes, D[:599], "length mismatch"), (closes, [D[0]] + D[2:] + [D[1]], "non-increasing days")):
        try:
            ribbon_state(bad_closes, bad_days)
            raise AssertionError(f"{why} must be rejected")
        except ValueError:
            pass
    stacked = [100.0] * 200 + [100.0 * (1.0 + 0.003) ** t for t in range(400)]
    full_w = ribbon_state(stacked, D)["weight"]
    e400 = {n: ema_last(stacked[-400:], n) for n in LENGTHS}
    w400 = 1 if e400[8] > e400[21] > e400[50] > e400[200] else 0
    assert full_w == w400 == 1
    window_differs = [100.0 + 40.0 * math.sin(t / 7.0) for t in range(200)] + [100.0 + 0.002 * t for t in range(400)]
    eF = {n: ema_last(window_differs, n) for n in LENGTHS}
    eW = {n: ema_last(window_differs[-400:], n) for n in LENGTHS}
    assert abs(eF[200] - eW[200]) > 1e-9 and abs(eF[8] - eW[8]) < 1e-6
    sys.path.insert(0, os.path.join(WORK, "reference_strategies_v1"))
    import reference_strategies as RS
    ref = RS.ema_series(closes, 21)[-1]
    assert abs(ref - e[21]) < 1e-9
    base = r"G:\NoSock\stock\penny\data\kr_etf"
    rec = decide_rule_from_files(os.path.join(base, "kodex_daily_raw.json"), os.path.join(base, "kodex_daily_adj.json"), 0, 0)
    assert rec["decision_session"] == "2026-11-05" and [d["day"] for d in rec["dropped_incomplete_rows"]] == ["2026-11-06"] and rec["weight"] in (0, 1)
    rec_q = decide_rule_from_files(os.path.join(base, "kodex_daily_raw.json"), os.path.join(base, "kodex_daily_adj.json"), 100000, 0, next_open=80000)
    assert rec_q["computed_quantity"]["price_basis"].startswith("assumed") and rec_q["computed_quantity"]["buy_price_assumed"] == 80005
    try:
        decide_rule_from_files(os.path.join(base, "kodex_daily_raw.json"), os.path.join(base, "kodex_daily_adj.json"), -1, 0)
        raise AssertionError("negative cash must be rejected")
    except ValueError:
        pass
    print("selftest ok (rule state checks, EMA parity with reference_strategies, frozen snapshot decision)")
    print(json.dumps(dict(decision_session=rec["decision_session"], close=rec["close"], emas={k: round(v, 2) for k, v in rec["rule"]["emas"].items()}, trend=rec["rule"]["trend"], weight=rec["weight"], action=rec["action"], fetched_at=rec["bar_completion"]["fetched_at"]), ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description="dobby-kr-rule-core-v1: EMA ribbon rule decision and assumed quantity from files, reusing the approved dobby_kr_core checks. LIMITATIONS: " + " | ".join(LIMITATIONS))
    sub = ap.add_subparsers(dest="cmd")
    d = sub.add_parser("decide")
    d.add_argument("--raw", required=True)
    d.add_argument("--adj", required=True)
    d.add_argument("--cash", type=int, required=True)
    d.add_argument("--qty", type=int, required=True)
    d.add_argument("--next-open", type=int)
    d.add_argument("--observed-at")
    d.add_argument("--fetched-at")
    d.add_argument("--decision-session")
    d.add_argument("--out")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "decide":
        rec = decide_rule_from_files(a.raw, a.adj, a.cash, a.qty, a.next_open, fetched_at=a.fetched_at, decision_session=a.decision_session, observed_at=a.observed_at)
        if a.out:
            with open(a.out, "xb") as f:
                f.write(json.dumps(rec, ensure_ascii=False, indent=1).encode("utf-8"))
            rec["record_sha256"] = sha_file(a.out)
        print(json.dumps(rec, ensure_ascii=False, indent=1))
    else:
        ap.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
