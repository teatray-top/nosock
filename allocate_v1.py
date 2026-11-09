import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone

SPEC = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work\allocate_v1_spec_20261106.json")
SPEC_SHA256 = "8d8c5946add95d1b4f1afcda23067beaaa41f824dcac727f5170da748f08bf7d"
STATE_SPEC = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work\state_vol_v1_spec_20261106.json")
OUT_DIR = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work\allocate_v1_20261106")
SCHEMA = "allocate-v1"
MODULE = "allocate_v1"
SYM = "069500"
LAST = "2026-11-05"
DEPOSIT, TICK = 10000.0, 5.0
FEE_NUM, FEE_DEN = 15, 100000
VARIANTS = ("A_const_state", "B_ewma_state")
PARENT_ORDER = ("raw_prices", "state_results", "state_spec", "ladder_results", "allocation_spec")
ANCHORS = {"common": dict(pnl=4779819.0, buys=130, sells=0, max_dd_won=-7920376.0), "bull_2025": dict(pnl=1154595.0, buys=58, sells=0, max_dd_won=-298659.0), "crash_2026": dict(pnl=-124863.0, buys=7, sells=0, max_dd_won=-155087.0)}


class InvalidInput(SystemExit):
    pass


def sha_full(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def fee(notional):
    if not (isinstance(notional, (int, float)) and math.isfinite(notional) and notional >= 0 and float(notional).is_integer()):
        raise InvalidInput(f"non-integer notional {notional}")
    return float(int(notional) * FEE_NUM // FEE_DEN)


def load_spec(path=SPEC):
    h = sha_full(path)
    if h != SPEC_SHA256:
        raise InvalidInput(f"spec hash {h} != {SPEC_SHA256}; refusing")
    s = json.load(open(path, encoding="utf-8"))
    if s["version"] != SCHEMA or s["outputs"]["output_dir"] != OUT_DIR or s["outputs"]["overwrite"] or s["resources"]["cpu_threads"] != 1:
        raise InvalidInput("spec settings differ from module constants")
    parents = {k: dict(path=v.get("path"), sha256=v["sha256"]) for k, v in s["parents"].items()}
    if parents["state_spec"]["path"] is None:
        parents["state_spec"]["path"] = STATE_SPEC
    parents["allocation_spec"] = dict(path=path, sha256=SPEC_SHA256)
    if set(parents) != set(PARENT_ORDER) or any(p["path"] is None for p in parents.values()):
        raise InvalidInput("parent set incomplete")
    periods = {k: tuple(v) for k, v in s["periods"].items() if k != "note"}
    return dict(path=path, sha256=h, parents=parents, periods=periods, primary=s["primary_period"], secondary=list(s["secondary_periods"]), external_timeout_seconds=int(s["resources"]["external_timeout_seconds"]))


def check_parents(spec):
    for name in PARENT_ORDER:
        p = spec["parents"][name]
        h = sha_full(p["path"])
        if h != p["sha256"]:
            raise InvalidInput(f"parent {name} hash changed: {h}")
    return len(PARENT_ORDER)


def validate_prices(days, O, C):
    if days != sorted(days) or len(set(days)) != len(days):
        raise InvalidInput("days not sorted/unique")
    for o, c in zip(O, C):
        if not (math.isfinite(o) and math.isfinite(c) and o > TICK and c > 0 and float(o).is_integer() and float(c).is_integer()):
            raise InvalidInput("invalid price (must be finite, positive, integer won, open > tick)")


def load_prices(spec):
    p = spec["parents"]["raw_prices"]["path"]
    r = json.load(open(p, encoding="utf-8"))[SYM]
    keep = [i for i, d in enumerate(r["days"]) if d <= LAST]
    days = [r["days"][i] for i in keep]
    O = [float(r["O"][i]) for i in keep]
    C = [float(r["C"][i]) for i in keep]
    validate_prices(days, O, C)
    return days, O, C


def state_from_obj(obj):
    fr = obj["forecast_records"]
    h_ref = float(obj["envelope"]["reference"]["h"])
    if not (math.isfinite(h_ref) and h_ref > 0):
        raise InvalidInput("invalid h_ref")
    lookup = {}
    for k, rec in enumerate(fr):
        d, nxt = rec["decision_session"], rec["predicted_for_session"]
        he, hr = float(rec["h_forecast"]["ewma"]), float(rec["h_forecast"]["reference"])
        if d in lookup:
            raise InvalidInput(f"duplicate decision_session {d}")
        if not (math.isfinite(he) and he > 0) or hr != h_ref or not nxt > d:
            raise InvalidInput(f"invalid forecast record {d}")
        lookup[d] = dict(next_session=nxt, h_ewma=he, provenance=f"forecast_records[{k}].h_forecast.ewma")
    return dict(h_ref=h_ref, lookup=lookup, n=len(fr))


def load_state(spec):
    p = spec["parents"]["state_results"]["path"]
    obj = json.load(open(p, encoding="utf-8"))
    st = state_from_obj(dict(forecast_records=obj["forecast_records"], envelope=dict(reference=dict(h=obj["envelope"]["reference"]["h"]))))
    del obj
    return st


def weight(variant, h_ref, h_ewma):
    if variant == "A_const_state":
        return min(1.0, math.sqrt(h_ref / h_ref))
    if variant == "B_ewma_state":
        return min(1.0, math.sqrt(h_ref / h_ewma))
    raise InvalidInput(variant)


def decisions_for(days, idx, state):
    out = []
    for i in idx[:-1]:
        rec = state["lookup"].get(days[i])
        if rec is None:
            raise InvalidInput(f"missing state record for decision session {days[i]}")
        if rec["next_session"] != days[i + 1]:
            raise InvalidInput(f"misaligned state record {days[i]} -> {rec['next_session']} != {days[i + 1]}")
        out.append(dict(weight_decision_session=days[i], next_session=days[i + 1], h_ref=state["h_ref"], h_ewma=rec["h_ewma"], w={v: weight(v, state["h_ref"], rec["h_ewma"]) for v in VARIANTS}, provenance=rec["provenance"]))
    return out


def run_ledger(days, O, C, idx, w_list):
    if len(w_list) != len(idx) - 1:
        raise InvalidInput("weights must be one per decision opportunity")
    cash, qty, deposits = 0.0, 0, 0.0
    buys = sells = 0
    fees = tick_cost = turnover = 0.0
    daily, trades = [], []
    for k, i in enumerate(idx):
        executed_w = None
        if k > 0:
            w = w_list[k - 1]
            if not (0.0 < w <= 1.0):
                raise InvalidInput("weight outside (0,1]")
            executed_w = w
            o = O[i]
            w_open = cash + qty * o
            target = math.floor(w * w_open / o)
            cash_before, qty_before = cash, qty
            side, n, px, f = None, 0, None, 0.0
            if target < qty:
                n, px = qty - target, o - TICK
                notional = n * px
                f = fee(notional)
                cash += notional - f
                qty -= n
                sells += 1
                side = "sell"
            elif target > qty:
                px = o + TICK
                q = min(target - qty, int(cash // px))
                while q > 0 and q * px + fee(q * px) > cash:
                    q -= 1
                if q >= 1:
                    f = fee(q * px)
                    cash -= q * px + f
                    qty += q
                    buys += 1
                    side, n = "buy", q
            if side:
                fees += f
                tick_cost += n * TICK
                turnover += n * o
                eq_change = (cash + qty * o) - (cash_before + qty_before * o)
                if abs(eq_change + n * TICK + f) > 1e-6 or cash < 0 or qty < 0:
                    raise InvalidInput("ledger invariant violated")
                trades.append(dict(weight_decision_session=days[idx[k - 1]], order_decision_session=days[i], execution_session=days[i], open_observed=o, w=w, target_qty=target, side=side, qty=n, price=px, fee=f, cash_before=cash_before, cash_after=cash, qty_before=qty_before, qty_after=qty))
        deposits += DEPOSIT
        cash += DEPOSIT
        value = cash + qty * C[i]
        daily.append(dict(session=days[i], cash=cash, qty=qty, value=value, deposits=deposits, pnl=value - deposits, exposure=qty * C[i] / value if value > 0 else 0.0, executed_w_from_previous_close=executed_w, w_for_next_session=w_list[k] if k < len(idx) - 1 else None))
    pnl = [d["pnl"] for d in daily]
    peak, dd = 0.0, 0.0
    for x in pnl:
        peak = max(peak, x)
        dd = min(dd, x - peak)
    met = dict(start=days[idx[0]], end=days[idx[-1]], sessions=len(idx), decisions=len(w_list), deposits=deposits, pnl_won_after_costs=pnl[-1], pnl_pct_of_deposits=pnl[-1] / deposits, max_dd_won=dd, exposure_avg=sum(d["exposure"] for d in daily) / len(daily), buys=buys, sells=sells, fees_total_won=fees, tick_cost_won=tick_cost, turnover_won=turnover, qty_end=qty, final_cash=cash, final_value=daily[-1]["value"], w_mean=sum(w_list) / len(w_list) if w_list else None, w_min=min(w_list) if w_list else None, sessions_w_below_1=sum(1 for w in w_list if w < 1.0))
    return dict(metrics=met, daily=daily, trades=trades)


def period_index(days, start, end):
    idx = [i for i, d in enumerate(days) if start <= d <= end]
    if not idx:
        raise InvalidInput(f"empty period {start}~{end}")
    return idx


def anchor_check(name, met, ladder):
    if name not in ANCHORS:
        return None
    a = ANCHORS[name]
    lad = ladder["periods"][name]["R1_buyhold"]
    checks = dict(pnl=(met["pnl_won_after_costs"], a["pnl"], lad["pnl"]), buys=(met["buys"], a["buys"], lad["buys"]), sells=(met["sells"], a["sells"], lad["sells"]), max_dd_won=(met["max_dd_won"], a["max_dd_won"], lad["max_dd_won"]), deposits=(met["deposits"], lad["deposits"], lad["deposits"]), qty_end=(met["qty_end"], lad["qty_end"], lad["qty_end"]), final_value=(met["final_value"], lad["final_value"], lad["final_value"]))
    bad = {k: v for k, v in checks.items() if not (v[0] == v[1] == v[2])}
    if bad:
        raise InvalidInput(f"anchor mismatch {name}: {bad}")
    return {k: v[0] for k, v in checks.items()}


def paired(a, b):
    diff = [db["pnl"] - da["pnl"] for da, db in zip(a["daily"], b["daily"])]
    return dict(b_minus_a_terminal_pnl=diff[-1], daily_diff_min=min(diff), daily_diff_max=max(diff), sessions_b_fewer_shares=sum(1 for da, db in zip(a["daily"], b["daily"]) if db["qty"] < da["qty"]))


def summary_lines(results):
    lines = []
    for name, res in results.items():
        a, b = res["variants"]["A_const_state"]["metrics"], res["variants"]["B_ewma_state"]["metrics"]
        p = res["paired"]
        lines.append(f"[{name} {a['start']}~{a['end']} {a['sessions']}세션 입금 {a['deposits']:,.0f}]")
        for tag, m in (("A 고정상태", a), ("B EWMA상태", b)):
            lines.append(f"  {tag}: 손익 {m['pnl_won_after_costs']:+,.0f} ({m['pnl_pct_of_deposits']*100:+.1f}%) 낙폭 {m['max_dd_won']:+,.0f} 노출 {m['exposure_avg']*100:.0f}% 매수 {m['buys']} 매도 {m['sells']} 수수료 {m['fees_total_won']:,.0f} 틱 {m['tick_cost_won']:,.0f} 회전 {m['turnover_won']:,.0f} w평균 {m['w_mean']:.3f} w최소 {m['w_min']:.3f} w<1 {m['sessions_w_below_1']}/{m['decisions']}")
        lines.append(f"  B−A 기말 {p['b_minus_a_terminal_pnl']:+,.0f} | 일별 차이 최소 {p['daily_diff_min']:+,.0f} 최대 {p['daily_diff_max']:+,.0f} | B 보유주 더 적은 세션 {p['sessions_b_fewer_shares']}" + (f" | 앵커 일치 {res['anchor_check']['pnl']:+,.0f}" if res.get("anchor_check") else ""))
    return lines


def build_payload(spec, results, code_sha, now_utc):
    artifact_id = f"{MODULE}_{now_utc.strftime('%Y%m%d_%H%M%S')}_{code_sha[:8]}"
    parents = spec["parents"]
    env = dict(schema_version=SCHEMA, artifact_id=artifact_id, parent_artifact_hashes=[parents[k]["sha256"] for k in PARENT_ORDER], parent_artifact_roles={k: dict(path=parents[k]["path"], sha256=parents[k]["sha256"]) for k in PARENT_ORDER}, code_sha256=code_sha, spec_sha256=spec["sha256"], asof_utc=now_utc.isoformat(timespec="seconds"), exchange_session=dict(exchange="KRX", symbol=SYM, calendar="raw daily session dates", timezone="Asia/Seoul", utc_mapping="session date D in KST (UTC+09:00); no intraday clock times used", weight_marker="previous_session_close", quantity_marker="next_session_open after observing raw open", execution="idealized fill at open +5 won (buy) / -5 won (sell), fee floor(notional x 15 / 100000) won per side on integer won notional, no tax"), symbol=SYM, availability_status="available", decision_at="previous_session_close (weight only; logical marker)", execution_not_before="next_session_open (quantity decided after observing the open; idealized fill, no actual order)", units=dict(cash="krw", value="krw", deposits="krw", pnl="krw", fees_total_won="krw", tick_cost_won="krw", turnover_won="krw", qty="shares", w="dimensionless weight in (0,1]", h="log_return second moment (moment_order 2)"), assumptions="frozen research assumptions: fee 0.015% floored per side (integer arithmetic), +/-5 won tick, zero tax, zero cash yield, no dividends/distributions, whole shares; price-change PnL only", primary_period=spec["primary"], secondary_periods=spec["secondary"], variants=dict(A_const_state="w=min(1,sqrt(h_ref/h_ref))=1 (ladder R1 by construction)", B_ewma_state="w=min(1,sqrt(h_ref/h_ewma_i)), h_ewma_i from state forecast_records by decision_session"), limitation="all periods already viewed; rule chosen after the 2026 crash; exploratory price-ledger comparison; lower exposure mechanically changes market participation; not evidence of predictive alpha or total return", resources=dict(cpu_threads=1, external_timeout_seconds=spec["external_timeout_seconds"], experiments=1, variants=2, periods=len(results), ledgers=2 * len(results), new_fit_optimizations=0, tuned_parameters=0, api_calls=0, orders=0, random_paths=0))
    ref = dict(envelope_artifact_id=artifact_id)
    periods = {}
    for name, res in results.items():
        periods[name] = dict(decision_records=[dict(**d, **ref) for d in res["decisions"]], variants={v: dict(metrics=r["metrics"], trade_records=[dict(**t, **ref) for t in r["trades"]], daily_records=[dict(**d, **ref) for d in r["daily"]]) for v, r in res["variants"].items()}, paired=res["paired"], anchor_check=res.get("anchor_check"), cash_reference=dict(final_value="deposits", pnl=0.0, max_dd_won=0.0, note="R0 identity at zero cash yield; no extra ledger"))
    return dict(envelope=env, lines=summary_lines(results), periods=periods)


def compute(periods, days, O, C, state, ladder):
    results = {}
    for name, (start, end) in periods.items():
        idx = period_index(days, start, end)
        decisions = decisions_for(days, idx, state)
        variants = {v: run_ledger(days, O, C, idx, [d["w"][v] for d in decisions]) for v in VARIANTS}
        results[name] = dict(decisions=decisions, variants=variants, paired=paired(variants["A_const_state"], variants["B_ewma_state"]), anchor_check=anchor_check(name, variants["A_const_state"]["metrics"], ladder) if ladder is not None else None)
    return results


def run(spec_path=SPEC):
    spec = load_spec(spec_path)
    check_parents(spec)
    days, O, C = load_prices(spec)
    state = load_state(spec)
    ladder = json.load(open(spec["parents"]["ladder_results"]["path"], encoding="utf-8"))
    results = compute(spec["periods"], days, O, C, state, ladder)
    check_parents(spec)
    code_sha = sha_full(os.path.abspath(__file__))
    payload = build_payload(spec, results, code_sha, datetime.now(timezone.utc))
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "allocate_v1_results.json")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; refusing to overwrite")
    with open(out, "xb") as f:
        f.write(json.dumps(payload, ensure_ascii=False, indent=1).encode("utf-8"))
    print("\n".join(payload["lines"]))
    print("saved", out, "sha", sha_full(out)[:16], "artifact", payload["envelope"]["artifact_id"], "code", code_sha[:16], "spec", spec["sha256"][:16])
    return payload


def synthetic(n=200):
    days = [f"s{i:04d}" for i in range(n)]
    O = [float(10000 + 5 * ((i * 37) % 97) + 5 * ((i * 11) % 7)) for i in range(n)]
    C = [float(O[i] + 5 * (((i * 53) % 41) - 20)) for i in range(n)]
    validate_prices(days, O, C)
    h_ref = 4e-4
    recs = [dict(decision_session=days[i], predicted_for_session=days[i + 1], h_forecast=dict(reference=h_ref, ewma=h_ref * (0.5 + ((i * 17) % 13) / 6.0))) for i in range(n - 1)]
    return days, O, C, state_from_obj(dict(envelope=dict(reference=dict(h=h_ref)), forecast_records=recs))


def expect_reject(fn, *a):
    try:
        fn(*a)
    except InvalidInput:
        return
    raise AssertionError("expected rejection")


def selftest():
    assert [fee(x) for x in (20000, 40000, 100000, 19999, 6666, 6667, 0, 10010, 25025, 5005, 4995)] == [3.0, 6.0, 15.0, 2.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0]
    expect_reject(fee, 10000.5)
    expect_reject(fee, -5)
    days5 = ["d0", "d1", "d2", "d3", "d4"]
    O5 = [5000.0] * 5
    C5 = [5000.0] * 5
    led = run_ledger(days5, O5, C5, list(range(5)), [1.0, 1.0, 0.5, 1.0])
    m, t, dl = led["metrics"], led["trades"], led["daily"]
    assert [(x["side"], x["qty"], x["price"], x["fee"]) for x in t] == [("buy", 1, 5005.0, 0.0), ("buy", 2, 5005.0, 1.0), ("sell", 1, 4995.0, 0.0), ("buy", 5, 5005.0, 3.0)], t
    assert [(d["cash"], d["qty"], d["value"], d["deposits"], d["pnl"]) for d in dl] == [(10000.0, 0, 10000.0, 10000.0, 0.0), (14995.0, 1, 19995.0, 20000.0, -5.0), (14984.0, 3, 29984.0, 30000.0, -16.0), (29979.0, 2, 39979.0, 40000.0, -21.0), (14951.0, 7, 49951.0, 50000.0, -49.0)], dl
    assert m["buys"] == 3 and m["sells"] == 1 and m["fees_total_won"] == 4.0 and m["tick_cost_won"] == 45.0 and m["turnover_won"] == 45000.0 and m["pnl_won_after_costs"] == -49.0 and m["max_dd_won"] == -49.0
    assert dl[0]["executed_w_from_previous_close"] is None and dl[-1]["w_for_next_session"] is None and dl[1]["executed_w_from_previous_close"] == 1.0 and dl[2]["w_for_next_session"] == 0.5
    assert t[2]["weight_decision_session"] == "d2" and t[2]["order_decision_session"] == "d3" and t[2]["target_qty"] == 2
    one = run_ledger(days5[:1], O5[:1], C5[:1], [0], [])
    assert len(one["daily"]) == 1 and one["trades"] == [] and one["metrics"]["pnl_won_after_costs"] == 0.0 and one["metrics"]["final_value"] == 10000.0 and one["metrics"]["decisions"] == 0
    two = run_ledger(days5[:2], O5[:2], C5[:2], [0, 1], [1.0])
    assert len(two["daily"]) == 2 and len(two["trades"]) == 1 and two["metrics"]["decisions"] == 1 and two["daily"][1]["deposits"] == 20000.0
    fake = dict(envelope=dict(reference=dict(h=4e-4)), forecast_records=[dict(decision_session="d0", predicted_for_session="d1", h_forecast=dict(reference=4e-4, ewma=1e-4)), dict(decision_session="d1", predicted_for_session="d2", h_forecast=dict(reference=4e-4, ewma=1.6e-3)), dict(decision_session="d2", predicted_for_session="d3", h_forecast=dict(reference=4e-4, ewma=4e-4))])
    st = state_from_obj(fake)
    dec = decisions_for(days5[:4], [0, 1, 2, 3], st)
    assert [d["w"]["A_const_state"] for d in dec] == [1.0, 1.0, 1.0] and [d["w"]["B_ewma_state"] for d in dec] == [1.0, 0.5, 1.0] and dec[1]["provenance"] == "forecast_records[1].h_forecast.ewma"
    fr = fake["forecast_records"]
    expect_reject(state_from_obj, dict(fake, forecast_records=fr + [fr[0]]))
    expect_reject(state_from_obj, dict(fake, forecast_records=fr[:1] + [dict(fr[1], h_forecast=dict(reference=4e-4, ewma=0.0))] + fr[2:]))
    expect_reject(state_from_obj, dict(fake, forecast_records=fr[:1] + [dict(fr[1], h_forecast=dict(reference=5e-4, ewma=1e-4))] + fr[2:]))
    expect_reject(decisions_for, ["d0", "d1", "d2", "d3", "d4"], [0, 1, 2, 3, 4], st)
    expect_reject(decisions_for, ["d0", "d1", "dX", "d3"], [0, 1, 2, 3], st)
    expect_reject(validate_prices, ["a", "b"], [10000.0, 10000.5], [10000.0, 10000.0])
    expect_reject(validate_prices, ["a", "b"], [10000.0, 5.0], [10000.0, 10000.0])
    days, O, C, state = synthetic()
    idx = list(range(len(days)))
    dec = decisions_for(days, idx, state)
    wb = [d["w"]["B_ewma_state"] for d in dec]
    assert any(w < 1.0 for w in wb) and any(w == 1.0 for w in wb)
    base = run_ledger(days, O, C, idx, wb)
    assert base["metrics"]["sells"] > 0 and base["metrics"]["buys"] > 0
    k = 80
    O2, C2 = list(O), list(C)
    for i in idx[k + 1:]:
        O2[i] += 500.0
        C2[i] -= 300.0
    pert = run_ledger(days, O2, C2, idx, wb)
    assert pert["daily"][:k + 1] == base["daily"][:k + 1] and [t for t in pert["trades"] if t["execution_session"] <= days[k]] == [t for t in base["trades"] if t["execution_session"] <= days[k]]
    jt = next(t for t in base["trades"] if t["execution_session"] >= days[k])
    j = days.index(jt["execution_session"])
    C3 = list(C)
    C3[j] -= 1000.0
    pert_c = run_ledger(days, O, C3, idx, wb)
    assert [t for t in pert_c["trades"] if t["execution_session"] <= days[j]] == [t for t in base["trades"] if t["execution_session"] <= days[j]] and pert_c["daily"][j]["value"] != base["daily"][j]["value"]
    O4 = list(O)
    O4[j] += 2000.0
    pert_o = run_ledger(days, O4, C, idx, wb)
    assert pert_o["daily"][:j] == base["daily"][:j] and pert_o["daily"][j]["executed_w_from_previous_close"] == base["daily"][j]["executed_w_from_previous_close"] and [t for t in pert_o["trades"] if t["execution_session"] == days[j]] != [t for t in base["trades"] if t["execution_session"] == days[j]]
    st2 = dict(state, lookup={d: (dict(v, h_ewma=v["h_ewma"] * 9.0) if d > days[k] else v) for d, v in state["lookup"].items()})
    dec2 = decisions_for(days, idx, st2)
    assert dec2[:k + 1] == dec[:k + 1] and dec2[k + 1:] != dec[k + 1:]
    for v in VARIANTS:
        led = run_ledger(days, O, C, idx, [d["w"][v] for d in dec])
        assert all(d["cash"] >= 0 and isinstance(d["qty"], int) and d["qty"] >= 0 for d in led["daily"])
        for t in led["trades"]:
            signed = t["qty"] * t["price"] * (1 if t["side"] == "sell" else -1)
            assert abs((t["cash_after"] - t["cash_before"]) - (signed - t["fee"])) < 1e-6 and t["fee"] == fee(t["qty"] * t["price"])
    spec = load_spec()
    assert spec["sha256"] == SPEC_SHA256 and len(SPEC_SHA256) == 64 and spec["primary"] == "common" and set(spec["periods"]) == {"state_eval", "common", "bull_2025", "crash_2026"}
    assert check_parents(spec) == 5 and spec["parents"]["state_spec"]["path"] == STATE_SPEC and spec["parents"]["allocation_spec"]["path"] == SPEC
    syn_results = compute({"syn_all": (days[0], days[-1]), "syn_tail": (days[150], days[-1])}, days, O, C, state, None)
    pay = build_payload(spec, syn_results, "cd" * 32, datetime(2026, 11, 6, 8, 0, 0, tzinfo=timezone.utc))
    env = pay["envelope"]
    for key in ("schema_version", "artifact_id", "parent_artifact_hashes", "code_sha256", "spec_sha256", "asof_utc", "exchange_session", "symbol", "availability_status", "decision_at", "execution_not_before"):
        assert key in env, key
    assert env["artifact_id"] == "allocate_v1_20261106_080000_cdcdcdcd" and len(env["parent_artifact_hashes"]) == 5 and env["parent_artifact_hashes"][-1] == SPEC_SHA256 and env["parent_artifact_hashes"][2] == "5df26ad21d582694f4a425500727260b3d5f1f8bcfdce17d0c9a17b92fd57225"
    assert len(pay["periods"]["syn_all"]["variants"]["B_ewma_state"]["daily_records"]) == 200 and len(pay["periods"]["syn_all"]["decision_records"]) == 199 and pay["periods"]["syn_tail"]["anchor_check"] is None and len(pay["lines"]) == 8
    json.dumps(pay)
    rdays, rO, rC = load_prices(spec)
    rstate = load_state(spec)
    assert len(rdays) == 5938 and rstate["n"] == 2906 and abs(rstate["h_ref"] - 2.1990e-4) < 1e-7
    ladder = json.load(open(spec["parents"]["ladder_results"]["path"], encoding="utf-8"))
    for name in ("common", "bull_2025", "crash_2026"):
        ridx = period_index(rdays, *spec["periods"][name])
        rdec = decisions_for(rdays, ridx, rstate)
        a = run_ledger(rdays, rO, rC, ridx, [1.0] * len(rdec))
        assert anchor_check(name, a["metrics"], ladder) is not None and len(a["daily"]) == len(ridx) == len(set(d["session"] for d in a["daily"])) and a["daily"][0]["session"] >= spec["periods"][name][0] and a["daily"][-1]["session"] <= spec["periods"][name][1]
    assert check_parents(spec) == 5
    print("selftest ok", dict(code_sha256_16=sha_full(os.path.abspath(__file__))[:16], here=os.path.dirname(os.path.abspath(__file__)), spec_sha256_16=spec["sha256"][:16], real_data_ledgers="A anchors only (common, bull_2025, crash_2026)", b_ledgers_on_real_data=0))


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    elif not sys.argv[1:]:
        run()
    else:
        print("usage: allocate_v1.py | selftest")
