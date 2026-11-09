import hashlib
import json
import math
import os
import sys
from datetime import date, datetime, timedelta, timezone

SPEC = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work\allocate_h20_v1_spec_20261106.json")
SPEC_SHA256 = "c29b64d5890482264b3f24f0c60536643e212f3c868e5f248ccf75bae4e6cdc9"
OUT_DIR = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work\allocate_h20_v1_20261106")
SCHEMA = "allocate-h20-v1"
MODULE = "allocate_h20_v1"
SYM = "069500"
CUTOFF = "2026-11-05"
N_SESSIONS = 5938
GRID_ORIGIN = 3031
GRID_STEP = 20
N_GRID = 146
PERIOD_START = 3031
PERIOD_END = 5937
DEPOSIT = 10000
TICK = 5
FEE_NUM, FEE_DEN = 15, 100000
VARIANTS = ("A_full_grid20", "B_ratio_grid20")
PARENT_ORDER = ("raw_prices", "predict_h20_v3_results", "allocate_v1_spec", "allocate_v1_reviewed_code", "allocation_h20_spec")
ANCHOR_SCHEMA = "predict-h20-v3"
LEDGER_CALLS = [0]


class InvalidInput(SystemExit):
    pass


def sha_full(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_spec(path=SPEC):
    h = sha_full(path)
    if h != SPEC_SHA256:
        raise InvalidInput(f"spec hash {h} != {SPEC_SHA256}; refusing")
    s = json.load(open(path, encoding="utf-8"))
    if s["version"] != SCHEMA or s["outputs"]["output_dir"] != OUT_DIR or s["outputs"]["overwrite"] or s["resources"]["cpu_threads"] != 1 or list(s["parent_order"]) != list(PARENT_ORDER):
        raise InvalidInput("spec settings differ from module constants")
    if s["scope_decisions"]["clip_bounds"] != [0, 1] or list(s["scope_decisions"]["variants"]) != list(VARIANTS):
        raise InvalidInput("scope decisions differ from module constants")
    parents = {k: dict(path=s["parents"][k]["path"], sha256=s["parents"][k]["sha256"]) for k in ("raw_prices", "predict_h20_v3_results", "allocate_v1_spec", "allocate_v1_reviewed_code")}
    parents["allocation_h20_spec"] = dict(path=path, sha256=SPEC_SHA256)
    return dict(path=path, sha256=h, parents=parents, anchor_artifact_id=s["anchor_artifact_id"], counts=s["outputs"]["counts"], external_timeout_seconds=int(s["resources"]["external_timeout_seconds"]))


def check_parents(spec):
    for name in PARENT_ORDER:
        p = spec["parents"][name]
        if not os.path.exists(p["path"]):
            raise InvalidInput(f"parent {name} missing: {p['path']}")
        h = sha_full(p["path"])
        if h != p["sha256"]:
            raise InvalidInput(f"parent {name} hash changed: {h}")
    return len(PARENT_ORDER)


def validate_prices(days, O, C):
    if not (len(days) == len(O) == len(C)):
        raise InvalidInput("days/O/C length mismatch")
    if days != sorted(days) or len(set(days)) != len(days):
        raise InvalidInput("sessions not sorted/unique")
    out_O, out_C = [], []
    for o, c in zip(O, C):
        fo, fc = float(o), float(c)
        if not (math.isfinite(fo) and math.isfinite(fc) and fo.is_integer() and fc.is_integer() and fo > TICK and fc > 0):
            raise InvalidInput("raw prices must be finite integer won with O > tick and C > 0")
        out_O.append(int(fo))
        out_C.append(int(fc))
    return out_O, out_C


def load_prices(spec):
    raw = json.load(open(spec["parents"]["raw_prices"]["path"], encoding="utf-8"))[SYM]
    days_all, O_all, C_all = raw["days"], raw["O"], raw["C"]
    if not (len(days_all) == len(O_all) == len(C_all)):
        raise InvalidInput("raw full arrays misaligned")
    keep = [k for k, d in enumerate(days_all) if d <= CUTOFF]
    days = [days_all[k] for k in keep]
    O, C = validate_prices(days, [O_all[k] for k in keep], [C_all[k] for k in keep])
    if len(days) != N_SESSIONS or days[0] != "2002-10-14" or days[-1] != CUTOFF:
        raise InvalidInput("raw calendar differs from the fixed facts")
    return days, O, C


def anchor_from_obj(obj, days, expected_sha=None, expected_artifact=None, raw_sha=None):
    if expected_sha is not None and obj.get("_sha256") != expected_sha:
        raise InvalidInput("anchor hash mismatch")
    env = obj["envelope"]
    if env.get("schema_version") != ANCHOR_SCHEMA or env.get("symbol") != SYM or env.get("availability_status") != "available":
        raise InvalidInput("anchor schema/symbol/availability mismatch")
    if expected_artifact is not None and env.get("artifact_id") != expected_artifact:
        raise InvalidInput("anchor artifact id mismatch")
    if raw_sha is not None and env["parent_artifact_roles"]["raw_prices"]["sha256"] != raw_sha:
        raise InvalidInput("anchor raw parent hash mismatch")
    recs = obj["prediction_records"]
    if len(recs) != N_GRID:
        raise InvalidInput(f"anchor has {len(recs)} records, expected {N_GRID}")
    seen = set()
    out = []
    for k, r in enumerate(recs):
        idx = GRID_ORIGIN + GRID_STEP * k
        if r["block_k"] != k or r["decision_index"] != idx or r["decision_session"] != days[idx] or r["entry_session"] != days[idx + 1]:
            raise InvalidInput(f"anchor record {k} misaligned (block/index/session/entry)")
        if r["decision_session"] in seen:
            raise InvalidInput("duplicate anchor decision")
        seen.add(r["decision_session"])
        if r.get("envelope_artifact_id") != env["artifact_id"] or r.get("value_unit") != "simple_return" or r.get("moment_order") != 1 or r.get("forecast_kind") != "conditional_mean" or r.get("symbol") != SYM:
            raise InvalidInput(f"anchor record {k} metadata mismatch")
        if r["train_max_label_available_session"] is not None and r["train_max_label_available_session"] > r["decision_session"]:
            raise InvalidInput(f"anchor record {k} uses a label after its decision")
        H3 = float(r["mu_hat_H3"])
        H1 = float(r["anchor_mu_loaded"]["H1_expanding_mean"])
        if not (math.isfinite(H3) and math.isfinite(H1)) or H1 <= 0:
            raise InvalidInput(f"anchor record {k} invalid H3/H1 (finite, H1 > 0 required)")
        out.append(dict(k=k, decision_index=idx, decision_session=r["decision_session"], entry_session=r["entry_session"], H3=H3, H1=H1))
    return dict(artifact_id=env["artifact_id"], records=out)


def load_anchor(spec, days):
    p = spec["parents"]["predict_h20_v3_results"]
    h = sha_full(p["path"])
    if h != p["sha256"]:
        raise InvalidInput("anchor hash changed")
    obj = json.load(open(p["path"], encoding="utf-8"))
    obj.pop("outcome_records", None)
    obj["_sha256"] = h
    return anchor_from_obj(obj, days, expected_sha=h, expected_artifact=spec["anchor_artifact_id"], raw_sha=spec["parents"]["raw_prices"]["sha256"])


def weights_from_anchor(anchor):
    decisions = []
    for r in anchor["records"]:
        ratio = r["H3"] / r["H1"]
        if not math.isfinite(ratio):
            raise InvalidInput(f"ratio non-finite at {r['decision_session']}")
        wB = max(0.0, min(1.0, ratio))
        if not (0.0 <= wB <= 1.0):
            raise InvalidInput("clipped weight outside [0,1]")
        decisions.append(dict(decision_index=r["decision_index"], block_k=r["k"], decision_session=r["decision_session"], entry_session=r["entry_session"], H3=r["H3"], H1=r["H1"], ratio=ratio, w_A=1.0, w_B=wB, source_artifact_id=anchor["artifact_id"], input_record_index=r["k"]))
    return decisions


def fee(notional):
    if not isinstance(notional, int) or notional < 0:
        raise InvalidInput("notional must be a non-negative integer")
    return notional * FEE_NUM // FEE_DEN


def ledger(days, O, C, start, end, weights):
    LEDGER_CALLS[0] += 1
    if not (0 <= start <= end < len(days)):
        raise InvalidInput("period out of range")
    for i, w in weights.items():
        if not (start <= i <= end) or not (isinstance(w, float) and math.isfinite(w) and 0.0 <= w <= 1.0):
            raise InvalidInput("weights must be finite floats in [0,1] at in-period grid closes")
    cash, qty, deposits = 0, 0, 0
    pending = None
    daily, execs, trades = [], [], []
    buys = sells = 0
    fees = tick_cost = turnover = 0
    for t in range(start, end + 1):
        executed_w = None
        if pending is not None:
            w, dec_i = pending
            o = O[t]
            W_open = cash + qty * o
            target = math.floor(w * W_open / o)
            cash_b, qty_b = cash, qty
            side, n, px, f = None, 0, None, 0
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
                q = min(target - qty, cash // px)
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
                if (cash + qty * o) - (cash_b + qty_b * o) != -(n * TICK + f) or cash < 0 or qty < 0:
                    raise InvalidInput("ledger invariant violated")
                trades.append(dict(weight_decision_session=days[dec_i], execution_session=days[t], side=side, qty=n, price=px, fee=f, cash_before=cash_b, cash_after=cash, qty_before=qty_b, qty_after=qty))
            execs.append(dict(weight_decision_session=days[dec_i], order_decision_session=days[t], execution_session=days[t], observed_open=o, w=w, W_open=W_open, target_qty=target, side=side, realized_qty=n, cash_before=cash_b, cash_after=cash, qty_before=qty_b, qty_after=qty, fee=f, tick=n * TICK))
            executed_w = w
            pending = None
        deposits += DEPOSIT
        cash += DEPOSIT
        value = cash + qty * C[t]
        w_next = None
        if t in weights and t + 1 <= end:
            w_next = weights[t]
            pending = (w_next, t)
        daily.append(dict(session=days[t], cash=cash, qty=qty, value=value, deposits=deposits, pnl=value - deposits, exposure=(qty * C[t] / value) if value > 0 else 0.0, executed_w_from_previous_close=executed_w, w_for_next_session=w_next))
    pnl = [d["pnl"] for d in daily]
    peak, dd = 0, 0
    for x in pnl:
        peak = max(peak, x)
        dd = min(dd, x - peak)
    ws = [weights[i] for i in sorted(weights)]
    met = dict(start=days[start], end=days[end], sessions=len(daily), unique_daily_rows=len({d["session"] for d in daily}), deposit_events=len(daily), deposits=deposits, pnl_won_after_costs=pnl[-1], pnl_pct_of_deposits=pnl[-1] / deposits if deposits else None, max_dd_won=dd, exposure_avg=sum(d["exposure"] for d in daily) / len(daily), buys=buys, sells=sells, trades=len(trades), execution_opportunities=len(execs), fees_total_won=fees, tick_cost_won=tick_cost, turnover_won=turnover, qty_end=qty, final_cash=cash, final_value=daily[-1]["value"], w_mean=sum(ws) / len(ws) if ws else None, w_min=min(ws) if ws else None, w_max=max(ws) if ws else None, decisions_w_below_1=sum(1 for w in ws if w < 1.0), decisions_w_zero=sum(1 for w in ws if w == 0.0), weights_count=len(ws))
    return dict(metrics=met, daily=daily, executions=execs, trades=trades)


def yearly_increments(daily):
    years = sorted({d["session"][:4] for d in daily})
    last_by_year = {}
    for d in daily:
        last_by_year[d["session"][:4]] = d["pnl"]
    out, prev = {}, 0
    for y in years:
        out[y] = last_by_year[y] - prev
        prev = last_by_year[y]
    return out


def paired(a, b):
    diff = [db["pnl"] - da["pnl"] for da, db in zip(a["daily"], b["daily"])]
    ya, yb = yearly_increments(a["daily"]), yearly_increments(b["daily"])
    return dict(b_minus_a_terminal_pnl=diff[-1], daily_diff_min=min(diff), daily_diff_max=max(diff), sessions_b_fewer_shares=sum(1 for da, db in zip(a["daily"], b["daily"]) if db["qty"] < da["qty"]), yearly_increments=dict(A=ya, B=yb, B_minus_A={y: yb[y] - ya[y] for y in ya}))


def compute(days, O, C, decisions, start=PERIOD_START, end=PERIOD_END):
    wA = {d["decision_index"]: d["w_A"] for d in decisions}
    wB = {d["decision_index"]: d["w_B"] for d in decisions}
    A = ledger(days, O, C, start, end, wA)
    B = ledger(days, O, C, start, end, wB)
    return dict(decisions=decisions, variants={"A_full_grid20": A, "B_ratio_grid20": B}, paired=paired(A, B))


def summary_lines(res):
    a, b = res["variants"]["A_full_grid20"]["metrics"], res["variants"]["B_ratio_grid20"]["metrics"]
    p = res["paired"]
    lines = [f"[{a['start']}~{a['end']} {a['sessions']}세션 입금 {a['deposits']:,} 재조정 기회 {a['execution_opportunities']} 원장 호출 {LEDGER_CALLS[0]}]"]
    for tag, m in (("A 격자 전액", a), ("B 격자 비율", b)):
        lines.append(f"  {tag}: 손익 {m['pnl_won_after_costs']:+,} ({m['pnl_pct_of_deposits']*100:+.1f}%) 낙폭 {m['max_dd_won']:+,} 노출 {m['exposure_avg']*100:.0f}% 매수 {m['buys']} 매도 {m['sells']} 수수료 {m['fees_total_won']:,} 틱 {m['tick_cost_won']:,} 회전 {m['turnover_won']:,} 기말 현금 {m['final_cash']:,} 수량 {m['qty_end']} 가치 {m['final_value']:,} w 평균 {m['w_mean']:.3f} 최소 {m['w_min']:.3f} 최대 {m['w_max']:.3f} w<1 {m['decisions_w_below_1']} w=0 {m['decisions_w_zero']}")
    lines.append(f"  B−A 기말 {p['b_minus_a_terminal_pnl']:+,} | 일별 차이 최소 {p['daily_diff_min']:+,} 최대 {p['daily_diff_max']:+,} | B 보유주 더 적은 세션 {p['sessions_b_fewer_shares']}")
    lines.append("  연도별 증분 B−A: " + ", ".join(f"{y} {v:+,}" for y, v in p["yearly_increments"]["B_minus_A"].items()))
    lines.append("  연도별 증분 A: " + ", ".join(f"{y} {v:+,}" for y, v in p["yearly_increments"]["A"].items()))
    lines.append("  연도별 증분 B: " + ", ".join(f"{y} {v:+,}" for y, v in p["yearly_increments"]["B"].items()))
    return lines


def build_payload(spec, res, anchor, code_sha, now_utc):
    artifact_id = f"{MODULE}_{now_utc.strftime('%Y%m%d_%H%M%S')}_{code_sha[:8]}"
    p = spec["parents"]
    env = dict(schema_version=SCHEMA, artifact_id=artifact_id, parent_artifact_hashes=[p[k]["sha256"] for k in PARENT_ORDER], parent_artifact_roles={k: dict(path=p[k]["path"], sha256=p[k]["sha256"]) for k in PARENT_ORDER}, source_forecast_artifact_id=anchor["artifact_id"], code_sha256=code_sha, spec_sha256=spec["sha256"], asof_utc=now_utc.isoformat(timespec="seconds"), exchange_session=dict(exchange="KRX", symbol=SYM, calendar="raw daily session dates", timezone="Asia/Seoul", utc_mapping="session date D in KST (UTC+09:00); no intraday clock times used", weight_marker="grid_session_close", quantity_marker="next_session_open after observing raw open", execution="idealized fill at open +5 won (buy) / -5 won (sell), fee (notional x 15) // 100000 won per side on integer won notional, no tax"), symbol=SYM, availability_status="available", decision_at="grid_session_close (weight only; logical marker)", execution_not_before="next_session_open (quantity decided after observing the open; idealized fill, no actual order)", units=dict(cash="krw (integer)", value="krw (integer)", deposits="krw (integer)", pnl="krw (integer)", fees_total_won="krw", tick_cost_won="krw", turnover_won="krw", qty="shares (integer)", w="dimensionless weight in [0,1]", ratio="dimensionless (H3/H1, both simple_return 20-session means)", pnl_pct_of_deposits="fraction PnL/cumulative deposits, not annualized"), variants=dict(A_full_grid20="w = 1 at every grid close; deposits idle between grid opens (not the daily R1)", B_ratio_grid20="w = clip(H3/H1, 0, 1) from stored predict-h20-v3 forecasts; no daily h, no outcomes"), grid=dict(origin_decision_index=GRID_ORIGIN, step_sessions=GRID_STEP, decisions=N_GRID, period=[PERIOD_START, PERIOD_END]), assumptions="frozen research costs: fee 0.015% floored per side, +/-5 won tick, zero tax, zero cash yield, no dividends/distributions, whole shares; price-change PnL only; two-stage close-weight/open-quantity with idealized fills; no volume/queue/latency model", interpretation="exploratory ledger comparison; the weight ratio was proposed after viewing H3 results; no independent holdout remains; a lower drawdown alone does not establish higher profit; H3 and H1 have different training support so this does not isolate a volatility feature causally; not an optimal allocation or a live edge", counts=dict(forecasts_loaded=len(res["decisions"]), ratio_evaluations=len(res["decisions"]), weights_per_variant=len(res["decisions"]), rebalance_opportunities_per_variant={v: r["metrics"]["execution_opportunities"] for v, r in res["variants"].items()}, realized_trades_per_variant={v: r["metrics"]["trades"] for v, r in res["variants"].items()}, daily_rows_per_variant={v: r["metrics"]["unique_daily_rows"] for v, r in res["variants"].items()}, deposit_events_per_variant={v: r["metrics"]["deposit_events"] for v, r in res["variants"].items()}, deposit_won_per_variant={v: r["metrics"]["deposits"] for v, r in res["variants"].items()}, ledger_calls=LEDGER_CALLS[0], optimizer_calls=0, predictor_recomputations=0), resources=dict(cpu_threads=1, external_timeout_seconds=spec["external_timeout_seconds"], experiments=1, ledgers=2, optimizer_calls=0, tuned_parameters=0, api_calls=0, orders=0, random_paths=0))
    ref = dict(envelope_artifact_id=artifact_id, source_forecast_artifact_id=anchor["artifact_id"])
    return dict(envelope=env, lines=summary_lines(res), decision_records=[dict(d, **ref) for d in res["decisions"]], variants={v: dict(metrics=r["metrics"], execution_records=[dict(e, **ref) for e in r["executions"]], trade_records=[dict(t, **ref) for t in r["trades"]], daily_records=[dict(d, **ref) for d in r["daily"]]) for v, r in res["variants"].items()}, paired=res["paired"], cash_reference=dict(final_value="deposits", pnl=0, max_dd_won=0, note="analytical identity at zero cash yield; no third ledger"))


def run(spec_path=SPEC):
    spec = load_spec(spec_path)
    check_parents(spec)
    days, O, C = load_prices(spec)
    anchor = load_anchor(spec, days)
    decisions = weights_from_anchor(anchor)
    LEDGER_CALLS[0] = 0
    res = compute(days, O, C, decisions)
    exp = spec["counts"]
    for v, r in res["variants"].items():
        m = r["metrics"]
        if m["execution_opportunities"] != exp["expected_rebalance_opportunities_per_variant"] or m["unique_daily_rows"] != exp["expected_daily_rows_per_variant"] or m["deposit_events"] != exp["expected_deposit_events_per_variant"] or m["deposits"] != exp["expected_deposit_won_per_variant"] or m["weights_count"] != exp["expected_weights_per_variant"]:
            raise InvalidInput(f"counts differ from spec for {v}: {m}")
    if len(decisions) != exp["expected_forecasts_loaded"] or LEDGER_CALLS[0] != exp["expected_ledger_calls"]:
        raise InvalidInput("decision/ledger counts differ from spec")
    check_parents(spec)
    code_sha = sha_full(os.path.abspath(__file__))
    payload = build_payload(spec, res, anchor, code_sha, datetime.now(timezone.utc))
    data = json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False).encode("utf-8")
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "allocate_h20_v1_results.json")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; refusing to overwrite")
    with open(out, "xb") as f:
        f.write(data)
    print("\n".join(payload["lines"]))
    print("saved", out, "sha", sha_full(out)[:16], "artifact", payload["envelope"]["artifact_id"], "code", code_sha[:16], "spec", spec["sha256"][:16])
    return payload


def synthetic_days(n, start=date(2015, 1, 1)):
    return [(start + timedelta(days=k)).isoformat() for k in range(n)]


def synthetic_anchor(days, H3s, H1s, artifact_id="predict_h20_v3_synthetic", raw_sha="rawsha"):
    recs = []
    for k in range(N_GRID):
        idx = GRID_ORIGIN + GRID_STEP * k
        recs.append(dict(symbol=SYM, block_k=k, decision_index=idx, decision_session=days[idx], entry_session=days[idx + 1], train_max_label_available_session=days[idx], mu_hat_H3=H3s[k], anchor_mu_loaded={"H1_expanding_mean": H1s[k]}, envelope_artifact_id=artifact_id, value_unit="simple_return", moment_order=1, forecast_kind="conditional_mean"))
    return dict(envelope=dict(schema_version=ANCHOR_SCHEMA, artifact_id=artifact_id, symbol=SYM, availability_status="available", parent_artifact_roles=dict(raw_prices=dict(sha256=raw_sha))), prediction_records=recs)


def expect_reject(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except InvalidInput:
        return
    raise AssertionError("expected rejection")


def selftest():
    days5 = synthetic_days(5)
    O5, C5 = [5000] * 5, [5000] * 5
    led = ledger(days5, O5, C5, 0, 4, {0: 1.0, 1: 1.0, 2: 0.5, 3: 1.0})
    m, t, dl = led["metrics"], led["trades"], led["daily"]
    assert [(x["side"], x["qty"], x["price"], x["fee"]) for x in t] == [("buy", 1, 5005, 0), ("buy", 2, 5005, 1), ("sell", 1, 4995, 0), ("buy", 5, 5005, 3)], t
    assert [(d["cash"], d["qty"], d["value"], d["deposits"], d["pnl"]) for d in dl] == [(10000, 0, 10000, 10000, 0), (14995, 1, 19995, 20000, -5), (14984, 3, 29984, 30000, -16), (29979, 2, 39979, 40000, -21), (14951, 7, 49951, 50000, -49)], dl
    assert m["final_cash"] == 14951 and m["qty_end"] == 7 and m["final_value"] == 49951 and m["deposits"] == 50000 and m["pnl_won_after_costs"] == -49 and m["fees_total_won"] == 4 and m["tick_cost_won"] == 45 and m["turnover_won"] == 45000 and m["max_dd_won"] == -49 and m["execution_opportunities"] == 4
    assert dl[-1]["executed_w_from_previous_close"] == 1.0 and dl[-1]["w_for_next_session"] is None and all(isinstance(d["cash"], int) and isinstance(d["qty"], int) for d in dl)
    n = 40
    days = synthetic_days(n)
    O = [5000 + 5 * ((k * 37) % 23) for k in range(n)]
    C = [O[k] + 5 * (((k * 11) % 7) - 3) for k in range(n)]
    gw = {i: 1.0 for i in range(6, n - 1, 3)}
    sp = ledger(days, O, C, 0, n - 1, gw)
    ex_sessions = [e["execution_session"] for e in sp["executions"]]
    assert ex_sessions == [days[i + 1] for i in sorted(gw)] and sp["metrics"]["execution_opportunities"] == len(gw)
    for d in sp["daily"]:
        i = days.index(d["session"])
        assert (d["w_for_next_session"] is not None) == (i in gw and i + 1 <= n - 1) and (d["executed_w_from_previous_close"] is not None) == (i - 1 in gw)
    first = sp["executions"][0]
    assert first["cash_before"] == 7 * DEPOSIT and first["W_open"] == 7 * DEPOSIT and first["observed_open"] == O[7]
    before_second = [d for d in sp["daily"] if d["session"] == days[9]][0]
    second = sp["executions"][1]
    assert second["cash_before"] == before_second["cash"] and second["cash_before"] == first["cash_after"] + 3 * DEPOSIT
    one = ledger(days[:1], O[:1], C[:1], 0, 0, {})
    assert one["executions"] == [] and one["metrics"]["final_value"] == DEPOSIT and one["metrics"]["pnl_won_after_costs"] == 0 and one["metrics"]["unique_daily_rows"] == 1
    two = ledger(days[:2], O[:2], C[:2], 0, 1, {0: 1.0})
    assert len(two["executions"]) == 1 and two["executions"][0]["execution_session"] == days[1] and two["daily"][1]["executed_w_from_previous_close"] == 1.0 and two["daily"][1]["w_for_next_session"] is None and two["executions"][0]["realized_qty"] == DEPOSIT // (O[1] + TICK)
    expect_reject(ledger, days[:2], O[:2], C[:2], 0, 1, {1: 1.0, 0: 1.0} if False else {5: 1.0})
    expect_reject(ledger, days[:2], O[:2], C[:2], 0, 1, {0: 1.5})
    expect_reject(ledger, days[:2], O[:2], C[:2], 0, 1, {0: 1})
    sdays = synthetic_days(N_SESSIONS)
    sO = [20000 + 5 * ((k * 37) % 401) for k in range(N_SESSIONS)]
    sC = [sO[k] + 5 * (((k * 53) % 41) - 20) for k in range(N_SESSIONS)]
    sO, sC = validate_prices(sdays, sO, sC)
    ratios = [-0.5, 0.0, 0.4, 1.0, 1.7] * 30
    H1s = [0.01] * N_GRID
    H3s = [r * 0.01 for r in ratios[:N_GRID]]
    anc = anchor_from_obj(synthetic_anchor(sdays, H3s, H1s), sdays, expected_artifact="predict_h20_v3_synthetic", raw_sha="rawsha")
    dec = weights_from_anchor(anc)
    assert [d["w_B"] for d in dec[:5]] == [0.0, 0.0, 0.4, 1.0, 1.0] and all(d["w_A"] == 1.0 for d in dec) and dec[2]["ratio"] == 0.4 / 1.0 * 1.0 if False else abs(dec[2]["ratio"] - 0.4) < 1e-15
    LEDGER_CALLS[0] = 0
    res = compute(sdays, sO, sC, dec)
    assert LEDGER_CALLS[0] == 2
    A, B = res["variants"]["A_full_grid20"], res["variants"]["B_ratio_grid20"]
    assert A["metrics"]["execution_opportunities"] == N_GRID == B["metrics"]["execution_opportunities"] and A["metrics"]["unique_daily_rows"] == PERIOD_END - PERIOD_START + 1 == 2907 and A["metrics"]["deposits"] == 2907 * DEPOSIT
    e_zero = [e for e in B["executions"] if e["w"] == 0.0]
    assert e_zero and all(e["target_qty"] == 0 and e["qty_after"] == 0 for e in e_zero) and any(e["side"] == "sell" for e in e_zero)
    e_part = [e for e in B["executions"] if e["w"] == 0.4]
    assert e_part and all(e["target_qty"] == math.floor(0.4 * e["W_open"] / e["observed_open"]) for e in e_part)
    noop = [e for e in A["executions"] if e["side"] is None]
    assert len(noop) + A["metrics"]["trades"] == N_GRID
    assert all(e["execution_session"] == sdays[GRID_ORIGIN + GRID_STEP * k + 1] for k, e in enumerate(A["executions"]))
    for v in (A, B):
        for tr in v["trades"]:
            signed = tr["qty"] * tr["price"] * (1 if tr["side"] == "sell" else -1)
            assert tr["cash_after"] - tr["cash_before"] == signed - tr["fee"] and tr["cash_after"] >= 0
        assert all(d["cash"] >= 0 and isinstance(d["qty"], int) and d["qty"] >= 0 for d in v["daily"]) and len({d["session"] for d in v["daily"]}) == len(v["daily"])
    p = res["paired"]
    ya, yb = p["yearly_increments"]["A"], p["yearly_increments"]["B"]
    assert sum(ya.values()) == A["metrics"]["pnl_won_after_costs"] and sum(yb.values()) == B["metrics"]["pnl_won_after_costs"] and sum(p["yearly_increments"]["B_minus_A"].values()) == p["b_minus_a_terminal_pnl"]
    for bad_H3, bad_H1 in ((H3s, [0.0] + H1s[1:]), (H3s, [-0.01] + H1s[1:]), (H3s, [float("nan")] + H1s[1:]), ([float("inf")] + H3s[1:], H1s), ([1e308] + H3s[1:], [1e-308] + H1s[1:])):
        calls = LEDGER_CALLS[0]
        try:
            a2 = anchor_from_obj(synthetic_anchor(sdays, bad_H3, bad_H1), sdays)
            compute(sdays, sO, sC, weights_from_anchor(a2))
            raise AssertionError("expected rejection")
        except InvalidInput:
            assert LEDGER_CALLS[0] == calls
    expect_reject(validate_prices, ["a", "b"], [100, 100], [100])
    expect_reject(validate_prices, ["b", "a"], [100, 100], [100, 100])
    expect_reject(validate_prices, ["a", "a"], [100, 100], [100, 100])
    expect_reject(validate_prices, ["a", "b"], [100.5, 100], [100, 100])
    expect_reject(validate_prices, ["a", "b"], [5, 100], [100, 100])
    expect_reject(validate_prices, ["a", "b"], [100, 100], [0, 100])
    for mutate in (lambda a: a["prediction_records"].pop(), lambda a: a["prediction_records"].append(dict(a["prediction_records"][0])), lambda a: a["prediction_records"][3].__setitem__("decision_index", 3032), lambda a: a["prediction_records"][3].__setitem__("decision_session", sdays[0]), lambda a: a["prediction_records"][3].__setitem__("entry_session", sdays[0]), lambda a: a["prediction_records"][3].__setitem__("block_k", 99), lambda a: a["envelope"].__setitem__("artifact_id", "other"), lambda a: a["envelope"].__setitem__("symbol", "000000"), lambda a: a["prediction_records"][2].__setitem__("value_unit", "log_return"), lambda a: a["prediction_records"][2].__setitem__("moment_order", 2), lambda a: a["prediction_records"][2].__setitem__("forecast_kind", "other"), lambda a: a["envelope"].__setitem__("availability_status", "unsupported_history"), lambda a: a["prediction_records"][2].__setitem__("train_max_label_available_session", sdays[-1]), lambda a: a["envelope"]["parent_artifact_roles"]["raw_prices"].__setitem__("sha256", "x")):
        obj = synthetic_anchor(sdays, H3s, H1s)
        mutate(obj)
        expect_reject(anchor_from_obj, obj, sdays, expected_artifact="predict_h20_v3_synthetic", raw_sha="rawsha")
    obj = synthetic_anchor(sdays, H3s, H1s)
    obj["_sha256"] = "00" * 32
    expect_reject(anchor_from_obj, obj, sdays, expected_sha="11" * 32)

    def state(r):
        return [(d["session"], d["cash"], d["qty"], d["value"]) for d in r["daily"]]

    k = GRID_ORIGIN + 20 * 30 + 1
    O_f, C_f = list(sO), list(sC)
    for t in range(k + 1, N_SESSIONS):
        O_f[t] += 500
        C_f[t] += 500
    pf = compute(sdays, O_f, C_f, dec)
    kpos = k - PERIOD_START
    assert state(pf["variants"]["B_ratio_grid20"])[:kpos + 1] == state(B)[:kpos + 1] and [tr for tr in pf["variants"]["B_ratio_grid20"]["trades"] if tr["execution_session"] <= sdays[k]] == [tr for tr in B["trades"] if tr["execution_session"] <= sdays[k]]
    C_k = list(sC)
    C_k[k] += 1000
    pc = compute(sdays, sO, C_k, dec)
    assert [e for e in pc["variants"]["B_ratio_grid20"]["executions"] if e["execution_session"] == sdays[k]] == [e for e in B["executions"] if e["execution_session"] == sdays[k]]
    O_k = list(sO)
    O_k[k] += 500
    po = compute(sdays, O_k, sC, dec)
    eo, eb = [e for e in po["variants"]["B_ratio_grid20"]["executions"] if e["execution_session"] == sdays[k]][0], [e for e in B["executions"] if e["execution_session"] == sdays[k]][0]
    assert eo["w"] == eb["w"] and eo["observed_open"] != eb["observed_open"] and po["decisions"] == res["decisions"]
    m_idx = 2
    H3m = list(H3s)
    H3m[m_idx] = 0.009
    dec_m = weights_from_anchor(anchor_from_obj(synthetic_anchor(sdays, H3m, H1s), sdays))
    pm = compute(sdays, sO, sC, dec_m)
    Bm = pm["variants"]["B_ratio_grid20"]
    m_sess = sdays[GRID_ORIGIN + GRID_STEP * m_idx]
    mpos = GRID_ORIGIN + GRID_STEP * m_idx - PERIOD_START
    assert abs(dec_m[m_idx]["w_B"] - 0.9) < 1e-12 and dec_m[m_idx]["w_B"] != dec[m_idx]["w_B"] and state(Bm)[:mpos + 1] == state(B)[:mpos + 1] and Bm["daily"][mpos]["w_for_next_session"] != B["daily"][mpos]["w_for_next_session"]
    em, eb2 = [e for e in Bm["executions"] if e["weight_decision_session"] == m_sess][0], [e for e in B["executions"] if e["weight_decision_session"] == m_sess][0]
    assert em["realized_qty"] != eb2["realized_qty"] or em["side"] != eb2["side"]
    spec = load_spec()
    assert spec["sha256"] == SPEC_SHA256 and check_parents(spec) == 5
    cdays = synthetic_days(N_SESSIONS)
    rc = compute(cdays, [10000] * N_SESSIONS, [10000] * N_SESSIONS, dec)
    assert rc["variants"]["A_full_grid20"]["metrics"]["pnl_won_after_costs"] == -(rc["variants"]["A_full_grid20"]["metrics"]["fees_total_won"] + rc["variants"]["A_full_grid20"]["metrics"]["tick_cost_won"])
    pay = build_payload(spec, res, anc, "ef" * 32, datetime(2026, 11, 6, 12, 10, 0, tzinfo=timezone.utc))
    env = pay["envelope"]
    for key in ("schema_version", "artifact_id", "parent_artifact_hashes", "code_sha256", "spec_sha256", "asof_utc", "exchange_session", "symbol", "availability_status", "decision_at", "execution_not_before"):
        assert key in env, key
    assert env["artifact_id"] == "allocate_h20_v1_20261106_121000_efefefef" and env["parent_artifact_hashes"] == [spec["parents"][kk]["sha256"] for kk in PARENT_ORDER] and len(pay["decision_records"]) == N_GRID and all("observed_open" not in d for d in pay["decision_records"])
    assert len(pay["variants"]["B_ratio_grid20"]["execution_records"]) == N_GRID and len(pay["variants"]["B_ratio_grid20"]["daily_records"]) == 2907
    json.dumps(pay, allow_nan=False)
    payc = build_payload(spec, rc, anc, "ab" * 32, datetime(2026, 11, 6, 12, 11, 0, tzinfo=timezone.utc))
    json.dumps(payc, allow_nan=False)
    rdays, rO, rC = load_prices(spec)
    ranc = load_anchor(spec, rdays)
    assert len(ranc["records"]) == N_GRID and ranc["records"][0]["decision_session"] == "2014-12-30" and ranc["records"][-1]["decision_session"] == "2026-10-28" and all(r["H1"] > 0 and math.isfinite(r["H3"]) for r in ranc["records"]) and ranc["artifact_id"] == spec["anchor_artifact_id"]
    assert check_parents(spec) == 5
    print("selftest ok", dict(code_sha256_16=sha_full(os.path.abspath(__file__))[:16], here=os.path.dirname(os.path.abspath(__file__)), spec_sha256_16=spec["sha256"][:16], real_data="5 hashes, raw calendar/schema, anchor metadata and H1>0 guards only; no real ratios, weights or ledgers"))


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    elif not sys.argv[1:]:
        run()
    else:
        print("usage: allocate_h20.py | selftest")
