import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction

SPEC = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work\timing_costband_reuse_v1_spec_20261106.json")
SPEC_SHA256 = "f92baf8f955fec9014c387476e98fc06e8e9a920ee8b391e1f71bd88d53c08db"
OUT_DIR = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work\timing_costband_reuse_v1_20261106")
SCHEMA = "timing-costband-reuse-v1"
MODULE = "timing_costband_reuse_v1"
SYM = "069500"
CUTOFF = "2026-11-05"
FULL_ROWS, RETAINED_ROWS = 5939, 5938
H = 20
DEPOSIT = 10000
TICK = 5
FEE_NUM, FEE_DEN = 15, 100000
FEATURES = ("logret5", "logret20", "logret60", "ma200_distance", "sd20", "held")
HELD_INDEX = 5
PARENT_ORDER = ("raw_prices", "adjusted_prices", "allocation_mechanics_spec", "module_contract", "prior_timing_spec", "frozen_M1_R1_result", "prior_lgbm_spec", "frozen_M2_result", "costband_reuse_spec")
M2_SCHEMA, M2_ARTIFACT = "timing-lgbm-v1", "timing_lgbm_v1_20261106_124902_ab14f47a"
M1_SCHEMA, M1_ARTIFACT = "timing-actionvalue-v1", "timing_actionvalue_v1_20261106_122319_f745f401"
FIRST_DECISION, LAST_DECISION = 3031, 5936
LEDGER_CALLS = [0]
TRAVERSALS = [0]


class InvalidInput(SystemExit):
    pass


def sha_full(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def canonical_json_bytes(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def load_spec(path=SPEC):
    h = sha_full(path)
    if h != SPEC_SHA256:
        raise InvalidInput(f"spec hash {h} != {SPEC_SHA256}; refusing")
    s = json.load(open(path, encoding="utf-8"))
    if s["version"] != SCHEMA or s["output"]["directory"] != OUT_DIR or s["resources"]["cpu_threads"] != 1 or list(s["parent_order"]) != list(PARENT_ORDER):
        raise InvalidInput("spec settings differ from module constants")
    if s["data"]["cutoff_session"] != CUTOFF or s["data"]["full_input_rows"] != FULL_ROWS or s["data"]["retained_rows"] != RETAINED_ROWS or s["counts"]["source_trees"] != 2400 or s["counts"]["source_models"] != 12:
        raise InvalidInput("data/count constants differ from module constants")
    parents = {k: dict(path=s["parents"][k]["path"], sha256=s["parents"][k]["sha256"]) for k in PARENT_ORDER if k != "costband_reuse_spec"}
    parents["costband_reuse_spec"] = dict(path=path, sha256=SPEC_SHA256)
    return dict(path=path, sha256=h, parents=parents, folds={f["entry_year"]: f for f in s["walk_forward"]["folds"]}, counts=s["counts"], period=s["ledger"]["period"], external_timeout_seconds=int(s["resources"]["external_timeout_seconds"]))


def check_parents(spec):
    for name in PARENT_ORDER:
        p = spec["parents"][name]
        if not os.path.exists(p["path"]):
            raise InvalidInput(f"parent {name} missing")
        h = sha_full(p["path"])
        if h != p["sha256"]:
            raise InvalidInput(f"parent {name} hash changed: {h}")
    return len(PARENT_ORDER)


def fee(x):
    if not isinstance(x, int) or x < 0:
        raise InvalidInput("fee argument must be a non-negative integer")
    return x * FEE_NUM // FEE_DEN


def buy_cost(P):
    return P + TICK + fee(P + TICK)


def sell_proceeds(P):
    return P - TICK - fee(P - TICK)


def delta(P):
    if not isinstance(P, int) or P <= TICK:
        raise InvalidInput("delta requires an integer price above the tick")
    return (buy_cost(P) - sell_proceeds(P)) / P


def targets(O, i, h=H):
    P, Q = O[i + 1], O[i + h + 1]
    return (sell_proceeds(Q) - buy_cost(P)) / P, (sell_proceeds(Q) - sell_proceeds(P)) / P


def half_cost_from_close(C):
    return delta(C) / 2.0


def action_values(p, half_cost):
    if not (math.isfinite(p) and math.isfinite(half_cost)) or half_cost <= 0:
        raise InvalidInput("invalid prediction or cost band")
    return p - half_cost, p + half_cost


def decide_from_values(v0, v1, held):
    score = v1 if held == 1 else v0
    if score > 0:
        return 1.0, "long", score
    if score < 0:
        return 0.0, "flat", score
    return float(held), "keep", score


def validate_series(days, O, C_raw):
    if not (len(days) == len(O) == len(C_raw)):
        raise InvalidInput("array length mismatch")
    if days != sorted(days) or len(set(days)) != len(days):
        raise InvalidInput("sessions not sorted/unique")
    Oi, Ci = [], []
    for o, c in zip(O, C_raw):
        fo, fc = float(o), float(c)
        if not (math.isfinite(fo) and math.isfinite(fc) and fo.is_integer() and fc.is_integer() and fo > TICK and fc > TICK):
            raise InvalidInput("raw prices must be finite integer won above the tick")
        Oi.append(int(fo))
        Ci.append(int(fc))
    return Oi, Ci


def prepare_prices(raw, adj):
    if not (len(raw["days"]) == len(raw["O"]) == len(raw["C"])) or not (len(adj["days"]) == len(adj["C"])):
        raise InvalidInput("full parallel arrays misaligned")
    if raw["days"] != sorted(raw["days"]) or len(set(raw["days"])) != len(raw["days"]) or adj["days"] != sorted(adj["days"]) or len(set(adj["days"])) != len(adj["days"]):
        raise InvalidInput("full date arrays not sorted/unique")
    if len(raw["days"]) != FULL_ROWS or len(adj["days"]) != FULL_ROWS:
        raise InvalidInput("full input row count differs from the fixed facts")
    if raw["days"] != adj["days"]:
        raise InvalidInput("raw and adjusted full date arrays differ (checked before slicing)")
    keep = [k for k, d in enumerate(raw["days"]) if d <= CUTOFF]
    rd = [raw["days"][k] for k in keep]
    if len(rd) != RETAINED_ROWS or rd[-1] != CUTOFF or rd[0] != "2002-10-14":
        raise InvalidInput("retained calendar differs from the fixed facts")
    O, Cr = validate_series(rd, [raw["O"][k] for k in keep], [raw["C"][k] for k in keep])
    return rd, O, Cr


def load_prices(spec):
    raw = json.load(open(spec["parents"]["raw_prices"]["path"], encoding="utf-8"))[SYM]
    adj = json.load(open(spec["parents"]["adjusted_prices"]["path"], encoding="utf-8"))[SYM]
    return prepare_prices(raw, adj)


def traverse_dump(dump, x):
    TRAVERSALS[0] += 1
    total = 0.0
    for tree in dump["tree_info"]:
        node = tree["tree_structure"]
        while "leaf_value" not in node:
            node = node["left_child"] if x[node["split_feature"]] <= node["threshold"] else node["right_child"]
        total += node["leaf_value"]
    return total


def split_structure(dump):
    used = set()
    n_trees = 0
    for tree in dump["tree_info"]:
        n_trees += 1
        stack = [tree["tree_structure"]]
        while stack:
            node = stack.pop()
            if "leaf_value" in node:
                continue
            if node.get("decision_type", "<=") != "<=":
                raise InvalidInput("non-numerical split in stored tree")
            f = int(node["split_feature"])
            if f < 0 or f > 5:
                raise InvalidInput("split feature index out of range")
            used.add(f)
            stack.append(node["left_child"])
            stack.append(node["right_child"])
    return used, n_trees


def m2_from_obj(obj, days, folds=None, expected_sha=None, expected_spec_sha=None, expected_artifact=M2_ARTIFACT, first=FIRST_DECISION, last=LAST_DECISION, trees_per_model=200):
    if expected_sha is not None and obj.get("_sha256") != expected_sha:
        raise InvalidInput("M2 source hash mismatch")
    env = obj["envelope"]
    if env.get("schema_version") != M2_SCHEMA or env.get("symbol") != SYM or (expected_artifact is not None and env.get("artifact_id") != expected_artifact):
        raise InvalidInput("M2 source schema/symbol/artifact mismatch")
    if expected_spec_sha is not None and env.get("spec_sha256") != expected_spec_sha:
        raise InvalidInput("M2 source spec hash mismatch")
    if list(env["model"]["features"]) != list(FEATURES):
        raise InvalidInput("M2 feature order mismatch")
    fits = obj["fits"]
    if len(fits) != 12:
        raise InvalidInput("M2 must hold 12 models")
    total_trees = 0
    for y, f in fits.items():
        used, n_trees = split_structure(f["dump"])
        if n_trees != f["num_trees"] or n_trees != trees_per_model:
            raise InvalidInput(f"fit {y} tree count {n_trees} != {trees_per_model}")
        if sha_bytes(canonical_json_bytes(f["dump"])) != f["dump_sha256"] or sha_bytes(f["model_string"].encode("utf-8")) != f["model_string_sha256"]:
            raise InvalidInput(f"fit {y} stored hashes do not match stored content")
        if HELD_INDEX in used:
            raise InvalidInput("stored M2 trees split on held; reuse as a state-independent common component is not justified")
        if not used.issubset({0, 1, 2, 3, 4}):
            raise InvalidInput("split indices outside 0..4")
        total_trees += n_trees
    recs = obj["decision_records"]
    if len(recs) != last - first + 1:
        raise InvalidInput(f"M2 has {len(recs)} decisions, expected {last - first + 1}")
    preds = {}
    prev = -1
    for r in recs:
        i = r["decision_index"]
        s = r["decision_session"]
        if i <= prev or i < first or i > last or days[i] != s or days[i + 1] != r["entry_session"]:
            raise InvalidInput(f"M2 decision {s} not sorted/unique or misaligned with the calendar")
        prev = i
        y = r["fit_entry_year"]
        if str(y) != r["entry_session"][:4]:
            raise InvalidInput(f"M2 decision {s} fit year differs from entry year")
        fit = fits[str(y)]
        if fit["fit_cutoff_session"] != r["fit_cutoff_session"] or fit["train_max_label_available_session"] != r["train_max_label_available_session"] or r["train_max_label_available_session"] > s or r["fit_cutoff_session"] > s:
            raise InvalidInput(f"M2 decision {s} fold cutoff/maturity inconsistent")
        if folds is not None:
            fd = folds[y]
            if fit["fit_cutoff_index"] != fd["fit_cutoff_index"] or fit["fit_cutoff_session"] != fd["fit_cutoff_session"] or fit["train_max_label_available_session"] != fd["train_max_label_available_session"]:
                raise InvalidInput(f"M2 fit {y} differs from the frozen fold metadata")
        m = float(r["predicted_contrast"])
        feats = [float(r["features"][k]) for k in FEATURES]
        if not math.isfinite(m) or not all(math.isfinite(v) for v in feats):
            raise InvalidInput("non-finite stored prediction or feature")
        preds[i] = dict(p=m, features=feats, fit_entry_year=y, fit_cutoff_session=r["fit_cutoff_session"], train_max_label_available_session=r["train_max_label_available_session"], m2_held_feature=int(r["held_at_close"]))
    if first not in preds or last not in preds:
        raise InvalidInput("M2 decision range incomplete")
    m2m = obj["M2"]["metrics"]
    daily = [(d["session"], d["pnl"], d["qty"]) for d in obj["M2"]["daily_records"]]
    return dict(artifact_id=env["artifact_id"], spec_sha256=env.get("spec_sha256"), predictions=preds, n=len(preds), total_trees=total_trees, fits=fits, metrics=m2m, daily=daily, execution_opportunities=m2m["execution_opportunities"])


def traversal_check(m2, tol=1e-12):
    max_err, sign_agree = 0.0, 0
    for i, p in sorted(m2["predictions"].items()):
        t = traverse_dump(m2["fits"][str(p["fit_entry_year"])]["dump"], p["features"])
        e = abs(t - p["p"])
        max_err = max(max_err, e)
        same_sign = (t > 0) == (p["p"] > 0) and (t < 0) == (p["p"] < 0) and (t == 0) == (p["p"] == 0)
        sign_agree += int(same_sign)
        if e > tol or not same_sign:
            raise InvalidInput(f"stored prediction at index {i} not reproduced by its stored dump (err {e}, sign match {same_sign})")
    return dict(checked=len(m2["predictions"]), max_abs_error=max_err, sign_agreement=sign_agree, tolerance=tol)


def anchor_m1r1_from_obj(obj, expected_sha=None, expected_spec_sha=None, expected_artifact=M1_ARTIFACT):
    if expected_sha is not None and obj.get("_sha256") != expected_sha:
        raise InvalidInput("M1/R1 anchor hash mismatch")
    env = obj["envelope"]
    if env.get("schema_version") != M1_SCHEMA or env.get("symbol") != SYM or (expected_artifact is not None and env.get("artifact_id") != expected_artifact):
        raise InvalidInput("M1/R1 anchor schema/symbol/artifact mismatch")
    if expected_spec_sha is not None and env.get("spec_sha256") != expected_spec_sha:
        raise InvalidInput("M1/R1 anchor spec hash mismatch")
    out = {}
    for k in ("M1", "R1"):
        v = obj["variants"][k]
        daily = v["daily_records"]
        sessions = [d["session"] for d in daily]
        if sessions != sorted(sessions) or len(set(sessions)) != len(sessions):
            raise InvalidInput(f"anchor {k} daily rows not unique/sorted")
        out[k] = dict(metrics=v["metrics"], daily=[(d["session"], d["pnl"], d["qty"]) for d in daily])
    return dict(artifact_id=env["artifact_id"], spec_sha256=env.get("spec_sha256"), variants=out)


def anchor_calendar_check(anchors, days, start, end, counts):
    for k, v in anchors.items():
        sess = [s for s, _, _ in v["daily"]]
        if sess != days[start:end + 1] or len(sess) != counts["daily_rows_per_ledger"] or v["metrics"]["deposits"] != counts["deposits_won_per_ledger"] or v["metrics"]["execution_opportunities"] != counts["execution_opportunities_per_ledger"]:
            raise InvalidInput(f"anchor {k} calendar/counts differ from spec")
    return len(anchors)


def ledger_run(days, O, C_raw, start, end, decide):
    LEDGER_CALLS[0] += 1
    cash, qty, deposits = 0, 0, 0
    pending = None
    daily, execs, trades, decisions = [], [], [], []
    buys = sells = entries = exits = maintenance = 0
    fees = tick_cost = turnover = 0
    for t in range(start, end + 1):
        executed_w = None
        if pending is not None:
            w, dec_i = pending
            o = O[t]
            W_open = cash + qty * o
            target = math.floor(w * W_open / o) if w > 0 else 0
            cash_b, qty_b = cash, qty
            side, n, px, f = None, 0, None, 0
            if target < qty:
                n, px = qty - target, o - TICK
                f = fee(n * px)
                cash += n * px - f
                qty -= n
                sells += 1
                side = "sell"
                if qty == 0:
                    exits += 1
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
                    if qty_b == 0:
                        entries += 1
                    else:
                        maintenance += 1
            if side:
                fees += f
                tick_cost += n * TICK
                turnover += n * o
                if (cash + qty * o) - (cash_b + qty_b * o) != -(n * TICK + f) or cash < 0 or qty < 0:
                    raise InvalidInput("ledger invariant violated")
                trades.append(dict(weight_decision_session=days[dec_i], execution_session=days[t], side=side, kind=("entry" if side == "buy" and qty_b == 0 else "maintenance" if side == "buy" else "exit" if qty == 0 else "partial_sell"), qty=n, price=px, fee=f, cash_before=cash_b, cash_after=cash, qty_before=qty_b, qty_after=qty))
            execs.append(dict(weight_decision_session=days[dec_i], execution_session=days[t], observed_open=o, w=w, W_open=W_open, target_qty=target, side=side, realized_qty=n, cash_before=cash_b, cash_after=cash, qty_before=qty_b, qty_after=qty, fee=f, tick=n * TICK))
            executed_w = w
            pending = None
        deposits += DEPOSIT
        cash += DEPOSIT
        value = cash + qty * C_raw[t]
        held = 1 if qty > 0 else 0
        w_next = None
        if t + 1 <= end:
            dec = decide(t, held)
            w_next = dec["w"]
            pending = (w_next, t)
            decisions.append(dict(decision_index=t, decision_session=days[t], entry_session=days[t + 1], m3_held_at_close=held, **dec))
        daily.append(dict(session=days[t], cash=cash, qty=qty, value=value, deposits=deposits, pnl=value - deposits, exposure=(qty * C_raw[t] / value) if value > 0 else 0.0, held=held, executed_w_from_previous_close=executed_w, w_for_next_session=w_next))
    pnl = [d["pnl"] for d in daily]
    peak, dd = 0, 0
    for x in pnl:
        peak = max(peak, x)
        dd = min(dd, x - peak)
    met = dict(start=days[start], end=days[end], sessions=len(daily), unique_daily_rows=len({d["session"] for d in daily}), deposit_events=len(daily), deposits=deposits, pnl_won_after_costs=pnl[-1], pnl_pct_of_deposits=pnl[-1] / deposits if deposits else None, max_dd_won=dd, exposure_avg=sum(d["exposure"] for d in daily) / len(daily), days_held=sum(1 for d in daily if d["held"] == 1), buys=buys, sells=sells, flat_to_long_entries=entries, long_to_flat_exits=exits, long_maintenance_buys=maintenance, trades=len(trades), execution_opportunities=len(execs), fees_total_won=fees, tick_cost_won=tick_cost, turnover_won=turnover, qty_end=qty, final_cash=cash, final_value=daily[-1]["value"])
    return dict(metrics=met, daily=daily, executions=execs, trades=trades, decisions=decisions)


def yearly_increments(pnl_by_session):
    last = {}
    for s, v in pnl_by_session:
        last[s[:4]] = v
    out, prev = {}, 0
    for y in sorted(last):
        out[y] = last[y] - prev
        prev = last[y]
    return out


def holding_streaks(daily):
    runs, cur = [], 0
    for d in daily:
        if d["held"] == 1:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    if not runs:
        return dict(count=0, min=None, median=None, max=None, mean=None)
    s = sorted(runs)
    k = len(s)
    med = float(s[k // 2]) if k % 2 else (s[k // 2 - 1] + s[k // 2]) / 2.0
    return dict(count=k, min=s[0], median=med, max=s[-1], mean=sum(s) / k)


def exit_episodes(trades):
    eps, open_exit = [], None
    for t in trades:
        if t["kind"] == "exit":
            open_exit = t["execution_session"]
        elif t["kind"] == "entry" and open_exit is not None:
            eps.append(dict(exit_session=open_exit, reentry_session=t["execution_session"]))
            open_exit = None
    if open_exit is not None:
        eps.append(dict(exit_session=open_exit, reentry_session=None))
    return eps


def make_decider(days, C_raw, m2):
    def decide(i, held):
        p = m2["predictions"].get(i)
        if p is None:
            raise InvalidInput(f"no stored M2 prediction for decision {days[i]}")
        C = C_raw[i]
        hc = half_cost_from_close(C)
        v0, v1 = action_values(p["p"], hc)
        w, action, score = decide_from_values(v0, v1, held)
        return dict(source_artifact_id=m2["artifact_id"], source_fit_entry_year=p["fit_entry_year"], source_fit_cutoff_session=p["fit_cutoff_session"], source_train_max_label_available_session=p["train_max_label_available_session"], reused_p=p["p"], observed_raw_close=C, delta_hat=2.0 * hc, half_cost=hc, cash_value=v0, held_value=v1, score=score, action=action, w=w)

    return decide


def posthoc_cost_records(days, O, C_raw, decisions):
    recs, errs = [], []
    for d in decisions:
        i = d["decision_index"]
        if i + 1 >= len(days):
            continue
        P = O[i + 1]
        da, dh = delta(P), d["delta_hat"]
        e = da - dh
        errs.append(e)
        recs.append(dict(posthoc_only=True, available_at="next_session_open", decision_session=days[i], execution_session=days[i + 1], observed_next_open=P, close_proxy=C_raw[i], delta_actual=da, delta_hat=dh, signed_error=e))
    summary = dict(n=len(errs), min_error=min(errs) if errs else None, max_error=max(errs) if errs else None, mean_abs_error=(sum(abs(e) for e in errs) / len(errs)) if errs else None, max_abs_error=max(abs(e) for e in errs) if errs else None, note="delta(raw O[i+1]) minus delta_hat(raw C[i]); exact integer fee expression; computed after the completed ledger, never fed to decisions")
    return recs, summary


def compute(days, O, C_raw, start, end, m2):
    LEDGER_CALLS[0] = 0
    M3 = ledger_run(days, O, C_raw, start, end, make_decider(days, C_raw, m2))
    regions = {}
    for d in M3["decisions"]:
        y = d["entry_session"][:4]
        g = regions.setdefault(y, dict(n=0, long=0, flat=0, keep=0, both_long=0, in_band=0, both_flat=0))
        g["n"] += 1
        g[d["action"]] += 1
        if d["cash_value"] > 0:
            g["both_long"] += 1
        elif d["held_value"] < 0:
            g["both_flat"] += 1
        else:
            g["in_band"] += 1
    return dict(M3=M3, regions=regions, yearly_M3=yearly_increments([(d["session"], d["pnl"]) for d in M3["daily"]]), streaks=holding_streaks(M3["daily"]), episodes=exit_episodes(M3["trades"]), ledger_calls=LEDGER_CALLS[0])


def compare(M3, anchors):
    out = {}
    m3 = {d["session"]: d for d in M3["daily"]}
    for k, v in anchors.items():
        if [s for s, _, _ in v["daily"]] != [d["session"] for d in M3["daily"]]:
            raise InvalidInput(f"anchor {k} sessions differ from M3")
        diffs = [m3[s]["pnl"] - p for s, p, _ in v["daily"]]
        inc_a = yearly_increments([(s, p) for s, p, _ in v["daily"]])
        inc_m3 = yearly_increments([(d["session"], d["pnl"]) for d in M3["daily"]])
        am = v["metrics"]
        out[f"M3_minus_{k}"] = dict(role="primary" if k == "R1" else "secondary", terminal_pnl=M3["metrics"]["pnl_won_after_costs"] - am["pnl_won_after_costs"], max_dd_won_diff=M3["metrics"]["max_dd_won"] - am["max_dd_won"], daily_diff_min=min(diffs), daily_diff_max=max(diffs), sessions_m3_fewer_shares=sum(1 for s, _, q in v["daily"] if m3[s]["qty"] < q), yearly_increments_diff={y: inc_m3[y] - inc_a[y] for y in inc_m3}, fees_diff=M3["metrics"]["fees_total_won"] - am["fees_total_won"], tick_diff=M3["metrics"]["tick_cost_won"] - am["tick_cost_won"], turnover_diff=M3["metrics"]["turnover_won"] - am["turnover_won"], exposure_diff=M3["metrics"]["exposure_avg"] - am["exposure_avg"], days_held_diff=M3["metrics"]["days_held"] - am["days_held"], counts=dict(M3=dict(buys=M3["metrics"]["buys"], sells=M3["metrics"]["sells"], entries=M3["metrics"]["flat_to_long_entries"], exits=M3["metrics"]["long_to_flat_exits"], maintenance=M3["metrics"]["long_maintenance_buys"]), anchor=dict(buys=am["buys"], sells=am["sells"], entries=am["flat_to_long_entries"], exits=am["long_to_flat_exits"], maintenance=am["long_maintenance_buys"])))
    return out


def summary_lines(res, anchors, cmp, m2, trav, post_summary):
    m = res["M3"]["metrics"]
    lines = [f"[{m['start']}~{m['end']} {m['sessions']}세션 입금 {m['deposits']:,} 새 적합 0 원장 {res['ledger_calls']}개 결정 {len(res['M3']['decisions'])} | M2 저장 예측 {m2['n']}개 재사용, 트리 {m2['total_trees']}개 held 분할 0, 사전 순회 {trav['checked']}건 최대오차 {trav['max_abs_error']:.1e} 부호일치 {trav['sign_agreement']}]"]
    lines.append(f"  M3 비용밴드: 손익 {m['pnl_won_after_costs']:+,} ({m['pnl_pct_of_deposits']*100:+.1f}%) 낙폭 {m['max_dd_won']:+,} 노출 {m['exposure_avg']*100:.0f}% 보유일 {m['days_held']} 진입 {m['flat_to_long_entries']} 청산 {m['long_to_flat_exits']} 유지매수 {m['long_maintenance_buys']} 매수 {m['buys']} 매도 {m['sells']} 수수료 {m['fees_total_won']:,} 틱 {m['tick_cost_won']:,} 회전 {m['turnover_won']:,} 기말 현금 {m['final_cash']:,} 수량 {m['qty_end']} 가치 {m['final_value']:,}")
    for k, v in anchors.items():
        am = v["metrics"]
        lines.append(f"  앵커 {k}(로드): 손익 {am['pnl_won_after_costs']:+,} 낙폭 {am['max_dd_won']:+,} 보유일 {am['days_held']} 진입 {am['flat_to_long_entries']} 청산 {am['long_to_flat_exits']} 수수료 {am['fees_total_won']:,} 틱 {am['tick_cost_won']:,}")
    for k, c in cmp.items():
        lines.append(f"  {k} [{c['role']}]: 기말 {c['terminal_pnl']:+,} 낙폭차 {c['max_dd_won_diff']:+,} 일별 최소 {c['daily_diff_min']:+,} 최대 {c['daily_diff_max']:+,} 보유주 적은 세션 {c['sessions_m3_fewer_shares']} 보유일차 {c['days_held_diff']:+} 수수료차 {c['fees_diff']:+,} 틱차 {c['tick_diff']:+,} | 연도별 증분차: " + ", ".join(f"{y} {x:+,}" for y, x in c["yearly_increments_diff"].items()))
    lines.append("  연도별 증분 M3: " + ", ".join(f"{y} {v:+,}" for y, v in res["yearly_M3"].items()))
    lines.append("  영역(진입연도): " + ", ".join(f"{y} 양쪽롱 {g['both_long']}/밴드내 {g['in_band']}/양쪽플랫 {g['both_flat']} → 롱 {g['long']}·플랫 {g['flat']}·유지 {g['keep']}" for y, g in sorted(res["regions"].items())))
    lines.append(f"  보유 연속구간: {res['streaks']} | 청산→재진입 {len(res['episodes'])}구간")
    lines.append(f"  사후 d(P)−d̂(C) (posthoc_only, 결정 미반영): n {post_summary['n']} 최소 {post_summary['min_error']} 최대 {post_summary['max_error']} 평균절대 {post_summary['mean_abs_error']} 최대절대 {post_summary['max_abs_error']}")
    return lines


def build_payload(spec, res, m2, trav, anchors_all, anchor_ids, cmp, post_recs, post_summary, code_sha, now_utc):
    artifact_id = f"{MODULE}_{now_utc.strftime('%Y%m%d_%H%M%S')}_{code_sha[:8]}"
    p = spec["parents"]
    env = dict(schema_version=SCHEMA, artifact_id=artifact_id, parent_artifact_hashes=[p[k]["sha256"] for k in PARENT_ORDER], parent_artifact_roles={k: dict(path=p[k]["path"], sha256=p[k]["sha256"]) for k in PARENT_ORDER}, source_artifact_ids=anchor_ids, code_sha256=code_sha, spec_sha256=spec["sha256"], asof_utc=now_utc.isoformat(timespec="seconds"), thread_env={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}, exchange_session=dict(exchange="KRX", symbol=SYM, calendar="raw daily session dates", timezone="Asia/Seoul", utc_mapping="session date D in KST (UTC+09:00); no intraday clock times used", decision_marker="session_close", execution_marker="next_session_open"), symbol=SYM, availability_status="available", decision_at="session_close (reused frozen M2 prediction, observed raw close cost proxy, actual M3 held state; logical marker)", execution_not_before="next_session_open (quantity decided after observing the open; idealized fill at O +/- 5, no actual order)", decomposition=dict(identity="(f - z0)^2 + (f - z1)^2 = 2 (f - zbar)^2 + delta^2 / 2 with zbar = (z0 + z1)/2 and delta = z1 - z0 = d(P)", source_interpretation="frozen M2 prediction p is the learned common mean component (no held splits in 2400 trees); not a proof of optimality, calibration or conditional-expectation accuracy"), policy=dict(cost_proxy="delta_hat = (10 + F(C+5) + F(C-5)) / C with C = raw close at the decision; half_cost = delta_hat / 2", cash_value="v0 = p - half_cost", held_value="v1 = p + half_cost", score="v1 if actual M3 qty > 0 else v0; > 0 weight 1; < 0 weight 0; exactly 0 keep", forbidden="asymmetric v0 = p, v1 = p + delta_hat is not this experiment", limitation="delta_hat approximates the unknown next-open one-share cost gap with the observed close; repeated transitions pay true integer ledger costs"), source_checks=dict(models=12, trees=m2["total_trees"], held_split_trees=0, prediction_traversal=trav, reused_predictions=m2["n"]), ledger=dict(mechanics="allocate-v1 timeline; integer won; deposit 10,000 at each close; idealized fills; zero tax, zero cash yield, no dividends/distributions", anchors="R1, M1 (78176801) and M2 (d9556ca9) loaded only; never replayed"), units=dict(cash="krw (integer)", value="krw (integer)", pnl="krw (integer)", qty="shares (integer)", w="weight in {0,1}", p="dimensionless per-share value contrast", delta_hat="dimensionless cost gap", pnl_pct_of_deposits="fraction, not annualized"), counts=dict(real_model_fits=0, new_native_prediction_calls=0, new_feature_rows=0, source_models=12, source_trees=m2["total_trees"], source_decision_manual_checks=trav["checked"], decision_predictions_reused=m2["n"], M3_decision_records=len(res["M3"]["decisions"]), ledger_paths=res["ledger_calls"], daily_rows_per_ledger=res["M3"]["metrics"]["unique_daily_rows"], execution_opportunities_per_ledger=res["M3"]["metrics"]["execution_opportunities"], deposit_events_per_ledger=res["M3"]["metrics"]["deposit_events"], deposits_won_per_ledger=res["M3"]["metrics"]["deposits"], frozen_anchor_paths_loaded=len(anchors_all), traversal_calls=TRAVERSALS[0]), caveats="annual differences include carried share valuation; cost differences are included and not a zero-cost counterfactual; lower exposure can reduce risk without timing skill; retrospective diagnostic on viewed data; no calibrated expected value, optimal or live timing claim", resources=dict(cpu_threads=1, external_timeout_seconds=spec["external_timeout_seconds"], real_experiments=1, new_fits=0, gpu=0, api_calls=0, orders=0))
    ref = dict(envelope_artifact_id=artifact_id)
    return dict(envelope=env, lines=summary_lines(res, anchors_all, cmp, m2, trav, post_summary), decision_records=[dict(d, **ref) for d in res["M3"]["decisions"]], M3=dict(metrics=res["M3"]["metrics"], execution_records=[dict(e, **ref) for e in res["M3"]["executions"]], trade_records=[dict(t, **ref) for t in res["M3"]["trades"]], daily_records=[dict(d, **ref) for d in res["M3"]["daily"]]), anchors_loaded={k: dict(metrics=v["metrics"], loaded_from=anchor_ids["M2" if k == "M2" else "M1_R1"]) for k, v in anchors_all.items()}, comparisons=cmp, annual=dict(M3=res["yearly_M3"]), regions=res["regions"], holding_streaks=res["streaks"], exit_reentry_episodes=res["episodes"], posthoc_cost=dict(summary=post_summary, records=post_recs))


def run(spec_path=SPEC, out_path=None):
    spec = load_spec(spec_path)
    check_parents(spec)
    out = out_path or os.path.join(OUT_DIR, "timing_costband_reuse_v1_results.json")
    if os.path.exists(out):
        raise InvalidInput(f"{out} exists; refusing before loading prices or computing")
    TRAVERSALS[0] = 0
    m2obj = json.load(open(spec["parents"]["frozen_M2_result"]["path"], encoding="utf-8"))
    m2obj["_sha256"] = spec["parents"]["frozen_M2_result"]["sha256"]
    days, O, Cr = load_prices(spec)
    m2 = m2_from_obj(m2obj, days, folds=spec["folds"], expected_sha=spec["parents"]["frozen_M2_result"]["sha256"], expected_spec_sha=spec["parents"]["prior_lgbm_spec"]["sha256"])
    del m2obj
    a1obj = json.load(open(spec["parents"]["frozen_M1_R1_result"]["path"], encoding="utf-8"))
    a1obj["_sha256"] = spec["parents"]["frozen_M1_R1_result"]["sha256"]
    a1 = anchor_m1r1_from_obj(a1obj, expected_sha=a1obj["_sha256"], expected_spec_sha=spec["parents"]["prior_timing_spec"]["sha256"])
    del a1obj
    start, end = FIRST_DECISION, len(days) - 1
    if days[start] != spec["period"][0] or days[end] != spec["period"][1]:
        raise InvalidInput("ledger period differs from spec")
    anchors_all = dict(a1["variants"], M2=dict(metrics=m2["metrics"], daily=m2["daily"]))
    anchor_calendar_check(anchors_all, days, start, end, spec["counts"])
    trav = traversal_check(m2)
    if trav["checked"] != spec["counts"]["source_decision_manual_checks"]:
        raise InvalidInput("traversal count differs from spec")
    res = compute(days, O, Cr, start, end, m2)
    c = spec["counts"]
    m = res["M3"]["metrics"]
    if m["unique_daily_rows"] != c["daily_rows_per_ledger"] or m["execution_opportunities"] != c["execution_opportunities_per_ledger"] or m["deposit_events"] != c["deposit_events_per_ledger"] or m["deposits"] != c["deposits_won_per_ledger"] or len(res["M3"]["decisions"]) != c["M3_decision_records"] or res["ledger_calls"] != c["ledger_paths"]:
        raise InvalidInput("counts differ from spec")
    cmp = compare(res["M3"], anchors_all)
    post_recs, post_summary = posthoc_cost_records(days, O, Cr, res["M3"]["decisions"])
    check_parents(spec)
    code_sha = sha_full(os.path.abspath(__file__))
    payload = build_payload(spec, res, m2, trav, anchors_all, dict(M2=m2["artifact_id"], M1_R1=a1["artifact_id"]), cmp, post_recs, post_summary, code_sha, datetime.now(timezone.utc))
    data = json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False).encode("utf-8")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; refusing to overwrite")
    with open(out, "xb") as f:
        f.write(data)
    print("\n".join(payload["lines"]))
    print("saved", out, "sha", sha_full(out)[:16], "artifact", payload["envelope"]["artifact_id"], "code", code_sha[:16], "spec", spec["sha256"][:16])
    return payload


def synthetic_days(n, start=date(2003, 1, 1)):
    return [(start + timedelta(days=k)).isoformat() for k in range(n)]


def expect_reject(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except InvalidInput:
        return
    raise AssertionError("expected rejection")


def leaf(v):
    return {"leaf_value": v, "leaf_count": 1}


def split(feature, thr, left, right):
    return {"split_index": 0, "split_feature": feature, "threshold": thr, "decision_type": "<=", "default_left": True, "left_child": left, "right_child": right}


def synthetic_m2(days, dump, first, last, held_fn=lambda i: i % 2, cutoff_session=None, artifact="m2_synthetic", spec_sha="specsha", n_trees=None):
    trees = n_trees if n_trees is not None else len(dump["tree_info"])
    fit = dict(dump=dump, num_trees=trees, fit_cutoff_session=cutoff_session or days[first], fit_cutoff_index=first, train_max_label_available_session=cutoff_session or days[first], model_string="synthetic", model_string_sha256=sha_bytes(b"synthetic"), dump_sha256=sha_bytes(canonical_json_bytes(dump)))
    recs = []
    for i in range(first, last + 1):
        feats = [((i * 7) % 5 - 2) / 10.0, 0.0, 0.0, 0.0, ((i * 3) % 4) / 4.0, float(held_fn(i))]
        recs.append(dict(decision_index=i, decision_session=days[i], entry_session=days[i + 1], held_at_close=held_fn(i), fit_entry_year=int(days[i + 1][:4]), fit_cutoff_session=fit["fit_cutoff_session"], train_max_label_available_session=fit["train_max_label_available_session"], predicted_contrast=traverse_dump(dump, feats), features=dict(zip(FEATURES, feats))))
    years = sorted({r["fit_entry_year"] for r in recs})
    fits = {str(y): dict(fit) for y in years}
    while len(fits) < 12:
        fits[str(1900 + len(fits))] = dict(fit)
    n = last + 1
    return dict(envelope=dict(schema_version=M2_SCHEMA, symbol=SYM, artifact_id=artifact, spec_sha256=spec_sha, model=dict(features=list(FEATURES))), fits=fits, decision_records=recs, M2=dict(metrics=dict(pnl_won_after_costs=0, max_dd_won=0, days_held=0, flat_to_long_entries=0, long_to_flat_exits=0, long_maintenance_buys=0, buys=0, sells=0, fees_total_won=0, tick_cost_won=0, turnover_won=0, exposure_avg=0.0, deposits=(n - first) * DEPOSIT, execution_opportunities=n - first - 1), daily_records=[dict(session=days[t], pnl=0, qty=0) for t in range(first, n)]))


def selftest_synthetic():
    for P, Q in ((5000, 5005), (20000, 20600), (6667, 6600), (84610, 108820)):
        O = [0, P] + [0] * 19 + [Q]
        z0, z1 = targets(O, 0)
        Fz0, Fz1 = Fraction(sell_proceeds(Q) - buy_cost(P), P), Fraction(sell_proceeds(Q) - sell_proceeds(P), P)
        zbar, dF = (Fz0 + Fz1) / 2, Fz1 - Fz0
        for f in (Fraction(0), Fraction(1, 1000), Fraction(-3, 500)):
            assert (f - Fz0) ** 2 + (f - Fz1) ** 2 == 2 * (f - zbar) ** 2 + dF ** 2 / 2
        assert dF == Fraction(buy_cost(P) - sell_proceeds(P), P) and abs(float(dF) - delta(P)) <= 1e-18 and abs((z1 - z0) - delta(P)) <= 1e-15 * max(1.0, abs(z0), abs(z1))
    assert abs((delta(20600) - delta(20000)) - 2.6699029126213593e-05) < 1e-15 and abs(10 * (1 / 20600 - 1 / 20000) + 1.4563106796116505e-05) < 1e-15
    hc = 0.001
    for p, h, exp_w in ((-0.0005, 0, 0.0), (-0.0005, 1, 1.0), (0.0005, 0, 0.0), (0.0005, 1, 1.0), (0.0011, 0, 1.0), (0.0011, 1, 1.0), (-0.0011, 1, 0.0), (-0.0011, 0, 0.0)):
        w, action, score = decide_from_values(*action_values(p, hc), h)
        assert w == exp_w, (p, h, w, exp_w)
        if -hc < p < hc:
            assert w == float(h) and action != "keep"
    assert decide_from_values(*action_values(hc, hc), 0) == (0.0, "keep", 0.0) and decide_from_values(*action_values(hc, hc), 1) == (1.0, "long", 2 * hc)
    assert decide_from_values(*action_values(-hc, hc), 1) == (1.0, "keep", 0.0) and decide_from_values(*action_values(-hc, hc), 0) == (0.0, "flat", -2 * hc)
    assert action_values(0.0, hc) == (-hc, hc) and action_values(0.0005, hc) != (0.0005, 0.0005 + 2 * hc)
    expect_reject(action_values, float("nan"), hc)
    expect_reject(action_values, 0.0, 0.0)
    expect_reject(delta, 5)
    assert half_cost_from_close(20000) == delta(20000) / 2.0
    n = 70
    days = synthetic_days(n)
    O = [20000 + 10 * ((k * 37) % 23) for k in range(n)]
    C = [o + 5 * (((k * 11) % 7) - 3) for k, o in enumerate(O)]
    Oi, Ci = validate_series(days, O, C)
    dump = {"tree_info": [{"tree_structure": split(4, 0.5, leaf(0.0004), split(0, 0.0, leaf(-0.001), leaf(0.0009)))}, {"tree_structure": leaf(0.0001)}]}
    used, nt = split_structure(dump)
    assert used == {0, 4} and nt == 2
    assert abs(traverse_dump(dump, [0.1, 0, 0, 0, 0.2, 0]) - 0.0005) < 1e-15 and abs(traverse_dump(dump, [0, 0, 0, 0, 0.7, 0]) - (-0.0009)) < 1e-15 and abs(traverse_dump(dump, [0.3, 0, 0, 0, 0.7, 0]) - 0.0010) < 1e-15
    first, last = 10, n - 2
    m2obj = synthetic_m2(days, dump, first, last)
    m2 = m2_from_obj(m2obj, days, expected_artifact="m2_synthetic", first=first, last=last, trees_per_model=2)
    assert m2["n"] == last - first + 1 and m2["total_trees"] == 2 * 12
    trav = traversal_check(m2)
    assert trav["checked"] == m2["n"] and trav["max_abs_error"] == 0.0 and trav["sign_agreement"] == m2["n"]
    bad = json.loads(json.dumps(m2obj))
    bad["fits"][sorted(bad["fits"])[0]]["dump"] = {"tree_info": [{"tree_structure": split(5, 0.5, leaf(0.0), leaf(0.001))}, {"tree_structure": leaf(0.0)}]}
    bad["fits"][sorted(bad["fits"])[0]]["dump_sha256"] = sha_bytes(canonical_json_bytes(bad["fits"][sorted(bad["fits"])[0]]["dump"]))
    expect_reject(m2_from_obj, bad, days, expected_artifact="m2_synthetic", first=first, last=last, trees_per_model=2)
    bad = json.loads(json.dumps(m2obj))
    bad["decision_records"][3]["predicted_contrast"] += 1e-6
    expect_reject(traversal_check, m2_from_obj(bad, days, expected_artifact="m2_synthetic", first=first, last=last, trees_per_model=2))
    bad = json.loads(json.dumps(m2obj))
    bad["decision_records"][3]["predicted_contrast"] = -bad["decision_records"][3]["predicted_contrast"] if bad["decision_records"][3]["predicted_contrast"] else 1e-13
    expect_reject(traversal_check, m2_from_obj(bad, days, expected_artifact="m2_synthetic", first=first, last=last, trees_per_model=2))
    for mutate in (lambda b: b["decision_records"].append(dict(b["decision_records"][-1])), lambda b: b["decision_records"][3].__setitem__("decision_session", days[0]), lambda b: b["decision_records"][3].__setitem__("fit_entry_year", 1999), lambda b: b["decision_records"][3].__setitem__("fit_cutoff_session", days[n - 1]), lambda b: b["envelope"].__setitem__("artifact_id", "other"), lambda b: b["envelope"].__setitem__("spec_sha256", "wrong"), lambda b: b["envelope"]["model"]["features"].reverse(), lambda b: b["fits"][sorted(b["fits"])[0]].__setitem__("dump_sha256", "0" * 64), lambda b: b["decision_records"][2]["features"].__setitem__("sd20", float("nan"))):
        bad = json.loads(json.dumps(m2obj))
        mutate(bad)
        expect_reject(m2_from_obj, bad, days, expected_artifact="m2_synthetic", expected_spec_sha="specsha", first=first, last=last, trees_per_model=2)
    bad = json.loads(json.dumps(m2obj))
    bad["_sha256"] = "00" * 32
    expect_reject(m2_from_obj, bad, days, expected_sha="11" * 32, expected_artifact="m2_synthetic", first=first, last=last, trees_per_model=2)
    start, end = first, n - 1
    res = compute(days, Oi, Ci, start, end, m2)
    M3 = res["M3"]
    assert res["ledger_calls"] == 1 and len(M3["decisions"]) == end - start and M3["metrics"]["execution_opportunities"] == end - start and M3["metrics"]["unique_daily_rows"] == end - start + 1
    for d in M3["decisions"]:
        i = d["decision_index"]
        assert d["observed_raw_close"] == Ci[i] and d["half_cost"] == delta(Ci[i]) / 2.0 and d["cash_value"] == d["reused_p"] - d["half_cost"] and d["held_value"] == d["reused_p"] + d["half_cost"] and d["score"] == (d["held_value"] if d["m3_held_at_close"] == 1 else d["cash_value"])
        assert "observed_next_open" not in d and (d["action"], d["w"]) == {"long": ("long", 1.0), "flat": ("flat", 0.0), "keep": ("keep", float(d["m3_held_at_close"]))}[d["action"]]
    O_f = list(Oi)
    for t in range(start + 1, n):
        O_f[t] += 500
    res_f = compute(days, O_f, Ci, start, end, m2)
    assert [(d["reused_p"], d["half_cost"], d["w"]) for d in res_f["M3"]["decisions"]][0] == [(d["reused_p"], d["half_cost"], d["w"]) for d in M3["decisions"]][0]
    recs_a, sum_a = posthoc_cost_records(days, Oi, Ci, M3["decisions"])
    recs_b, sum_b = posthoc_cost_records(days, O_f, Ci, M3["decisions"])
    assert sum_a["n"] == len(M3["decisions"]) and recs_a[0]["posthoc_only"] and recs_a[0]["available_at"] == "next_session_open" and sum_b != sum_a and [d["w"] for d in M3["decisions"]] == [d["w"] for d in M3["decisions"]]
    C_k = list(Ci)
    C_k[start] = 90000
    res_c = compute(days, Oi, C_k, start, end, m2)
    assert res_c["M3"]["decisions"][0]["half_cost"] != M3["decisions"][0]["half_cost"]
    assert sum(res["yearly_M3"].values()) == M3["metrics"]["pnl_won_after_costs"]
    d5 = synthetic_days(5)
    O5, C5 = [5000] * 5, [5000] * 5
    seq = {0: 1.0, 1: 1.0, 2: 0.5, 3: 1.0}
    led = ledger_run(d5, O5, C5, 0, 4, lambda i, held: dict(action="x", w=seq[i]))
    assert [(x["side"], x["qty"], x["price"], x["fee"]) for x in led["trades"]] == [("buy", 1, 5005, 0), ("buy", 2, 5005, 1), ("sell", 1, 4995, 0), ("buy", 5, 5005, 3)] and led["metrics"]["final_cash"] == 14951 and led["metrics"]["qty_end"] == 7 and led["metrics"]["pnl_won_after_costs"] == -49 and led["metrics"]["fees_total_won"] == 4 and led["metrics"]["tick_cost_won"] == 45
    one = ledger_run(d5[:1], O5[:1], C5[:1], 0, 0, lambda i, held: dict(action="long", w=1.0))
    assert one["executions"] == [] and one["metrics"]["final_value"] == DEPOSIT
    two = ledger_run(d5[:2], O5[:2], C5[:2], 0, 1, lambda i, held: dict(action="long", w=1.0))
    assert len(two["executions"]) == 1 and two["daily"][1]["w_for_next_session"] is None
    d6 = synthetic_days(6)
    poor6 = ledger_run(d6, [45000] * 6, [45000] * 6, 0, 5, lambda i, held: dict(action="long", w=1.0))
    assert poor6["metrics"]["trades"] == 1 and poor6["trades"][0]["execution_session"] == d6[5]
    seqf = {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0}
    ff = ledger_run(d5, O5, C5, 0, 4, lambda i, held: dict(action="x", w=seqf[i]))
    assert [t["kind"] for t in ff["trades"]] == ["entry", "exit", "entry"] and exit_episodes(ff["trades"]) == [dict(exit_session=d5[2], reentry_session=d5[4])]
    keep = ledger_run(d5, O5, C5, 0, 4, lambda i, held: dict(action="keep", w=float(held)))
    assert keep["metrics"]["trades"] == 0
    hs = lambda pattern: holding_streaks([dict(held=v) for v in pattern])
    assert hs([1, 0, 1, 1, 1])["median"] == 2.0 and hs([0, 0])["count"] == 0 and hs([0, 1, 1, 0])["median"] == 2.0 and hs([1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1])["median"] == 2.5
    anchors_obj = dict(envelope=dict(schema_version=M1_SCHEMA, symbol=SYM, artifact_id="anchor_synthetic", spec_sha256="s1"), variants={k: dict(metrics=dict(M3["metrics"], pnl_won_after_costs=M3["metrics"]["pnl_won_after_costs"] + 7), daily_records=[dict(session=d["session"], pnl=d["pnl"] + 7, qty=d["qty"]) for d in M3["daily"]]) for k in ("M1", "R1")})
    anc = anchor_m1r1_from_obj(anchors_obj, expected_artifact="anchor_synthetic", expected_spec_sha="s1")
    all_anchors = dict(anc["variants"], M2=dict(metrics=M3["metrics"], daily=[(d["session"], d["pnl"], d["qty"]) for d in M3["daily"]]))
    counts_s = dict(daily_rows_per_ledger=len(M3["daily"]), deposits_won_per_ledger=M3["metrics"]["deposits"], execution_opportunities_per_ledger=M3["metrics"]["execution_opportunities"])
    assert anchor_calendar_check(all_anchors, days, start, end, counts_s) == 3
    cmp = compare(M3, all_anchors)
    assert cmp["M3_minus_R1"]["role"] == "primary" and cmp["M3_minus_R1"]["terminal_pnl"] == -7 and cmp["M3_minus_M2"]["terminal_pnl"] == 0 and sum(cmp["M3_minus_R1"]["yearly_increments_diff"].values()) == -7
    expect_reject(anchor_m1r1_from_obj, anchors_obj, expected_artifact="other")
    expect_reject(anchor_m1r1_from_obj, anchors_obj, expected_spec_sha="other")
    bad = json.loads(json.dumps(anchors_obj))
    bad["variants"]["R1"]["daily_records"][0]["session"] = "1999-01-01"
    expect_reject(anchor_m1r1_from_obj, bad)
    expect_reject(anchor_calendar_check, all_anchors, days, start + 1, end, counts_s)
    lines = summary_lines(res, all_anchors, cmp, m2, trav, sum_a)
    assert lines and "M3" in lines[1]
    expect_reject(validate_series, ["a", "b"], [100.5, 100], [100, 100])
    expect_reject(validate_series, ["b", "a"], [100, 100], [100, 100])
    expect_reject(validate_series, ["a", "b"], [100, 100], [5, 100])
    d0, d1 = date(2002, 10, 14), date.fromisoformat(CUTOFF)
    span = (d1 - d0).days
    sd_full = [(d0 + timedelta(days=round(k * span / (RETAINED_ROWS - 1)))).isoformat() for k in range(RETAINED_ROWS)] + ["2026-11-06"]
    ok_raw = dict(days=list(sd_full), O=[20000] * FULL_ROWS, C=[20000] * FULL_ROWS)
    ok_adj = dict(days=list(sd_full), C=[20000.0] * FULL_ROWS)
    pd_, pO, pC = prepare_prices(ok_raw, ok_adj)
    assert len(pd_) == RETAINED_ROWS and pd_[-1] == CUTOFF
    expect_reject(prepare_prices, ok_raw, dict(ok_adj, days=list(sd_full[:-1]) + ["2026-11-09"]))
    expect_reject(prepare_prices, dict(ok_raw, O=ok_raw["O"][:-1]), ok_adj)
    return dict(code_sha256_16=sha_full(os.path.abspath(__file__))[:16], here=os.path.dirname(os.path.abspath(__file__)))


def selftest_real_metadata():
    spec = load_spec()
    assert spec["sha256"] == SPEC_SHA256 and check_parents(spec) == 9
    import tempfile
    tmp = tempfile.NamedTemporaryFile(prefix="costband_existing_", suffix=".json", delete=False)
    tmp.write(b"{}")
    tmp.close()
    l0, t0 = LEDGER_CALLS[0], TRAVERSALS[0]
    try:
        expect_reject(run, SPEC, tmp.name)
        assert LEDGER_CALLS[0] == l0 and TRAVERSALS[0] == t0
    finally:
        os.unlink(tmp.name)
    days, O, Cr = load_prices(spec)
    m2obj = json.load(open(spec["parents"]["frozen_M2_result"]["path"], encoding="utf-8"))
    m2obj["_sha256"] = spec["parents"]["frozen_M2_result"]["sha256"]
    t_before = TRAVERSALS[0]
    m2 = m2_from_obj(m2obj, days, folds=spec["folds"], expected_sha=m2obj["_sha256"], expected_spec_sha=spec["parents"]["prior_lgbm_spec"]["sha256"])
    assert TRAVERSALS[0] == t_before and m2["n"] == 2906 and m2["total_trees"] == 2400 and m2["artifact_id"] == M2_ARTIFACT
    a1obj = json.load(open(spec["parents"]["frozen_M1_R1_result"]["path"], encoding="utf-8"))
    a1obj["_sha256"] = spec["parents"]["frozen_M1_R1_result"]["sha256"]
    a1 = anchor_m1r1_from_obj(a1obj, expected_sha=a1obj["_sha256"], expected_spec_sha=spec["parents"]["prior_timing_spec"]["sha256"])
    all_anchors = dict(a1["variants"], M2=dict(metrics=m2["metrics"], daily=m2["daily"]))
    assert anchor_calendar_check(all_anchors, days, FIRST_DECISION, len(days) - 1, spec["counts"]) == 3 and days[FIRST_DECISION] == spec["period"][0] and days[-1] == spec["period"][1]
    assert check_parents(spec) == 9
    return dict(spec_sha256_16=spec["sha256"][:16], m2_decisions=m2["n"], trees=m2["total_trees"], anchors=3, real_data="hashes, metadata, calendar, counts and stored-tree split structure only; no traversal of the 2906 stored predictions, no new features, no M3 ledger")


def selftest():
    info = selftest_synthetic()
    info.update(selftest_real_metadata())
    print("selftest ok", info)


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    elif not sys.argv[1:]:
        run()
    else:
        print("usage: timing_costband.py | selftest")
