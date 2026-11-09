import hashlib
import json
import math
import os
import sys
from datetime import date, datetime, timedelta, timezone

SPEC = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work\predict_v2_spec_20261106.json")
SPEC_SHA256 = "462e9749a2fb8fa306728148045d19188105cb935be1e6ae7e08e5278e59d76f"
OUT_DIR = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work\predict_v2_20261106")
SCHEMA = "predict-v2"
MODULE = "predict_v2"
SYM = "069500"
CUTOFF = "2026-11-05"
POOL_START = 199
ENTRY_START = "2015-01-01"
P3 = "P3_sign_prev_return_conditional_mean"
BASELINES = ("P0_zero", "P1_expanding_mean", "P2_ma200_conditional_mean")
PARENT_ORDER = ("raw_prices", "adjusted_prices", "spec", "predict_v1_results")
TOL = 1e-15


class InvalidInput(SystemExit):
    pass


def sha_full(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_spec(path=SPEC):
    h = sha_full(path)
    if h != SPEC_SHA256:
        raise InvalidInput(f"spec hash {h} != {SPEC_SHA256}; refusing")
    s = json.load(open(path, encoding="utf-8"))
    if s["version"] != SCHEMA or s["outputs"]["output_dir"] != OUT_DIR or s["outputs"]["overwrite"] or s["resources"]["cpu_threads"] != 1:
        raise InvalidInput("spec settings differ from module constants")
    if s["evaluation"]["entry_session_start"] != ENTRY_START or s["source"]["cutoff"] != CUTOFF or s["source"]["symbol"] != SYM or s["features"]["cells"] != 2 or s["features"]["threshold"] != 0 or s["features"]["magnitude_threshold"] is not None or list(s["parent_order"]) != list(PARENT_ORDER):
        raise InvalidInput("spec design constants differ from module constants")
    parents = {k: dict(path=s["parents"][k]["path"], sha256=s["parents"][k]["sha256"]) for k in ("raw_prices", "adjusted_prices", "predict_v1_results")}
    parents["spec"] = dict(path=path, sha256=SPEC_SHA256)
    return dict(path=path, sha256=h, parents=parents, expected=s["evaluation"], external_timeout_seconds=int(s["resources"]["external_timeout_seconds"]))


def check_parents(spec):
    for name in PARENT_ORDER:
        p = spec["parents"][name]
        h = sha_full(p["path"])
        if h != p["sha256"]:
            raise InvalidInput(f"parent {name} hash changed: {h}")
    return len(PARENT_ORDER)


def validate_inputs(days, O, C):
    if len(days) != len(O) or len(days) != len(C):
        raise InvalidInput("length mismatch")
    if days != sorted(days) or len(set(days)) != len(days):
        raise InvalidInput("sessions not sorted/unique")
    for o, c in zip(O, C):
        if not (math.isfinite(o) and math.isfinite(c) and o > 0 and c > 0 and float(o).is_integer()):
            raise InvalidInput("prices must be finite, positive; raw open integral won")


def load_prices(spec):
    raw = json.load(open(spec["parents"]["raw_prices"]["path"], encoding="utf-8"))[SYM]
    adj = json.load(open(spec["parents"]["adjusted_prices"]["path"], encoding="utf-8"))[SYM]
    rd = [d for d in raw["days"] if d <= CUTOFF]
    ad = [d for d in adj["days"] if d <= CUTOFF]
    if rd != ad:
        raise InvalidInput("raw and adjusted session lists differ")
    O = [float(x) for x in raw["O"][:len(rd)]]
    C = [float(x) for x in adj["C"][:len(ad)]]
    validate_inputs(rd, O, C)
    return rd, O, C


def cells(C):
    return [None] + [1 if C[i] > C[i - 1] else 0 for i in range(1, len(C))]


def label(O, j):
    return O[j + 2] / O[j + 1] - 1.0


def engine(days, O, C, pool_start=POOL_START, entry_start=ENTRY_START):
    n = len(days)
    c = cells(C)
    if pool_start < 1:
        raise InvalidInput("pool must start at or after the first defined cell")
    g_sum, g_n = 0.0, 0
    c_sum, c_n = [0.0, 0.0], [0, 0]
    updates, warmup = 0, 0
    last_label_session = None
    records = []
    for i in range(pool_start, n - 1):
        j = i - 2
        if j >= pool_start:
            y = label(O, j)
            g_sum += y
            g_n += 1
            c_sum[c[j]] += y
            c_n[c[j]] += 1
            updates += 1
            last_label_session = days[j + 2]
        if c_n[0] + c_n[1] != g_n:
            raise InvalidInput("sample pool invariant violated")
        ci = c[i]
        g_mean = g_sum / g_n if g_n else 0.0
        f_global = g_n == 0
        f_cell = c_n[ci] == 0
        mu = c_sum[ci] / c_n[ci] if c_n[ci] else g_mean
        entry = days[i + 1]
        if entry < entry_start:
            warmup += 1
            continue
        exit_s = days[i + 2] if i + 2 < n else None
        records.append(dict(symbol=SYM, decision_index=i, decision_session=days[i], entry_session=entry, exit_session=exit_s, decision_at="session_close (logical marker; information through close i)", execution_not_before="next_session_open (logical marker; no order)", label_available_session=exit_s, target_status="mature_for_scoring" if exit_s else "pending_outcome", c_i=ci, mu_hat_P3=mu, n_global=g_n, n_cell0=c_n[0], n_cell1=c_n[1], n_selected=c_n[ci], global_mean_for_fallback=g_mean, cell_means={"cell0": c_sum[0] / c_n[0] if c_n[0] else None, "cell1": c_sum[1] / c_n[1] if c_n[1] else None}, train_max_label_available_session=last_label_session, fallback_cell_to_global=f_cell, fallback_global_empty=f_global, value_unit="simple_return", moment_order=1, forecast_kind="conditional_mean"))
    outcomes = [dict(decision_session=r["decision_session"], entry_session=r["entry_session"], exit_session=r["exit_session"], realized_simple_price_return=label(O, r["decision_index"]), label_available_session=r["exit_session"]) for r in records if r["exit_session"]]
    counts = dict(sessions=n, pool_start=pool_start, matured_label_updates=updates, global_statistic_updates=updates, warmup_decisions=warmup, prediction_records=len(records), mature_scored_records=len(outcomes), pending_outcomes=len(records) - len(outcomes), fallback_cell_to_global=sum(r["fallback_cell_to_global"] for r in records), fallback_global_empty=sum(r["fallback_global_empty"] for r in records), conditional_estimates=dict(warmup=warmup, scored=len(outcomes), pending=len(records) - len(outcomes)))
    return dict(records=records, outcomes=outcomes, counts=counts)


def anchor_from_obj(obj, expected_sha=None):
    if expected_sha is not None and obj.get("_sha256") != expected_sha:
        raise InvalidInput("anchor hash mismatch")
    recs = obj["prediction_records"]
    outs = obj["outcome_records"]
    seen = set()
    for r in recs:
        if r["decision_session"] in seen:
            raise InvalidInput(f"duplicate anchor decision {r['decision_session']}")
        seen.add(r["decision_session"])
        for v in BASELINES:
            if not math.isfinite(float(r["mu_hat"][v])):
                raise InvalidInput("non-finite anchor mu")
    om = {}
    for o in outs:
        if o["decision_session"] in om:
            raise InvalidInput("duplicate anchor outcome")
        om[o["decision_session"]] = o
    return dict(artifact_id=obj["envelope"]["artifact_id"], records=recs, outcomes=om, n_records=len(recs), n_outcomes=len(outs))


def load_anchor(spec):
    p = spec["parents"]["predict_v1_results"]
    h = sha_full(p["path"])
    if h != p["sha256"]:
        raise InvalidInput("anchor hash changed")
    obj = json.load(open(p["path"], encoding="utf-8"))
    obj["_sha256"] = h
    return anchor_from_obj(obj, expected_sha=h)


def align(res, anchor):
    recs, arecs = res["records"], anchor["records"]
    if len(recs) != len(arecs):
        raise InvalidInput(f"record count {len(recs)} != anchor {len(arecs)}")
    max_mu, max_y = 0.0, 0.0
    for r, a in zip(recs, arecs):
        if (r["decision_index"], r["decision_session"], r["entry_session"], r["exit_session"]) != (a["decision_index"], a["decision_session"], a["entry_session"], a["exit_session"]):
            raise InvalidInput(f"anchor session mismatch at {r['decision_session']}")
        if (r["target_status"] == "pending_outcome") != (a["target_status"] == "pending_outcome"):
            raise InvalidInput("pending flag mismatch")
        if r["n_global"] != a["n_samples_P1"]:
            raise InvalidInput(f"pool count mismatch at {r['decision_session']}: {r['n_global']} vs {a['n_samples_P1']}")
        d = abs(r["global_mean_for_fallback"] - float(a["mu_hat"]["P1_expanding_mean"]))
        max_mu = max(max_mu, d)
        if d > TOL:
            raise InvalidInput(f"global mean differs from stored P1 at {r['decision_session']}: {d}")
    for o in res["outcomes"]:
        a = anchor["outcomes"].get(o["decision_session"])
        if a is None or a["exit_session"] != o["exit_session"]:
            raise InvalidInput(f"anchor outcome missing/misaligned at {o['decision_session']}")
        d = abs(o["realized_simple_price_return"] - float(a["realized_simple_price_return"]))
        max_y = max(max_y, d)
        if d > TOL:
            raise InvalidInput(f"target differs from stored outcome at {o['decision_session']}: {d}")
    if len(res["outcomes"]) != anchor["n_outcomes"]:
        raise InvalidInput("outcome count mismatch")
    return dict(max_abs_global_mean_minus_stored_P1=max_mu, max_abs_target_minus_stored_outcome=max_y, records=len(recs), outcomes=len(res["outcomes"]))


def pop_sd(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def score(res, anchor):
    amap = {a["decision_session"]: a for a in anchor["records"]}
    rows = []
    for r in res["records"]:
        if r["exit_session"] is None:
            continue
        a = amap[r["decision_session"]]
        y = anchor["outcomes"][r["decision_session"]]["realized_simple_price_return"]
        rows.append(dict(entry_year=r["entry_session"][:4], y=float(y), mu={P3: r["mu_hat_P3"], **{v: float(a["mu_hat"][v]) for v in BASELINES}}))
    if not rows:
        raise InvalidInput("nothing to score")
    years = sorted({x["entry_year"] for x in rows})
    names = (P3,) + BASELINES
    sq = {v: [(x["mu"][v] - x["y"]) ** 2 for x in rows] for v in names}
    yr = [x["entry_year"] for x in rows]
    ys = [x["y"] for x in rows]

    def mean(v):
        return sum(v) / len(v)

    def by_year(vals):
        return {Y: mean([x for x, g in zip(vals, yr) if g == Y]) for Y in years}

    met = dict(n_scored=len(rows), years=years, mse={v: mean(sq[v]) for v in names}, mse_by_year={v: by_year(sq[v]) for v in names}, paired={}, secondary_P3={})
    for b in BASELINES:
        d = [x - z for x, z in zip(sq[P3], sq[b])]
        met["paired"][f"{P3}_minus_{b}"] = dict(mean_sq_error_diff=mean(d), by_entry_year=by_year(d), share_rows_P3_better=sum(1 for x in d if x < 0) / len(d), role="primary" if b == "P1_expanding_mean" else "secondary")
    mu = [x["mu"][P3] for x in rows]
    elig = [(m, y) for m, y in zip(mu, ys) if m != 0.0 and y != 0.0]
    mse0 = met["mse"]["P0_zero"]
    met["secondary_P3"] = dict(mse_ratio_to_P0=met["mse"][P3] / mse0 if mse0 else None, uncentered_cross_moment=mean([m * y for m, y in zip(mu, ys)]), centered_covariance=mean([m * y for m, y in zip(mu, ys)]) - mean(mu) * mean(ys), sign_accuracy=(sum(1 for m, y in elig if (m > 0) == (y > 0)) / len(elig)) if elig else None, sign_eligible_count=len(elig), zero_target_count=sum(1 for y in ys if y == 0.0), zero_mu_count=sum(1 for m in mu if m == 0.0), mu_mean=mean(mu), mu_pop_sd=pop_sd(mu))
    first, last = res["records"][0], res["records"][-1]
    met["cell_diagnostics"] = {k: dict(decision_session=r["decision_session"], n_cell0=r["n_cell0"], n_cell1=r["n_cell1"], cell_means=r["cell_means"], c_i=r["c_i"]) for k, r in (("first_stored_prediction", first), ("last_stored_prediction", last))}
    met["note"] = "daily errors are dependent; no p-values or CI; cross moments are not covariance-of-returns or PnL; baselines loaded verbatim from the pinned predict-v1 artifact"
    return met


def fmt_null(x, spec_fmt):
    return format(x, spec_fmt) if x is not None else "해당 없음"


def summary_lines(res, met, al):
    c = res["counts"]
    s = met["secondary_P3"]
    lines = [f"세션 {c['sessions']} | 성숙 라벨 갱신 {c['matured_label_updates']} | 워밍업 결정 {c['warmup_decisions']} | 예측 기록 {c['prediction_records']} (채점 {c['mature_scored_records']}, 미확정 {c['pending_outcomes']}) | 대체: 셀→전역 {c['fallback_cell_to_global']}, 전역 빈 {c['fallback_global_empty']} | 앵커 대조 최대차 전역평균−저장P1 {al['max_abs_global_mean_minus_stored_P1']:.2e}, 라벨−저장결과 {al['max_abs_target_minus_stored_outcome']:.2e}"]
    acc = f"{s['sign_accuracy']*100:.1f}% ({s['sign_eligible_count']}건)" if s["sign_accuracy"] is not None else "해당 없음"
    lines.append(f"{P3}: MSE {met['mse'][P3]:.6e} | P0 대비 비율 {fmt_null(s['mse_ratio_to_P0'], '.6f')} | μ 평균 {s['mu_mean']:+.3e} sd {s['mu_pop_sd']:.3e} | 비중심 교차모멘트 {s['uncentered_cross_moment']:+.3e} 중심 공분산 {s['centered_covariance']:+.3e} | 부호 적중 {acc} (y=0 {s['zero_target_count']})")
    lines.append("저장 기준 MSE: " + ", ".join(f"{v} {met['mse'][v]:.6e}" for v in BASELINES))
    for k, p in met["paired"].items():
        lines.append(f"{k} [{p['role']}]: 제곱오차 차이 평균 {p['mean_sq_error_diff']:+.3e} (P3 나은 행 {p['share_rows_P3_better']*100:.1f}%) | 진입연도별: " + ", ".join(f"{y} {x:+.2e}" for y, x in p["by_entry_year"].items()))
    for k, d in met["cell_diagnostics"].items():
        lines.append(f"셀 진단 {k} ({d['decision_session']}, c_i={d['c_i']}): n0 {d['n_cell0']} 평균 {fmt_null(d['cell_means']['cell0'], '+.3e')} | n1 {d['n_cell1']} 평균 {fmt_null(d['cell_means']['cell1'], '+.3e')}")
    return lines


def build_payload(spec, res, met, al, anchor, code_sha, now_utc):
    artifact_id = f"{MODULE}_{now_utc.strftime('%Y%m%d_%H%M%S')}_{code_sha[:8]}"
    p = spec["parents"]
    env = dict(schema_version=SCHEMA, artifact_id=artifact_id, parent_artifact_hashes=[p[k]["sha256"] for k in PARENT_ORDER], parent_artifact_roles={k: dict(path=p[k]["path"], sha256=p[k]["sha256"]) for k in PARENT_ORDER}, anchor_artifact_id=anchor["artifact_id"], code_sha256=code_sha, spec_sha256=spec["sha256"], asof_utc=now_utc.isoformat(timespec="seconds"), exchange_session=dict(exchange="KRX", symbol=SYM, calendar="raw daily session dates (identical to adjusted)", timezone="Asia/Seoul", utc_mapping="session date D in KST (UTC+09:00); no intraday clock times used", decision_marker="session_close", entry_marker="next_session_open", exit_marker="following_session_open"), symbol=SYM, availability_status="available", decision_at="session_close (logical marker; information through close i only)", execution_not_before="next_session_open (logical marker; no order placed)", target=dict(definition="y_j = O[j+2]/O[j+1] - 1, raw open to following raw open, gross simple price return", value_unit="simple_return", moment_order=1, forecast_kind="conditional_mean", available_at="open of session j+2 (exit_session)"), maturity=dict(rule="pool j >= 199 and j+2 <= i; j = i-1 excluded", update_order="at close i incorporate matured label j = i-2 into the cell observed at its own decision j, then compute c_i and emit"), feature=dict(name="latest_completed_adjusted_close_return_sign", definition="c_i = 1 if adjusted C[i] > adjusted C[i-1] else 0; equality -> 0; uses C[i-1], C[i] only", cells=2, threshold=0, pool="common j >= 199, not enlarged"), new_predictor=dict(name=P3, formula="mean of y_j over pool with c_j == c_i; empty cell -> contemporaneous global pool mean (counted); 0 if global pool empty (counted)"), baselines=dict(policy="P0/P1/P2 loaded verbatim from the pinned predict-v1 artifact; not recomputed", anchor_artifact_id=anchor["artifact_id"], anchor_sha256=p["predict_v1_results"]["sha256"], alignment=al), feedback="matured raw-price labels update expanding sufficient statistics by the fixed rule; scores, diagnostics and immature labels never update predictors", assumptions="price return excluding cash distributions/dividends and costs; adjusted series is a current snapshot, not a point-in-time archive; all evaluation years already viewed (exploratory)", statistical_learning=dict(optimizer_fits=0, new_predictors=1, loaded_baselines=3, estimators="expanding conditional cell means plus one global mean kept for fallback/anchor comparison", matured_label_updates=res["counts"]["matured_label_updates"], global_statistic_updates=res["counts"]["global_statistic_updates"], conditional_estimates=res["counts"]["conditional_estimates"]), usage_limit="not combinable with state-vol-v1 h or allocation weights without a separate scope; no exp(mu)-1 conversion", resources=dict(cpu_threads=1, external_timeout_seconds=spec["external_timeout_seconds"], experiments=1, new_predictors=1, loaded_baselines=3, optimizer_fits=0, api_calls=0, orders=0, random_paths=0))
    ref = dict(envelope_artifact_id=artifact_id, anchor_artifact_id=anchor["artifact_id"])
    amap = {a["decision_session"]: a for a in anchor["records"]}
    preds = [dict(r, anchor_mu_loaded={v: float(amap[r["decision_session"]]["mu_hat"][v]) for v in BASELINES}, **ref) for r in res["records"]]
    outs = [dict(o, **ref) for o in res["outcomes"]]
    return dict(envelope=env, counts=res["counts"], alignment=al, metrics=met, lines=summary_lines(res, met, al), prediction_records=preds, outcome_records=outs)


def real_calendar_checks(spec, days):
    n = len(days)
    first = next(i for i in range(POOL_START, n - 1) if days[i + 1] >= ENTRY_START)
    facts = dict(price_sessions=n, first_session=days[0], first_common_training_decision=days[POOL_START], first_common_label_available_session=days[POOL_START + 2], first_decision=days[first], first_entry=days[first + 1], first_exit=days[first + 2], first_prediction_mature_training_count=(first - 2) - POOL_START + 1, prediction_records=(n - 2) - first + 1, mature_scored_records=(n - 3) - first + 1, pending_outcomes=1, last_scored_decision=days[n - 3], last_scored_entry=days[n - 2], last_scored_exit=days[n - 1], pending_decision=days[n - 2], pending_entry=days[n - 1])
    e = spec["expected"]
    exp = dict(price_sessions=5938, first_session="2002-10-14", first_common_training_decision=e["first_common_training_decision"], first_common_label_available_session=e["first_common_label_available_session"], first_decision=e["first_decision"], first_entry=e["first_entry"], first_exit=e["first_exit"], first_prediction_mature_training_count=e["first_prediction_mature_training_count"], prediction_records=e["prediction_records"], mature_scored_records=e["mature_scored_records"], pending_outcomes=e["pending_outcomes"], last_scored_decision=e["last_scored_decision"], last_scored_entry=e["last_scored_entry"], last_scored_exit=e["last_scored_exit"], pending_decision=e["pending"]["decision_session"], pending_entry=e["pending"]["entry_session"])
    bad = {k: (facts[k], exp[k]) for k in exp if facts[k] != exp[k]}
    if bad:
        raise InvalidInput(f"calendar facts differ from spec: {bad}")
    return facts


def anchor_metadata_checks(anchor, days, facts):
    recs = anchor["records"]
    if len(recs) != facts["prediction_records"] or anchor["n_outcomes"] != facts["mature_scored_records"]:
        raise InvalidInput("anchor counts differ from calendar facts")
    first = days.index(facts["first_decision"])
    for k, a in enumerate(recs):
        i = first + k
        if a["decision_index"] != i or a["decision_session"] != days[i] or a["entry_session"] != days[i + 1] or a["exit_session"] != (days[i + 2] if i + 2 < len(days) else None):
            raise InvalidInput(f"anchor record {k} misaligned with calendar")
    if recs[-1]["target_status"] != "pending_outcome" or any(a["target_status"] != "mature_for_scoring" for a in recs[:-1]):
        raise InvalidInput("anchor pending flags unexpected")
    return dict(records=len(recs), outcomes=anchor["n_outcomes"], first=recs[0]["decision_session"], last=recs[-1]["decision_session"])


def run(spec_path=SPEC):
    spec = load_spec(spec_path)
    check_parents(spec)
    days, O, C = load_prices(spec)
    facts = real_calendar_checks(spec, days)
    anchor = load_anchor(spec)
    anchor_metadata_checks(anchor, days, facts)
    res = engine(days, O, C)
    al = align(res, anchor)
    met = score(res, anchor)
    check_parents(spec)
    code_sha = sha_full(os.path.abspath(__file__))
    payload = build_payload(spec, res, met, al, anchor, code_sha, datetime.now(timezone.utc))
    payload["calendar_facts"] = facts
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "predict_v2_results.json")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; refusing to overwrite")
    with open(out, "xb") as f:
        f.write(json.dumps(payload, ensure_ascii=False, indent=1).encode("utf-8"))
    print("\n".join(payload["lines"]))
    print("saved", out, "sha", sha_full(out)[:16], "artifact", payload["envelope"]["artifact_id"], "code", code_sha[:16], "spec", spec["sha256"][:16])
    return payload


def synthetic_days(n, start=date(2015, 1, 1)):
    return [(start + timedelta(days=k)).isoformat() for k in range(n)]


def synthetic_anchor(res, p2_shift=1e-5, artifact_id="predict_v1_synthetic"):
    recs = [dict(decision_index=r["decision_index"], decision_session=r["decision_session"], entry_session=r["entry_session"], exit_session=r["exit_session"], target_status=r["target_status"], n_samples_P1=r["n_global"], mu_hat={"P0_zero": 0.0, "P1_expanding_mean": r["global_mean_for_fallback"], "P2_ma200_conditional_mean": r["global_mean_for_fallback"] + p2_shift}) for r in res["records"]]
    outs = [dict(o) for o in res["outcomes"]]
    return dict(envelope=dict(artifact_id=artifact_id), prediction_records=recs, outcome_records=outs)


def expect_reject(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except InvalidInput:
        return
    raise AssertionError("expected rejection")


def selftest():
    days = synthetic_days(8)
    O = [100.0, 110.0, 121.0, 115.0, 120.0, 126.0, 120.0, 130.0]
    C = [100.0, 110.0, 110.0, 105.0, 120.0, 118.0, 120.0, 130.0]
    assert cells(C) == [None, 1, 0, 0, 1, 0, 1, 1]
    y = [None, 115 / 121 - 1, 120 / 115 - 1, 126 / 120 - 1, 120 / 126 - 1, 130 / 120 - 1]
    res = engine(days, O, C, pool_start=1, entry_start=days[0])
    rec = {r["decision_index"]: r for r in res["records"]}
    tol = 1e-15
    assert sorted(rec) == [1, 2, 3, 4, 5, 6] and res["counts"]["matured_label_updates"] == 4
    for i in (1, 2):
        assert rec[i]["mu_hat_P3"] == 0.0 and rec[i]["fallback_global_empty"] and rec[i]["fallback_cell_to_global"] and rec[i]["cell_means"] == {"cell0": None, "cell1": None}
    assert rec[3]["c_i"] == 0 and rec[3]["n_cell1"] == 1 and rec[3]["n_cell0"] == 0 and rec[3]["fallback_cell_to_global"] and not rec[3]["fallback_global_empty"] and abs(rec[3]["mu_hat_P3"] - y[1]) < tol and abs(rec[3]["cell_means"]["cell1"] - y[1]) < tol and rec[3]["cell_means"]["cell0"] is None
    assert rec[4]["c_i"] == 1 and abs(rec[4]["mu_hat_P3"] - y[1]) < tol and rec[4]["n_cell0"] == 1 and abs(rec[4]["global_mean_for_fallback"] - (y[1] + y[2]) / 2) < tol
    assert rec[5]["c_i"] == 0 and abs(rec[5]["mu_hat_P3"] - (y[2] + y[3]) / 2) < tol
    assert rec[6]["c_i"] == 1 and abs(rec[6]["mu_hat_P3"] - (y[1] + y[4]) / 2) < tol and rec[6]["target_status"] == "pending_outcome" and rec[6]["train_max_label_available_session"] == days[6]
    assert all(r["n_cell0"] + r["n_cell1"] == r["n_global"] and r["n_selected"] == (r["n_cell1"] if r["c_i"] == 1 else r["n_cell0"]) for r in res["records"])
    anc = anchor_from_obj(synthetic_anchor(res))
    al = align(res, anc)
    assert al["records"] == 6 and al["outcomes"] == 5 and al["max_abs_global_mean_minus_stored_P1"] == 0.0
    met = score(res, anc)
    assert met["n_scored"] == 5 and abs(met["mse"]["P0_zero"] - sum(v * v for v in y[1:]) / 5) < tol and met["paired"][f"{P3}_minus_P1_expanding_mean"]["role"] == "primary" and abs(met["paired"][f"{P3}_minus_P1_expanding_mean"]["mean_sq_error_diff"] - (met["mse"][P3] - met["mse"]["P1_expanding_mean"])) < tol
    assert met["cell_diagnostics"]["last_stored_prediction"]["decision_session"] == days[6] and met["cell_diagnostics"]["first_stored_prediction"]["cell_means"] == {"cell0": None, "cell1": None}
    bad = synthetic_anchor(res)
    bad["prediction_records"][3]["mu_hat"]["P1_expanding_mean"] += 1e-12
    expect_reject(align, res, anchor_from_obj(bad))
    bad = synthetic_anchor(res)
    bad["outcome_records"][2]["realized_simple_price_return"] += 1e-12
    expect_reject(align, res, anchor_from_obj(bad))
    bad = synthetic_anchor(res)
    bad["prediction_records"][2]["entry_session"] = days[0]
    expect_reject(align, res, anchor_from_obj(bad))
    bad = synthetic_anchor(res)
    bad["prediction_records"].append(dict(bad["prediction_records"][0]))
    expect_reject(anchor_from_obj, bad)
    bad = synthetic_anchor(res)
    bad["prediction_records"][1]["n_samples_P1"] += 1
    expect_reject(align, res, anchor_from_obj(bad))
    bad = synthetic_anchor(res)
    bad["_sha256"] = "00" * 32
    expect_reject(anchor_from_obj, bad, expected_sha="11" * 32)
    n = 60
    days60 = synthetic_days(n)
    O60 = [float(10000 + 5 * ((k * 37) % 97) + 5 * ((k * 11) % 7)) for k in range(n)]
    C60 = [1000.0 + 3.0 * (((k * 53) % 41) - 20) + 0.5 * k for k in range(n)]
    validate_inputs(days60, O60, C60)
    base = engine(days60, O60, C60, pool_start=1, entry_start=days60[0])

    def keyed(r):
        return {x["decision_index"]: (x["mu_hat_P3"], x["c_i"], x["n_global"], x["n_cell0"], x["n_cell1"], x["global_mean_for_fallback"]) for x in r["records"]}

    kb = keyed(base)
    k = 30
    C_after = list(C60)
    for t in range(k + 1, n):
        C_after[t] += 50.0
    pa = keyed(engine(days60, O60, C_after, pool_start=1, entry_start=days60[0]))
    assert all(pa[i] == kb[i] for i in kb if i <= k)
    Cx = list(C60)
    Cx[k - 1], Cx[k], Cx[k + 1] = 100.0, 101.0, 100.5
    basex = engine(days60, O60, Cx, pool_start=1, entry_start=days60[0])
    cx = cells(Cx)
    assert cx[k] == 1 and cx[k + 1] == 0
    Cy = list(Cx)
    Cy[k] = 99.0
    cy = cells(Cy)
    assert cy[k] == 0 and cy[k + 1] == 1 and all(cy[t] == cx[t] for t in range(1, n) if t not in (k, k + 1))
    pert = engine(days60, O60, Cy, pool_start=1, entry_start=days60[0])
    kx, ky = keyed(basex), keyed(pert)
    assert all(ky[i] == kx[i] for i in kx if i < k) and ky[k][1] != kx[k][1] and ky[k + 1][1] != kx[k + 1][1]
    assert (ky[k + 2][3], ky[k + 2][4]) != (kx[k + 2][3], kx[k + 2][4]) and ky[k + 2][2] == kx[k + 2][2] and (ky[k + 3][3], ky[k + 3][4]) == (kx[k + 3][3], kx[k + 3][4])
    O_after = list(O60)
    for t in range(k + 1, n):
        O_after[t] += 500.0
    po = keyed(engine(days60, O_after, C60, pool_start=1, entry_start=days60[0]))
    assert all(po[i] == kb[i] for i in kb if i <= k) and any(po[i] != kb[i] for i in kb if i > k)
    O_k = list(O60)
    O_k[k] += 500.0
    pk = keyed(engine(days60, O_k, C60, pool_start=1, entry_start=days60[0]))
    assert all(pk[i] == kb[i] for i in kb if i < k) and pk[k] != kb[k]
    rwarm = engine(days60, O60, C60, pool_start=1, entry_start=days60[20])
    assert rwarm["counts"]["warmup_decisions"] == 18 and rwarm["records"][0]["decision_session"] == days60[19] and rwarm["counts"]["matured_label_updates"] == base["counts"]["matured_label_updates"] and keyed(rwarm)[25] == kb[25]
    expect_reject(validate_inputs, ["a", "b"], [100.5, 100.0], [1.0, 1.0])
    expect_reject(validate_inputs, ["b", "a"], [100.0, 100.0], [1.0, 1.0])
    spec = load_spec()
    assert spec["sha256"] == SPEC_SHA256 and check_parents(spec) == 4
    daysc = synthetic_days(8)
    rc = engine(daysc, [100.0] * 8, [100.0] * 8, pool_start=1, entry_start=daysc[0])
    ac = anchor_from_obj(synthetic_anchor(rc))
    alc = align(rc, ac)
    mc = score(rc, ac)
    assert mc["mse"]["P0_zero"] == 0.0 and mc["secondary_P3"]["mse_ratio_to_P0"] is None and mc["secondary_P3"]["sign_accuracy"] is None and mc["secondary_P3"]["zero_target_count"] == 5 and all(r["c_i"] == 0 for r in rc["records"])
    payc = build_payload(spec, rc, mc, alc, ac, "ab" * 32, datetime(2026, 11, 6, 8, 40, 0, tzinfo=timezone.utc))
    assert "해당 없음" in payc["lines"][1] and json.loads(json.dumps(payc))["metrics"]["secondary_P3"]["mse_ratio_to_P0"] is None and len(payc["outcome_records"]) == 5
    anb = anchor_from_obj(synthetic_anchor(base))
    pay = build_payload(spec, base, score(base, anb), align(base, anb), anb, "ef" * 32, datetime(2026, 11, 6, 8, 41, 0, tzinfo=timezone.utc))
    env = pay["envelope"]
    for key in ("schema_version", "artifact_id", "parent_artifact_hashes", "code_sha256", "spec_sha256", "asof_utc", "exchange_session", "symbol", "availability_status", "decision_at", "execution_not_before"):
        assert key in env, key
    assert env["artifact_id"] == "predict_v2_20261106_084100_efefefef" and env["parent_artifact_hashes"] == [spec["parents"][k]["sha256"] for k in PARENT_ORDER] and env["anchor_artifact_id"] == "predict_v1_synthetic"
    assert len(pay["prediction_records"]) == 58 and all(r["anchor_mu_loaded"]["P1_expanding_mean"] == r["global_mean_for_fallback"] and r["anchor_artifact_id"] == "predict_v1_synthetic" for r in pay["prediction_records"])
    json.dumps(pay)
    rdays, rO, rC = load_prices(spec)
    facts = real_calendar_checks(spec, rdays)
    ranchor = load_anchor(spec)
    meta = anchor_metadata_checks(ranchor, rdays, facts)
    assert meta["records"] == 2906 and meta["outcomes"] == 2905 and meta["first"] == "2014-12-30" and meta["last"] == "2026-11-04" and facts["first_prediction_mature_training_count"] == 2831
    assert check_parents(spec) == 4
    print("selftest ok", dict(code_sha256_16=sha_full(os.path.abspath(__file__))[:16], here=os.path.dirname(os.path.abspath(__file__)), spec_sha256_16=spec["sha256"][:16], real_data="hashes, calendar, counts, stored-anchor alignment only; no real-data estimation"))


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    elif not sys.argv[1:]:
        run()
    else:
        print("usage: predict_v2.py | selftest")
