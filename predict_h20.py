import hashlib
import json
import math
import os
import sys
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction

SPEC = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work\predict_h20_v1_spec_20261106.json")
SPEC_SHA256 = "fd4dcd3edc20c2abcfd3f368486f7d19ae0aa45454442c588882f824969379a0"
OUT_DIR = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work\predict_h20_v1_20261106")
SCHEMA = "predict-h20-v1"
MODULE = "predict_h20_v1"
SYM = "069500"
CUTOFF = "2026-11-05"
H = 20
POOL_START = 199
ENTRY_START = "2015-01-01"
GRID_ORIGIN = 3031
GRID_STEP = 20
H0, H1 = "H0_zero", "H1_expanding_mean"
PARENT_ORDER = ("raw_prices", "spec")


class InvalidInput(SystemExit):
    pass


def sha_full(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_spec(path=SPEC):
    h = sha_full(path)
    if h != SPEC_SHA256:
        raise InvalidInput(f"spec hash {h} != {SPEC_SHA256}; refusing")
    s = json.load(open(path, encoding="utf-8"))
    g = s["grid"]
    if s["version"] != SCHEMA or s["outputs"]["output_dir"] != OUT_DIR or s["outputs"]["overwrite"] or s["resources"]["cpu_threads"] != 1 or list(s["parent_order"]) != list(PARENT_ORDER):
        raise InvalidInput("spec settings differ from module constants")
    if g["origin_decision_index"] != GRID_ORIGIN or g["step_sessions"] != GRID_STEP or g["calendar_facts"]["horizon_sessions"] != H or s["training"]["pool_start_index"] != POOL_START or s["source"]["cutoff_session"] != CUTOFF or s["source"]["symbol"] != SYM:
        raise InvalidInput("spec design constants differ from module constants")
    parents = {"raw_prices": dict(path=s["parents"]["raw_prices"]["path"], sha256=s["parents"]["raw_prices"]["sha256"]), "spec": dict(path=path, sha256=SPEC_SHA256)}
    return dict(path=path, sha256=h, parents=parents, facts=g["calendar_facts"], last_grid_decision_index=g["last_grid_decision_index"], last_grid_mature_pool_count=g["last_grid_mature_pool_count"], update_count=s["estimation"]["update_count"], first_pool=tuple(s["training"]["first_pool_inclusive"]), last_pool=tuple(s["training"]["last_pool_inclusive"]), external_timeout_seconds=int(s["resources"]["external_timeout_seconds"]))


def check_parents(spec):
    for name in PARENT_ORDER:
        p = spec["parents"][name]
        h = sha_full(p["path"])
        if h != p["sha256"]:
            raise InvalidInput(f"parent {name} hash changed: {h}")
    return len(PARENT_ORDER)


def validate_inputs(days, O):
    if len(days) != len(O):
        raise InvalidInput("length mismatch")
    if days != sorted(days) or len(set(days)) != len(days):
        raise InvalidInput("sessions not sorted/unique")
    for o in O:
        if not (math.isfinite(o) and o > 0 and float(o).is_integer()):
            raise InvalidInput("raw open must be finite, positive, integral won")


def load_prices(spec):
    raw = json.load(open(spec["parents"]["raw_prices"]["path"], encoding="utf-8"))[SYM]
    days = [d for d in raw["days"] if d <= CUTOFF]
    O = [float(x) for x in raw["O"][:len(days)]]
    validate_inputs(days, O)
    return days, O


def label(O, j, h=H):
    return O[j + h + 1] / O[j + 1] - 1.0


def grid_indices(n_days, origin, step):
    return list(range(origin, n_days - 1, step))


def engine(days, O, h=H, pool_start=POOL_START, origin=GRID_ORIGIN, step=GRID_STEP):
    n_days = len(days)
    grid = grid_indices(n_days, origin, step)
    if not grid or origin < pool_start:
        raise InvalidInput("grid empty or origin before pool start")
    gset = set(grid)
    last = grid[-1]
    s, n = 0.0, 0
    updates = 0
    last_label_session = None
    records = []
    for i in range(pool_start, last + 1):
        j = i - h - 1
        if j >= pool_start:
            y = label(O, j, h)
            if not math.isfinite(y):
                raise InvalidInput("non-finite training label")
            s += y
            n += 1
            updates += 1
            last_label_session = days[j + h + 1]
        if i not in gset:
            continue
        mu1 = s / n if n else 0.0
        if not math.isfinite(mu1):
            raise InvalidInput("non-finite estimate")
        k = (i - origin) // step
        entry = days[i + 1]
        exit_s = days[i + h + 1] if i + h + 1 < n_days else None
        records.append(dict(symbol=SYM, decision_index=i, decision_session=days[i], block_k=k, entry_session=entry, exit_session=exit_s, label_available_session=exit_s, target_status="mature_for_scoring" if exit_s else "pending_outcome", decision_at="session_close (logical marker; information through close i)", execution_not_before="next_session_open (logical marker; no order)", mu_hat={H0: 0.0, H1: mu1}, n_pool=n, fallback_empty=n == 0, train_max_label_available_session=last_label_session, value_unit="simple_return", moment_order=1, forecast_kind="conditional_mean"))
    outcomes = [dict(decision_session=r["decision_session"], block_k=r["block_k"], entry_session=r["entry_session"], exit_session=r["exit_session"], realized_simple_price_return=label(O, r["decision_index"], h), label_available_session=r["exit_session"]) for r in records if r["exit_session"]]
    scored = len(outcomes)
    pend = len(records) - scored
    counts = dict(sessions=n_days, horizon_sessions=h, pool_start=pool_start, grid_origin=origin, grid_step=step, last_grid_decision_index=last, mature_label_updates=updates, H1_estimate_counts=dict(warmup=0, off_grid=0, scored=scored, pending=pend, total=len(records)), H0_forecast_counts=dict(scored=scored, pending=pend, total=len(records)), prediction_records=len(records), mature_scored_records=scored, pending_outcomes=pend, fallback_empty=sum(r["fallback_empty"] for r in records), last_pool_count=n)
    return dict(records=records, outcomes=outcomes, counts=counts)


def pop_sd(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def score(res):
    om = {o["decision_session"]: o["realized_simple_price_return"] for o in res["outcomes"]}
    rows = [(r, om[r["decision_session"]]) for r in res["records"] if r["exit_session"]]
    if not rows:
        raise InvalidInput("nothing to score")
    years = sorted({r["entry_session"][:4] for r, _ in rows})
    yr = [r["entry_session"][:4] for r, _ in rows]
    ys = [y for _, y in rows]
    sq = {v: [(r["mu_hat"][v] - y) ** 2 for r, y in rows] for v in (H0, H1)}

    def mean(v):
        return sum(v) / len(v)

    def by_year(vals):
        return {Y: mean([x for x, g in zip(vals, yr) if g == Y]) for Y in years}

    d = [a - b for a, b in zip(sq[H1], sq[H0])]
    mu = [r["mu_hat"][H1] for r, _ in rows]
    elig = [(m, y) for m, y in zip(mu, ys) if m != 0.0 and y != 0.0]
    mse0 = mean(sq[H0])
    met = dict(n_scored=len(rows), years=years, blocks_per_entry_year={Y: sum(1 for g in yr if g == Y) for Y in years}, mse={H0: mse0, H1: mean(sq[H1])}, mse_by_year={v: by_year(sq[v]) for v in (H0, H1)}, paired_H1_minus_H0=dict(mean_sq_error_diff=mean(d), by_entry_year=by_year(d), share_rows_H1_strictly_better=sum(1 for x in d if x < 0) / len(d)), secondary_H1=dict(mse_ratio_to_H0=mean(sq[H1]) / mse0 if mse0 else None, uncentered_cross_moment=mean([m * y for m, y in zip(mu, ys)]), centered_covariance=mean([m * y for m, y in zip(mu, ys)]) - mean(mu) * mean(ys), sign_accuracy=(sum(1 for m, y in elig if (m > 0) == (y > 0)) / len(elig)) if elig else None, sign_eligible_count=len(elig), zero_target_count=sum(1 for y in ys if y == 0.0), zero_mu_count=sum(1 for m in mu if m == 0.0), mu_mean=mean(mu), mu_pop_sd=pop_sd(mu)), note="145 dependent blocks; descriptive only; not comparable with 1-session MSE magnitudes; no p-values or CI; cross moments are not PnL")
    return met


def fmt_null(x, spec_fmt):
    return format(x, spec_fmt) if x is not None else "해당 없음"


def summary_lines(res, met):
    c = res["counts"]
    s = met["secondary_H1"]
    p = met["paired_H1_minus_H0"]
    acc = f"{s['sign_accuracy']*100:.1f}% ({s['sign_eligible_count']}건)" if s["sign_accuracy"] is not None else "해당 없음"
    lines = [f"세션 {c['sessions']} | 지평 {c['horizon_sessions']} | 성숙 라벨 갱신 {c['mature_label_updates']} | 격자 예측 {c['prediction_records']} (채점 {c['mature_scored_records']}, 미확정 {c['pending_outcomes']}) | H1 추정 워밍업 0·비격자 0 | 빈 풀 대체 {c['fallback_empty']} | 끝 풀 {c['last_pool_count']}"]
    lines.append(f"MSE H0 {met['mse'][H0]:.6e} | H1 {met['mse'][H1]:.6e} | 비율 H1/H0 {fmt_null(s['mse_ratio_to_H0'], '.6f')} | 짝 차이 H1−H0 {p['mean_sq_error_diff']:+.3e} (H1 엄격 우세 블록 {p['share_rows_H1_strictly_better']*100:.1f}%)")
    lines.append("진입연도별 H1−H0: " + ", ".join(f"{y} {x:+.2e} ({met['blocks_per_entry_year'][y]})" for y, x in p["by_entry_year"].items()))
    lines.append(f"H1 μ 평균 {s['mu_mean']:+.3e} sd {s['mu_pop_sd']:.3e} | 비중심 교차모멘트 {s['uncentered_cross_moment']:+.3e} 중심 공분산 {s['centered_covariance']:+.3e} | 부호 적중 {acc} (Y=0 {s['zero_target_count']}, μ=0 {s['zero_mu_count']})")
    return lines


def build_payload(spec, res, met, code_sha, now_utc):
    artifact_id = f"{MODULE}_{now_utc.strftime('%Y%m%d_%H%M%S')}_{code_sha[:8]}"
    p = spec["parents"]
    env = dict(schema_version=SCHEMA, artifact_id=artifact_id, parent_artifact_hashes=[p[k]["sha256"] for k in PARENT_ORDER], parent_artifact_roles={k: dict(path=p[k]["path"], sha256=p[k]["sha256"]) for k in PARENT_ORDER}, code_sha256=code_sha, spec_sha256=spec["sha256"], asof_utc=now_utc.isoformat(timespec="seconds"), exchange_session=dict(exchange="KRX", symbol=SYM, calendar="raw daily session dates", timezone="Asia/Seoul", utc_mapping="session date D in KST (UTC+09:00); no intraday clock times used", decision_marker="session_close", entry_marker="next_session_open", exit_marker="open of session i+21"), symbol=SYM, availability_status="available", decision_at="session_close (logical marker; information through close i only)", execution_not_before="next_session_open (logical marker; no order placed)", target=dict(definition="Y_j = O[j+21]/O[j+1] - 1, raw open of session j+1 to raw open of session j+21, gross simple price return over 20 held sessions", value_unit="simple_return", moment_order=1, forecast_kind="conditional_mean", available_at="open of session j+21 (exit_session)", excludes="dividends/distributions and all costs"), maturity=dict(rule="pool j >= 199 and j + 21 <= i; labels for j in (i-21, i-1] are immature and excluded", update_order="at close i incorporate the newly matured label j = i-21; emit only on grid decisions; no label updates after the last grid decision"), grid=dict(origin_decision_index=GRID_ORIGIN, step_sessions=GRID_STEP, rule="i = 3031 + 20k, i + 1 < N; only grid decisions get forecasts; off-grid sessions update statistics only", nonoverlap="adjacent scored holding intervals share only the boundary open; this does not establish independence across blocks"), training=dict(policy="B: all mature rolling 20-session labels regardless of grid alignment", overlap_caveat="availability makes the estimator causal; overlapping windows remain dependent and weight shared daily moves repeatedly; raw label count is not an effective independent sample size"), variants={H0: "constant 0", H1: "arithmetic mean of all mature labels; 0 if empty (counted)"}, feedback="only mature labels update the expanding sum/count; scores and outcomes never tune the estimate", assumptions="price return excluding distributions and costs; provider quotes not independently reconciled (2026-07-31 jump as supplied); all evaluation periods already viewed (exploratory)", statistical_learning=dict(optimizer_calls=0, new_estimated_predictors=1, mature_label_updates=res["counts"]["mature_label_updates"], H1_estimate_counts=res["counts"]["H1_estimate_counts"], H0_forecast_counts=res["counts"]["H0_forecast_counts"], note="closed-form mean is fitted estimation even with zero optimizer calls"), limitations="145 scored blocks, about 12 per entry year, potential serial dependence; descriptive only; not comparable with 1-day MSE; no PnL, calibration, p-values or horizon/offset selection", resources=dict(cpu_threads=1, external_timeout_seconds=spec["external_timeout_seconds"], experiments=1, variants=2, new_estimated_predictors=1, optimizer_calls=0, api_calls=0, orders=0, random_paths=0))
    ref = dict(envelope_artifact_id=artifact_id)
    return dict(envelope=env, counts=res["counts"], metrics=met, lines=summary_lines(res, met), prediction_records=[dict(r, **ref) for r in res["records"]], outcome_records=[dict(o, **ref) for o in res["outcomes"]])


def real_calendar_checks(spec, days):
    n = len(days)
    first = next(i for i in range(POOL_START, n - 1) if days[i + 1] >= ENTRY_START)
    grid = grid_indices(n, GRID_ORIGIN, GRID_STEP)
    scored = [i for i in grid if i + H + 1 <= n - 1]
    pending = [i for i in grid if i + H + 1 > n - 1]
    facts = dict(price_sessions=n, horizon_sessions=H, pool_start_index=POOL_START, first_common_training_decision=days[POOL_START], first_common_label_available_session=days[POOL_START + H + 1], grid_origin_decision_index=first, grid_origin_decision=days[GRID_ORIGIN], grid_origin_entry=days[GRID_ORIGIN + 1], grid_origin_exit=days[GRID_ORIGIN + H + 1], grid_step_sessions=GRID_STEP, grid_predictions=len(grid), scored_blocks=len(scored), pending_blocks=len(pending), first_prediction_mature_training_count=(GRID_ORIGIN - H - 1) - POOL_START + 1, last_scored_decision=days[scored[-1]], last_scored_entry=days[scored[-1] + 1], last_scored_exit=days[scored[-1] + H + 1], pending=[dict(decision_session=days[i], entry_session=days[i + 1], exit_session=None, sessions_missing=(i + H + 1) - (n - 1)) for i in pending], last_grid_decision_index=grid[-1], last_grid_mature_pool_count=(grid[-1] - H - 1) - POOL_START + 1, update_count=(grid[-1] - H - 1) - POOL_START + 1, first_pool=(POOL_START, GRID_ORIGIN - H - 1), last_pool=(POOL_START, grid[-1] - H - 1))
    e = spec["facts"]
    exp = {k: e[k] for k in ("price_sessions", "horizon_sessions", "pool_start_index", "first_common_training_decision", "first_common_label_available_session", "grid_origin_decision_index", "grid_origin_decision", "grid_origin_entry", "grid_origin_exit", "grid_step_sessions", "grid_predictions", "scored_blocks", "pending_blocks", "first_prediction_mature_training_count", "last_scored_decision", "last_scored_entry", "last_scored_exit", "pending")}
    exp.update(last_grid_decision_index=spec["last_grid_decision_index"], last_grid_mature_pool_count=spec["last_grid_mature_pool_count"], update_count=spec["update_count"], first_pool=tuple(spec["first_pool"]), last_pool=tuple(spec["last_pool"]))
    bad = {k: (facts[k], exp[k]) for k in exp if facts[k] != exp[k]}
    if bad:
        raise InvalidInput(f"calendar facts differ from spec: {bad}")
    if first != GRID_ORIGIN:
        raise InvalidInput("grid origin is not the first evaluation decision")
    return facts


def run(spec_path=SPEC):
    spec = load_spec(spec_path)
    check_parents(spec)
    days, O = load_prices(spec)
    facts = real_calendar_checks(spec, days)
    res = engine(days, O)
    c = res["counts"]
    if c["prediction_records"] != facts["grid_predictions"] or c["mature_scored_records"] != facts["scored_blocks"] or c["mature_label_updates"] != facts["update_count"] or c["last_pool_count"] != facts["last_grid_mature_pool_count"] or res["records"][0]["n_pool"] != facts["first_prediction_mature_training_count"]:
        raise InvalidInput("engine counts differ from calendar facts")
    met = score(res)
    check_parents(spec)
    code_sha = sha_full(os.path.abspath(__file__))
    payload = build_payload(spec, res, met, code_sha, datetime.now(timezone.utc))
    payload["calendar_facts"] = facts
    data = json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False).encode("utf-8")
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "predict_h20_v1_results.json")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; refusing to overwrite")
    with open(out, "xb") as f:
        f.write(data)
    print("\n".join(payload["lines"]))
    print("saved", out, "sha", sha_full(out)[:16], "artifact", payload["envelope"]["artifact_id"], "code", code_sha[:16], "spec", spec["sha256"][:16])
    return payload


def synthetic_days(n, start=date(2015, 1, 1)):
    return [(start + timedelta(days=k)).isoformat() for k in range(n)]


def expect_reject(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except InvalidInput:
        return
    raise AssertionError("expected rejection")


def exact_mean(O, js, h):
    ys = [Fraction(int(O[j + h + 1])) / Fraction(int(O[j + 1])) - 1 for j in js]
    return sum(ys) / len(ys) if ys else Fraction(0)


def selftest():
    h, ps, org, st = 3, 1, 6, 3
    n = 40
    days = synthetic_days(n)
    O = [float(100 + 7 * ((k * 37) % 23) + 3 * ((k * 11) % 5)) for k in range(n)]
    validate_inputs(days, O)
    res = engine(days, O, h=h, pool_start=ps, origin=org, step=st)
    grid = grid_indices(n, org, st)
    assert [r["decision_index"] for r in res["records"]] == grid and all(r["block_k"] == k for k, r in enumerate(res["records"]))
    assert all(i not in set(grid) or True for i in range(n)) and set(r["decision_index"] for r in res["records"]).isdisjoint(set(range(ps, grid[-1] + 1)) - set(grid))
    for r in res["records"]:
        i = r["decision_index"]
        js = [j for j in range(ps, i - h) if j + h + 1 <= i]
        assert js == list(range(ps, i - h)) and r["n_pool"] == len(js)
        assert any(j not in set(grid) for j in js) or not js
        ex = exact_mean(O, js, h)
        assert abs(r["mu_hat"][H1] - float(ex)) <= 1e-14 + 1e-12 * abs(float(ex)) and r["mu_hat"][H0] == 0.0
        assert r["fallback_empty"] == (len(js) == 0)
        assert (r["exit_session"] is None) == (i + h + 1 >= n) and r["entry_session"] == days[i + 1]
    assert res["counts"]["mature_label_updates"] == (grid[-1] - h - 1) - ps + 1 and res["counts"]["H1_estimate_counts"]["warmup"] == 0 and res["counts"]["H1_estimate_counts"]["total"] == len(grid)
    assert res["records"][-1]["target_status"] == "pending_outcome" and res["records"][-2]["target_status"] == "mature_for_scoring"
    small = engine(days[:12], O[:12], h=h, pool_start=ps, origin=2, step=st)
    recs = {r["decision_index"]: r for r in small["records"]}
    assert recs[2]["n_pool"] == 0 and recs[2]["fallback_empty"] and recs[2]["mu_hat"][H1] == 0.0 and recs[5]["n_pool"] == 1 and abs(recs[5]["mu_hat"][H1] - label(O, 1, h)) < 1e-15 and recs[8]["n_pool"] == 4
    met = score(res)
    scored = [r for r in res["records"] if r["exit_session"]]
    assert met["n_scored"] == len(scored) and abs(met["paired_H1_minus_H0"]["mean_sq_error_diff"] - (met["mse"][H1] - met["mse"][H0])) < 1e-15 and sum(met["blocks_per_entry_year"].values()) == len(scored)

    def keyed(r):
        return {x["decision_index"]: (x["mu_hat"][H1], x["n_pool"]) for x in r["records"]}

    kb = keyed(res)
    k = 24
    assert k in set(grid) and (k + h) in set(grid)
    O_after = list(O)
    for t in range(k + 1, n):
        O_after[t] += 50.0
    pa = keyed(engine(days, O_after, h=h, pool_start=ps, origin=org, step=st))
    assert all(pa[i] == kb[i] for i in kb if i <= k) and any(pa[i] != kb[i] for i in kb if i > k)
    O_k = list(O)
    O_k[k] += 50.0
    pk = keyed(engine(days, O_k, h=h, pool_start=ps, origin=org, step=st))
    assert all(pk[i] == kb[i] for i in kb if i < k)
    j_num, j_den = k - h - 1, k - 1
    d_num = label(O_k, j_num, h) - label(O, j_num, h)
    d_den = label(O_k, j_den, h) - label(O, j_den, h)
    assert d_num != 0.0 and d_den != 0.0
    assert abs((pk[k][0] - kb[k][0]) - d_num / kb[k][1]) < 1e-15
    for i in kb:
        if k < i < k + h:
            assert i - h - 1 < j_den and abs((pk[i][0] - kb[i][0]) - d_num / kb[i][1]) < 1e-15
    assert abs((pk[k + h][0] - kb[k + h][0]) - (d_num + d_den) / kb[k + h][1]) < 1e-15
    expect_reject(validate_inputs, ["a", "b"], [100.5, 100.0])
    expect_reject(validate_inputs, ["b", "a"], [100.0, 100.0])
    expect_reject(validate_inputs, ["a", "a"], [100.0, 100.0])
    expect_reject(validate_inputs, ["a", "b", "c"], [100.0, 100.0])
    expect_reject(validate_inputs, ["a", "b"], [100.0, -5.0])
    expect_reject(validate_inputs, ["a", "b"], [100.0, float("inf")])
    expect_reject(engine, days, O, h=h, pool_start=ps, origin=0, step=st)
    spec = load_spec()
    assert spec["sha256"] == SPEC_SHA256 and check_parents(spec) == 2
    cdays = synthetic_days(30)
    rc = engine(cdays, [100.0] * 30, h=h, pool_start=ps, origin=org, step=st)
    mc = score(rc)
    assert mc["mse"][H0] == 0.0 and mc["mse"][H1] == 0.0 and mc["secondary_H1"]["mse_ratio_to_H0"] is None and mc["secondary_H1"]["sign_accuracy"] is None and mc["secondary_H1"]["zero_target_count"] == mc["n_scored"] and rc["records"][-1]["exit_session"] is None
    payc = build_payload(spec, rc, mc, "ab" * 32, datetime(2026, 11, 6, 9, 0, 0, tzinfo=timezone.utc))
    assert "해당 없음" in payc["lines"][1] and json.loads(json.dumps(payc, allow_nan=False))["metrics"]["secondary_H1"]["mse_ratio_to_H0"] is None and len(payc["outcome_records"]) == mc["n_scored"]
    pay = build_payload(spec, res, met, "ef" * 32, datetime(2026, 11, 6, 9, 1, 0, tzinfo=timezone.utc))
    env = pay["envelope"]
    for key in ("schema_version", "artifact_id", "parent_artifact_hashes", "code_sha256", "spec_sha256", "asof_utc", "exchange_session", "symbol", "availability_status", "decision_at", "execution_not_before"):
        assert key in env, key
    assert env["artifact_id"] == "predict_h20_v1_20261106_090100_efefefef" and env["parent_artifact_hashes"] == [spec["parents"][kk]["sha256"] for kk in PARENT_ORDER] and len(pay["prediction_records"]) == len(grid) and all(r["envelope_artifact_id"] == env["artifact_id"] for r in pay["prediction_records"])
    json.dumps(pay, allow_nan=False)
    rdays, rO = load_prices(spec)
    facts = real_calendar_checks(spec, rdays)
    assert facts["grid_predictions"] == 146 and facts["scored_blocks"] == 145 and facts["pending_blocks"] == 1 and facts["first_prediction_mature_training_count"] == 2812 and facts["update_count"] == 5712 and facts["last_grid_decision_index"] == 5931
    assert check_parents(spec) == 2
    print("selftest ok", dict(code_sha256_16=sha_full(os.path.abspath(__file__))[:16], here=os.path.dirname(os.path.abspath(__file__)), spec_sha256_16=spec["sha256"][:16], real_data="hashes, calendar facts, counts only; no real-data mean or score"))


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    elif not sys.argv[1:]:
        run()
    else:
        print("usage: predict_h20.py | selftest")
