import hashlib
import json
import math
import os
import sys
from datetime import date, datetime, timedelta, timezone

SPEC = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work\predict_v1_spec_20261106.json")
SPEC_SHA256 = "4d05b55bdfd1bedb80972388695ecbf8610c6acabcdb309d47267b6b8bf8d747"
OUT_DIR = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work\predict_v1_20261106")
SCHEMA = "predict-v1"
MODULE = "predict_v1"
SYM = "069500"
CUTOFF = "2026-11-05"
WINDOW = 200
ENTRY_START = "2015-01-01"
VARIANTS = ("P0_zero", "P1_expanding_mean", "P2_ma200_conditional_mean")
PARENT_ORDER = ("raw_prices", "adjusted_prices", "spec")


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
    if s["features"]["ma_window"] != WINDOW or s["evaluation"]["entry_session_start"] != ENTRY_START or s["source"]["cutoff"] != CUTOFF or s["source"]["symbol"] != SYM:
        raise InvalidInput("spec design constants differ from module constants")
    parents = {k: dict(path=s["parents"][k]["path"], sha256=s["parents"][k]["sha256"]) for k in ("raw_prices", "adjusted_prices")}
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


def regimes(C, window):
    n = len(C)
    s = [None] * n
    run = 0.0
    for j in range(n):
        run += C[j]
        if j >= window:
            run -= C[j - window]
        if j >= window - 1:
            s[j] = 1 if C[j] > run / window else 0
    return s


def label(O, j):
    return O[j + 2] / O[j + 1] - 1.0


def engine(days, O, C, window=WINDOW, entry_start=ENTRY_START):
    n = len(days)
    w0 = window - 1
    s = regimes(C, window)
    sum_all, n_all = 0.0, 0
    sum_r, n_r = [0.0, 0.0], [0, 0]
    updates, warmup = 0, 0
    last_label_session = None
    records = []
    for i in range(w0, n - 1):
        j = i - 2
        if j >= w0:
            y = label(O, j)
            sum_all += y
            n_all += 1
            sum_r[s[j]] += y
            n_r[s[j]] += 1
            updates += 1
            last_label_session = days[j + 2]
        if n_r[0] + n_r[1] != n_all:
            raise InvalidInput("sample pool invariant violated")
        si = s[i]
        f1 = n_all == 0
        mu1 = sum_all / n_all if n_all else 0.0
        f2 = n_r[si] == 0
        mu2 = sum_r[si] / n_r[si] if n_r[si] else mu1
        entry = days[i + 1]
        if entry < entry_start:
            warmup += 1
            continue
        exit_s = days[i + 2] if i + 2 < n else None
        records.append(dict(symbol=SYM, decision_index=i, decision_session=days[i], entry_session=entry, exit_session=exit_s, decision_at="session_close (logical marker; information through close i)", execution_not_before="next_session_open (logical marker; no order)", label_available_session=exit_s, target_status="mature_for_scoring" if exit_s else "pending_outcome", mu_hat={"P0_zero": 0.0, "P1_expanding_mean": mu1, "P2_ma200_conditional_mean": mu2}, regime_s_i=si, n_samples_P1=n_all, n_samples_regime0=n_r[0], n_samples_regime1=n_r[1], n_samples_P2_selected=n_r[si], train_max_label_available_session=last_label_session, fallback_P1_empty=f1, fallback_P2_to_P1=f2, value_unit="simple_return", moment_order=1, forecast_kind="conditional_mean"))
    outcomes = [dict(decision_session=r["decision_session"], entry_session=r["entry_session"], exit_session=r["exit_session"], realized_simple_price_return=label(O, r["decision_index"]), label_available_session=r["exit_session"]) for r in records if r["exit_session"]]
    return dict(records=records, outcomes=outcomes, counts=dict(sessions=n, window=window, first_regime_index=w0, matured_label_updates=updates, warmup_decisions=warmup, prediction_records=len(records), mature_scored_records=len(outcomes), pending_outcomes=len(records) - len(outcomes), fallback_P1_empty=sum(r["fallback_P1_empty"] for r in records), fallback_P2_to_P1=sum(r["fallback_P2_to_P1"] for r in records), warmup_fallback_note="fallback counts above are over stored prediction records only"))


def pop_sd(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def score(records, outcomes):
    by_dec = {r["decision_session"]: r for r in records}
    rows = [(by_dec[o["decision_session"]], o["realized_simple_price_return"]) for o in outcomes]
    if not rows:
        raise InvalidInput("nothing to score")
    years = sorted({r["entry_session"][:4] for r, _ in rows})
    sq = {v: [(r["mu_hat"][v] - y) ** 2 for r, y in rows] for v in VARIANTS}
    yr = [r["entry_session"][:4] for r, _ in rows]
    ys = [y for _, y in rows]

    def mean_by_year(vals):
        return {Y: sum(x for x, g in zip(vals, yr) if g == Y) / sum(1 for g in yr if g == Y) for Y in years}

    def mean(v):
        return sum(v) / len(v)

    met = dict(n_scored=len(rows), years=years, mse={v: mean(sq[v]) for v in VARIANTS}, mse_by_year={v: mean_by_year(sq[v]) for v in VARIANTS}, paired={}, secondary={})
    for a, b in (("P1_expanding_mean", "P0_zero"), ("P2_ma200_conditional_mean", "P0_zero"), ("P2_ma200_conditional_mean", "P1_expanding_mean")):
        d = [x - z for x, z in zip(sq[a], sq[b])]
        met["paired"][f"{a}_minus_{b}"] = dict(mean_sq_error_diff=mean(d), by_entry_year=mean_by_year(d), share_rows_first_better=sum(1 for x in d if x < 0) / len(d))
    mse0 = met["mse"]["P0_zero"]
    for v in VARIANTS:
        mu = [r["mu_hat"][v] for r, _ in rows]
        elig = [(m, y) for m, y in zip(mu, ys) if m != 0.0 and y != 0.0]
        met["secondary"][v] = dict(mse_ratio_to_P0=met["mse"][v] / mse0 if mse0 else None, uncentered_cross_moment=mean([m * y for m, y in zip(mu, ys)]), centered_covariance=mean([m * y for m, y in zip(mu, ys)]) - mean(mu) * mean(ys), sign_accuracy=(sum(1 for m, y in elig if (m > 0) == (y > 0)) / len(elig)) if elig else None, sign_eligible_count=len(elig), zero_target_count=sum(1 for y in ys if y == 0.0), zero_mu_count=sum(1 for m in mu if m == 0.0), mu_mean=mean(mu), mu_pop_sd=pop_sd(mu))
    met["note"] = "daily errors are dependent; no p-values or CI; cross moments are not covariance-of-returns or PnL"
    return met


def summary_lines(res, met):
    c = res["counts"]
    lines = [f"세션 {c['sessions']} | 성숙 라벨 갱신 {c['matured_label_updates']} | 워밍업 결정 {c['warmup_decisions']} | 예측 기록 {c['prediction_records']} (채점 {c['mature_scored_records']}, 미확정 {c['pending_outcomes']}) | 대체: P1 빈 {c['fallback_P1_empty']}, P2→P1 {c['fallback_P2_to_P1']}"]
    for v in VARIANTS:
        s = met["secondary"][v]
        acc = f"{s['sign_accuracy']*100:.1f}% ({s['sign_eligible_count']}건)" if s["sign_accuracy"] is not None else "해당 없음"
        ratio = f"{s['mse_ratio_to_P0']:.6f}" if s["mse_ratio_to_P0"] is not None else "해당 없음"
        lines.append(f"{v}: MSE {met['mse'][v]:.6e} | P0 대비 비율 {ratio} | μ 평균 {s['mu_mean']:+.3e} sd {s['mu_pop_sd']:.3e} | 비중심 교차모멘트 {s['uncentered_cross_moment']:+.3e} 중심 공분산 {s['centered_covariance']:+.3e} | 부호 적중 {acc} (y=0 {s['zero_target_count']})")
    for k, p in met["paired"].items():
        lines.append(f"{k}: 제곱오차 차이 평균 {p['mean_sq_error_diff']:+.3e} (앞이 나은 행 {p['share_rows_first_better']*100:.1f}%) | 진입연도별: " + ", ".join(f"{y} {x:+.2e}" for y, x in p["by_entry_year"].items()))
    return lines


def build_payload(spec, res, met, code_sha, now_utc):
    artifact_id = f"{MODULE}_{now_utc.strftime('%Y%m%d_%H%M%S')}_{code_sha[:8]}"
    p = spec["parents"]
    env = dict(schema_version=SCHEMA, artifact_id=artifact_id, parent_artifact_hashes=[p[k]["sha256"] for k in PARENT_ORDER], parent_artifact_roles={k: dict(path=p[k]["path"], sha256=p[k]["sha256"]) for k in PARENT_ORDER}, code_sha256=code_sha, spec_sha256=spec["sha256"], asof_utc=now_utc.isoformat(timespec="seconds"), exchange_session=dict(exchange="KRX", symbol=SYM, calendar="raw daily session dates (identical to adjusted)", timezone="Asia/Seoul", utc_mapping="session date D in KST (UTC+09:00); no intraday clock times used", decision_marker="session_close", entry_marker="next_session_open", exit_marker="following_session_open"), symbol=SYM, availability_status="available", decision_at="session_close (logical marker; information through close i only)", execution_not_before="next_session_open (logical marker; no order placed)", target=dict(definition="y_j = O[j+2]/O[j+1] - 1, raw open to following raw open, gross simple price return", value_unit="simple_return", moment_order=1, forecast_kind="conditional_mean", available_at="open of session j+2 (exit_session)"), maturity=dict(rule="at decision close i only pairs with j >= window-1 and j+2 <= i are in the training pool; the j=i-1 pair is excluded because it needs O[i+1]", update_order="at close i incorporate the newly matured label j=i-2 into the regime observed at its own decision j, then compute s_i and emit"), feature=dict(regime="s_j = 1 if adjusted C[j] > mean(adjusted C[j-199..j]) else 0; equality -> 0; undefined before 200 closes", ma_window=WINDOW, role="fixed design constant inherited from ladder R2/R3, not tuned; P1 and P2 share the identical eligible pool"), variants=dict(P0_zero="0", P1_expanding_mean="pool mean; 0 if empty (counted)", P2_ma200_conditional_mean="pool mean within s_j == s_i; P1 mean if the regime has no sample (counted); 0 if P1 also empty"), feedback="matured raw-price labels update expanding sufficient statistics by the fixed rule during warm-up and evaluation; scores, diagnostics and immature labels never update predictors", assumptions="price return excluding cash distributions/dividends and costs; adjusted series is a current snapshot, not a point-in-time archive; all evaluation years already viewed (exploratory)", statistical_learning=dict(optimizer_fits=0, estimators="expanding sample means (P1: 1 per decision, P2: 1 of 2 conditional means per decision)", matured_label_updates=res["counts"]["matured_label_updates"], warmup_decisions=res["counts"]["warmup_decisions"], scored_forecasts=res["counts"]["mature_scored_records"], pending_forecasts=res["counts"]["pending_outcomes"]), usage_limit="not combinable with state-vol-v1 h without a separate scope (different series, horizon and units); no exp(mu)-1 conversion", resources=dict(cpu_threads=1, external_timeout_seconds=spec["external_timeout_seconds"], experiments=1, variants=3, optimizer_fits=0, api_calls=0, orders=0, random_paths=0))
    ref = dict(envelope_artifact_id=artifact_id)
    return dict(envelope=env, counts=res["counts"], metrics=met, lines=summary_lines(res, met), prediction_records=[dict(r, **ref) for r in res["records"]], outcome_records=[dict(o, **ref) for o in res["outcomes"]])


def real_calendar_checks(spec, days):
    n = len(days)
    w0 = WINDOW - 1
    first = next(i for i in range(w0, n - 1) if days[i + 1] >= ENTRY_START)
    facts = dict(price_sessions=n, first_session=days[0], first_common_training_decision=days[w0], first_common_label_available_session=days[w0 + 2], first_decision=days[first], first_entry=days[first + 1], first_exit=days[first + 2], first_prediction_mature_training_count=(first - 2) - w0 + 1, prediction_records=(n - 2) - first + 1, mature_scored_records=(n - 3) - first + 1, pending_outcomes=1, last_scored_decision=days[n - 3], last_scored_entry=days[n - 2], last_scored_exit=days[n - 1], pending_decision=days[n - 2], pending_entry=days[n - 1])
    e = spec["expected"]
    exp = dict(price_sessions=5938, first_session="2002-10-14", first_common_training_decision=e["first_common_training_decision"], first_common_label_available_session=e["first_common_label_available_session"], first_decision=e["first_decision"], first_entry=e["first_entry"], first_exit=e["first_exit"], first_prediction_mature_training_count=e["first_prediction_mature_training_count"], prediction_records=e["prediction_records"], mature_scored_records=e["mature_scored_records"], pending_outcomes=e["pending_outcomes"], last_scored_decision=e["last_scored_decision"], last_scored_entry=e["last_scored_entry"], last_scored_exit=e["last_scored_exit"], pending_decision=e["pending"]["decision_session"], pending_entry=e["pending"]["entry_session"])
    bad = {k: (facts[k], exp[k]) for k in exp if facts[k] != exp[k]}
    if bad:
        raise InvalidInput(f"calendar facts differ from spec: {bad}")
    return facts


def run(spec_path=SPEC):
    spec = load_spec(spec_path)
    check_parents(spec)
    days, O, C = load_prices(spec)
    facts = real_calendar_checks(spec, days)
    res = engine(days, O, C)
    if res["counts"]["prediction_records"] != facts["prediction_records"] or res["counts"]["mature_scored_records"] != facts["mature_scored_records"] or res["records"][0]["n_samples_P1"] != facts["first_prediction_mature_training_count"]:
        raise InvalidInput("engine counts differ from calendar facts")
    met = score(res["records"], res["outcomes"])
    check_parents(spec)
    code_sha = sha_full(os.path.abspath(__file__))
    payload = build_payload(spec, res, met, code_sha, datetime.now(timezone.utc))
    payload["calendar_facts"] = facts
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "predict_v1_results.json")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; refusing to overwrite")
    with open(out, "xb") as f:
        f.write(json.dumps(payload, ensure_ascii=False, indent=1).encode("utf-8"))
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


def selftest():
    days = synthetic_days(8)
    O = [100.0, 110.0, 121.0, 115.0, 120.0, 126.0, 120.0, 130.0]
    C = [100.0, 110.0, 110.0, 105.0, 120.0, 118.0, 120.0, 130.0]
    assert regimes(C, 2) == [None, 1, 0, 0, 1, 0, 1, 1]
    y = [None, 115 / 121 - 1, 120 / 115 - 1, 126 / 120 - 1, 120 / 126 - 1, 130 / 120 - 1]
    res = engine(days, O, C, window=2, entry_start=days[0])
    rec = {r["decision_index"]: r for r in res["records"]}
    assert sorted(rec) == [1, 2, 3, 4, 5, 6] and res["counts"]["matured_label_updates"] == 4 and res["counts"]["warmup_decisions"] == 0
    P1, P2 = "P1_expanding_mean", "P2_ma200_conditional_mean"
    tol = 1e-15
    for i in (1, 2):
        assert rec[i]["mu_hat"][P1] == 0.0 and rec[i]["mu_hat"][P2] == 0.0 and rec[i]["fallback_P1_empty"] and rec[i]["fallback_P2_to_P1"] and rec[i]["n_samples_P1"] == 0 and rec[i]["train_max_label_available_session"] is None
    assert abs(rec[3]["mu_hat"][P1] - y[1]) < tol and abs(rec[3]["mu_hat"][P2] - y[1]) < tol and rec[3]["fallback_P2_to_P1"] and not rec[3]["fallback_P1_empty"] and rec[3]["regime_s_i"] == 0 and rec[3]["n_samples_regime1"] == 1 and rec[3]["n_samples_regime0"] == 0 and rec[3]["train_max_label_available_session"] == days[3]
    assert abs(rec[4]["mu_hat"][P1] - (y[1] + y[2]) / 2) < tol and abs(rec[4]["mu_hat"][P2] - y[1]) < tol and rec[4]["regime_s_i"] == 1 and rec[4]["n_samples_P2_selected"] == 1 and not rec[4]["fallback_P2_to_P1"]
    assert abs(rec[5]["mu_hat"][P1] - (y[1] + y[2] + y[3]) / 3) < tol and abs(rec[5]["mu_hat"][P2] - (y[2] + y[3]) / 2) < tol and rec[5]["regime_s_i"] == 0
    assert abs(rec[6]["mu_hat"][P1] - (y[1] + y[2] + y[3] + y[4]) / 4) < tol and abs(rec[6]["mu_hat"][P2] - (y[1] + y[4]) / 2) < tol and rec[6]["regime_s_i"] == 1 and rec[6]["target_status"] == "pending_outcome" and rec[6]["exit_session"] is None
    assert all(r["n_samples_regime0"] + r["n_samples_regime1"] == r["n_samples_P1"] for r in res["records"])
    assert [o["decision_session"] for o in res["outcomes"]] == [days[i] for i in (1, 2, 3, 4, 5)] and all(abs(o["realized_simple_price_return"] - y[i]) < tol for o, i in zip(res["outcomes"], (1, 2, 3, 4, 5))) and res["outcomes"][0]["label_available_session"] == days[3]
    assert rec[2]["entry_session"] == days[3] and rec[2]["exit_session"] == days[4] and rec[2]["decision_session"] == days[2]
    met = score(res["records"], res["outcomes"])
    assert met["n_scored"] == 5 and abs(met["mse"]["P0_zero"] - sum(v * v for v in y[1:]) / 5) < tol and met["years"] == ["2015"]
    assert abs(met["paired"]["P1_expanding_mean_minus_P0_zero"]["mean_sq_error_diff"] - (met["mse"][P1] - met["mse"]["P0_zero"])) < tol
    assert met["secondary"]["P0_zero"]["sign_accuracy"] is None and met["secondary"]["P0_zero"]["zero_mu_count"] == 5 and met["secondary"][P1]["sign_eligible_count"] == 3
    Oz = [100.0, 110.0, 121.0, 121.0, 120.0, 126.0, 120.0, 130.0]
    rz = engine(days, Oz, C, window=2, entry_start=days[0])
    mz = score(rz["records"], rz["outcomes"])
    assert mz["secondary"][P1]["zero_target_count"] == 1 and mz["secondary"][P1]["sign_eligible_count"] == 2
    n = 60
    days60 = synthetic_days(n)
    O60 = [float(10000 + 5 * ((k * 37) % 97) + 5 * ((k * 11) % 7)) for k in range(n)]
    C60 = [O60[k] * 1.001 + 0.3 * (((k * 53) % 41) - 20) for k in range(n)]
    validate_inputs(days60, O60, C60)
    base = engine(days60, O60, C60, window=2, entry_start=days60[0])

    def keyed(r):
        return {x["decision_index"]: (x["mu_hat"], x["regime_s_i"], x["n_samples_P1"], x["n_samples_regime0"], x["n_samples_regime1"], x["fallback_P2_to_P1"]) for x in r["records"]}

    k = 30
    O_f = list(O60)
    for t in range(k + 1, n):
        O_f[t] += 500.0
    pf = keyed(engine(days60, O_f, C60, window=2, entry_start=days60[0]))
    kb = keyed(base)
    assert all(pf[i] == kb[i] for i in kb if i <= k) and any(pf[i] != kb[i] for i in kb if i > k)
    O_k = list(O60)
    O_k[k] += 500.0
    pk = keyed(engine(days60, O_k, C60, window=2, entry_start=days60[0]))
    assert all(pk[i] == kb[i] for i in kb if i < k) and pk[k] != kb[k]
    C_f = list(C60)
    for t in range(k + 1, n):
        C_f[t] *= 1.5
    pc = keyed(engine(days60, O_f, C_f, window=2, entry_start=days60[0]))
    assert all(pc[i] == kb[i] for i in kb if i <= k)
    rwarm = engine(days60, O60, C60, window=2, entry_start=days60[20])
    assert rwarm["counts"]["warmup_decisions"] == 18 and rwarm["records"][0]["decision_session"] == days60[19] and rwarm["records"][0]["entry_session"] == days60[20] and rwarm["counts"]["matured_label_updates"] == base["counts"]["matured_label_updates"] and keyed(rwarm)[25] == kb[25]
    expect_reject(validate_inputs, ["a", "b"], [100.5, 100.0], [1.0, 1.0])
    expect_reject(validate_inputs, ["b", "a"], [100.0, 100.0], [1.0, 1.0])
    spec = load_spec()
    assert spec["sha256"] == SPEC_SHA256 and check_parents(spec) == 3
    daysc = synthetic_days(8)
    rc = engine(daysc, [100.0] * 8, [100.0] * 8, window=2, entry_start=daysc[0])
    mc = score(rc["records"], rc["outcomes"])
    assert mc["mse"]["P0_zero"] == 0.0 and mc["secondary"][P1]["mse_ratio_to_P0"] is None and mc["secondary"][P1]["sign_accuracy"] is None and mc["secondary"][P1]["zero_target_count"] == 5 and all(r["regime_s_i"] == 0 for r in rc["records"])
    payc = build_payload(spec, rc, mc, "ab" * 32, datetime(2026, 11, 6, 8, 31, 0, tzinfo=timezone.utc))
    assert "해당 없음" in payc["lines"][1] and json.loads(json.dumps(payc))["metrics"]["secondary"][P1]["mse_ratio_to_P0"] is None
    pay = build_payload(spec, base, score(base["records"], base["outcomes"]), "ef" * 32, datetime(2026, 11, 6, 8, 30, 0, tzinfo=timezone.utc))
    env = pay["envelope"]
    for key in ("schema_version", "artifact_id", "parent_artifact_hashes", "code_sha256", "spec_sha256", "asof_utc", "exchange_session", "symbol", "availability_status", "decision_at", "execution_not_before"):
        assert key in env, key
    assert env["artifact_id"] == "predict_v1_20261106_083000_efefefef" and env["parent_artifact_hashes"] == [spec["parents"]["raw_prices"]["sha256"], spec["parents"]["adjusted_prices"]["sha256"], SPEC_SHA256] and len(pay["prediction_records"]) == 58 and len(pay["outcome_records"]) == 57 and all(r["envelope_artifact_id"] == env["artifact_id"] for r in pay["prediction_records"])
    json.dumps(pay)
    rdays, rO, rC = load_prices(spec)
    facts = real_calendar_checks(spec, rdays)
    assert facts["prediction_records"] == 2906 and facts["mature_scored_records"] == 2905 and facts["first_prediction_mature_training_count"] == 2831 and facts["first_decision"] == "2014-12-30"
    assert check_parents(spec) == 3
    print("selftest ok", dict(code_sha256_16=sha_full(os.path.abspath(__file__))[:16], here=os.path.dirname(os.path.abspath(__file__)), spec_sha256_16=spec["sha256"][:16], real_data="hashes, calendar alignment, counts, maturity indices only; no real-data estimation"))


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    elif not sys.argv[1:]:
        run()
    else:
        print("usage: predict_v1.py | selftest")
