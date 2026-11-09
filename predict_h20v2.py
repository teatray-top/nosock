import hashlib
import json
import math
import os
import sys
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction

SPEC = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work\predict_h20_v2_spec_20261106.json")
SPEC_SHA256 = "8560bf53c27758c952ed367023a7edc12bcc6977cecdedf52df329e23bd09f80"
OUT_DIR = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work\predict_h20_v2_20261106")
SCHEMA = "predict-h20-v2"
MODULE = "predict_h20_v2"
SYM = "069500"
CUTOFF = "2026-11-05"
H = 20
WINDOW = 200
POOL_START = 199
ENTRY_START = "2015-01-01"
GRID_ORIGIN = 3031
GRID_STEP = 20
H0, H1, H2 = "H0_zero", "H1_expanding_mean", "H2_ma200_conditional_mean"
PARENT_ORDER = ("raw_prices", "adjusted_prices", "spec", "predict_h20_v1_results")
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
    g = s["grid"]
    if s["version"] != SCHEMA or s["outputs"]["output_dir"] != OUT_DIR or s["outputs"]["overwrite"] or s["resources"]["cpu_threads"] != 1 or list(s["parent_order"]) != list(PARENT_ORDER):
        raise InvalidInput("spec settings differ from module constants")
    if g["origin_decision_index"] != GRID_ORIGIN or g["step_sessions"] != GRID_STEP or g["calendar_facts"]["horizon_sessions"] != H or s["training"]["pool_start_index"] != POOL_START or s["regime_feature"]["window"] != WINDOW or s["source"]["cutoff_session"] != CUTOFF or s["source"]["symbol"] != SYM:
        raise InvalidInput("spec design constants differ from module constants")
    parents = {k: dict(path=s["parents"][k]["path"], sha256=s["parents"][k]["sha256"]) for k in ("raw_prices", "adjusted_prices", "predict_h20_v1_results")}
    parents["spec"] = dict(path=path, sha256=SPEC_SHA256)
    return dict(path=path, sha256=h, parents=parents, facts=g["calendar_facts"], last_grid_decision_index=g["last_grid_decision_index"], last_grid_mature_pool_count=g["last_grid_mature_pool_count"], update_count=s["estimation"]["mature_label_updates"], first_pool=tuple(s["training"]["first_pool_inclusive"]), last_pool=tuple(s["training"]["last_pool_inclusive"]), anchor_artifact_id=s["baselines"]["artifact_id"], external_timeout_seconds=int(s["resources"]["external_timeout_seconds"]))


def check_parents(spec):
    for name in PARENT_ORDER:
        p = spec["parents"][name]
        h = sha_full(p["path"])
        if h != p["sha256"]:
            raise InvalidInput(f"parent {name} hash changed: {h}")
    return len(PARENT_ORDER)


def slice_series(days_all, vals_all, cutoff):
    if len(days_all) != len(vals_all):
        raise InvalidInput("full array length mismatch between days and prices")
    keep = [k for k, d in enumerate(days_all) if d <= cutoff]
    return [days_all[k] for k in keep], [float(vals_all[k]) for k in keep]


def validate_inputs(days, O, C):
    if len(days) != len(O) or len(days) != len(C):
        raise InvalidInput("retained array length mismatch")
    if days != sorted(days) or len(set(days)) != len(days):
        raise InvalidInput("sessions not sorted/unique")
    for o, c in zip(O, C):
        if not (math.isfinite(o) and o > 0 and float(o).is_integer() and math.isfinite(c) and c > 0):
            raise InvalidInput("raw open must be finite positive integral won; adjusted close finite positive")


def load_prices(spec):
    raw = json.load(open(spec["parents"]["raw_prices"]["path"], encoding="utf-8"))[SYM]
    adj = json.load(open(spec["parents"]["adjusted_prices"]["path"], encoding="utf-8"))[SYM]
    rd, O = slice_series(raw["days"], raw["O"], CUTOFF)
    ad, C = slice_series(adj["days"], adj["C"], CUTOFF)
    if rd != ad:
        raise InvalidInput("raw and adjusted retained session lists differ")
    validate_inputs(rd, O, C)
    return rd, O, C


def regimes(C, window=WINDOW):
    n = len(C)
    s = [None] * n
    run = 0.0
    for t in range(n):
        run += C[t]
        if t >= window:
            run -= C[t - window]
        if t >= window - 1:
            s[t] = 1 if C[t] > run / window else 0
    return s


def label(O, j, h=H):
    return O[j + h + 1] / O[j + 1] - 1.0


def grid_indices(n_days, origin, step):
    return list(range(origin, n_days - 1, step))


def engine(days, O, C, h=H, pool_start=POOL_START, origin=GRID_ORIGIN, step=GRID_STEP, window=WINDOW):
    n_days = len(days)
    if pool_start < window - 1 or origin < pool_start:
        raise InvalidInput("pool must start where the regime is defined and before the grid origin")
    grid = grid_indices(n_days, origin, step)
    if not grid:
        raise InvalidInput("grid empty")
    gset = set(grid)
    last = grid[-1]
    s = regimes(C, window)
    g_sum, g_n = 0.0, 0
    c_sum, c_n = [0.0, 0.0], [0, 0]
    updates = 0
    last_label_session = None
    records = []
    for i in range(pool_start, last + 1):
        j = i - h - 1
        if j >= pool_start:
            y = label(O, j, h)
            if not math.isfinite(y) or s[j] is None:
                raise InvalidInput("non-finite label or undefined regime for a pooled label")
            g_sum += y
            g_n += 1
            c_sum[s[j]] += y
            c_n[s[j]] += 1
            updates += 1
            last_label_session = days[j + h + 1]
            if not math.isfinite(g_sum) or not math.isfinite(c_sum[0]) or not math.isfinite(c_sum[1]):
                raise InvalidInput("non-finite running statistic")
        if c_n[0] + c_n[1] != g_n:
            raise InvalidInput("cell invariant violated")
        if i not in gset:
            continue
        si = s[i]
        if si is None:
            raise InvalidInput("undefined regime at a grid decision")
        g_mean = g_sum / g_n if g_n else 0.0
        f_empty = g_n == 0
        f_cell = c_n[si] == 0
        mu = c_sum[si] / c_n[si] if c_n[si] else g_mean
        if not math.isfinite(mu):
            raise InvalidInput("non-finite prediction")
        entry = days[i + 1]
        exit_s = days[i + h + 1] if i + h + 1 < n_days else None
        records.append(dict(symbol=SYM, decision_index=i, decision_session=days[i], block_k=(i - origin) // step, entry_session=entry, exit_session=exit_s, label_available_session=exit_s, target_status="mature_for_scoring" if exit_s else "pending_outcome", decision_at="session_close (logical marker; information through close i)", execution_not_before="next_session_open (logical marker; no order)", s_i=si, mu_hat_H2=mu, n_global=g_n, n_cell0=c_n[0], n_cell1=c_n[1], n_selected=c_n[si], global_mean_for_fallback=g_mean, cell_means={"cell0": c_sum[0] / c_n[0] if c_n[0] else None, "cell1": c_sum[1] / c_n[1] if c_n[1] else None}, fallback_cell_to_global=f_cell, fallback_empty=f_empty, train_max_label_available_session=last_label_session, value_unit="simple_return", moment_order=1, forecast_kind="conditional_mean"))
    labels = [dict(decision_session=r["decision_session"], block_k=r["block_k"], entry_session=r["entry_session"], exit_session=r["exit_session"], realized_simple_price_return=label(O, r["decision_index"], h), label_available_session=r["exit_session"]) for r in records if r["exit_session"]]
    scored = len(labels)
    pend = len(records) - scored
    counts = dict(sessions=n_days, horizon_sessions=h, window=window, pool_start=pool_start, grid_origin=origin, grid_step=step, last_grid_decision_index=last, mature_label_updates=updates, global_statistic_updates=updates, cell_statistic_updates_total=updates, H2_forecast_estimates=dict(warmup=0, off_grid=0, scored=scored, pending=pend, total=len(records)), global_mean_alignment_reads=len(records), prediction_records=len(records), mature_scored_records=scored, pending_outcomes=pend, fallback_cell_to_global=sum(r["fallback_cell_to_global"] for r in records), fallback_empty=sum(r["fallback_empty"] for r in records), last_pool_count=g_n)
    return dict(records=records, labels=labels, counts=counts)


def anchor_from_obj(obj, expected_sha=None, expected_artifact=None):
    if expected_sha is not None and obj.get("_sha256") != expected_sha:
        raise InvalidInput("anchor hash mismatch")
    aid = obj["envelope"]["artifact_id"]
    if expected_artifact is not None and aid != expected_artifact:
        raise InvalidInput("anchor artifact id mismatch")
    recs = obj["prediction_records"]
    seen_d, seen_b = set(), set()
    for r in recs:
        if r["decision_session"] in seen_d or r["block_k"] in seen_b:
            raise InvalidInput("duplicate anchor decision/block")
        seen_d.add(r["decision_session"])
        seen_b.add(r["block_k"])
        for v in (H0, H1):
            if not math.isfinite(float(r["mu_hat"][v])):
                raise InvalidInput("non-finite anchor mu")
    om = {}
    for o in obj["outcome_records"]:
        if o["decision_session"] in om:
            raise InvalidInput("duplicate anchor outcome")
        om[o["decision_session"]] = o
    return dict(artifact_id=aid, records=recs, outcomes=om, n_records=len(recs), n_outcomes=len(om), mse=obj["metrics"]["mse"], blocks_per_entry_year=obj["metrics"]["blocks_per_entry_year"], counts=obj["counts"])


def load_anchor(spec):
    p = spec["parents"]["predict_h20_v1_results"]
    h = sha_full(p["path"])
    if h != p["sha256"]:
        raise InvalidInput("anchor hash changed")
    obj = json.load(open(p["path"], encoding="utf-8"))
    obj["_sha256"] = h
    return anchor_from_obj(obj, expected_sha=h, expected_artifact=spec["anchor_artifact_id"])


def align(res, anchor):
    recs, arecs = res["records"], anchor["records"]
    if len(recs) != anchor["n_records"] or len(res["labels"]) != anchor["n_outcomes"]:
        raise InvalidInput(f"record/outcome counts {len(recs)}/{len(res['labels'])} vs anchor {anchor['n_records']}/{anchor['n_outcomes']}")
    max_mu, max_y = 0.0, 0.0
    for r, a in zip(recs, arecs):
        if (r["decision_index"], r["decision_session"], r["block_k"], r["entry_session"], r["exit_session"], r["label_available_session"], r["target_status"]) != (a["decision_index"], a["decision_session"], a["block_k"], a["entry_session"], a["exit_session"], a["label_available_session"], a["target_status"]):
            raise InvalidInput(f"anchor session/status mismatch at {r['decision_session']}")
        if r["n_global"] != a["n_pool"]:
            raise InvalidInput(f"pool count mismatch at {r['decision_session']}")
        d = abs(r["global_mean_for_fallback"] - float(a["mu_hat"][H1]))
        max_mu = max(max_mu, d)
        if d > TOL:
            raise InvalidInput(f"global mean differs from stored H1 at {r['decision_session']}: {d}")
    for o in res["labels"]:
        a = anchor["outcomes"].get(o["decision_session"])
        if a is None or a["exit_session"] != o["exit_session"] or a["block_k"] != o["block_k"]:
            raise InvalidInput(f"anchor outcome missing/misaligned at {o['decision_session']}")
        d = abs(o["realized_simple_price_return"] - float(a["realized_simple_price_return"]))
        max_y = max(max_y, d)
        if d > TOL:
            raise InvalidInput(f"label differs from stored outcome at {o['decision_session']}: {d}")
    years = {}
    for r in recs:
        if r["exit_session"]:
            years[r["entry_session"][:4]] = years.get(r["entry_session"][:4], 0) + 1
    if years != {k: int(v) for k, v in anchor["blocks_per_entry_year"].items()}:
        raise InvalidInput("blocks per entry year differ from anchor")
    return dict(max_abs_global_mean_minus_stored_H1=max_mu, max_abs_label_minus_stored_outcome=max_y, records=len(recs), outcomes=len(res["labels"]), anchor_artifact_id=anchor["artifact_id"], blocks_per_entry_year=years)


def pop_sd(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def score(res, anchor):
    amap = {a["decision_session"]: a for a in anchor["records"]}
    rows = []
    for r in res["records"]:
        if r["exit_session"] is None:
            continue
        y = float(anchor["outcomes"][r["decision_session"]]["realized_simple_price_return"])
        a = amap[r["decision_session"]]
        rows.append(dict(year=r["entry_session"][:4], y=y, mu={H2: r["mu_hat_H2"], H0: float(a["mu_hat"][H0]), H1: float(a["mu_hat"][H1])}))
    if not rows:
        raise InvalidInput("nothing to score")
    years = sorted({x["year"] for x in rows})
    yr = [x["year"] for x in rows]
    ys = [x["y"] for x in rows]
    sq = {v: [(x["mu"][v] - x["y"]) ** 2 for x in rows] for v in (H2, H0, H1)}

    def mean(v):
        return sum(v) / len(v)

    def by_year(vals):
        return {Y: mean([x for x, g in zip(vals, yr) if g == Y]) for Y in years}

    met = dict(n_scored=len(rows), years=years, blocks_per_entry_year={Y: sum(1 for g in yr if g == Y) for Y in years}, mse={v: mean(sq[v]) for v in (H2, H0, H1)}, mse_by_year={v: by_year(sq[v]) for v in (H2, H0, H1)}, paired={})
    for b, role in ((H1, "primary"), (H0, "secondary")):
        d = [x - z for x, z in zip(sq[H2], sq[b])]
        met["paired"][f"{H2}_minus_{b}"] = dict(mean_sq_error_diff=mean(d), by_entry_year=by_year(d), share_blocks_H2_strictly_better=sum(1 for x in d if x < 0) / len(d), role=role)
    for v in (H0, H1):
        if abs(met["mse"][v] - float(anchor["mse"][v])) > TOL:
            raise InvalidInput(f"loaded {v} MSE differs from pinned h20-v1 metrics")
    mu = [x["mu"][H2] for x in rows]
    elig = [(m, y) for m, y in zip(mu, ys) if m != 0.0 and y != 0.0]
    mse0 = met["mse"][H0]
    met["secondary_H2"] = dict(mse_ratio_to_H0=met["mse"][H2] / mse0 if mse0 else None, uncentered_cross_moment=mean([m * y for m, y in zip(mu, ys)]), centered_covariance=mean([m * y for m, y in zip(mu, ys)]) - mean(mu) * mean(ys), sign_accuracy=(sum(1 for m, y in elig if (m > 0) == (y > 0)) / len(elig)) if elig else None, sign_eligible_count=len(elig), zero_target_count=sum(1 for y in ys if y == 0.0), zero_mu_count=sum(1 for m in mu if m == 0.0), mu_mean=mean(mu), mu_pop_sd=pop_sd(mu))
    first, last = res["records"][0], res["records"][-1]
    met["cell_diagnostics"] = {k: dict(decision_session=r["decision_session"], s_i=r["s_i"], n_cell0=r["n_cell0"], n_cell1=r["n_cell1"], cell_means=r["cell_means"], n_global=r["n_global"]) for k, r in (("first_grid_prediction", first), ("last_grid_prediction", last))}
    met["note"] = "145 dependent blocks; descriptive only; not comparable with 1-session MSE magnitudes; no p-values or CI; cross moments are not PnL; MA200 partition changes sample weighting and proves no timing benefit"
    return met


def fmt_null(x, spec_fmt):
    return format(x, spec_fmt) if x is not None else "해당 없음"


def summary_lines(res, met, al):
    c = res["counts"]
    s = met["secondary_H2"]
    acc = f"{s['sign_accuracy']*100:.1f}% ({s['sign_eligible_count']}건)" if s["sign_accuracy"] is not None else "해당 없음"
    lines = [f"세션 {c['sessions']} | 지평 {c['horizon_sessions']} | 성숙 라벨 갱신 {c['mature_label_updates']} (전역 {c['global_statistic_updates']}·셀 {c['cell_statistic_updates_total']}, 라벨당 1회) | 격자 예측 {c['prediction_records']} (채점 {c['mature_scored_records']}, 미확정 {c['pending_outcomes']}) | H2 추정 워밍업 0·비격자 0 | H1 정렬 확인 {c['global_mean_alignment_reads']} | 대체: 셀→전역 {c['fallback_cell_to_global']}, 전역 빈 {c['fallback_empty']} | 앵커 최대차 전역−저장H1 {al['max_abs_global_mean_minus_stored_H1']:.2e}, 라벨−저장결과 {al['max_abs_label_minus_stored_outcome']:.2e}"]
    lines.append(f"MSE H2 {met['mse'][H2]:.6e} | 저장 H1 {met['mse'][H1]:.6e} | 저장 H0 {met['mse'][H0]:.6e} | 비율 H2/H0 {fmt_null(s['mse_ratio_to_H0'], '.6f')}")
    for k, p in met["paired"].items():
        lines.append(f"{k} [{p['role']}]: 제곱오차 차이 평균 {p['mean_sq_error_diff']:+.3e} (H2 엄격 우세 블록 {p['share_blocks_H2_strictly_better']*100:.1f}%) | 진입연도별: " + ", ".join(f"{y} {x:+.2e} ({met['blocks_per_entry_year'][y]})" for y, x in p["by_entry_year"].items()))
    lines.append(f"H2 μ 평균 {s['mu_mean']:+.3e} sd {s['mu_pop_sd']:.3e} | 비중심 교차모멘트 {s['uncentered_cross_moment']:+.3e} 중심 공분산 {s['centered_covariance']:+.3e} | 부호 적중 {acc} (Y=0 {s['zero_target_count']}, μ=0 {s['zero_mu_count']})")
    for k, d in met["cell_diagnostics"].items():
        lines.append(f"셀 진단 {k} ({d['decision_session']}, s_i={d['s_i']}, n {d['n_global']}): cell0 n {d['n_cell0']} 평균 {fmt_null(d['cell_means']['cell0'], '+.4e')} | cell1 n {d['n_cell1']} 평균 {fmt_null(d['cell_means']['cell1'], '+.4e')}")
    return lines


def build_payload(spec, res, met, al, anchor, code_sha, now_utc):
    artifact_id = f"{MODULE}_{now_utc.strftime('%Y%m%d_%H%M%S')}_{code_sha[:8]}"
    p = spec["parents"]
    env = dict(schema_version=SCHEMA, artifact_id=artifact_id, parent_artifact_hashes=[p[k]["sha256"] for k in PARENT_ORDER], parent_artifact_roles={k: dict(path=p[k]["path"], sha256=p[k]["sha256"]) for k in PARENT_ORDER}, anchor_artifact_id=anchor["artifact_id"], code_sha256=code_sha, spec_sha256=spec["sha256"], asof_utc=now_utc.isoformat(timespec="seconds"), exchange_session=dict(exchange="KRX", symbol=SYM, calendar="raw daily session dates (identical to adjusted)", timezone="Asia/Seoul", utc_mapping="session date D in KST (UTC+09:00); no intraday clock times used", decision_marker="session_close", entry_marker="next_session_open", exit_marker="open of session i+21"), symbol=SYM, availability_status="available", decision_at="session_close (logical marker; information through close i only)", execution_not_before="next_session_open (logical marker; no order placed)", target=dict(definition="Y_j = O[j+21]/O[j+1] - 1, raw open of session j+1 to raw open of session j+21, gross simple price return over 20 held sessions", value_unit="simple_return", moment_order=1, forecast_kind="conditional_mean", available_at="open of session j+21 (exit_session)", excludes="dividends/distributions and all costs"), maturity=dict(rule="pool j >= 199 and j + 21 <= i", update_order="at close i add Y_j for j = i-21 to the global accumulator and to cell s_j (regime at its own decision); emit H2 only on grid decisions; no updates after the last grid decision 5931 (last label j = 5910)"), grid=dict(origin_decision_index=GRID_ORIGIN, step_sessions=GRID_STEP, rule="i = 3031 + 20k, i + 1 < N; grid-only forecasts", nonoverlap="adjacent scored holding intervals share only the boundary open; not independence"), regime_feature=dict(definition="s_t = 1 if adjusted C[t] > mean(adjusted C[t-199..t]) else 0; equality -> 0; undefined before index 199 and never used", window=WINDOW, source="adjusted C only (same as predict-v1); fixed design constant, not tuned"), new_predictor=dict(name=H2, formula="mean of mature labels with s_j == s_i; empty cell -> contemporaneous global mean (fallback_cell_to_global); empty global -> 0 (fallback_empty)", invariants="n_cell0 + n_cell1 == n_global; n_selected == n_cell[s_i]; empty cell mean null"), baselines=dict(policy="H0/H1 forecasts and outcomes loaded verbatim from the pinned predict-h20-v1 artifact; not recomputed; running global mean compared to stored H1 within 1e-15 as an arithmetic alignment check", anchor_artifact_id=anchor["artifact_id"], anchor_sha256=p["predict_h20_v1_results"]["sha256"], alignment=al), training=dict(policy="B: all mature rolling 20-session labels regardless of grid alignment", overlap_caveat="availability makes the estimator causal; overlapping windows remain dependent and weight shared daily moves repeatedly; raw label count is not an effective independent sample size"), feedback="only newly mature labels update global/cell statistics; membership is fixed by s_j at decision j; current s_i selects the forecast cell but never reassigns old labels; outcomes and scores never tune estimates", assumptions="price return excluding distributions and costs; provider quotes not independently reconciled (2026-07-31 jump as supplied); adjusted series is a current snapshot; all evaluation periods already viewed (exploratory)", statistical_learning=dict(optimizer_calls=0, new_estimated_predictors=1, loaded_baselines=2, mature_label_updates=res["counts"]["mature_label_updates"], global_statistic_updates=res["counts"]["global_statistic_updates"], cell_statistic_updates_total=res["counts"]["cell_statistic_updates_total"], H2_forecast_estimates=res["counts"]["H2_forecast_estimates"], global_mean_alignment_reads=res["counts"]["global_mean_alignment_reads"], note="one mature label updates one own-decision cell and the global accumulator (not counted twice); conditional and global means are fitted statistics despite zero optimizer calls"), limitations="145 scored blocks, about 12 per entry year, potential serial dependence; descriptive only; not comparable with 1-day MSE; MA200 partition alone proves no timing benefit; no PnL, calibration, p-values or horizon/offset selection", resources=dict(cpu_threads=1, external_timeout_seconds=spec["external_timeout_seconds"], experiments=1, new_estimated_predictors=1, loaded_baselines=2, optimizer_calls=0, api_calls=0, orders=0, random_paths=0))
    ref = dict(envelope_artifact_id=artifact_id, anchor_artifact_id=anchor["artifact_id"])
    amap = {a["decision_session"]: a for a in anchor["records"]}
    preds = [dict(r, anchor_mu_loaded={H0: float(amap[r["decision_session"]]["mu_hat"][H0]), H1: float(amap[r["decision_session"]]["mu_hat"][H1])}, **ref) for r in res["records"]]
    outs = [dict(anchor["outcomes"][o["decision_session"]], loaded_from_anchor=True, **ref) for o in res["labels"]]
    return dict(envelope=env, counts=res["counts"], alignment=al, metrics=met, lines=summary_lines(res, met, al), prediction_records=preds, outcome_records=outs)


def real_calendar_checks(spec, days):
    n = len(days)
    first = next(i for i in range(POOL_START, n - 1) if days[i + 1] >= ENTRY_START)
    grid = grid_indices(n, GRID_ORIGIN, GRID_STEP)
    scored = [i for i in grid if i + H + 1 <= n - 1]
    pending = [i for i in grid if i + H + 1 > n - 1]
    years = {}
    for i in scored:
        years[days[i + 1][:4]] = years.get(days[i + 1][:4], 0) + 1
    facts = dict(price_sessions=n, horizon_sessions=H, pool_start_index=POOL_START, first_common_training_decision=days[POOL_START], first_common_label_available_session=days[POOL_START + H + 1], grid_origin_decision_index=first, grid_origin_decision=days[GRID_ORIGIN], grid_origin_entry=days[GRID_ORIGIN + 1], grid_origin_exit=days[GRID_ORIGIN + H + 1], grid_step_sessions=GRID_STEP, grid_predictions=len(grid), scored_blocks=len(scored), pending_blocks=len(pending), first_prediction_mature_training_count=(GRID_ORIGIN - H - 1) - POOL_START + 1, last_scored_decision=days[scored[-1]], last_scored_entry=days[scored[-1] + 1], last_scored_exit=days[scored[-1] + H + 1], pending=[dict(decision_session=days[i], entry_session=days[i + 1], exit_session=None, sessions_missing=(i + H + 1) - (n - 1)) for i in pending], blocks_per_entry_year=years, last_grid_decision_index=grid[-1], last_grid_mature_pool_count=(grid[-1] - H - 1) - POOL_START + 1, update_count=(grid[-1] - H - 1) - POOL_START + 1, first_pool=(POOL_START, GRID_ORIGIN - H - 1), last_pool=(POOL_START, grid[-1] - H - 1))
    e = spec["facts"]
    exp = {k: e[k] for k in ("price_sessions", "horizon_sessions", "pool_start_index", "first_common_training_decision", "first_common_label_available_session", "grid_origin_decision_index", "grid_origin_decision", "grid_origin_entry", "grid_origin_exit", "grid_step_sessions", "grid_predictions", "scored_blocks", "pending_blocks", "first_prediction_mature_training_count", "last_scored_decision", "last_scored_entry", "last_scored_exit", "pending", "blocks_per_entry_year")}
    exp["blocks_per_entry_year"] = {k: int(v) for k, v in exp["blocks_per_entry_year"].items()}
    exp.update(last_grid_decision_index=spec["last_grid_decision_index"], last_grid_mature_pool_count=spec["last_grid_mature_pool_count"], update_count=spec["update_count"], first_pool=tuple(spec["first_pool"]), last_pool=tuple(spec["last_pool"]))
    bad = {k: (facts[k], exp[k]) for k in exp if facts[k] != exp[k]}
    if bad:
        raise InvalidInput(f"calendar facts differ from spec: {bad}")
    if first != GRID_ORIGIN:
        raise InvalidInput("grid origin is not the first evaluation decision")
    return facts


def anchor_metadata_checks(anchor, days, facts):
    recs = anchor["records"]
    if len(recs) != facts["grid_predictions"] or anchor["n_outcomes"] != facts["scored_blocks"]:
        raise InvalidInput("anchor counts differ from calendar facts")
    for k, a in enumerate(recs):
        i = GRID_ORIGIN + GRID_STEP * k
        if a["decision_index"] != i or a["block_k"] != k or a["decision_session"] != days[i] or a["entry_session"] != days[i + 1] or a["exit_session"] != (days[i + H + 1] if i + H + 1 < len(days) else None) or a["n_pool"] != (i - H - 1) - POOL_START + 1:
            raise InvalidInput(f"anchor record {k} misaligned with the grid calendar")
    if recs[-1]["target_status"] != "pending_outcome" or any(a["target_status"] != "mature_for_scoring" for a in recs[:-1]):
        raise InvalidInput("anchor pending flags unexpected")
    if {k: int(v) for k, v in anchor["blocks_per_entry_year"].items()} != facts["blocks_per_entry_year"]:
        raise InvalidInput("anchor blocks per entry year differ from calendar facts")
    return dict(records=len(recs), outcomes=anchor["n_outcomes"], first=recs[0]["decision_session"], last=recs[-1]["decision_session"], artifact_id=anchor["artifact_id"])


def run(spec_path=SPEC):
    spec = load_spec(spec_path)
    check_parents(spec)
    days, O, C = load_prices(spec)
    facts = real_calendar_checks(spec, days)
    anchor = load_anchor(spec)
    anchor_metadata_checks(anchor, days, facts)
    res = engine(days, O, C)
    c = res["counts"]
    if c["prediction_records"] != facts["grid_predictions"] or c["mature_scored_records"] != facts["scored_blocks"] or c["mature_label_updates"] != facts["update_count"] or c["last_pool_count"] != facts["last_grid_mature_pool_count"] or res["records"][0]["n_global"] != facts["first_prediction_mature_training_count"]:
        raise InvalidInput("engine counts differ from calendar facts")
    al = align(res, anchor)
    met = score(res, anchor)
    check_parents(spec)
    code_sha = sha_full(os.path.abspath(__file__))
    payload = build_payload(spec, res, met, al, anchor, code_sha, datetime.now(timezone.utc))
    payload["calendar_facts"] = facts
    data = json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False).encode("utf-8")
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "predict_h20_v2_results.json")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; refusing to overwrite")
    with open(out, "xb") as f:
        f.write(data)
    print("\n".join(payload["lines"]))
    print("saved", out, "sha", sha_full(out)[:16], "artifact", payload["envelope"]["artifact_id"], "code", code_sha[:16], "spec", spec["sha256"][:16])
    return payload


def synthetic_days(n, start=date(2015, 1, 1)):
    return [(start + timedelta(days=k)).isoformat() for k in range(n)]


def synthetic_anchor(res, artifact_id="predict_h20_v1_synthetic"):
    recs = [dict(decision_index=r["decision_index"], decision_session=r["decision_session"], block_k=r["block_k"], entry_session=r["entry_session"], exit_session=r["exit_session"], label_available_session=r["label_available_session"], target_status=r["target_status"], n_pool=r["n_global"], mu_hat={H0: 0.0, H1: r["global_mean_for_fallback"]}) for r in res["records"]]
    outs = [dict(o) for o in res["labels"]]
    om = {o["decision_session"]: o for o in outs}
    rows = [(r, om[r["decision_session"]]["realized_simple_price_return"]) for r in recs if r["exit_session"]]
    mse = {v: sum((r["mu_hat"][v] - y) ** 2 for r, y in rows) / len(rows) for v in (H0, H1)} if rows else {H0: 0.0, H1: 0.0}
    years = {}
    for r, _ in rows:
        years[r["entry_session"][:4]] = years.get(r["entry_session"][:4], 0) + 1
    return dict(envelope=dict(artifact_id=artifact_id), prediction_records=recs, outcome_records=outs, metrics=dict(mse=mse, blocks_per_entry_year=years), counts=dict(prediction_records=len(recs)))


def exact_mean(O, js, h):
    ys = [Fraction(int(O[j + h + 1])) / Fraction(int(O[j + 1])) - 1 for j in js]
    return sum(ys) / len(ys) if ys else None


def near(a, b, atol=1e-14, rtol=1e-12):
    return abs(a - float(b)) <= atol + rtol * abs(float(b))


def expect_reject(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except InvalidInput:
        return
    raise AssertionError("expected rejection")


def selftest():
    h, win, ps, org, st = 3, 2, 1, 6, 3
    n = 48
    days = synthetic_days(n)
    O = [float(100 + 7 * ((k * 37) % 23) + 3 * ((k * 11) % 5)) for k in range(n)]
    C = [float(1000 + 9 * ((k * 53) % 41) - 4 * ((k * 7) % 13)) for k in range(n)]
    C[9] = C[8]
    validate_inputs(days, O, C)
    s = regimes(C, win)
    assert s[0] is None and all(s[t] == (1 if C[t] > C[t - 1] else 0) for t in range(1, n)) and s[9] == 0
    res = engine(days, O, C, h=h, pool_start=ps, origin=org, step=st, window=win)
    grid = grid_indices(n, org, st)
    assert [r["decision_index"] for r in res["records"]] == grid and all(r["block_k"] == k for k, r in enumerate(res["records"]))
    attribution_differs = False
    for r in res["records"]:
        i = r["decision_index"]
        js = list(range(ps, i - h))
        assert r["n_global"] == len(js) and r["n_cell0"] + r["n_cell1"] == r["n_global"] and r["n_selected"] == (r["n_cell1"] if r["s_i"] == 1 else r["n_cell0"]) and r["s_i"] == s[i]
        cell_js = [j for j in js if s[j] == s[i]]
        other = [j for j in js if s[j] != s[i]]
        if other and i - h - 1 in other:
            attribution_differs = True
        ex_cell, ex_glob = exact_mean(O, cell_js, h), exact_mean(O, js, h)
        if not js:
            assert r["fallback_empty"] and r["fallback_cell_to_global"] and r["mu_hat_H2"] == 0.0 and r["cell_means"] == {"cell0": None, "cell1": None}
        elif not cell_js:
            assert r["fallback_cell_to_global"] and not r["fallback_empty"] and near(r["mu_hat_H2"], ex_glob) and r["cell_means"]["cell1" if r["s_i"] == 1 else "cell0"] is None
        else:
            assert not r["fallback_cell_to_global"] and near(r["mu_hat_H2"], ex_cell)
        assert near(r["global_mean_for_fallback"], ex_glob) if js else r["global_mean_for_fallback"] == 0.0
        for cell in (0, 1):
            cj = [j for j in js if s[j] == cell]
            m = r["cell_means"]["cell1" if cell else "cell0"]
            assert (m is None) == (not cj) and (m is None or near(m, exact_mean(O, cj, h)))
        assert (r["exit_session"] is None) == (i + h + 1 >= n)
    assert attribution_differs, "fixture must contain a matured label whose own-decision regime differs from the current regime"
    sd, sO = synthetic_days(12), [100.0, 110.0, 121.0, 115.0, 120.0, 126.0, 120.0, 130.0, 128.0, 135.0, 131.0, 140.0]
    sC = [100.0, 110.0, 105.0, 108.0, 112.0, 111.0, 115.0, 120.0, 118.0, 125.0, 124.0, 130.0]
    ss = regimes(sC, win)
    assert ss[1] == 1 and ss[5] == 0
    small = engine(sd, sO, sC, h=h, pool_start=ps, origin=2, step=1, window=win)
    sm = {r["decision_index"]: r for r in small["records"]}
    assert sm[2]["fallback_empty"] and sm[2]["fallback_cell_to_global"] and sm[2]["mu_hat_H2"] == 0.0 and sm[2]["n_global"] == 0
    assert sm[5]["n_global"] == 1 and sm[5]["fallback_cell_to_global"] and not sm[5]["fallback_empty"] and sm[5]["s_i"] == 0 and sm[5]["n_cell1"] == 1 and sm[5]["n_cell0"] == 0 and abs(sm[5]["mu_hat_H2"] - label(sO, 1, h)) < 1e-15 and sm[5]["cell_means"]["cell0"] is None
    assert res["counts"]["mature_label_updates"] == (grid[-1] - h - 1) - ps + 1 and res["counts"]["H2_forecast_estimates"]["total"] == len(grid) and res["counts"]["global_mean_alignment_reads"] == len(grid)
    anc = anchor_from_obj(synthetic_anchor(res))
    al = align(res, anc)
    assert al["records"] == len(grid) and al["max_abs_global_mean_minus_stored_H1"] == 0.0 and al["outcomes"] == len(res["labels"])
    met = score(res, anc)
    assert met["n_scored"] == len(res["labels"]) and met["paired"][f"{H2}_minus_{H1}"]["role"] == "primary" and abs(met["paired"][f"{H2}_minus_{H1}"]["mean_sq_error_diff"] - (met["mse"][H2] - met["mse"][H1])) < 1e-15
    for mutate in (lambda a: a["prediction_records"][4]["mu_hat"].__setitem__(H1, a["prediction_records"][4]["mu_hat"][H1] + 1e-12), lambda a: a["outcome_records"][2].__setitem__("realized_simple_price_return", a["outcome_records"][2]["realized_simple_price_return"] + 1e-12), lambda a: a["prediction_records"][3].__setitem__("entry_session", days[0]), lambda a: a["prediction_records"][3].__setitem__("n_pool", 99), lambda a: a["prediction_records"][-1].__setitem__("target_status", "mature_for_scoring"), lambda a: a["metrics"]["blocks_per_entry_year"].__setitem__("2015", 1), lambda a: a["metrics"]["mse"].__setitem__(H1, 0.5)):
        obj = synthetic_anchor(res)
        mutate(obj)
        try:
            anc_b = anchor_from_obj(obj)
            align(res, anc_b)
            score(res, anc_b)
            raise AssertionError("expected rejection")
        except InvalidInput:
            pass
    obj = synthetic_anchor(res)
    obj["prediction_records"].append(dict(obj["prediction_records"][0]))
    expect_reject(anchor_from_obj, obj)
    obj = synthetic_anchor(res)
    obj["_sha256"] = "00" * 32
    expect_reject(anchor_from_obj, obj, expected_sha="11" * 32)
    expect_reject(anchor_from_obj, synthetic_anchor(res), expected_artifact="other_artifact")

    def keyed(r):
        return {x["decision_index"]: (x["mu_hat_H2"], x["s_i"], x["n_global"], x["n_cell0"], x["n_cell1"], x["global_mean_for_fallback"]) for x in r["records"]}

    kb = keyed(res)
    k = 24
    C_after = list(C)
    for t in range(k + 1, n):
        C_after[t] += 500.0
    pc = keyed(engine(days, O, C_after, h=h, pool_start=ps, origin=org, step=st, window=win))
    assert all(pc[i] == kb[i] for i in kb if i <= k)
    C_k = list(C)
    C_k[k] = C[k - 1] - 1.0 if s[k] == 1 else C[k - 1] + 1.0
    s_k = regimes(C_k, win)
    changed = [t for t in range(1, n) if s_k[t] != s[t]]
    assert changed and all(k <= t <= k + win - 1 for t in changed)
    pk = keyed(engine(days, O, C_k, h=h, pool_start=ps, origin=org, step=st, window=win))
    assert all(pk[i] == kb[i] for i in kb if i < k)
    O_after = list(O)
    for t in range(k + 1, n):
        O_after[t] += 50.0
    po = keyed(engine(days, O_after, C, h=h, pool_start=ps, origin=org, step=st, window=win))
    assert all(po[i] == kb[i] for i in kb if i <= k) and any(po[i] != kb[i] for i in kb if i > k)
    step1 = engine(days, O, C, h=h, pool_start=ps, origin=org, step=1, window=win)
    k1 = keyed(step1)
    O_k = list(O)
    O_k[k] += 50.0
    pk1 = keyed(engine(days, O_k, C, h=h, pool_start=ps, origin=org, step=1, window=win))
    j_num, j_den = k - h - 1, k - 1
    d_num = label(O_k, j_num, h) - label(O, j_num, h)
    d_den = label(O_k, j_den, h) - label(O, j_den, h)
    assert d_num != 0.0 and d_den != 0.0
    assert all(pk1[i] == k1[i] for i in k1 if i < k)
    for i in range(k, k + h):
        base_i, pert_i = k1[i], pk1[i]
        assert pert_i[2] == base_i[2] and abs((pert_i[5] - base_i[5]) - d_num / base_i[2]) < 1e-15
        if s[j_num] == s[i]:
            n_cell = base_i[4] if s[i] == 1 else base_i[3]
            assert abs((pert_i[0] - base_i[0]) - d_num / n_cell) < 1e-15
        else:
            assert pert_i[0] == base_i[0]
    base_e, pert_e = k1[k + h], pk1[k + h]
    assert abs((pert_e[5] - base_e[5]) - (d_num + d_den) / base_e[2]) < 1e-15
    expect_reject(slice_series, ["a", "b", "c"], [1.0, 2.0], "c")
    expect_reject(validate_inputs, ["a", "b"], [100.5, 100.0], [1.0, 1.0])
    expect_reject(validate_inputs, ["b", "a"], [100.0, 100.0], [1.0, 1.0])
    expect_reject(validate_inputs, ["a", "b"], [100.0, -5.0], [1.0, 1.0])
    expect_reject(validate_inputs, ["a", "b"], [100.0, 100.0], [1.0, float("nan")])
    expect_reject(validate_inputs, ["a", "b"], [100.0, 100.0], [1.0])
    expect_reject(engine, days, O, C, h=h, pool_start=0, origin=org, step=st, window=win)
    spec = load_spec()
    assert spec["sha256"] == SPEC_SHA256 and check_parents(spec) == 4
    cdays = synthetic_days(40)
    rc = engine(cdays, [100.0] * 40, [100.0] * 40, h=h, pool_start=ps, origin=org, step=st, window=win)
    assert all(r["s_i"] == 0 for r in rc["records"])
    ac = anchor_from_obj(synthetic_anchor(rc))
    alc = align(rc, ac)
    mc = score(rc, ac)
    assert mc["mse"][H0] == 0.0 and mc["secondary_H2"]["mse_ratio_to_H0"] is None and mc["secondary_H2"]["sign_accuracy"] is None and mc["secondary_H2"]["zero_target_count"] == mc["n_scored"]
    payc = build_payload(spec, rc, mc, alc, ac, "ab" * 32, datetime(2026, 11, 6, 9, 10, 0, tzinfo=timezone.utc))
    assert "해당 없음" in payc["lines"][1] and json.loads(json.dumps(payc, allow_nan=False))["metrics"]["secondary_H2"]["mse_ratio_to_H0"] is None and len(payc["outcome_records"]) == mc["n_scored"] and all(o["loaded_from_anchor"] for o in payc["outcome_records"])
    pay = build_payload(spec, res, met, al, anc, "ef" * 32, datetime(2026, 11, 6, 9, 11, 0, tzinfo=timezone.utc))
    env = pay["envelope"]
    for key in ("schema_version", "artifact_id", "parent_artifact_hashes", "code_sha256", "spec_sha256", "asof_utc", "exchange_session", "symbol", "availability_status", "decision_at", "execution_not_before"):
        assert key in env, key
    assert env["artifact_id"] == "predict_h20_v2_20261106_091100_efefefef" and env["parent_artifact_hashes"] == [spec["parents"][kk]["sha256"] for kk in PARENT_ORDER] and env["anchor_artifact_id"] == "predict_h20_v1_synthetic" and all(r["anchor_mu_loaded"][H1] == r["global_mean_for_fallback"] for r in pay["prediction_records"])
    json.dumps(pay, allow_nan=False)
    rdays, rO, rC = load_prices(spec)
    facts = real_calendar_checks(spec, rdays)
    ranchor = load_anchor(spec)
    meta = anchor_metadata_checks(ranchor, rdays, facts)
    assert meta["records"] == 146 and meta["outcomes"] == 145 and meta["first"] == "2014-12-30" and meta["last"] == "2026-10-28" and meta["artifact_id"] == spec["anchor_artifact_id"] and facts["update_count"] == 5712 and sum(facts["blocks_per_entry_year"].values()) == 145
    assert check_parents(spec) == 4
    print("selftest ok", dict(code_sha256_16=sha_full(os.path.abspath(__file__))[:16], here=os.path.dirname(os.path.abspath(__file__)), spec_sha256_16=spec["sha256"][:16], real_data="hashes, calendar facts incl. yearly blocks, stored-anchor alignment only; no real-data mean, regime or score"))


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    elif not sys.argv[1:]:
        run()
    else:
        print("usage: predict_h20v2.py | selftest")
