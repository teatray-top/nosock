import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction

import numpy as np

SPEC = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work\predict_h20_v3_spec_20261106.json")
SPEC_SHA256 = "97c64470b1ff4116866c886fdfcc544204a3af357929817c6dc3ed4a063112a6"
OUT_DIR = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work\predict_h20_v3_20261106")
SCHEMA = "predict-h20-v3"
MODULE = "predict_h20_v3"
SYM = "069500"
CUTOFF = "2026-11-05"
H = 20
POOL_START = 199
ENTRY_START = "2015-01-01"
GRID_ORIGIN = 3031
GRID_STEP = 20
INIT_INDEX = 252
INIT_N = 252
LAMBDA = 0.94
CAL_END_MARKER = "2014-12-31"
CAL_LAST_INDEX = 3031
H0, H1, H3 = "H0_zero", "H1_expanding_mean", "H3_volstate_conditional_mean"
PARENT_ORDER = ("raw_prices", "adjusted_prices", "state_vol_v1_spec", "state_vol_v1_results", "predict_h20_v1_results", "spec")
TOL = 1e-15
STATE_TOL = 1e-18
CAVEAT = "v[j] for labels before the first evaluation decision applies h_ref fitted through 2014-12-30 retrospectively; such membership was not known at the old close j. Preprocessing fitted as of close 3031 and frozen; no historical availability claim."


class InvalidInput(SystemExit):
    pass


def sha_full(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_spec(path=SPEC):
    h = sha_full(path)
    if h != SPEC_SHA256:
        raise InvalidInput(f"spec hash {h} != {SPEC_SHA256}; refusing")
    s = json.load(open(path, encoding="utf-8"))
    g, sf = s["grid"], s["state_feature"]
    if s["version"] != SCHEMA or s["outputs"]["output_dir"] != OUT_DIR or s["outputs"]["overwrite"] or s["resources"]["cpu_threads"] != 1 or list(s["parent_order"]) != list(PARENT_ORDER):
        raise InvalidInput("spec settings differ from module constants")
    if g["origin_decision_index"] != GRID_ORIGIN or g["step_sessions"] != GRID_STEP or g["calendar_facts"]["horizon_sessions"] != H or s["training"]["pool_start_index"] != POOL_START or s["source"]["cutoff_session"] != CUTOFF or s["source"]["symbol"] != SYM:
        raise InvalidInput("spec design constants differ from module constants")
    if sf["initial_index"] != INIT_INDEX or sf["initial_n"] != INIT_N or sf["lambda_"] != LAMBDA or sf["calibration_end_marker"] != CAL_END_MARKER or sf["calibration_last_index"] != CAL_LAST_INDEX:
        raise InvalidInput("state constants differ from module constants")
    parents = {k: dict(path=s["parents"][k]["path"], sha256=s["parents"][k]["sha256"]) for k in ("raw_prices", "adjusted_prices", "state_vol_v1_spec", "state_vol_v1_results", "predict_h20_v1_results")}
    parents["spec"] = dict(path=path, sha256=SPEC_SHA256)
    return dict(path=path, sha256=h, parents=parents, facts=g["calendar_facts"], last_grid_decision_index=g["last_grid_decision_index"], last_grid_mature_pool_count=g["last_grid_mature_pool_count"], update_count=s["estimation"]["mature_label_updates"], defined_updates=s["estimation"]["defined_state_label_updates"], undefined_updates=s["estimation"]["undefined_state_label_updates"], state_recursion_updates=s["estimation"]["state_recursion_updates"], stored_state_comparisons=s["estimation"]["stored_state_alignment_comparisons"], first_pool=tuple(s["training"]["first_pool_inclusive"]), last_pool=tuple(s["training"]["last_pool_inclusive"]), anchor_artifact_id=s["baselines"]["artifact_id"], h_initial=float(sf["h_initial"]), h_ref=float(sf["h_ref"]), external_timeout_seconds=int(s["resources"]["external_timeout_seconds"]))


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


def log_returns(C):
    arr = np.array(C, dtype=float)
    return np.log(arr[1:] / arr[:-1])


def calibration_constants(r, cal_last_index=CAL_LAST_INDEX, init_n=INIT_N):
    h_ref = float(np.mean(r[:cal_last_index] ** 2))
    h_initial = float(np.mean(r[:init_n] ** 2))
    return h_ref, h_initial


def replay_state(r, n_prices, h_initial, init_index=INIT_INDEX, lam=LAMBDA, end=None):
    end = n_prices - 2 if end is None else end
    if init_index < 1 or end < init_index or end > n_prices - 1:
        raise InvalidInput("state replay bounds invalid")
    hv = h_initial
    h = [None] * n_prices
    h[init_index] = float(hv)
    updates = 0
    for i in range(init_index + 1, end + 1):
        x = r[i - 1]
        hv = lam * hv + (1 - lam) * x ** 2
        if not math.isfinite(hv) or hv < 0:
            raise InvalidInput("non-finite or negative state")
        h[i] = float(hv)
        updates += 1
    return h, updates


def states(h, h_ref):
    return [None if hv is None else (1 if hv > h_ref else 0) for hv in h]


def label(O, j, h=H):
    return O[j + h + 1] / O[j + 1] - 1.0


def grid_indices(n_days, origin, step):
    return list(range(origin, n_days - 1, step))


def engine(days, O, v, h_vals, h=H, pool_start=POOL_START, origin=GRID_ORIGIN, step=GRID_STEP):
    n_days = len(days)
    grid = grid_indices(n_days, origin, step)
    if not grid or origin < pool_start:
        raise InvalidInput("grid empty or origin before pool start")
    gset = set(grid)
    last = grid[-1]
    g_sum, g_n = 0.0, 0
    c_sum, c_n = {0: 0.0, 1: 0.0, "undefined": 0.0}, {0: 0, 1: 0, "undefined": 0}
    updates, defined_updates, undefined_updates = 0, 0, 0
    last_label_session = None
    records = []
    for i in range(pool_start, last + 1):
        j = i - h - 1
        if j >= pool_start:
            y = label(O, j, h)
            if not math.isfinite(y):
                raise InvalidInput("non-finite label")
            g_sum += y
            g_n += 1
            cell = v[j] if v[j] is not None else "undefined"
            c_sum[cell] += y
            c_n[cell] += 1
            updates += 1
            if cell == "undefined":
                undefined_updates += 1
            else:
                defined_updates += 1
            last_label_session = days[j + h + 1]
            if not math.isfinite(g_sum) or not all(math.isfinite(x) for x in c_sum.values()):
                raise InvalidInput("non-finite running statistic")
        if c_n[0] + c_n[1] + c_n["undefined"] != g_n:
            raise InvalidInput("cell invariant violated")
        if i not in gset:
            continue
        vi = v[i]
        if vi is None:
            raise InvalidInput("undefined state at a grid decision")
        g_mean = g_sum / g_n if g_n else 0.0
        f_empty = g_n == 0
        f_cell = c_n[vi] == 0
        mu = c_sum[vi] / c_n[vi] if c_n[vi] else g_mean
        if not math.isfinite(mu):
            raise InvalidInput("non-finite prediction")
        entry = days[i + 1]
        exit_s = days[i + h + 1] if i + h + 1 < n_days else None
        records.append(dict(symbol=SYM, decision_index=i, decision_session=days[i], block_k=(i - origin) // step, entry_session=entry, exit_session=exit_s, label_available_session=exit_s, target_status="mature_for_scoring" if exit_s else "pending_outcome", decision_at="session_close (logical marker; information through close i)", execution_not_before="next_session_open (logical marker; no order)", h_i=h_vals[i], v_i=vi, mu_hat_H3=mu, n_global=g_n, n_cell0=c_n[0], n_cell1=c_n[1], n_undefined=c_n["undefined"], n_selected=c_n[vi], global_mean_for_fallback=g_mean, cell_means={"cell0": c_sum[0] / c_n[0] if c_n[0] else None, "cell1": c_sum[1] / c_n[1] if c_n[1] else None, "undefined": c_sum["undefined"] / c_n["undefined"] if c_n["undefined"] else None}, fallback_cell_to_global=f_cell, fallback_empty=f_empty, train_max_label_available_session=last_label_session, calibration_caveat=CAVEAT, value_unit="simple_return", moment_order=1, forecast_kind="conditional_mean"))
    labels = [dict(decision_session=r["decision_session"], block_k=r["block_k"], entry_session=r["entry_session"], exit_session=r["exit_session"], realized_simple_price_return=label(O, r["decision_index"], h), label_available_session=r["exit_session"]) for r in records if r["exit_session"]]
    scored = len(labels)
    pend = len(records) - scored
    counts = dict(sessions=n_days, horizon_sessions=h, pool_start=pool_start, grid_origin=origin, grid_step=step, last_grid_decision_index=last, mature_label_updates=updates, global_statistic_updates=updates, cell_statistic_updates_total=updates, defined_state_label_updates=defined_updates, undefined_state_label_updates=undefined_updates, H3_forecast_estimates=dict(warmup=0, off_grid=0, scored=scored, pending=pend, total=len(records)), global_mean_alignment_reads=len(records), prediction_records=len(records), mature_scored_records=scored, pending_outcomes=pend, fallback_cell_to_global=sum(r["fallback_cell_to_global"] for r in records), fallback_empty=sum(r["fallback_empty"] for r in records), last_pool_count=g_n)
    return dict(records=records, labels=labels, counts=counts)


def h20_anchor_from_obj(obj, expected_sha=None, expected_artifact=None):
    if expected_sha is not None and obj.get("_sha256") != expected_sha:
        raise InvalidInput("h20 anchor hash mismatch")
    aid = obj["envelope"]["artifact_id"]
    if expected_artifact is not None and aid != expected_artifact:
        raise InvalidInput("h20 anchor artifact id mismatch")
    recs = obj["prediction_records"]
    seen_d, seen_b = set(), set()
    for r in recs:
        if r["decision_session"] in seen_d or r["block_k"] in seen_b:
            raise InvalidInput("duplicate h20 anchor decision/block")
        seen_d.add(r["decision_session"])
        seen_b.add(r["block_k"])
        for vname in (H0, H1):
            if not math.isfinite(float(r["mu_hat"][vname])):
                raise InvalidInput("non-finite h20 anchor mu")
    om = {}
    for o in obj["outcome_records"]:
        if o["decision_session"] in om:
            raise InvalidInput("duplicate h20 anchor outcome")
        om[o["decision_session"]] = o
    return dict(artifact_id=aid, records=recs, outcomes=om, n_records=len(recs), n_outcomes=len(om), mse=obj["metrics"]["mse"], blocks_per_entry_year=obj["metrics"]["blocks_per_entry_year"])


def state_anchor_from_obj(obj, expected_sha=None):
    if expected_sha is not None and obj.get("_sha256") != expected_sha:
        raise InvalidInput("state anchor hash mismatch")
    env = obj["envelope"]
    recs = obj["forecast_records"]
    h_ref_env = float(env["reference"]["h"])
    if not math.isfinite(h_ref_env) or h_ref_env < 0:
        raise InvalidInput("state anchor envelope h_ref invalid")
    seen = set()
    fmap = {}
    for r in recs:
        d = r["decision_session"]
        if d in seen:
            raise InvalidInput("duplicate state anchor decision")
        seen.add(d)
        he = float(r["h_forecast"]["ewma"])
        href = float(r["h_forecast"]["reference"])
        if not math.isfinite(he) or r.get("value_unit") != "log_return" or r.get("moment_order") != 2:
            raise InvalidInput("state anchor record invalid (value/units/order)")
        if not math.isfinite(href) or href != h_ref_env:
            raise InvalidInput(f"state anchor record reference differs from envelope h_ref at {d}")
        fmap[d] = dict(predicted_for_session=r["predicted_for_session"], h=he)
    return dict(artifact_id=env["artifact_id"], h_ref=float(env["reference"]["h"]), h_initial=float(env["candidate"]["h_initial"]), lambda_=float(env["candidate"]["lambda_"]), initial_n=int(env["candidate"]["initial_n"]), forecasts=fmap, n_records=len(recs))


def load_anchors(spec):
    p = spec["parents"]["predict_h20_v1_results"]
    h = sha_full(p["path"])
    if h != p["sha256"]:
        raise InvalidInput("h20 anchor hash changed")
    obj = json.load(open(p["path"], encoding="utf-8"))
    obj["_sha256"] = h
    a20 = h20_anchor_from_obj(obj, expected_sha=h, expected_artifact=spec["anchor_artifact_id"])
    q = spec["parents"]["state_vol_v1_results"]
    hs = sha_full(q["path"])
    if hs != q["sha256"]:
        raise InvalidInput("state anchor hash changed")
    sobj = json.load(open(q["path"], encoding="utf-8"))
    sobj["_sha256"] = hs
    ast = state_anchor_from_obj(sobj, expected_sha=hs)
    if ast["h_ref"] != spec["h_ref"] or ast["h_initial"] != spec["h_initial"] or ast["lambda_"] != LAMBDA or ast["initial_n"] != INIT_N:
        raise InvalidInput("pinned state constants differ between spec and stored state envelope")
    return a20, ast


def align_state(h, days, ast, first=GRID_ORIGIN, last=None):
    last = len(days) - 2 if last is None else last
    expected = [days[i] for i in range(first, last + 1)]
    if sorted(ast["forecasts"]) != sorted(expected) or len(ast["forecasts"]) != len(expected):
        raise InvalidInput("stored state decisions do not match the expected session range (missing/extra/shifted)")
    max_err = 0.0
    for i in range(first, last + 1):
        rec = ast["forecasts"][days[i]]
        if rec["predicted_for_session"] != days[i + 1]:
            raise InvalidInput(f"state anchor target session mismatch at {days[i]}")
        if h[i] is None:
            raise InvalidInput(f"replayed state undefined at {days[i]}")
        e = abs(h[i] - rec["h"])
        max_err = max(max_err, e)
        if e > STATE_TOL:
            raise InvalidInput(f"replayed state differs from stored at {days[i]}: {e}")
    return dict(comparisons=len(expected), max_abs_error=max_err, first=days[first], last=days[last], state_artifact_id=ast["artifact_id"])


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
        if a is None or a["exit_session"] != o["exit_session"] or a["block_k"] != o["block_k"] or a["entry_session"] != o["entry_session"] or a["label_available_session"] != o["label_available_session"]:
            raise InvalidInput(f"anchor outcome missing/misaligned at {o['decision_session']}")
        d = abs(o["realized_simple_price_return"] - float(a["realized_simple_price_return"]))
        max_y = max(max_y, d)
        if d > TOL:
            raise InvalidInput(f"label differs from stored outcome at {o['decision_session']}: {d}")
    years = {}
    for r in recs:
        if r["exit_session"]:
            years[r["entry_session"][:4]] = years.get(r["entry_session"][:4], 0) + 1
    if years != {k: int(x) for k, x in anchor["blocks_per_entry_year"].items()}:
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
        rows.append(dict(year=r["entry_session"][:4], y=y, mu={H3: r["mu_hat_H3"], H0: float(a["mu_hat"][H0]), H1: float(a["mu_hat"][H1])}))
    if not rows:
        raise InvalidInput("nothing to score")
    years = sorted({x["year"] for x in rows})
    yr = [x["year"] for x in rows]
    ys = [x["y"] for x in rows]
    sq = {vn: [(x["mu"][vn] - x["y"]) ** 2 for x in rows] for vn in (H3, H0, H1)}

    def mean(v):
        return sum(v) / len(v)

    def by_year(vals):
        return {Y: mean([x for x, g in zip(vals, yr) if g == Y]) for Y in years}

    met = dict(n_scored=len(rows), years=years, blocks_per_entry_year={Y: sum(1 for g in yr if g == Y) for Y in years}, mse={vn: mean(sq[vn]) for vn in (H3, H0, H1)}, mse_by_year={vn: by_year(sq[vn]) for vn in (H3, H0, H1)}, paired={})
    for b, role in ((H1, "primary"), (H0, "secondary")):
        d = [x - z for x, z in zip(sq[H3], sq[b])]
        met["paired"][f"{H3}_minus_{b}"] = dict(mean_sq_error_diff=mean(d), by_entry_year=by_year(d), share_blocks_H3_strictly_better=sum(1 for x in d if x < 0) / len(d), role=role)
    for vn in (H0, H1):
        if abs(met["mse"][vn] - float(anchor["mse"][vn])) > TOL:
            raise InvalidInput(f"loaded {vn} MSE differs from pinned h20-v1 metrics")
    mu = [x["mu"][H3] for x in rows]
    elig = [(m, y) for m, y in zip(mu, ys) if m != 0.0 and y != 0.0]
    mse0 = met["mse"][H0]
    met["secondary_H3"] = dict(mse_ratio_to_H0=met["mse"][H3] / mse0 if mse0 else None, uncentered_cross_moment=mean([m * y for m, y in zip(mu, ys)]), centered_covariance=mean([m * y for m, y in zip(mu, ys)]) - mean(mu) * mean(ys), sign_accuracy=(sum(1 for m, y in elig if (m > 0) == (y > 0)) / len(elig)) if elig else None, sign_eligible_count=len(elig), zero_target_count=sum(1 for y in ys if y == 0.0), zero_mu_count=sum(1 for m in mu if m == 0.0), mu_mean=mean(mu), mu_pop_sd=pop_sd(mu))
    first, last = res["records"][0], res["records"][-1]
    met["cell_diagnostics"] = {k: dict(decision_session=r["decision_session"], h_i=r["h_i"], v_i=r["v_i"], n_global=r["n_global"], n_cell0=r["n_cell0"], n_cell1=r["n_cell1"], n_undefined=r["n_undefined"], cell_means=r["cell_means"]) for k, r in (("first_grid_prediction", first), ("last_grid_prediction", last))}
    met["support_limitation"] = "relative to H1, the selected H3 cell omits the undefined-prefix labels as well as the opposite state; the score difference does not isolate the binary feature alone"
    met["note"] = "145 dependent blocks; descriptive only; not comparable with 1-session MSE magnitudes; no p-values or CI; cross moments are not PnL; state partition is sample weighting, not a timing rule"
    return met


def fmt_null(x, spec_fmt):
    return format(x, spec_fmt) if x is not None else "해당 없음"


def summary_lines(res, met, al, sal, cal):
    c = res["counts"]
    s = met["secondary_H3"]
    acc = f"{s['sign_accuracy']*100:.1f}% ({s['sign_eligible_count']}건)" if s["sign_accuracy"] is not None else "해당 없음"
    lines = [f"세션 {c['sessions']} | 보정상수 재구성 검사 {cal['checks']} (h_ref 차 {cal['h_ref_abs_diff']:.1e}, h_init 차 {cal['h_initial_abs_diff']:.1e}) | 상태 재귀 {sal['recursion_updates']} (초기 포함 {sal['recursion_updates']+1}) | 저장 상태 정렬 {sal['comparisons']} 최대오차 {sal['max_abs_error']:.1e} | 성숙 라벨 {c['mature_label_updates']} = 정의 {c['defined_state_label_updates']} + 미정의 {c['undefined_state_label_updates']} | 격자 예측 {c['prediction_records']} (채점 {c['mature_scored_records']}, 미확정 {c['pending_outcomes']}) | H1 정렬 {c['global_mean_alignment_reads']} 최대차 {al['max_abs_global_mean_minus_stored_H1']:.1e}, 라벨 {al['max_abs_label_minus_stored_outcome']:.1e} | 대체: 셀→전역 {c['fallback_cell_to_global']}, 전역 빈 {c['fallback_empty']}"]
    lines.append(f"MSE H3 {met['mse'][H3]:.6e} | 저장 H1 {met['mse'][H1]:.6e} | 저장 H0 {met['mse'][H0]:.6e} | 비율 H3/H0 {fmt_null(s['mse_ratio_to_H0'], '.6f')}")
    for k, p in met["paired"].items():
        lines.append(f"{k} [{p['role']}]: 제곱오차 차이 평균 {p['mean_sq_error_diff']:+.3e} (H3 엄격 우세 블록 {p['share_blocks_H3_strictly_better']*100:.1f}%) | 진입연도별: " + ", ".join(f"{y} {x:+.2e} ({met['blocks_per_entry_year'][y]})" for y, x in p["by_entry_year"].items()))
    lines.append(f"H3 μ 평균 {s['mu_mean']:+.3e} sd {s['mu_pop_sd']:.3e} | 비중심 교차모멘트 {s['uncentered_cross_moment']:+.3e} 중심 공분산 {s['centered_covariance']:+.3e} | 부호 적중 {acc} (Y=0 {s['zero_target_count']}, μ=0 {s['zero_mu_count']})")
    for k, d in met["cell_diagnostics"].items():
        lines.append(f"셀 진단 {k} ({d['decision_session']}, h_i {d['h_i']:.4e}, v_i={d['v_i']}, n {d['n_global']}): cell0(h≤h_ref) n {d['n_cell0']} 평균 {fmt_null(d['cell_means']['cell0'], '+.4e')} | cell1(h>h_ref) n {d['n_cell1']} 평균 {fmt_null(d['cell_means']['cell1'], '+.4e')} | 미정의 n {d['n_undefined']} 평균 {fmt_null(d['cell_means']['undefined'], '+.4e')}")
    return lines


def build_payload(spec, res, met, al, sal, cal, a20, ast, code_sha, now_utc):
    artifact_id = f"{MODULE}_{now_utc.strftime('%Y%m%d_%H%M%S')}_{code_sha[:8]}"
    p = spec["parents"]
    env = dict(schema_version=SCHEMA, artifact_id=artifact_id, parent_artifact_hashes=[p[k]["sha256"] for k in PARENT_ORDER], parent_artifact_roles={k: dict(path=p[k]["path"], sha256=p[k]["sha256"]) for k in PARENT_ORDER}, anchor_artifact_ids=dict(h20_v1=a20["artifact_id"], state_vol_v1=ast["artifact_id"]), code_sha256=code_sha, spec_sha256=spec["sha256"], asof_utc=now_utc.isoformat(timespec="seconds"), exchange_session=dict(exchange="KRX", symbol=SYM, calendar="raw daily session dates (identical to adjusted)", timezone="Asia/Seoul", utc_mapping="session date D in KST (UTC+09:00); no intraday clock times used", decision_marker="session_close", entry_marker="next_session_open", exit_marker="open of session i+21"), symbol=SYM, availability_status="available", decision_at="session_close (logical marker; information through close i only)", execution_not_before="next_session_open (logical marker; no order placed)", target=dict(definition="Y_j = O[j+21]/O[j+1] - 1, raw open of session j+1 to raw open of session j+21, gross simple price return over 20 held sessions", value_unit="simple_return", moment_order=1, forecast_kind="conditional_mean", available_at="open of session j+21 (exit_session)", excludes="dividends/distributions and all costs"), maturity=dict(rule="pool j >= 199 and j + 21 <= i", update_order="at close i add Y_j for j = i-21 to the global accumulator and to cell v[j] (or undefined); emit H3 only on grid decisions; no updates after the last grid decision 5931 (last label j = 5910)"), grid=dict(origin_decision_index=GRID_ORIGIN, step_sessions=GRID_STEP, rule="i = 3031 + 20k, i + 1 < N; grid-only forecasts", nonoverlap="adjacent scored holding intervals share only the boundary open; not independence"), state_feature=dict(input="adjusted close C; r_t = log(C[t]/C[t-1]); h[252] = mean(r[1..252]^2); h[i] = 0.94 h[i-1] + 0.06 r[i]^2 for i >= 253 (expression lam*h + (1-lam)*x**2 with numpy float64 x, state-vol-v1 order)", units="log_return second moment (moment_order 2); a variance only under a zero-conditional-mean assumption", h_initial=spec["h_initial"], h_ref=spec["h_ref"], threshold="v[i] = 1 iff un-floored h[i] > h_ref else 0 (equality 0)", calibration=dict(end_marker=CAL_END_MARKER, last_session=cal["calibration_last_session"], last_index=CAL_LAST_INDEX, availability="parameters available at close of the last calibration session on the retained calendar", reconstruction_checks=cal), historical_membership=CAVEAT, undefined_prefix="labels j in 199..251 (53) have no h[j]; counted in the undefined cell, never selected, kept in the global/H1 pool", replay_extent=f"h replayed through index {sal['last_index']} ({sal['last']}) for state alignment only; H3 statistics stop at grid 5931", alignment=sal), new_predictor=dict(name=H3, formula="mean of mature labels with v[j] == v[i] over defined states only; empty selected cell -> contemporaneous global mean incl. undefined labels (fallback_cell_to_global); empty global -> 0 (fallback_empty)", invariants="n_cell0 + n_cell1 + n_undefined == n_global; n_selected == n_cell[v[i]]; undefined never selected; empty cell mean null"), baselines=dict(policy="H0/H1 forecasts and outcomes loaded verbatim from the pinned predict-h20-v1 artifact; not recomputed; running global mean compared to stored H1 within 1e-15", anchor_artifact_id=a20["artifact_id"], alignment=al), training=dict(policy="B: all mature rolling 20-session labels regardless of grid alignment", overlap_caveat="availability makes the estimator causal; overlapping windows remain dependent and weight shared daily moves repeatedly; raw label count is not an effective independent sample size"), support_limitation=met["support_limitation"], selection_caveat="candidate chosen after prior prediction and allocation results were viewed; constants frozen before this run only; not a pre-outcome-selected hypothesis", feedback="only newly mature labels update global/cell statistics; membership fixed by v[j]; current v[i] selects the cell; outcomes and scores never tune estimates", assumptions="price return excluding distributions and costs; provider quotes not independently reconciled (2026-07-31 jump as supplied); adjusted series is a current snapshot; all evaluation periods already viewed (exploratory)", statistical_learning=dict(optimizer_calls=0, new_hyperparameter_searches=0, new_estimated_predictors=1, loaded_baselines=2, loaded_calibration_parameters=2, source_calibration_reconstruction_checks=cal["checks"], state_recursion_updates=sal["recursion_updates"], state_values_including_initial=sal["recursion_updates"] + 1, stored_state_alignment_comparisons=sal["comparisons"], mature_label_updates=res["counts"]["mature_label_updates"], defined_state_label_updates=res["counts"]["defined_state_label_updates"], undefined_state_label_updates=res["counts"]["undefined_state_label_updates"], H3_forecast_estimates=res["counts"]["H3_forecast_estimates"], global_mean_alignment_reads=res["counts"]["global_mean_alignment_reads"], note="deterministic replay is still computation of statistical state estimates; one mature label updates one cell and the global accumulator"), limitations="145 scored blocks, about 12 per entry year, potential serial dependence; descriptive only; not comparable with 1-day MSE; no PnL, calibration, p-values or horizon/offset selection; no timing or edge inference", resources=dict(cpu_threads=1, external_timeout_seconds=spec["external_timeout_seconds"], experiments=1, new_estimated_predictors=1, loaded_baselines=2, optimizer_calls=0, api_calls=0, orders=0, random_paths=0))
    ref = dict(envelope_artifact_id=artifact_id, anchor_artifact_ids=dict(h20_v1=a20["artifact_id"], state_vol_v1=ast["artifact_id"]))
    amap = {a["decision_session"]: a for a in a20["records"]}
    preds = [dict(r, anchor_mu_loaded={H0: float(amap[r["decision_session"]]["mu_hat"][H0]), H1: float(amap[r["decision_session"]]["mu_hat"][H1])}, **ref) for r in res["records"]]
    outs = [dict(a20["outcomes"][o["decision_session"]], loaded_from_anchor=True, **ref) for o in res["labels"]]
    return dict(envelope=env, counts=res["counts"], alignment=al, state_alignment=sal, calibration_reconstruction=cal, metrics=met, lines=summary_lines(res, met, al, sal, cal), prediction_records=preds, outcome_records=outs)


def real_calendar_checks(spec, days):
    n = len(days)
    first = next(i for i in range(POOL_START, n - 1) if days[i + 1] >= ENTRY_START)
    grid = grid_indices(n, GRID_ORIGIN, GRID_STEP)
    scored = [i for i in grid if i + H + 1 <= n - 1]
    pending = [i for i in grid if i + H + 1 > n - 1]
    years = {}
    for i in scored:
        years[days[i + 1][:4]] = years.get(days[i + 1][:4], 0) + 1
    facts = dict(price_sessions=n, horizon_sessions=H, pool_start_index=POOL_START, first_common_training_decision=days[POOL_START], first_common_label_available_session=days[POOL_START + H + 1], grid_origin_decision_index=first, grid_origin_decision=days[GRID_ORIGIN], grid_origin_entry=days[GRID_ORIGIN + 1], grid_origin_exit=days[GRID_ORIGIN + H + 1], grid_step_sessions=GRID_STEP, grid_predictions=len(grid), scored_blocks=len(scored), pending_blocks=len(pending), first_prediction_mature_training_count=(GRID_ORIGIN - H - 1) - POOL_START + 1, last_scored_decision=days[scored[-1]], last_scored_entry=days[scored[-1] + 1], last_scored_exit=days[scored[-1] + H + 1], pending=[dict(decision_session=days[i], entry_session=days[i + 1], exit_session=None, sessions_missing=(i + H + 1) - (n - 1)) for i in pending], blocks_per_entry_year=years, last_grid_decision_index=grid[-1], last_grid_mature_pool_count=(grid[-1] - H - 1) - POOL_START + 1, update_count=(grid[-1] - H - 1) - POOL_START + 1, defined_updates=(grid[-1] - H - 1) - INIT_INDEX + 1, undefined_updates=INIT_INDEX - POOL_START, state_recursion_updates=(n - 2) - INIT_INDEX, stored_state_comparisons=(n - 2) - GRID_ORIGIN + 1, first_pool=(POOL_START, GRID_ORIGIN - H - 1), last_pool=(POOL_START, grid[-1] - H - 1), calibration_last_session=days[CAL_LAST_INDEX], calibration_marker_ok=days[CAL_LAST_INDEX] <= CAL_END_MARKER < days[CAL_LAST_INDEX + 1])
    e = spec["facts"]
    exp = {k: e[k] for k in ("price_sessions", "horizon_sessions", "pool_start_index", "first_common_training_decision", "first_common_label_available_session", "grid_origin_decision_index", "grid_origin_decision", "grid_origin_entry", "grid_origin_exit", "grid_step_sessions", "grid_predictions", "scored_blocks", "pending_blocks", "first_prediction_mature_training_count", "last_scored_decision", "last_scored_entry", "last_scored_exit", "pending", "blocks_per_entry_year")}
    exp["blocks_per_entry_year"] = {k: int(x) for k, x in exp["blocks_per_entry_year"].items()}
    exp.update(last_grid_decision_index=spec["last_grid_decision_index"], last_grid_mature_pool_count=spec["last_grid_mature_pool_count"], update_count=spec["update_count"], defined_updates=spec["defined_updates"], undefined_updates=spec["undefined_updates"], state_recursion_updates=spec["state_recursion_updates"], stored_state_comparisons=spec["stored_state_comparisons"], first_pool=tuple(spec["first_pool"]), last_pool=tuple(spec["last_pool"]), calibration_last_session="2014-12-30", calibration_marker_ok=True)
    bad = {k: (facts[k], exp[k]) for k in exp if facts[k] != exp[k]}
    if bad:
        raise InvalidInput(f"calendar facts differ from spec: {bad}")
    if first != GRID_ORIGIN:
        raise InvalidInput("grid origin is not the first evaluation decision")
    return facts


def anchor_metadata_checks(a20, ast, days, facts):
    recs = a20["records"]
    if len(recs) != facts["grid_predictions"] or a20["n_outcomes"] != facts["scored_blocks"]:
        raise InvalidInput("h20 anchor counts differ from calendar facts")
    for k, a in enumerate(recs):
        i = GRID_ORIGIN + GRID_STEP * k
        if a["decision_index"] != i or a["block_k"] != k or a["decision_session"] != days[i] or a["entry_session"] != days[i + 1] or a["exit_session"] != (days[i + H + 1] if i + H + 1 < len(days) else None) or a["n_pool"] != (i - H - 1) - POOL_START + 1:
            raise InvalidInput(f"h20 anchor record {k} misaligned with the grid calendar")
    if recs[-1]["target_status"] != "pending_outcome" or any(a["target_status"] != "mature_for_scoring" for a in recs[:-1]):
        raise InvalidInput("h20 anchor pending flags unexpected")
    if {k: int(x) for k, x in a20["blocks_per_entry_year"].items()} != facts["blocks_per_entry_year"]:
        raise InvalidInput("h20 anchor blocks per entry year differ from calendar facts")
    n = len(days)
    expected = [days[i] for i in range(GRID_ORIGIN, n - 1)]
    if ast["n_records"] != facts["stored_state_comparisons"] or sorted(ast["forecasts"]) != sorted(expected):
        raise InvalidInput("state anchor decisions differ from the expected range")
    for i in range(GRID_ORIGIN, n - 1):
        if ast["forecasts"][days[i]]["predicted_for_session"] != days[i + 1]:
            raise InvalidInput("state anchor target session misaligned")
    return dict(h20_records=len(recs), h20_outcomes=a20["n_outcomes"], h20_artifact_id=a20["artifact_id"], state_records=ast["n_records"], state_artifact_id=ast["artifact_id"], state_first=expected[0], state_last=expected[-1])


def run(spec_path=SPEC):
    spec = load_spec(spec_path)
    check_parents(spec)
    days, O, C = load_prices(spec)
    facts = real_calendar_checks(spec, days)
    a20, ast = load_anchors(spec)
    anchor_metadata_checks(a20, ast, days, facts)
    r = log_returns(C)
    h_ref_rec, h_init_rec = calibration_constants(r)
    cal = dict(checks=2, h_ref_reconstructed=h_ref_rec, h_ref_pinned=spec["h_ref"], h_ref_abs_diff=abs(h_ref_rec - spec["h_ref"]), h_initial_reconstructed=h_init_rec, h_initial_pinned=spec["h_initial"], h_initial_abs_diff=abs(h_init_rec - spec["h_initial"]), calibration_last_session=facts["calibration_last_session"], calibration_returns=CAL_LAST_INDEX)
    if cal["h_ref_abs_diff"] > STATE_TOL or cal["h_initial_abs_diff"] > STATE_TOL:
        raise InvalidInput(f"calibration constants do not reconstruct: {cal}")
    n = len(days)
    h, rec_updates = replay_state(r, n, spec["h_initial"], end=n - 2)
    if rec_updates != facts["state_recursion_updates"]:
        raise InvalidInput("state recursion count differs from spec")
    sal = align_state(h, days, ast, first=GRID_ORIGIN, last=n - 2)
    sal.update(recursion_updates=rec_updates, last_index=n - 2)
    v = states(h, spec["h_ref"])
    res = engine(days, O, v, h)
    c = res["counts"]
    if c["prediction_records"] != facts["grid_predictions"] or c["mature_scored_records"] != facts["scored_blocks"] or c["mature_label_updates"] != facts["update_count"] or c["defined_state_label_updates"] != facts["defined_updates"] or c["undefined_state_label_updates"] != facts["undefined_updates"] or c["last_pool_count"] != facts["last_grid_mature_pool_count"] or res["records"][0]["n_global"] != facts["first_prediction_mature_training_count"]:
        raise InvalidInput("engine counts differ from calendar facts")
    al = align(res, a20)
    met = score(res, a20)
    check_parents(spec)
    code_sha = sha_full(os.path.abspath(__file__))
    payload = build_payload(spec, res, met, al, sal, cal, a20, ast, code_sha, datetime.now(timezone.utc))
    payload["calendar_facts"] = facts
    data = json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False).encode("utf-8")
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "predict_h20_v3_results.json")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; refusing to overwrite")
    with open(out, "xb") as f:
        f.write(data)
    print("\n".join(payload["lines"]))
    print("saved", out, "sha", sha_full(out)[:16], "artifact", payload["envelope"]["artifact_id"], "code", code_sha[:16], "spec", spec["sha256"][:16])
    return payload


def synthetic_days(n, start=date(2015, 1, 1)):
    return [(start + timedelta(days=k)).isoformat() for k in range(n)]


def synthetic_h20_anchor(res, artifact_id="predict_h20_v1_synthetic"):
    recs = [dict(decision_index=r["decision_index"], decision_session=r["decision_session"], block_k=r["block_k"], entry_session=r["entry_session"], exit_session=r["exit_session"], label_available_session=r["label_available_session"], target_status=r["target_status"], n_pool=r["n_global"], mu_hat={H0: 0.0, H1: r["global_mean_for_fallback"]}) for r in res["records"]]
    outs = [dict(o) for o in res["labels"]]
    om = {o["decision_session"]: o for o in outs}
    rows = [(r, om[r["decision_session"]]["realized_simple_price_return"]) for r in recs if r["exit_session"]]
    mse = {vn: sum((r["mu_hat"][vn] - y) ** 2 for r, y in rows) / len(rows) for vn in (H0, H1)} if rows else {H0: 0.0, H1: 0.0}
    years = {}
    for r, _ in rows:
        years[r["entry_session"][:4]] = years.get(r["entry_session"][:4], 0) + 1
    return dict(envelope=dict(artifact_id=artifact_id), prediction_records=recs, outcome_records=outs, metrics=dict(mse=mse, blocks_per_entry_year=years))


def synthetic_state_anchor(days, h, first, last, h_ref, h_initial, artifact_id="state_vol_synthetic"):
    recs = [dict(decision_session=days[i], predicted_for_session=days[i + 1], h_forecast=dict(ewma=h[i], reference=h_ref), value_unit="log_return", moment_order=2) for i in range(first, last + 1)]
    return dict(envelope=dict(artifact_id=artifact_id, reference=dict(h=h_ref), candidate=dict(h_initial=h_initial, lambda_=LAMBDA, initial_n=INIT_N)), forecast_records=recs)


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
    hz, ps = 3, 1
    init_index, init_n = 4, 4
    n = 70
    days = synthetic_days(n)
    O = [float(100 + 7 * ((k * 37) % 23) + 3 * ((k * 11) % 5)) for k in range(n)]
    C = [1000.0 * (1.0 + 0.02 * math.sin(k * 0.9) + 0.004 * ((k * 53) % 7)) for k in range(n)]
    validate_inputs(days, O, C)
    r = log_returns(C)
    assert len(r) == n - 1 and abs(float(r[0]) - math.log(C[1] / C[0])) < 1e-15
    h_init = float(np.mean(r[:init_n] ** 2))
    h, upd = replay_state(r, n, h_init, init_index=init_index, lam=LAMBDA, end=n - 2)
    assert upd == (n - 2) - init_index and h[init_index - 1] is None and h[init_index] == h_init and h[n - 1] is None
    lam_f = Fraction(94, 100)
    hf = Fraction(h_init)
    for i in range(init_index + 1, n - 1):
        hf = lam_f * hf + (1 - lam_f) * Fraction(float(r[i - 1])) ** 2
        assert abs(h[i] - float(hf)) <= 1e-17 + 1e-14 * float(hf), (i, h[i], float(hf))
    h_ref = sorted(x for x in h if x is not None)[len([x for x in h if x is not None]) // 2]
    v = states(h, h_ref)
    assert v[:init_index] == [None] * init_index and sum(1 for x in v if x == 1) > 0 and sum(1 for x in v if x == 0) > 0
    i_eq = next(i for i in range(init_index, n - 1) if h[i] == h_ref)
    assert v[i_eq] == 0
    C_fut = list(C)
    C_fut[-1] *= 1.5
    h_fut, _ = replay_state(log_returns(C_fut), n, h_init, init_index=init_index, end=n - 2)
    assert all(h_fut[i] == h[i] for i in range(init_index, n - 1))
    org, st = 8, 3
    res = engine(days, O, v, h, h=hz, pool_start=ps, origin=org, step=st)
    grid = grid_indices(n, org, st)
    assert [x["decision_index"] for x in res["records"]] == grid
    attribution_differs = False
    for rec in res["records"]:
        i = rec["decision_index"]
        js = list(range(ps, i - hz))
        cells = {0: [j for j in js if v[j] == 0], 1: [j for j in js if v[j] == 1], "undefined": [j for j in js if v[j] is None]}
        assert rec["n_global"] == len(js) and rec["n_cell0"] == len(cells[0]) and rec["n_cell1"] == len(cells[1]) and rec["n_undefined"] == len(cells["undefined"]) and rec["n_cell0"] + rec["n_cell1"] + rec["n_undefined"] == rec["n_global"]
        assert rec["v_i"] == v[i] and rec["h_i"] == h[i] and rec["n_selected"] == len(cells[rec["v_i"]])
        if i - hz - 1 >= ps and v[i - hz - 1] is not None and v[i - hz - 1] != v[i]:
            attribution_differs = True
        sel = cells[rec["v_i"]]
        if not js:
            assert rec["fallback_empty"] and rec["fallback_cell_to_global"] and rec["mu_hat_H3"] == 0.0
        elif not sel:
            assert rec["fallback_cell_to_global"] and not rec["fallback_empty"] and near(rec["mu_hat_H3"], exact_mean(O, js, hz))
        else:
            assert not rec["fallback_cell_to_global"] and near(rec["mu_hat_H3"], exact_mean(O, sel, hz))
        for key, cj in (("cell0", cells[0]), ("cell1", cells[1]), ("undefined", cells["undefined"])):
            m = rec["cell_means"][key]
            assert (m is None) == (not cj) and (m is None or near(m, exact_mean(O, cj, hz)))
        assert (rec["exit_session"] is None) == (i + hz + 1 >= n)
    assert attribution_differs
    assert res["counts"]["undefined_state_label_updates"] == init_index - ps and res["counts"]["defined_state_label_updates"] + res["counts"]["undefined_state_label_updates"] == res["counts"]["mature_label_updates"] and res["counts"]["H3_forecast_estimates"]["total"] == len(grid)
    assert any(x["n_undefined"] > 0 and x["n_selected"] == (x["n_cell1"] if x["v_i"] == 1 else x["n_cell0"]) for x in res["records"])
    small_days, sO = synthetic_days(14), [100.0, 110.0, 121.0, 115.0, 120.0, 126.0, 120.0, 130.0, 128.0, 135.0, 131.0, 140.0, 138.0, 145.0]
    v_small = [None, None, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1]
    h_small = [None, None] + [1.0] * 12
    sm = {x["decision_index"]: x for x in engine(small_days, sO, v_small, h_small, h=hz, pool_start=ps, origin=2, step=1)["records"]}
    assert sm[2]["fallback_empty"] and sm[2]["fallback_cell_to_global"] and sm[2]["mu_hat_H3"] == 0.0
    assert sm[5]["n_global"] == 1 and sm[5]["n_undefined"] == 1 and sm[5]["fallback_cell_to_global"] and not sm[5]["fallback_empty"] and abs(sm[5]["mu_hat_H3"] - label(sO, 1, hz)) < 1e-15 and sm[5]["cell_means"]["cell0"] is None and sm[5]["cell_means"]["undefined"] is not None
    assert sm[7]["n_global"] == 3 and sm[7]["n_undefined"] == 1 and sm[7]["v_i"] == 0 and sm[7]["n_cell0"] == 0 and sm[7]["n_cell1"] == 2 and sm[7]["fallback_cell_to_global"] and near(sm[7]["mu_hat_H3"], exact_mean(sO, [1, 2, 3], hz))
    a20 = h20_anchor_from_obj(synthetic_h20_anchor(res))
    al = align(res, a20)
    assert al["records"] == len(grid) and al["max_abs_global_mean_minus_stored_H1"] == 0.0
    met = score(res, a20)
    assert met["n_scored"] == len(res["labels"]) and met["paired"][f"{H3}_minus_{H1}"]["role"] == "primary"
    ast = state_anchor_from_obj(synthetic_state_anchor(days, h, org, n - 2, h_ref, h_init))
    sal = align_state(h, days, ast, first=org, last=n - 2)
    assert sal["comparisons"] == (n - 2) - org + 1 and sal["max_abs_error"] == 0.0
    for mutate in (lambda a: a["forecast_records"][3].__setitem__("decision_session", days[0]), lambda a: a["forecast_records"].pop(), lambda a: a["forecast_records"].append(dict(a["forecast_records"][0], decision_session=days[n - 1], predicted_for_session=days[n - 1])), lambda a: a["forecast_records"][2]["h_forecast"].__setitem__("ewma", a["forecast_records"][2]["h_forecast"]["ewma"] + 1e-15), lambda a: a["forecast_records"][2].__setitem__("predicted_for_session", days[0]), lambda a: a["forecast_records"][1].__setitem__("moment_order", 1), lambda a: a["forecast_records"][2]["h_forecast"].__setitem__("reference", a["forecast_records"][2]["h_forecast"]["reference"] * 1.5), lambda a: a["forecast_records"][2]["h_forecast"].__setitem__("reference", float("nan")), lambda a: a["envelope"]["reference"].__setitem__("h", float("inf"))):
        obj = synthetic_state_anchor(days, h, org, n - 2, h_ref, h_init)
        mutate(obj)
        try:
            align_state(h, days, state_anchor_from_obj(obj), first=org, last=n - 2)
            raise AssertionError("expected state rejection")
        except InvalidInput:
            pass
    obj = synthetic_state_anchor(days, h, org, n - 2, h_ref, h_init)
    obj["forecast_records"].append(dict(obj["forecast_records"][0]))
    expect_reject(state_anchor_from_obj, obj)
    obj["_sha256"] = "00" * 32
    expect_reject(state_anchor_from_obj, obj, expected_sha="11" * 32)
    for mutate in (lambda a: a["prediction_records"][4]["mu_hat"].__setitem__(H1, a["prediction_records"][4]["mu_hat"][H1] + 1e-12), lambda a: a["outcome_records"][2].__setitem__("realized_simple_price_return", a["outcome_records"][2]["realized_simple_price_return"] + 1e-12), lambda a: a["prediction_records"][3].__setitem__("entry_session", days[0]), lambda a: a["prediction_records"][3].__setitem__("n_pool", 99), lambda a: a["prediction_records"][-1].__setitem__("target_status", "mature_for_scoring"), lambda a: a["metrics"]["blocks_per_entry_year"].__setitem__("2015", 1), lambda a: a["metrics"]["mse"].__setitem__(H1, 0.5), lambda a: a["outcome_records"][1].__setitem__("entry_session", days[0]), lambda a: a["outcome_records"][1].__setitem__("label_available_session", days[0])):
        obj = synthetic_h20_anchor(res)
        mutate(obj)
        try:
            b = h20_anchor_from_obj(obj)
            align(res, b)
            score(res, b)
            raise AssertionError("expected h20 rejection")
        except InvalidInput:
            pass
    expect_reject(h20_anchor_from_obj, synthetic_h20_anchor(res), expected_artifact="other")

    def keyed(rr):
        return {x["decision_index"]: (x["mu_hat_H3"], x["v_i"], x["n_global"], x["n_cell0"], x["n_cell1"], x["n_undefined"], x["global_mean_for_fallback"]) for x in rr["records"]}

    kb = keyed(res)
    k = 32
    C_after = list(C)
    for t in range(k + 1, n):
        C_after[t] *= 1.3
    h_a, _ = replay_state(log_returns(C_after), n, h_init, init_index=init_index, end=n - 2)
    pa = keyed(engine(days, O, states(h_a, h_ref), h_a, h=hz, pool_start=ps, origin=org, step=st))
    assert all(h_a[i] == h[i] for i in range(init_index, k + 1)) and all(pa[i] == kb[i] for i in kb if i <= k)
    C_k = list(C)
    C_k[k] *= 1.2
    r_k = log_returns(C_k)
    assert float(r_k[k - 1]) != float(r[k - 1]) and float(r_k[k]) != float(r[k]) and all(float(r_k[t]) == float(r[t]) for t in range(len(r)) if t not in (k - 1, k))
    h_k, _ = replay_state(r_k, n, h_init, init_index=init_index, end=n - 2)
    pk = keyed(engine(days, O, states(h_k, h_ref), h_k, h=hz, pool_start=ps, origin=org, step=st))
    assert all(h_k[i] == h[i] for i in range(init_index, k)) and all(pk[i] == kb[i] for i in kb if i < k)
    cal_end = 30
    kc = 12
    h_ref_cal = float(np.mean(r[:cal_end] ** 2))
    v_cal = states(h, h_ref_cal)
    C_c = list(C)
    C_c[kc] *= 1.25
    r_c = log_returns(C_c)
    h_fixed_run, _ = replay_state(r_c, n, h_init, init_index=init_index, end=n - 2)
    v_fixed = states(h_fixed_run, h_ref_cal)
    assert all(h_fixed_run[i] == h[i] and v_fixed[i] == v_cal[i] for i in range(init_index, kc))
    h_ref_re = float(np.mean(r_c[:cal_end] ** 2))
    h_init_re = float(np.mean(r_c[:init_n] ** 2))
    assert h_init_re == h_init and h_ref_re != h_ref_cal
    v_re = states(replay_state(r_c, n, h_init_re, init_index=init_index, end=n - 2)[0], h_ref_re)
    changed_past = [i for i in range(init_index, kc) if v_re[i] != v_cal[i]]
    assert changed_past, "recomputing the threshold after a calibration change must reclassify at least one past label in this fixture"
    step1 = engine(days, O, v, h, h=hz, pool_start=ps, origin=org, step=1)
    k1 = keyed(step1)
    ko = 30
    O_k = list(O)
    O_k[ko] += 50.0
    pk1 = keyed(engine(days, O_k, v, h, h=hz, pool_start=ps, origin=org, step=1))
    j_num, j_den = ko - hz - 1, ko - 1
    d_num = label(O_k, j_num, hz) - label(O, j_num, hz)
    d_den = label(O_k, j_den, hz) - label(O, j_den, hz)
    assert d_num != 0.0 and d_den != 0.0 and all(pk1[i] == k1[i] for i in k1 if i < ko)
    for i in range(ko, ko + hz):
        b_i, p_i = k1[i], pk1[i]
        assert p_i[2] == b_i[2] and abs((p_i[6] - b_i[6]) - d_num / b_i[2]) < 1e-15
        cell_num = v[j_num]
        sel_n = b_i[4] if v[i] == 1 else b_i[3]
        if sel_n == 0:
            assert abs((p_i[0] - b_i[0]) - d_num / b_i[2]) < 1e-15
        elif cell_num == v[i]:
            assert abs((p_i[0] - b_i[0]) - d_num / sel_n) < 1e-15
        else:
            assert p_i[0] == b_i[0]
    b_e, p_e = k1[ko + hz], pk1[ko + hz]
    assert abs((p_e[6] - b_e[6]) - (d_num + d_den) / b_e[2]) < 1e-15
    O_after = list(O)
    for t in range(k + 1, n):
        O_after[t] += 50.0
    po = keyed(engine(days, O_after, v, h, h=hz, pool_start=ps, origin=org, step=st))
    assert all(po[i] == kb[i] for i in kb if i <= k) and any(po[i] != kb[i] for i in kb if i > k)
    expect_reject(slice_series, ["a", "b", "c"], [1.0, 2.0], "c")
    expect_reject(validate_inputs, ["a", "b"], [100.5, 100.0], [1.0, 1.0])
    expect_reject(validate_inputs, ["b", "a"], [100.0, 100.0], [1.0, 1.0])
    expect_reject(validate_inputs, ["a", "b"], [100.0, 100.0], [1.0, float("nan")])
    expect_reject(validate_inputs, ["a", "b"], [100.0, 100.0], [1.0])
    expect_reject(replay_state, r, n, h_init, init_index=0, end=n - 2)
    expect_reject(engine, days, O, v, h, h=hz, pool_start=ps, origin=2, step=st)
    spec = load_spec()
    assert spec["sha256"] == SPEC_SHA256 and check_parents(spec) == 6 and spec["h_ref"] == 0.00021990165124809738 and spec["h_initial"] == 0.0003266282769865406
    cdays = synthetic_days(40)
    Cc = [100.0] * 40
    rc_ = log_returns(Cc)
    hc, _ = replay_state(rc_, 40, float(np.mean(rc_[:init_n] ** 2)), init_index=init_index, end=38)
    vc = states(hc, 0.0)
    assert all(x == 0 for x in vc[init_index:39])
    rcp = engine(cdays, [100.0] * 40, vc, hc, h=hz, pool_start=ps, origin=org, step=st)
    ac = h20_anchor_from_obj(synthetic_h20_anchor(rcp))
    alc = align(rcp, ac)
    mc = score(rcp, ac)
    assert mc["mse"][H0] == 0.0 and mc["secondary_H3"]["mse_ratio_to_H0"] is None and mc["secondary_H3"]["sign_accuracy"] is None
    astc = state_anchor_from_obj(synthetic_state_anchor(cdays, hc, org, 38, 0.0, hc[init_index]))
    salc = align_state(hc, cdays, astc, first=org, last=38)
    salc.update(recursion_updates=38 - init_index, last_index=38)
    calc = dict(checks=2, h_ref_reconstructed=0.0, h_ref_pinned=0.0, h_ref_abs_diff=0.0, h_initial_reconstructed=hc[init_index], h_initial_pinned=hc[init_index], h_initial_abs_diff=0.0, calibration_last_session=cdays[20], calibration_returns=20)
    payc = build_payload(spec, rcp, mc, alc, salc, calc, ac, astc, "ab" * 32, datetime(2026, 11, 6, 11, 40, 0, tzinfo=timezone.utc))
    assert "해당 없음" in payc["lines"][1] and json.loads(json.dumps(payc, allow_nan=False))["metrics"]["secondary_H3"]["mse_ratio_to_H0"] is None and all(o["loaded_from_anchor"] for o in payc["outcome_records"])
    sal.update(recursion_updates=upd, last_index=n - 2)
    cal = dict(checks=2, h_ref_reconstructed=h_ref, h_ref_pinned=h_ref, h_ref_abs_diff=0.0, h_initial_reconstructed=h_init, h_initial_pinned=h_init, h_initial_abs_diff=0.0, calibration_last_session=days[30], calibration_returns=30)
    pay = build_payload(spec, res, met, al, sal, cal, a20, ast, "ef" * 32, datetime(2026, 11, 6, 11, 41, 0, tzinfo=timezone.utc))
    env = pay["envelope"]
    for key in ("schema_version", "artifact_id", "parent_artifact_hashes", "code_sha256", "spec_sha256", "asof_utc", "exchange_session", "symbol", "availability_status", "decision_at", "execution_not_before"):
        assert key in env, key
    assert env["artifact_id"] == "predict_h20_v3_20261106_114100_efefefef" and env["parent_artifact_hashes"] == [spec["parents"][kk]["sha256"] for kk in PARENT_ORDER] and env["anchor_artifact_ids"] == dict(h20_v1="predict_h20_v1_synthetic", state_vol_v1="state_vol_synthetic") and all(x["calibration_caveat"] == CAVEAT for x in pay["prediction_records"])
    json.dumps(pay, allow_nan=False)
    rdays, rO, rC = load_prices(spec)
    facts = real_calendar_checks(spec, rdays)
    ra20, rast = load_anchors(spec)
    meta = anchor_metadata_checks(ra20, rast, rdays, facts)
    assert meta["h20_records"] == 146 and meta["h20_outcomes"] == 145 and meta["state_records"] == 2906 and meta["state_first"] == "2014-12-30" and meta["state_last"] == "2026-11-04" and facts["defined_updates"] == 5659 and facts["undefined_updates"] == 53 and facts["state_recursion_updates"] == 5684 and facts["calibration_last_session"] == "2014-12-30"
    assert check_parents(spec) == 6
    print("selftest ok", dict(code_sha256_16=sha_full(os.path.abspath(__file__))[:16], here=os.path.dirname(os.path.abspath(__file__)), spec_sha256_16=spec["sha256"][:16], real_data="hashes, calendar facts, pinned constants, stored metadata only; no real state replay, calibration reconstruction, means or scores"))


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    elif not sys.argv[1:]:
        run()
    else:
        print("usage: predict_h20v3.py | selftest")
