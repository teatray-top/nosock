import hashlib
import json
import math
import os
import sys
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction

SPEC = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work\predict_v3_spec_20261106.json")
SPEC_SHA256 = "6e7aa09753c75c0c316b138df6bed511d38287879ba6ade58a3f35969c08f4ca"
OUT_DIR = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work\predict_v3_20261106")
SCHEMA = "predict-v3"
MODULE = "predict_v3"
SYM = "069500"
CUTOFF = "2026-11-05"
POOL_START = 199
ENTRY_START = "2015-01-01"
P4 = "P4_expanding_ols_prev_return"
P3 = "P3_sign_prev_return_conditional_mean"
V1_BASELINES = ("P0_zero", "P1_expanding_mean", "P2_ma200_conditional_mean")
BASELINES = V1_BASELINES + (P3,)
PARENT_ORDER = ("raw_prices", "adjusted_prices", "spec", "predict_v1_results", "predict_v2_results")
GUARD = 1e-24
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
    if s["evaluation"]["entry_session_start"] != ENTRY_START or s["source"]["cutoff"] != CUTOFF or s["source"]["symbol"] != SYM or list(s["parent_order"]) != list(PARENT_ORDER):
        raise InvalidInput("spec design constants differ from module constants")
    parents = {k: dict(path=s["parents"][k]["path"], sha256=s["parents"][k]["sha256"]) for k in ("raw_prices", "adjusted_prices", "predict_v1_results", "predict_v2_results")}
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


def feature(C, i):
    return C[i] / C[i - 1] - 1.0


def label(O, j):
    return O[j + 2] / O[j + 1] - 1.0


def finite(*vals):
    return all(math.isfinite(v) for v in vals)


def coefficients(n, xbar, ybar, M2x, Cxy, x_i):
    if not finite(xbar, ybar, M2x, Cxy, x_i):
        raise InvalidInput("non-finite statistic or feature")
    if M2x < 0:
        raise InvalidInput("negative M2x")
    if n == 0:
        return dict(alpha=0.0, beta=0.0, mu=0.0, fallback_empty=True, fallback_degenerate=False)
    if n < 2 or M2x <= n * GUARD:
        return dict(alpha=ybar, beta=0.0, mu=ybar, fallback_empty=False, fallback_degenerate=True)
    beta = Cxy / M2x
    alpha = ybar - beta * xbar
    mu = alpha + beta * x_i
    if not finite(alpha, beta, mu):
        raise InvalidInput("non-finite coefficient or prediction")
    return dict(alpha=alpha, beta=beta, mu=mu, fallback_empty=False, fallback_degenerate=False)


def engine(days, O, C, pool_start=POOL_START, entry_start=ENTRY_START):
    n_days = len(days)
    if pool_start < 1:
        raise InvalidInput("pool must start at or after the first defined feature")
    n, xbar, ybar, M2x, Cxy = 0, 0.0, 0.0, 0.0, 0.0
    updates, warmup = 0, 0
    est = dict(warmup=0, scored=0, pending=0)
    last_label_session = None
    records = []
    for i in range(pool_start, n_days - 1):
        j = i - 2
        if j >= pool_start:
            x, y = feature(C, j), label(O, j)
            if not finite(x, y):
                raise InvalidInput("non-finite training pair")
            n += 1
            dx = x - xbar
            xbar += dx / n
            dy = y - ybar
            ybar += dy / n
            M2x += dx * (x - xbar)
            Cxy += dx * (y - ybar)
            if not finite(xbar, ybar, M2x, Cxy) or M2x < 0:
                raise InvalidInput("invalid running statistic")
            updates += 1
            last_label_session = days[j + 2]
        x_i = feature(C, i)
        co = coefficients(n, xbar, ybar, M2x, Cxy, x_i)
        entry = days[i + 1]
        exit_s = days[i + 2] if i + 2 < n_days else None
        if entry < entry_start:
            warmup += 1
            est["warmup"] += 1
            continue
        est["scored" if exit_s else "pending"] += 1
        records.append(dict(symbol=SYM, decision_index=i, decision_session=days[i], entry_session=entry, exit_session=exit_s, decision_at="session_close (logical marker; information through close i)", execution_not_before="next_session_open (logical marker; no order)", label_available_session=exit_s, target_status="mature_for_scoring" if exit_s else "pending_outcome", x_i=x_i, alpha_i=co["alpha"], beta_i=co["beta"], mu_hat_P4=co["mu"], n_pool=n, xbar=xbar, ybar=ybar, M2x=M2x, Cxy=Cxy, fallback_empty=co["fallback_empty"], fallback_degenerate=co["fallback_degenerate"], train_max_label_available_session=last_label_session, value_unit="simple_return", moment_order=1, forecast_kind="conditional_mean"))
    outcomes = [dict(decision_session=r["decision_session"], entry_session=r["entry_session"], exit_session=r["exit_session"], realized_simple_price_return=label(O, r["decision_index"]), label_available_session=r["exit_session"]) for r in records if r["exit_session"]]
    counts = dict(sessions=n_days, pool_start=pool_start, mature_pair_updates=updates, coefficient_estimates=est, coefficient_estimates_total=sum(est.values()), warmup_decisions=warmup, prediction_records=len(records), mature_scored_records=len(outcomes), pending_outcomes=len(records) - len(outcomes), fallback_empty=sum(r["fallback_empty"] for r in records), fallback_degenerate=sum(r["fallback_degenerate"] for r in records))
    return dict(records=records, outcomes=outcomes, counts=counts)


def pool_indices(i, pool_start=POOL_START):
    return list(range(pool_start, i - 1))


def two_pass(days, O, C, i, pool_start=POOL_START):
    js = pool_indices(i, pool_start)
    n = len(js)
    xs = [feature(C, j) for j in js]
    ys = [label(O, j) for j in js]
    if n == 0:
        xbar, ybar, M2x, Cxy = 0.0, 0.0, 0.0, 0.0
    else:
        xbar, ybar = math.fsum(xs) / n, math.fsum(ys) / n
        M2x = math.fsum((x - xbar) ** 2 for x in xs)
        Cxy = math.fsum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    x_i = feature(C, i)
    co = coefficients(n, xbar, ybar, M2x, Cxy, x_i)
    degenerate = co["fallback_empty"] or co["fallback_degenerate"]
    sd = math.sqrt(M2x / n) if n and not degenerate else None
    max_std = max(abs(x - xbar) for x in xs) / sd if sd else None
    if n and not degenerate:
        hs = [1.0 / n + (x - xbar) ** 2 / M2x for x in xs]
        k = max(range(n), key=lambda t: (hs[t], -t))
        lev = dict(max_hat_diagonal=hs[k], session=days[js[k]], x=xs[k], sum_hat_diagonal=math.fsum(hs))
    elif n:
        lev = dict(max_hat_diagonal=1.0 / n, session=None, x=None, sum_hat_diagonal=None)
    else:
        lev = dict(max_hat_diagonal=None, session=None, x=None, sum_hat_diagonal=None)
    return dict(decision_session=days[i], n=n, xbar=xbar, ybar=ybar, M2x=M2x, Cxy=Cxy, alpha=co["alpha"], beta=co["beta"], mu=co["mu"], degenerate=degenerate, population_sd_x=sd, max_abs_standardized_deviation=max_std, leverage=lev)


def close_enough(a, b):
    return abs(a - b) <= 1e-12 + 1e-10 * max(abs(a), abs(b))


def crosscheck(res, days, O, C, pool_start=POOL_START):
    out = {}
    for tag, r in (("first_stored_prediction", res["records"][0]), ("last_stored_prediction", res["records"][-1])):
        tp = two_pass(days, O, C, r["decision_index"], pool_start)
        dev = dict(alpha=abs(tp["alpha"] - r["alpha_i"]), beta=abs(tp["beta"] - r["beta_i"]), mu=abs(tp["mu"] - r["mu_hat_P4"]), xbar=abs(tp["xbar"] - r["xbar"]), ybar=abs(tp["ybar"] - r["ybar"]), M2x=abs(tp["M2x"] - r["M2x"]), Cxy=abs(tp["Cxy"] - r["Cxy"]))
        ok = tp["n"] == r["n_pool"] and all(close_enough(tp[k], r[m]) for k, m in (("alpha", "alpha_i"), ("beta", "beta_i"), ("mu", "mu_hat_P4")))
        if not ok:
            raise InvalidInput(f"two-pass crosscheck failed at {tag}: {dev}")
        out[tag] = dict(two_pass=tp, running=dict(alpha=r["alpha_i"], beta=r["beta_i"], mu=r["mu_hat_P4"], n=r["n_pool"]), abs_deviation=dev, within_tolerance=True)
    out["crosscheck_count"] = 2
    return out


def anchor_from_obj(obj, kind, expected_sha=None):
    if expected_sha is not None and obj.get("_sha256") != expected_sha:
        raise InvalidInput(f"{kind} anchor hash mismatch")
    recs = obj["prediction_records"]
    seen = set()
    for r in recs:
        if r["decision_session"] in seen:
            raise InvalidInput(f"duplicate {kind} anchor decision {r['decision_session']}")
        seen.add(r["decision_session"])
        if kind == "v1":
            for v in V1_BASELINES:
                if not math.isfinite(float(r["mu_hat"][v])):
                    raise InvalidInput("non-finite v1 anchor mu")
        else:
            if not math.isfinite(float(r["mu_hat_P3"])):
                raise InvalidInput("non-finite v2 anchor mu")
    om = {}
    for o in obj.get("outcome_records", []):
        if o["decision_session"] in om:
            raise InvalidInput(f"duplicate {kind} anchor outcome")
        om[o["decision_session"]] = o
    return dict(kind=kind, artifact_id=obj["envelope"]["artifact_id"], records=recs, outcomes=om, n_records=len(recs), n_outcomes=len(om))


def load_anchor(spec, name, kind):
    p = spec["parents"][name]
    h = sha_full(p["path"])
    if h != p["sha256"]:
        raise InvalidInput(f"{kind} anchor hash changed")
    obj = json.load(open(p["path"], encoding="utf-8"))
    obj["_sha256"] = h
    return anchor_from_obj(obj, kind, expected_sha=h)


def align(res, a1, a2):
    recs = res["records"]
    if len(recs) != a1["n_records"] or len(recs) != a2["n_records"]:
        raise InvalidInput(f"record count {len(recs)} vs anchors {a1['n_records']}/{a2['n_records']}")
    max_mu, max_y = 0.0, 0.0
    for r, x, z in zip(recs, a1["records"], a2["records"]):
        key = (r["decision_index"], r["decision_session"], r["entry_session"], r["exit_session"], r["target_status"] == "pending_outcome")
        if key != (x["decision_index"], x["decision_session"], x["entry_session"], x["exit_session"], x["target_status"] == "pending_outcome"):
            raise InvalidInput(f"v1 anchor session mismatch at {r['decision_session']}")
        if key != (z["decision_index"], z["decision_session"], z["entry_session"], z["exit_session"], z["target_status"] == "pending_outcome"):
            raise InvalidInput(f"v2 anchor session mismatch at {r['decision_session']}")
        if r["n_pool"] != x["n_samples_P1"] or r["n_pool"] != z["n_global"]:
            raise InvalidInput(f"pool count mismatch at {r['decision_session']}")
        d = abs(r["ybar"] - float(x["mu_hat"]["P1_expanding_mean"]))
        max_mu = max(max_mu, d)
        if d > TOL:
            raise InvalidInput(f"ybar differs from stored P1 at {r['decision_session']}: {d}")
    for o in res["outcomes"]:
        a = a1["outcomes"].get(o["decision_session"])
        b = a2["outcomes"].get(o["decision_session"])
        if a is None or b is None or a["exit_session"] != o["exit_session"] or b["exit_session"] != o["exit_session"]:
            raise InvalidInput(f"anchor outcome missing/misaligned at {o['decision_session']}")
        d = abs(o["realized_simple_price_return"] - float(a["realized_simple_price_return"]))
        max_y = max(max_y, d)
        if d > TOL or float(a["realized_simple_price_return"]) != float(b["realized_simple_price_return"]):
            raise InvalidInput(f"target differs from stored outcomes at {o['decision_session']}")
    if len(res["outcomes"]) != a1["n_outcomes"] or len(res["outcomes"]) != a2["n_outcomes"]:
        raise InvalidInput("outcome count mismatch")
    return dict(max_abs_ybar_minus_stored_P1=max_mu, max_abs_target_minus_stored_outcome=max_y, records=len(recs), outcomes=len(res["outcomes"]), anchors=dict(v1=a1["artifact_id"], v2=a2["artifact_id"]))


def pop_sd(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def loaded_mu(a1map, a2map, session):
    x, z = a1map[session], a2map[session]
    return {**{v: float(x["mu_hat"][v]) for v in V1_BASELINES}, P3: float(z["mu_hat_P3"])}


def score(res, a1, a2):
    a1map = {a["decision_session"]: a for a in a1["records"]}
    a2map = {a["decision_session"]: a for a in a2["records"]}
    rows = []
    for r in res["records"]:
        if r["exit_session"] is None:
            continue
        y = float(a1["outcomes"][r["decision_session"]]["realized_simple_price_return"])
        rows.append(dict(entry_year=r["entry_session"][:4], session=r["decision_session"], y=y, beta=r["beta_i"], mu={P4: r["mu_hat_P4"], **loaded_mu(a1map, a2map, r["decision_session"])}))
    if not rows:
        raise InvalidInput("nothing to score")
    years = sorted({x["entry_year"] for x in rows})
    names = (P4,) + BASELINES
    sq = {v: [(x["mu"][v] - x["y"]) ** 2 for x in rows] for v in names}
    yr = [x["entry_year"] for x in rows]
    ys = [x["y"] for x in rows]

    def mean(v):
        return sum(v) / len(v)

    def by_year(vals):
        return {Y: mean([x for x, g in zip(vals, yr) if g == Y]) for Y in years}

    met = dict(n_scored=len(rows), years=years, mse={v: mean(sq[v]) for v in names}, mse_by_year={v: by_year(sq[v]) for v in names}, paired={}, secondary_P4={})
    for b in ("P1_expanding_mean", "P0_zero", P3, "P2_ma200_conditional_mean"):
        d = [x - z for x, z in zip(sq[P4], sq[b])]
        met["paired"][f"{P4}_minus_{b}"] = dict(mean_sq_error_diff=mean(d), by_entry_year=by_year(d), share_rows_P4_better=sum(1 for x in d if x < 0) / len(d), role="primary" if b == "P1_expanding_mean" else "secondary", note="not nested; difference does not isolate the value of magnitude alone" if b == P3 else None)
    mu = [x["mu"][P4] for x in rows]
    elig = [(m, y) for m, y in zip(mu, ys) if m != 0.0 and y != 0.0]
    mse0 = met["mse"]["P0_zero"]
    met["secondary_P4"] = dict(mse_ratio_to_P0=met["mse"][P4] / mse0 if mse0 else None, uncentered_cross_moment=mean([m * y for m, y in zip(mu, ys)]), centered_covariance=mean([m * y for m, y in zip(mu, ys)]) - mean(mu) * mean(ys), sign_accuracy=(sum(1 for m, y in elig if (m > 0) == (y > 0)) / len(elig)) if elig else None, sign_eligible_count=len(elig), zero_target_count=sum(1 for y in ys if y == 0.0), zero_mu_count=sum(1 for m in mu if m == 0.0), mu_mean=mean(mu), mu_pop_sd=pop_sd(mu))
    bmin = min(rows, key=lambda x: (x["beta"], x["session"]))
    bmax = max(rows, key=lambda x: (x["beta"], [-ord(ch) for ch in x["session"]]))
    met["beta_extremes_scored"] = dict(min=dict(beta=bmin["beta"], decision_session=bmin["session"]), max=dict(beta=bmax["beta"], decision_session=bmax["session"]))
    first, last = res["records"][0], res["records"][-1]
    met["coefficient_endpoints"] = {k: dict(decision_session=r["decision_session"], alpha=r["alpha_i"], beta=r["beta_i"], n=r["n_pool"], xbar=r["xbar"], ybar=r["ybar"], M2x=r["M2x"], Cxy=r["Cxy"], x_i=r["x_i"], mu=r["mu_hat_P4"]) for k, r in (("first_stored_prediction", first), ("last_stored_prediction", last))}
    met["note"] = "daily errors are dependent; no p-values or CI; cross moments are not covariance-of-returns or PnL; baselines loaded verbatim from pinned predict-v1/v2 artifacts"
    return met


def fmt_null(x, spec_fmt):
    return format(x, spec_fmt) if x is not None else "해당 없음"


def summary_lines(res, met, al, cc):
    c = res["counts"]
    s = met["secondary_P4"]
    lines = [f"세션 {c['sessions']} | 성숙 쌍 갱신 {c['mature_pair_updates']} | 계수 추정 {c['coefficient_estimates_total']} (워밍업 {c['coefficient_estimates']['warmup']}, 채점 {c['coefficient_estimates']['scored']}, 미확정 {c['coefficient_estimates']['pending']}) | 예측 기록 {c['prediction_records']} | 대체: 빈 {c['fallback_empty']}, 퇴화 {c['fallback_degenerate']} | 앵커 최대차 ȳ−저장P1 {al['max_abs_ybar_minus_stored_P1']:.2e}, 라벨−저장결과 {al['max_abs_target_minus_stored_outcome']:.2e} | 2-pass 검산 {cc['crosscheck_count']}회 통과"]
    acc = f"{s['sign_accuracy']*100:.1f}% ({s['sign_eligible_count']}건)" if s["sign_accuracy"] is not None else "해당 없음"
    lines.append(f"{P4}: MSE {met['mse'][P4]:.6e} | P0 대비 비율 {fmt_null(s['mse_ratio_to_P0'], '.6f')} | μ 평균 {s['mu_mean']:+.3e} sd {s['mu_pop_sd']:.3e} | 비중심 교차모멘트 {s['uncentered_cross_moment']:+.3e} 중심 공분산 {s['centered_covariance']:+.3e} | 부호 적중 {acc} (y=0 {s['zero_target_count']}, μ=0 {s['zero_mu_count']})")
    lines.append("저장 기준 MSE: " + ", ".join(f"{v} {met['mse'][v]:.6e}" for v in BASELINES))
    for k, p in met["paired"].items():
        lines.append(f"{k} [{p['role']}]: 제곱오차 차이 평균 {p['mean_sq_error_diff']:+.3e} (P4 나은 행 {p['share_rows_P4_better']*100:.1f}%) | 진입연도별: " + ", ".join(f"{y} {x:+.2e}" for y, x in p["by_entry_year"].items()))
    for k, e in met["coefficient_endpoints"].items():
        lines.append(f"계수 {k} ({e['decision_session']}): α {e['alpha']:+.4e} β {e['beta']:+.4f} n {e['n']} x̄ {e['xbar']:+.3e} ȳ {e['ybar']:+.3e} M2x {e['M2x']:.4e} Cxy {e['Cxy']:+.4e} x_i {e['x_i']:+.3e} μ {e['mu']:+.3e}")
    be = met["beta_extremes_scored"]
    lines.append(f"채점 구간 β 최소 {be['min']['beta']:+.4f} ({be['min']['decision_session']}) 최대 {be['max']['beta']:+.4f} ({be['max']['decision_session']})")
    tp = cc["last_stored_prediction"]["two_pass"]
    lv = tp["leverage"]
    lines.append(f"끝 풀 진단 ({tp['decision_session']}, n {tp['n']}): x 모표준편차 {fmt_null(tp['population_sd_x'], '.4e')} | 최대 표준화 편차 {fmt_null(tp['max_abs_standardized_deviation'], '.2f')} | 최대 햇 대각 h_jj {fmt_null(lv['max_hat_diagonal'], '.5f')} ({lv['session']}, x {fmt_null(lv['x'], '+.4f')}) | Σh_jj {fmt_null(lv['sum_hat_diagonal'], '.6f')}")
    for k in ("first_stored_prediction", "last_stored_prediction"):
        d = cc[k]["abs_deviation"]
        lines.append(f"2-pass 편차 {k}: α {d['alpha']:.2e} β {d['beta']:.2e} μ {d['mu']:.2e} x̄ {d['xbar']:.2e} ȳ {d['ybar']:.2e} M2x {d['M2x']:.2e} Cxy {d['Cxy']:.2e}")
    return lines


def build_payload(spec, res, met, al, cc, a1, a2, code_sha, now_utc):
    artifact_id = f"{MODULE}_{now_utc.strftime('%Y%m%d_%H%M%S')}_{code_sha[:8]}"
    p = spec["parents"]
    env = dict(schema_version=SCHEMA, artifact_id=artifact_id, parent_artifact_hashes=[p[k]["sha256"] for k in PARENT_ORDER], parent_artifact_roles={k: dict(path=p[k]["path"], sha256=p[k]["sha256"]) for k in PARENT_ORDER}, anchor_artifact_ids=dict(v1=a1["artifact_id"], v2=a2["artifact_id"]), code_sha256=code_sha, spec_sha256=spec["sha256"], asof_utc=now_utc.isoformat(timespec="seconds"), exchange_session=dict(exchange="KRX", symbol=SYM, calendar="raw daily session dates (identical to adjusted)", timezone="Asia/Seoul", utc_mapping="session date D in KST (UTC+09:00); no intraday clock times used", decision_marker="session_close", entry_marker="next_session_open", exit_marker="following_session_open"), symbol=SYM, availability_status="available", decision_at="session_close (logical marker; information through close i only)", execution_not_before="next_session_open (logical marker; no order placed)", target=dict(definition="y_j = O[j+2]/O[j+1] - 1, raw open to following raw open, gross simple price return", value_unit="simple_return", moment_order=1, forecast_kind="conditional_mean", available_at="open of session j+2 (exit_session)"), maturity=dict(rule="pool j >= 199 and j+2 <= i; j = i-1 excluded", update_order="at close i incorporate pair (x_j, y_j) for j = i-2 into the running centered statistics, then compute x_i and emit; x_i joins the statistics only when its own label matures at close i+2"), feature=dict(name="latest_completed_adjusted_close_simple_return", definition="x_i = adjusted C[i]/C[i-1] - 1; no transform"), new_predictor=dict(name=P4, formula="mu_i = alpha_i + beta_i x_i; beta = Cxy/M2x, alpha = ybar - beta xbar over the mature pool", running_statistics="Welford: n+=1; dx=x-xbar; xbar+=dx/n; dy=y-ybar; ybar+=dy/n; M2x+=dx*(x-xbar_new); Cxy+=dx*(y-ybar_new)", fallbacks=dict(empty="n==0 -> alpha=beta=mu=0 (fallback_empty)", degenerate=f"n<2 or M2x<=n*{GUARD:g} -> alpha=ybar, beta=0, mu=ybar (fallback_degenerate); effective fallback coefficients, not identified slopes", invalid="M2x<0 or any non-finite value -> abort invalid_input"), estimation_label="closed-form OLS is fitted estimation at every decision; counted separately from pair updates"), baselines=dict(policy="P0/P1/P2 loaded verbatim from pinned predict-v1, P3 from pinned predict-v2; not recomputed", alignment=al), units=dict(x="simple_return", y="simple_return", alpha="simple_return", mu="simple_return", beta="dimensionless", M2x="simple_return_squared summed over training samples", Cxy="simple_return_squared summed over training samples"), feedback="running statistics change only on mature pairs; outcomes, diagnostics and scores never update the model", assumptions="price return excluding cash distributions/dividends and costs; adjusted series is a current snapshot, not a point-in-time archive; all evaluation years already viewed (exploratory)", statistical_learning=dict(optimizer_calls=0, new_predictors=1, loaded_baselines=4, mature_pair_updates=res["counts"]["mature_pair_updates"], coefficient_estimates=res["counts"]["coefficient_estimates"], fallback_empty=res["counts"]["fallback_empty"], fallback_degenerate=res["counts"]["fallback_degenerate"], numerical_crosschecks=cc["crosscheck_count"]), usage_limit="not combinable with state-vol-v1 h or allocation weights without a separate scope; no exp(mu)-1 conversion", interpretation="exploratory; no expected gain, significance, live profitability or causal magnitude-value claim; P4 vs P3 are not nested", resources=dict(cpu_threads=1, external_timeout_seconds=spec["external_timeout_seconds"], experiments=1, new_predictors=1, loaded_baselines=4, optimizer_calls=0, api_calls=0, orders=0, random_paths=0))
    ref = dict(envelope_artifact_id=artifact_id, anchor_artifact_ids=dict(v1=a1["artifact_id"], v2=a2["artifact_id"]))
    a1map = {a["decision_session"]: a for a in a1["records"]}
    a2map = {a["decision_session"]: a for a in a2["records"]}
    preds = [dict(r, anchor_mu_loaded=loaded_mu(a1map, a2map, r["decision_session"]), **ref) for r in res["records"]]
    outs = [dict(o, **ref) for o in res["outcomes"]]
    return dict(envelope=env, counts=res["counts"], alignment=al, crosscheck=cc, metrics=met, lines=summary_lines(res, met, al, cc), prediction_records=preds, outcome_records=outs)


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
        raise InvalidInput(f"{anchor['kind']} anchor counts differ from calendar facts")
    first = days.index(facts["first_decision"])
    for k, a in enumerate(recs):
        i = first + k
        if a["decision_index"] != i or a["decision_session"] != days[i] or a["entry_session"] != days[i + 1] or a["exit_session"] != (days[i + 2] if i + 2 < len(days) else None):
            raise InvalidInput(f"{anchor['kind']} anchor record {k} misaligned with calendar")
    if recs[-1]["target_status"] != "pending_outcome" or any(a["target_status"] != "mature_for_scoring" for a in recs[:-1]):
        raise InvalidInput(f"{anchor['kind']} anchor pending flags unexpected")
    return dict(kind=anchor["kind"], records=len(recs), outcomes=anchor["n_outcomes"], first=recs[0]["decision_session"], last=recs[-1]["decision_session"])


def run(spec_path=SPEC):
    spec = load_spec(spec_path)
    check_parents(spec)
    days, O, C = load_prices(spec)
    facts = real_calendar_checks(spec, days)
    a1 = load_anchor(spec, "predict_v1_results", "v1")
    a2 = load_anchor(spec, "predict_v2_results", "v2")
    anchor_metadata_checks(a1, days, facts)
    anchor_metadata_checks(a2, days, facts)
    res = engine(days, O, C)
    al = align(res, a1, a2)
    cc = crosscheck(res, days, O, C)
    met = score(res, a1, a2)
    check_parents(spec)
    code_sha = sha_full(os.path.abspath(__file__))
    payload = build_payload(spec, res, met, al, cc, a1, a2, code_sha, datetime.now(timezone.utc))
    payload["calendar_facts"] = facts
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "predict_v3_results.json")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; refusing to overwrite")
    data = json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False).encode("utf-8")
    with open(out, "xb") as f:
        f.write(data)
    print("\n".join(payload["lines"]))
    print("saved", out, "sha", sha_full(out)[:16], "artifact", payload["envelope"]["artifact_id"], "code", code_sha[:16], "spec", spec["sha256"][:16])
    return payload


def synthetic_days(n, start=date(2015, 1, 1)):
    return [(start + timedelta(days=k)).isoformat() for k in range(n)]


def synthetic_anchors(res, p2_shift=1e-5, p3_shift=-2e-5):
    v1 = dict(envelope=dict(artifact_id="predict_v1_synthetic"), prediction_records=[dict(decision_index=r["decision_index"], decision_session=r["decision_session"], entry_session=r["entry_session"], exit_session=r["exit_session"], target_status=r["target_status"], n_samples_P1=r["n_pool"], mu_hat={"P0_zero": 0.0, "P1_expanding_mean": r["ybar"], "P2_ma200_conditional_mean": r["ybar"] + p2_shift}) for r in res["records"]], outcome_records=[dict(o) for o in res["outcomes"]])
    v2 = dict(envelope=dict(artifact_id="predict_v2_synthetic"), prediction_records=[dict(decision_index=r["decision_index"], decision_session=r["decision_session"], entry_session=r["entry_session"], exit_session=r["exit_session"], target_status=r["target_status"], n_global=r["n_pool"], mu_hat_P3=r["ybar"] + p3_shift) for r in res["records"]], outcome_records=[dict(o) for o in res["outcomes"]])
    return v1, v2


def exact_ols(O, C, js, i):
    xs = [Fraction(int(C[j])) / Fraction(int(C[j - 1])) - 1 for j in js]
    ys = [Fraction(int(O[j + 2])) / Fraction(int(O[j + 1])) - 1 for j in js]
    n = len(js)
    xbar, ybar = sum(xs) / n, sum(ys) / n
    M2x = sum((x - xbar) ** 2 for x in xs)
    Cxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    beta = Cxy / M2x
    alpha = ybar - beta * xbar
    x_i = Fraction(int(C[i])) / Fraction(int(C[i - 1])) - 1
    return dict(n=n, xbar=xbar, ybar=ybar, M2x=M2x, Cxy=Cxy, alpha=alpha, beta=beta, mu=alpha + beta * x_i)


def near(a, b, atol=1e-14, rtol=1e-12):
    return abs(a - float(b)) <= atol + rtol * abs(float(b))


def expect_reject(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except InvalidInput:
        return
    raise AssertionError("expected rejection")


def selftest():
    co = coefficients(5, 0.001, 0.0004, 5 * GUARD, 1e-9, 0.01)
    assert co["fallback_degenerate"] and co["beta"] == 0.0 and co["alpha"] == 0.0004 and co["mu"] == 0.0004
    co = coefficients(5, 0.001, 0.0004, 5 * GUARD * (1 + 1e-6), 1e-9, 0.01)
    assert not co["fallback_degenerate"] and co["beta"] == 1e-9 / (5 * GUARD * (1 + 1e-6))
    co = coefficients(1, 0.001, 0.0004, 0.0, 0.0, 0.01)
    assert co["fallback_degenerate"] and not co["fallback_empty"] and co["mu"] == 0.0004
    co = coefficients(0, 0.0, 0.0, 0.0, 0.0, 0.01)
    assert co["fallback_empty"] and not co["fallback_degenerate"] and co["mu"] == 0.0
    expect_reject(coefficients, 5, 0.001, 0.0004, -1e-30, 0.0, 0.01)
    expect_reject(coefficients, 5, 0.001, float("nan"), 1e-5, 0.0, 0.01)
    expect_reject(coefficients, 5, 0.001, 0.0004, 1e-5, float("inf"), 0.01)
    n = 12
    days = synthetic_days(n)
    O = [100.0, 110.0, 121.0, 115.0, 120.0, 126.0, 120.0, 130.0, 128.0, 135.0, 131.0, 140.0]
    C = [100.0, 110.0, 110.0, 105.0, 120.0, 118.0, 120.0, 130.0, 127.0, 133.0, 134.0, 139.0]
    validate_inputs(days, O, C)
    res = engine(days, O, C, pool_start=1, entry_start=days[0])
    rec = {r["decision_index"]: r for r in res["records"]}
    assert sorted(rec) == list(range(1, n - 1)) and res["counts"]["mature_pair_updates"] == n - 4
    assert rec[1]["fallback_empty"] and rec[2]["fallback_empty"] and rec[1]["n_pool"] == 0
    assert rec[3]["n_pool"] == 1 and rec[3]["fallback_degenerate"] and rec[3]["mu_hat_P4"] == rec[3]["ybar"] and rec[3]["beta_i"] == 0.0
    for i in range(4, n - 1):
        js = pool_indices(i, 1)
        assert js == list(range(1, i - 1)) and rec[i]["n_pool"] == len(js) and (i - 1) not in js
        ex = exact_ols(O, C, js, i)
        r = rec[i]
        assert near(r["xbar"], ex["xbar"]) and near(r["ybar"], ex["ybar"]) and near(r["M2x"], ex["M2x"]) and near(r["Cxy"], ex["Cxy"]) and near(r["alpha_i"], ex["alpha"]) and near(r["beta_i"], ex["beta"]) and near(r["mu_hat_P4"], ex["mu"]), (i, r, ex)
        assert not r["fallback_degenerate"] and not r["fallback_empty"]
    assert rec[n - 2]["target_status"] == "pending_outcome" and rec[n - 2]["train_max_label_available_session"] == days[n - 2]
    cc = crosscheck(res, days, O, C, pool_start=1)
    assert cc["crosscheck_count"] == 2 and cc["last_stored_prediction"]["within_tolerance"] and abs(cc["last_stored_prediction"]["two_pass"]["leverage"]["sum_hat_diagonal"] - 2.0) < 1e-12
    tp = two_pass(days, O, C, n - 2, pool_start=1)
    js = pool_indices(n - 2, 1)
    xs = [feature(C, j) for j in js]
    xbar = math.fsum(xs) / len(xs)
    M2x = math.fsum((x - xbar) ** 2 for x in xs)
    hs = [1 / len(xs) + (x - xbar) ** 2 / M2x for x in xs]
    assert abs(tp["leverage"]["max_hat_diagonal"] - max(hs)) < 1e-15 and tp["leverage"]["session"] == days[js[hs.index(max(hs))]] and abs(tp["max_abs_standardized_deviation"] - max(abs(x - xbar) for x in xs) / math.sqrt(M2x / len(xs))) < 1e-12
    a1, a2 = (anchor_from_obj(o, k) for o, k in zip(synthetic_anchors(res), ("v1", "v2")))
    al = align(res, a1, a2)
    assert al["records"] == n - 2 and al["outcomes"] == n - 3 and al["max_abs_ybar_minus_stored_P1"] == 0.0
    met = score(res, a1, a2)
    assert met["n_scored"] == n - 3 and met["paired"][f"{P4}_minus_P1_expanding_mean"]["role"] == "primary" and met["paired"][f"{P4}_minus_{P3}"]["note"] and set(met["mse"]) == {P4, *BASELINES}
    assert met["beta_extremes_scored"]["min"]["beta"] <= met["beta_extremes_scored"]["max"]["beta"]
    for which, mutate in (("v1", lambda v1, v2: v1["prediction_records"][4]["mu_hat"].__setitem__("P1_expanding_mean", v1["prediction_records"][4]["mu_hat"]["P1_expanding_mean"] + 1e-12)), ("v1", lambda v1, v2: v1["outcome_records"][2].__setitem__("realized_simple_price_return", v1["outcome_records"][2]["realized_simple_price_return"] + 1e-12)), ("v2", lambda v1, v2: v2["prediction_records"][3].__setitem__("entry_session", days[0])), ("v2", lambda v1, v2: v2["prediction_records"][3].__setitem__("n_global", 99)), ("v1", lambda v1, v2: v1["prediction_records"][-1].__setitem__("target_status", "mature_for_scoring")), ("v2", lambda v1, v2: v2["outcome_records"][1].__setitem__("realized_simple_price_return", 0.5))):
        v1o, v2o = synthetic_anchors(res)
        mutate(v1o, v2o)
        expect_reject(align, res, anchor_from_obj(v1o, "v1"), anchor_from_obj(v2o, "v2"))
    v1o, v2o = synthetic_anchors(res)
    v2o["prediction_records"].append(dict(v2o["prediction_records"][0]))
    expect_reject(anchor_from_obj, v2o, "v2")
    v1o, v2o = synthetic_anchors(res)
    v1o["_sha256"] = "00" * 32
    expect_reject(anchor_from_obj, v1o, "v1", expected_sha="11" * 32)
    v2o["_sha256"] = "00" * 32
    expect_reject(anchor_from_obj, v2o, "v2", expected_sha="11" * 32)
    Cc = [100.0] * n
    rc = engine(days, O, Cc, pool_start=1, entry_start=days[0])
    assert all(r["fallback_degenerate"] or r["fallback_empty"] for r in rc["records"]) and all(r["M2x"] == 0.0 for r in rc["records"]) and all(r["beta_i"] == 0.0 for r in rc["records"]) and all(r["mu_hat_P4"] == r["ybar"] for r in rc["records"] if r["n_pool"] > 0)
    ccz = crosscheck(rc, days, O, Cc, pool_start=1)
    assert ccz["last_stored_prediction"]["two_pass"]["degenerate"] and ccz["last_stored_prediction"]["two_pass"]["population_sd_x"] is None and ccz["last_stored_prediction"]["two_pass"]["leverage"]["max_hat_diagonal"] == 1.0 / rc["records"][-1]["n_pool"]
    m = 60
    days60 = synthetic_days(m)
    O60 = [float(10000 + 5 * ((k * 37) % 97) + 5 * ((k * 11) % 7)) for k in range(m)]
    C60 = [1000.0 + 3.0 * (((k * 53) % 41) - 20) + 0.5 * k for k in range(m)]
    validate_inputs(days60, O60, C60)
    base = engine(days60, O60, C60, pool_start=1, entry_start=days60[0])

    def keyed(r):
        return {x["decision_index"]: (x["mu_hat_P4"], x["x_i"], x["alpha_i"], x["beta_i"], x["n_pool"], x["ybar"]) for x in r["records"]}

    kb = keyed(base)
    k = 30
    C_after = list(C60)
    for t in range(k + 1, m):
        C_after[t] += 50.0
    pa = keyed(engine(days60, O60, C_after, pool_start=1, entry_start=days60[0]))
    assert all(pa[i] == kb[i] for i in kb if i <= k) and any(pa[i] != kb[i] for i in kb if i > k)
    assert two_pass(days60, O60, C_after, k, 1) == two_pass(days60, O60, C60, k, 1)
    C_k = list(C60)
    C_k[k] += 40.0
    pk = keyed(engine(days60, O60, C_k, pool_start=1, entry_start=days60[0]))
    assert all(pk[i] == kb[i] for i in kb if i < k) and pk[k][1] != kb[k][1] and pk[k + 1][1] != kb[k + 1][1] and pk[k][4] == kb[k][4] and pk[k][5] == kb[k][5] and pk[k + 2][2:4] != kb[k + 2][2:4]
    assert all(r["n_pool"] == max(0, r["decision_index"] - 2) for r in base["records"])
    O_after = list(O60)
    for t in range(k + 1, m):
        O_after[t] += 500.0
    po = keyed(engine(days60, O_after, C60, pool_start=1, entry_start=days60[0]))
    assert all(po[i] == kb[i] for i in kb if i <= k) and any(po[i] != kb[i] for i in kb if i > k)
    O_k = list(O60)
    O_k[k] += 500.0
    pok = keyed(engine(days60, O_k, C60, pool_start=1, entry_start=days60[0]))
    assert all(pok[i] == kb[i] for i in kb if i < k) and pok[k][5] != kb[k][5]
    rwarm = engine(days60, O60, C60, pool_start=1, entry_start=days60[20])
    assert rwarm["counts"]["warmup_decisions"] == 18 and rwarm["counts"]["coefficient_estimates"]["warmup"] == 18 and rwarm["records"][0]["decision_session"] == days60[19] and rwarm["counts"]["mature_pair_updates"] == base["counts"]["mature_pair_updates"] and keyed(rwarm)[25] == kb[25]
    expect_reject(validate_inputs, ["a", "b"], [100.5, 100.0], [1.0, 1.0])
    expect_reject(validate_inputs, ["b", "a"], [100.0, 100.0], [1.0, 1.0])
    spec = load_spec()
    assert spec["sha256"] == SPEC_SHA256 and check_parents(spec) == 5
    daysc = synthetic_days(8)
    rcp = engine(daysc, [100.0] * 8, [100.0] * 8, pool_start=1, entry_start=daysc[0])
    a1c, a2c = (anchor_from_obj(o, kd) for o, kd in zip(synthetic_anchors(rcp), ("v1", "v2")))
    alc = align(rcp, a1c, a2c)
    ccc = crosscheck(rcp, daysc, [100.0] * 8, [100.0] * 8, pool_start=1)
    mc = score(rcp, a1c, a2c)
    assert mc["mse"]["P0_zero"] == 0.0 and mc["secondary_P4"]["mse_ratio_to_P0"] is None and mc["secondary_P4"]["sign_accuracy"] is None and mc["secondary_P4"]["zero_target_count"] == 5
    payc = build_payload(spec, rcp, mc, alc, ccc, a1c, a2c, "ab" * 32, datetime(2026, 11, 6, 8, 50, 0, tzinfo=timezone.utc))
    assert "해당 없음" in payc["lines"][1] and json.loads(json.dumps(payc, allow_nan=False))["metrics"]["secondary_P4"]["mse_ratio_to_P0"] is None and len(payc["outcome_records"]) == 5
    a1b, a2b = (anchor_from_obj(o, kd) for o, kd in zip(synthetic_anchors(base), ("v1", "v2")))
    pay = build_payload(spec, base, score(base, a1b, a2b), align(base, a1b, a2b), crosscheck(base, days60, O60, C60, pool_start=1), a1b, a2b, "ef" * 32, datetime(2026, 11, 6, 8, 51, 0, tzinfo=timezone.utc))
    env = pay["envelope"]
    for key in ("schema_version", "artifact_id", "parent_artifact_hashes", "code_sha256", "spec_sha256", "asof_utc", "exchange_session", "symbol", "availability_status", "decision_at", "execution_not_before"):
        assert key in env, key
    assert env["artifact_id"] == "predict_v3_20261106_085100_efefefef" and env["parent_artifact_hashes"] == [spec["parents"][kk]["sha256"] for kk in PARENT_ORDER] and env["anchor_artifact_ids"] == dict(v1="predict_v1_synthetic", v2="predict_v2_synthetic")
    assert len(pay["prediction_records"]) == 58 and all(r["anchor_mu_loaded"]["P1_expanding_mean"] == r["ybar"] and r["anchor_mu_loaded"][P3] == r["ybar"] - 2e-5 for r in pay["prediction_records"]) and pay["crosscheck"]["crosscheck_count"] == 2
    json.dumps(pay, allow_nan=False)
    rdays, rO, rC = load_prices(spec)
    facts = real_calendar_checks(spec, rdays)
    ra1 = load_anchor(spec, "predict_v1_results", "v1")
    ra2 = load_anchor(spec, "predict_v2_results", "v2")
    m1 = anchor_metadata_checks(ra1, rdays, facts)
    m2 = anchor_metadata_checks(ra2, rdays, facts)
    assert m1["records"] == m2["records"] == 2906 and m1["outcomes"] == m2["outcomes"] == 2905 and m1["first"] == "2014-12-30" and m2["last"] == "2026-11-04" and facts["first_prediction_mature_training_count"] == 2831
    assert check_parents(spec) == 5
    print("selftest ok", dict(code_sha256_16=sha_full(os.path.abspath(__file__))[:16], here=os.path.dirname(os.path.abspath(__file__)), spec_sha256_16=spec["sha256"][:16], real_data="hashes, calendar, counts, stored-anchor alignment only; no real-data estimation"))


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    elif not sys.argv[1:]:
        run()
    else:
        print("usage: predict_v3.py | selftest")
