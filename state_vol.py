import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
import sys
from datetime import datetime, timezone

import numpy as np

SPEC = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work\state_vol_v1_spec_20261106.json")
SPEC_SHA256 = "5df26ad21d582694f4a425500727260b3d5f1f8bcfdce17d0c9a17b92fd57225"
OUT_DIR = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work\state_vol_v1_20261106")
SCHEMA = "state-vol-v1"
MODULE = "state_vol"
EXPECT = dict(symbol="069500", cutoff="2026-11-05", calibration_end="2014-12-31", evaluation_start="2015-01-01", lambda_=0.94, initial_n=252, numeric_floor=1e-12, n_returns=5937, calibration_returns=3031, evaluation_returns=2906, first_price_day="2002-10-14", cpu_threads=1)


def sha_full(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_spec(path=SPEC):
    h = sha_full(path)
    if h != SPEC_SHA256:
        raise SystemExit(f"spec hash {h} != {SPEC_SHA256}; refusing")
    s = json.load(open(path, encoding="utf-8"))
    eff = dict(symbol=s["source"]["symbol"], cutoff=s["source"]["cutoff"], calibration_end=s["source"]["calibration_end"], evaluation_start=s["source"]["evaluation_start"], lambda_=float(s["candidate"]["lambda"]), initial_n=int(s["candidate"]["initial_n"]), numeric_floor=float(s["candidate"]["numeric_floor"]), n_returns=int(s["source"]["n_returns"]), calibration_returns=int(s["source"]["calibration_returns"]), evaluation_returns=int(s["source"]["evaluation_returns"]), first_price_day=s["source"]["first_price_day"], cpu_threads=int(s["resources"]["cpu_threads"]))
    bad = {k: (eff[k], EXPECT[k]) for k in EXPECT if eff[k] != EXPECT[k]}
    if bad:
        raise SystemExit(f"spec settings differ from module constants: {bad}")
    if s["version"] != SCHEMA or s["source"]["field"] != "adjusted C":
        raise SystemExit("spec version/field mismatch")
    return dict(path=path, sha256=h, source_path=s["source"]["path"], source_sha256=s["source"]["sha256"], out_dir=s["outputs"]["output_dir"], overwrite=bool(s["outputs"]["overwrite"]), external_timeout_seconds=int(s["resources"]["external_timeout_seconds"]), **eff)


def load(spec):
    p = spec["source_path"]
    h = sha_full(p)
    if h != spec["source_sha256"]:
        raise SystemExit("source hash changed; refusing")
    a = json.load(open(p, encoding="utf-8"))[spec["symbol"]]
    days = [d for d in a["days"] if d <= spec["cutoff"]]
    C = np.array(a["C"][:len(days)], float)
    if days != sorted(days) or len(set(days)) != len(days):
        raise SystemExit("days not sorted/unique")
    if not (np.isfinite(C).all() and (C > 0).all()):
        raise SystemExit("non-finite or non-positive close")
    if days[0] != spec["first_price_day"]:
        raise SystemExit("first price day differs from spec")
    r = np.log(C[1:] / C[:-1])
    if len(r) != spec["n_returns"]:
        raise SystemExit("return count differs from spec")
    return days[1:], r


def split(days, cal_end, ev_start):
    cal = np.array([d <= cal_end for d in days])
    ev = np.array([d >= ev_start for d in days])
    if cal.sum() + ev.sum() != len(days) or (cal & ev).any():
        raise SystemExit("calibration/evaluation masks must partition the returns")
    return cal, ev


def floor_h(h, floor, counter):
    if h < floor:
        counter[0] += 1
        return floor
    return h


def run_variants(days, r, lam, init_n, floor, cal_end, ev_start):
    cal, ev = split(days, cal_end, ev_start)
    r_cal = r[cal]
    ref_h = float(np.mean(r_cal ** 2))
    if len(r_cal) < init_n + 1:
        raise SystemExit("calibration too short")
    h = float(np.mean(r_cal[:init_n] ** 2))
    h_init = h
    for x in r_cal[init_n:]:
        h = lam * h + (1 - lam) * x ** 2
    h_at_eval_start = h
    ev_idx = np.flatnonzero(ev)
    floors = {"reference": [0], "ewma": [0]}
    fc_ref, fc_ewma, prev_sessions = [], [], []
    for i in ev_idx:
        fc_ewma.append(h)
        fc_ref.append(ref_h)
        prev_sessions.append(days[i - 1])
        h = lam * h + (1 - lam) * r[i] ** 2
    fc_ref, fc_ewma = np.array(fc_ref), np.array(fc_ewma)
    r_ev = r[ev_idx]
    out = dict(ref_h=ref_h, h_init=h_init, h_at_eval_start=h_at_eval_start, n_cal=int(cal.sum()), n_eval=int(ev.sum()), eval_days=[days[i] for i in ev_idx], prev_sessions=prev_sessions, r_eval=r_ev, forecasts=dict(reference=fc_ref, ewma=fc_ewma), scored={})
    for name, fc in out["forecasts"].items():
        hs = np.array([floor_h(v, floor, floors[name]) for v in fc])
        q = np.log(hs) + r_ev ** 2 / hs
        out["scored"][name] = dict(qlike=q, ratio=r_ev ** 2 / hs, mse=(r_ev ** 2 - hs) ** 2, h=hs, floors=floors[name][0])
    return out


def metrics(res):
    days_eval = res["eval_days"]
    years = sorted({d[:4] for d in days_eval})
    yr = np.array([d[:4] for d in days_eval])
    m = {}
    for name, s in res["scored"].items():
        m[name] = dict(mean_qlike=float(s["qlike"].mean()), mean_ratio_r2_over_h=float(s["ratio"].mean()), mean_h=float(s["h"].mean()), mean_r2=float((res["r_eval"] ** 2).mean()), mse_vs_r2=float(s["mse"].mean()), floor_count=int(s["floors"]), by_year={y: dict(mean_qlike=float(s["qlike"][yr == y].mean()), mean_ratio=float(s["ratio"][yr == y].mean()), n=int((yr == y).sum())) for y in years})
    d = res["scored"]["ewma"]["qlike"] - res["scored"]["reference"]["qlike"]
    m["paired_diff_ewma_minus_reference"] = dict(mean_qlike_diff=float(d.mean()), by_year={y: float(d[yr == y].mean()) for y in years}, share_days_ewma_better=float((d < 0).mean()), note="daily scores are dependent; no p-value or CI")
    m["n_eval"] = int(len(days_eval))
    return m


def summary_lines(res, met, spec):
    lines = [f"보정 {res['n_cal']}수익(≤{spec['calibration_end']}) 기준 h {res['ref_h']:.4e} (sd {np.sqrt(res['ref_h'])*100:.2f}%) | EWMA 초기 h {res['h_init']:.4e} → 평가 시작 h {res['h_at_eval_start']:.4e} | 평가 {res['eval_days'][0]}~{res['eval_days'][-1]} {res['n_eval']}수익"]
    for name in ("reference", "ewma"):
        mm = met[name]
        lines.append(f"{name}: QLIKE 평균 {mm['mean_qlike']:.4f} | r²/h 평균 {mm['mean_ratio_r2_over_h']:.3f} | h 평균 {mm['mean_h']:.3e} (r² 평균 {mm['mean_r2']:.3e}) | MSE {mm['mse_vs_r2']:.3e} | 바닥 적용 {mm['floor_count']}")
    pd = met["paired_diff_ewma_minus_reference"]
    lines.append(f"EWMA−기준 QLIKE 차이 평균 {pd['mean_qlike_diff']:+.4f} (EWMA가 나은 날 비율 {pd['share_days_ewma_better']*100:.1f}%) | 연도별: " + ", ".join(f"{y} {v:+.3f}" for y, v in pd["by_year"].items()))
    lines.append("연도별 교정비 r²/h (기준 / EWMA): " + ", ".join(f"{y} {met['reference']['by_year'][y]['mean_ratio']:.2f}/{met['ewma']['by_year'][y]['mean_ratio']:.2f}" for y in met["reference"]["by_year"]))
    return lines


def build_payload(spec, res, met, code_sha, now_utc):
    artifact_id = f"{MODULE}_{now_utc.strftime('%Y%m%d_%H%M%S')}_{code_sha[:8]}"
    env = dict(schema_version=SCHEMA, artifact_id=artifact_id, parent_artifact_hashes=[spec["source_sha256"], spec["sha256"]], parent_artifact_roles=dict(source=spec["source_sha256"], spec=spec["sha256"]), code_sha256=code_sha, spec_sha256=spec["sha256"], asof_utc=now_utc.isoformat(timespec="seconds"), exchange_session=dict(exchange="KRX", symbol=spec["symbol"], calendar="session dates as listed in the adjusted daily source series", timezone="Asia/Seoul", utc_mapping="session date D in KST (UTC+09:00); no intraday clock times used or verified", decision_marker="previous_session_close", target_marker="predicted_for_session", execution_marker="next_session_open"), symbol=spec["symbol"], availability_status="available", decision_at="previous_session_close (logical marker; no verified clock time)", execution_not_before="next_session_open (logical marker; no actual fill)", source=dict(path=spec["source_path"], sha256=spec["source_sha256"], field="adjusted C", cutoff=spec["cutoff"], first_price_day=spec["first_price_day"], n_returns=spec["n_returns"]), spec=dict(path=spec["path"], sha256=spec["sha256"]), calibration_end=spec["calibration_end"], evaluation=[res["eval_days"][0], res["eval_days"][-1]], n_cal=res["n_cal"], n_eval=res["n_eval"], assumption="conditional mean fixed at zero; h = E[r^2 | past] under that assumption only; no full-period demeaning", reference=dict(name="constant_calibration_second_moment", h=res["ref_h"]), candidate=dict(name="ewma_fixed_094", lambda_=spec["lambda_"], initial_n=spec["initial_n"], h_initial=res["h_init"], h_at_evaluation_start=res["h_at_eval_start"], numeric_floor=spec["numeric_floor"]), value_unit="log_return", moment_order=2, forecast_kind="conditional_second_moment", limitation="adjusted series is a current provider snapshot, not a proven point-in-time corporate-action archive; all evaluation years were viewed in earlier research (exploratory); better volatility forecast is not a trading edge claim", resources=dict(cpu_threads=spec["cpu_threads"], env_threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}, external_timeout_seconds=spec["external_timeout_seconds"], experiments=1, variants=2, new_fit_optimizations=0, calibration_estimates=2, api_calls=0, orders=0, random_paths=0))
    ref = dict(envelope_artifact_id=artifact_id, source_sha256_16=spec["source_sha256"][:16], code_sha256_16=code_sha[:16], spec_sha256_16=spec["sha256"][:16])
    fc_records = [dict(symbol=spec["symbol"], decision_session=ps, decision_at=env["decision_at"], execution_not_before=env["execution_not_before"], predicted_for_session=d, h_forecast=dict(reference=float(res["forecasts"]["reference"][k]), ewma=float(res["forecasts"]["ewma"][k])), value_unit="log_return", moment_order=2, forecast_kind="conditional_second_moment", **ref) for k, (ps, d) in enumerate(zip(res["prev_sessions"], res["eval_days"]))]
    oc_records = [dict(predicted_for_session=d, realized_log_return=float(x), realized_squared_log_return=float(x * x), **ref) for d, x in zip(res["eval_days"], res["r_eval"])]
    return dict(envelope=env, metrics=met, lines=summary_lines(res, met, spec), forecast_records=fc_records, outcome_records=oc_records)


def run(spec_path=SPEC):
    spec = load_spec(spec_path)
    if spec["out_dir"] != OUT_DIR or spec["overwrite"]:
        raise SystemExit("output settings differ from spec")
    days, r = load(spec)
    cal, ev = split(days, spec["calibration_end"], spec["evaluation_start"])
    if int(cal.sum()) != spec["calibration_returns"] or int(ev.sum()) != spec["evaluation_returns"]:
        raise SystemExit("calibration/evaluation counts differ from spec")
    res = run_variants(days, r, spec["lambda_"], spec["initial_n"], spec["numeric_floor"], spec["calibration_end"], spec["evaluation_start"])
    met = metrics(res)
    code_sha = sha_full(os.path.abspath(__file__))
    payload = build_payload(spec, res, met, code_sha, datetime.now(timezone.utc))
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "state_vol_v1_results.json")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; refusing to overwrite")
    with open(out, "xb") as f:
        f.write(json.dumps(payload, ensure_ascii=False, indent=1).encode("utf-8"))
    print("\n".join(payload["lines"]))
    print("saved", out, "sha", sha_full(out)[:16], "artifact", payload["envelope"]["artifact_id"], "code", code_sha[:16], "spec", spec["sha256"][:16])
    return payload


def selftest():
    days = ["2014-12-29", "2014-12-30", "2014-12-31", "2015-01-02", "2015-01-05", "2015-01-06"]
    r = np.array([0.01, -0.02, 0.005, 0.03, -0.01, 0.02])
    kw = dict(lam=0.5, init_n=2, floor=1e-12, cal_end="2014-12-31", ev_start="2015-01-01")
    res = run_variants(days, r, **kw)
    ref = np.mean(np.array([0.01, -0.02, 0.005]) ** 2)
    assert abs(res["ref_h"] - ref) < 1e-18 and res["n_cal"] == 3 and res["n_eval"] == 3
    h0 = np.mean(np.array([0.01, -0.02]) ** 2)
    h1 = 0.5 * h0 + 0.5 * 0.005 ** 2
    assert abs(res["h_init"] - h0) < 1e-18 and abs(res["h_at_eval_start"] - h1) < 1e-18 and abs(res["forecasts"]["ewma"][0] - h1) < 1e-18
    h2 = 0.5 * h1 + 0.5 * 0.03 ** 2
    h3 = 0.5 * h2 + 0.5 * 0.01 ** 2
    assert np.allclose(res["forecasts"]["ewma"], [h1, h2, h3]) and np.allclose(res["forecasts"]["reference"], [ref] * 3)
    assert res["prev_sessions"] == ["2014-12-31", "2015-01-02", "2015-01-05"] and res["eval_days"] == ["2015-01-02", "2015-01-05", "2015-01-06"]
    q = res["scored"]["ewma"]["qlike"]
    assert abs(q[0] - (np.log(h1) + 0.03 ** 2 / h1)) < 1e-15
    r_p = r.copy()
    r_p[5] = 0.5
    res_p = run_variants(days, r_p, **kw)
    assert np.allclose(res_p["forecasts"]["ewma"], res["forecasts"]["ewma"]) and res_p["ref_h"] == res["ref_h"]
    r_q = r.copy()
    r_q[4] = 0.2
    res_q = run_variants(days, r_q, **kw)
    assert np.allclose(res_q["forecasts"]["ewma"][:2], res["forecasts"]["ewma"][:2]) and res_q["forecasts"]["ewma"][2] != res["forecasts"]["ewma"][2]
    met = metrics(res)
    assert met["n_eval"] == 3 and set(met["paired_diff_ewma_minus_reference"]["by_year"]) == {"2015"} and met["reference"]["floor_count"] == 0
    tiny = run_variants(days, np.array([0.0, 0.0, 0.0, 0.01, 0.0, 0.0]), **kw)
    assert tiny["scored"]["reference"]["floors"] == 3 and tiny["scored"]["ewma"]["floors"] >= 1
    spec = load_spec()
    assert spec["sha256"] == SPEC_SHA256 and len(SPEC_SHA256) == 64
    assert spec["out_dir"] == OUT_DIR and not spec["overwrite"] and spec["cpu_threads"] == 1
    assert all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"))
    days_r, r_r = load(spec)
    assert len(r_r) == 5937 and days_r[0] == "2002-10-15" and days_r[-1] == spec["cutoff"]
    cal_r, ev_r = split(days_r, spec["calibration_end"], spec["evaluation_start"])
    assert cal_r.sum() == 3031 and ev_r.sum() == 2906 and days_r[int(np.flatnonzero(ev_r)[0])] >= spec["evaluation_start"] and days_r[int(np.flatnonzero(cal_r)[-1])] <= spec["calibration_end"]
    fake_spec = dict(spec, lambda_=0.5, initial_n=2, numeric_floor=1e-12, calibration_end="2014-12-31", evaluation_start="2015-01-01")
    pay = build_payload(fake_spec, res, met, "ab" * 32, datetime(2026, 11, 6, 7, 30, 0, tzinfo=timezone.utc))
    env = pay["envelope"]
    for k in ("schema_version", "artifact_id", "parent_artifact_hashes", "code_sha256", "spec_sha256", "asof_utc", "exchange_session", "symbol", "availability_status", "decision_at", "execution_not_before"):
        assert k in env, k
    assert env["artifact_id"] == "state_vol_20261106_073000_abababab" and env["parent_artifact_hashes"] == [spec["source_sha256"], spec["sha256"]] and env["asof_utc"] == "2026-11-06T07:30:00+00:00"
    assert len(pay["forecast_records"]) == 3 and all(fr["envelope_artifact_id"] == env["artifact_id"] and fr["decision_session"] < fr["predicted_for_session"] for fr in pay["forecast_records"])
    assert pay["forecast_records"][0]["h_forecast"]["ewma"] == h1 and pay["outcome_records"][0]["realized_log_return"] == 0.03 and len(pay["lines"]) == 5
    json.dumps(pay)
    print("selftest ok", dict(code_sha256_16=sha_full(os.path.abspath(__file__))[:16], here=os.path.dirname(os.path.abspath(__file__)), n_returns=len(r_r), n_cal=int(cal_r.sum()), n_eval=int(ev_r.sum()), spec_sha256_16=spec["sha256"][:16]))


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    elif not sys.argv[1:]:
        run()
    else:
        print("usage: state_vol.py | selftest")
