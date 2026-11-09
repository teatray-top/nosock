import math
from datetime import datetime, timedelta, timezone

VERSION = "council-io-v1"
CONTRACT_SHA256 = "2603f7a077bf6497fb3eb303eaa5582fb1f3b1bda813f537e6962bd303299b8e"
FEATURE_ORDER = ["logret5", "logret20", "logret60", "ma200_distance", "sd20", "logret120", "log_slope20", "drawdown60", "down_count10", "vol_ratio"]
MIN_HISTORY = 200
KST = timezone(timedelta(hours=9))
VINTAGE_MODES = ("point_in_time", "retrospective_latest_vintage")
VALUE_TYPES = ("scalar_score", "probability_vector", "label", "ranked_records", "fraction", "abstain")
ENVELOPE_REQUIRED = ("axis_id", "market_as_of", "decision_at", "decision_not_before", "input_available_at", "source_received_at", "produced_at", "vintage_mode", "completion_source", "question", "value", "value_type", "calibration_status", "target_id", "horizon_sessions", "confidence_basis", "evidence", "provenance")
EVENT_THRESHOLD = -0.05
H = 20
TARGET_HORIZONS = {"E_close20": 20, "E_peak20": 20, "B_teacher_20": 20, "S_generated_state_at_D": None}
CALIBRATION_FIT_SETS = ("train", "validation", "train_validation")
RETRIEVAL_K_MAX = 10
RETRIEVAL_MIN_SEPARATION = 21
RETRIEVAL_TARGETS = ("E_close20",)
TARGET_DECLARED_LAG_SECONDS = {"E_close20": 60, "E_peak20": 60}
AXIS_TARGETS = {
    "B_direction": ("B_teacher_20",),
    "T_state": ("S_generated_state_at_D", None),
    "T_forward_risk": ("E_close20",),
    "F1_fly_state": ("S_generated_state_at_D", None),
    "F1_fly_forward_risk": ("E_close20",),
    "F2_fly_similar_cases": ("E_close20",),
    "R_feasibility_rules": (None,),
    "J_jev_blind_stage1": (None, "E_close20"),
}


class ContractError(ValueError):
    pass


def _finite_positive(xs):
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0 for x in xs)


def _pstd(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def features(closes, D):
    if not isinstance(D, int) or D < 0 or D >= len(closes):
        raise ContractError("D must index an existing close")
    window = closes[: D + 1]
    if not _finite_positive(window):
        raise ContractError("closes up to D must be finite positive numbers")
    if D < MIN_HISTORY - 1:
        return dict(status="abstain", reason=f"insufficient history: need {MIN_HISTORY} closes through D, have {D + 1}", order=FEATURE_ORDER)
    c = [float(x) for x in window]
    lr = lambda k: math.log(c[D] / c[D - k])
    r20 = [math.log(c[t] / c[t - 1]) for t in range(D - 19, D + 1)]
    sd20 = _pstd(r20)
    if sd20 == 0.0:
        return dict(status="abstain", reason="sd20 = 0: vol_ratio denominator undefined", order=FEATURE_ORDER)
    ma200 = sum(c[D - 199: D + 1]) / 200.0
    r60 = [math.log(c[t] / c[t - 1]) for t in range(D - 59, D + 1)]
    slope = sum((j - 9.5) * math.log(c[D - 19 + j]) for j in range(20)) / 665.0
    dd60 = c[D] / max(c[D - 59: D + 1]) - 1.0
    down10 = sum(1 for t in range(D - 9, D + 1) if c[t] < c[t - 1])
    vals = [lr(5), lr(20), lr(60), c[D] / ma200 - 1.0, sd20, lr(120), slope, dd60, float(down10), _pstd(r60) / sd20]
    if not all(math.isfinite(v) for v in vals):
        return dict(status="abstain", reason="non-finite feature", order=FEATURE_ORDER)
    return dict(status="ok", values=vals, order=FEATURE_ORDER, last_row_used=D)


def label_e_close20(closes, D):
    if not isinstance(D, int) or D < 0 or D >= len(closes):
        raise ContractError("D must index an existing close")
    end = D + H
    if end >= len(closes):
        return dict(status="rows_incomplete", target_id="E_close20", label_end_index=end, rows_needed=end + 1, rows_available=len(closes))
    seg = closes[D: end + 1]
    if not _finite_positive(seg):
        raise ContractError("closes in the label window must be finite positive numbers")
    ratios = [closes[s] / closes[D] - 1.0 for s in range(D + 1, end + 1)]
    m = min(ratios)
    return dict(status="rows_complete", target_id="E_close20", event=m <= EVENT_THRESHOLD, min_ratio=m, label_start_index=D + 1, label_end_index=end, definition="min_{s=D+1..D+20} C_s/C_D - 1 <= -0.05", note="raw offline row computation; availability is judged only by gated_label")


def label_e_peak20(closes, D):
    if not isinstance(D, int) or D < 0 or D >= len(closes):
        raise ContractError("D must index an existing close")
    end = D + H
    if end >= len(closes):
        return dict(status="rows_incomplete", target_id="E_peak20", label_end_index=end, rows_needed=end + 1, rows_available=len(closes))
    seg = closes[D: end + 1]
    if not _finite_positive(seg):
        raise ContractError("closes in the label window must be finite positive numbers")
    peak = closes[D]
    dds = []
    for s in range(D + 1, end + 1):
        peak = max(peak, closes[s])
        dds.append(closes[s] / peak - 1.0)
    m = min(dds)
    return dict(status="rows_complete", target_id="E_peak20", event=m <= EVENT_THRESHOLD, min_drawdown=m, drawdowns=dds, label_start_index=D + 1, label_end_index=end, definition="min_{s=D+1..D+20} (C_s / max_{u=D..s} C_u - 1) <= -0.05", note="raw offline row computation; availability is judged only by gated_label")


def label_b_open_endpoints(opens, D):
    if not isinstance(D, int) or D < 0:
        raise ContractError("D must be a non-negative index")
    end = D + H + 1
    if end >= len(opens):
        return dict(status="rows_incomplete", target_id="B_teacher_20", label_end_index=end, rows_needed=end + 1, rows_available=len(opens))
    if not _finite_positive([opens[D + 1], opens[end]]):
        raise ContractError("opens at the endpoints must be finite positive numbers")
    return dict(status="rows_complete", target_id="B_teacher_20", P=opens[D + 1], Q=opens[end], label_start_index=D + 1, label_end_index=end, note="raw offline row computation; availability is judged only by gated_label")


OUTCOME_FIELDS = ("event", "min_ratio", "min_drawdown", "drawdowns", "P", "Q")


def gated_label(raw, mode, event_end_at, decision_at, source_received_at=None, lag_seconds=None):
    dec = parse_time(decision_at, "decision_at")
    if lag_seconds is None:
        lag_seconds = TARGET_DECLARED_LAG_SECONDS.get(raw.get("target_id"))
        if lag_seconds is None:
            raise ContractError(f"no declared confirmation lag for target {raw.get('target_id')!r}; pass lag_seconds explicitly (open-event lag is not the close-bar lag)")
    public = {k: v for k, v in raw.items() if k not in OUTCOME_FIELDS}
    if raw["status"] != "rows_complete":
        public["status"] = "rows_incomplete"
        return public
    av = availability(mode, event_end_at, source_received_at, lag_seconds)
    if parse_time(av["available_at"], "available_at") > dec:
        public["status"] = "not_yet_available"
        public["availability"] = av
        public["decision_at"] = dec.isoformat()
        return public
    out = dict(raw)
    out["status"] = "available"
    out["availability"] = av
    out["decision_at"] = dec.isoformat()
    return out


def parse_time(value, name):
    if not isinstance(value, str):
        raise ContractError(f"{name} must be an ISO 8601 string with offset")
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        raise ContractError(f"{name} {value!r} is not ISO 8601")
    if dt.tzinfo is None:
        raise ContractError(f"{name} must carry a UTC offset")
    return dt


def session_time(day, hms):
    datetime.strptime(day, "%Y-%m-%d")
    h, m, s = hms
    return datetime(int(day[:4]), int(day[5:7]), int(day[8:10]), h, m, s, tzinfo=KST)


def _check_lag(lag_seconds):
    if not isinstance(lag_seconds, (int, float)) or isinstance(lag_seconds, bool) or not math.isfinite(lag_seconds) or lag_seconds < 0:
        raise ContractError("lag_seconds must be a finite non-negative number")


def completion_time(day, calendar, lag_seconds=60):
    _check_lag(lag_seconds)
    entry = calendar.get(day)
    if entry is None or "close_hms" not in entry:
        return dict(status="abstain", reason=f"no calendar entry with a close time for {day}; cannot determine completion")
    close = session_time(day, tuple(entry["close_hms"]))
    return dict(status="ok", close_at=close.isoformat(), completion_at=(close + timedelta(seconds=lag_seconds)).isoformat(), lag_seconds=lag_seconds, completion_source=entry.get("source", "fixture calendar"), lag_applied_once=True)


def availability(mode, event_end_at, source_received_at=None, lag_seconds=60):
    if mode not in VINTAGE_MODES:
        raise ContractError("unknown vintage mode")
    _check_lag(lag_seconds)
    end = parse_time(event_end_at, "event_end_at")
    if mode == "point_in_time":
        if source_received_at is None:
            raise ContractError("point_in_time availability requires source_received_at")
        rec = parse_time(source_received_at, "source_received_at")
        if rec < end:
            raise ContractError("source received before the event ended: inconsistent")
        confirmed = end + timedelta(seconds=lag_seconds)
        return dict(mode=mode, available_at=max(rec, confirmed).isoformat(), kind="observed", source_received_at=rec.isoformat(), confirmation_at=confirmed.isoformat(), lag_seconds=lag_seconds, event_end_at=end.isoformat(), rule="available_at = max(declared confirmation, actual receipt)")
    modeled = end + timedelta(seconds=lag_seconds)
    return dict(mode=mode, available_at=modeled.isoformat(), kind="modeled", lag_seconds=lag_seconds, event_end_at=end.isoformat(), source_received_at=(parse_time(source_received_at, "source_received_at").isoformat() if source_received_at is not None else None), note="modeled replay availability; not observed point-in-time availability")


def envelope(record):
    missing = [k for k in ENVELOPE_REQUIRED if k not in record]
    if missing:
        raise ContractError(f"missing envelope fields: {missing}")
    r = dict(record)
    datetime.strptime(r["market_as_of"], "%Y-%m-%d")
    dec = parse_time(r["decision_at"], "decision_at")
    nb = parse_time(r["decision_not_before"], "decision_not_before")
    if dec < nb:
        raise ContractError("decision_at precedes decision_not_before")
    if r["vintage_mode"] not in VINTAGE_MODES:
        raise ContractError("unknown vintage mode")
    if not isinstance(r["source_received_at"], dict) or not r["source_received_at"]:
        raise ContractError("source_received_at must be a non-empty mapping input -> time")
    received = {k: parse_time(v, f"source_received_at[{k}]") for k, v in r["source_received_at"].items()}
    ia = r["input_available_at"]
    if r["vintage_mode"] == "point_in_time":
        if any(v > dec for v in received.values()):
            raise ContractError("point_in_time: an input was received after decision_at")
        avail = parse_time(ia, "input_available_at")
        if avail != max(received.values()):
            raise ContractError("point_in_time: input_available_at must equal the latest source_received_at")
        r["availability_kind"] = "observed"
    else:
        if not isinstance(ia, dict) or ia.get("kind") != "modeled" or "available_at" not in ia:
            raise ContractError("retrospective_latest_vintage: input_available_at must be a modeled availability record")
        if parse_time(ia["available_at"], "input_available_at.available_at") > dec:
            raise ContractError("retrospective_latest_vintage: modeled input availability is after decision_at")
        r["availability_kind"] = "modeled"
    parse_time(r["produced_at"], "produced_at")
    if r["value_type"] not in VALUE_TYPES:
        raise ContractError("unknown value_type")
    validate_value(r["value_type"], r["value"], r.get("abstain_reason"))
    validate_axis_target(r["axis_id"], r["target_id"])
    validate_calibration(r["calibration_status"], r["market_as_of"], dec, r["target_id"])
    if r["value_type"] == "probability_vector" and r["calibration_status"] == "uncalibrated" and r.get("claims_probability_calibrated"):
        raise ContractError("uncalibrated probabilities cannot claim calibration")
    validate_target_horizon(r["target_id"], r["horizon_sessions"])
    if not isinstance(r["provenance"], dict) or not r["provenance"]:
        raise ContractError("provenance must be a non-empty mapping")
    r["contract_sha256"] = CONTRACT_SHA256
    return r


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_value(value_type, value, abstain_reason=None):
    if value_type == "abstain":
        if value is not None:
            raise ContractError("abstain records carry no value")
        if not abstain_reason:
            raise ContractError("abstain requires abstain_reason")
        return
    if value is None:
        raise ContractError("non-abstain record requires a value")
    if value_type == "scalar_score":
        if not _num(value):
            raise ContractError("scalar_score must be a finite number")
    elif value_type == "fraction":
        if not _num(value) or not 0.0 <= value <= 1.0:
            raise ContractError("fraction must be a finite number in [0, 1]")
    elif value_type == "probability_vector":
        if not isinstance(value, list) or len(value) < 2 or not all(_num(x) and 0.0 <= x <= 1.0 for x in value) or abs(sum(value) - 1.0) > 1e-9:
            raise ContractError("probability_vector must be a list of finite numbers in [0, 1] summing to 1")
    elif value_type == "label":
        if not (isinstance(value, str) or (isinstance(value, dict) and all(isinstance(v, str) for v in value.values()))):
            raise ContractError("label must be a string or a mapping of strings")
    elif value_type == "ranked_records":
        if not isinstance(value, list) or not all(isinstance(x, dict) for x in value):
            raise ContractError("ranked_records must be a list of records")


def validate_axis_target(axis_id, target_id):
    if axis_id not in AXIS_TARGETS:
        raise ContractError(f"unknown axis_id {axis_id!r}")
    if target_id not in AXIS_TARGETS[axis_id]:
        raise ContractError(f"axis {axis_id} does not emit target {target_id!r}")


def validate_calibration(cs, market_as_of, decision_at, target_id):
    if cs in ("uncalibrated", "n/a"):
        return
    if isinstance(cs, str):
        raise ContractError("a calibration claim must be a structured record, not a string prefix")
    if not isinstance(cs, dict) or cs.get("status") != "calibrated":
        raise ContractError("calibration_status must be uncalibrated, n/a or a record with status calibrated")
    for k in ("fit_set", "target_id", "last_anchor_session", "fit_cutoff_at", "max_label_available_at", "method", "frozen_at"):
        if k not in cs:
            raise ContractError(f"calibration record lacks {k}")
    if cs["fit_set"] not in CALIBRATION_FIT_SETS:
        raise ContractError("calibration may be fitted only on train, validation or train_validation; never test")
    if cs["target_id"] != target_id:
        raise ContractError("calibration target differs from the record target")
    datetime.strptime(cs["last_anchor_session"], "%Y-%m-%d")
    cutoff = parse_time(cs["fit_cutoff_at"], "fit_cutoff_at")
    frozen = parse_time(cs["frozen_at"], "frozen_at")
    max_label = parse_time(cs["max_label_available_at"], "max_label_available_at")
    if not (max_label <= cutoff <= frozen <= decision_at):
        raise ContractError("calibration clocks must satisfy max_label_available_at <= fit_cutoff_at <= frozen_at <= decision_at")
    if cs["last_anchor_session"] > cutoff.astimezone(KST).date().isoformat():
        raise ContractError("last calibration anchor session is after the fit cutoff")
    if cs["last_anchor_session"] > market_as_of:
        raise ContractError("last calibration anchor session is after market_as_of")
    if "period_end" in cs:
        datetime.strptime(cs["period_end"], "%Y-%m-%d")
        if cs["period_end"] > cs["last_anchor_session"]:
            raise ContractError("calibration period_end is after the last anchor session")
    if not isinstance(cs["method"], str) or not cs["method"]:
        raise ContractError("calibration method must be a non-empty string")


def validate_target_horizon(target_id, hs):
    if not (hs is None or (isinstance(hs, int) and not isinstance(hs, bool) and hs >= 0)):
        raise ContractError("horizon_sessions must be a non-negative int or null")
    if target_id is None:
        if hs not in (None, 0):
            raise ContractError("state records (no target) must have null or zero horizon")
        return
    if target_id not in TARGET_HORIZONS:
        raise ContractError(f"unknown target_id {target_id!r}")
    expected = TARGET_HORIZONS[target_id]
    if expected is None:
        if hs not in (None, 0):
            raise ContractError("generated-state target must have null or zero horizon")
    elif hs != expected:
        raise ContractError(f"target {target_id} requires horizon_sessions {expected}")


def cosine(a, b):
    if len(a) != len(b) or not a:
        raise ContractError("vector dimension mismatch")
    if not all(_num(x) for x in a) or not all(_num(x) for x in b):
        raise ContractError("non-finite vector")
    sa = max(abs(x) for x in a)
    sb = max(abs(x) for x in b)
    if sa == 0.0 or sb == 0.0:
        raise ContractError("zero vector")
    ua = [x / sa for x in a]
    ub = [x / sb for x in b]
    na = math.sqrt(sum(x * x for x in ua))
    nb = math.sqrt(sum(x * x for x in ub))
    val = sum(x * y for x, y in zip(ua, ub)) / (na * nb)
    if not math.isfinite(val):
        raise ContractError("non-finite cosine")
    return max(-1.0, min(1.0, val))


def _valid_outcome(o):
    return isinstance(o, dict) and isinstance(o.get("event"), bool) and _num(o.get("ret20")) and _num(o.get("max_dd")) and o["max_dd"] >= 0.0


def _identity(x):
    return isinstance(x, str) and x.strip() != ""


def retrieve(query, candidates, decision_at, k=10, min_separation=21, target_id="E_close20"):
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= RETRIEVAL_K_MAX:
        raise ContractError(f"k must be an integer in 1..{RETRIEVAL_K_MAX}")
    if not isinstance(min_separation, int) or isinstance(min_separation, bool) or min_separation < RETRIEVAL_MIN_SEPARATION:
        raise ContractError(f"min_separation must be an integer >= {RETRIEVAL_MIN_SEPARATION}")
    if target_id not in RETRIEVAL_TARGETS:
        raise ContractError(f"retrieval supports only the primary target(s) {RETRIEVAL_TARGETS}")
    horizon = TARGET_HORIZONS[target_id]
    dec = parse_time(decision_at, "decision_at")
    if not isinstance(query.get("session_index"), int) or isinstance(query.get("session_index"), bool):
        raise ContractError("query session_index must be an integer")
    if not _identity(query.get("encoder_id")):
        raise ContractError("query encoder_id must be a non-empty identity")
    qcode = query["code"]
    if not isinstance(qcode, list) or not qcode or not all(_num(x) for x in qcode) or all(x == 0 for x in qcode):
        raise ContractError("query code must be a finite non-zero vector")
    excluded = dict(future_or_same_session=0, endpoint_not_before_query=0, label_immature=0, encoder_mismatch=0, invalid_vector=0, dimension_mismatch=0, target_mismatch=0, outcome_missing_or_invalid=0)
    eligible = []
    for c in candidates:
        if not _identity(c.get("encoder_id")) or c.get("encoder_id") != query["encoder_id"]:
            excluded["encoder_mismatch"] += 1
            continue
        code = c.get("code")
        if not isinstance(code, list) or len(code) != len(qcode):
            excluded["dimension_mismatch"] += 1
            continue
        if not all(_num(x) for x in code) or all(x == 0 for x in code):
            excluded["invalid_vector"] += 1
            continue
        if c.get("target_id") != target_id or c.get("label_end_index") != c.get("session_index", -10 ** 9) + horizon:
            excluded["target_mismatch"] += 1
            continue
        if not isinstance(c.get("session_index"), int) or c["session_index"] >= query["session_index"]:
            excluded["future_or_same_session"] += 1
            continue
        if c["label_end_index"] > query["session_index"]:
            excluded["endpoint_not_before_query"] += 1
            continue
        if parse_time(c["label_available_at"], "label_available_at") > dec:
            excluded["label_immature"] += 1
            continue
        if not _valid_outcome(c.get("outcome")):
            excluded["outcome_missing_or_invalid"] += 1
            continue
        eligible.append(c)
    ranked = sorted(eligible, key=lambda c: (-cosine(qcode, c["code"]), c["session_id"]))
    chosen = []
    for c in ranked:
        if all(abs(c["session_index"] - x["session_index"]) >= min_separation for x in chosen):
            chosen.append(c)
        if len(chosen) == k:
            break
    cases = [dict(session_id=c["session_id"], session_index=c["session_index"], similarity=cosine(qcode, c["code"]), outcome=c["outcome"], max_drawdown_in_window=c["outcome"]["max_dd"], label_end_index=c["label_end_index"], label_available_at=c["label_available_at"]) for c in chosen]
    sims = [x["similarity"] for x in cases]
    spread = (max(sims) - min(sims)) if sims else None
    return dict(encoder_id=query["encoder_id"], target_id=target_id, candidate_count=len(candidates), eligible_count=len(eligible), returned_count=len(cases), k=k, min_separation=min_separation, cases=cases, similarity_spread=spread, excluded=excluded, diagnostics_note="max_drawdown_in_window = max(0, -min drawdown) inside (s, s+20], a diagnostic only; the primary target is E_close20", confidence_note="k and similarity are not calibrated confidence")
