import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import date, datetime, timezone, timedelta

WORK = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work")
M2_CODE = os.path.join(WORK, "timing_lgbm_v1_20261106", "timing_lgbm_v1.py")
M2_CODE_SHA = "ab14f47ad1813e480d2bc878feb4de08de334505b61c70bfeb88d871632bccaf"
RESULT = os.path.join(WORK, "mixed_training_v1_20261106", "mixed_training_v1_results.json")
RESULT_SHA = "0b10b9da9328cb6f47c93f86c5e75b3b1b2e50cee4f5e6af342be9ab978c12c4"
MODEL_YEAR = "2026"
SYM = "069500"
TICK = 5
FEE_NUM, FEE_DEN = 15, 100000
VERSION = "dobby-kr-core-v1"
KR_CLOSE_CONFIRMED = "15:31:00+09:00"
KST = timezone(timedelta(hours=9))
LIMITATIONS = [
    "15:31 KST is a regular-session completion assumption; exchange holidays, special trading hours and provider finalization are not verified by this tool",
    "observed_at is a user-provided observation claim; it is not a next-session confirmation and not a fill confirmation",
    "the computed quantity is a file-level calculation under idealized fill assumptions (next open +-5 won, integer fee); it is not an order and no order is placed by this tool",
    "the decision uses the frozen B 2026 fold only; decision sessions outside 2026 or not after the fit cutoff 2025-12-30 are rejected",
    "inputs are files supplied by the caller; this tool performs no data collection, no API call and no account access",
]


def sha_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def fee(notional):
    if not is_int(notional) or notional < 0:
        raise ValueError("fee argument must be a non-negative integer")
    return notional * FEE_NUM // FEE_DEN


def features_fn():
    if sha_file(M2_CODE) != M2_CODE_SHA:
        raise SystemExit("frozen M2 code hash changed")
    sys.dont_write_bytecode = True
    if os.path.dirname(M2_CODE) not in sys.path:
        sys.path.insert(0, os.path.dirname(M2_CODE))
    import timing_lgbm_v1 as m2
    return m2.features


def load_model(result_path=RESULT, expected_sha=RESULT_SHA, year=MODEL_YEAR):
    h = sha_file(result_path)
    if h != expected_sha:
        raise SystemExit(f"frozen result hash changed: {h}")
    import lightgbm as lgb
    fit = json.load(open(result_path, encoding="utf-8"))["fits"][year]
    if sha_bytes(fit["model_string"].encode("utf-8")) != fit["model_string_sha256"]:
        raise SystemExit("model string hash mismatch")
    booster = lgb.Booster(model_str=fit["model_string"])
    if int(booster.num_trees()) != fit["num_trees"]:
        raise SystemExit("tree count mismatch")
    return dict(booster=booster, year=year, fit_cutoff_session=fit["fit_cutoff_session"], first_entry_session=fit["first_entry_session"], model_string_sha256=fit["model_string_sha256"], num_trees=fit["num_trees"], feature_names=list(fit["feature_names"]), result_sha256=h)


def validate_series(raw, adj, min_rows=200):
    for name, s, keys in (("raw", raw, ("days", "O", "C")), ("adj", adj, ("days", "C"))):
        for k in keys:
            if k not in s or not isinstance(s[k], list):
                raise ValueError(f"{name} lacks list field {k}")
        n = len(s["days"])
        if any(len(s[k]) != n for k in keys):
            raise ValueError(f"{name} arrays differ in length")
    days = list(raw["days"])
    if days != list(adj["days"]):
        raise ValueError("raw and adjusted calendars differ")
    for d in days:
        if not isinstance(d, str) or len(d) != 10:
            raise ValueError(f"invalid session date {d!r}")
        date.fromisoformat(d)
    if days != sorted(days) or len(set(days)) != len(days) or len(days) < min_rows:
        raise ValueError("calendar not sorted/unique or too short")
    for o, c in zip(raw["O"], raw["C"]):
        fo, fc = float(o), float(c)
        if not (math.isfinite(fo) and math.isfinite(fc) and fo.is_integer() and fc.is_integer() and fo > TICK and fc > 0 and int(fo) % TICK == 0 and int(fc) % TICK == 0):
            raise ValueError(f"invalid raw price {o} {c}")
    for c in adj["C"]:
        fc = float(c)
        if not (math.isfinite(fc) and fc > 0):
            raise ValueError("invalid adjusted close")
    return days


def validate_account(cash, qty):
    if not is_int(cash) or not is_int(qty) or cash < 0 or qty < 0:
        raise ValueError("cash and qty must be non-negative integers")
    return cash, qty


def validate_price(px, name):
    if not is_int(px) or px <= TICK or px % TICK:
        raise ValueError(f"{name} must be an integer won price above {TICK} on the {TICK}-won tick")
    return px


def signal(model, features, C_adj, held):
    if held not in (0, 1):
        raise ValueError("held must be 0 or 1")
    i = len(C_adj) - 1
    f = features([float(c) for c in C_adj], i)
    if f is None:
        raise ValueError("not enough history for features")
    pred = float(model["booster"].predict([f + [float(held)]], num_threads=1)[0])
    if not math.isfinite(pred):
        raise ValueError("non-finite prediction")
    w = 1 if pred > 0 else 0 if pred < 0 else int(held)
    return dict(pred=pred, w=w, features=dict(zip(model["feature_names"], f + [float(held)])), index=i)


def quantity(cash, qty, w, buy_px, sell_px):
    cash, qty = validate_account(cash, qty)
    if w not in (0, 1):
        raise ValueError("w must be 0 or 1")
    buy_px = validate_price(buy_px, "buy price")
    sell_px = validate_price(sell_px, "sell price")
    if w == 1:
        W = cash + qty * buy_px
        n = W // buy_px - qty
        while n > 0 and n * buy_px + fee(n * buy_px) > cash:
            n -= 1
        if n >= 1:
            f = fee(n * buy_px)
            return dict(side="BUY", qty=n, price=buy_px, fee=f, cash_after=cash - n * buy_px - f, qty_after=qty + n)
        return dict(side=None, qty=0, price=None, fee=0, cash_after=cash, qty_after=qty)
    if qty > 0:
        f = fee(qty * sell_px)
        return dict(side="SELL", qty=qty, price=sell_px, fee=f, cash_after=cash + qty * sell_px - f, qty_after=0)
    return dict(side=None, qty=0, price=None, fee=0, cash_after=cash, qty_after=qty)


def completion(days, fetched_at):
    if not fetched_at:
        raise ValueError("fetched_at (ISO 8601 with offset) is required to judge bar completion")
    f = datetime.fromisoformat(fetched_at)
    if f.tzinfo is None:
        raise ValueError("fetched_at must carry a UTC offset")
    incomplete = []
    for d in reversed(days):
        if f >= datetime.fromisoformat(f"{d}T{KR_CLOSE_CONFIRMED}"):
            break
        incomplete.append(d)
    incomplete.reverse()
    return dict(rule=f"a bar for day D is complete iff fetched_at >= D T{KR_CLOSE_CONFIRMED}; trailing incomplete bars are dropped", fetched_at=f.isoformat(), last_row_day=days[-1], complete=not incomplete, incomplete_days=incomplete)


def _parse_fetched(value, name):
    note = None
    if isinstance(value, str) and len(value) == 19:
        value = value + "+09:00"
        note = "file field carries no offset; assumed Asia/Seoul local time of the fetch process"
    try:
        dt = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} fetch time {value!r} is not ISO 8601")
    if dt.tzinfo is None:
        raise ValueError(f"{name} fetch time has no UTC offset")
    return dt, note


def fetch_sources(raw, adj, fetched_at=None):
    arg = None
    if fetched_at is not None:
        arg, _ = _parse_fetched(fetched_at, "argument")
        if len(fetched_at) == 19:
            raise ValueError("fetched_at argument must carry a UTC offset")
    per_file = {}
    for name, series in (("raw", raw), ("adj", adj)):
        f = series.get("fetched")
        if f:
            dt, note = _parse_fetched(f, name + " file fetched field")
            src = dict(source=f"{name} file fetched field", fetched_at=dt.isoformat(), assumption=note)
            if arg is not None and arg < dt:
                src = dict(source=f"argument (earlier than the {name} file fetched field {dt.isoformat()})", fetched_at=arg.isoformat(), assumption=None)
        elif arg is not None:
            src = dict(source=f"argument substituting for the missing {name} file fetched field", fetched_at=arg.isoformat(), assumption=None)
        else:
            raise ValueError(f"{name} file carries no fetched field and no --fetched-at was given; a fetch time is required per file")
        per_file[name] = src
    eff = min(per_file.values(), key=lambda c: datetime.fromisoformat(c["fetched_at"]))
    return dict(per_file=per_file, all=[dict(file=k, **v) for k, v in per_file.items()], effective=eff["fetched_at"], effective_source=eff["source"], rule="each file needs its own fetch time (file field, or the explicit argument as recorded substitute); the argument may only make a time earlier; the effective time is the earliest across files; no implicit substitution from the other file")


def truncate_incomplete(raw, adj, comp):
    k = len(comp["incomplete_days"])
    if k == 0:
        return dict(days=list(raw["days"]), O=list(raw["O"]), C=list(raw["C"])), dict(days=list(adj["days"]), C=list(adj["C"])), []
    if k >= len(raw["days"]):
        raise ValueError("every row is incomplete relative to the fetch time")
    dropped = [dict(day=raw["days"][i], O=float(raw["O"][i]), C=float(raw["C"][i]), note="dropped: fetched before the confirmed close of this day") for i in range(len(raw["days"]) - k, len(raw["days"]))]
    r = dict(days=raw["days"][:-k], O=raw["O"][:-k], C=raw["C"][:-k])
    a = dict(days=adj["days"][:-k], C=adj["C"][:-k])
    return r, a, dropped


def model_applicability(model, decision_session, fetched_at, now_kst=None):
    now_kst = now_kst or datetime.now(KST)
    f = datetime.fromisoformat(fetched_at)
    if decision_session <= model["fit_cutoff_session"]:
        raise ValueError(f"decision session {decision_session} is not after the model fit cutoff {model['fit_cutoff_session']}")
    if decision_session[:4] != model["year"]:
        raise ValueError(f"decision session {decision_session} is outside the model application year {model['year']}")
    if decision_session > f.date().isoformat() or decision_session > now_kst.date().isoformat():
        raise ValueError(f"decision session {decision_session} is in the future relative to fetch time or now")
    return dict(fit_cutoff_session=model["fit_cutoff_session"], application_year=model["year"], first_entry_session=model["first_entry_session"], checked_against=dict(fetched_date=f.date().isoformat(), now_kst=now_kst.isoformat(timespec="seconds")))


def decide_from_files(raw_path, adj_path, cash, qty, next_open=None, result_path=RESULT, expected_sha=RESULT_SHA, year=MODEL_YEAR, fetched_at=None, decision_session=None, observed_at=None, now_kst=None):
    cash, qty = validate_account(cash, qty)
    raw_full = json.load(open(raw_path, encoding="utf-8"))
    adj_full = json.load(open(adj_path, encoding="utf-8"))
    raw = raw_full.get(SYM, raw_full)
    adj = adj_full.get(SYM, adj_full)
    sources = fetch_sources(raw, adj, fetched_at)
    now_kst = now_kst or datetime.now(KST)
    if datetime.fromisoformat(sources["effective"]) > now_kst:
        raise ValueError(f"fetch time {sources['effective']} is in the future relative to now {now_kst.isoformat(timespec='seconds')}")
    days_all = validate_series(raw, adj)
    comp = completion(days_all, sources["effective"])
    comp["fetched_at_sources"] = sources["all"]
    comp["effective_source"] = sources["effective_source"]
    comp["effective_rule"] = sources["rule"]
    comp["per_file"] = sources["per_file"]
    raw, adj, dropped = truncate_incomplete(raw, adj, comp)
    days = validate_series(raw, adj)
    if decision_session is not None and decision_session != days[-1]:
        raise ValueError(f"requested decision session {decision_session} differs from the last completed session {days[-1]}")
    model = load_model(result_path, expected_sha, year)
    applicability = model_applicability(model, days[-1], comp["fetched_at"], now_kst)
    held = 1 if qty > 0 else 0
    sig = signal(model, features_fn(), adj["C"], held)
    rec = dict(version=VERSION, code_sha256=sha_file(os.path.abspath(__file__)), symbol=SYM, decision_session=days[-1], close=float(raw["C"][-1]), open=float(raw["O"][-1]), rows=len(days), input_source=dict(raw_path=os.path.abspath(raw_path), adj_path=os.path.abspath(adj_path), raw_sha256=sha_file(raw_path), adj_sha256=sha_file(adj_path), fetched_at=comp["fetched_at"], rows_in_file=len(days_all)), bar_completion=comp, dropped_incomplete_rows=dropped, model=dict(year=model["year"], fit_cutoff_session=model["fit_cutoff_session"], model_string_sha256=model["model_string_sha256"], result_sha256=model["result_sha256"], num_trees=model["num_trees"], applicability=applicability), account=dict(cash=cash, qty=qty, held=held), prediction=sig["pred"], weight=sig["w"], features=sig["features"], action="long" if sig["w"] == 1 else "flat", teacher_unit="signed regression of cost-inclusive one-share targets; not pre-cost, not a probability", limitations=LIMITATIONS, created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if next_open is not None:
        O = validate_price(next_open, "next open")
        obs = validate_observed_at(observed_at, days[-1], now_kst)
        q = quantity(cash, qty, sig["w"], O + TICK, O - TICK)
        rec["computed_quantity"] = dict(label="computed quantity under idealized fill assumptions; not an order, not a fill", execution_session="the KR session following the decision session; not determined from the input file", next_open_input=O, price_basis=obs["price_basis"], next_open_observed_at=obs["observed_at"], observed_session=obs["observed_session"], buy_price_assumed=O + TICK, sell_price_assumed=O - TICK, fee_rule="floor(n x 15 / 100000) per notional, no tax for the ETF", sizing_rule="target = floor((cash + qty x buy_price) / buy_price); buy target - qty while affordable including fee; flat sells all", **q)
    return rec


def validate_observed_at(observed_at, decision_session, now_kst):
    if observed_at is None:
        return dict(observed_at=None, observed_session=None, price_basis="assumed: next open supplied without an observation time; hypothetical price computation")
    try:
        dt = datetime.fromisoformat(observed_at)
    except ValueError:
        raise ValueError(f"observed_at {observed_at!r} is not ISO 8601")
    if dt.tzinfo is None:
        raise ValueError("observed_at must carry a UTC offset")
    day = dt.astimezone(KST).date().isoformat()
    if day <= decision_session:
        raise ValueError(f"observed_at {observed_at} is not after the decision session {decision_session}")
    if dt > now_kst:
        raise ValueError(f"observed_at {observed_at} is in the future")
    return dict(observed_at=dt.isoformat(), observed_session=day, price_basis="observed: next open observed at the stated time on the stated session")


def write_record(rec, out_path):
    with open(out_path, "xb") as f:
        f.write(json.dumps(rec, ensure_ascii=False, indent=1).encode("utf-8"))
    return sha_file(out_path)


def expect(fn, exc, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} from {fn.__name__}")


def selftest():
    import tempfile
    assert fee(10000) == 1 and fee(81340) == 12 and fee(0) == 0 and fee(20000) == 3
    expect(fee, ValueError, 1.5)
    expect(fee, ValueError, True)
    q = quantity(100000, 0, 1, 81005, 80995)
    assert q == dict(side="BUY", qty=1, price=81005, fee=12, cash_after=100000 - 81005 - 12, qty_after=1)
    assert quantity(50000, 0, 1, 81005, 80995)["side"] is None
    q = quantity(0, 3, 0, 81005, 80995)
    assert q == dict(side="SELL", qty=3, price=80995, fee=fee(3 * 80995), cash_after=3 * 80995 - fee(3 * 80995), qty_after=0)
    assert quantity(0, 3, 1, 81005, 80995)["side"] is None and quantity(10000, 0, 0, 1005, 995)["side"] is None
    q = quantity(10000, 0, 1, 1005, 995)
    assert q["side"] == "BUY" and q["qty"] == 9 and q["fee"] == 1 and q["cash_after"] == 10000 - 9 * 1005 - 1
    assert quantity(10954, 9, 1, 1005, 995)["qty"] == 10
    expect(quantity, ValueError, -1, 0, 1, 1005, 995)
    expect(quantity, ValueError, 100, -1, 1, 1005, 995)
    expect(quantity, ValueError, 100.0, 0, 1, 1005, 995)
    expect(quantity, ValueError, 100, 2.5, 1, 1005, 995)
    expect(quantity, ValueError, 100, 0, 2, 1005, 995)
    expect(quantity, ValueError, 100, 0, 1, 1003, 995)
    expect(quantity, ValueError, 100, 0, 1, 1005, 0)
    expect(validate_account, ValueError, True, 0)
    base = r"G:\NoSock\stock\penny\data\kr_etf"
    frozen_raw_path, frozen_adj_path = os.path.join(base, "kodex_daily_raw.json"), os.path.join(base, "kodex_daily_adj.json")
    raw = json.load(open(frozen_raw_path, encoding="utf-8"))[SYM]
    adj = json.load(open(frozen_adj_path, encoding="utf-8"))[SYM]
    keep = [i for i, d in enumerate(raw["days"]) if d <= "2026-11-04"]
    tmp = tempfile.mkdtemp()
    rp, ap = os.path.join(tmp, "raw.json"), os.path.join(tmp, "adj.json")
    sub_raw = dict(days=[raw["days"][i] for i in keep], O=[raw["O"][i] for i in keep], C=[raw["C"][i] for i in keep])
    sub_adj = dict(days=[adj["days"][i] for i in keep], C=[adj["C"][i] for i in keep])
    json.dump(sub_raw, open(rp, "w"))
    json.dump(sub_adj, open(ap, "w"))
    res = json.load(open(RESULT, encoding="utf-8"))
    stored = [d for d in res["real_B"]["decision_records"] if d["decision_session"] == "2026-11-04"][0]
    held_qty = 1 if stored["held_at_close"] else 0
    nxt = int(raw["O"][keep[-1] + 1])
    rec = decide_from_files(rp, ap, 100000, held_qty, next_open=nxt, fetched_at="2026-11-04T18:00:00+09:00", decision_session="2026-11-04", observed_at="2026-11-05T09:00:30+09:00")
    assert rec["computed_quantity"]["price_basis"].startswith("observed") and rec["computed_quantity"]["observed_session"] == "2026-11-05"
    rec_noobs = decide_from_files(rp, ap, 100000, held_qty, next_open=nxt, fetched_at="2026-11-04T18:00:00+09:00")
    assert rec_noobs["computed_quantity"]["price_basis"].startswith("assumed") and rec_noobs["computed_quantity"]["next_open_observed_at"] is None
    for bad_obs in ("not-a-time", "2026-11-05T09:00:30", "2026-11-04T09:00:30+09:00", "2026-11-03T09:00:30+09:00", "2099-01-01T09:00:30+09:00"):
        expect(decide_from_files, ValueError, rp, ap, 100000, held_qty, next_open=nxt, fetched_at="2026-11-04T18:00:00+09:00", observed_at=bad_obs)
    expect(decide_from_files, ValueError, rp, ap, 100000, held_qty, fetched_at="2099-01-01T18:00:00+09:00")
    expect(decide_from_files, ValueError, rp, ap, 100000, held_qty, fetched_at="2026-11-04T18:00:00+09:00", now_kst=datetime(2026, 11, 4, 17, 0, tzinfo=KST))
    assert rec["decision_session"] == "2026-11-04" and abs(rec["prediction"] - stored["predicted_contrast"]) <= 1e-9 and (rec["weight"] == 1) == (stored["w"] == 1.0)
    cq = rec["computed_quantity"]
    assert cq["next_open_input"] == nxt and cq["buy_price_assumed"] == nxt + TICK and cq["sell_price_assumed"] == nxt - TICK and "not an order" in cq["label"]
    assert cq == dict(cq, **quantity(100000, held_qty, rec["weight"], nxt + TICK, nxt - TICK))
    assert rec["bar_completion"]["complete"] and rec["dropped_incomplete_rows"] == [] and rec["input_source"]["rows_in_file"] == rec["rows"]
    assert rec["model"]["applicability"]["fit_cutoff_session"] == "2025-12-30"
    rec_mid = decide_from_files(rp, ap, 100000, held_qty, fetched_at="2026-11-04T12:02:00+09:00")
    assert rec_mid["decision_session"] == "2026-11-03" and [d["day"] for d in rec_mid["dropped_incomplete_rows"]] == ["2026-11-04"] and rec_mid["rows"] == rec["rows"] - 1 and "computed_quantity" not in rec_mid
    expect(decide_from_files, ValueError, rp, ap, 100000, held_qty)
    expect(decide_from_files, ValueError, rp, ap, 100000, held_qty, fetched_at="2026-11-04T18:00:00")
    expect(decide_from_files, ValueError, rp, ap, 100000, held_qty, fetched_at="2026-11-04T18:00:00+09:00", decision_session="2026-11-03")
    expect(decide_from_files, ValueError, rp, ap, -1, 0, fetched_at="2026-11-04T18:00:00+09:00")
    expect(decide_from_files, ValueError, rp, ap, 100000, -1, fetched_at="2026-11-04T18:00:00+09:00")
    expect(decide_from_files, ValueError, rp, ap, 100000, 0, next_open=81003, fetched_at="2026-11-04T18:00:00+09:00")
    expect(decide_from_files, ValueError, rp, ap, 100000, 0, fetched_at="2026-11-04T18:00:00+09:00", now_kst=datetime(2026, 11, 3, 18, 0, tzinfo=KST))
    rec_frozen = decide_from_files(frozen_raw_path, frozen_adj_path, 0, 0)
    bc = rec_frozen["bar_completion"]
    assert bc["fetched_at"].startswith("2026-11-06T11:54:53") and not bc["complete"] and bc["effective_source"] == "adj file fetched field" and len(bc["fetched_at_sources"]) == 2
    assert all(c["assumption"] for c in bc["fetched_at_sources"])
    assert rec_frozen["decision_session"] == "2026-11-05" and [d["day"] for d in rec_frozen["dropped_incomplete_rows"]] == ["2026-11-06"]
    raw_only = dict(sub_raw, fetched="2026-11-04T18:00:00")
    rop = os.path.join(tmp, "raw_only_fetched.json")
    json.dump(raw_only, open(rop, "w"))
    expect(decide_from_files, ValueError, rop, ap, 100000, held_qty)
    rec_sub = decide_from_files(rop, ap, 100000, held_qty, fetched_at="2026-11-04T17:00:00+09:00")
    pf = rec_sub["bar_completion"]["per_file"]
    assert pf["raw"]["source"].startswith("argument (earlier than") and pf["adj"]["source"].startswith("argument substituting") and rec_sub["bar_completion"]["fetched_at"] == "2026-11-04T17:00:00+09:00" and rec_sub["decision_session"] == "2026-11-04"
    rec_sub2 = decide_from_files(rop, ap, 100000, held_qty, fetched_at="2026-11-04T19:00:00+09:00")
    pf2 = rec_sub2["bar_completion"]["per_file"]
    assert pf2["raw"]["source"] == "raw file fetched field" and pf2["adj"]["source"].startswith("argument substituting") and rec_sub2["bar_completion"]["fetched_at"] == "2026-11-04T18:00:00+09:00"
    adj_only = dict(sub_adj, fetched="2026-11-04T12:00:00+09:00")
    aop = os.path.join(tmp, "adj_only_fetched.json")
    json.dump(adj_only, open(aop, "w"))
    expect(decide_from_files, ValueError, rp, aop, 100000, held_qty)
    rec_adj = decide_from_files(rp, aop, 100000, held_qty, fetched_at="2026-11-04T18:00:00+09:00")
    assert rec_adj["bar_completion"]["effective_source"] == "adj file fetched field" and rec_adj["decision_session"] == "2026-11-03"
    bad_fetched = dict(sub_raw, fetched="yesterday")
    bfp = os.path.join(tmp, "bad_fetched.json")
    json.dump(bad_fetched, open(bfp, "w"))
    expect(decide_from_files, ValueError, bfp, ap, 100000, held_qty, fetched_at="2026-11-04T18:00:00+09:00")
    assert rec["limitations"] == LIMITATIONS and len(LIMITATIONS) == 5
    rec_late = decide_from_files(frozen_raw_path, frozen_adj_path, 0, 0, fetched_at="2026-11-06T18:00:00+09:00")
    assert rec_late["decision_session"] == "2026-11-05" and rec_late["bar_completion"]["effective_source"] == "adj file fetched field" and len(rec_late["bar_completion"]["fetched_at_sources"]) == 2
    rec_early = decide_from_files(frozen_raw_path, frozen_adj_path, 0, 0, fetched_at="2026-11-05T12:00:00+09:00")
    assert rec_early["decision_session"] == "2026-11-04" and rec_early["bar_completion"]["effective_source"].startswith("argument (earlier than") and [d["day"] for d in rec_early["dropped_incomplete_rows"]] == ["2026-11-05", "2026-11-06"]
    expect(decide_from_files, ValueError, frozen_raw_path, frozen_adj_path, 0, 0, fetched_at="2002-01-01T12:00:00+09:00")
    expect(decide_from_files, ValueError, frozen_raw_path, frozen_adj_path, 0, 0, fetched_at="2026-11-06T18:00:00")
    stored5 = [d for d in res["real_B"]["decision_records"] if d["decision_session"] == "2026-11-05"]
    assert not stored5
    bad_len = dict(days=sub_raw["days"], O=sub_raw["O"][:1], C=sub_raw["C"])
    bp = os.path.join(tmp, "bad_len.json")
    json.dump(bad_len, open(bp, "w"))
    expect(decide_from_files, ValueError, bp, ap, 100000, 0, fetched_at="2026-11-04T18:00:00+09:00")
    bad_date = dict(sub_raw, days=sub_raw["days"][:-1] + ["2026-13-40"])
    bdp = os.path.join(tmp, "bad_date.json")
    json.dump(bad_date, open(bdp, "w"))
    expect(decide_from_files, ValueError, bdp, ap, 100000, 0, fetched_at="2026-11-04T18:00:00+09:00")
    bad_tick = dict(sub_raw)
    bad_tick["C"] = sub_raw["C"][:-1] + [81342]
    btp = os.path.join(tmp, "bad_tick.json")
    json.dump(bad_tick, open(btp, "w"))
    expect(decide_from_files, ValueError, btp, ap, 100000, 0, fetched_at="2026-11-04T18:00:00+09:00")
    short = dict(days=raw["days"][:100], O=raw["O"][:100], C=raw["C"][:100])
    sp = os.path.join(tmp, "short.json")
    json.dump(short, open(sp, "w"))
    expect(decide_from_files, ValueError, sp, sp, 100000, 0, fetched_at="2026-11-04T18:00:00+09:00")
    old_keep = [i for i, d in enumerate(raw["days"]) if d <= "2025-12-30"]
    orp, oap = os.path.join(tmp, "old_raw.json"), os.path.join(tmp, "old_adj.json")
    json.dump(dict(days=[raw["days"][i] for i in old_keep], O=[raw["O"][i] for i in old_keep], C=[raw["C"][i] for i in old_keep]), open(orp, "w"))
    json.dump(dict(days=[adj["days"][i] for i in old_keep], C=[adj["C"][i] for i in old_keep]), open(oap, "w"))
    expect(decide_from_files, ValueError, orp, oap, 100000, 0, fetched_at="2025-12-30T18:00:00+09:00")
    fake_model = dict(fit_cutoff_session="2025-12-30", year="2026", first_entry_session="2026-01-02")
    expect(model_applicability, ValueError, fake_model, "2027-01-05", "2027-01-05T18:00:00+09:00")
    expect(model_applicability, ValueError, fake_model, "2026-11-10", "2026-11-06T18:00:00+09:00")
    assert model_applicability(fake_model, "2026-11-05", "2026-11-06T12:02:16+09:00")["application_year"] == "2026"
    out = os.path.join(tmp, "rec.json")
    write_record(rec, out)
    expect(write_record, FileExistsError, rec, out)
    print("selftest ok (file fixtures only; no API, no order, no job, nothing posted)")
    print(json.dumps(dict(parity_2026_11_04=dict(stored=stored["predicted_contrast"], recomputed=rec["prediction"], w=rec["weight"], computed_quantity={k: cq[k] for k in ("side", "qty", "price", "fee", "cash_after", "qty_after")}), frozen_snapshot=dict(fetched_at=rec_frozen["bar_completion"]["fetched_at"], effective_source=rec_frozen["bar_completion"]["effective_source"], decided_on=rec_frozen["decision_session"], dropped=[d["day"] for d in rec_frozen["dropped_incomplete_rows"]], pred_2026_11_05_held0=rec_frozen["prediction"], w=rec_frozen["weight"]))))


def main():
    ap = argparse.ArgumentParser(description="dobby-kr-core-v1: file-input decision and quantity calculation for 069500 with the frozen B 2026 model. LIMITATIONS: " + " | ".join(LIMITATIONS))
    sub = ap.add_subparsers(dest="cmd")
    d = sub.add_parser("decide")
    d.add_argument("--raw", required=True)
    d.add_argument("--adj", required=True)
    d.add_argument("--cash", type=int, required=True)
    d.add_argument("--qty", type=int, required=True)
    d.add_argument("--next-open", type=int)
    d.add_argument("--observed-at", help="ISO 8601 with offset: when the next open price was observed")
    d.add_argument("--fetched-at", help="ISO 8601 with offset; defaults to the raw file fetched field")
    d.add_argument("--decision-session", help="expected last completed session; mismatch is an error")
    d.add_argument("--out")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "decide":
        rec = decide_from_files(a.raw, a.adj, a.cash, a.qty, a.next_open, fetched_at=a.fetched_at, decision_session=a.decision_session, observed_at=a.observed_at)
        if a.out:
            rec["record_sha256"] = write_record(rec, a.out)
        print(json.dumps(rec, ensure_ascii=False, indent=1))
    else:
        ap.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
