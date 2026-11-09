import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import hashlib
import json
import math
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
KST = ZoneInfo("Asia/Seoul")
SYM = "069500"
FEE_NUM, FEE_DEN = 15, 100000
WORK = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work")
M2_CODE = os.path.join(WORK, "timing_lgbm_v1_20261106", "timing_lgbm_v1.py")
M2_CODE_SHA = "ab14f47ad1813e480d2bc878feb4de08de334505b61c70bfeb88d871632bccaf"
RESULT = os.path.join(WORK, "mixed_training_v1_20261106", "mixed_training_v1_results.json")
RESULT_SHA = "0b10b9da9328cb6f47c93f86c5e75b3b1b2e50cee4f5e6af342be9ab978c12c4"
MODEL_YEAR = "2026"
RULE_CORE = os.path.join(WORK, "dobby_kr_rule_core_v1", "dobby_kr_rule_core.py")
RULE_CORE_SHA = "09d0cf4234ff7d180ccfd9ee17aa295b217b7cf0e04e424716938e045f47b3a9"
SIGNAL_SOURCE = os.environ.get("DOBBY_KR_SIGNAL", "ribbon")
FROZEN_RAW = os.path.join(HERE, "data", "kr_etf", "kodex_daily_raw.json")
FROZEN_ADJ = os.path.join(HERE, "data", "kr_etf", "kodex_daily_adj.json")
LIVE_RAW = os.path.join(HERE, "data", "kr_etf", "live_daily_raw.json")
LIVE_ADJ = os.path.join(HERE, "data", "kr_etf", "live_daily_adj.json")
LEDGER = os.environ.get("DOBBY_KR_LEDGER", os.path.join(HERE, "runs", "dobby_kr_ledger.jsonl"))
LIVE = os.environ.get("DOBBY_LIVE") == "1"
DECIDE_AT = (15, 35)
COMPLETE_AFTER = "15:31:00"
EXEC_FROM = (9, 0, 30)
EXEC_TO = (9, 10, 0)
EXEC_FROM_S = "%02d:%02d:%02d" % EXEC_FROM
EXEC_TO_S = "%02d:%02d:%02d" % EXEC_TO
SESSION_OPEN = "09:00:00"
OB_MAX_AGE = 120.0
DECIDE_RETRY_UNTIL = "15:45:00"
DECIDE_RETRY_EVERY = 120
DAILY_LOSS_FLOOR = 0.97
DRAWDOWN_FLOOR = 0.85
DEPOSIT = 10000
SNAP_DIR = os.path.join(HERE, "data", "kr_etf", "snapshots")
IRC = os.path.join(HERE, "irc.py")
QUIET = os.environ.get("DOBBY_KR_QUIET") == "1"
TEST_FIXTURE = False


def sha_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def fee(notional):
    return int(notional) * FEE_NUM // FEE_DEN


def now_kst():
    return datetime.now(KST)


def say(msg):
    if QUIET or TEST_FIXTURE:
        print("TEST_FIXTURE(not posted):", msg)
        return
    try:
        subprocess.run([sys.executable, "-X", "utf8", IRC, "say", msg], timeout=30, cwd=HERE, capture_output=True)
    except Exception:
        pass


def append(row, path=None):
    path = path or LEDGER
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = dict(ts=now_kst().isoformat(timespec="seconds"), mode="live" if LIVE else "paper", **row)
    if TEST_FIXTURE:
        row["test_fixture"] = True
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def load_ledger(path=None):
    path = path or LEDGER
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def features_fn():
    if sha_file(M2_CODE) != M2_CODE_SHA:
        raise SystemExit("frozen M2 code hash changed")
    sys.dont_write_bytecode = True
    sys.path.insert(0, os.path.dirname(M2_CODE))
    import timing_lgbm_v1 as m2
    return m2.features, m2


def rule_core():
    if sha_file(RULE_CORE) != RULE_CORE_SHA:
        raise SystemExit("rule core hash changed")
    import importlib.util
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("dobby_kr_rule_core_frozen", RULE_CORE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model():
    if sha_file(RESULT) != RESULT_SHA:
        raise SystemExit("frozen result hash changed")
    import lightgbm as lgb
    fit = json.load(open(RESULT, encoding="utf-8"))["fits"][MODEL_YEAR]
    if sha_bytes(fit["model_string"].encode("utf-8")) != fit["model_string_sha256"]:
        raise SystemExit("model string hash mismatch")
    booster = lgb.Booster(model_str=fit["model_string"])
    return dict(booster=booster, year=MODEL_YEAR, fit_cutoff_session=fit["fit_cutoff_session"], model_string_sha256=fit["model_string_sha256"], num_trees=fit["num_trees"], feature_names=fit["feature_names"])


def bar_complete(day, fetched):
    if not fetched or len(fetched) < 19:
        return False
    return day < fetched[:10] or (day == fetched[:10] and fetched[11:19] >= COMPLETE_AFTER)


def seed_rows(d):
    days, O, C = list(d["days"]), [float(x) for x in d["O"]], [float(x) for x in d["C"]]
    dropped = []
    while days and not bar_complete(days[-1], d.get("fetched")):
        dropped.append(dict(day=days.pop(), O=O.pop(), C=C.pop()))
    return dict(days=days, O=O, C=C, seed_fetched=d.get("fetched"), dropped_incomplete=dropped[::-1])


def load_series():
    for src, dst in ((FROZEN_RAW, LIVE_RAW), (FROZEN_ADJ, LIVE_ADJ)):
        if not os.path.exists(dst):
            d = json.load(open(src, encoding="utf-8"))[SYM]
            seeded = seed_rows(d)
            seeded.update(seeded_from=os.path.basename(src), seed_sha256=sha_file(src))
            json.dump(seeded, open(dst, "w", encoding="utf-8"))
    raw = json.load(open(LIVE_RAW, encoding="utf-8"))
    adj = json.load(open(LIVE_ADJ, encoding="utf-8"))
    if raw["days"] != adj["days"]:
        raise RuntimeError("operating raw and adjusted calendars differ")
    return raw, adj


def differs(stored, fetched, exact):
    if exact:
        return tuple(stored) != tuple(fetched)
    return any(abs(a - b) > 1e-6 * max(1.0, abs(b)) for a, b in zip(stored, fetched))


def revise_last(series, rows, exact, label):
    last = series["days"][-1]
    if last not in rows:
        return None
    stored = (series["O"][-1], series["C"][-1])
    if not differs(stored, rows[last], exact):
        return None
    series["O"][-1], series["C"][-1] = float(rows[last][0]), float(rows[last][1])
    rev = dict(file=label, day=last, stored=list(stored), fetched=list(rows[last]))
    append(dict(kind="revision", **rev))
    return rev


def save_series(raw, adj):
    json.dump(raw, open(LIVE_RAW, "w", encoding="utf-8"))
    json.dump(adj, open(LIVE_ADJ, "w", encoding="utf-8"))


def candles_to_rows(candles):
    rows = {}
    for c in candles:
        day = c["timestamp"][:10]
        rows[day] = (float(c["openPrice"]), float(c["closePrice"]))
    return rows


def merge(series, rows, exact, today=None):
    idx = {d: i for i, d in enumerate(series["days"])}
    mismatches, added = [], []
    for day in sorted(rows):
        o, c = rows[day]
        if today and day > today:
            continue
        if day in idx:
            i = idx[day]
            if exact:
                same = series["O"][i] == o and series["C"][i] == c
            else:
                same = abs(series["O"][i] - o) <= 1e-6 * max(1.0, abs(o)) and abs(series["C"][i] - c) <= 1e-6 * max(1.0, abs(c))
            if not same:
                mismatches.append(dict(day=day, stored=(series["O"][i], series["C"][i]), fetched=(o, c)))
        elif day > series["days"][-1]:
            series["days"].append(day)
            series["O"].append(o)
            series["C"].append(c)
            added.append(day)
    return mismatches, added


def fetch_rows(t, adjusted, count=40, before=None):
    r = t.candles(SYM, "1d", count=count, before=before, adjusted=adjusted)
    return candles_to_rows(r["candles"]), r["candles"]


def refetch_adjusted_full(t, first_day):
    rows, before = {}, None
    for _ in range(60):
        page, candles = fetch_rows(t, True, count=200, before=before)
        if not candles:
            break
        rows.update(page)
        oldest = min(c["timestamp"] for c in candles)
        if oldest[:10] <= first_day or len(candles) < 200:
            break
        before = oldest
        time.sleep(0.2)
    return rows


def timed_fetch(t, adjusted, clock):
    req = clock()
    rows, candles = fetch_rows(t, adjusted)
    rec = clock()
    return rows, candles, dict(requested_at=req.isoformat(timespec="milliseconds"), received_at=rec.isoformat(timespec="milliseconds"))


def receipt_complete(receipt, today):
    floor = f"{today}T{COMPLETE_AFTER}"
    return all(receipt[k]["requested_at"][:19] >= floor for k in ("raw", "adj"))


def refresh(t, today, clock=None):
    clock = clock or now_kst
    raw, adj = load_series()
    rraw, craw, rt_raw = timed_fetch(t, False, clock)
    radj, cadj, rt_adj = timed_fetch(t, True, clock)
    receipt = dict(raw=rt_raw, adj=rt_adj)
    fetched_at = rt_raw["requested_at"]
    prov = snapshot_provider(craw, cadj, fetched_at, today, receipt=receipt)
    revisions = [r for r in (revise_last(raw, rraw, True, "raw"), revise_last(adj, radj, False, "adj")) if r]
    if revisions:
        say(f"[dobby_kr] 마지막 저장봉 수정: {json.dumps(revisions, ensure_ascii=False)[:200]}")
    mm_raw, add_raw = merge(raw, rraw, True, today)
    if mm_raw:
        raise RuntimeError(f"raw history mismatch {mm_raw[:3]}")
    mm_adj, add_adj = merge(adj, radj, False, today)
    note = None
    if mm_adj:
        full = refetch_adjusted_full(t, raw["days"][0])
        days = [d for d in sorted(full) if d in set(raw["days"])]
        if days != raw["days"]:
            raise RuntimeError(f"adjusted refetch calendar differs: {len(days)} vs {len(raw['days'])}")
        adj = dict(days=list(days), O=[full[d][0] for d in days], C=[full[d][1] for d in days], refetched=now_kst().isoformat(timespec="seconds"), reason=mm_adj[:3])
        note = f"adjusted history changed on {len(mm_adj)} days; refetched full series"
    if raw["days"] != adj["days"]:
        raise RuntimeError("raw and adjusted calendars differ")
    save_series(raw, adj)
    return raw, adj, dict(added=add_raw, adjusted_note=note, revisions=revisions, provider_snapshot=prov, fetched_at=fetched_at, receipt=receipt)


def signal(model, features, C_adj, held):
    i = len(C_adj) - 1
    f = features(C_adj, i)
    pred = float(model["booster"].predict([f + [float(held)]], num_threads=1)[0])
    if not math.isfinite(pred):
        raise RuntimeError("non-finite prediction")
    w = 1 if pred > 0 else 0 if pred < 0 else int(held)
    return dict(pred=pred, w=w, features=dict(zip(model["feature_names"], f + [float(held)])), index=i)


def account(t):
    cash = float(t.buying_power("KRW")["cashBuyingPower"])
    qty, avg = 0, None
    for it in t.holdings()["items"]:
        if it.get("symbol") == SYM:
            qty = int(float(it["quantity"]))
            avg = float(it["averagePurchasePrice"])
    return dict(cash=cash, qty=qty, avg=avg)


def paper_state(rows):
    st = None
    for r in rows:
        if r.get("kind") == "paper_init":
            st = dict(cash=r["cash"], qty=r["qty"])
        elif r.get("kind") == "paper_buy" and st:
            st["cash"] -= r["cost"]
            st["qty"] += r["qty"]
        elif r.get("kind") == "paper_sell" and st:
            st["cash"] += r["proceeds"]
            st["qty"] -= r["qty"]
        elif r.get("kind") == "paper_deposit" and st:
            st["cash"] += r["amount"]
    return st


def position(t, rows):
    acc = account(t)
    if LIVE:
        return acc, acc
    st = paper_state(rows)
    if st is None:
        st = dict(cash=acc["cash"], qty=acc["qty"])
        append(dict(kind="paper_init", cash=st["cash"], qty=st["qty"], note="paper state seeded from the real account"))
    return acc, st


def account_flows(rows, today, st, close):
    idx = max([i for i, r in enumerate(rows) if r.get("kind") == "decision"], default=-1)
    recorded = [r for r in rows[idx + 1:] if r.get("kind") == "paper_deposit"]
    need = (not LIVE) and not any(r.get("day") == today for r in recorded)
    flows = sum(r["amount"] for r in recorded) + (DEPOSIT if need else 0)
    value = st["cash"] + (DEPOSIT if need else 0) + st["qty"] * close
    return dict(need_deposit=need, flows_since_prev_decision=flows, recorded_deposits=[dict(day=r.get("day"), amount=r["amount"]) for r in recorded], value=value, value_ex_flows=value - flows)


def decide(t, now=None):
    clock = (lambda: now) if now else now_kst
    now = now or now_kst()
    today = now.date().isoformat()
    rows = load_ledger()
    if any(r.get("kind") == "decision" and r.get("day") == today for r in rows):
        return None
    if paused(rows):
        say(f"[dobby_kr] 정지 상태(낙폭 한도) → {today} 판단 생략; 재개는 dobby_kr.py resume")
        return append(dict(kind="paused_skip", day=today))
    if now.strftime("%H:%M:%S") < COMPLETE_AFTER:
        return append(dict(kind="too_early", day=today, now=now.isoformat(timespec="seconds"), complete_after=COMPLETE_AFTER))
    raw, adj, info = refresh(t, today, clock)
    if not receipt_complete(info["receipt"], today):
        return append(dict(kind="incomplete_receipt", day=today, receipt=info["receipt"], complete_after=COMPLETE_AFTER, provider_snapshot=info["provider_snapshot"]))
    if raw["days"][-1] != today:
        return append(dict(kind="no_close_yet", day=today, last_day=raw["days"][-1], fetched_at=info["fetched_at"], receipt=info["receipt"], provider_snapshot=info["provider_snapshot"]))
    snap = snapshot_series(raw, adj, today)
    acc, st = position(t, rows)
    held = 1 if st["qty"] > 0 else 0
    fl = account_flows(rows, today, st, raw["C"][-1])
    deposit = 0
    if fl["need_deposit"]:
        append(dict(kind="paper_deposit", day=today, amount=DEPOSIT, note="paper mirror of the backtest's 10,000 won per close"))
        st["cash"] += DEPOSIT
        deposit = DEPOSIT
    value = st["cash"] + st["qty"] * raw["C"][-1]
    if abs(value - fl["value"]) > 1e-6:
        raise RuntimeError("flow accounting mismatch")
    value_before_deposit = fl["value_ex_flows"]
    lim = apply_limits(rows, value_before_deposit)
    limits = lim["limits"]
    if SIGNAL_SOURCE == "ribbon":
        rc = rule_core()
        state = rc.ribbon_state([float(c) for c in adj["C"]], list(adj["days"]))
        w_rule = state["weight"]
        w = 0 if lim["force_flat"] else w_rule
        desc = f"리본 {state['trend']} (EMA8 {state['emas']['8']:,.0f} / 21 {state['emas']['21']:,.0f} / 50 {state['emas']['50']:,.0f} / 200 {state['emas']['200']:,.0f})"
        row = append(dict(kind="decision", day=today, close=raw["C"][-1], open=raw["O"][-1], held=held, signal_source="ribbon", rule=state["rule"], trend=state["trend"], emas=state["emas"], w_rule=w_rule, w=w, limits=limits, no_buy=lim["no_buy"], period_return=lim["period_return"], unit_index=lim["unit_index"], unit_peak=lim["unit_peak"], prev_value=lim["prev_value"], value_before_deposit=value_before_deposit, value_ex_flows=fl["value_ex_flows"], flows_since_prev_decision=fl["flows_since_prev_decision"], recorded_deposits=fl["recorded_deposits"], deposit=deposit, series_snapshot=snap, cash=st["cash"], qty=st["qty"], value=value, account_cash=acc["cash"], account_qty=acc["qty"], rule_core_sha=RULE_CORE_SHA[:16], added_days=info["added"], adjusted_note=info["adjusted_note"], revisions=info["revisions"], fetched_at=info["fetched_at"], receipt=info["receipt"], provider_snapshot=info["provider_snapshot"], executed=False))
    else:
        features, _ = features_fn()
        model = load_model()
        sig = signal(model, features, adj["C"], held)
        w_rule = sig["w"]
        w = 0 if lim["force_flat"] else w_rule
        desc = f"예측 {sig['pred']:+.5f}"
        row = append(dict(kind="decision", day=today, close=raw["C"][-1], open=raw["O"][-1], held=held, signal_source="model_B_2026", pred=sig["pred"], w_rule=w_rule, w=w, limits=limits, no_buy=lim["no_buy"], period_return=lim["period_return"], unit_index=lim["unit_index"], unit_peak=lim["unit_peak"], prev_value=lim["prev_value"], value_before_deposit=value_before_deposit, value_ex_flows=fl["value_ex_flows"], flows_since_prev_decision=fl["flows_since_prev_decision"], recorded_deposits=fl["recorded_deposits"], deposit=deposit, series_snapshot=snap, features=sig["features"], cash=st["cash"], qty=st["qty"], value=value, account_cash=acc["cash"], account_qty=acc["qty"], model_year=model["year"], model_sha=model["model_string_sha256"][:16], added_days=info["added"], adjusted_note=info["adjusted_note"], revisions=info["revisions"], fetched_at=info["fetched_at"], receipt=info["receipt"], provider_snapshot=info["provider_snapshot"], executed=False))
    act = "매수/보유" if w == 1 else "매도/현금"
    limtxt = f" | 한도 {','.join(limits)} (규칙 w={w_rule}, 매수금지)" if limits else ""
    say(f"[dobby_kr {'LIVE' if LIVE else '모의'}] {today} 종가 {raw['C'][-1]:,.0f} 보유 {st['qty']}주 현금 {st['cash']:,.0f} 평가 {value:,.0f} (기간수익 {lim['period_return']*100:+.2f}%, 단위지수 {lim['unit_index']:.4f}) | {desc} → 다음 시가 {act}{limtxt}")
    if lim["force_flat"]:
        append(dict(kind="pause", day=today, reason="drawdown", unit_index=lim["unit_index"], unit_peak=lim["unit_peak"], floor=DRAWDOWN_FLOOR))
        say(f"[dobby_kr] 낙폭 한도 발동: 단위지수 {lim['unit_index']:.4f} < 고점 {lim['unit_peak']:.4f} × {DRAWDOWN_FLOOR}. 다음 시가 청산 후 정지, 재개는 주인님이 dobby_kr.py resume")
    return row


def pending_decision(rows, today):
    dec = [r for r in rows if r.get("kind") == "decision" and r["day"] < today]
    if not dec:
        return None
    last = dec[-1]
    if any(r.get("decision_day") == last["day"] and (r.get("kind") == "execution" or (r.get("kind") == "execution_skip" and r.get("terminal"))) for r in rows):
        return None
    return last


def session_evidence(ob, received_at, today):
    ts = ob.get("timestamp")
    try:
        obt = datetime.fromisoformat(str(ts))
    except (TypeError, ValueError):
        return False, dict(reason="no parsable orderbook timestamp", ob_ts=ts)
    obt = obt.astimezone(KST)
    age = (received_at - obt).total_seconds()
    detail = dict(ob_ts=obt.isoformat(timespec="milliseconds"), received_at=received_at.isoformat(timespec="milliseconds"), age_seconds=round(age, 3))
    if obt.date().isoformat() != today:
        return False, dict(reason="orderbook timestamp is not today (no session today)", **detail)
    if obt.strftime("%H:%M:%S") < SESSION_OPEN:
        return False, dict(reason="orderbook timestamp before the regular open", **detail)
    if age > OB_MAX_AGE or age < -5.0:
        return False, dict(reason="orderbook timestamp too old or ahead of the receipt clock", **detail)
    return True, dict(reason="orderbook updated today after the open", **detail)


def pause_reason(rows):
    reason = None
    for r in rows:
        if r.get("kind") == "pause":
            reason = r.get("reason") or "unspecified"
        elif r.get("kind") == "resume":
            reason = None
    return reason


def weekdays_between(a, b):
    d = datetime.fromisoformat(a).date() + timedelta(days=1)
    end = datetime.fromisoformat(b).date()
    out = []
    while d < end:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def decision_is_fresh(dec_day, today):
    gap = weekdays_between(dec_day, today)
    if not gap:
        return True, dict(gap=[], reason="next weekday")
    return False, dict(gap=gap, reason="weekday gap between decision and execution; no independent holiday evidence, so the decision is dropped")


def snapshot_provider(craw, cadj, fetched_at, today, out_dir=None, receipt=None):
    out_dir = out_dir or SNAP_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"provider_1d_{today}_{fetched_at[11:23].replace(':', '').replace('.', '')}.json")
    with open(path, "xb") as f:
        f.write(json.dumps(dict(symbol=SYM, fetched_at=fetched_at, receipt=receipt, raw_candles=craw, adj_candles=cadj), ensure_ascii=False).encode("utf-8"))
    return dict(path=os.path.abspath(path), sha256=sha_file(path), fetched_at=fetched_at, receipt=receipt, raw_count=len(craw), adj_count=len(cadj))


def paused(rows):
    state = False
    for r in rows:
        if r.get("kind") == "pause":
            state = True
        elif r.get("kind") == "resume":
            state = False
    return state


def apply_limits(rows, value_before_deposit):
    decs = [r for r in rows if r.get("kind") == "decision" and "unit_index" in r]
    if decs and decs[-1]["value"] > 0:
        prev_value = decs[-1]["value"]
        r = value_before_deposit / prev_value - 1.0
        index = decs[-1]["unit_index"] * (1.0 + r)
        peak = max(decs[-1]["unit_peak"], index)
    else:
        prev_value, r, index, peak = None, 0.0, 1.0, 1.0
    limits = []
    if r < DAILY_LOSS_FLOOR - 1.0:
        limits.append("daily_loss")
    if index < DRAWDOWN_FLOOR * peak:
        limits.append("drawdown")
    return dict(limits=limits, no_buy=bool(limits), force_flat="drawdown" in limits, period_return=r, unit_index=index, unit_peak=peak, prev_value=prev_value)


def order_anomaly(o):
    return o.get("status") != "FILLED"


def snapshot_series(raw, adj, today, out_dir=None):
    out_dir = out_dir or SNAP_DIR
    os.makedirs(out_dir, exist_ok=True)
    stamp = now_kst().strftime("%H%M%S")
    out = {}
    for label, series in (("raw", raw), ("adj", adj)):
        path = os.path.join(out_dir, f"live_daily_{label}_{today}_{stamp}.json")
        with open(path, "xb") as f:
            f.write(json.dumps(series).encode("utf-8"))
        out[label] = dict(path=os.path.abspath(path), sha256=sha_file(path), last_day=series["days"][-1], rows=len(series["days"]))
    return out


def wait_order(t, oid, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        o = t.order(oid)
        if o["status"] in ("FILLED", "CANCELED", "REJECTED"):
            return o
        time.sleep(2)
    return t.order(oid)


def in_window(ts, today):
    return ts.date().isoformat() == today and EXEC_FROM_S <= ts.strftime("%H:%M:%S") <= EXEC_TO_S


def fill_gate(tn, sess, today):
    if not in_window(tn, today):
        return dict(reason="outside execution window before fill", checked_at=tn.isoformat(timespec="milliseconds"), window=[EXEC_FROM_S, EXEC_TO_S])
    age = (tn - datetime.fromisoformat(sess["ob_ts"])).total_seconds()
    if age > OB_MAX_AGE:
        return dict(reason="orderbook too old before fill", checked_at=tn.isoformat(timespec="milliseconds"), age_seconds=round(age, 3), max_age=OB_MAX_AGE)
    return None


def execute(t, now=None, clock=None):
    if clock is None:
        clock = (lambda: now) if now else now_kst
    now = now or clock()
    today = now.date().isoformat()
    rows = load_ledger()
    dec = pending_decision(rows, today)
    acc, st = position(t, rows)
    cash, qty = st["cash"], st["qty"]
    flatten = pause_reason(rows) == "drawdown" and qty > 0
    if dec is None and not flatten:
        return None
    dday = dec["day"] if dec else None
    hms = now.strftime("%H:%M:%S")
    if not (EXEC_FROM_S <= hms <= EXEC_TO_S):
        return append(dict(kind="execution_skip", decision_day=dday, day=today, reason="outside execution window", now=now.isoformat(timespec="seconds"), window=[EXEC_FROM_S, EXEC_TO_S], flatten_pending=flatten))
    ob_req = clock()
    ob = t.orderbook(SYM)
    ob_rec = clock()
    if not in_window(ob_rec, today):
        return append(dict(kind="execution_skip", decision_day=dday, day=today, reason="outside execution window at orderbook receipt", requested_at=ob_req.isoformat(timespec="milliseconds"), received_at=ob_rec.isoformat(timespec="milliseconds"), window=[EXEC_FROM_S, EXEC_TO_S], flatten_pending=flatten))
    sess_ok, sess = session_evidence(ob, ob_rec, today)
    sess["requested_at"] = ob_req.isoformat(timespec="milliseconds")
    if not sess_ok:
        say(f"[dobby_kr] {today} 세션 미확인 → 주문 생략: {sess['reason']}")
        return append(dict(kind="execution_skip", decision_day=dday, day=today, reason="current session not confirmed", detail=sess, flatten_pending=flatten))
    ask = float(ob["asks"][0]["price"]) if ob.get("asks") else None
    bid = float(ob["bids"][0]["price"]) if ob.get("bids") else None
    if ask is None or bid is None:
        return append(dict(kind="execution_skip", decision_day=dday, day=today, reason="no orderbook", session=sess, flatten_pending=flatten))
    side, n, px, f, result = None, 0, None, 0, None
    if flatten:
        side, n, px, w, purpose = "SELL", qty, bid, 0, "drawdown_flatten"
    else:
        fresh, why = decision_is_fresh(dec["day"], today)
        if not fresh:
            say(f"[dobby_kr] {dec['day']} 판단 미실행: {why['reason']} {json.dumps({k: v for k, v in why.items() if k != 'reason'}, ensure_ascii=False)[:160]}")
            return append(dict(kind="execution_skip", decision_day=dec["day"], day=today, reason="stale or unconfirmed decision", detail=why, terminal=True, session=sess))
        if pause_reason(rows) is not None and dec["w"] != 0:
            return append(dict(kind="execution_skip", decision_day=dec["day"], day=today, reason="paused", pause_reason=pause_reason(rows), terminal=True, session=sess))
        w, purpose = dec["w"], "rule"
    if not flatten and w == 1 and dec.get("no_buy"):
        row = append(dict(kind="execution", decision_day=dec["day"], day=today, action="none", reason="no_buy limit", limits=dec.get("limits"), w=w, ask=ask, bid=bid, cash=cash, qty=qty, session=sess))
        say(f"[dobby_kr {'LIVE' if LIVE else '모의'}] {today} 한도({','.join(dec.get('limits') or [])})로 매수 생략, 보유 {qty}주 유지")
        return row
    if flatten:
        pass
    elif w == 1:
        W = cash + qty * ask
        target = math.floor(W / ask)
        n = target - qty
        px = ask
        if n >= 1:
            while n > 0 and n * px + fee(n * px) > cash:
                n -= 1
        if n >= 1:
            side = "BUY"
    elif w == 0 and qty > 0:
        side, n, px = "SELL", qty, bid
    if side is None:
        row = append(dict(kind="execution", decision_day=dday, day=today, action="none", w=w, ask=ask, bid=bid, cash=cash, qty=qty, session=sess))
        say(f"[dobby_kr {'LIVE' if LIVE else '모의'}] {today} 시가 주문 없음 (목표 {'보유' if w == 1 else '현금'}, 보유 {qty}주 현금 {cash:,.0f}, 매수호가 {ask:,.0f})")
        return row
    gate = fill_gate(clock(), sess, today)
    if gate:
        return append(dict(kind="execution_skip", decision_day=dday, day=today, reason=gate["reason"], detail=gate, session=sess, flatten_pending=flatten))
    if not LIVE:
        if side == "BUY":
            f = fee(n * px)
            cost = n * px + f
            append(dict(kind="paper_buy", decision_day=dday, day=today, qty=n, price=px, fee=f, cost=cost))
            result = dict(status="PAPER", qty=n, price=px, fee=f)
        else:
            f = fee(n * px)
            proceeds = n * px - f
            append(dict(kind="paper_sell", decision_day=dday, day=today, qty=n, price=px, fee=f, proceeds=proceeds, purpose=purpose))
            result = dict(status="PAPER", qty=n, price=px, fee=f)
    else:
        if side == "SELL":
            sq = t.get("/api/v1/sellable-quantity", account=True, symbol=SYM)["sellableQuantity"]
            n = int(float(sq))
            if n <= 0:
                return append(dict(kind="execution_skip", decision_day=dday, day=today, reason="no sellable qty", flatten_pending=flatten))
        gate = fill_gate(clock(), sess, today)
        if gate:
            return append(dict(kind="execution_skip", decision_day=dday, day=today, reason=gate["reason"], detail=gate, session=sess, flatten_pending=flatten))
        r = t.place(SYM, side, "MARKET", quantity=str(n), client_order_id="dobbykr-" + uuid.uuid4().hex[:12])
        o = wait_order(t, r["orderId"])
        if order_anomaly(o):
            append(dict(kind="pause", day=today, reason="order_anomaly", order_id=r["orderId"], status=o.get("status")))
            say(f"[dobby_kr LIVE] 주문 이상 상태 {o.get('status')} (주문 {r['orderId']}) → 루프 정지, 재개는 주인님이 dobby_kr.py resume")
        ex = o.get("execution", {})
        result = dict(status=o["status"], order_id=r["orderId"], qty=ex.get("filledQuantity"), price=ex.get("averageFilledPrice"), commission=ex.get("commission"), tax=ex.get("tax"))
    row = append(dict(kind="execution", decision_day=dday, day=today, action=side, purpose=purpose, w=w, ask=ask, bid=bid, cash_before=cash, qty_before=qty, order=result, session=sess))
    tag = " (낙폭 청산)" if purpose == "drawdown_flatten" else ""
    say(f"[dobby_kr {'LIVE' if LIVE else '모의'}] {today} 시가 {side} {n}주 @ {px:,.0f} ({result.get('status')}) 수수료 {result.get('fee', result.get('commission'))}{tag}")
    return row


def status():
    rows = load_ledger()
    dec = [r for r in rows if r.get("kind") == "decision"]
    ex = [r for r in rows if r.get("kind") == "execution"]
    st = paper_state(rows)
    return dict(decisions=len(dec), executions=len(ex), last_decision=dec[-1] if dec else None, last_execution=ex[-1] if ex else None, paper_state=st, mode="live" if LIVE else "paper")


def next_event(now):
    d = now.date()
    cands = []
    for k in range(0, 8):
        day = d + timedelta(days=k)
        if day.weekday() >= 5:
            continue
        for kind, hms in (("execute", EXEC_FROM), ("decide", (DECIDE_AT[0], DECIDE_AT[1], 0))):
            at = datetime(day.year, day.month, day.day, hms[0], hms[1], hms[2], tzinfo=KST)
            if at > now:
                cands.append((at, kind))
    return min(cands)


def loop():
    from toss import Toss
    t = Toss()
    t.account = t.accounts()[0]["accountSeq"]
    append(dict(kind="loop_start", code_sha=sha_file(os.path.abspath(__file__))[:16], model_year=MODEL_YEAR, result_sha=RESULT_SHA[:16]))
    say(f"[dobby_kr {'LIVE' if LIVE else '모의'}] 루프 시작: 15:35 종가 신호 → 다음 거래일 09:00:30 시가 주문, 종목 {SYM}, 신호 {SIGNAL_SOURCE}")
    while True:
        now = now_kst()
        at, kind = next_event(now)
        time.sleep(max(1.0, (at - now).total_seconds()))
        while True:
            res, failed = None, False
            try:
                res = execute(t) if kind == "execute" else decide(t)
            except Exception as e:
                failed = True
                append(dict(kind="error", phase=kind, error=repr(e)[:500]))
                say(f"[dobby_kr] {kind} 오류: {repr(e)[:160]}")
            retry = kind == "decide" and (failed or (res or {}).get("kind") in ("no_close_yet", "incomplete_receipt"))
            if not retry or now_kst().strftime("%H:%M:%S") >= DECIDE_RETRY_UNTIL:
                break
            time.sleep(DECIDE_RETRY_EVERY)
        if kind == "execute":
            time.sleep(60)


def selftest():
    import tempfile
    global LEDGER, TEST_FIXTURE
    TEST_FIXTURE = True
    assert fee(10000) == 1 and fee(81340) == 12 and fee(0) == 0
    s = dict(days=["2026-11-04", "2026-11-05"], O=[100.0, 101.0], C=[100.5, 101.5])
    mm, add = merge(s, {"2026-11-05": (101.0, 101.5), "2026-11-06": (102.0, 103.0), "2026-11-07": (1.0, 1.0)}, True, today="2026-11-06")
    assert mm == [] and add == ["2026-11-06"] and s["days"][-1] == "2026-11-06" and s["C"][-1] == 103.0
    mm, add = merge(s, {"2026-11-05": (101.0, 999.0)}, True)
    assert len(mm) == 1 and add == []
    mm, add = merge(dict(days=["d"], O=[100.0], C=[100.0]), {"d": (100.0000001, 100.0)}, False)
    assert mm == []
    rows = candles_to_rows([{"timestamp": "2026-11-06T00:00:00+09:00", "openPrice": "81000", "closePrice": "81340", "volume": "1"}])
    assert rows == {"2026-11-06": (81000.0, 81340.0)}
    rc = rule_core()
    frozen_adj_chk = json.load(open(FROZEN_ADJ, encoding="utf-8"))[SYM]
    keepR = [i for i, d in enumerate(frozen_adj_chk["days"]) if d <= "2026-11-05"]
    stR = rc.ribbon_state([float(frozen_adj_chk["C"][i]) for i in keepR], [frozen_adj_chk["days"][i] for i in keepR])
    assert stR["weight"] in (0, 1) and stR["trend"] in ("up", "down", "mixed") and set(stR["emas"]) == {"8", "21", "50", "200"} and stR["origin_day"] == "2002-10-14" and stR["last_day"] == "2026-11-05"
    assert bar_complete("2026-11-05", "2026-11-06T12:02:16") and not bar_complete("2026-11-06", "2026-11-06T12:02:16") and bar_complete("2026-11-06", "2026-11-06T15:31:00") and not bar_complete("2026-11-06", None)
    sd = seed_rows(dict(days=["2026-11-04", "2026-11-05", "2026-11-06"], O=[1, 2, 3], C=[1.5, 2.5, 3.5], fetched="2026-11-06T12:02:16"))
    assert sd["days"] == ["2026-11-04", "2026-11-05"] and sd["C"] == [1.5, 2.5] and [d["day"] for d in sd["dropped_incomplete"]] == ["2026-11-06"]
    sd2 = seed_rows(dict(days=["2026-11-05", "2026-11-06"], O=[1, 2], C=[1.5, 2.5], fetched="2026-11-06T15:35:00"))
    assert sd2["days"] == ["2026-11-05", "2026-11-06"] and sd2["dropped_incomplete"] == []
    ser = dict(days=["2026-11-04", "2026-11-05"], O=[100.0, 101.0], C=[100.5, 101.5])
    assert revise_last(ser, {"2026-11-05": (101.0, 101.5)}, True, "raw") is None and revise_last(ser, {"2026-11-04": (1.0, 1.0)}, True, "raw") is None and ser["C"] == [100.5, 101.5]
    assert differs((100.0, 101.0), (100.0, 101.0000001), True) and not differs((100.0, 101.0), (100.0, 101.0000001), False)
    feats, m2 = features_fn()
    model = load_model()
    assert model["num_trees"] == 200 and model["fit_cutoff_session"] == "2025-12-30"
    frozen_adj = json.load(open(FROZEN_ADJ, encoding="utf-8"))[SYM]
    frozen_raw = json.load(open(FROZEN_RAW, encoding="utf-8"))[SYM]
    keep = [i for i, d in enumerate(frozen_adj["days"]) if d <= "2026-11-05"]
    C_adj = [float(frozen_adj["C"][i]) for i in keep]
    res = json.load(open(RESULT, encoding="utf-8"))
    last_dec = [d for d in res["real_B"]["decision_records"] if d["decision_session"] == "2026-11-04"][0]
    C_adj_1104 = C_adj[:-1]
    sig = signal(model, feats, C_adj_1104, last_dec["held_at_close"])
    assert abs(sig["pred"] - last_dec["predicted_contrast"]) <= 1e-9 and (sig["w"] == 1.0) == (last_dec["w"] == 1.0)
    sig0 = signal(model, feats, C_adj, 0)
    sig1 = signal(model, feats, C_adj, 1)
    full_C = [float(x) for x in frozen_adj["C"]]
    sigF0 = signal(model, feats, full_C, 0)
    sigF1 = signal(model, feats, full_C, 1)
    tmp = tempfile.mkdtemp()
    real_ledger = LEDGER
    LEDGER = os.path.join(tmp, "ledger.jsonl")
    real_before = os.path.getsize(real_ledger) if os.path.exists(real_ledger) else None
    append(dict(kind="paper_init", cash=100000.0, qty=0))
    append(dict(kind="decision", day="2026-11-06", w=1, pred=0.01))
    rows = load_ledger()
    st = paper_state(rows)
    assert st == dict(cash=100000.0, qty=0)
    dec = pending_decision(rows, "2026-11-09")
    assert dec and dec["day"] == "2026-11-06"
    assert pending_decision(rows, "2026-11-06") is None
    class FakeToss:
        sessions = []
        clock = None
        ob_ts = None

        def orderbook(self, sym):
            ts = self.ob_ts if self.ob_ts is not None else (self.clock - timedelta(seconds=5)).isoformat()
            return {"timestamp": ts, "asks": [{"price": "81000"}], "bids": [{"price": "80995"}]}

        def candles(self, sym, interval, count=40, before=None, adjusted=False):
            return {"candles": [{"timestamp": f"{d}T00:00:00+09:00", "openPrice": "1", "closePrice": "1", "volume": "1"} for d in self.sessions]}
        def buying_power(self, cur):
            return {"cashBuyingPower": "100000"}
        def holdings(self):
            return {"items": []}
    def ex(ft, dt):
        ft.clock = dt
        return execute(ft, dt)

    def KD(*a):
        return datetime(*a, tzinfo=KST)

    early = decide(FakeToss(), KD(2026, 11, 12, 15, 0, 0))
    assert early["kind"] == "too_early" and early["day"] == "2026-11-12"
    ser2 = dict(days=["2026-11-04", "2026-11-05"], O=[100.0, 101.0], C=[100.5, 101.5])
    rev = revise_last(ser2, {"2026-11-05": (101.0, 102.0)}, True, "raw")
    assert rev["day"] == "2026-11-05" and ser2["C"][-1] == 102.0 and load_ledger()[-1]["kind"] == "revision"
    row = ex(FakeToss(), KD(2026, 11, 9, 9, 0, 40))
    assert row["action"] == "BUY" and row["order"]["qty"] == 1 and row["order"]["price"] == 81000.0 and row["order"]["fee"] == 12 and row["session"]["reason"].startswith("orderbook updated today")
    st = paper_state(load_ledger())
    assert st["qty"] == 1 and st["cash"] == 100000.0 - 81012.0
    assert pending_decision(load_ledger(), "2026-11-09") is None
    append(dict(kind="decision", day="2026-11-09", w=0, pred=-0.01))
    row = ex(FakeToss(), KD(2026, 11, 10, 9, 0, 40))
    assert row["action"] == "SELL" and row["order"]["qty"] == 1 and row["order"]["price"] == 80995.0
    st = paper_state(load_ledger())
    assert st["qty"] == 0 and st["cash"] == 100000.0 - 81012.0 + 80995.0 - 12
    append(dict(kind="decision", day="2026-11-10", w=1, pred=0.01))
    row = ex(FakeToss(), KD(2026, 11, 11, 9, 0, 40))
    assert row["action"] == "BUY" and paper_state(load_ledger())["qty"] == 1
    append(dict(kind="decision", day="2026-11-11", w=0, pred=-0.01))
    late = ex(FakeToss(), KD(2026, 11, 12, 9, 20, 0))
    assert late["kind"] == "execution_skip" and late["reason"] == "outside execution window" and not late.get("terminal")
    early2 = ex(FakeToss(), KD(2026, 11, 12, 8, 59, 59))
    assert early2["kind"] == "execution_skip" and early2["reason"] == "outside execution window"
    ft = FakeToss()
    ft.ob_ts = "2026-11-13T15:30:28.000+09:00"
    sat = ex(ft, KD(2026, 11, 14, 9, 0, 40))
    assert sat["kind"] == "execution_skip" and sat["reason"] == "current session not confirmed" and sat["detail"]["reason"].startswith("orderbook timestamp is not today") and not sat.get("terminal")
    ft.ob_ts = ""
    nots = ex(ft, KD(2026, 11, 12, 9, 0, 40))
    assert nots["reason"] == "current session not confirmed" and nots["detail"]["reason"] == "no parsable orderbook timestamp"
    ft.ob_ts = "2026-11-12T08:59:50.000+09:00"
    pre = ex(ft, KD(2026, 11, 12, 9, 0, 40))
    assert pre["reason"] == "current session not confirmed" and pre["detail"]["reason"] == "orderbook timestamp before the regular open"
    ft.ob_ts = "2026-11-12T09:00:05.000+09:00"
    old = ex(ft, KD(2026, 11, 12, 9, 8, 0))
    assert old["reason"] == "current session not confirmed" and old["detail"]["reason"].startswith("orderbook timestamp too old")
    assert pending_decision(load_ledger(), "2026-11-16")["day"] == "2026-11-11"
    assert weekdays_between("2026-11-06", "2026-11-09") == [] and weekdays_between("2026-11-11", "2026-11-16") == ["2026-11-12", "2026-11-13"]
    ok, why = decision_is_fresh("2026-11-11", "2026-11-16")
    assert not ok and why["gap"] == ["2026-11-12", "2026-11-13"]
    ok, why = decision_is_fresh("2026-11-13", "2026-11-16")
    assert ok and why["gap"] == []
    empty = ex(FakeToss(), KD(2026, 11, 20, 9, 0, 40))
    assert empty["kind"] == "execution_skip" and empty["reason"] == "stale or unconfirmed decision" and empty["terminal"] and paper_state(load_ledger())["qty"] == 1
    assert pending_decision(load_ledger(), "2026-11-23") is None and ex(FakeToss(), KD(2026, 11, 23, 9, 0, 40)) is None
    append(dict(kind="decision", day="2026-11-18", w=0, pred=-0.01))
    append(dict(kind="no_close_yet", day="2026-11-19", last_day="2026-11-18"))
    append(dict(kind="no_close_yet", day="2026-11-20", last_day="2026-11-18"))
    holiday = ex(FakeToss(), KD(2026, 11, 23, 9, 0, 40))
    assert holiday["kind"] == "execution_skip" and holiday["terminal"] and holiday["detail"]["gap"] == ["2026-11-19", "2026-11-20"] and paper_state(load_ledger())["qty"] == 1
    append(dict(kind="decision", day="2026-11-23", w=0, pred=-0.01))

    def seq(*ts):
        it = iter(ts)
        return lambda: next(it)

    ft = FakeToss()
    ft.ob_ts = "2026-11-24T09:10:19.000+09:00"
    g1 = execute(ft, clock=seq(KD(2026, 11, 24, 9, 9, 50), KD(2026, 11, 24, 9, 9, 51), KD(2026, 11, 24, 9, 10, 20)))
    assert g1["kind"] == "execution_skip" and g1["reason"] == "outside execution window at orderbook receipt" and not g1.get("terminal") and paper_state(load_ledger())["qty"] == 1
    ft.ob_ts = "2026-11-24T09:09:51.000+09:00"
    g2 = execute(ft, clock=seq(KD(2026, 11, 24, 9, 9, 50), KD(2026, 11, 24, 9, 9, 51), KD(2026, 11, 24, 9, 9, 52), KD(2026, 11, 24, 9, 10, 5)))
    assert g2["kind"] == "execution_skip" and g2["reason"] == "outside execution window before fill" and paper_state(load_ledger())["qty"] == 1
    ft.ob_ts = "2026-11-24T09:00:41.000+09:00"
    g3 = execute(ft, clock=seq(KD(2026, 11, 24, 9, 0, 40), KD(2026, 11, 24, 9, 0, 41), KD(2026, 11, 24, 9, 0, 42), KD(2026, 11, 24, 9, 2, 50)))
    assert g3["kind"] == "execution_skip" and g3["reason"] == "orderbook too old before fill" and g3["detail"]["age_seconds"] == 129.0 and paper_state(load_ledger())["qty"] == 1
    ft.ob_ts = "2026-11-24T09:00:41.000+09:00"
    g4 = execute(ft, clock=seq(KD(2026, 11, 24, 9, 0, 40), KD(2026, 11, 24, 9, 0, 41), KD(2026, 11, 24, 9, 0, 42), KD(2026, 11, 24, 9, 0, 43)))
    assert g4["kind"] == "execution" and g4["action"] == "SELL" and paper_state(load_ledger())["qty"] == 0
    assert pending_decision(load_ledger(), "2026-11-24") is None
    append(dict(kind="paper_buy", decision_day="fixture-rebuy", day="2026-11-24", qty=1, price=81000.0, fee=12, cost=81012.0))
    r = ex(FakeToss(), KD(2026, 11, 24, 9, 0, 40))
    assert r is None
    append(dict(kind="decision", day="2026-11-24", w=0, pred=-0.01))
    r = ex(FakeToss(), KD(2026, 11, 25, 9, 0, 40))
    assert r["action"] == "SELL" and r["purpose"] == "rule" and paper_state(load_ledger())["qty"] == 0
    append(dict(kind="decision", day="2026-11-25", w=1, pred=0.01))
    assert ex(FakeToss(), KD(2026, 11, 26, 9, 0, 40))["action"] == "BUY" and paper_state(load_ledger())["qty"] == 1
    append(dict(kind="decision", day="2026-11-26", w=0, w_rule=1, limits=["drawdown"], no_buy=True))
    append(dict(kind="pause", day="2026-11-26", reason="drawdown"))
    assert decide(FakeToss(), KD(2026, 11, 27, 15, 35, 0))["kind"] == "paused_skip"
    missed = ex(FakeToss(), KD(2026, 11, 27, 9, 30, 0))
    assert missed["reason"] == "outside execution window" and missed["flatten_pending"]
    flat = ex(FakeToss(), KD(2026, 11, 30, 9, 0, 40))
    assert flat["kind"] == "execution" and flat["action"] == "SELL" and flat["purpose"] == "drawdown_flatten" and flat["decision_day"] == "2026-11-26" and paper_state(load_ledger())["qty"] == 0
    assert ex(FakeToss(), KD(2026, 12, 1, 9, 0, 40)) is None
    append(dict(kind="decision", day="2026-11-27", w=1, pred=0.01))
    pz = ex(FakeToss(), KD(2026, 11, 30, 9, 0, 40))
    assert pz["kind"] == "execution_skip" and pz["reason"] == "paused" and pz["pause_reason"] == "drawdown" and pz["terminal"]
    append(dict(kind="resume", by="owner_cli"))
    assert not paused(load_ledger()) and pause_reason(load_ledger()) is None
    append(dict(kind="pause", day="2026-11-30", reason="order_anomaly"))
    assert pause_reason(load_ledger()) == "order_anomaly"
    append(dict(kind="decision", day="2026-11-30", w=1, pred=0.01))
    pa = ex(FakeToss(), KD(2026, 12, 1, 9, 0, 40))
    assert pa["kind"] == "execution_skip" and pa["reason"] == "paused" and pa["pause_reason"] == "order_anomaly"
    append(dict(kind="resume", by="owner_cli"))
    append(dict(kind="decision", day="2026-12-01", w=1, no_buy=True, limits=["daily_loss"]))
    nb = ex(FakeToss(), KD(2026, 12, 2, 9, 0, 40))
    assert nb["kind"] == "execution" and nb["action"] == "none" and nb["reason"] == "no_buy limit"
    assert pending_decision(load_ledger(), "2026-12-03") is None
    pv = snapshot_provider([{"timestamp": "2026-11-11T00:00:00+09:00"}], [{"timestamp": "2026-11-11T00:00:00+09:00"}], "2026-11-16T15:35:01.123+09:00", "2026-11-16", out_dir=os.path.join(tmp, "snap"), receipt=dict(raw=dict(requested_at="2026-11-16T15:35:01.123+09:00", received_at="2026-11-16T15:35:01.300+09:00")))
    assert os.path.exists(pv["path"]) and len(pv["sha256"]) == 64 and pv["raw_count"] == 1 and json.load(open(pv["path"], encoding="utf-8"))["receipt"]["raw"]["received_at"].endswith("01.300+09:00")
    ticks = iter([KD(2026, 11, 16, 15, 35, 1), KD(2026, 11, 16, 15, 35, 2)])
    ft = FakeToss()
    ft.sessions = ["2026-11-16"]
    trows, tc, tr = timed_fetch(ft, False, lambda: next(ticks))
    assert list(trows) == ["2026-11-16"] and tr["requested_at"].startswith("2026-11-16T15:35:01") and tr["received_at"].startswith("2026-11-16T15:35:02")
    assert receipt_complete(dict(raw=dict(requested_at="2026-11-16T15:31:00.000+09:00"), adj=dict(requested_at="2026-11-16T15:31:00.400+09:00")), "2026-11-16")
    assert not receipt_complete(dict(raw=dict(requested_at="2026-11-16T15:30:59.900+09:00"), adj=dict(requested_at="2026-11-16T15:31:01.000+09:00")), "2026-11-16")
    prev = [dict(kind="decision", day="2026-11-13", value=100000.0, unit_index=1.0, unit_peak=1.0)]
    f1 = account_flows(prev, "2026-11-16", dict(cash=96000.0, qty=0), 80000.0)
    assert f1["need_deposit"] and f1["value"] == 106000.0 and f1["value_ex_flows"] == 96000.0
    retry_rows = prev + [dict(kind="paper_deposit", day="2026-11-16", amount=DEPOSIT)]
    f2 = account_flows(retry_rows, "2026-11-16", dict(cash=106000.0, qty=0), 80000.0)
    assert not f2["need_deposit"] and f2["value"] == 106000.0 and f2["value_ex_flows"] == 96000.0
    assert apply_limits(prev, f1["value_ex_flows"])["no_buy"] and apply_limits(retry_rows, f2["value_ex_flows"])["no_buy"] and abs(apply_limits(retry_rows, f2["value_ex_flows"])["period_return"] + 0.04) < 1e-12
    gap_rows = prev + [dict(kind="paper_deposit", day="2026-11-14", amount=DEPOSIT)]
    f3 = account_flows(gap_rows, "2026-11-16", dict(cash=106000.0, qty=0), 80000.0)
    assert f3["need_deposit"] and f3["flows_since_prev_decision"] == 2 * DEPOSIT and f3["value"] == 116000.0 and f3["value_ex_flows"] == 96000.0
    held_rows = prev + [dict(kind="paper_deposit", day="2026-11-16", amount=DEPOSIT)]
    f4 = account_flows(held_rows, "2026-11-16", dict(cash=26000.0, qty=1), 80000.0)
    assert not f4["need_deposit"] and f4["value"] == 106000.0 and f4["value_ex_flows"] == 96000.0
    good = session_evidence({"timestamp": "2026-11-16T09:00:35.000+09:00"}, KD(2026, 11, 16, 9, 0, 40), "2026-11-16")
    assert good[0] and good[1]["age_seconds"] == 5.0
    assert not session_evidence({"timestamp": "2026-11-16T09:00:35.000+09:00"}, KD(2026, 11, 17, 9, 0, 40), "2026-11-17")[0]
    assert not session_evidence({}, KD(2026, 11, 16, 9, 0, 40), "2026-11-16")[0]
    rows_l = [dict(kind="decision", value=100000.0, unit_index=1.0, unit_peak=1.0)]
    L = apply_limits(rows_l, 96000.0)
    assert L["limits"] == ["daily_loss"] and L["no_buy"] and not L["force_flat"] and abs(L["period_return"] + 0.04) < 1e-12 and abs(L["unit_index"] - 0.96) < 1e-12 and L["prev_value"] == 100000.0
    assert apply_limits(rows_l, 97000.0)["limits"] == [] and not apply_limits(rows_l, 97000.0)["no_buy"]
    rows_l2 = rows_l + [dict(kind="decision", value=98000.0, unit_index=0.98, unit_peak=1.0)]
    L2 = apply_limits(rows_l2, 84000.0)
    assert L2["limits"] == ["daily_loss", "drawdown"] and L2["force_flat"] and abs(L2["unit_index"] - 0.84) < 1e-9 and L2["unit_peak"] == 1.0
    L0 = apply_limits([], 50000.0)
    assert L0["limits"] == [] and L0["unit_index"] == 1.0 and L0["prev_value"] is None
    assert not order_anomaly({"status": "FILLED"}) and order_anomaly({"status": "CANCELED"}) and order_anomaly({"status": "REJECTED"}) and order_anomaly({})
    snap = snapshot_series(dict(days=["2026-11-05"], O=[1.0], C=[1.5]), dict(days=["2026-11-05"], O=[1.0], C=[1.5]), "2026-11-05", out_dir=os.path.join(tmp, "snap"))
    assert set(snap) == {"raw", "adj"} and all(os.path.exists(v["path"]) and len(v["sha256"]) == 64 and v["last_day"] == "2026-11-05" for v in snap.values())
    assert paused([dict(kind="pause")]) and not paused([dict(kind="pause"), dict(kind="resume")]) and not paused([])
    assert next_event(datetime(2026, 11, 6, 23, 30, tzinfo=KST)) == (datetime(2026, 11, 9, 9, 0, 30, tzinfo=KST), "execute")
    assert next_event(datetime(2026, 11, 9, 9, 5, tzinfo=KST)) == (datetime(2026, 11, 9, 15, 35, 0, tzinfo=KST), "decide")
    assert next_event(datetime(2026, 11, 9, 16, 0, tzinfo=KST)) == (datetime(2026, 11, 10, 9, 0, 30, tzinfo=KST), "execute")
    assert all(r.get("test_fixture") for r in load_ledger())
    assert LEDGER != real_ledger and (os.path.getsize(real_ledger) if os.path.exists(real_ledger) else None) == real_before
    LEDGER = real_ledger
    print("selftest ok (TEST_FIXTURE only: FakeToss, no API call, no order, no job, nothing posted)")
    print(json.dumps(dict(parity_2026_11_04=dict(stored=last_dec["predicted_contrast"], recomputed=sig["pred"], w=last_dec["w"]), signal_at_2026_11_05_close=dict(held0=sig0, held1=sig1), signal_at_2026_11_06_close=dict(held0=sigF0, held1=sigF1, last_close=frozen_raw["C"][-1], last_day=frozen_raw["days"][-1])), ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    cmd = sys.argv[1:] or ["status"]
    if cmd == ["selftest"]:
        selftest()
    elif cmd == ["status"]:
        print(json.dumps(status(), ensure_ascii=False, indent=1))
    elif cmd == ["loop"]:
        loop()
    elif cmd in (["pause"], ["resume"]):
        print(json.dumps(append(dict(kind=cmd[0], by="owner_cli")), ensure_ascii=False))
    elif cmd[:1] == ["once"] and cmd[1:] in (["decide"], ["execute"]):
        from toss import Toss
        t = Toss()
        t.account = t.accounts()[0]["accountSeq"]
        print(json.dumps(decide(t) if cmd[1] == "decide" else execute(t), ensure_ascii=False, indent=1))
    else:
        print("usage: dobby_kr.py selftest | status | loop | pause | resume | once decide | once execute")
        sys.exit(2)
