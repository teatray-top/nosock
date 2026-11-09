import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timedelta

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import f1_fill_schedule as F

SYM = "069500"
FEE = 0.00015
SLIP = 5
CAPITAL = 100_000_000
START = "2026-11-09"
CAL = os.path.join(HERE, "session_calendar_kr_etf_f1_v1.json")
CAL_SHA = "aa12305513f739387e82fb31a586f034a1f798a4c81a47ed529dadec1a0d62af"
HIST_ROOT = os.path.join(ROOT, "data", "kr_etf", "forward_daily_adj_history")
MIN_ROOT = os.path.join(ROOT, "data", "kr_etf", "forward_1m")
OUT_DIR = os.path.join(HERE, "results", "forward_shadow_v1")
FIX_DIR = os.path.join(HERE, "results", "forward_shadow_fixtures_v1")
ORIGIN = "2002-10-14"
INT_TXT = re.compile(r"^\d+$")
TS_MIN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:00$")


class Refused(Exception):
    pass


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def ema(xs, n):
    a = 2.0 / (n + 1)
    out, e = [], None
    for x in xs:
        e = x if e is None else a * x + (1 - a) * e
        out.append(e)
    return out


def ribbon(closes):
    e = {n: ema(closes, n)[-1] for n in (8, 21, 50, 200)}
    return 1 if e[8] > e[21] > e[50] > e[200] else 0


def donchian(closes):
    held = 0
    for i in range(20, len(closes)):
        if held == 0 and closes[i] > max(closes[i - 20:i]):
            held = 1
        elif held == 1 and closes[i] < min(closes[i - 10:i]):
            held = 0
    return held


def hold(closes):
    return 1


RULES = {"ribbon": ribbon, "donchian20_exit10": donchian, "hold_reference": hold}


def session_times(cal, d):
    b = F.bounds(d, cal)
    return b["start"], b["cont_end"]


def eligible(rows, cal, d, activation):
    start, cont_end = session_times(cal, d)
    for k, r in enumerate(sorted(rows, key=lambda r: r["label"])):
        s = r["label"] - F.MIN
        if s >= cont_end:
            return None
        if s < activation:
            continue
        if k == 0 and activation <= start:
            continue
        if r["volume"] > 0:
            return s, float(r["open"])
    return None


def ledger(rule, signal, sessions, histories, minutes, cal, as_of, capital=CAPITAL):
    cash, shares, order = float(capital), 0, None
    decisions, orders, fills, equity = [], [], [], []
    status = dict(state="complete_through_input", at=None)
    prev = {}
    allcal = sorted(cal)
    for d in sessions:
        prev[d] = allcal[allcal.index(d) - 1]
    for d in sessions:
        T, cont_end = session_times(cal, d)
        if T > as_of:
            status = dict(state="waiting_decision_time", at=d)
            break
        p = prev[d]
        avail = [h for h in histories if h["available_at"] <= T]
        h = max(avail, key=lambda h: (h["available_at"], h["vintage"])) if avail else None
        if h is None or p not in h["days"]:
            dec = dict(session=d, decision_time=F.iso(T), prev_session=p, status="late_input", history_vintage=h["vintage"] if h else None, history_available_at=F.iso(h["available_at"]) if h else None, history_last_day=h["days"][-1] if h else None)
            target = None
        else:
            i = h["days"].index(p)
            closes = h["closes"][:i + 1]
            target = signal(closes)
            dec = dict(session=d, decision_time=F.iso(T), prev_session=p, status="decided", target=target, history_vintage=h["vintage"], history_available_at=F.iso(h["available_at"]), prev_close=closes[-1])
        decisions.append(dec)
        if order is None and target is not None:
            if target == 1 and shares == 0:
                n_req = int(cash // ((dec["prev_close"] + SLIP) * (1 + FEE)))
                if n_req > 0:
                    order = dict(order_id=f"{rule}:{d}:buy", side="buy", requested=n_req, activation=T, session=d, status="active")
                    orders.append(order)
            elif target == 0 and shares > 0:
                order = dict(order_id=f"{rule}:{d}:sell", side="sell", requested=shares, activation=T, session=d, status="active")
                orders.append(order)
        elif order is not None and target is not None:
            dec["note"] = "an earlier order is still active; no new order"
        m = minutes.get(d)
        if m is None or m["available_at"] > as_of:
            status = dict(state="waiting_minutes", at=d, minute_available_at=F.iso(m["available_at"]) if m else None)
            break
        if m.get("problems"):
            status = dict(state="held_input", at=d, problems=m["problems"])
            break
        rows = m["rows"]
        if order is not None:
            hit = eligible(rows, cal, d, order["activation"])
            if order["side"] == "buy":
                if hit is None:
                    order.update(status="expired_session_end", expired=order["requested"], expired_at=F.iso(cont_end))
                    order = None
                else:
                    s, o = hit
                    px = o + SLIP
                    afford = int(cash // (px * (1 + FEE)))
                    n = min(order["requested"], afford)
                    if n > 0:
                        cash -= n * px * (1 + FEE)
                        shares += n
                        fills.append(dict(order_id=order["order_id"], side="buy", shares=n, model_price=o, price=px, fee=n * px * FEE, model_fill_time=F.iso(s), model_notification_time=F.iso(s + F.MIN), minute_vintage=m["vintage"], minute_received_at=m["received_at"], minute_available_at=F.iso(m["available_at"])))
                    order.update(status="filled" if n == order["requested"] else ("partially_filled_rest_cancelled" if n > 0 else "cancelled_cash"), filled=n, cancelled=order["requested"] - n)
                    order = None
            else:
                if hit is not None:
                    s, o = hit
                    px = o - SLIP
                    n = min(order["requested"], shares)
                    cash += n * px * (1 - FEE)
                    shares -= n
                    fills.append(dict(order_id=order["order_id"], side="sell", shares=n, model_price=o, price=px, fee=n * px * FEE, model_fill_time=F.iso(s), model_notification_time=F.iso(s + F.MIN), minute_vintage=m["vintage"], minute_received_at=m["received_at"], minute_available_at=F.iso(m["available_at"])))
                    order.update(status="filled", filled=n, cancelled=order["requested"] - n)
                    order = None
                else:
                    order["carried_from"] = order.get("carried_from") or d
        close = float(sorted(rows, key=lambda r: r["label"])[-1]["close"])
        equity.append(dict(session=d, close=close, cash=cash, shares=shares, equity=cash + shares * close))
    final = None
    if equity:
        last = equity[-1]
        final = last["cash"] + last["shares"] * last["close"] * (1 - FEE)
        peak, dd = float(capital), 0.0
        for e in equity:
            peak = max(peak, e["equity"])
            dd = min(dd, e["equity"] / peak - 1)
        final = dict(final_after_sell_fee=final, return_pct=100 * (final / capital - 1), max_drawdown_pct=100 * dd, sessions=len(equity), through=last["session"])
    return dict(rule=rule, decisions=decisions, orders=orders, fills=fills, equity=equity, status=status, summary=final, cash=cash, shares=shares, open_order=order)


def seal_records(root):
    recs = {}
    p = os.path.join(root, "seals.jsonl")
    if not os.path.exists(p):
        return recs
    for line in open(p, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["file"] in recs:
            raise Refused(f"two seals for {r['file']}")
        recs[r["file"]] = r
    return recs


def checked(root, rel, recs):
    if rel not in recs:
        raise Refused(f"no seal for {rel}")
    data = open(os.path.join(root, rel), "rb").read()
    if sha_bytes(data) != recs[rel]["sha256"] or len(data) != recs[rel]["bytes"]:
        raise Refused(f"sealed hash differs for {rel}")
    return data


def at_iso(s):
    t = datetime.fromisoformat(s)
    if t.tzinfo is None:
        raise Refused(f"naive time {s}")
    return t


def load_histories(root=HIST_ROOT, sym=SYM):
    recs = seal_records(root)
    out, by_sha, skipped = [], {}, []
    for v in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        if v.startswith("_") or not os.path.isdir(os.path.join(root, v)):
            continue
        if f"{v}/.commit" not in recs:
            skipped.append(v)
            continue
        meta_b = checked(root, f"{v}/meta.json", recs)
        if recs[f"{v}/.commit"]["sha256"] != sha_bytes(meta_b):
            raise Refused(f"commit record of {v} does not match its meta")
        if not os.path.exists(os.path.join(root, v, f"{sym}.json")):
            continue
        data = checked(root, f"{v}/{sym}.json", recs)
        meta = json.loads(meta_b.decode("utf-8"))
        if meta["files"].get(sym) != sha_bytes(data):
            raise Refused(f"{v}/{sym}.json is not the file its meta lists")
        h = json.loads(data.decode("utf-8"))
        closes = [float(c) for c in h["close"]]
        if len(closes) != len(h["days"]) or not all(math.isfinite(c) and c > 0 for c in closes) or h["days"] != sorted(set(h["days"])) or h["days"][0] != ORIGIN:
            raise Refused(f"{v}/{sym}.json has invalid days or closes")
        e = dict(vintage=v, available_at=at_iso(recs[f"{v}/.commit"]["sealed_at"]), days=h["days"], closes=closes, sha256=sha_bytes(data), kind=h["kind"], parent=h.get("parent"), cutoff_day=h["cutoff_day"])
        by_sha[e["sha256"]] = e
        out.append(e)
    for e in out:
        if e["kind"] != "base":
            par = by_sha.get((e["parent"] or {}).get("sha256"))
            if par is None or not par["cutoff_day"] < e["cutoff_day"] or e["days"][:len(par["days"])] != par["days"]:
                raise Refused(f"broken history chain at {e['vintage']}")
    return sorted(out, key=lambda e: (e["available_at"], e["vintage"])), skipped


def load_minutes(days, root=MIN_ROOT, sym=SYM, cal=None):
    recs = seal_records(root)
    out, cut = {}, []
    for d in days:
        cands = []
        for v in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            rel = f"{v}/{d}_{sym}.json"
            if not os.path.exists(os.path.join(root, rel)):
                continue
            if f"{v}/meta.json" not in recs:
                cut.append(rel)
                continue
            meta = json.loads(checked(root, f"{v}/meta.json", recs).decode("utf-8"))
            rec = json.loads(checked(root, rel, recs).decode("utf-8"))
            times = [r.get("received_at") for r in meta["requests"] if r.get("received_at")]
            cands.append(dict(vintage=v, available_at=at_iso(recs[f"{v}/meta.json"]["sealed_at"]), received_at=max(times) if times else None, held=rec["summary"]["held"] or rec["day"] != d or rec["symbol"] != sym, raw=rec["rows"], sha256=recs[rel]["sha256"]))
        cands = [c for c in cands if not c["held"]]
        if not cands:
            continue
        c = min(cands, key=lambda c: (c["available_at"], c["vintage"]))
        rows, problems = [], []
        for k, r in enumerate(c["raw"]):
            if not (isinstance(r, list) and len(r) == 6 and isinstance(r[0], str) and TS_MIN.match(r[0]) and INT_TXT.match(str(r[5]))):
                problems.append(f"row {k} malformed")
                continue
            vals = []
            for x in r[1:5]:
                try:
                    vals.append(float(x))
                except (TypeError, ValueError):
                    vals.append(float("nan"))
            rows.append(dict(label=datetime.fromisoformat(r[0]).replace(tzinfo=F.KST), open=vals[0], high=vals[1], low=vals[2], close=vals[3], volume=int(r[5])))
        if cal is not None and not problems:
            pc = F.precheck([d], {d: rows}, cal)
            problems += pc.get(d, [])
        out[d] = dict(vintage=c["vintage"], available_at=c["available_at"], received_at=c["received_at"], rows=rows, problems=problems, sha256=c["sha256"])
    return out, cut


def load_cal():
    data = open(CAL, "rb").read()
    if sha_bytes(data) != CAL_SHA:
        raise Refused("calendar hash differs")
    return json.loads(data.decode("utf-8"))["days"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "inputs"])
    a = ap.parse_args()
    if a.cmd == "inputs":
        cal = load_cal()
        hist, skipped = load_histories()
        sessions = [d for d in sorted(cal) if d >= START]
        mins, cut = load_minutes([d for d in sorted(cal) if "2026-11-06" <= d <= "2026-11-13"], cal=cal)
        print("histories", [(h["vintage"], F.iso(h["available_at"]), h["days"][-1]) for h in hist], "uncommitted", skipped)
        print("minutes", {d: (m["vintage"], F.iso(m["available_at"]), len(m["rows"]), m["problems"]) for d, m in mins.items()}, "cut", cut)
        print("first decision", sessions[0], F.iso(session_times(cal, sessions[0])[0]))
    else:
        import forward_shadow_fixtures as FX
        FX.main()


if __name__ == "__main__":
    main()
