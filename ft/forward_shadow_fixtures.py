import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timedelta

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import f1_fill_schedule as F
import forward_shadow as S

OUT = os.path.join(S.FIX_DIR, "result.json")
D0, D1, D2, D3 = "2030-01-04", "2030-01-07", "2030-01-08", "2030-01-09"
CAP = 1_000_000


def entry(first, last):
    hm = lambda s, d: (datetime.strptime(s, "%H:%M") + timedelta(minutes=d)).strftime("%H:%M")
    return dict(cls="synthetic", first=first, last=last, session_start=hm(first, -1), continuous_end=hm(last, -10))


CAL = {d: entry("09:01", "15:30") for d in (D0, D1, D2, D3)}
SESS = [D1, D2, D3]


def t(day, hhmm):
    return F.at(day, hhmm)


def rows(day, opens=None, zero=(), base=100.0):
    out, x = [], t(day, "09:01")
    k = 0
    while x <= t(day, "15:30"):
        hh = x.strftime("%H:%M")
        o = (opens or {}).get(hh, base)
        out.append(dict(label=x, open=o, high=o + 1, low=o - 1, close=o, volume=0 if hh in zero else 1000))
        x += F.MIN
        k += 1
    return out


def mins(day, avail, **kw):
    return dict(vintage=f"m_{day}", available_at=avail, received_at=F.iso(avail), rows=rows(day, **kw), problems=[], sha256="x")


def hist(name, avail, days, closes):
    return dict(vintage=name, available_at=avail, days=days, closes=closes, sha256=name, kind="x", parent=None, cutoff_day=days[-1])


UP = lambda closes: 1 if closes[-1] >= 100 else 0


def accounting_ok(r, capital):
    cash = capital - sum(f["shares"] * f["price"] * (1 + S.FEE) for f in r["fills"] if f["side"] == "buy") + sum(f["shares"] * f["price"] * (1 - S.FEE) for f in r["fills"] if f["side"] == "sell")
    sh = sum(f["shares"] for f in r["fills"] if f["side"] == "buy") - sum(f["shares"] for f in r["fills"] if f["side"] == "sell")
    nonneg = all(e["cash"] >= -1e-9 for e in r["equity"])
    return abs(cash - r["cash"]) < 1e-6 and sh == r["shares"] and nonneg


def cases():
    out = []

    def add(cid, desc, ok, detail):
        out.append(dict(id=cid, case=desc, passed=bool(ok), detail=detail))

    base_days = [D0, D1, D2]
    h_all = hist("h1", t(D0, "16:00"), [D0, D1, D2], [101, 101, 101])
    full = {d: mins(d, t(d, "16:10")) for d in SESS}
    late = hist("late", t(D1, "09:30"), [D0], [101])
    later = hist("next", t(D1, "16:20"), [D0, D1], [101, 101])
    r = S.ledger("fx", UP, SESS, [late, later], full, CAL, t(D3, "16:30"), CAP)
    d1, d2 = r["decisions"][0], r["decisions"][1]
    add("FX1", "history for the previous close arrives after the decision time", d1["status"] == "late_input" and d1["history_vintage"] is None and not any(o["session"] == D1 for o in r["orders"]) and d2["status"] == "decided" and d2["history_vintage"] == "next", dict(d1=d1, d2=d2))

    v1 = hist("v1", t(D0, "16:00"), [D0], [101])
    v2 = hist("v2", t(D1, "12:00"), [D0, D1], [99, 99])
    r_early = S.ledger("fx", UP, SESS, [v1, v2], full, CAL, t(D1, "16:30"), CAP)
    r_late = S.ledger("fx", UP, SESS, [v1, v2], full, CAL, t(D3, "16:30"), CAP)
    a, b = r_early["decisions"][0], r_late["decisions"][0]
    add("FX2", "a later revised vintage is not used for an earlier decision and does not change it on a later run", a["history_vintage"] == "v1" and a["target"] == 1 and a == b and r_late["decisions"][1]["history_vintage"] == "v2" and r_late["decisions"][1]["target"] == 0, dict(early=a, late=b, d2=r_late["decisions"][1]))

    gap = {D1: mins(D1, t(D1, "16:10"), opens={"09:02": 150.0}), D2: full[D2], D3: full[D3]}
    r = S.ledger("fx", UP, SESS, [hist("h", t(D0, "16:00"), [D0, D1, D2], [100, 100, 100])], gap, CAL, t(D3, "16:30"), CAP)
    o = r["orders"][0]
    n_req = int(CAP // ((100 + S.SLIP) * (1 + S.FEE)))
    afford = int(CAP // ((150 + S.SLIP) * (1 + S.FEE)))
    add("FX3", "gap up at the fill: requested from the previous close, reduced to what cash covers at the fill price, rest cancelled", o["requested"] == n_req and o["filled"] == afford and o["cancelled"] == n_req - afford and o["status"] == "partially_filled_rest_cancelled" and r["fills"][0]["price"] == 155.0 and accounting_ok(r, CAP), dict(order=o, fill=r["fills"][0]))

    flip = hist("h", t(D0, "16:00"), [D0, D1, D2], [101, 99, 99])
    r = S.ledger("fx", UP, SESS, [flip], full, CAL, t(D3, "16:30"), CAP)
    sells = [f for f in r["fills"] if f["side"] == "sell"]
    add("FX4", "sell at the model open minus 5 won, never more than the shares held", len(sells) == 1 and sells[0]["price"] == 95.0 and sells[0]["shares"] == r["fills"][0]["shares"] and r["shares"] == 0 and accounting_ok(r, CAP), dict(fills=r["fills"]))

    r_missing = S.ledger("fx", UP, SESS, [h_all], {D1: full[D1]}, CAL, t(D3, "16:30"), CAP)
    r_future = S.ledger("fx", UP, SESS, [h_all], {D1: full[D1], D2: mins(D2, t(D3, "08:30"))}, CAL, t(D2, "16:30"), CAP)
    bad = dict(full[D2], problems=["missing 1 from 11:00"])
    r_held = S.ledger("fx", UP, SESS, [h_all], {D1: full[D1], D2: bad, D3: full[D3]}, CAL, t(D3, "16:30"), CAP)
    add("FX5", "minutes missing, not yet available at as_of, or damaged: the ledger stops at that session", r_missing["status"]["state"] == "waiting_minutes" and r_missing["status"]["at"] == D2 and r_future["status"]["state"] == "waiting_minutes" and r_future["status"]["at"] == D2 and r_held["status"]["state"] == "held_input" and r_held["status"]["at"] == D2 and len(r_held["equity"]) == 1, dict(missing=r_missing["status"], future=r_future["status"], held=r_held["status"]))

    r = S.ledger("fx", UP, SESS, [h_all], full, CAL, t(D1, "08:59"), CAP)
    add("FX6", "a decision time after as_of is not recorded", r["decisions"] == [] and r["status"]["state"] == "waiting_decision_time" and r["status"]["at"] == D1, dict(status=r["status"]))

    zz = {D1: mins(D1, t(D1, "16:10"), zero=("09:02", "09:03", "09:04"), opens={"09:01": 90.0, "09:05": 102.0}), D2: full[D2], D3: full[D3]}
    r = S.ledger("fx", UP, SESS, [h_all], zz, CAL, t(D3, "16:30"), CAP)
    f = r["fills"][0]
    add("FX7", "an order active at the session start skips the opening interval and waits through zero-volume minutes", f["model_fill_time"] == F.iso(t(D1, "09:04")) and f["model_price"] == 102.0, dict(fill=f))

    dead = [x.strftime("%H:%M") for x in (t(D1, "09:01") + F.MIN * k for k in range(0, 380))]
    zx = {D1: mins(D1, t(D1, "16:10"), zero=tuple(dead)), D2: full[D2], D3: full[D3]}
    r = S.ledger("fx", UP, SESS, [h_all], zx, CAL, t(D3, "16:30"), CAP)
    add("FX8", "a buy with no positive-volume minute before continuous_end expires; the next session decides again", r["orders"][0]["status"] == "expired_session_end" and r["orders"][0]["expired_at"] == F.iso(t(D1, "15:20")) and r["orders"][1]["session"] == D2 and r["orders"][1]["status"] == "filled", dict(orders=r["orders"]))

    import compare_trend_v2 as CT
    w, days = CT.weights("069500")
    adj = json.load(open(CT.ADJ, encoding="utf-8"))["069500"]
    closes = [float(c) for c in adj["C"][:len(days)]]
    lo = days.index("2025-06-02")
    mism = [d for i, d in enumerate(days) if i >= lo and (S.ribbon(closes[:i + 1]) != w["ribbon"][d] or S.donchian(closes[:i + 1]) != w["donchian20_exit10"][d])]
    add("FX9", "ribbon and donchian signals equal compare_trend_v2 on every 069500 day from 2025-06-02 to 2026-11-05", not mism and len(days) - lo > 300, dict(days=len(days) - lo, mismatches=mism[:5]))

    cal = S.load_cal()
    hist_real, skipped = S.load_histories()
    sess = [d for d in sorted(cal) if d >= S.START][:5]
    mreal, cut = S.load_minutes(sess, cal=cal)
    now = datetime.now(F.KST)
    r_now = S.ledger("ribbon", S.ribbon, sess, hist_real, mreal, cal, now)
    add("FX10", "real stored inputs at the current time: no decision is recorded before 2026-11-09 09:00", r_now["decisions"] == [] and r_now["status"] == dict(state="waiting_decision_time", at=S.START), dict(as_of=F.iso(now), histories=[(h["vintage"], F.iso(h["available_at"])) for h in hist_real], status=r_now["status"]))
    return out


def main():
    res = cases()
    for c in res:
        print("PASS" if c["passed"] else "FAIL", c["id"], c["case"], "" if c["passed"] else json.dumps(c["detail"], ensure_ascii=False, default=str)[:800])
    print("all passed", all(c["passed"] for c in res), "cases", len(res))
    if "--write" in sys.argv:
        sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
        doc = dict(version="forward-shadow-fixtures-v1", code={os.path.basename(p): sha(p) for p in (S.__file__, os.path.abspath(__file__), F.__file__)}, cases=res, all_passed=all(c["passed"] for c in res), imports=sorted(n for n, m in sys.modules.items() if getattr(m, "__file__", None) and os.path.abspath(m.__file__).startswith(S.ROOT)), forbidden_loaded=sorted(n for n in sys.modules if n in ("toss", "cred", "forward_history_kr_etf", "forward_collect_kr_etf", "backfill_kr_etf_1m", "dobby_kr", "forward_runner_kr_etf")), run_meta=dict(generated_at=datetime.now(F.KST).isoformat(timespec="seconds"), python=platform.python_version()))
        os.makedirs(S.FIX_DIR, exist_ok=True)
        data = json.dumps(doc, ensure_ascii=False, indent=1, default=str).encode("utf-8")
        with open(OUT, "xb") as f:
            f.write(data)
        print(OUT, hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    main()
