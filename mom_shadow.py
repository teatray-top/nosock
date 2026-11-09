import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import bisect
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ft"))
import numpy as np
import msma_v1 as M1
import ms_role_v2 as R
import ms_alloc_v7 as A7
import ms_alloc_v9 as A9
import sel_diag_v1 as D1
import sel_diag_v2 as D2
import port_c_v1 as C1
import port_c_v2 as C2

KST = ZoneInfo("Asia/Seoul")
ANCHOR = M1.END
US_FEE, US_FX = 0.001, 0.0005
BOOKS = {"U29_topk": ("U29", "topk"), "U29_dual": ("U29", "dual")}
K_2026 = {"U29_topk": 2, "U29_dual": 2}
KR_DOMESTIC = ["069500", "229200", "091160", "091170", "114800", "148070", "114260", "153130"]
LEDGER = os.path.join(HERE, "runs", "mom_shadow_ledger.jsonl")
SNAP_DIR = os.path.join(HERE, "data", "kr_etf", "mom_shadow_snaps")
FX_LOG = os.path.join(HERE, "runs", "mom_shadow_fx.jsonl")
RUN_AT = (16, 5)
FX_MIN_WON = 10_000.0
COMPLETE_AFTER = "15:31:00"
IRC = os.path.join(HERE, "irc.py")


def sha_b(b):
    return hashlib.sha256(b).hexdigest()


def now_kst():
    return datetime.now(KST)


def append(row, path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")


def read_rows(path, kind=None):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if kind is None or r.get("kind") == kind:
            out.append(r)
    return out


def make_ins(rec, code, market, costf_maker, fx=None, end=None):
    keep = [i for i, d in enumerate(rec["days"]) if (end is None or d <= end) and (fx is None or d in fx)]
    days = [rec["days"][i] for i in keep]
    n = len(days)
    m = np.array([fx[d] for d in days]) if fx is not None else np.ones(n)
    O, H, L, C = (np.asarray([rec[k][i] for i in keep], float) * m for k in ("O", "H", "L", "C"))
    r = np.r_[np.nan, np.diff(np.log(C))]
    v20 = np.array([np.std(r[max(1, i - 19):i + 1]) if i >= 20 else np.nan for i in range(n)])
    ins = dict(code=code, market=market, days=days, O=O, H=H, L=L, C=C, v20=v20, fxv=m)
    ins["cost"] = costf_maker(ins)
    return ins


def pinned():
    man_us, us = A7.load_us()
    man_kus, kus = A9.load_krus()
    recs = R.load()
    kr = {c: recs[c] for c in M1.CLASS}
    kus_adj = {c: kus[f"{c}_adj"] for c in man_kus["classes"]}
    us_adj = {c: us[f"{c}_adj"] for c in man_us["classes"]}
    fxrec = us["KRW_X_adj"]
    fx = {d: c for d, c in zip(fxrec["days"], fxrec["C"]) if c > 500}
    return kr, kus_adj, us_adj, fx


def cut(rec, end):
    n = sum(1 for d in rec["days"] if d <= end)
    return dict(days=list(rec["days"][:n]), **{k: [float(x) for x in rec[k][:n]] for k in ("O", "H", "L", "C")})


def splice(rec, rows):
    if ANCHOR not in rows:
        raise RuntimeError("anchor day missing from the fetch")
    s = rec["C"][-1] / rows[ANCHOR][3]
    out = {k: list(v) for k, v in rec.items()}
    for d in sorted(x for x in rows if x > ANCHOR):
        for k, v in zip("OHLC", rows[d]):
            out[k].append(v * s)
        out["days"].append(d)
    return out


def fx_ff(us_days, fx_hist):
    fdays = sorted(fx_hist)
    out, last, j = {}, None, 0
    for d in us_days:
        while j < len(fdays) and fdays[j] <= d:
            last = fx_hist[fdays[j]]
            j += 1
        if last is not None:
            out[d] = last
    return out


def build(kr, kus, us, fx_hist, end=None):
    us_days = sorted(set(d for v in us.values() for d in v["days"]))
    fxm = fx_ff(us_days, fx_hist)
    inst = {}
    for c, v in kr.items():
        inst[c] = make_ins(v, c, "KR", A7.kr_cost, end=end)
    for c, v in kus.items():
        inst[c] = make_ins(v, c, "KR", A7.kr_cost, end=end)
    for c, v in us.items():
        inst[c] = make_ins(v, c, "US", A7.us_cost_maker(US_FEE, US_FX), fxm, end=end)
    groups = dict(U28=list(kr) + list(kus), U36=list(kr) + list(us), U29=KR_DOMESTIC + list(us))
    return inst, groups


def world_for(inst, codes):
    simple = {c: D1.simple_scores(inst[c]) for c in codes}
    dates = D2.by_date(inst, codes)
    scores = dict(mom=D2.composite(inst, codes, simple, dates))
    absmom = {c: simple[c]["mom252_21"] for c in codes}
    w = C2.World2(inst, codes, ("ewma", 0.94, 0.0), scores, absmom)
    w.init_args = (inst, codes, ("ewma", 0.94, 0.0), scores, absmom)
    return w


def k_for_year(world, mode, y, cache):
    if y in cache:
        return cache[y]
    best, best_v = None, -1e18
    for k in C2.KS:
        r = world.run(f"{y - C1.LOOKBACK_YEARS}-01-01", f"{y - 1}-12-31", lambda D, k=k: k, mode)
        if r["total_won"] > best_v + 1e-6:
            best, best_v = k, r["total_won"]
    cache[y] = best
    return best


def bar_complete(day, fetched, market):
    if market == "US":
        return day < fetched[:10]
    return day < fetched[:10] or (day == fetched[:10] and fetched[11:19] >= COMPLETE_AFTER)


def fetch_all(client_factory, kr_codes, us_codes):
    t = client_factory()
    snap = {}
    count = 200
    for c in kr_codes + us_codes:
        snap[c] = t.candles(c, "1d", count=count, adjusted=True)["candles"]
        time.sleep(0.2)
    fx = t.fx("USD", "KRW")
    return snap, dict(midRate=float(fx["midRate"]), rate=float(fx["rate"]), validFrom=fx.get("validFrom")), now_kst().isoformat(timespec="milliseconds")


def to_rows(candles, fetched, market):
    out = {}
    for x in candles:
        d = x["timestamp"][:10]
        if bar_complete(d, fetched, market):
            out[d] = (float(x["openPrice"]), float(x["highPrice"]), float(x["lowPrice"]), float(x["closePrice"]))
    return out


class ShadowWorld(C2.World2):
    def run_state(self, start, end, lam_of, mode):
        cash, vault, dep, hwm = 0.0, 0.0, 0.0, 0.0
        units = {}
        pending = []
        kdays = sorted(d for d in self.days if start <= d <= end and d in self.kr_days)
        reb = set(kdays[::C1.H])
        vault_year, pay_days, eq_reb = {}, [], None
        for D in sorted(d for d in self.days if start <= d <= end):
            keep = []
            for (c, val, d_exec) in pending:
                if d_exec != D:
                    keep.append((c, val, d_exec))
                    continue
                ins = self.inst[c]
                i = self.ipos[c][D]
                o = ins["O"][i]
                nu = val / o
                du = nu - units.get(c, 0.0)
                cost = abs(du) * o * ins["cost"](i, o)
                cash -= du * o + cost
                units[c] = nu
            pending = keep
            if D in self.kr_days:
                cash += C1.DEPOSIT
                dep += C1.DEPOSIT
            if D in reb:
                eq = cash + sum(u * self.close_at(c, D)[0] for c, u in units.items() if u)
                P = eq + vault - dep
                if P > hwm:
                    pay = C1.VAULT_SHARE * (P - hwm)
                    hwm = P
                    vault += pay
                    cash -= pay
                    eq -= pay
                    vault_year[D[:4]] = vault_year.get(D[:4], 0.0) + pay
                    pay_days.append(D)
                eq_reb = (D, eq)
                tw = self.targets(D, lam_of(D), mode)
                for c in set(units) | set(tw):
                    o, i, d_exec = self.next_open(c, D)
                    if o is None or d_exec > end:
                        continue
                    pending.append((c, tw.get(c, 0.0) * eq * (o / self.close_at(c, D)[0]), d_exec))
        eq = cash + sum(u * self.close_at(c, end)[0] for c, u in units.items() if u)
        hold = {c: dict(units=u, value_won=u * self.close_at(c, end)[0], market=self.inst[c]["market"]) for c, u in units.items() if u > 1e-12}
        return dict(deposits_won=dep, vault_won=vault, acct_won=eq, total_won=eq + vault, cash_won=cash, payouts=len(pay_days), vault_by_year_won=vault_year, holdings=hold, eq_at_last_rebalance=eq_reb, kdays=kdays)


def fx_request(book, today, tgt, hold, eq, fx_mid):
    need = sum(w * eq for c, w in tgt.items() if c.isalpha()) - sum(h["value_won"] for c, h in hold.items() if h["market"] == "US")
    if abs(need) < FX_MIN_WON:
        return None
    return dict(id=f"{today}-{book}", book=book, direction="KRW->USD" if need > 0 else "USD->KRW", amount_won=round(abs(need)), amount_usd=round(abs(need) / fx_mid, 2), fx_mid=fx_mid, reason="monthly rebalance of the US positions" if need > 0 else "US positions shrink (sells or vault payout)", convert_by="next KR business day, Toss business hours (95% preference)")


def names_map():
    names = json.load(open(os.path.join(R.ETF_DIR, "manifest.json"), encoding="utf-8"))["names"]
    man_us, _ = A7.load_us()
    return dict(names, **man_us["names"])


def replay(inst, groups, start, today, kcache, fx_mid=None):
    from nosock import engine as E
    import ms_alloc_v3 as A3
    out = {}
    names = names_map()
    raw = A3.load_raw()
    for book, (g, mode) in BOOKS.items():
        codes = groups[g]
        w = world_for(inst, codes)
        cache = kcache.setdefault(book, {2026: K_2026[book]})
        kfun = lambda D: k_for_year(w, mode, int(D[:4]), cache)
        m = E.Market(inst, codes, names)
        m.raw = raw
        strat = E.Momentum(m, w.scores["mom"], w.absmom, {c: D1.simple_scores(inst[c])["mom126"] for c in codes}, mode)
        r = E.run(m, strat, start, today, kfun, whole=True, tax=True, record=True)
        kd = sorted(d for d in m.days if start <= d <= today and d in m.kr_days)
        rebalance_today = bool(kd and kd[-1] == today and (len(kd) - 1) % C1.H == 0)
        dec = r["decisions"][-1] if (r["decisions"] and r["decisions"][-1]["day"] == today) else None
        tgt = dec["targets"] if dec else {}
        hold = {c: dict(units=h["units"], value_won=h["value"], market=h["market"], shares=h.get("shares")) for c, h in r["holdings"].items()}
        fxr = fx_request(book, today, tgt, hold, dec["equity"], fx_mid) if (dec and fx_mid and g in ("U29", "U36")) else None
        nxt = (C1.H - (len(kd) - 1) % C1.H) % C1.H if kd else None
        vy = {}
        for x in r["payouts"]:
            vy[x["day"][:4]] = vy.get(x["day"][:4], 0.0) + x["amount"]
        out[book] = dict(universe=g, mode=mode, k=kfun(today), total_won=r["total"], vault_won=r["vault"], acct_won=r["account"], cash_won=r["cash"], deposits_won=r["deposits"], taxes_won=r["taxes"], payouts=len(r["payouts"]), rebalance_today=rebalance_today, sessions_to_next_rebalance=nxt, sessions_since_start=len(kd), targets=tgt, holdings=hold, fx_request=fxr, decision=dec, fills_today=[f for f in r["fills"] if f["day"] == today], payouts_today=[x for x in r["payouts"] if x["day"] == today], vault_by_year_won=vy, engine="nosock.engine realistic (whole KR shares, KR 15.4% / US 22% tax)")
    return out


def run_once(today=None, ledger=LEDGER, fx_log=FX_LOG, client_factory=None, snap_dir=SNAP_DIR, quiet=False, start=None):
    today = today or now_kst().strftime("%Y-%m-%d")
    days_done = [r for r in read_rows(ledger, "mom_day")]
    if days_done and days_done[-1]["day"] >= today:
        return []
    start = start or (days_done[0]["start"] if days_done else today)
    kr, kus, us, fx_hist = pinned()
    kr = {c: cut(v, ANCHOR) for c, v in kr.items()}
    kus = {c: cut(v, ANCHOR) for c, v in kus.items()}
    us = {c: cut(v, ANCHOR) for c, v in us.items()}
    fx_hist = {d: v for d, v in fx_hist.items() if d <= ANCHOR}
    if client_factory is None:
        import alloc_shadow
        client_factory = alloc_shadow.read_only_client
    try:
        snap, fx_now, fetched = fetch_all(client_factory, list(kr) + list(kus), list(us))
    except Exception as e:
        row = dict(kind="mom_skip", day=today, reason="fetch failed: " + type(e).__name__)
        append(row, ledger)
        return [row]
    os.makedirs(snap_dir, exist_ok=True)
    sb = json.dumps(dict(fetched_at=fetched, fx=fx_now, candles=snap), ensure_ascii=False).encode("utf-8")
    sp = os.path.join(snap_dir, f"{today}_{fetched[11:19].replace(':', '')}.json")
    with open(sp, "xb") as f:
        f.write(sb)
    rows_kr = {c: to_rows(snap[c], fetched, "KR") for c in list(kr) + list(kus)}
    rows_us = {c: to_rows(snap[c], fetched, "US") for c in us}
    last_us = max((max(v) for v in rows_us.values() if v), default=None)
    if last_us and last_us > ANCHOR:
        append(dict(kind="fx", us_day=last_us, midRate=fx_now["midRate"], recorded_at=fetched), fx_log)
    for r in read_rows(fx_log, "fx"):
        fx_hist[r["us_day"]] = r["midRate"]
    kr = {c: splice(v, rows_kr[c]) for c, v in kr.items()}
    kus = {c: splice(v, rows_kr[c]) for c, v in kus.items()}
    us = {c: splice(v, rows_us[c]) for c, v in us.items()}
    if today not in kr["069500"]["days"]:
        row = dict(kind="mom_skip", day=today, reason="no completed KR bar for today")
        append(row, ledger)
        return [row]
    inst, groups = build(kr, kus, us, fx_hist)
    kcache = {}
    books = replay(inst, groups, start, today, kcache, fx_now["midRate"])
    row = dict(kind="mom_day", day=today, start=start, fetched_at=fetched, snapshot=os.path.basename(sp), snapshot_sha256=sha_b(sb), fx_mid=fx_now["midRate"], books=books)
    append(row, ledger)
    if not quiet:
        subprocess.run([sys.executable, "-X", "utf8", IRC, "say", message(row)], cwd=HERE, timeout=60)
    return [row]


def message(row):
    if row["kind"] != "mom_day":
        return f"[모멘텀 모의 {row['day']}] 건너뜀: {row['reason']}"
    parts = []
    for b, x in row["books"].items():
        t = ", ".join(f"{c} {100 * w:.0f}%" for c, w in sorted(x["targets"].items(), key=lambda z: -z[1])) if x["rebalance_today"] else ""
        fx = x.get("fx_request")
        fxs = f" | 환전 요청: {'원화→달러' if fx['direction'] == 'KRW->USD' else '달러→원화'} {fx['amount_won']:,}원 (약 ${fx['amount_usd']:,.2f}), 다음 영업일 토스 영업시간에" if fx else ""
        parts.append(f"{b}(K={x['k']}): 입금 {x['deposits_won']:,.0f} → 금고 {x['vault_won']:,.0f} + 계좌 {x['acct_won']:,.0f} = {x['total_won']:,.0f}" + (f" | 오늘 리밸런싱 목표: {t or '전부 현금'}" if x["rebalance_today"] else "") + fxs)
    return f"[모멘텀 모의 {row['day']}] " + " / ".join(parts)


def next_run(now):
    t = now.replace(hour=RUN_AT[0], minute=RUN_AT[1], second=0, microsecond=0)
    if now >= t:
        t += timedelta(days=1)
    while t.weekday() >= 5:
        t += timedelta(days=1)
    return t


def loop():
    attempts = {}
    while True:
        now = now_kst()
        today = now.strftime("%Y-%m-%d")
        done = read_rows(LEDGER, "mom_day")
        due = now.weekday() < 5 and (now.hour, now.minute) >= RUN_AT and not (done and done[-1]["day"] >= today) and attempts.get(today, 0) < 3
        if due:
            attempts[today] = attempts.get(today, 0) + 1
            try:
                run_once(today, quiet=attempts[today] > 1)
            except Exception as e:
                append(dict(kind="mom_error", day=today, error=type(e).__name__), LEDGER)
        t = next_run(now_kst())
        time.sleep(max(30.0, min(3600.0, (t - now_kst()).total_seconds())))


def selftest():
    import tempfile
    tmp = tempfile.mkdtemp(prefix="TEST_FIXTURE_mom_")
    checks = []
    kr, kus, us, fx = pinned()
    inst, groups = build({c: cut(v, ANCHOR) for c, v in kr.items()}, {c: cut(v, ANCHOR) for c, v in kus.items()}, {c: cut(v, ANCHOR) for c, v in us.items()}, {d: v for d, v in fx.items() if d <= ANCHOR}, end=ANCHOR)
    for book, (g, mode) in BOOKS.items():
        pv6 = json.load(open(os.path.join(HERE, "ft", "results", "port_c_v6.json"), encoding="utf-8"))["results"][mode]
        prior = dict(pv6["fractional"], k_by_year=pv6["k_by_year"])
        kby = {int(y): k for y, k in prior["k_by_year"].items()}
        w = world_for(inst, groups[g])
        r = w.run("2016-01-01", "2026-11-05", lambda D: kby[int(D[:4])], mode)
        checks.append((f"{book}: shadow builder + engine reproduce port_c_v6 total {prior['total_won'] / 1e4:.1f}만", abs(r["total_won"] - prior["total_won"]) < 1.0))
    full_kr, full_kus, full_us = kr, kus, us
    man_us, usd = A7.load_us()
    man_kus, kusd = A9.load_krus()

    class Fake:
        def candles(self, sym, interval, count=200, before=None, adjusted=False):
            src = full_kr.get(sym) or full_kus.get(sym) or full_us.get(sym)
            ds = list(range(len(src["days"])))[-count:]
            return {"candles": [{"timestamp": src["days"][i] + "T00:00:00+09:00", "openPrice": str(src["O"][i]), "highPrice": str(src["H"][i]), "lowPrice": str(src["L"][i]), "closePrice": str(src["C"][i])} for i in ds]}

        def fx(self, a, b):
            return {"midRate": "1400.0", "rate": "1400.5", "validFrom": "2026-11-06T16:00:00+09:00"}

    led, fxl = os.path.join(tmp, "ledger.jsonl"), os.path.join(tmp, "fx.jsonl")
    r1 = run_once("2026-11-06", led, fxl, lambda: Fake(), os.path.join(tmp, "snap"), quiet=True)
    ok1 = len(r1) == 1 and r1[0]["kind"] == "mom_day" and all(abs(b["deposits_won"] - 10_000.0) < 1e-9 for b in r1[0]["books"].values()) and all(b["rebalance_today"] for b in r1[0]["books"].values())
    checks.append(("first run: one KR session, one deposit per book, rebalance on day one", ok1))
    checks.append(("targets respect K and no leverage", all(len(b["targets"]) <= b["k"] and sum(b["targets"].values()) <= 1 + 1e-9 for b in r1[0]["books"].values())))
    r2 = run_once("2026-11-06", led, fxl, lambda: Fake(), os.path.join(tmp, "snap"), quiet=True)
    checks.append(("same day is idempotent", r2 == []))
    r3 = run_once("2026-11-09", led, fxl, lambda: Fake(), os.path.join(tmp, "snap"), quiet=True)
    checks.append(("no KR bar for today is a skip", len(r3) == 1 and r3[0]["kind"] == "mom_skip"))
    fxrows = read_rows(fxl, "fx")
    checks.append(("fx recorded for the newest US day after the anchor", len(fxrows) >= 1 and fxrows[0]["us_day"] > ANCHOR and fxrows[0]["midRate"] == 1400.0))
    checks.append(("message renders", message(r1[0]).startswith("[모멘텀 모의 2026-11-06]")))
    g, mode = BOOKS["U29_topk"]
    pv6 = json.load(open(os.path.join(HERE, "ft", "results", "port_c_v6.json"), encoding="utf-8"))["results"][mode]
    prior = dict(pv6["fractional"], k_by_year=pv6["k_by_year"])
    kby = {int(y): k for y, k in prior["k_by_year"].items()}
    sw = ShadowWorld(*world_for(inst, groups[g]).init_args)
    rs = sw.run_state("2016-01-01", "2026-11-05", lambda D: kby[int(D[:4])], mode)
    checks.append(("state-capturing replay equals the engine total", abs(rs["total_won"] - prior["total_won"]) < 1.0 and abs(sum(h["value_won"] for h in rs["holdings"].values()) + rs["cash_won"] - rs["acct_won"]) < 1e-3))
    fr = fx_request("U29_topk", "2026-11-06", {"SPY": 0.5, "069500": 0.5}, {"QQQ": dict(units=1, value_won=100000.0, market="US")}, 1_000_000.0, 1400.0)
    checks.append(("fx request = US target minus US holdings, in won and dollars", fr["direction"] == "KRW->USD" and fr["amount_won"] == 400000 and abs(fr["amount_usd"] - 285.71) < 0.01))
    us_book = r1[0]["books"]["U29_topk"]
    checks.append(("first-day US book carries an fx request when it targets US ETFs", (us_book["fx_request"] is not None) == any(c.isalpha() for c in us_book["targets"])))
    for name, okk in checks:
        print("PASS" if okk else "FAIL", name)
    print("first-day targets", {b: x["targets"] for b, x in r1[0]["books"].items()}, "fixture", tmp)
    return all(ok for _, ok in checks)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    elif cmd == "once":
        run_once()
    elif cmd == "loop":
        loop()
    else:
        raise SystemExit("usage: selftest | once | loop")
