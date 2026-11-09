import csv
import hashlib
import json
import os
import sys
from datetime import datetime

import numpy as np

import mh_data as md

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "kr_etf", "kodex_daily_raw.json")
ADJ = os.path.join(HERE, "data", "kr_etf", "kodex_daily_adj.json")
TICK, FEE = 5.0, 0.00015
BEAR_DD, BEAR_LOOKBACK, BEAR_SYM, BEAR_HORIZONS = 0.20, 250, "069500", (60, 120, 250)
DEPOSIT = 10000.0
VAULT_SHARE = 0.5
WINDOW = ("2024-04-01", "2026-11-05")
SHARES = dict(combined=(0.3, 0.7), scalp_only=(1.0, 0.0), long_only=(0.0, 1.0), cash=(1.0, 0.0))
S_STOP = -0.02
OK_ENTRY = (md.FLAG["ok"], md.FLAG["delayed"], md.FLAG["unexited"], md.FLAG["exit_unknown"])


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def fee(x):
    return float(np.floor(x * FEE))


def affordable(cash, px):
    if px <= 0:
        return 0
    q = int(cash // px)
    while q > 0 and q * px + fee(q * px) > cash:
        q -= 1
    return q


class Ledger:
    def __init__(self, name, share_s, share_l, unknown_next_day=False):
        self.name, self.share, self.unknown_next_day = name, (share_s, share_l), unknown_next_day
        self.cash = {"S": 0.0, "L": 0.0}
        self.vault = 0.0
        self.vault_today = {"S": 0.0, "L": 0.0}
        self.pos = {"S": {}, "L": {}}
        self.deposits = 0.0
        self.trades = []
        self.daily = []
        self.next_id = 1
        self.pnl_unavailable_from = None
        self.cnt = dict(S_trades=0, L_trades=0, S_unfilled=0, S_delayed=0, S_carried=0, S_exit_unknown_open=0, S_stop_days=0, S_skipped_cash=0, L_skipped_cash=0, S_attempts=0, L_attempts=0, S_blocked_position=0, L_blocked_position=0, S_days_with_data_issues=0)

    def deposit(self):
        self.cash["S"] += DEPOSIT * self.share[0]
        self.cash["L"] += DEPOSIT * self.share[1]
        self.deposits += DEPOSIT

    def open(self, head, sym, px, day, ts, meta):
        if sym in self.pos[head]:
            self.cnt[f"{head}_blocked_position"] += 1
            return None
        q = affordable(self.cash[head], px)
        if q < 1:
            self.cnt[f"{head}_skipped_cash"] += 1
            return None
        cost = q * px + fee(q * px)
        self.cash[head] -= cost
        pid = self.next_id
        self.next_id += 1
        self.pos[head][sym] = dict(id=pid, q=q, px=px, cost=cost, day=day, ts=ts, fill_days=[day], **meta)
        return pid

    def close(self, head, sym, px, day, ts, reason, pid=None):
        p = self.pos[head].get(sym)
        if p is None or (pid is not None and p["id"] != pid):
            return None
        if ts is not None and p["ts"] is not None and ts < p["ts"]:
            raise RuntimeError(f"exit {ts} before entry {p['ts']} for {head} {sym}")
        self.pos[head].pop(sym)
        gross = p["q"] * px
        proceeds = gross - fee(gross)
        profit = proceeds - p["cost"]
        to_vault = VAULT_SHARE * profit if profit > 0 else 0.0
        self.vault += to_vault
        self.vault_today[head] += to_vault
        self.cash[head] += proceeds - to_vault
        self.cnt[f"{head}_trades"] += 1
        self.trades.append(dict(id=p["id"], head=head, sym=sym, q=p["q"], entry_day=p["day"], entry_ts=p["ts"], entry_px=p["px"], exit_day=day, exit_ts=ts, exit_px=px, cost=p["cost"], proceeds=proceeds, profit=profit, vault=to_vault, reason=reason))
        return profit

    def marked(self, head, marks, uncertain=None):
        return sum(p["q"] * marks.get(s, p["px"]) for s, p in self.pos[head].items() if uncertain is None or bool(p.get("uncertain")) == uncertain)

    def total(self, marks, include_uncertain=True):
        base = self.cash["S"] + self.cash["L"] + self.vault + self.marked("S", marks, uncertain=False) + self.marked("L", marks, uncertain=False)
        return base + (self.marked("S", marks, uncertain=True) if include_uncertain else 0.0)


def load_inputs(tables, train_dir, run_dirs):
    z = np.load(tables, allow_pickle=True)
    res = json.load(open(os.path.join(train_dir, "results.json"), encoding="utf-8"))
    pz = np.load(os.path.join(train_dir, "predictions.npz"), allow_pickle=True)
    s_feats = list(z["s_feats"].astype(str))
    l_feats = list(z["l_feats"].astype(str))

    def preds(head):
        out = {}
        for r in res[head]:
            if r.get("status") != "fitted":
                continue
            for i, p, s in zip(pz[f"{head}_{r['fold']}_idx"], pz[f"{head}_{r['fold']}_pred"], pz[f"{head}_{r['fold']}_signal"]):
                out[int(i)] = (float(p), bool(s), r["tau"])
        return out
    s_pred, l_pred = preds("S"), preds("L")
    S = []
    for i in range(len(z["s_y"])):
        day = str(z["s_day"][i])
        p, s, tau = s_pred.get(i, (np.nan, False, None))
        X = z["s_X"][i]
        S.append(dict(i=i, sym=str(z["s_sym"][i]), day=day, t=str(z["s_t"][i])[11:16], flag=int(z["s_flag"][i]), pb=float(z["s_pb"][i]), ps=float(z["s_ps"][i]), entry_tod=str(z["s_entry_tod"][i]), exit_tod=str(z["s_exit_ts"][i])[11:16] if z["s_exit_ts"][i] else None, pred=p, signal=s, tau=tau, r15=float(X[s_feats.index("r15")]), brk60=float(z["s_brk60"][i]), r1=float(X[s_feats.index("r1")]), day_bad=bool(z["s_day_incomplete"][i]) or int(z["s_day_conflicts"][i]) > 0))
    L = []
    for i in range(len(z["l_y"])):
        day = str(z["l_day"][i])
        p, s, tau = l_pred.get(i, (np.nan, False, None))
        X = z["l_X"][i]
        L.append(dict(i=i, sym=str(z["l_sym"][i]), day=day, pb=float(z["l_pb"][i]), ps=float(z["l_ps"][i]), entry_day=str(z["l_entry_day"][i]), exit_day=str(z["l_exit_day"][i]) if z["l_exit_day"][i] else None, pending=bool(z["l_pending"][i]), pred=p, signal=s, tau=tau, dip_first=bool(z["l_dip_first"][i]), ma120=float(X[l_feats.index("ma120")]), r1=float(X[l_feats.index("r1")])))
    bars = md.load_run(run_dirs) if run_dirs else {}
    raw = json.load(open(RAW, encoding="utf-8"))
    closes = {sym: dict(zip(raw[sym]["days"], map(float, raw[sym]["C"]))) for sym in md.SYMS}
    all_sessions = [d for d in raw["069500"]["days"] if d <= "2026-11-05"]
    sessions = [d for d in all_sessions if WINDOW[0] <= d <= WINDOW[1]]
    return S, L, bars, closes, sessions, all_sessions


def bear_windows(days, C, dd=BEAR_DD, look=BEAR_LOOKBACK):
    out, i, n = [], look - 1, len(C)
    while i < n:
        hi = float(C[i - look + 1:i + 1].max())
        if C[i] <= (1 - dd) * hi:
            j = i + 1
            while j < n and C[j] < hi:
                j += 1
            end = min(j, n - 1)
            out.append(dict(start=days[i], end=days[end], recovered=j < n, peak=hi, trough=float(C[i:end + 1].min()), sessions=end - i + 1, start_idx=i, end_idx=end))
            i = j + 1
        else:
            i += 1
    return out


def rule_signal(kind, r):
    if kind == "model":
        return r["signal"], (r["pred"] - (r["tau"] if r["tau"] is not None else 0.0))
    if kind == "S-D1":
        return r["r15"] <= -0.005, -r["r15"]
    if kind == "S-D2":
        return r["r15"] >= 0.005 and r["brk60"] > 0, r["brk60"]
    if kind == "L-D1":
        return r["dip_first"], -r["r1"]
    if kind == "L-D2":
        return r["ma120"] > 1.0, r["ma120"]
    return False, 0.0


def prev_close(closes, sym, day, all_sessions, fallback):
    i = all_sessions.index(day) if day in all_sessions else -1
    for k in range(i - 1, -1, -1):
        c = closes.get(sym, {}).get(all_sessions[k])
        if c is not None:
            return c
    return fallback


def session_close(day):
    return md.SESSION_LAST_EXT if day in md.SUNEUNG_DAYS else md.SESSION_LAST


def carried_exit(b, day):
    if b is None:
        return None, None, False
    cur = md.LATE_OPEN if day in md.LATE_OPEN_DAYS else md.SESSION_FIRST
    last = session_close(day)
    while cur <= last:
        k = b["idx"].get(cur)
        if k is None:
            return None, None, False
        if b["V"][k] > 0:
            return cur, b["O"][k] - TICK, True
        cur = md.tod_add(cur, 1)
    return None, None, True


def simulate(name, share, S, L, bars, closes, sessions, s_kind, l_kind, unknown_next_day=False, all_sessions=None):
    all_sessions = all_sessions or sessions
    led = Ledger(name, *share, unknown_next_day=unknown_next_day)
    s_by_day, l_by_day = {}, {}
    for r in S:
        s_by_day.setdefault(r["day"], []).append(r)
    for r in L:
        l_by_day.setdefault(r["day"], []).append(r)
    planned_l = []
    for day in sessions:
        led.deposit()
        led.vault_today = {"S": 0.0, "L": 0.0}
        e0 = led.cash["S"] + sum(p["q"] * prev_close(closes, s, day, all_sessions, p["px"]) for s, p in led.pos["S"].items())
        for sym, p in list(led.pos["L"].items()):
            if p["exit_day"] == day:
                led.close("L", sym, p["ps"], day, day, "scheduled", pid=p["id"])
        for p in [p for p in planned_l if p["entry_day"] == day]:
            led.cnt["L_attempts"] += 1
            led.open("L", p["sym"], p["pb"], day, day, dict(exit_day=p["exit_day"], ps=p["ps"], pending=p["pending"]))
        planned_l = [p for p in planned_l if p["entry_day"] > day]
        pending_exits = []
        for sym, p in list(led.pos["S"].items()):
            if p.get("uncertain") and not unknown_next_day:
                continue
            if p.get("flag") == md.FLAG["exit_unknown"] and not unknown_next_day:
                continue
            b = bars.get(sym, {}).get(day)
            if unknown_next_day:
                j = next((k for k in range(len(b["V"])) if b["V"][k] > 0), None) if b is not None else None
                if j is not None:
                    pending_exits.append(dict(id=p["id"], sym=sym, exit_tod=b["tod"][j], ps=b["O"][j] - TICK, reason="unknown_next_day_scenario" if (p.get("flag") == md.FLAG["exit_unknown"] or p.get("path_unknown")) else "carried_next_session", delayed=False))
                continue
            exit_tod, ps, known = carried_exit(b, day)
            if not known:
                p["uncertain"] = True
                p["path_unknown"] = True
                led.cnt["S_carried_path_unknown"] = led.cnt.get("S_carried_path_unknown", 0) + 1
                if led.pnl_unavailable_from is None:
                    led.pnl_unavailable_from = day
                continue
            if exit_tod is not None:
                pending_exits.append(dict(id=p["id"], sym=sym, exit_tod=exit_tod, ps=ps, reason="carried_next_session", delayed=False))

        def flush_exits(before_tod=None):
            for ex in sorted([x for x in pending_exits if before_tod is None or x["exit_tod"] < before_tod], key=lambda x: x["exit_tod"]):
                if led.close("S", ex["sym"], ex["ps"], day, day + "T" + ex["exit_tod"], ex["reason"], pid=ex["id"]) is not None and ex["delayed"]:
                    led.cnt["S_delayed"] += 1
                pending_exits.remove(ex)
        stop = False
        if led.share[0] > 0 and s_kind:
            rows = s_by_day.get(day, [])
            if any(r.get("day_bad") for r in rows):
                led.cnt["S_days_with_data_issues"] += 1
            for t in sorted({r["t"] for r in rows}):
                entry_tod = md.tod_add(t, md.S_ENTRY)
                flush_exits(entry_tod)
                mark = {}
                for s, p in led.pos["S"].items():
                    bb = bars.get(s, {}).get(day)
                    k = bb["idx"].get(t) if bb else None
                    mark[s] = bb["C"][k] if k is not None else prev_close(closes, s, day, all_sessions, p["px"])
                pnl_day = led.cash["S"] + led.marked("S", mark) + led.vault_today["S"] - e0
                if not stop and (e0 <= 0 or pnl_day <= S_STOP * e0):
                    stop = True
                    if e0 > 0:
                        led.cnt["S_stop_days"] += 1
                if stop:
                    continue
                cands = [(rule_signal(s_kind, r)[1], r) for r in rows if r["t"] == t and rule_signal(s_kind, r)[0]]
                for strength, r in sorted(cands, key=lambda x: (-x[0], x[1]["sym"])):
                    led.cnt["S_attempts"] += 1
                    if r["flag"] not in OK_ENTRY:
                        if r["flag"] == md.FLAG["unfilled"]:
                            led.cnt["S_unfilled"] += 1
                        continue
                    pid = led.open("S", r["sym"], r["pb"], day, day + "T" + r["entry_tod"], dict(exit_tod=r["exit_tod"], ps=r["ps"], flag=r["flag"]))
                    if pid is not None and r["exit_tod"]:
                        pending_exits.append(dict(id=pid, sym=r["sym"], exit_tod=r["exit_tod"], ps=r["ps"], reason="scheduled" if r["flag"] == md.FLAG["ok"] else "delayed", delayed=r["flag"] == md.FLAG["delayed"]))
        flush_exits()
        for sym, p in led.pos["S"].items():
            if p.get("flag") == md.FLAG["exit_unknown"] and not unknown_next_day:
                if not p.get("uncertain"):
                    p["uncertain"] = True
                    led.cnt["S_exit_unknown_open"] += 1
                    if led.pnl_unavailable_from is None:
                        led.pnl_unavailable_from = day
            elif p["day"] == day and not p.get("uncertain"):
                led.cnt["S_carried"] += 1
        if led.share[1] > 0 and l_kind:
            cands = []
            for r in l_by_day.get(day, []):
                on, strength = rule_signal(l_kind, r)
                if not on:
                    continue
                if r["sym"] in led.pos["L"] or any(p["sym"] == r["sym"] for p in planned_l):
                    led.cnt["L_blocked_position"] += 1
                    continue
                cands.append((strength, r))
            for strength, r in sorted(cands, key=lambda x: (-x[0], x[1]["sym"])):
                planned_l.append(dict(sym=r["sym"], entry_day=r["entry_day"], pb=r["pb"], ps=r["ps"], exit_day=r["exit_day"], pending=r["pending"]))
        marks = {s: closes[s].get(day, p["px"]) for h in ("S", "L") for s, p in led.pos[h].items()}
        tot = led.total(marks, include_uncertain=False)
        tot_assumed = led.total(marks, include_uncertain=True)
        avail = led.pnl_unavailable_from is None
        led.daily.append(dict(day=day, pnl_available=avail, total=tot if avail else None, total_incl_uncertain=tot_assumed, deposits=led.deposits, adj_pnl=(tot - led.deposits) if avail else None, adj_pnl_incl_uncertain=tot_assumed - led.deposits, cash_S=led.cash["S"], cash_L=led.cash["L"], vault=led.vault, open_S=len(led.pos["S"]), open_L=len(led.pos["L"]), certain_marked=led.marked("S", marks, uncertain=False) + led.marked("L", marks, uncertain=False), uncertain_marked=led.marked("S", marks, uncertain=True), stop=stop))
    return led


def simulate_buyhold(name, L, closes, sessions, sym="069500"):
    led = Ledger(name, 0.0, 1.0)
    by_entry = {}
    for r in L:
        if r["sym"] == sym:
            by_entry[r["entry_day"]] = r["pb"]
    for day in sessions:
        led.deposit()
        pb = by_entry.get(day)
        if pb is not None and pb > 0:
            q = affordable(led.cash["L"], pb)
            if q >= 1:
                cost = q * pb + fee(q * pb)
                led.cash["L"] -= cost
                p = led.pos["L"].get(sym)
                if p is None:
                    led.pos["L"][sym] = dict(id=led.next_id, q=q, px=pb, cost=cost, day=day, ts=day, exit_day=None, ps=np.nan, pending=True, lots=1, fill_days=[day])
                    led.next_id += 1
                else:
                    p["q"] += q
                    p["cost"] += cost
                    p["px"] = p["cost"] / p["q"]
                    p["lots"] += 1
                    p["fill_days"].append(day)
                led.cnt["L_attempts"] += 1
            else:
                led.cnt["L_skipped_cash"] += 1
        marks = {s: closes[s].get(day, p["px"]) for s, p in led.pos["L"].items()}
        tot = led.total(marks, include_uncertain=False)
        led.daily.append(dict(day=day, pnl_available=True, total=tot, total_incl_uncertain=tot, deposits=led.deposits, adj_pnl=tot - led.deposits, adj_pnl_incl_uncertain=tot - led.deposits, cash_S=0.0, cash_L=led.cash["L"], vault=0.0, open_S=0, open_L=len(led.pos["L"]), certain_marked=led.marked("L", marks), uncertain_marked=0.0, stop=False))
    return led


FIXED_SUBPERIODS = {"2020": [("2020-02-20", "2020-03-23")]}


def period_metrics(led, start, end):
    rows = [d for d in led.daily if start <= d["day"] <= end]
    if not rows:
        return None
    prev = [d for d in led.daily if d["day"] < start]
    start_total = prev[-1]["total"] if prev else 0.0
    start_dep = prev[-1]["deposits"] if prev else 0.0
    end_row = rows[-1]
    avail = all(d["pnl_available"] for d in rows) and (not prev or prev[-1]["pnl_available"])
    dep = end_row["deposits"] - start_dep
    pnl = (end_row["total"] - start_total - dep) if avail else None
    adj = np.array([d["adj_pnl"] for d in rows], float) if avail else None
    dd = None
    if avail and len(adj):
        base = (prev[-1]["adj_pnl"] if prev else 0.0)
        series = np.concatenate([[base], adj])
        peak = np.maximum.accumulate(series)
        dd = float((series - peak).min())
    expo = [d["certain_marked"] / d["total"] for d in rows if d["pnl_available"] and d["total"] and d["total"] > 0]
    return dict(start=rows[0]["day"], end=end_row["day"], sessions=len(rows), start_total=start_total, deposits=dep, end_total=end_row["total"], pnl=pnl, pnl_available=avail, max_dd_won=dd, exposure_avg=float(np.mean(expo)) if expo else None, exposure_max=float(np.max(expo)) if expo else None, open_marked_end=end_row["certain_marked"], uncertain_marked_end=end_row["uncertain_marked"])


def yearly_ledgers(L, closes, all_sessions, windows, years):
    out = []
    for y in years:
        ysess = [d for d in all_sessions if d[:4] == y]
        if not ysess:
            continue
        has_pred = any(np.isfinite(r["pred"]) for r in L if r["day"][:4] == y)
        subs = [(s, e) for s, e in FIXED_SUBPERIODS.get(y, [])]
        for w in windows:
            s, e = max(w["start"], ysess[0]), min(w["end"], ysess[-1])
            if s <= e:
                subs.append((s, e))
        rep = dict(year=y, sessions=len(ysess), model_available=has_pred, note="ledger starts the year with cash 0 (limitation); sub-periods inherit positions from the year ledger", ledgers={})
        cfgs = [("long_only_D1", "L-D1"), ("long_only_D2", "L-D2"), ("cash", None)] + ([("long_only_model", "model")] if has_pred else [])
        for name, lk in cfgs:
            led = simulate(name, SHARES["long_only"] if lk else SHARES["cash"], [], L, {}, closes, ysess, None, lk, all_sessions=all_sessions)
            rep["ledgers"][name] = dict(full_year=period_metrics(led, ysess[0], ysess[-1]), n_trades=len(led.trades), sub_periods={f"{s}~{e}": period_metrics(led, s, e) for s, e in subs})
        led = simulate_buyhold("long_only_buyhold", L, closes, ysess)
        rep["ledgers"]["long_only_buyhold"] = dict(full_year=period_metrics(led, ysess[0], ysess[-1]), n_trades=0, sub_periods={f"{s}~{e}": period_metrics(led, s, e) for s, e in subs})
        out.append(rep)
    return out


def cutoff_metrics(led, start, cutoff):
    m = period_metrics(led, start, cutoff)
    if m is None:
        return None
    m["trades_closed"] = sum(1 for t in led.trades if t["exit_day"] <= cutoff)
    m["positions_open_at_cutoff"] = sum(1 for h in ("S", "L") for p in led.pos[h].values() if p["day"] <= cutoff) + sum(1 for t in led.trades if t["entry_day"] <= cutoff < t["exit_day"])
    m["buy_fills_by_cutoff"] = sum(1 for t in led.trades if t["entry_day"] <= cutoff) + sum(sum(1 for d in p.get("fill_days", [p["day"]]) if d <= cutoff) for h in ("S", "L") for p in led.pos[h].values())
    return m


def window_metrics(led, closes, w, all_sessions):
    i0 = all_sessions.index(w["start"])
    last = led.daily[-1]["day"] if led.daily else None
    at_end = cutoff_metrics(led, w["start"], w["end"])
    out = dict(start=w["start"], end=w["end"], recovered=w["recovered"], complete_to_end=w["recovered"], at_end=at_end, deposits_at_end=at_end["deposits"] if at_end else None, adj_pnl_at_end=at_end["pnl"] if at_end else None, total_at_end=at_end["end_total"] if at_end else None, pnl_available_at_end=at_end["pnl_available"] if at_end else None, max_dd_won_pre_unavailable=at_end["max_dd_won"] if at_end else None, n_trades=at_end["trades_closed"] if at_end else None, horizons={}, run_end=last)
    for h in BEAR_HORIZONS:
        k = i0 + h
        if k < len(all_sessions) and last is not None and all_sessions[k] <= last:
            m = cutoff_metrics(led, w["start"], all_sessions[k])
            out["horizons"][str(h)] = dict(day=all_sessions[k], adj_pnl=m["pnl"], deposits=m["deposits"], max_dd_won=m["max_dd_won"], trades_closed=m["trades_closed"], exposure_avg=m["exposure_avg"]) if m else None
        else:
            out["horizons"][str(h)] = None
    return out


def report(led, closes, last_day):
    avail_rows = [d for d in led.daily if d["pnl_available"]]
    adj = np.array([d["adj_pnl"] for d in avail_rows], float)
    tot = np.array([d["total"] for d in avail_rows], float)
    dep = np.array([d["deposits"] for d in avail_rows], float)
    if len(adj):
        peak = np.maximum.accumulate(np.concatenate([[0.0], adj]))[1:]
        dd_won = float((adj - peak).min())
        dd_perf_pct = float(((adj - peak) / np.where(dep > 0, dep, np.nan)).min())
        peak_t = np.maximum.accumulate(tot)
        dd_total_pct = float(((tot - peak_t) / np.where(peak_t > 0, peak_t, np.nan)).min())
        daily_change = np.diff(np.concatenate([[0.0], adj]))
        top3 = np.sort(daily_change)[-3:] if len(daily_change) >= 3 else daily_change
    else:
        dd_won = dd_perf_pct = dd_total_pct = None
        top3 = np.array([])
    full = led.pnl_unavailable_from is None
    last = led.daily[-1] if led.daily else {}
    open_pos = [dict(head=h, sym=s, q=p["q"], entry=p["px"], entry_day=p["day"], mark=closes[s].get(last_day), value=p["q"] * closes[s].get(last_day, p["px"]), uncertain=bool(p.get("uncertain"))) for h in ("S", "L") for s, p in led.pos[h].items()]
    return dict(name=led.name, share=led.share, unknown_next_day_scenario=led.unknown_next_day, deposits=led.deposits, pnl_available=full, pnl_unavailable_from=led.pnl_unavailable_from, pnl_available_through=avail_rows[-1]["day"] if avail_rows else None, final_total_confirmed=float(tot[-1]) if full and len(tot) else None, adj_pnl_confirmed=float(adj[-1]) if full and len(adj) else None, components=dict(cash_S=led.cash["S"], cash_L=led.cash["L"], vault=led.vault, certain_marked=last.get("certain_marked"), uncertain_marked=last.get("uncertain_marked"), deposits=led.deposits), adj_pnl_incl_uncertain_marks_scenario=float(last["adj_pnl_incl_uncertain"]) if last else None, vault=led.vault, cash=dict(led.cash), max_dd_won=dd_won, max_dd_perf_pct_of_deposits=dd_perf_pct, max_dd_total_assets_pct=dd_total_pct, top3_day_contrib=float(top3.sum()) if len(top3) else None, adj_pnl_ex_top3=float(adj[-1] - top3.sum()) if full and len(adj) else None, counts=led.cnt, n_trades=len(led.trades), realized=float(sum(t["profit"] for t in led.trades)), open_positions=open_pos, uncertain_open_value=float(sum(p["value"] for p in open_pos if p["uncertain"])), note="main P&L/drawdown are unavailable from the first exit-unknown position onward; components and assumption scenarios are reported separately")


def write_ledger_files(out_dir, name, led):
    with open(os.path.join(out_dir, f"{name}_equity.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(led.daily[0].keys()))
        w.writeheader()
        w.writerows(led.daily)
    with open(os.path.join(out_dir, f"{name}_trades.jsonl"), "w", encoding="utf-8") as f:
        for t in led.trades:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


def main(tables, train_dir, run_dirs):
    S, L, bars, closes, sessions, all_sessions = load_inputs(tables, train_dir, run_dirs)
    out_dir = os.path.join(train_dir, "ledger")
    os.makedirs(out_dir, exist_ok=False)
    configs = [("combined_model", SHARES["combined"], "model", "model", False), ("scalp_only_model", SHARES["scalp_only"], "model", None, False), ("long_only_model", SHARES["long_only"], None, "model", False), ("cash", SHARES["cash"], None, None, False), ("scalp_only_D1", SHARES["scalp_only"], "S-D1", None, False), ("scalp_only_D2", SHARES["scalp_only"], "S-D2", None, False), ("long_only_D1", SHARES["long_only"], None, "L-D1", False), ("long_only_D2", SHARES["long_only"], None, "L-D2", False), ("combined_model_unknown_nextday_scenario", SHARES["combined"], "model", "model", True), ("scalp_only_model_unknown_nextday_scenario", SHARES["scalp_only"], "model", None, True)]
    if not S:
        configs = [c for c in configs if c[2] is None]
    reports = {}
    for name, share, sk, lk, unk in configs:
        led = simulate(name, share, S, L, bars, closes, sessions, sk, lk, unknown_next_day=unk, all_sessions=all_sessions)
        reports[name] = report(led, closes, sessions[-1])
        write_ledger_files(out_dir, name, led)
    led = simulate_buyhold("long_only_buyhold", L, closes, sessions)
    reports["long_only_buyhold"] = report(led, closes, sessions[-1])
    write_ledger_files(out_dir, "long_only_buyhold", led)
    adj = json.load(open(ADJ, encoding="utf-8"))[BEAR_SYM]
    adj_days = [d for d in adj["days"] if d <= "2026-11-05"]
    adj_C = np.array(adj["C"][:len(adj_days)], float)
    windows = bear_windows(adj_days, adj_C)
    bear = []
    for w in windows:
        i0 = all_sessions.index(w["start"])
        i1 = max(all_sessions.index(w["end"]), min(i0 + max(BEAR_HORIZONS), len(all_sessions) - 1))
        wsess = all_sessions[i0:i1 + 1]
        rows_w = [r for r in L if r["day"] >= w["start"]]
        in_w = [r for r in rows_w if r["day"] <= w["end"]]
        support = float(np.mean([np.isfinite(r["pred"]) for r in in_w])) if in_w else 0.0
        model_full = bool(in_w) and support == 1.0
        wrep = dict(window=w, run_end=wsess[-1], entries_from=all_sessions[i0 + 1] if i0 + 1 < len(all_sessions) else None, model_available=model_full, model_support_fraction=support, model_support="full" if model_full else ("partial_not_run" if support > 0 else "unsupported_pre2015"), ledgers={})
        wconfigs = [("long_only_D1", None, "L-D1"), ("long_only_D2", None, "L-D2"), ("cash", None, None)] + ([("long_only_model", None, "model")] if model_full else [])
        for name, sk, lk in wconfigs:
            share = SHARES["long_only"] if lk else SHARES["cash"]
            ledw = simulate(name, share, [], rows_w, {}, closes, wsess, sk, lk, all_sessions=all_sessions)
            wrep["ledgers"][name] = window_metrics(ledw, closes, w, all_sessions)
        ledw = simulate_buyhold("long_only_buyhold", rows_w, closes, wsess)
        wrep["ledgers"]["long_only_buyhold"] = window_metrics(ledw, closes, w, all_sessions)
        bear.append(wrep)
    meta = dict(script_sha256=sha(os.path.abspath(__file__)), tables_sha256=sha(tables), train_dir=os.path.relpath(train_dir, HERE), window=WINDOW, sessions=len(sessions), deposit=DEPOSIT, vault_share=VAULT_SHARE, shares=SHARES, s_stop=S_STOP, run_dirs=[os.path.relpath(d, HERE) for d in run_dirs], generated=datetime.now().isoformat(timespec="seconds"))
    years = sorted({d[:4] for d in all_sessions if d >= "2015-01-01"})
    yearly = yearly_ledgers(L, closes, all_sessions, windows, years)
    meta.update(bear_rule=dict(sym=BEAR_SYM, drawdown=BEAR_DD, lookback=BEAR_LOOKBACK, horizons=list(BEAR_HORIZONS), basis="adjusted close; start = first session at or below (1-dd) x rolling high; end = first session at or above that high, else data end"), fixed_subperiods=FIXED_SUBPERIODS, yearly_years=years, adj_sha256=sha(ADJ))
    json.dump(dict(meta=meta, reports=reports, bear_windows=bear, yearly=yearly), open(os.path.join(out_dir, "ledger_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for yrep in yearly:
        line = f"연도 {yrep['year']} ({yrep['sessions']}세션, 모델 {'있음' if yrep['model_available'] else '없음'}):"
        for name, m in yrep["ledgers"].items():
            fy = m["full_year"]
            pnl = "불가" if fy["pnl"] is None else f"{fy['pnl']:+,.0f}"
            line += f" {name} {pnl}/{fy['deposits']:,.0f} 노출 {fy['exposure_avg'] if fy['exposure_avg'] is None else round(fy['exposure_avg'] * 100)}%"
        print(line)
    for wrep in bear:
        w = wrep["window"]
        line = f"하락장 {w['start']}~{w['end']} ({'회복' if w['recovered'] else '미회복'}, {w['sessions']}세션, 저점 {w['trough']/w['peak']-1:+.1%}) 모델 {'있음' if wrep['model_available'] else '없음'}:"
        for name, m in wrep["ledgers"].items():
            pnl = "불가" if m["adj_pnl_at_end"] is None else f"{m['adj_pnl_at_end']:+,.0f}"
            line += f" {name} 끝 {pnl}/{m['deposits_at_end']:,.0f}원"
        print(line)
    for name, r in reports.items():
        pnl = f"{r['adj_pnl_confirmed']:>+10,.0f}" if r["adj_pnl_confirmed"] is not None else f"불가({r['pnl_unavailable_from']}부터)"
        dd = f"{r['max_dd_won']:>+9,.0f}원" if r["max_dd_won"] is not None else "n/a"
        print(f"{name:42s} 입금 {r['deposits']:>12,.0f} 확정조정손익 {pnl} 가정포함 {r['adj_pnl_incl_uncertain_marks_scenario']:>+10,.0f} 금고 {r['vault']:>9,.0f} 낙폭 {dd} 거래 {r['n_trades']:>4} 미체결 {r['counts']['S_unfilled']} 지연 {r['counts']['S_delayed']} 이월 {r['counts']['S_carried']} 불확실 {r['counts']['S_exit_unknown_open']}({r['uncertain_open_value']:,.0f}원) 중단일 {r['counts']['S_stop_days']} 자료문제일 {r['counts']['S_days_with_data_issues']} 미청산 {len(r['open_positions'])}")
    print("saved", out_dir)


def _srow(day, t, b, sym="069500", **kw):
    o = md.s_outcome(b, b["idx"][t])
    r = dict(i=0, sym=sym, day=day, t=t, flag=o["flag"], pb=o["pb"], ps=o["ps"], entry_tod=o["entry_tod"], exit_tod=o["exit_ts"][11:16] if o["exit_ts"] else None, pred=0.01, signal=True, tau=0.001, r15=0.0, brk60=0.0, r1=0.0)
    r.update(kw)
    return r


def selftest():
    assert affordable(1000.0, 1000.0) == 1 and affordable(10000.0, 3000.0) == 3 and affordable(999.0, 1000.0) == 0 and affordable(1000.15, 1000.0) == 1
    day1, day2, day3 = "2026-01-05", "2026-01-06", "2026-01-07"
    b1, b2, b3 = md.synth_day(day1), md.synth_day(day2, base=81000.0), md.synth_day(day3, base=82000.0)
    bars = {"069500": {day1: b1, day2: b2, day3: b3}}
    closes = {"069500": {day1: float(b1["C"][-1]), day2: float(b2["C"][-1]), day3: float(b3["C"][-1])}, "229200": {}}
    sessions = [day1, day2, day3]
    led0 = simulate("t0", (1.0, 0.0), [_srow(day1, "10:05", b1)], [], bars, closes, sessions, "model", None)
    assert led0.cnt["S_trades"] == 0 and led0.cnt["S_skipped_cash"] == 1
    Sb = [_srow(day1, "10:05", b1, pb=3000.0, ps=3100.0), _srow(day1, "10:10", b1, pred=0.02, pb=3050.0, ps=3100.0)]
    led = simulate("t", (1.0, 0.0), Sb, [], bars, closes, sessions, "model", None)
    assert led.cnt["S_trades"] == 1 and led.cnt["S_blocked_position"] == 1 and led.trades[0]["entry_px"] == 3000.0, led.cnt
    Sc = [_srow(day1, "10:05", b1, pb=3000.0, ps=3100.0), _srow(day1, "10:25", b1, pred=0.02, pb=3050.0, ps=3200.0)]
    ledc = simulate("tc", (1.0, 0.0), Sc, [], bars, closes, sessions, "model", None)
    assert ledc.cnt["S_trades"] == 2 and ledc.cnt["S_blocked_position"] == 0, ledc.cnt
    t1, t2 = ledc.trades
    assert t1["exit_ts"].endswith("10:22") and t1["exit_px"] == 3100.0 and t2["entry_ts"].endswith("10:27") and t2["exit_ts"].endswith("10:42") and t2["exit_px"] == 3200.0 and all(t["exit_ts"] > t["entry_ts"] for t in ledc.trades)
    S2 = [_srow(day1, "10:05", b1, pb=3000.0, ps=3100.0)]
    led2 = simulate("t2", (1.0, 0.0), S2, [], bars, closes, sessions, "model", None)
    t = led2.trades[0]
    cost = t["q"] * 3000.0 + fee(t["q"] * 3000.0)
    proceeds = t["q"] * 3100.0 - fee(t["q"] * 3100.0)
    profit = proceeds - cost
    assert t["q"] == 3 and abs(t["profit"] - profit) < 1e-9 and abs(led2.vault - 0.5 * profit) < 1e-9
    assert abs(led2.cash["S"] - (3 * DEPOSIT - cost + proceeds - 0.5 * profit)) < 1e-9
    r2 = report(led2, closes, day3)
    assert r2["pnl_available"] and abs(r2["final_total_confirmed"] - (led2.cash["S"] + led2.vault)) < 1e-9 and abs(r2["adj_pnl_confirmed"] - (r2["final_total_confirmed"] - 3 * DEPOSIT)) < 1e-9
    S3 = [_srow(day1, "10:05", b1, pb=3000.0, ps=2900.0)]
    led3 = simulate("t3", (1.0, 0.0), S3, [], bars, closes, sessions, "model", None)
    r3 = report(led3, closes, day3)
    assert led3.vault == 0.0 and r3["max_dd_won"] < 0 and abs(r3["max_dd_won"] - led3.trades[0]["profit"]) < 1e-9 and r3["max_dd_perf_pct_of_deposits"] < 0
    Sstop = [_srow(day1, "10:05", b1, pb=3000.0, ps=2900.0), _srow(day1, "10:30", b1, pred=0.02, pb=3000.0, ps=3100.0)]
    leds = simulate("ts", (1.0, 0.0), Sstop, [], bars, closes, sessions, "model", None)
    assert leds.cnt["S_stop_days"] == 1 and leds.cnt["S_trades"] == 1, leds.cnt
    bdip = md.synth_day(day1)
    bdip["C"][bdip["idx"]["10:08"]:bdip["idx"]["10:21"]] = 2865.0
    Sdip = [_srow(day1, "10:05", bdip, pb=3000.0, ps=3000.0), _srow(day1, "10:10", bdip, pred=0.0, signal=False, pb=3000.0, ps=3000.0), _srow(day1, "10:30", bdip, pred=0.02, pb=3000.0, ps=3100.0)]
    leddip = simulate("tdip", (1.0, 0.0), Sdip, [], {"069500": {day1: bdip, day2: b2, day3: b3}}, closes, sessions, "model", None)
    assert leddip.cnt["S_stop_days"] == 1 and leddip.cnt["S_trades"] == 1, leddip.cnt
    Lwin = [dict(i=0, sym="229200", day=day1, pb=1000.0, ps=1500.0, entry_day=day2, exit_day=day3, pending=False, pred=0.05, signal=True, tau=0.01, dip_first=False, ma120=1.1, r1=0.0)]
    Sstop2 = [_srow(day3, "10:05", b3, pb=3000.0, ps=2900.0), _srow(day3, "10:30", b3, pred=0.02, pb=3000.0, ps=3100.0)]
    closes2 = dict(closes, **{"229200": {day1: 1000.0, day2: 1000.0, day3: 1500.0}})
    ledm = simulate("tm", (0.5, 0.5), Sstop2, Lwin, bars, closes2, sessions, "model", "model")
    assert ledm.cnt["L_trades"] == 1 and ledm.trades[0]["profit"] > 0 and ledm.cnt["S_stop_days"] == 1
    bu = md.synth_day(day1)
    bu["V"][bu["idx"]["15:17"]:] = 0
    S4 = [_srow(day1, "15:00", bu, pb=3000.0)]
    assert S4[0]["flag"] == md.FLAG["unexited"] and S4[0]["exit_tod"] is None
    b2hi = md.synth_day(day2, base=3100.0)
    b2hi["V"][:b2hi["idx"]["12:00"]] = 0
    b229 = md.synth_day(day2, base=12000.0)
    bars4 = {"069500": {day1: bu, day2: b2hi, day3: b3}, "229200": {day2: b229}}
    closes4 = {"069500": {day1: 3000.0, day2: float(b2hi["C"][-1]), day3: float(b3["C"][-1])}, "229200": {day1: 12000.0, day2: 12000.0, day3: 12000.0}}
    S4.append(_srow(day2, "10:05", b229, sym="229200", pred=0.02, pb=12001.0, ps=12500.0))
    S4.append(_srow(day2, "12:30", b229, sym="229200", pred=0.02, pb=10000.0, ps=10500.0))
    led4 = simulate("t4", (1.0, 0.0), S4, [], bars4, closes4, sessions, "model", None)
    carried = [t for t in led4.trades if t["reason"] == "carried_next_session"]
    assert led4.cnt["S_carried"] == 1 and len(carried) == 1 and carried[0]["exit_ts"].endswith("12:00") and abs(carried[0]["exit_px"] - (b2hi["O"][b2hi["idx"]["12:00"]] - TICK)) < 1e-9, (led4.cnt, led4.trades)
    assert led4.cnt["S_skipped_cash"] == 1 and led4.cnt["S_stop_days"] == 0, led4.cnt
    t229 = [t for t in led4.trades if t["sym"] == "229200"]
    assert len(t229) == 1 and t229[0]["entry_ts"].endswith("12:32") and t229[0]["entry_px"] == 10000.0, led4.trades
    assert led4.daily[0]["open_S"] == 1 and abs(led4.daily[0]["certain_marked"] - 3 * 3000.0) < 1e-9
    b2low = md.synth_day(day2, base=2000.0)
    bars4b = {"069500": {day1: bu, day2: b2low, day3: b3}}
    closes4b = {"069500": {day1: 3000.0, day2: float(b2low["C"][-1]), day3: float(b3["C"][-1])}, "229200": {}}
    S4b = [S4[0], _srow(day2, "10:30", b2low, pred=0.02, pb=2000.0, ps=2100.0)]
    led4b = simulate("t4b", (1.0, 0.0), S4b, [], bars4b, closes4b, sessions, "model", None)
    assert led4b.cnt["S_stop_days"] == 1 and led4b.cnt["S_trades"] == 1 and led4b.trades[0]["reason"] == "carried_next_session", (led4b.cnt, led4b.trades)
    b2gap = md.synth_day(day2, base=3100.0, drop=("09:01",))
    ledg = simulate("tg", (1.0, 0.0), [S4[0]], [], {"069500": {day1: bu, day2: b2gap, day3: b3}}, closes4b, sessions, "model", None)
    assert len(ledg.trades) == 0 and ledg.pos["S"]["069500"].get("uncertain") and ledg.cnt.get("S_carried_path_unknown") == 1 and ledg.pnl_unavailable_from == day2 and ledg.cnt["S_carried"] == 1, (ledg.cnt, ledg.pnl_unavailable_from)
    rg = report(ledg, closes4b, day3)
    assert not rg["pnl_available"] and rg["pnl_available_through"] == day1 and rg["max_dd_won"] is not None and rg["adj_pnl_confirmed"] is None, rg
    assert abs(ledg.cash["S"] - (3 * DEPOSIT - 9001.0)) < 1e-9
    ledgs = simulate("tgs", (1.0, 0.0), [S4[0]], [], {"069500": {day1: bu, day2: b2gap, day3: b3}}, closes4b, sessions, "model", None, unknown_next_day=True)
    assert len(ledgs.trades) == 1 and ledgs.trades[0]["exit_ts"].endswith("09:02") and ledgs.trades[0]["reason"] == "carried_next_session", ledgs.trades
    lednd = simulate("tnd", (1.0, 0.0), [S4[0]], [], {"069500": {day1: bu, day3: b3}}, closes4b, sessions, "model", None)
    assert len(lednd.trades) == 0 and lednd.pos["S"]["069500"].get("uncertain") and lednd.pnl_unavailable_from == day2, lednd.cnt
    lednds = simulate("tnds", (1.0, 0.0), [S4[0]], [], {"069500": {day1: bu, day3: b3}}, closes4b, sessions, "model", None, unknown_next_day=True)
    assert len(lednds.trades) == 1 and lednds.trades[0]["exit_day"] == day3, lednds.trades
    b2mid = md.synth_day(day2, base=3100.0, drop=("09:02",))
    b2mid["V"][b2mid["idx"]["09:01"]] = 0
    ledmid = simulate("tmid", (1.0, 0.0), [S4[0]], [], {"069500": {day1: bu, day2: b2mid, day3: b3}}, closes4b, sessions, "model", None)
    assert len(ledmid.trades) == 0 and ledmid.pnl_unavailable_from == day2, ledmid.cnt
    assert carried_exit(b2hi, day2) == ("12:00", b2hi["O"][b2hi["idx"]["12:00"]] - TICK, True) and carried_exit(None, day2) == (None, None, False)
    sday = "2024-11-14"
    assert sday in md.SUNEUNG_DAYS and session_close(sday) == "16:30" and session_close("2025-01-02") == "15:30"
    su = md.synth_day(sday, n=390, start=(10, 1))
    su["V"][:su["idx"]["16:00"]] = 0
    assert carried_exit(su, sday) == ("16:00", su["O"][su["idx"]["16:00"]] - TICK, True)
    su_gap = md.synth_day(sday, n=390, start=(10, 1), drop=("15:45",))
    su_gap["V"][:] = 0
    su_gap["V"][su_gap["idx"]["16:00"]] = 1
    assert carried_exit(su_gap, sday) == (None, None, False)
    su_zero = md.synth_day(sday, n=390, start=(10, 1))
    su_zero["V"][:] = 0
    assert carried_exit(su_zero, sday) == (None, None, True)
    fd = md.synth_day("2025-01-02", n=330, start=(10, 1))
    fd["V"][:] = 0
    assert carried_exit(fd, "2025-01-02") == (None, None, True)
    sess_s = ["2024-11-13", sday, "2024-11-15"]
    bu2 = md.synth_day("2024-11-13")
    bu2["V"][bu2["idx"]["15:17"]:] = 0
    Ssu = [_srow("2024-11-13", "15:00", bu2, pb=3000.0)]
    assert Ssu[0]["flag"] == md.FLAG["unexited"]
    b15 = md.synth_day("2024-11-15", base=3200.0)
    bars_su = {"069500": {"2024-11-13": bu2, sday: su, "2024-11-15": b15}}
    closes_su = {"069500": {"2024-11-13": 3000.0, sday: float(su["C"][-1]), "2024-11-15": float(b15["C"][-1])}, "229200": {}}
    Lsu = [dict(i=0, sym="229200", day="2024-11-13", pb=1000.0, ps=1100.0, entry_day=sday, exit_day="2024-11-15", pending=False, pred=0.05, signal=True, tau=0.01, dip_first=False, ma120=1.1, r1=0.0)]
    closes_su["229200"] = {"2024-11-13": 1000.0, sday: 1000.0, "2024-11-15": 1100.0}
    ledsu = simulate("tsu", (0.5, 0.5), Ssu, Lsu, bars_su, closes_su, sess_s, "model", "model")
    ct = [t for t in ledsu.trades if t["reason"] == "carried_next_session"]
    assert len(ct) == 1 and ct[0]["exit_day"] == sday and ct[0]["exit_ts"].endswith("16:00") and ledsu.pnl_unavailable_from is None, (ledsu.trades, ledsu.cnt)
    lt = [t for t in ledsu.trades if t["head"] == "L"]
    assert len(lt) == 1 and lt[0]["entry_day"] == sday and lt[0]["exit_day"] == "2024-11-15" and ledsu.deposits == 3 * DEPOSIT, lt
    bx = md.synth_day(day1, drop=("10:22",))
    S5 = [_srow(day1, "10:05", bx, pb=3000.0)]
    assert S5[0]["flag"] == md.FLAG["exit_unknown"]
    bars5 = {"069500": {day1: bx, day2: b2, day3: b3}}
    ledu = simulate("tu", (1.0, 0.0), S5, [], bars5, closes, sessions, "model", None)
    assert len(ledu.trades) == 0 and ledu.pos["S"]["069500"].get("uncertain") and ledu.cnt["S_exit_unknown_open"] == 1 and ledu.cnt["S_carried"] == 0 and ledu.pnl_unavailable_from == day1
    ru = report(ledu, closes, day3)
    assert not ru["pnl_available"] and ru["adj_pnl_confirmed"] is None and ru["final_total_confirmed"] is None and ru["max_dd_won"] is None and ru["pnl_unavailable_from"] == day1 and ru["pnl_available_through"] is None, ru
    assert abs(ru["components"]["cash_S"] - ledu.cash["S"]) < 1e-9 and ru["components"]["uncertain_marked"] == 3 * closes["069500"][day3] and ru["uncertain_open_value"] == 3 * closes["069500"][day3]
    assert abs(ru["adj_pnl_incl_uncertain_marks_scenario"] - (ledu.cash["S"] + 3 * closes["069500"][day3] - 3 * DEPOSIT)) < 1e-9
    assert all(d["adj_pnl"] is None and not d["pnl_available"] for d in ledu.daily)
    ledv = simulate("tv", (1.0, 0.0), S5, [], bars5, closes, sessions, "model", None, unknown_next_day=True)
    assert len(ledv.trades) == 1 and ledv.trades[0]["reason"] == "unknown_next_day_scenario" and ledv.trades[0]["exit_day"] == day2 and not ledv.pos["S"] and report(ledv, closes, day3)["pnl_available"]
    Sdb = [dict(_srow(day1, "10:05", b1, pb=3000.0, ps=3100.0), day_bad=True)]
    leddb = simulate("tdb", (1.0, 0.0), Sdb, [], bars, closes, sessions, "model", None)
    assert leddb.cnt["S_trades"] == 1 and leddb.cnt["S_days_with_data_issues"] == 1
    L = [dict(i=0, sym="069500", day=day1, pb=3000.0, ps=3300.0, entry_day=day2, exit_day=day3, pending=False, pred=0.05, signal=True, tau=0.01, dip_first=False, ma120=1.1, r1=0.0)]
    led5 = simulate("t5", (0.3, 0.7), [], L, bars, closes, sessions, None, "model")
    t5 = led5.trades[0]
    assert t5["entry_day"] == day2 and t5["exit_day"] == day3 and t5["q"] == affordable(2 * DEPOSIT * 0.7, 3000.0) == 4 and abs(led5.cash["S"] - 3 * DEPOSIT * 0.3) < 1e-9
    L6 = [dict(L[0]), dict(L[0], i=1, sym="229200", day=day2, pb=12000.0, ps=12500.0, entry_day=day3, exit_day=None, pending=True)]
    closes6 = dict(closes, **{"229200": {day1: 12000.0, day2: 12000.0, day3: 12000.0}})
    led6 = simulate("t6", (0.0, 1.0), [], L6, bars, closes6, sessions, None, "model")
    assert led6.cnt["L_trades"] == 1 and "229200" in led6.pos["L"] and led6.cnt["L_skipped_cash"] == 0
    led7 = simulate("cash", SHARES["cash"], Sb, L, bars, closes, sessions, None, None)
    r7 = report(led7, closes, day3)
    assert r7["final_total_confirmed"] == 3 * DEPOSIT and r7["adj_pnl_confirmed"] == 0.0 and r7["max_dd_won"] == 0.0
    Lbh = [dict(i=0, sym="069500", day=day1, pb=3000.0, ps=np.nan, entry_day=day2, exit_day=None, pending=True, pred=0.0, signal=False, tau=None, dip_first=False, ma120=1.0, r1=0.0), dict(i=1, sym="069500", day=day2, pb=3000.0, ps=np.nan, entry_day=day3, exit_day=None, pending=True, pred=0.0, signal=False, tau=None, dip_first=False, ma120=1.0, r1=0.0)]
    ledbh = simulate_buyhold("bh", Lbh, closes, sessions)
    p = ledbh.pos["L"]["069500"]
    assert p["q"] == 6 + 3 and p["lots"] == 2 and abs(p["cost"] - (6 * 3000 + fee(18000) + 3 * 3000 + fee(9000))) < 1e-9 and len(ledbh.trades) == 0, p
    rbh = report(ledbh, closes, day3)
    assert rbh["pnl_available"] and abs(rbh["final_total_confirmed"] - (ledbh.cash["L"] + 9 * closes["069500"][day3])) < 1e-9 and rbh["deposits"] == 3 * DEPOSIT and len(rbh["open_positions"]) == 1, rbh
    n = 700
    Cb = np.full(n, 100.0)
    Cb[300:320] = np.linspace(100, 75, 20)
    Cb[320:400] = 75.0
    Cb[400:420] = np.linspace(75, 101, 20)
    Cb[420:600] = 101.0
    Cb[600:] = 70.0
    daysb = [f"d{i:04d}" for i in range(n)]
    wins = bear_windows(daysb, Cb)
    assert len(wins) == 2, wins
    first_below = next(i for i in range(300, 320) if Cb[i] <= 80.0)
    assert wins[0]["start"] == daysb[first_below] and wins[0]["recovered"] and Cb[int(wins[0]["end"][1:])] >= 100.0 and int(wins[0]["end"][1:]) >= 400 and wins[0]["trough"] == 75.0 and wins[0]["peak"] == 100.0, wins[0]
    assert wins[1]["start"] == daysb[600] and not wins[1]["recovered"] and wins[1]["end"] == daysb[n - 1], wins[1]
    assert bear_windows(daysb, np.full(n, 100.0)) == []
    wsess = sessions
    w = dict(start=day1, end=day3, recovered=False, peak=1.0, trough=1.0, sessions=3)
    ledc = simulate("cash_w", SHARES["cash"], [], [], {}, closes, wsess, None, None, all_sessions=sessions)
    m = window_metrics(ledc, closes, w, sessions)
    assert m["adj_pnl_at_end"] == 0.0 and m["deposits_at_end"] == 3 * DEPOSIT and m["horizons"]["60"] is None and m["max_dd_won_pre_unavailable"] == 0.0 and m["n_trades"] == 0 and not m["complete_to_end"], m
    w2 = dict(start=day1, end=day2, recovered=True, peak=1.0, trough=1.0, sessions=2)
    Lw = [dict(i=0, sym="069500", day=day1, pb=3000.0, ps=2000.0, entry_day=day2, exit_day=day3, pending=False, pred=0.05, signal=True, tau=0.01, dip_first=False, ma120=1.1, r1=0.0)]
    ledw = simulate("mw", SHARES["long_only"], [], Lw, {}, closes, sessions, None, "model", all_sessions=sessions)
    assert len(ledw.trades) == 1 and ledw.trades[0]["exit_day"] == day3 and ledw.trades[0]["profit"] < 0
    mw = window_metrics(ledw, closes, w2, sessions)
    assert mw["n_trades"] == 0 and mw["at_end"]["buy_fills_by_cutoff"] == 1 and mw["at_end"]["positions_open_at_cutoff"] == 1 and mw["complete_to_end"], mw
    mbh = cutoff_metrics(ledbh, day1, day3)
    assert mbh["buy_fills_by_cutoff"] == 2 and mbh["positions_open_at_cutoff"] == 1 and mbh["trades_closed"] == 0, mbh
    mbh2 = cutoff_metrics(ledbh, day1, day2)
    assert mbh2["buy_fills_by_cutoff"] == 1 and mbh2["positions_open_at_cutoff"] == 1, mbh2
    assert ledbh.pos["L"]["069500"]["fill_days"] == [day2, day3]
    mbh1 = cutoff_metrics(ledbh, day1, day1)
    assert mbh1["buy_fills_by_cutoff"] == 0 and mbh1["positions_open_at_cutoff"] == 0, mbh1
    assert abs(mw["adj_pnl_at_end"] - (ledw.daily[1]["total"] - 2 * DEPOSIT)) < 1e-9 and mw["max_dd_won_pre_unavailable"] >= min(0.0, ledw.daily[1]["adj_pnl"]) - 1e-9, mw
    Lbh2 = [dict(i=0, sym="069500", day="2026-01-02", pb=3000.0, ps=np.nan, entry_day=day1, exit_day=None, pending=True, pred=0.0, signal=False, tau=None, dip_first=False, ma120=1.0, r1=0.0), dict(i=1, sym="069500", day=day1, pb=3000.0, ps=np.nan, entry_day=day2, exit_day=None, pending=True, pred=0.0, signal=False, tau=None, dip_first=False, ma120=1.0, r1=0.0)]
    rows_w = [r for r in Lbh2 if r["day"] >= day1]
    ledbw = simulate_buyhold("bhw", rows_w, closes, sessions)
    assert ledbw.pos["L"]["069500"]["day"] == day2 and ledbw.pos["L"]["069500"]["lots"] == 1 and ledbw.cnt["L_attempts"] == 1, ledbw.pos
    assert simulate_buyhold("bhall", Lbh2, closes, sessions).pos["L"]["069500"]["lots"] == 2
    pm = period_metrics(led5, day2, day3)
    assert pm["start_total"] == led5.daily[0]["total"] and pm["deposits"] == 2 * DEPOSIT and abs(pm["pnl"] - (led5.daily[-1]["total"] - led5.daily[0]["total"] - 2 * DEPOSIT)) < 1e-9 and pm["sessions"] == 2 and pm["pnl_available"], pm
    assert pm["exposure_avg"] is not None and 0 <= pm["exposure_avg"] <= 1
    pmu = period_metrics(ledu, day2, day3)
    assert pmu["pnl"] is None and not pmu["pnl_available"]
    yl = yearly_ledgers(L, closes, sessions, [dict(start=day2, end=day3)], ["2026"])
    assert len(yl) == 1 and set(yl[0]["ledgers"]) == {"long_only_D1", "long_only_D2", "cash", "long_only_model", "long_only_buyhold"} and f"{day2}~{day3}" in yl[0]["ledgers"]["cash"]["sub_periods"] and yl[0]["ledgers"]["cash"]["full_year"]["pnl"] == 0.0, yl[0]["ledgers"]["cash"]
    assert yl[0]["ledgers"]["long_only_model"]["n_trades"] == 1
    print("selftest ok", dict(sha=sha(os.path.abspath(__file__))))


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    elif len(sys.argv) >= 3:
        main(sys.argv[1], sys.argv[2], sys.argv[3:])
    else:
        print("usage: mh_ledger.py <tables.npz> <train_dir> [<run_dir> ...] | selftest")
