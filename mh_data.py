import hashlib
import json
import os
import sys
from datetime import date

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ADJ = os.path.join(HERE, "data", "kr_etf", "kodex_daily_adj.json")
RAW = os.path.join(HERE, "data", "kr_etf", "kodex_daily_raw.json")
OUT_DIR = os.path.join(HERE, "data", "mh")
SYMS = ("069500", "229200")
TICK, FEE = 5.0, 0.00015
SESSION_FIRST, SESSION_LAST = "09:01", "15:30"
S_GRID = [f"{h:02d}:{m:02d}" for h in range(10, 16) for m in range(0, 60, 5) if "10:05" <= f"{h:02d}:{m:02d}" <= "15:00"]
S_HIST, S_ENTRY, S_EXIT, S_LAST_EXIT = 60, 2, 17, "15:19"
S_FEATS = ["r1", "r5", "r15", "r30", "r60", "ret_open", "range_pos", "vol_ratio", "rv15", "grid_idx", "dow", "prev_ret", "prev_r5", "prev_r20", "prev_vol20", "sym_id"]
L_FEATS = ["r1", "r5", "r20", "r60", "r120", "r250", "dd60", "dd250", "vol20", "volratio", "ma20", "ma60", "ma120", "month", "sym_id"]
L_H, L_WARM, DAILY_WARM = 60, 250, 22
FLAG = dict(ok=0, unfilled=1, delayed=2, unexited=3, exit_unknown=4, entry_missing=5, feature_gap=6, decision_missing=7, special_session=8)
LATE_OPEN = "10:01"
FIRST_SESSION_DAYS = {"2023-01-02", "2024-01-02", "2025-01-02", "2026-01-02"}
SUNEUNG_DAYS = {"2022-11-17", "2023-11-16", "2024-11-14", "2025-11-13", "2026-11-19"}
LATE_OPEN_DAYS = FIRST_SESSION_DAYS | SUNEUNG_DAYS
SESSION_LAST_EXT = "16:30"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def fee(x):
    return np.floor(np.asarray(x, float) * FEE)


def net_ret(pb, ps):
    fb, fs = fee(pb), fee(ps)
    return (ps - fs - pb - fb) / (pb + fb)


def tod_add(tod, minutes):
    h, m = int(tod[:2]), int(tod[3:]) + minutes
    h += m // 60
    m %= 60
    return f"{h:02d}:{m:02d}"


def load_daily():
    adj = json.load(open(ADJ, encoding="utf-8"))
    raw = json.load(open(RAW, encoding="utf-8"))
    out = {}
    for sym in SYMS:
        a, r = adj[sym], raw[sym]
        if a["days"] != r["days"]:
            raise SystemExit(f"{sym}: adjusted/raw day lists differ")
        keep = [i for i, d in enumerate(a["days"]) if d <= "2026-11-05"]
        out[sym] = dict(days=[a["days"][i] for i in keep], Ca=np.array([a["C"][i] for i in keep], float), Oa=np.array([a["O"][i] for i in keep], float), Va=np.array([a["V"][i] for i in keep], float), Or=np.array([r["O"][i] for i in keep], float), Cr=np.array([r["C"][i] for i in keep], float))
    return out


def make_day(ts, O, H, L, C, V, cls="normal", incomplete=False, conflicts=0, dups=0):
    tod = [t[11:16] for t in ts]
    return dict(ts=list(ts), tod=tod, idx={t: i for i, t in enumerate(tod)}, O=np.asarray(O, float), H=np.asarray(H, float), L=np.asarray(L, float), C=np.asarray(C, float), V=np.asarray(V, float), cls=cls, incomplete=incomplete, conflicts=conflicts, dups=dups)


def load_run(run_dirs):
    out = {}
    for run_dir in run_dirs:
        ddir = os.path.join(run_dir, "days")
        for fn in sorted(os.listdir(ddir)):
            sym, day = fn[:-5].split("_", 1)
            if day in out.get(sym, {}):
                raise SystemExit(f"{sym} {day} present in more than one run dir")
            d = json.load(open(os.path.join(ddir, fn), encoding="utf-8"))
            last = SESSION_LAST_EXT if day in SUNEUNG_DAYS else SESSION_LAST
            rows = [r for r in d["rows"] if SESSION_FIRST <= r[0][11:16] <= last]
            if not rows:
                continue
            if len({r[0] for r in rows}) != len(rows):
                raise SystemExit(f"{sym} {day}: duplicate timestamps inside day rows")
            out.setdefault(sym, {})[day] = make_day([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows], [r[4] for r in rows], [r[5] for r in rows], d["cls"], bool(d.get("incomplete")), len(d.get("content_conflicts", [])), int(d.get("duplicate_ts", 0)))
    return out


def daily_prev_feats(dd, day):
    days = dd["days"]
    if day not in days:
        return None
    i = days.index(day)
    if i < DAILY_WARM:
        return None
    C = dd["Ca"]
    lr = np.diff(np.log(C[i - 21:i]))
    return [C[i - 1] / C[i - 2] - 1, C[i - 1] / C[i - 6] - 1, C[i - 1] / C[i - 21] - 1, float(np.std(lr, ddof=1))]


def session_open(b):
    return LATE_OPEN if b["ts"][0][:10] in LATE_OPEN_DAYS else SESSION_FIRST


def window_ok(b, k):
    t = b["tod"][k]
    if k < S_HIST or b["tod"][0] != session_open(b):
        return False
    return all(b["idx"].get(tod_add(t, -i)) == k - i for i in range(1, k + 1))


def s_features(b, k, prev, sym_id):
    O, H, L, C, V = b["O"], b["H"], b["L"], b["C"], b["V"]
    c = C[k]
    hi, lo = H[:k + 1].max(), L[:k + 1].min()
    vm = V[:k + 1].mean()
    lr = np.diff(np.log(C[k - 15:k + 1]))
    return [c / C[k - 1] - 1, c / C[k - 5] - 1, c / C[k - 15] - 1, c / C[k - 30] - 1, c / C[k - 60] - 1, c / O[0] - 1, (c - lo) / (hi - lo) if hi > lo else 0.5, V[k - 4:k + 1].mean() / vm if vm > 0 else 0.0, float(np.std(lr, ddof=1)), float(S_GRID.index(b["tod"][k])), float(date.fromisoformat(b["ts"][0][:10]).weekday())] + prev + [float(sym_id)]


def s_outcome(b, k):
    t = b["tod"][k]
    e_tod, x_tod = tod_add(t, S_ENTRY), tod_add(t, S_EXIT)
    base = dict(entry_tod=e_tod, exit_tod=x_tod, pb=np.nan, ps=np.nan, y=np.nan, entry_ts=None, exit_ts=None, exit_reason="")
    e = b["idx"].get(e_tod)
    if e is None:
        return dict(base, flag=FLAG["entry_missing"], exit_reason="entry_bar_missing")
    pb = b["O"][e] + TICK
    base.update(pb=pb, entry_ts=b["ts"][e])
    if b["V"][e] <= 0:
        return dict(base, flag=FLAG["unfilled"], exit_reason="entry_zero_volume")
    cur = x_tod
    while cur <= S_LAST_EXIT:
        j = b["idx"].get(cur)
        if j is None:
            return dict(base, flag=FLAG["exit_unknown"], exit_reason=("exit_bar_missing" if cur == x_tod else "exit_bar_missing_after_zero_volume") + "_" + cur)
        if b["V"][j] > 0:
            ps = b["O"][j] - TICK
            return dict(base, flag=FLAG["ok"] if cur == x_tod else FLAG["delayed"], ps=ps, y=float(net_ret(pb, ps)), exit_ts=b["ts"][j], exit_reason="ok" if cur == x_tod else "exit_zero_volume")
        cur = tod_add(cur, 1)
    return dict(base, flag=FLAG["unexited"], exit_reason="exit_zero_volume_until_" + S_LAST_EXIT)


def s_table(run, daily):
    rows = []
    for sym_id, sym in enumerate(SYMS):
        for day, b in sorted(run.get(sym, {}).items()):
            prev = daily_prev_feats(daily[sym], day)
            if prev is None:
                continue
            dayinfo = dict(day_cls=b["cls"], day_incomplete=b["incomplete"], day_conflicts=b["conflicts"], day_dups=b["dups"])
            for tod in S_GRID:
                k = b["idx"].get(tod)
                if day in SUNEUNG_DAYS:
                    rows.append(dict(sym=sym, day=day, t=f"{day}T{tod}:00", feats=[np.nan] * len(S_FEATS), brk60=np.nan, flag=FLAG["special_session"], entry_tod=tod_add(tod, S_ENTRY), exit_tod=tod_add(tod, S_EXIT), pb=np.nan, ps=np.nan, y=np.nan, entry_ts=None, exit_ts=None, exit_reason="suneung_session_10:00-16:30_excluded", **dayinfo))
                    continue
                if k is None:
                    rows.append(dict(sym=sym, day=day, t=f"{day}T{tod}:00", feats=[np.nan] * len(S_FEATS), brk60=np.nan, flag=FLAG["decision_missing"], entry_tod=tod_add(tod, S_ENTRY), exit_tod=tod_add(tod, S_EXIT), pb=np.nan, ps=np.nan, y=np.nan, entry_ts=None, exit_ts=None, exit_reason="decision_bar_missing", **dayinfo))
                    continue
                if not window_ok(b, k):
                    rows.append(dict(sym=sym, day=day, t=b["ts"][k], feats=[np.nan] * len(S_FEATS), brk60=np.nan, flag=FLAG["feature_gap"], entry_tod=tod_add(tod, S_ENTRY), exit_tod=tod_add(tod, S_EXIT), pb=np.nan, ps=np.nan, y=np.nan, entry_ts=None, exit_ts=None, exit_reason="feature_window_gap", **dayinfo))
                    continue
                rows.append(dict(sym=sym, day=day, t=b["ts"][k], feats=s_features(b, k, prev, sym_id), brk60=float(b["C"][k] / b["H"][k - S_HIST:k].max() - 1), **s_outcome(b, k), **dayinfo))
    return rows


def dip_first_flags(C, lookback=20, thr=-0.04):
    r = np.full(len(C), np.nan)
    r[1:] = C[1:] / C[:-1] - 1
    dip = r <= thr
    out = np.zeros(len(C), bool)
    for i in np.flatnonzero(dip):
        out[i] = not dip[max(0, i - lookback):i].any()
    return out


def l_features(dd, i, sym_id):
    C, V = dd["Ca"], dd["Va"]
    c = C[i]
    lr = np.diff(np.log(C[i - 20:i + 1]))
    return [c / C[i - 1] - 1, c / C[i - 5] - 1, c / C[i - 20] - 1, c / C[i - 60] - 1, c / C[i - 120] - 1, c / C[i - 250] - 1, c / C[i - 59:i + 1].max() - 1, c / C[i - 249:i + 1].max() - 1, float(np.std(lr, ddof=1)), V[i - 19:i + 1].mean() / V[i - 119:i + 1].mean() if V[i - 119:i + 1].mean() > 0 else 0.0, c / C[i - 19:i + 1].mean(), c / C[i - 59:i + 1].mean(), c / C[i - 119:i + 1].mean(), float(int(dd["days"][i][5:7])), float(sym_id)]


def l_table(daily):
    rows = []
    for sym_id, sym in enumerate(SYMS):
        dd = daily[sym]
        n = len(dd["days"])
        dip = dip_first_flags(dd["Ca"])
        for i in range(L_WARM, n - 1):
            feats = l_features(dd, i, sym_id)
            pb = dd["Or"][i + 1] + TICK
            if i + 1 + L_H <= n - 1:
                ps = dd["Or"][i + 1 + L_H] - TICK
                rows.append(dict(sym=sym, day=dd["days"][i], feats=feats, dip_first=bool(dip[i]), pb=pb, ps=ps, y=float(net_ret(pb, ps)), entry_day=dd["days"][i + 1], exit_day=dd["days"][i + 1 + L_H], pending=False))
            else:
                rows.append(dict(sym=sym, day=dd["days"][i], feats=feats, dip_first=bool(dip[i]), pb=pb, ps=np.nan, y=np.nan, entry_day=dd["days"][i + 1], exit_day=None, pending=True))
    return rows


def s_eligible(r):
    return r["flag"] in (FLAG["ok"], FLAG["delayed"]) and np.isfinite(r["y"]) and all(np.isfinite(v) for v in r["feats"]) and r["day_conflicts"] == 0 and not r["day_incomplete"]


def l_eligible(r):
    return (not r["pending"]) and np.isfinite(r["y"]) and all(np.isfinite(v) for v in r["feats"])


def save(rows_s, rows_l, meta, out_path):
    def col(rows, key, dtype=object):
        return np.array([r[key] for r in rows], dtype=dtype)
    np.savez(out_path, s_eligible=np.array([s_eligible(r) for r in rows_s], bool), l_eligible=np.array([l_eligible(r) for r in rows_l], bool), s_sym=col(rows_s, "sym"), s_day=col(rows_s, "day"), s_t=col(rows_s, "t"), s_X=np.array([r["feats"] for r in rows_s], np.float64).reshape(len(rows_s), len(S_FEATS)), s_y=col(rows_s, "y", float), s_flag=col(rows_s, "flag", np.int8), s_pb=col(rows_s, "pb", float), s_ps=col(rows_s, "ps", float), s_entry_tod=col(rows_s, "entry_tod"), s_exit_tod=col(rows_s, "exit_tod"), s_entry_ts=col(rows_s, "entry_ts"), s_exit_ts=col(rows_s, "exit_ts"), s_exit_reason=col(rows_s, "exit_reason"), s_day_cls=col(rows_s, "day_cls"), s_day_incomplete=col(rows_s, "day_incomplete", bool), s_day_conflicts=col(rows_s, "day_conflicts", np.int32), s_day_dups=col(rows_s, "day_dups", np.int32), s_brk60=col(rows_s, "brk60", float), l_dip_first=col(rows_l, "dip_first", bool), l_sym=col(rows_l, "sym"), l_day=col(rows_l, "day"), l_X=np.array([r["feats"] for r in rows_l], np.float64), l_y=col(rows_l, "y", float), l_pb=col(rows_l, "pb", float), l_ps=col(rows_l, "ps", float), l_entry_day=col(rows_l, "entry_day"), l_exit_day=col(rows_l, "exit_day"), l_pending=col(rows_l, "pending", bool), s_feats=np.array(S_FEATS), l_feats=np.array(L_FEATS), flags=np.array(json.dumps(FLAG)), meta=np.array(json.dumps(meta, ensure_ascii=False)))


def main(run_dirs):
    daily = load_daily()
    run = load_run(run_dirs) if run_dirs else {}
    rows_s, rows_l = (s_table(run, daily) if run_dirs else []), l_table(daily)
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = "_".join(os.path.basename(d.rstrip("\\/")).replace("1m_backfill_", "") for d in run_dirs) if run_dirs else "Lonly"
    out_path = os.path.join(OUT_DIR, f"mh_tables_{tag}.npz")
    if os.path.exists(out_path):
        raise SystemExit(f"{out_path} exists; refusing to overwrite")
    flags = {k: int(sum(1 for r in rows_s if r["flag"] == v)) for k, v in FLAG.items()}
    meta = dict(script_sha256=sha(os.path.abspath(__file__)), mode="L-only" if not run_dirs else "S+L", run_dirs=[os.path.relpath(d, HERE) for d in run_dirs], run_summary_sha256=[sha(os.path.join(d, "summary.json")) for d in run_dirs], adj_sha256=sha(ADJ), raw_sha256=sha(RAW), s_rows=len(rows_s), s_eligible=int(sum(s_eligible(r) for r in rows_s)), l_eligible=int(sum(l_eligible(r) for r in rows_l)), eligibility=dict(S="flag in (ok, delayed) and finite label and all 16 features finite and day has no content conflicts and day not incomplete", L="not pending and finite label and all 15 features finite"), s_flags=flags, s_days={s: len(run.get(s, {})) for s in SYMS}, s_day_classes={s: dict(zip(*np.unique([b["cls"] for b in run.get(s, {}).values()], return_counts=True))) if run.get(s) else {} for s in SYMS}, l_rows=len(rows_l), l_pending=int(sum(1 for r in rows_l if r["pending"])), grid=S_GRID[:3] + ["..."] + S_GRID[-2:], s_entry_exit_minutes=[S_ENTRY, S_EXIT], s_last_exit=S_LAST_EXIT, s_hist_minutes=S_HIST, l_h=L_H, l_warm=L_WARM, tick=TICK, fee=FEE, s_feats=S_FEATS, l_feats=L_FEATS, flags=FLAG)
    meta["s_day_classes"] = {s: {str(k): int(v) for k, v in c.items()} for s, c in meta["s_day_classes"].items()}
    save(rows_s, rows_l, meta, out_path)
    print("META", json.dumps(meta, ensure_ascii=False))
    print("saved", out_path, "sha", sha(out_path))


def synth_day(day="2026-01-05", n=390, start=(9, 1), slope=10.0, base=80000.0, drop=()):
    h, m = start
    ts = []
    for _ in range(n):
        ts.append(f"{day}T{h:02d}:{m:02d}:00")
        m += 1
        if m == 60:
            h, m = h + 1, 0
    C = base + slope * np.arange(n)
    keep = [i for i, t in enumerate(ts) if t[11:16] not in drop]
    return make_day([ts[i] for i in keep], (C - 2)[keep], (C + 3)[keep], (C - 4)[keep], C[keep], np.ones(len(keep)))


def selftest():
    b = synth_day()
    k = b["idx"]["10:05"]
    o = s_outcome(b, k)
    e, x = b["idx"]["10:07"], b["idx"]["10:22"]
    pb, ps = b["O"][e] + 5, b["O"][x] - 5
    exp = (ps - np.floor(ps * FEE) - pb - np.floor(pb * FEE)) / (pb + np.floor(pb * FEE))
    assert o["flag"] == 0 and abs(o["y"] - exp) < 1e-12 and o["entry_ts"].endswith("10:07:00") and o["exit_ts"].endswith("10:22:00") and o["exit_reason"] == "ok", o
    b2 = synth_day()
    b2["V"][e] = 0
    o2 = s_outcome(b2, k)
    assert o2["flag"] == FLAG["unfilled"] and o2["entry_ts"].endswith("10:07:00") and o2["pb"] == pb, o2
    b3 = synth_day()
    b3["V"][x] = 0
    b3["V"][x + 1] = 0
    o3 = s_outcome(b3, k)
    assert o3["flag"] == FLAG["delayed"] and o3["exit_ts"].endswith("10:24:00") and o3["exit_reason"] == "exit_zero_volume", o3
    bm = synth_day(drop=("10:06",))
    om = s_outcome(bm, bm["idx"]["10:05"])
    assert om["flag"] == 0 and om["entry_ts"].endswith("10:07:00") and om["exit_ts"].endswith("10:22:00"), om
    bx = synth_day(drop=("10:22",))
    ox = s_outcome(bx, bx["idx"]["10:05"])
    assert ox["flag"] == FLAG["exit_unknown"] and ox["entry_ts"].endswith("10:07:00") and ox["pb"] == pb and ox["exit_reason"] == "exit_bar_missing_10:22" and np.isnan(ox["y"]), ox
    bz = synth_day(drop=("10:23",))
    bz["V"][bz["idx"]["10:22"]] = 0
    oz = s_outcome(bz, bz["idx"]["10:05"])
    assert oz["flag"] == FLAG["exit_unknown"] and oz["exit_reason"] == "exit_bar_missing_after_zero_volume_10:23", oz
    be = synth_day(drop=("10:07",))
    oe = s_outcome(be, be["idx"]["10:05"])
    assert oe["flag"] == FLAG["entry_missing"] and oe["entry_ts"] is None and oe["entry_tod"] == "10:07", oe
    bt = synth_day(n=75)
    assert bt["tod"][-1] == "10:15"
    ot = s_outcome(bt, bt["idx"]["10:05"])
    assert ot["flag"] == FLAG["exit_unknown"] and ot["entry_ts"].endswith("10:07:00") and ot["pb"] == pb and ot["exit_reason"] == "exit_bar_missing_10:22", ot
    b4 = synth_day()
    k4 = b4["idx"]["15:00"]
    x4 = b4["idx"]["15:17"]
    b4["V"][x4:x4 + 3] = 0
    o4 = s_outcome(b4, k4)
    assert o4["flag"] == FLAG["unexited"] and np.isnan(o4["y"]) and o4["exit_reason"] == "exit_zero_volume_until_15:19", o4
    bg = synth_day(drop=("09:30",))
    assert not window_ok(bg, bg["idx"]["10:05"]) and not window_ok(bg, bg["idx"]["10:35"]) and not window_ok(bg, bg["idx"]["15:00"])
    bo = synth_day(drop=("09:01",))
    assert not window_ok(bo, bo["idx"]["10:05"])
    assert window_ok(b, b["idx"]["15:00"])
    prev = [0.01, 0.02, 0.03, 0.004]
    f1 = s_features(b, k, prev, 0)
    b5 = synth_day()
    b5["C"][k + 1:] *= 1.5
    b5["H"][k + 1:] *= 1.5
    b5["V"][k + 1:] *= 7
    assert s_features(b5, k, prev, 0) == f1 and len(f1) == len(S_FEATS) == 16
    assert f1[S_FEATS.index("grid_idx")] == 0.0 and abs(f1[S_FEATS.index("r60")] - (b["C"][k] / b["C"][b["idx"]["09:05"]] - 1)) < 1e-12
    late = synth_day(day="2024-11-14", n=330, start=(10, 1))
    late["cls"] = "late_open_special"
    ks = [kk for kk, tod in enumerate(late["tod"]) if tod in S_GRID and window_ok(late, kk)]
    assert late["tod"][ks[0]] == "11:05", late["tod"][ks[0]]
    late_n = synth_day(day="2024-11-15", n=330, start=(10, 1))
    late_n["cls"] = "late_open_special"
    assert not any(window_ok(late_n, kk) for kk, tod in enumerate(late_n["tod"]) if tod in S_GRID)
    full_on_late_day = synth_day(day="2024-11-14")
    assert not window_ok(full_on_late_day, full_on_late_day["idx"]["10:05"]) and not window_ok(full_on_late_day, full_on_late_day["idx"]["15:00"])
    first_day = synth_day(day="2025-01-02", n=330, start=(10, 1))
    assert window_ok(first_day, first_day["idx"]["11:05"]) and not window_ok(first_day, first_day["idx"]["11:00"])
    n = 400
    days = [f"2020-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(n)]
    Cr = 10000 + 10.0 * np.arange(n)
    dd = dict(days=days, Ca=Cr * 0.9, Oa=(Cr - 3) * 0.9, Va=np.ones(n), Or=Cr - 3, Cr=Cr)
    rows = l_table({"069500": dd, "229200": dd})
    r0 = [r for r in rows if r["sym"] == "069500"][0]
    i = L_WARM
    pbl, psl = dd["Or"][i + 1] + 5, dd["Or"][i + 61] - 5
    assert r0["day"] == days[i] and r0["exit_day"] == days[i + 61] and abs(r0["y"] - float(net_ret(pbl, psl))) < 1e-12 and len(r0["feats"]) == len(L_FEATS) == 15, r0
    pend = [r for r in rows if r["sym"] == "069500" and r["pending"]]
    assert len(pend) == 60 and pend[0]["day"] == days[n - 1 - 60], len(pend)
    dd2 = dict(dd, Ca=dd["Ca"].copy())
    dd2["Ca"][i + 1:] *= 2
    assert l_features(dd2, i, 0) == l_features(dd, i, 0)
    Cd = np.full(60, 100.0)
    Cd[10] = 95.0
    Cd[11:] = 95.0
    Cd[15] = 90.0
    Cd[16:] = 90.0
    Cd[40] = 85.0
    Cd[41:] = 85.0
    df = dip_first_flags(Cd)
    assert list(np.flatnonzero(df)) == [10, 40], list(np.flatnonzero(df))
    rb = s_table({"069500": {"2026-01-05": b}}, {"069500": dict(days=[f"2025-12-{d:02d}" for d in range(1, 29)] + ["2026-01-05"], Ca=np.linspace(100, 110, 29), Oa=np.linspace(100, 110, 29), Va=np.ones(29), Or=np.linspace(100, 110, 29), Cr=np.linspace(100, 110, 29)), "229200": dict(days=[], Ca=np.array([]), Oa=np.array([]), Va=np.array([]), Or=np.array([]), Cr=np.array([]))})
    r = [x for x in rb if x["t"].endswith("10:05:00")][0]
    assert abs(r["brk60"] - (b["C"][k] / b["H"][k - 60:k].max() - 1)) < 1e-12 and len(rb) == 60, (r["brk60"], len(rb))
    daily_fake = {"069500": dict(days=[f"2025-12-{d:02d}" for d in range(1, 29)] + ["2026-01-05"], Ca=np.linspace(100, 110, 29), Oa=np.linspace(100, 110, 29), Va=np.ones(29), Or=np.linspace(100, 110, 29), Cr=np.linspace(100, 110, 29)), "229200": dict(days=[], Ca=np.array([]), Oa=np.array([]), Va=np.array([]), Or=np.array([]), Cr=np.array([]))}
    bd = synth_day(drop=("10:05",))
    rd = s_table({"069500": {"2026-01-05": bd}}, daily_fake)
    assert len(rd) == 60 and [x for x in rd if x["t"].endswith("10:05:00")][0]["flag"] == FLAG["decision_missing"], len(rd)
    rg = s_table({"069500": {"2026-01-05": synth_day(drop=("09:01",))}}, daily_fake)
    assert len(rg) == 60 and all(x["flag"] == FLAG["feature_gap"] for x in rg)
    daily_su = {"069500": dict(days=[f"2024-10-{d:02d}" for d in range(1, 29)] + ["2024-11-14"], Ca=np.linspace(100, 110, 29), Oa=np.linspace(100, 110, 29), Va=np.ones(29), Or=np.linspace(100, 110, 29), Cr=np.linspace(100, 110, 29)), "229200": dict(days=[], Ca=np.array([]), Oa=np.array([]), Va=np.array([]), Or=np.array([]), Cr=np.array([]))}
    su = synth_day(day="2024-11-14", n=390, start=(10, 1))
    assert su["tod"][-1] == "16:30"
    rs = s_table({"069500": {"2024-11-14": su}}, daily_su)
    assert len(rs) == 60 and all(x["flag"] == FLAG["special_session"] for x in rs) and sum(s_eligible(x) for x in rs) == 0
    assert sum(s_eligible(x) for x in rb) == 60 and sum(s_eligible(x) for x in rd) == 0 and sum(s_eligible(x) for x in rg) == 0
    assert sum(1 for x in rd if x["flag"] == FLAG["feature_gap"]) == 59
    bc = synth_day()
    bc["conflicts"] = 1
    rc = s_table({"069500": {"2026-01-05": bc}}, daily_fake)
    assert sum(s_eligible(x) for x in rc) == 0 and all(x["flag"] == FLAG["ok"] for x in rc[:50])
    assert sum(l_eligible(x) for x in rows) == len(rows) - 2 * 60
    import tempfile
    tmp = os.path.join(tempfile.mkdtemp(), "t.npz")
    save([], rows, dict(test=True), tmp)
    z = np.load(tmp, allow_pickle=True)
    assert z["s_X"].shape == (0, len(S_FEATS)) and len(z["s_y"]) == 0 and len(z["l_y"]) == len(rows) and z["l_eligible"].sum() == len(rows) - 120, (z["s_X"].shape, len(z["l_y"]))
    print("selftest ok", dict(s_feats=len(S_FEATS), l_feats=len(L_FEATS), grid=len(S_GRID), flags=FLAG, sha=sha(os.path.abspath(__file__))))


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    elif sys.argv[1:] == ["--l-only"]:
        main([])
    elif len(sys.argv) >= 2:
        main(sys.argv[1:])
    else:
        print("usage: mh_data.py <run_dir> [<run_dir> ...] | --l-only | selftest")
