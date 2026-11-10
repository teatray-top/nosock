import glob
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get("NOSOCK_RUNS", os.path.join(HERE, "runs"))
ISSUES = os.path.join(RUNS, "issues")
DESK = os.environ.get("NOSOCK_SCENARIO_LOG", os.path.join(RUNS, "scenario_desk.jsonl"))
OUT = os.path.join(RUNS, "scorecard.json")
KST = ZoneInfo("Asia/Seoul")
NY = ZoneInfo("America/New_York")
KEEP = 160


def parse_ts(s):
    s = s.strip().replace("Z", "+00:00")
    if len(s) >= 5 and s[-5] in "+-" and s[-4:].isdigit():
        s = s[:-2] + ":" + s[-2:]
    return datetime.fromisoformat(s)


def ymd(day):
    return tuple(map(int, day.split("-")))


def close_at(market, day):
    y, m, d = ymd(day)
    return datetime(y, m, d, 15, 30, tzinfo=KST) if market == "KR" else datetime(y, m, d, 16, 0, tzinfo=NY)


def open_at(market, day):
    y, m, d = ymd(day)
    return datetime(y, m, d, 9, 0, tzinfo=KST) if market == "KR" else datetime(y, m, d, 9, 30, tzinfo=NY)


def series_from_inst(inst, now):
    out = {}
    for c, ins in inst.items():
        n = len(ins["days"])
        ks = [k for k in range(max(0, n - KEEP), n) if close_at(ins["market"], ins["days"][k]) <= now]
        out[c] = dict(market=ins["market"], days=[ins["days"][k] for k in ks],
                      po=[float(ins["O"][k] / ins["fxv"][k]) for k in ks], pc=[float(ins["C"][k] / ins["fxv"][k]) for k in ks],
                      ko=[float(ins["O"][k]) for k in ks], kc=[float(ins["C"][k]) for k in ks], cost=[float(ins["cost"](k, ins["C"][k])) for k in ks])
    return out


def load_calls(issue_dir=ISSUES):
    out = []
    for f in sorted(glob.glob(os.path.join(issue_dir, "*.json"))):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if d.get("kind") != "issues":
            continue
        p = d.get("payload") or {}
        at = parse_ts(d["received_at"])
        author = p.get("author") or d.get("source") or "?"
        for i, it in enumerate(p.get("issues") or []):
            for c, a in (it.get("affected") or {}).items():
                if not isinstance(a, dict):
                    continue
                try:
                    hz = int(float(a.get("horizon_days") or 0))
                except (TypeError, ValueError):
                    hz = 0
                out.append(dict(batch=d.get("id"), at=at, author=author, title=(it.get("title") or "")[:90], urgency=it.get("urgency"),
                                event_id=it.get("event_id"), signal=it.get("signal"), code=c, direction=a.get("direction"), magnitude=a.get("magnitude_pct"), horizon=hz))
    return out


def score_call(call, s):
    x = s.get(call["code"])
    if call["direction"] not in ("up", "down") or not x or call["horizon"] <= 0:
        return None
    after = [k for k, d in enumerate(x["days"]) if open_at(x["market"], d) > call["at"]]
    if not after:
        return dict(entry_day=None, end_day=None, matured=False, days=0, ret=0.0, net=0.0, hit=None)
    k0 = after[0]
    k1 = k0 + call["horizon"] - 1
    last = len(x["days"]) - 1
    ke = min(k1, last)
    sign = 1.0 if call["direction"] == "up" else -1.0
    r = x["pc"][ke] / x["po"][k0] - 1.0
    net = sign * (x["kc"][ke] / x["ko"][k0] - 1.0) - x["cost"][k0] - x["cost"][ke]
    return dict(entry_day=x["days"][k0], end_day=x["days"][ke], matured=k1 <= last, days=ke - k0 + 1, ret=r, net=net, hit=sign * r > 0)


def desk_rows(path=DESK):
    rows = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("kind") == "scenario" and r.get("opt_ok"):
                rows.append(dict(at=parse_ts(r["ts"]), tilt=r.get("tilt") or {}, trigger=r.get("trigger"), probs=r.get("probabilities") or [], scen=r.get("scenarios") or [], exp=r.get("exp_pct"), base=r.get("base_rates") or {}))
    return sorted(rows, key=lambda r: r["at"])


def desk_calls(rows, min_abs=0.005):
    out = []
    for r in rows:
        if r["trigger"] != "morning" or not r["scen"] or len(r["probs"]) != len(r["scen"]):
            continue
        exp = r.get("exp")
        codes = sorted(exp) if exp else sorted({c for sc in r["scen"] for c in (sc.get("returns_pct") or {})})
        for c in codes:
            e = float(exp[c]) / 100.0 if exp else sum(p * float((sc.get("returns_pct") or {}).get(c, 0.0)) for p, sc in zip(r["probs"], r["scen"])) / 100.0
            if abs(e) < min_abs:
                continue
            b = (r.get("base") or {}).get(c)
            out.append(dict(batch=r["at"].isoformat(timespec="minutes"), at=r["at"], author="desk", title="시나리오 기대수익", urgency=None, event_id=None, signal=None,
                            code=c, direction="up" if e > 0 else "down", magnitude=round(100 * e, 2), horizon=20, base=b["mean"] if b else None))
    return out


def tilt_on(rows, when, code):
    w = 0.0
    for r in rows:
        if r["at"] <= when:
            w = r["tilt"].get(code, 0.0)
    return w


def desk_pnl(rows, s):
    days = {}
    codes = sorted({c for r in rows for c in r["tilt"]})
    for c in codes:
        x = s.get(c)
        if not x:
            continue
        prev = 0.0
        for k in range(1, len(x["days"])):
            w = tilt_on(rows, open_at(x["market"], x["days"][k]), c)
            if w == 0.0 and prev == 0.0:
                continue
            g = prev * (x["ko"][k] / x["kc"][k - 1] - 1.0) + w * (x["kc"][k] / x["ko"][k] - 1.0)
            cost = abs(w - prev) * x["cost"][k]
            dk = close_at(x["market"], x["days"][k]).astimezone(KST).strftime("%Y-%m-%d")
            e = days.setdefault(dk, dict(gross=0.0, cost=0.0, codes={}))
            e["gross"] += g
            e["cost"] += cost
            e["codes"][c] = round(g - cost, 6)
            prev = w
    out = [dict(day=d, gross=v["gross"], cost=v["cost"], net=v["gross"] - v["cost"], codes=v["codes"]) for d, v in sorted(days.items())]
    tot = dict(gross=sum(v["gross"] for v in out), cost=sum(v["cost"] for v in out))
    tot["net"] = tot["gross"] - tot["cost"]
    return out, tot


def summarize(calls, s):
    items, by = [], {}
    for c in calls:
        a = by.setdefault(c["author"], dict(all=0, calls=0, matured=0, hits=0, net_sum=0.0, open=0, open_hits=0, open_net_sum=0.0, mag_n=0, mag_err_sum=0.0, mag_abs_sum=0.0, base_n=0, base_abs_sum=0.0, base_mag_abs_sum=0.0))
        a["all"] += 1
        sc = score_call(c, s)
        if sc is None:
            continue
        items.append(dict(c, at=c["at"].isoformat(timespec="minutes"), **sc))
        a["calls"] += 1
        if sc["matured"]:
            a["matured"] += 1
            a["hits"] += bool(sc["hit"])
            a["net_sum"] += sc["net"]
            if c.get("magnitude") is not None:
                err = sc["ret"] - float(c["magnitude"]) / 100.0
                a["mag_n"] += 1
                a["mag_err_sum"] += err
                a["mag_abs_sum"] += abs(err)
                if c.get("base") is not None:
                    a["base_n"] += 1
                    a["base_abs_sum"] += abs(sc["ret"] - float(c["base"]) / 100.0)
                    a["base_mag_abs_sum"] += abs(err)
        elif sc["days"] > 0:
            a["open"] += 1
            a["open_hits"] += bool(sc["hit"])
            a["open_net_sum"] += sc["net"]
    for a in by.values():
        a["hit_rate"] = a["hits"] / a["matured"] if a["matured"] else None
        a["net_mean"] = a["net_sum"] / a["matured"] if a["matured"] else None
        a["open_hit_rate"] = a["open_hits"] / a["open"] if a["open"] else None
        a["open_net_mean"] = a["open_net_sum"] / a["open"] if a["open"] else None
        a["mag_bias"] = a["mag_err_sum"] / a["mag_n"] if a["mag_n"] else None
        a["mag_mae"] = a["mag_abs_sum"] / a["mag_n"] if a["mag_n"] else None
        a["base_mae"] = a["base_abs_sum"] / a["base_n"] if a["base_n"] else None
        a["vs_base_mae"] = a["base_mag_abs_sum"] / a["base_n"] if a["base_n"] else None
    return items, by


def build(s, now, issue_dir=ISSUES, desk_path=DESK):
    rows = desk_rows(desk_path)
    calls = load_calls(issue_dir) + desk_calls(rows)
    items, by = summarize(calls, s)
    daily, tot = desk_pnl(rows, s)
    last = {c: x["days"][-1] for c, x in s.items() if x["days"]}
    return dict(asof=now.isoformat(timespec="seconds"), last_close=dict(KR=max((d for c, d in last.items() if s[c]["market"] == "KR"), default=None), US=max((d for c, d in last.items() if s[c]["market"] == "US"), default=None)),
                calls=dict(total=len(calls), scored=len(items), by_author=by, items=items),
                desk=dict(rows=len(rows), first=rows[0]["at"].isoformat(timespec="minutes") if rows else None, total=tot, daily=daily))


def pct(v):
    return "-" if v is None else f"{100 * v:+.2f}%"


def rate(v):
    return "-" if v is None else f"{100 * v:.0f}%"


def report(sc):
    lines = [f"점수판 {sc['asof'][:16]} (마지막 종가 국내 {sc['last_close']['KR']}, 미국 {sc['last_close']['US']})"]
    d = sc["desk"]
    lines.append(f"시나리오 데스크 조정을 그대로 따랐다면 (비용 후): {pct(d['total']['net'])} (조정 수익 {pct(d['total']['gross'])}, 비용 {pct(-d['total']['cost'])}), 데스크 결과 {d['rows']}건, {len(d['daily'])}일")
    for day in d["daily"][-10:]:
        lines.append(f"  {day['day']}: {pct(day['net'])}")
    c = sc["calls"]
    lines.append(f"예측 {c['total']}건 중 채점 가능 {c['scored']}건 (방향이 up/down이고 기간이 있는 것)")
    for au, a in sorted(c["by_author"].items()):
        lines.append(f"  {au} {a['all']}건 중 채점 {a['calls']}건: 기간 끝난 {a['matured']}건 방향 적중 {rate(a['hit_rate'])}, 그 방향으로 거래했다면 비용 후 평균 {pct(a['net_mean'])} / 진행 중 {a['open']}건 지금까지 방향 적중 {rate(a['open_hit_rate'])}, 비용 후 평균 {pct(a['open_net_mean'])}")
        if a["mag_n"]:
            lines.append(f"    예상 크기 대비 실제: 평균 차이 {pct(a['mag_bias'])}, 평균 절대 오차 {100 * a['mag_mae']:.2f}%p ({a['mag_n']}건)")
        if a.get("base_n"):
            lines.append(f"    같은 {a['base_n']}건을 과거 평균으로 예측했다면 평균 절대 오차 {100 * a['base_mae']:.2f}%p (이 예측 {100 * a['vs_base_mae']:.2f}%p)")
    return "\n".join(lines)


def live_series(now):
    import alloc_shadow
    import live_nosock as L
    t = alloc_shadow.read_only_client()
    extra = sorted(set(L.listfile("adopt.txt")) | set(L.listfile("candidates_extra.txt")))
    inst, groups, fx_now, fetched = L.build_market(t, extra)
    return series_from_inst(inst, now)


def main():
    now = datetime.now(KST)
    sc = build(live_series(now), now)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(sc, f, ensure_ascii=False, default=str)
    os.replace(tmp, OUT)
    print(report(sc))


if __name__ == "__main__":
    sys.exit(main())
