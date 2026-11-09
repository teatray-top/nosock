import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import learn_inverse as LI
import compare_trend_v2 as CT
import harvest_v2 as H2

ZIG = 0.15
BEFORE, AFTER = 20, 60
NAMES = ("hold", "ribbon", "A_long_tp10_sl10", "A_long_tp10_sl0", "B_long_gated_tp10_sl10", "C_with_inverse_tp10_sl10", "D_trend_switch_tp10_sl10")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def run_series(pol, px, cost, start, n, pred, bull_at, bear_at):
    cash, vault = H2.CAPITAL, 0.0
    leg, q, basis, entry = None, 0.0, 0.0, None
    series = {start: H2.CAPITAL}
    k = pol["kind"]
    for i in range(start, n - 1):
        j = i + 1
        act = None
        if k == "hold":
            if leg is None and i == start:
                act = ("buy", H2.LONG)
        elif k == "ribbon":
            if leg is None and bull_at(i):
                act = ("buy", H2.LONG)
            elif leg is not None and not bull_at(i):
                act = ("sell", None)
        else:
            if leg is not None:
                c_now = px[leg]["C"][i]
                if c_now >= entry * (1 + pol["tp"]) or (pol["sl"] and c_now <= entry * (1 - pol["sl"])):
                    act = ("sell", None)
            else:
                pl, ps = pred(i)
                tgt = H2.choose(k, pl, ps, bull_at(i), bear_at(i))
                if tgt:
                    act = ("buy", tgt)
        if act and act[0] == "sell" and leg is not None:
            c = cost(leg, j)
            proceeds = q * px[leg]["O"][j] * (1 - c)
            profit = proceeds - basis
            tv = H2.VAULT_SHARE * profit if profit > 0 else 0.0
            vault += tv
            cash += proceeds - tv
            leg, q, basis, entry = None, 0.0, 0.0, None
        elif act and act[0] == "buy" and leg is None and cash > 0:
            leg = act[1]
            c = cost(leg, j)
            q = cash * (1 - c) / px[leg]["O"][j]
            basis, entry, cash = cash, cash / q, 0.0
        series[j] = cash + (q * px[leg]["C"][j] if leg else 0.0) + vault
    final = cash + (q * px[leg]["C"][n - 1] * (1 - cost(leg, n - 1)) if leg else 0.0) + vault
    return series, final


def zigzag(C, thr):
    pts, mode = [], None
    hi_i = lo_i = 0
    for i in range(1, len(C)):
        if C[i] > C[hi_i]:
            hi_i = i
        if C[i] < C[lo_i]:
            lo_i = i
        if mode != "down" and C[i] <= C[hi_i] * (1 - thr):
            pts.append(("peak", hi_i))
            mode, lo_i = "down", i
        elif mode != "up" and C[i] >= C[lo_i] * (1 + thr):
            pts.append(("trough", lo_i))
            mode, hi_i = "up", i
    return pts


def main():
    raw = json.load(open(os.path.join(os.path.dirname(HERE), "data", "kr_etf", "kodex_daily_raw.json"), encoding="utf-8"))[H2.LONG]
    adj = json.load(open(CT.ADJ, encoding="utf-8"))[H2.LONG]
    iv = json.load(open(H2.INV_DAILY, encoding="utf-8"))[H2.INV]
    imap = {d: (float(o), float(c)) for d, o, c in zip(iv["days"], iv["O"], iv["C"])}
    days = [d for d in raw["days"] if d <= H2.REAL_TO]
    A = [float(x) for x in adj["C"][:len(days)]]
    e = {m: CT.ema(A, m) for m in (8, 21, 50, 200)}
    start = days.index(H2.REAL_FROM) - 1
    keep = list(range(start, len(days)))
    sub = [days[i] for i in keep]
    px = {H2.LONG: dict(O=[float(raw["O"][i]) for i in keep], C=[float(raw["C"][i]) for i in keep]), H2.INV: dict(O=[imap[days[i]][0] for i in keep], C=[imap[days[i]][1] for i in keep])}
    pr = json.load(open(H2.PRED_FILE, encoding="utf-8"))["preds"]
    bull = lambda i: e[8][keep[i]] > e[21][keep[i]] > e[50][keep[i]] > e[200][keep[i]]
    bear = lambda i: e[8][keep[i]] < e[21][keep[i]] < e[50][keep[i]] < e[200][keep[i]]
    pred = lambda i: tuple(pr[sub[i]]) if sub[i] in pr else (None, None)
    cost = lambda leg, j: LI.FEE + LI.TICK / px[leg]["O"][j]
    pols = {p["name"]: p for p in H2.policies()}
    v2 = json.load(open(os.path.join(HERE, "results", "harvest_v2.json"), encoding="utf-8"))["real"]["results"]
    series, check = {}, {}
    for n in NAMES:
        s, final = run_series(pols[n], px, cost, 0, len(sub), pred, bull, bear)
        series[n] = s
        check[n] = abs(100 * (final / H2.CAPITAL - 1) - v2[n]["total_pct"]) < 1e-9
    adjk = [A[i] for i in keep]
    pts = zigzag(adjk, ZIG)
    rows = []
    for kind, i in pts:
        a, b = max(1, i - BEFORE), min(len(sub) - 1, i + AFTER)
        r = dict(kind=kind, day=sub[i], window=[sub[a], sub[b]], market_pct=100 * (px[H2.LONG]["C"][b] / px[H2.LONG]["C"][a] - 1))
        for n in NAMES:
            r[n] = 100 * (series[n][b] / series[n][a] - 1)
        rows.append(r)
    summ = {}
    for kind in ("peak", "trough"):
        rs = [r for r in rows if r["kind"] == kind]
        summ[kind] = {n: dict(mean=sum(r[n] for r in rs) / len(rs), worst=min(r[n] for r in rs), lost_count=sum(r[n] < 0 for r in rs), beats_hold=sum(r[n] > r["hold"] for r in rs), n=len(rs)) for n in NAMES}
    out = os.path.join(HERE, "results", "turning_points_v1.json")
    rec = dict(version="turning-points-v1", code_sha256=sha(os.path.abspath(__file__)), zigzag=ZIG, window=[-BEFORE, AFTER], reproduces_harvest_v2=check, turning_points=rows, summary=summ, note="turning points found in hindsight on 069500 adjusted closes (15% zigzag); window value change of vault + trading money; descriptive check, strategies unchanged")
    with open(out, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print("reproduces harvest_v2 finals", check)
    for r in rows:
        print(f"{r['kind']:6s} {r['day']} window {r['window'][0]}..{r['window'][1]} market {r['market_pct']:+6.1f}% | " + " ".join(f"{n.split('_')[0]}{'' if n in ('hold', 'ribbon') else ('-' + n.split('_')[-1] if n.startswith('A') else '')} {r[n]:+6.1f}" for n in NAMES))
    for kind, s in summ.items():
        print(kind, json.dumps({n: (round(v["mean"], 1), round(v["worst"], 1), f"lost {v['lost_count']}/{v['n']}", f">hold {v['beats_hold']}") for n, v in s.items()}, ensure_ascii=False))
    print("saved", out, sha(out))


if __name__ == "__main__":
    main()
