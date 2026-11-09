import glob
import hashlib
import json
import os
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "compare_b_v1")
BARS = os.path.join(HERE, "user_data", "data_kodex_nfi_v1_2_algo", "binance", "ALGO_USDT-5m.feather")
ADJ = r"G:\NoSock\stock\penny\data\kr_etf\kodex_daily_adj.json"
ADJ_SHA = "844b27891727b1faa227ea3259cb47eee258f8bf85ec30a3fc6a41b5a299a609"
SCALE = 10000
FEE = 0.00015
CAPITAL = 100_000_000
START = pd.Timestamp("2026-01-26 00:30:00", tz="UTC")
END_DAY = "2026-11-05"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_bars():
    b = pd.read_feather(BARS)
    b["date"] = pd.to_datetime(b["date"], utc=True)
    for c in ("open", "high", "low", "close"):
        b[c] = (b[c] * SCALE).round(6)
    b["day"] = b["date"].dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    b = b[(b["date"] >= START) & (b["day"] <= END_DAY)].reset_index(drop=True)
    return b


class Book:
    def __init__(self, name):
        self.name, self.cash, self.shares, self.rows = name, float(CAPITAL), 0, []

    def trade(self, ts, side, n, px):
        f = n * px * FEE
        if side == "buy":
            self.cash -= n * px + f
            self.shares += n
        else:
            self.cash += n * px - f
            self.shares -= n
        if self.shares < 0:
            raise ValueError("negative shares")
        self.rows.append(dict(time_utc=str(ts), side=side, shares=n, price_won=px, fee_won=round(f, 4), cash_after=round(self.cash, 4), shares_after=self.shares))


def ribbon_weights():
    if sha(ADJ) != ADJ_SHA:
        raise SystemExit("adjusted file hash changed")
    adj = json.load(open(ADJ, encoding="utf-8"))["069500"]
    days = [d for d in adj["days"] if d <= END_DAY]
    C = [float(c) for c in adj["C"][:len(days)]]
    if days[0] != "2002-10-14":
        raise SystemExit("origin")
    e = {}
    for n in (8, 21, 50, 200):
        a, v, out = 2.0 / (n + 1), None, []
        for x in C:
            v = x if v is None else a * x + (1 - a) * v
            out.append(v)
        e[n] = out
    return {d: (1 if e[8][i] > e[21][i] > e[50][i] > e[200][i] else 0) for i, d in enumerate(days)}, days


def nfi_events(variant):
    z = sorted(glob.glob(os.path.join(HERE, "results", "nfi_kr_v1_2", "B", variant, "*.zip")))[-1]
    with zipfile.ZipFile(z) as f:
        n = [x for x in f.namelist() if x.endswith(".json") and not x.endswith("_config.json")][0]
        st = json.loads(f.read(n))["strategy"]["NostalgiaForInfinityX8"]
    ev = []
    for t in st["trades"]:
        for o in t["orders"]:
            ts = pd.Timestamp(o["order_filled_timestamp"], unit="ms", tz="UTC")
            ev.append((ts, "buy" if o["ft_is_entry"] else "sell", int(round(o["amount"])), round(o["safe_price"] * SCALE, 6)))
    ev.sort(key=lambda e: (e[0], e[1] != "buy"))
    wins = sum(1 for t in st["trades"] if t["profit_abs"] > 0)
    return ev, dict(result_zip=os.path.relpath(z, HERE), result_sha256=sha(z), trades=len(st["trades"]), wins=wins, losses=len(st["trades"]) - wins, engine_final_units=st["final_balance"])


def mark(book, bars, events):
    ei, daily, five = 0, [], []
    for i, r in bars.iterrows():
        while ei < len(events) and events[ei][0] < r["date"] + pd.Timedelta(minutes=5):
            ts, side, n, px = events[ei]
            book.trade(ts, side, n, px)
            ei += 1
        eq = book.cash + book.shares * r["close"]
        five.append(eq)
        last_of_day = i == len(bars) - 1 or bars.at[i + 1, "day"] != r["day"]
        if last_of_day:
            daily.append(dict(day=r["day"], close_won=r["close"], cash=round(book.cash, 4), shares=book.shares, equity=round(eq, 4), exposure=round(book.shares * r["close"] / eq, 6) if eq > 0 else 0.0))
    if ei != len(events):
        raise ValueError(f"{book.name}: {len(events) - ei} events after the last bar")
    return daily, five


def dd(series):
    s = pd.Series(series)
    return round(100 * float((s / s.cummax() - 1).min()), 4)


def summarize(book, daily, five, last_close, extra=None):
    nlv = book.cash + book.shares * last_close - book.shares * last_close * FEE
    out = dict(net_liquidation_value_won=round(nlv), return_pct=round(100 * (nlv / CAPITAL - 1), 4), terminal_note="valuation at the last 5m close of 2026-11-05 minus exit commission; not a strategy exit", shares_at_end=book.shares, max_dd_daily_close_pct=dd([d["equity"] for d in daily]), max_dd_5m_close_pct=dd(five), avg_exposure=round(sum(d["exposure"] for d in daily) / len(daily), 4), orders=len(book.rows), round_trips=sum(1 for r in book.rows if r["side"] == "sell" and r["shares_after"] == 0))
    if extra:
        out.update(extra)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    bars = load_bars()
    last_close = float(bars["close"].iloc[-1])
    first = bars.iloc[0]
    res = {}
    for variant in ("default", "single"):
        ev, meta = nfi_events(variant)
        book = Book(f"nfi_{variant}")
        daily, five = mark(book, bars, ev)
        meta["replay_matches_engine"] = abs(book.cash / SCALE - meta["engine_final_units"]) < 1e-4 and book.shares == 0
        res[book.name] = (book, daily, five, meta)
    hold = Book("hold")
    n = int(CAPITAL // (first["open"] * (1 + FEE)))
    daily, five = mark(hold, bars, [(first["date"], "buy", n, float(first["open"]))])
    res["hold"] = (hold, daily, five, None)
    w, wdays = ribbon_weights()
    sessions = list(dict.fromkeys(bars["day"]))
    first_bar_of = bars.groupby("day").first()
    prev_day = {d: wdays[wdays.index(d) - 1] for d in sessions}
    rb = Book("ribbon")
    ev, held = [], 0
    for d in sessions:
        target = w[prev_day[d]]
        fb = first_bar_of.loc[d]
        ts, px = fb["date"], float(fb["open"])
        if target == 1 and held == 0:
            ev.append((ts, "buy", None, px))
            held = 1
        elif target == 0 and held == 1:
            ev.append((ts, "sell", None, px))
            held = 0
    resolved, cash_sim, sh = [], float(CAPITAL), 0
    for ts, side, _, px in ev:
        if side == "buy":
            k = int(cash_sim // (px * (1 + FEE)))
            cash_sim -= k * px * (1 + FEE)
            sh += k
            resolved.append((ts, "buy", k, px))
        else:
            cash_sim += sh * px * (1 - FEE)
            resolved.append((ts, "sell", sh, px))
            sh = 0
    daily, five = mark(rb, bars, resolved)
    res["ribbon"] = (rb, daily, five, dict(first_weight_from=prev_day[sessions[0]], first_weight=w[prev_day[sessions[0]]]))
    summary = dict(version="kodex-window-b-same-basis-compare-v1", spec_sha256=sha(os.path.join(OUT, "compare_b_spec_v1.json")), code_sha256=sha(os.path.abspath(__file__)), bars_sha256=sha(BARS), adjusted_sha256=ADJ_SHA, window=[str(bars["date"].iloc[0]), str(bars["date"].iloc[-1])], sessions=len(sessions), first_bar_open_won=float(first["open"]), last_close_won=last_close, methods={})
    for name, (book, daily, five, meta) in res.items():
        pd.DataFrame(book.rows).to_csv(os.path.join(OUT, f"ledger_{name}.csv"), index=False)
        pd.DataFrame(daily).to_csv(os.path.join(OUT, f"daily_{name}.csv"), index=False)
        summary["methods"][name] = summarize(book, daily, five, last_close, meta)
        summary["methods"][name]["ledger_sha256"] = sha(os.path.join(OUT, f"ledger_{name}.csv"))
        summary["methods"][name]["daily_sha256"] = sha(os.path.join(OUT, f"daily_{name}.csv"))
    p = os.path.join(OUT, "compare_b_v1.json")
    json.dump(summary, open(p, "x", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print("window", summary["window"], "sessions", summary["sessions"], "first open", summary["first_bar_open_won"], "last close", last_close)
    for k, v in summary["methods"].items():
        print(f"{k:12s} NLV {v['net_liquidation_value_won']:13,d} return {v['return_pct']:8.3f}% ddDaily {v['max_dd_daily_close_pct']:8.3f}% dd5m {v['max_dd_5m_close_pct']:8.3f}% expo {v['avg_exposure']:.3f} orders {v['orders']} trips {v['round_trips']} " + (f"wins {v.get('wins')} losses {v.get('losses')} replay_ok {v.get('replay_matches_engine')}" if 'wins' in v else ""))
    print("saved", p, sha(p)[:8])


if __name__ == "__main__":
    main()
