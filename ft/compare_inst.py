import glob
import hashlib
import json
import os
import sys
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCALE = 10000
FEE = 0.00015
CAPITAL = 100_000_000


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


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
        if self.shares < -1e-9:
            raise ValueError("negative shares")
        self.rows.append(dict(time_utc=str(ts), side=side, shares=n, price_won=px, fee_won=round(f, 4), cash_after=round(self.cash, 4), shares_after=self.shares))


def load_bars(path, start, end_day):
    b = pd.read_feather(path)
    b["date"] = pd.to_datetime(b["date"], utc=True)
    for c in ("open", "high", "low", "close"):
        b[c] = (b[c] * SCALE).round(6)
    b["day"] = b["date"].dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    return b[(b["date"] >= start) & (b["day"] <= end_day)].reset_index(drop=True)


def ribbon_weights(adj_path, adj_sha, key, origin, end_day):
    if sha(adj_path) != adj_sha:
        raise SystemExit("adjusted file hash changed")
    adj = json.load(open(adj_path, encoding="utf-8"))[key]
    days = [d for d in adj["days"] if d <= end_day]
    C = [float(c) for c in adj["C"][:len(days)]]
    if days[0] != origin:
        raise SystemExit(f"origin {days[0]} != declared {origin}")
    e = {}
    for n in (8, 21, 50, 200):
        a, v, out = 2.0 / (n + 1), None, []
        for x in C:
            v = x if v is None else a * x + (1 - a) * v
            out.append(v)
        e[n] = out
    return {d: (1 if e[8][i] > e[21][i] > e[50][i] > e[200][i] else 0) for i, d in enumerate(days)}, days


def nfi_events(result_dir):
    z = sorted(glob.glob(os.path.join(result_dir, "*.zip")))[-1]
    with zipfile.ZipFile(z) as f:
        n = [x for x in f.namelist() if x.endswith(".json") and not x.endswith("_config.json")][0]
        st = json.loads(f.read(n))["strategy"]["NostalgiaForInfinityX8"]
    ev = []
    for t in st["trades"]:
        for o in t["orders"]:
            ts = pd.Timestamp(o["order_filled_timestamp"], unit="ms", tz="UTC")
            ev.append((ts, "buy" if o["ft_is_entry"] else "sell", o["amount"], round(o["safe_price"] * SCALE, 6)))
    ev.sort(key=lambda e: (e[0], e[1] != "buy"))
    wins = sum(1 for t in st["trades"] if t["profit_abs"] > 0)
    gross_win = sum(t["profit_abs"] for t in st["trades"] if t["profit_abs"] > 0)
    gross_loss = -sum(t["profit_abs"] for t in st["trades"] if t["profit_abs"] < 0)
    return ev, dict(result_zip=os.path.relpath(z, HERE), result_sha256=sha(z), backtest_start=st["backtest_start"], trades=len(st["trades"]), wins=wins, losses=len(st["trades"]) - wins, gross_win_units=gross_win, gross_loss_units=gross_loss, engine_final_units=st["final_balance"])


CAL = {}


def auction_buckets(kst_day):
    last = pd.Timestamp(f"{kst_day} {CAL[kst_day]['last']}")
    return {(last - pd.Timedelta(minutes=10)).strftime("%H:%M"), (last - pd.Timedelta(minutes=5)).strftime("%H:%M")}


def fill_check(book, r, n, px):
    v = book.violations
    if abs(n - round(n)) > 1e-9:
        v["non_integer_shares"] += 1
    if not (r["low"] - 1e-6 <= px <= r["high"] + 1e-6):
        v["outside_bar_range"] += 1
    if r["volume"] == 0:
        v["zero_volume_bar"] += 1
    kst = r["date"].tz_convert("Asia/Seoul")
    if kst.strftime("%H:%M") in auction_buckets(kst.strftime("%Y-%m-%d")):
        v["auction_bucket"] += 1
    if r["volume"] > 0:
        book.max_share = max(book.max_share, n / r["volume"])


def nfi_events_from_log(log_path, result_dir):
    sys.path.insert(0, HERE)
    import fills_source as FS
    orders, meta = FS.from_fillgate_log(os.path.join(HERE, log_path), os.path.join(HERE, result_dir), "NostalgiaForInfinityX8")
    ev = [(o["time"], o["side"], o["amount"], round(o["price"] * SCALE, 6)) for o in orders]
    ev.sort(key=lambda e: (e[0], e[1] != "buy"))
    m = dict(source="fillgate_log", log=log_path, log_sha256=sha(os.path.join(HERE, log_path)), fills=meta["fills"], blocked_attempts=meta["blocked"], blocked_force_exits=meta["blocked_force_exits"], open_trades_at_exit=len(meta["open_trades_at_exit"]), backtest_start=meta["backtest_start"], engine_final_units=meta["engine_final_balance"], engine_trades=meta["engine_trades"], force_exit_fills=sum(1 for o in orders if o["is_force_exit"]))
    return ev, m


def mark(book, bars, events):
    ei, daily, five = 0, [], []
    book.violations = dict(non_integer_shares=0, outside_bar_range=0, zero_volume_bar=0, auction_bucket=0, no_bar=0)
    book.no_bar_examples = []
    book.max_share = 0.0
    for i, r in bars.iterrows():
        while ei < len(events) and events[ei][0] < r["date"] + pd.Timedelta(minutes=5):
            ts, side, n, px = events[ei]
            if not (r["date"] <= ts):
                book.violations["no_bar"] += 1
                book.no_bar_examples.append(dict(event_utc=str(ts), applied_at_bar_utc=str(r["date"]), side=side, shares=n, price_won=px))
            fill_check(book, r, n, px)
            book.trade(ts, side, n, px)
            ei += 1
        eq = book.cash + book.shares * r["close"]
        five.append((eq, book.shares * r["close"] / eq if eq > 0 else 0.0))
        if i == len(bars) - 1 or bars.at[i + 1, "day"] != r["day"]:
            daily.append(dict(day=r["day"], close_won=r["close"], cash=round(book.cash, 4), shares=book.shares, equity=round(eq, 4), exposure=round(book.shares * r["close"] / eq, 6) if eq > 0 else 0.0))
    if ei != len(events):
        raise ValueError(f"{book.name}: {len(events) - ei} events after the last bar")
    return daily, five


def dd(series):
    s = pd.Series([float(CAPITAL)] + list(series))
    return round(100 * float((s / s.cummax() - 1).min()), 4)


def summarize(book, daily, five, last_close, extra=None):
    nlv = book.cash + book.shares * last_close - book.shares * last_close * FEE
    out = dict(fill_violations=dict(book.violations), no_bar_examples=book.no_bar_examples[:5], bar_rule="an order is checked against the bar with bar.date <= order time < bar.date + 5 min; an order whose time falls in a gap is counted as no_bar and applied at the next real bar only to keep the ledger continuous", max_order_share_of_bar_volume=round(book.max_share, 6), net_liquidation_value_won=round(nlv), return_pct=round(100 * (nlv / CAPITAL - 1), 4), drawdown_basis="equity series with the initial 100,000,000 won prepended", terminal_note="valuation at the window's last 5m close minus exit commission; not a strategy exit", shares_at_end=book.shares, max_dd_daily_close_pct=dd([d["equity"] for d in daily]), max_dd_5m_close_pct=dd([x[0] for x in five]), exposure_daily_sample_mean=round(sum(d["exposure"] for d in daily) / len(daily), 4), exposure_5m_mean=round(sum(x[1] for x in five) / len(five), 6), exposure_5m_max=round(max(x[1] for x in five), 6), orders=len(book.rows), round_trips=sum(1 for r in book.rows if r["side"] == "sell" and r["shares_after"] == 0))
    if extra:
        out.update(extra)
        if "trades" in out:
            if out["trades"] == 0:
                out["profit_factor"] = "no trades"
            elif out["gross_loss_units"] == 0:
                out["profit_factor"] = "loss denominator 0"
            else:
                out["profit_factor"] = round(out["gross_win_units"] / out["gross_loss_units"], 6)
    return out


def main(params_path):
    P = json.load(open(params_path, encoding="utf-8"))
    CAL.update(json.load(open(os.path.join(HERE, P["session_calendar"]), encoding="utf-8"))["days"])
    out_dir = os.path.join(HERE, P["out_dir"])
    os.makedirs(out_dir, exist_ok=False)
    start = pd.Timestamp(P["start_utc"], tz="UTC")
    bars = load_bars(os.path.join(HERE, P["bars"]), start, P["end_day"])
    if str(bars["date"].iloc[0]) != str(start):
        raise SystemExit(f"first bar {bars['date'].iloc[0]} != declared start {start}")
    last_close = float(bars["close"].iloc[-1])
    first = bars.iloc[0]
    res = {}
    for variant, rdir in P.get("nfi_dirs", {}).items():
        ev, meta = nfi_events(os.path.join(HERE, rdir))
        meta["backtest_start_matches"] = pd.Timestamp(meta["backtest_start"], tz="UTC") == start
        book = Book(f"nfi_{variant}")
        daily, five = mark(book, bars, ev)
        meta["replay_matches_engine"] = abs(book.cash / SCALE - meta["engine_final_units"]) < 1e-4 and abs(book.shares) < 1e-9
        res[book.name] = (book, daily, five, meta)
    for variant, src in P.get("nfi_logs", {}).items():
        ev, meta = nfi_events_from_log(src["log"], src["result_dir"])
        meta["backtest_start_matches"] = pd.Timestamp(meta["backtest_start"], tz="UTC") == start
        book = Book(f"nfi_{variant}")
        daily, five = mark(book, bars, ev)
        meta["engine_export_complete"] = meta["open_trades_at_exit"] == 0
        meta["replay_matches_engine"] = (abs(book.cash / SCALE - meta["engine_final_units"]) < 1e-4 and abs(book.shares) < 1e-9) if meta["engine_export_complete"] else "not comparable: open trades at exit are missing from the engine export"
        res[book.name] = (book, daily, five, meta)
    hold = Book("hold")
    n = int(CAPITAL // (first["open"] * (1 + FEE)))
    daily, five = mark(hold, bars, [(first["date"], "buy", n, float(first["open"]))])
    res["hold"] = (hold, daily, five, None)
    w, wdays = ribbon_weights(P["adj_path"], P["adj_sha256"], P["adj_key"], P["origin_day"], P["end_day"])
    sessions = list(dict.fromkeys(bars["day"]))
    first_bar_of = bars.groupby("day").first()
    prev_day = {d: wdays[wdays.index(d) - 1] for d in sessions}
    ev, held = [], 0
    for d in sessions:
        target = w[prev_day[d]]
        fb = first_bar_of.loc[d]
        if target == 1 and held == 0:
            ev.append((fb["date"], "buy", float(fb["open"])))
            held = 1
        elif target == 0 and held == 1:
            ev.append((fb["date"], "sell", float(fb["open"])))
            held = 0
    resolved, cash_sim, sh = [], float(CAPITAL), 0
    for ts, side, px in ev:
        if side == "buy":
            k = int(cash_sim // (px * (1 + FEE)))
            cash_sim -= k * px * (1 + FEE)
            sh += k
            resolved.append((ts, "buy", k, px))
        else:
            cash_sim += sh * px * (1 - FEE)
            resolved.append((ts, "sell", sh, px))
            sh = 0
    rb = Book("ribbon")
    daily, five = mark(rb, bars, resolved)
    res["ribbon"] = (rb, daily, five, dict(first_weight_from=prev_day[sessions[0]], first_weight=w[prev_day[sessions[0]]]))
    summary = dict(version="instrument-same-basis-compare-v2.3", params=P, params_sha256=sha(params_path), code_sha256=sha(os.path.abspath(__file__)), bars_sha256=sha(os.path.join(HERE, P["bars"])), window=[str(bars["date"].iloc[0]), str(bars["date"].iloc[-1])], sessions=len(sessions), first_bar_open_won=float(first["open"]), last_close_won=last_close, methods={})
    for name, (book, daily, five, meta) in res.items():
        pd.DataFrame(book.rows).to_csv(os.path.join(out_dir, f"ledger_{name}.csv"), index=False)
        pd.DataFrame(daily).to_csv(os.path.join(out_dir, f"daily_{name}.csv"), index=False)
        summary["methods"][name] = summarize(book, daily, five, last_close, meta)
        summary["methods"][name]["ledger_sha256"] = sha(os.path.join(out_dir, f"ledger_{name}.csv"))
        summary["methods"][name]["daily_sha256"] = sha(os.path.join(out_dir, f"daily_{name}.csv"))
    p = os.path.join(out_dir, "compare.json")
    json.dump(summary, open(p, "x", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print("window", summary["window"], "sessions", summary["sessions"])
    for k, v in summary["methods"].items():
        print(f"{k:12s} NLV {v['net_liquidation_value_won']:13,d} return {v['return_pct']:8.3f}% ddDaily {v['max_dd_daily_close_pct']:8.3f}% dd5m {v['max_dd_5m_close_pct']:8.3f}% expoDaily {v['exposure_daily_sample_mean']:.3f} expo5m {v['exposure_5m_mean']:.4f}/{v['exposure_5m_max']:.4f} orders {v['orders']}")
    print("saved", p, sha(p)[:8])


if __name__ == "__main__":
    main(sys.argv[1])
