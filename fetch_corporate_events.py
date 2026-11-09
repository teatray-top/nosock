import json
import os
import sys
import time
from datetime import date, datetime, timedelta

import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "events", "corporate.json")
EARN = {"NVDA": "엔비디아", "TSM": "TSMC", "AVGO": "브로드컴", "AMD": "AMD", "ASML": "ASML", "MU": "마이크론", "AAPL": "애플", "MSFT": "마이크로소프트", "GOOGL": "알파벳", "AMZN": "아마존", "META": "메타", "FIVN": "Five9", "005930.KS": "삼성전자", "000660.KS": "SK하이닉스"}


def earnings(t):
    out = []
    try:
        df = yf.Ticker(t).get_earnings_dates(limit=8)
        if df is not None:
            for ix in df.index:
                out.append(ix.date().isoformat())
    except Exception:
        pass
    try:
        cal = yf.Ticker(t).calendar or {}
        for d in cal.get("Earnings Date", []) or []:
            out.append(d.isoformat() if hasattr(d, "isoformat") else str(d)[:10])
    except Exception:
        pass
    return sorted(set(out))


def exdiv(t):
    try:
        s = yf.Ticker(t).dividends
    except Exception:
        return [], None
    if s is None or len(s) == 0:
        return [], None
    days = [ix.date().isoformat() for ix in s.index]
    amts = [float(x) for x in s.values]
    gaps = [(date.fromisoformat(b) - date.fromisoformat(a)).days for a, b in zip(days[-6:], days[-5:])]
    nxt = None
    if gaps:
        g = sorted(gaps)[len(gaps) // 2]
        n = date.fromisoformat(days[-1]) + timedelta(days=g)
        nxt = dict(date=n.isoformat(), estimated=True, last_amount=amts[-1], period_days=g)
    return list(zip(days[-8:], amts[-8:])), nxt


def main(etfs):
    res = dict(fetched_at=datetime.now().astimezone().isoformat(timespec="seconds"), earnings={}, exdiv={})
    for t, nm in EARN.items():
        res["earnings"][t] = dict(name=nm, dates=earnings(t))
        time.sleep(0.5)
    for c, y in etfs.items():
        hist, nxt = exdiv(y)
        res["exdiv"][c] = dict(yahoo=y, recent=hist, next=nxt)
        time.sleep(0.5)
    tmp = OUT + ".tmp"
    json.dump(res, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, OUT)
    return res


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    import mom_shadow as MS
    us = ["SPY", "QQQ", "DIA", "IWM", "SMH", "XLK", "XLE", "XLF", "XLV", "XLP", "XLU", "EEM", "EFA", "GLD", "SLV", "USO", "UUP", "TLT", "IEF", "SHY", "SH"]
    etfs = {c: c + ".KS" for c in MS.KR_DOMESTIC} | {c: c for c in us}
    r = main(etfs)
    today = date.today().isoformat()
    for t, v in r["earnings"].items():
        up = [d for d in v["dates"] if d >= today]
        print(t, v["name"], "next earnings", up[:1], "past", [d for d in v["dates"] if d < today][-1:])
    for c, v in r["exdiv"].items():
        print(c, "next ex-div", v["next"], "last", v["recent"][-1:] if v["recent"] else None)
