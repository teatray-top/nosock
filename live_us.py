import json
import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

import reader as R
import sim
from sim import Data
from toss import ApiError, Toss

ET = ZoneInfo("America/New_York")
KST = ZoneInfo("Asia/Seoul")
HERE = os.path.dirname(os.path.abspath(__file__))
H = 36
THR = float(os.environ.get("LIVE_THR", 0.55))
MODEL = os.environ.get("LIVE_MODEL", "runs/reader_us5m_h36_s0.06_e59_1793801624.pt")
EXIT_BAR = 64
ONCE = os.environ.get("LIVE_ONCE") == "1"
RPS = float(os.environ.get("TOSS_RPS", 15))
DRY = os.environ.get("DOBBY_LIVE") != "1"
B = 78
LOG = os.path.join(HERE, "runs", f"live_us_{datetime.now(ET).strftime('%Y%m%d')}.jsonl")
R.STOP = 0.06


def log(kind, **kw):
    row = dict(ts=datetime.now(KST).isoformat(timespec="seconds"), kind=kind, **kw)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(row, ensure_ascii=False, default=str)[:300], flush=True)


class Pacer:
    def __init__(self, rps):
        self.dt, self.t = 1.0 / rps, 0.0

    def wait(self):
        now = time.time()
        if now < self.t + self.dt:
            time.sleep(self.t + self.dt - now)
        self.t = time.time()


def et(ts):
    return pd.Timestamp(ts).tz_convert(ET)


def bar_index(ts_end, day):
    t = et(ts_end)
    if t.date() != day:
        return -1
    m = t.hour * 60 + t.minute - 570 - 1
    return m // 5 if 0 <= m < B * 5 else -1


class Live:
    def __init__(self):
        self.t = Toss()
        self.t.account = self.t.accounts()[0]["accountSeq"]
        self.pacer = Pacer(RPS)
        u = pd.read_csv(os.path.join(HERE, "data", "us", "universe.csv"), keep_default_na=False)
        self.syms = [s for s in u.sym.astype(str) if s]
        self.N = len(self.syms)
        self.day = datetime.now(ET).date()
        self.O, self.H, self.L, self.C, self.V = [np.full((B, self.N), np.nan) for _ in range(5)]
        self.V[:] = 0.0
        self.pc = np.zeros(self.N)
        self.pturn = np.zeros(self.N)
        self.model = None
        self.pos, self.entry, self.held_b, self.qty = None, 0.0, -1, 0.0
        self.buys = 0

    def candles(self, sym, interval, count, before=None):
        self.pacer.wait()
        return self.t.candles(sym, interval, count=count, before=before)["candles"]

    def load_daily(self):
        ok = 0
        for j, s in enumerate(self.syms):
            try:
                c = self.candles(s, "1d", 3)
            except ApiError as e:
                continue
            prev = [x for x in c if et(x["timestamp"]).date() < self.day]
            if prev:
                self.pc[j] = float(prev[0]["closePrice"])
                self.pturn[j] = float(prev[0]["closePrice"]) * float(prev[0]["volume"])
                ok += 1
        log("daily", symbols=self.N, with_prev_close=ok, elig=int((self.pturn >= 1e6).sum()))

    def ingest(self, j, bars):
        for x in bars:
            b = bar_index(x["timestamp"], self.day)
            if b < 0:
                continue
            o, h, l, c, v = (float(x[k]) for k in ("openPrice", "highPrice", "lowPrice", "closePrice", "volume"))
            if np.isnan(self.O[b, j]):
                self.O[b, j], self.H[b, j], self.L[b, j], self.C[b, j], self.V[b, j] = o, h, l, c, v
            else:
                self.H[b, j] = max(self.H[b, j], h)
                self.L[b, j] = min(self.L[b, j], l)
                self.C[b, j] = c
                self.V[b, j] += v
            self.seen.add((b, j, x["timestamp"]))

    def refresh(self, count):
        self.seen_new = 0
        for j, s in enumerate(self.syms):
            if self.pc[j] <= 0:
                continue
            try:
                bars = self.candles(s, "1m", count)
            except ApiError:
                continue
            fresh = [x for x in bars if (bar_index(x["timestamp"], self.day), j, x["timestamp"]) not in self.seen]
            self.ingest(j, fresh)

    def data(self):
        D = 2
        z = lambda: np.full((D, B, self.N), np.nan)
        O, H_, L, C, V = z(), z(), z(), z(), np.zeros((D, B, self.N))
        O[1], H_[1], L[1], C[1], V[1] = self.O, self.H, self.L, self.C, self.V
        bar0 = (np.arange(B) == 0)[None, :, None]
        has = ~np.isnan(C) & ((V > 0) | (bar0 & ~np.isnan(O)))
        Cf = C.copy()
        Cf[1] = pd.DataFrame(C[1]).ffill().values
        pc = np.zeros((D, self.N))
        pc[1] = self.pc
        turn = np.zeros((D, self.N))
        turn[0] = self.pturn
        oc = np.nan_to_num(Cf[:, -1, :])
        arrays = dict(O=O, H=H_, L=L, C=Cf, V=V, has=has, pc=pc, oc=oc, turn=turn, days=np.array(["prev", str(self.day)]), syms=np.array(self.syms))
        return Data(R.DEV, "us5m", arrays)

    def score(self, b):
        X = self.data()
        if self.model is None:
            hor = R.horizon(X, H)
            f = R.feats(X, [1], hor)
            self.model = R.Reader(f.shape[-1]).to(X.dev)
            self.model.load_state_dict(torch.load(os.path.join(HERE, MODEL), map_location=X.dev))
            self.model.eval()
        hor = R.horizon(X, H)
        f = R.feats(X, [1], hor)[0, b]
        with torch.no_grad():
            p = torch.sigmoid(self.model(f))[:, 0]
        mask = X.buyable[1, b] & (b + 1 <= X.entry_last)
        sc = p.masked_fill(~mask, 0)
        top = torch.topk(sc, 5)
        return X, p, sc, [(self.syms[int(i)], round(float(v), 3)) for v, i in zip(top.values, top.indices)]

    def buy(self, j, p):
        usd = float(self.t.buying_power("USD")["cashBuyingPower"])
        amt = f"{max(0.0, np.floor(usd * 100 - 1) / 100):.2f}"
        if float(amt) < 0.5:
            log("skip_buy", reason="usd too small", usd=usd)
            return
        if DRY:
            log("dry_buy", symbol=self.syms[j], p=round(p, 3), amount=amt)
            self.pos, self.entry, self.qty = j, float(self.C[self.last_b, j]), 0.0
            return
        r = self.t.place(self.syms[j], "BUY", "MARKET", amount=amt, client_order_id="live-" + uuid.uuid4().hex[:12])
        o = self.wait(r["orderId"])
        ex = o["execution"]
        log("buy", symbol=self.syms[j], p=round(p, 3), amount=amt, status=o["status"], qty=ex["filledQuantity"], price=ex["averageFilledPrice"], commission=ex["commission"])
        if o["status"] == "FILLED" and float(ex["filledQuantity"]) > 0:
            self.pos, self.entry, self.qty = j, float(ex["averageFilledPrice"]), float(ex["filledQuantity"])
            self.buys += 1

    def sell(self, reason):
        s = self.syms[self.pos]
        if DRY:
            log("dry_sell", symbol=s, reason=reason, last=float(self.C[self.last_b, self.pos]))
            self.pos = None
            return
        sq = self.t.get("/api/v1/sellable-quantity", account=True, symbol=s)["sellableQuantity"]
        if float(sq) <= 0:
            log("sell_skip", symbol=s, reason="no sellable qty")
            self.pos = None
            return
        r = self.t.place(s, "SELL", "MARKET", quantity=sq, client_order_id="live-" + uuid.uuid4().hex[:12])
        o = self.wait(r["orderId"])
        ex = o["execution"]
        pnl = (float(ex["filledAmount"] or 0) - float(ex["commission"] or 0) - float(ex["tax"] or 0)) - self.entry * self.qty
        log("sell", symbol=s, reason=reason, status=o["status"], qty=ex["filledQuantity"], price=ex["averageFilledPrice"], commission=ex["commission"], tax=ex["tax"], entry=self.entry, pnl_usd=round(pnl, 4))
        if o["status"] == "FILLED":
            self.pos = None

    def wait(self, oid, timeout=90):
        t0 = time.time()
        while time.time() - t0 < timeout:
            o = self.t.order(oid)
            if o["status"] in ("FILLED", "CANCELED", "REJECTED"):
                return o
            time.sleep(2)
        return self.t.order(oid)

    def run(self):
        self.seen = set()
        log("start", dry=DRY, thr=THR, model=MODEL, universe=self.N, day=str(self.day))
        self.load_daily()
        self.refresh(200)
        while True:
            now = datetime.now(ET)
            m = now.hour * 60 + now.minute - 570
            if m < 5:
                time.sleep(30)
                continue
            b = m // 5 - 1
            self.last_b = b
            if b >= EXIT_BAR or os.path.exists(os.path.join(HERE, "runs", "STOP")):
                if self.pos is not None:
                    self.sell("forced_exit" if b >= EXIT_BAR else "stop_file")
                log("end", bar=b, buys=self.buys)
                return
            X, p, sc, top = self.score(b)
            act = None
            if self.pos is not None:
                c = float(self.C[b, self.pos]) if not np.isnan(self.C[b, self.pos]) else self.entry
                if b - self.held_b >= H:
                    act = "time_exit"
                elif c <= self.entry * (1 - R.STOP):
                    act = "stop"
                log("hold", bar=b, symbol=self.syms[self.pos], last=c, entry=self.entry, upnl_pct=round((c / self.entry - 1) * 100, 3), top=top)
                if act:
                    self.sell(act)
            else:
                j = int(sc.argmax())
                log("scan", bar=b, best=self.syms[j], p=round(float(sc[j]), 3), n_buyable=int((sc > 0).sum()), top=top)
                if float(sc[j]) > THR and self.buys < 4:
                    self.held_b = b
                    self.buy(j, float(sc[j]))
            if ONCE:
                log("once", p_mean=round(float(p.mean()), 4), p_max=round(float(p.max()), 4), feat_dim=int(R.feats(X, [1], R.horizon(X, H)).shape[-1]), has_bars=int(X.has[1].sum()), buyable_now=int((sc > 0).sum()))
                return
            nxt = (now + timedelta(minutes=5 - (now.minute % 5))).replace(second=12, microsecond=0)
            time.sleep(max(1, (nxt - datetime.now(ET)).total_seconds()))
            self.refresh(12)


if __name__ == "__main__":
    Live().run()
