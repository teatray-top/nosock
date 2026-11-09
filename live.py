import json
import math
import os
import sys
import time
import uuid
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

import bill
import reader as R
import sim
from sim import Data
from toss import ApiError, Toss

HERE = os.path.dirname(os.path.abspath(__file__))
KST = ZoneInfo("Asia/Seoul")
MK = {
    "us": dict(tz=ZoneInfo("America/New_York"), open_min=570, B=78, iv="us5m", exit_bar=int(os.environ.get("LIVE_EXIT_BAR", 64)), model="runs/reader_us5m_h36_s0.06_e59_1793801624.pt", thr=0.55, universe=os.path.join(HERE, "data", "us", "universe.csv"), min_turn=1e6),
    "kr": dict(tz=KST, open_min=540, B=72, iv="all5m", exit_bar=70, model="runs/reader_all5m_h36_s0.06_e59_1793802831.pt", thr=0.4, universe=os.path.join(HERE, "data", "toss", "universe_kr.json"), min_turn=3e8),
}
H = int(os.environ.get("LIVE_H", 36))
TP = float(os.environ.get("SCALP_TP", 0))
MIN_PX = float(os.environ.get("LIVE_MIN_PX", 5.0 if (sys.argv[1] if len(sys.argv) > 1 else "us") == "us" else 0.0))
RPS = float(os.environ.get("TOSS_RPS", 15))
MAX_BOOK_AGE = float(os.environ.get("DRY_MAX_BOOK_AGE", 30))
LEDGER_V = 2
DRY = os.environ.get("DOBBY_LIVE") != "1"
ONCE = os.environ.get("LIVE_ONCE") == "1"
R.STOP = float(os.environ.get("READER_STOP", 0.06))


class Pacer:
    def __init__(self, rps):
        self.dt, self.t = 1.0 / rps, 0.0

    def wait(self):
        now = time.time()
        if now < self.t + self.dt:
            time.sleep(self.t + self.dt - now)
        self.t = time.time()


class Live:
    def __init__(self, mk):
        self.mk = mk
        self.cfg = MK[mk]
        self.tz, self.B = self.cfg["tz"], self.cfg["B"]
        self.thr = float(os.environ.get("LIVE_THR", self.cfg["thr"]))
        self.model_path = os.environ.get("LIVE_MODEL", self.cfg["model"] or "")
        self.day = date.fromisoformat(os.environ["LIVE_DAY"]) if os.environ.get("LIVE_DAY") else datetime.now(self.tz).date()
        self.log_path = os.environ.get("LIVE_LOG") or os.path.join(HERE, "runs", f"live_{mk}_{self.day.strftime('%Y%m%d')}.jsonl")
        self.t = Toss()
        self.t.account = self.t.accounts()[0]["accountSeq"]
        self.pacer = Pacer(RPS)
        if mk == "us":
            u = pd.read_csv(os.environ.get("LIVE_UNIVERSE", self.cfg["universe"]), keep_default_na=False)
            self.syms = [s for s in u.sym.astype(str) if s]
        else:
            self.syms = [s["symbol"] for s in json.load(open(self.cfg["universe"], encoding="utf-8"))]
        self.N = len(self.syms)
        self.O, self.H, self.L, self.C = [np.full((self.B, self.N), np.nan) for _ in range(4)]
        self.V = np.zeros((self.B, self.N))
        self.pc, self.pturn = np.zeros(self.N), np.zeros(self.N)
        self.seen = set()
        self.model = None
        self.pos, self.entry, self.held_b, self.qty, self.buys, self.last_b = None, 0.0, -1, 0.0, 0, 0
        self.cash, self.cost, self.realized, self.cash_src, self.skipped_quotes = None, 0.0, 0.0, None, 0
        self.resume_info = None
        self.entry_ref = os.environ.get("DRY_ENTRY_REF", "last")
        if self.entry_ref not in ("last", "fill"):
            raise SystemExit("DRY_ENTRY_REF must be 'last' or 'fill'")
        if DRY:
            if os.environ.get("DRY_CASH"):
                self.cash, self.cash_src = float(os.environ["DRY_CASH"]), "env DRY_CASH"
            else:
                cur = "USD" if mk == "us" else "KRW"
                self.cash, self.cash_src = float(self.t.buying_power(cur)["cashBuyingPower"]), f"buying_power {cur} at start"
            self.resume_dry()

    def ledger_state(self):
        return dict(v=LEDGER_V, cash=self.cash, qty=self.qty, cost=self.cost, entry=self.entry, realized=self.realized, buys=self.buys, held_b=self.held_b, pos=None if self.pos is None else self.syms[self.pos])

    def apply_ledger(self, r, need=("cash", "qty", "cost", "entry", "realized", "buys", "held_b", "pos")):
        L = r["ledger"]
        missing = [k for k in need if k not in L]
        if missing:
            raise SystemExit(f"ledger row {r.get('ts')} lacks {missing}")
        if L["pos"] is not None and L["pos"] not in self.syms:
            raise SystemExit(f"ledger row {r.get('ts')} holds {L['pos']} which is not in the universe")
        self.cash, self.qty, self.cost, self.entry, self.realized, self.buys, self.held_b = float(L["cash"]), float(L["qty"]), float(L["cost"]), float(L["entry"]), float(L["realized"]), int(L["buys"]), int(L["held_b"])
        self.pos = None if L["pos"] is None else self.syms.index(L["pos"])

    def resume_dry(self):
        mode = os.environ.get("DRY_RESUME", "1")
        self.resume_info = dict(mode=mode, log=self.log_path, seed_ts=None, applied=0, legacy=0, before_boundary=0, boundary_ts=None)
        rows = [json.loads(line) for line in open(self.log_path, encoding="utf-8")] if os.path.exists(self.log_path) else []
        is_v2 = lambda r: isinstance(r.get("ledger"), dict) and r["ledger"].get("v") == LEDGER_V
        cut = max([i for i, r in enumerate(rows) if r.get("kind") == "dry_reset"], default=-1)
        if cut >= 0:
            self.resume_info["boundary_ts"] = rows[cut].get("ts")
            self.resume_info["before_boundary"] = sum(r.get("kind") in ("dry_buy", "dry_sell") for r in rows[:cut])
        after = rows[cut + 1:]
        seeds = ([rows[cut]] if cut >= 0 else []) + [r for r in after if r.get("kind") == "dry_seed"]
        dry = [r for r in after if r.get("kind") in ("dry_buy", "dry_sell")]
        legacy = [r.get("ts") for r in dry if "cash" in r and not is_v2(r)]
        v2 = [r for r in dry if is_v2(r)]
        self.resume_info["legacy"] = len(legacy)
        if mode == "0":
            self.log("dry_reset", reason="DRY_RESUME=0: new ledger seed from here; earlier dry rows in this file are ignored by later resumes", ignored_v2_rows=len(v2), ignored_legacy_rows=len(legacy), ignored_seeds=len(seeds), last_ignored_ts=(dry or seeds or [{}])[-1].get("ts"), cash_src=self.cash_src, ledger=self.ledger_state())
            return
        if legacy:
            raise SystemExit(f"{self.log_path} has {len(legacy)} pre-v{LEDGER_V} ledger rows (first {legacy[0]}): point LIVE_LOG at a new file or set DRY_RESUME=0 to start a fresh ledger")
        if not seeds:
            funded = [r.get("ts") for r in after if (r.get("kind") == "start" and r.get("dry_cash") is not None) or r.get("kind") == "resume_skipped"]
            if v2 or funded:
                raise SystemExit(f"{self.log_path} has {len(v2)} ledger rows and {len(funded)} pre-seed funding records (start dry_cash / resume_skipped, first {(funded or [None])[0]}) but no seed row: point LIVE_LOG at a new file or set DRY_RESUME=0 to start a fresh ledger")
            self.log("dry_seed", reason="first dry start of this log: ledger seed", cash_src=self.cash_src, ledger=self.ledger_state())
            return
        bad = [s for s in seeds if not is_v2(s)]
        if bad:
            raise SystemExit(f"seed row {bad[0].get('ts')} lacks a v{LEDGER_V} ledger")
        self.apply_ledger(seeds[0])
        self.resume_info["seed_ts"] = seeds[0].get("ts")
        for r in v2:
            self.apply_ledger(r)
        self.resume_info["applied"] = len(v2)
        self.cash_src = f"resumed from {os.path.basename(self.log_path)}: seed {seeds[0].get('ts')} + {len(v2)} ledger rows"

    @staticmethod
    def pos_num(x):
        if isinstance(x, bool):
            return None
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) and v > 0 else None

    def quote(self, top):
        ask, bid = self.pos_num(top.get("ask1")), self.pos_num(top.get("bid1"))
        aq, bq = self.pos_num(top.get("ask1_qty")), self.pos_num(top.get("bid1_qty"))
        age = None
        try:
            age = round((datetime.now(KST) - datetime.fromisoformat(str(top.get("ob_ts")))).total_seconds(), 1)
        except (TypeError, ValueError):
            pass
        bad = []
        if "ob_error" in top:
            bad.append(top["ob_error"])
        if ask is not None and bid is not None and ask < bid:
            bad.append(f"crossed ask {ask} < bid {bid}")
        if age is None:
            bad.append("no parsable ob_ts")
        elif not -5 <= age <= MAX_BOOK_AGE:
            bad.append(f"book age {age}s")
        if ask is not None and aq is None:
            bad.append("ask1_qty invalid")
        if bid is not None and bq is None:
            bad.append("bid1_qty invalid")
        if bad:
            return None, None, age, bad
        return ask, bid, age, []

    def log(self, kind, **kw):
        row = dict(ts=datetime.now(KST).isoformat(timespec="seconds"), kind=kind, **kw)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        print(json.dumps(row, ensure_ascii=False, default=str)[:300], flush=True)

    def candles(self, sym, interval, count, before=None):
        self.pacer.wait()
        return self.t.candles(sym, interval, count=count, before=before)["candles"]

    def local(self, ts):
        return pd.Timestamp(ts).tz_convert(self.tz)

    def bar_index(self, ts_end):
        t = self.local(ts_end)
        if t.date() != self.day:
            return -1
        m = t.hour * 60 + t.minute - self.cfg["open_min"] - 1
        return m // 5 if 0 <= m < self.B * 5 else -1

    def load_daily(self):
        ok = 0
        prev_day = None
        for j, s in enumerate(self.syms):
            try:
                c = self.candles(s, "1d", 3)
            except ApiError:
                continue
            prev = [x for x in c if self.local(x["timestamp"]).date() < self.day]
            if not prev:
                continue
            self.pturn[j] = float(prev[0]["closePrice"]) * float(prev[0]["volume"])
            self.pc[j] = float(prev[0]["closePrice"])
            prev_day = prev_day or self.local(prev[0]["timestamp"]).date()
            ok += 1
        if self.mk == "kr" and prev_day:
            fixed = 0
            for j, s in enumerate(self.syms):
                if self.pturn[j] < self.cfg["min_turn"]:
                    continue
                try:
                    c = self.candles(s, "1m", 1, before=f"{prev_day}T15:31:00+09:00")
                except ApiError:
                    continue
                if c and self.local(c[0]["timestamp"]).date() == prev_day:
                    self.pc[j] = float(c[0]["closePrice"])
                    fixed += 1
            self.log("daily_kr_close", from_1531_bar=fixed)
        self.log("daily", symbols=self.N, with_prev_close=ok, elig=int((self.pturn >= self.cfg["min_turn"]).sum()))

    def ingest(self, j, bars):
        now = datetime.now(self.tz)
        for x in sorted(bars, key=lambda x: x["timestamp"]):
            if self.local(x["timestamp"]) > now:
                continue
            key = (j, x["timestamp"])
            if key in self.seen:
                continue
            self.seen.add(key)
            b = self.bar_index(x["timestamp"])
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

    def refresh(self, count, before=None):
        for j, s in enumerate(self.syms):
            if self.pturn[j] < self.cfg["min_turn"]:
                continue
            try:
                self.ingest(j, self.candles(s, "1m", count, before))
            except ApiError:
                continue

    def data(self):
        D = 2
        z = lambda: np.full((D, self.B, self.N), np.nan)
        O, H_, L, C, V = z(), z(), z(), z(), np.zeros((D, self.B, self.N))
        O[1], H_[1], L[1], C[1], V[1] = self.O, self.H, self.L, self.C, self.V
        bar0 = (np.arange(self.B) == 0)[None, :, None]
        has = ~np.isnan(C) & ((V > 0) | (bar0 & ~np.isnan(O)))
        Cf = C.copy()
        Cf[1] = pd.DataFrame(C[1]).ffill().values
        pc = np.zeros((D, self.N))
        pc[1] = self.pc
        turn = np.zeros((D, self.N))
        turn[0] = self.pturn
        arrays = dict(O=O, H=H_, L=L, C=Cf, V=V, has=has, pc=pc, oc=np.nan_to_num(Cf[:, -1, :]), turn=turn, days=np.array(["prev", str(self.day)]), syms=np.array(self.syms))
        return Data(R.DEV, self.cfg["iv"], arrays)

    def score(self, b):
        X = self.data()
        hor = R.horizon(X, H)
        f = R.feats(X, [1], hor)
        if self.model is None:
            self.model = R.Reader(f.shape[-1]).to(X.dev)
            self.model.load_state_dict(torch.load(os.path.join(HERE, self.model_path), map_location=X.dev))
            self.model.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.model(f[0, b]))[:, 0]
        mask = X.buyable[1, b] & (b + 1 <= X.entry_last) & (X.C[1, b] >= MIN_PX)
        sc = p.masked_fill(~mask, 0)
        top = torch.topk(sc, 5)
        return X, p, sc, [(self.syms[int(i)], round(float(v), 3)) for v, i in zip(top.values, top.indices)]

    def book_top(self, s):
        req = round(time.time(), 3)
        try:
            ob = self.t.orderbook(s)
            a, b = (ob.get("asks") or [{}])[0], (ob.get("bids") or [{}])[0]
            return dict(ob_ts=ob.get("timestamp"), ask1=a.get("price"), ask1_qty=a.get("volume"), bid1=b.get("price"), bid1_qty=b.get("volume"), ob_req=req, ob_rec=round(time.time(), 3))
        except Exception as e:
            return dict(ob_error=f"{type(e).__name__}: {str(e)[:60]}")

    def bill_ok(self, j, b):
        s = self.syms[j]
        last = float(self.C[b, j]) if not np.isnan(self.C[b, j]) else None
        try:
            ob = self.t.orderbook(s)
            tr = self.t.get("/api/v1/trades", symbol=s, count=50)
        except ApiError as e:
            self.log("bill", symbol=s, ok=False, reasons=[f"api {e.code}"])
            return False
        qty = None
        if self.mk == "kr" and last:
            krw = self.cash if DRY else float(self.t.buying_power("KRW")["cashBuyingPower"])
            qty = int(np.floor(krw / (last * 1.02 * (1 + sim.FEE))))
        ok, reasons, m = bill.judge(self.mk, ob, tr, last, qty=qty)
        self.log("bill", symbol=s, bar=b, ok=ok, reasons=reasons, **m)
        return ok

    def buy(self, j, p):
        s = self.syms[j]
        last = float(self.C[self.last_b, j])
        if self.mk == "us":
            usd = self.cash if DRY else float(self.t.buying_power("USD")["cashBuyingPower"])
            amt = f"{max(0.0, np.floor(usd * 100 - 1) / 100):.2f}"
            if float(amt) < 0.5:
                self.log("skip_buy", reason="usd too small", usd=usd)
                return
            kw, size = dict(amount=amt), amt + "$"
        else:
            krw = self.cash if DRY else float(self.t.buying_power("KRW")["cashBuyingPower"])
            n = int(np.floor(krw / (last * 1.02 * (1 + sim.FEE))))
            if n < 1:
                self.log("skip_buy", reason="krw too small", krw=krw, last=last)
                return
            kw, size = dict(quantity=str(n)), f"{n}주"
        if DRY:
            dts = round(time.time(), 3)
            top = self.book_top(s)
            px, _, age, bad = self.quote(top)
            if px is None:
                self.skipped_quotes += 1
                self.log("skip_buy", symbol=s, reason="no valid ask quote: " + "; ".join(bad or ["ask1 missing"]), p=round(p, 3), last=last, bar=self.last_b, book_age_s=age, **top)
                return
            if self.mk == "us":
                qty, cost = float(amt) / px, float(amt)
            else:
                qty = min(n, int(np.floor(self.cash / (px * (1 + sim.FEE)))))
                cost = qty * px * (1 + sim.FEE)
                size = f"{qty}주"
                if qty < 1:
                    self.log("skip_buy", symbol=s, reason="krw too small at ask", krw=self.cash, ask=px, gate_qty=n)
                    return
            self.cash -= cost
            self.pos, self.entry, self.qty, self.cost, self.held_b = j, (last if self.entry_ref == "last" else px), qty, cost, self.last_b
            self.buys += 1
            self.log("dry_buy", symbol=s, p=round(p, 3), size=size, last=last, entry=self.entry, bar=self.last_b, decision_ts=dts, fill_px=px, book_age_s=age, qty=round(qty, 6), cost=round(cost, 4), cash=round(self.cash, 4), ledger=self.ledger_state(), **top)
            return
        r = self.t.place(s, "BUY", "MARKET", client_order_id="live-" + uuid.uuid4().hex[:12], **kw)
        o = self.wait(r["orderId"])
        ex = o["execution"]
        self.log("buy", symbol=s, bar=self.last_b, p=round(p, 3), size=size, last=last, status=o["status"], qty=ex["filledQuantity"], price=ex["averageFilledPrice"], commission=ex["commission"])
        if o["status"] == "FILLED" and float(ex["filledQuantity"]) > 0:
            self.pos, self.entry, self.qty = j, float(ex["averageFilledPrice"]), float(ex["filledQuantity"])
            self.buys += 1
        elif o["status"] == "PARTIAL_FILLED" or (ex["filledQuantity"] and float(ex["filledQuantity"]) > 0):
            self.pos, self.entry, self.qty = j, float(ex["averageFilledPrice"]), float(ex["filledQuantity"])
            self.buys += 1

    def sell(self, reason):
        s = self.syms[self.pos]
        last = float(self.C[self.last_b, self.pos]) if not np.isnan(self.C[self.last_b, self.pos]) else self.entry
        if DRY:
            dts = round(time.time(), 3)
            top = self.book_top(s)
            _, px, age, bad = self.quote(top)
            if px is None:
                self.skipped_quotes += 1
                self.log("dry_sell_skip", symbol=s, reason="no valid bid quote: " + "; ".join(bad or ["bid1 missing"]), intent=reason, last=last, entry=self.entry, bar=self.last_b, qty=round(self.qty, 6), book_age_s=age, **top)
                return
            gross = self.qty * px
            if self.mk == "us":
                fee = gross * float(os.environ.get("DRY_US_COMM", 0.0))
                tax = max(sim.US_MIN_FEE, round(gross * sim.US_SEC, 2))
            else:
                fee = gross * sim.FEE
                tax = gross * sim.TAX
            proceeds = gross - fee - tax
            pnl = proceeds - self.cost
            self.cash += proceeds
            self.realized += pnl
            qty, cost, entry = self.qty, self.cost, self.entry
            self.pos, self.qty, self.cost, self.entry = None, 0.0, 0.0, 0.0
            self.log("dry_sell", symbol=s, reason=reason, last=last, entry=entry, bar=self.last_b, decision_ts=dts, fill_px=px, book_age_s=age, qty=round(qty, 6), gross=round(gross, 4), fee=round(fee, 4), tax=round(tax, 4), proceeds=round(proceeds, 4), cost=round(cost, 4), pnl=round(pnl, 4), cash=round(self.cash, 4), realized=round(self.realized, 4), ledger=self.ledger_state(), **top)
            return
        sq = self.t.get("/api/v1/sellable-quantity", account=True, symbol=s)["sellableQuantity"]
        if float(sq) <= 0:
            self.log("sell_skip", symbol=s, reason="no sellable qty")
            self.pos = None
            return
        r = self.t.place(s, "SELL", "MARKET", quantity=sq, client_order_id="live-" + uuid.uuid4().hex[:12])
        o = self.wait(r["orderId"])
        ex = o["execution"]
        pnl = (float(ex["filledAmount"] or 0) - float(ex["commission"] or 0) - float(ex["tax"] or 0)) - self.entry * self.qty
        self.log("sell", symbol=s, reason=reason, status=o["status"], qty=ex["filledQuantity"], price=ex["averageFilledPrice"], commission=ex["commission"], tax=ex["tax"], entry=self.entry, last=last, pnl=round(pnl, 4))
        if o["status"] in ("FILLED",) or float(ex["filledQuantity"] or 0) >= float(sq):
            self.pos = None

    def adopt(self):
        want = "US" if self.mk == "us" else "KR"
        ign = set(filter(None, os.environ.get("LIVE_IGNORE", "").split(",")))
        ign_path = os.path.join(HERE, "runs", f"ignore_{self.mk}.txt")
        if os.path.exists(ign_path):
            ign |= set(open(ign_path, encoding="utf-8").read().split())
        for it in self.t.holdings()["items"]:
            if it["marketCountry"] != want or float(it["quantity"]) <= 0 or it["symbol"] not in self.syms:
                continue
            if it["symbol"] in ign:
                self.log("ignore_holding", symbol=it["symbol"], qty=float(it["quantity"]), entry=float(it["averagePurchasePrice"]))
                continue
            self.pos, self.entry, self.qty = self.syms.index(it["symbol"]), float(it["averagePurchasePrice"]), float(it["quantity"])
            hb, last_scan = None, None
            if os.path.exists(self.log_path):
                for line in open(self.log_path, encoding="utf-8"):
                    r = json.loads(line)
                    if r.get("kind") in ("scan", "hold") and "bar" in r:
                        last_scan = r["bar"]
                    if r.get("kind") == "buy" and r.get("symbol") == it["symbol"]:
                        hb = r.get("bar", last_scan)
            self.held_b = hb if hb is not None else self.last_b
            self.buys += 1
            self.log("adopt", symbol=it["symbol"], qty=self.qty, entry=self.entry, held_b=self.held_b)
            return

    def wait(self, oid, timeout=90):
        t0 = time.time()
        while time.time() - t0 < timeout:
            o = self.t.order(oid)
            if o["status"] in ("FILLED", "CANCELED", "REJECTED"):
                return o
            time.sleep(2)
        return self.t.order(oid)

    def run(self):
        import hashlib
        try:
            sha = hashlib.sha256(open(os.path.join(HERE, self.model_path), "rb").read()).hexdigest()
        except OSError:
            sha = None
        sess = sim.SESSION[("us" if self.mk == "us" else "kr", "5m")]
        self.log("start", market=self.mk, dry=DRY, thr=self.thr, model=self.model_path, model_sha256=sha, universe=self.N, universe_path=os.environ.get("LIVE_UNIVERSE", self.cfg["universe"]), min_px=MIN_PX, day=str(self.day), h=H, stop=R.STOP, tp=TP, exit_bar=self.cfg["exit_bar"], data_entry_last=int(sess[0]), data_exit_bar=int(sess[1]), us_exit_env=os.environ.get("DOBBY_US_EXIT"), rps=RPS, pid=os.getpid(), dry_cash=self.cash, dry_cash_src=self.cash_src, dry_entry_ref=self.entry_ref, dry_max_book_age=MAX_BOOK_AGE, dry_resume=self.resume_info)
        self.load_daily()
        self.refresh(200, before=f"{self.day}T{(self.cfg['open_min'] + self.B * 5) // 60:02d}:{(self.cfg['open_min'] + self.B * 5) % 60:02d}:00+09:00" if self.mk == "kr" and os.environ.get("LIVE_DAY") else None)
        if self.mk == "kr" and os.environ.get("LIVE_DAY"):
            self.refresh(200, before=f"{self.day}T12:11:00+09:00")
        now = datetime.now(self.tz)
        self.last_b = max(0, min((now.hour * 60 + now.minute - self.cfg["open_min"]) // 5 - 1, self.B - 1))
        if not os.environ.get("LIVE_DAY"):
            if DRY:
                self.log("adopt_skipped", reason="dry ledger is independent of real holdings", resumed_pos=None if self.pos is None else self.syms[self.pos], qty=round(self.qty, 6), cash=round(self.cash, 4))
            else:
                self.adopt()
        while True:
            now = datetime.now(self.tz)
            m = now.hour * 60 + now.minute - self.cfg["open_min"]
            if os.environ.get("LIVE_DAY"):
                m = int(os.environ.get("LIVE_BAR", 30)) * 5 + 5
            if m < 5:
                time.sleep(30)
                continue
            b = min(m // 5 - 1, self.B - 1)
            self.last_b = b
            stop_file = os.path.exists(os.path.join(HERE, "runs", "STOP"))
            if b >= self.cfg["exit_bar"] or stop_file:
                if self.pos is not None:
                    self.sell("forced_exit" if not stop_file else "stop_file")
                if DRY and self.pos is not None:
                    c = float(self.C[b, self.pos]) if not np.isnan(self.C[b, self.pos]) else None
                    self.log("dry_unsettled", symbol=self.syms[self.pos], qty=round(self.qty, 6), cost=round(self.cost, 4), entry=self.entry, last=c, bar=b)
                self.log("end", bar=b, buys=self.buys, dry_cash=None if not DRY else round(self.cash, 4), dry_realized=None if not DRY else round(self.realized, 4), dry_unsettled=None if not DRY else (self.pos is not None), dry_skipped_quotes=None if not DRY else self.skipped_quotes)
                return
            X, p, sc, top = self.score(b)
            if self.pos is not None:
                c = float(self.C[b, self.pos]) if not np.isnan(self.C[b, self.pos]) else self.entry
                act = "time_exit" if b - self.held_b >= H else ("stop" if c <= self.entry * (1 - R.STOP) else ("tp" if TP > 0 and c >= self.entry * (1 + TP) else None))
                self.log("hold", bar=b, symbol=self.syms[self.pos], last=c, entry=self.entry, upnl_pct=round((c / self.entry - 1) * 100, 3), top=top)
                if act:
                    self.sell(act)
            else:
                j = int(sc.argmax())
                self.log("scan", bar=b, best=self.syms[j], p=round(float(sc[j]), 3), n_buyable=int((sc > 0).sum()), top=top)
                if self.buys < 4:
                    for v, i in zip(*[x.tolist() for x in torch.topk(sc, 3)]):
                        if v <= self.thr:
                            break
                        j = int(i)
                        if self.bill_ok(j, b):
                            self.held_b = b
                            self.buy(j, float(v))
                            break
            if ONCE:
                self.log("once", p_mean=round(float(p.mean()), 4), p_max=round(float(p.max()), 4), feat_dim=int(R.feats(X, [1], R.horizon(X, H)).shape[-1]), has_bars=int(X.has[1].sum()), buyable_now=int((sc > 0).sum()))
                return
            nxt = (now + timedelta(minutes=5 - (now.minute % 5))).replace(second=12, microsecond=0)
            time.sleep(max(1, (nxt - datetime.now(self.tz)).total_seconds()))
            self.refresh(12)


if __name__ == "__main__":
    Live(sys.argv[1] if len(sys.argv) > 1 else "us").run()
