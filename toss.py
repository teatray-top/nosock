import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import cred

BASE = "https://openapi.tossinvest.com"
TOKEN_FILE = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "penny", "toss_token.json")
CRED_TARGET = os.environ.get("TOSS_CRED", "TossOpenAPI")


class ApiError(Exception):
    def __init__(self, status, code, message, request_id=None, data=None):
        super().__init__(f"{status} {code}: {message} (requestId {request_id})")
        self.status, self.code, self.message, self.request_id, self.data = status, code, message, request_id, data


class _Lock:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.f = open(self.path, "a+")
        if os.name == "nt":
            import msvcrt
            self.f.seek(0)
            while True:
                try:
                    msvcrt.locking(self.f.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    time.sleep(0.2)
        else:
            import fcntl
            fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if os.name == "nt":
            import msvcrt
            self.f.seek(0)
            msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


class Toss:
    def __init__(self, account=None):
        self.client_id, self.client_secret = cred.load(CRED_TARGET)
        self.account = account
        self.tok, self.exp = None, 0
        self._load_token()

    def _load_token(self):
        try:
            t = json.load(open(TOKEN_FILE, encoding="utf-8"))
            if t.get("client_id") == self.client_id and t.get("exp", 0) > time.time() + 30:
                self.tok, self.exp = t["access_token"], t["exp"]
        except (OSError, ValueError, KeyError):
            pass

    def _issue(self):
        stale = self.tok
        with _Lock(TOKEN_FILE + ".lock"):
            self._load_token()
            if self.tok and self.tok != stale:
                return
            body = urllib.parse.urlencode(dict(grant_type="client_credentials", client_id=self.client_id, client_secret=self.client_secret)).encode()
            req = urllib.request.Request(BASE + "/oauth2/token", data=body, headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=20) as r:
                    t = json.loads(r.read())
            except urllib.error.HTTPError as e:
                raise ApiError(e.code, "oauth", e.read().decode("utf-8", "ignore")[:300])
            self.tok, self.exp = t["access_token"], time.time() + int(t.get("expires_in", 3600))
            tmp = TOKEN_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(dict(client_id=self.client_id, access_token=self.tok, exp=self.exp), f)
            os.replace(tmp, TOKEN_FILE)

    def _req(self, method, path, params=None, body=None, account=False, retry=3):
        if not self.tok or self.exp < time.time() + 30:
            self._issue()
        url = BASE + path + (("?" + urllib.parse.urlencode(params, doseq=True)) if params else "")
        headers = {"Authorization": f"Bearer {self.tok}", "Accept": "application/json"}
        if account:
            if self.account is None:
                self.account = self.accounts()[0]["accountSeq"]
            headers["X-Tossinvest-Account"] = str(self.account)
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "ignore")
            try:
                err = json.loads(raw).get("error", {})
            except ValueError:
                err = {"code": "http", "message": raw[:300]}
            code = err.get("code", "http")
            if e.code == 401 and code in ("expired-token", "invalid-token", "token-revoked") and retry > 0:
                old = self.tok
                self._load_token()
                if self.tok == old:
                    self._issue()
                return self._req(method, path, params, body, account, retry - 1)
            if e.code == 429 and retry > 0:
                time.sleep(float(e.headers.get("Retry-After", 1)) + 0.2)
                return self._req(method, path, params, body, account, retry - 1)
            raise ApiError(e.code, code, err.get("message", ""), err.get("requestId") or e.headers.get("X-Request-Id"), err.get("data"))

    def get(self, path, account=False, **params):
        return self._req("GET", path, {k: v for k, v in params.items() if v is not None}, account=account)["result"]

    def accounts(self):
        return self._req("GET", "/api/v1/accounts")["result"]

    def buying_power(self, currency):
        return self.get("/api/v1/buying-power", account=True, currency=currency)

    def holdings(self, symbol=None):
        return self.get("/api/v1/holdings", account=True, symbol=symbol)

    def commissions(self):
        return self.get("/api/v1/commissions", account=True)

    def fx(self, base="USD", quote="KRW"):
        return self.get("/api/v1/exchange-rate", baseCurrency=base, quoteCurrency=quote)

    def prices(self, symbols):
        return self.get("/api/v1/prices", symbols=",".join(symbols))

    def orderbook(self, symbol):
        return self.get("/api/v1/orderbook", symbol=symbol)

    def candles(self, symbol, interval="1m", count=200, before=None, adjusted=False):
        return self.get("/api/v1/candles", symbol=symbol, interval=interval, count=count, before=before, adjusted=str(adjusted).lower())

    def stocks(self, symbols):
        return self.get("/api/v1/stocks", symbols=",".join(symbols))

    def stocks_all(self, market, status=None, security_type=None, common=None):
        return self.get("/api/v1/stocks/all", market=market, status=status, securityType=security_type, commonShare=None if common is None else str(common).lower())

    def warnings(self, symbol):
        return self.get(f"/api/v1/stocks/{symbol}/warnings")

    def calendar(self, market="KR", date=None):
        return self.get(f"/api/v1/market-calendar/{market}", date=date)

    def orders(self, status="OPEN", **kw):
        return self.get("/api/v1/orders", account=True, status=status, **kw)

    def order(self, order_id):
        return self.get(f"/api/v1/orders/{order_id}", account=True)

    def place(self, symbol, side, order_type, quantity=None, price=None, amount=None, client_order_id=None, tif=None):
        if os.environ.get("DOBBY_LIVE") != "1":
            raise PermissionError("live orders disabled: set DOBBY_LIVE=1")
        body = dict(symbol=symbol, side=side, orderType=order_type)
        if client_order_id:
            body["clientOrderId"] = client_order_id
        if amount is not None:
            body["orderAmount"] = str(amount)
        else:
            body["quantity"] = str(quantity)
            if price is not None:
                body["price"] = str(price)
            if tif:
                body["timeInForce"] = tif
        return self._req("POST", "/api/v1/orders", body=body, account=True)["result"]

    def cancel(self, order_id):
        if os.environ.get("DOBBY_LIVE") != "1":
            raise PermissionError("live orders disabled: set DOBBY_LIVE=1")
        return self._req("POST", f"/api/v1/orders/{order_id}/cancel", body={}, account=True)["result"]


def public_ip():
    try:
        return urllib.request.urlopen("https://api.ipify.org", timeout=5).read().decode()
    except Exception as e:
        return f"? ({e})"


def check():
    t = Toss()
    try:
        acc = t.accounts()
    except ApiError as e:
        if e.code in ("ip-not-allowed", "access_denied") or e.status == 403:
            print(f"403 {e.code}: 허용 IP 미등록. 이 PC 공인 IP: {public_ip()}")
            return
        raise
    print("accounts:", [(a["accountSeq"], a["accountType"], a["accountNo"][:3] + "***") for a in acc])
    t.account = acc[0]["accountSeq"]
    for c in ("KRW", "USD"):
        print(f"buying power {c}:", t.buying_power(c)["cashBuyingPower"])
    print("commissions:", [(c["marketCountry"], c["commissionRate"], c.get("startDate"), c.get("endDate")) for c in t.commissions()])
    fx = t.fx("USD", "KRW")
    print(f"fx USD/KRW rate {fx['rate']} mid {fx['midRate']} bp {fx['basisPoint']} valid {fx['validFrom']}~{fx['validUntil']}")
    h = t.holdings()
    print("holdings:", len(h["items"]), "items, market value", h["marketValue"]["amount"])
    print("prices:", [(p["symbol"], p["lastPrice"], p["timestamp"]) for p in t.prices(["005930", "AAPL"])])
    for sym in ("005930", "AAPL"):
        c = t.candles(sym, "1m", count=5)["candles"]
        print(f"candles {sym} 1m latest 5:", [(x["timestamp"][5:16], x["closePrice"], x["volume"]) for x in c])
    d = t.candles("005930", "1d", count=3)["candles"]
    print("candles 005930 1d:", [(x["timestamp"][:10], x["closePrice"], x["volume"]) for x in d])
    cal = t.calendar("KR")["today"]
    print("KR calendar today:", cal["date"], json.dumps(cal.get("integrated"), ensure_ascii=False))
    print("open orders:", len(t.orders("OPEN")["orders"]))


if __name__ == "__main__":
    if sys.argv[1:] == ["check"]:
        check()
