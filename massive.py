import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import cred

BASE = "https://api.massive.com"
RPM = float(os.environ.get("MASSIVE_RPM", 4.8))


def key():
    k = os.environ.get("MASSIVE_API_KEY")
    if k:
        return k
    return cred.load("MassiveAPI")[1]


class Client:
    def __init__(self, rpm=RPM):
        self.k = key()
        self.dt, self.t = 60.0 / rpm, 0.0

    def get(self, url, params=None):
        if url.startswith("/"):
            url = BASE + url
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
        for attempt in range(8):
            now = time.time()
            if now < self.t + self.dt:
                time.sleep(self.t + self.dt - now)
            self.t = time.time()
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.k}"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                body = e.read()
                try:
                    j = json.loads(body)
                except Exception:
                    j = {"raw": body[:200].decode("utf-8", "replace")}
                if e.code == 429:
                    time.sleep(61)
                    continue
                if e.code >= 500:
                    time.sleep(10 * (attempt + 1))
                    continue
                raise ApiError(e.code, j)
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(10 * (attempt + 1))
        raise ApiError(0, {"message": "retries exhausted", "url": url})

    def pages(self, url, params=None):
        j = self.get(url, params)
        yield j
        while j.get("next_url"):
            j = self.get(j["next_url"])
            yield j

    def minute_bars(self, ticker, frm, to):
        out = []
        for j in self.pages(f"/v2/aggs/ticker/{ticker}/range/1/minute/{frm}/{to}", dict(adjusted="false", sort="asc", limit=50000)):
            out += j.get("results") or []
        return out

    def grouped_daily(self, date):
        j = self.get(f"/v2/aggs/grouped/locale/us/market/stocks/{date}", dict(adjusted="false", include_otc="false"))
        return j.get("results") or []


class ApiError(Exception):
    def __init__(self, status, body):
        super().__init__(f"{status} {str(body)[:200]}")
        self.status, self.body = status, body
