import json
import os
import sys
import time
import urllib.request

import numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "edgar")
UA = os.environ["EDGAR_UA"]
FORMS = {"8-K", "8-K/A", "10-Q", "10-K", "10-Q/A", "10-K/A", "6-K", "S-1", "424B4", "SC 13D", "SC 13G", "4"}
RPS = 5.0


class Pacer:
    def __init__(self, rps):
        self.dt, self.t = 1.0 / rps, 0.0

    def wait(self):
        now = time.time()
        if now < self.t + self.dt:
            time.sleep(self.t + self.dt - now)
        self.t = time.time()


def get(url, pacer):
    for i in range(5):
        pacer.wait()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip, deflate", "Host": url.split("/")[2]})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    import gzip
                    data = gzip.decompress(data)
                return json.loads(data)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if i == 4:
                raise
            time.sleep(5 * 3 ** i)
        except OSError:
            if i == 4:
                raise
            time.sleep(5 * 3 ** i)


def rows_from(block, start):
    out = []
    n = len(block.get("accessionNumber", []))
    for i in range(n):
        form = block["form"][i]
        fd = block["filingDate"][i]
        if fd < start:
            continue
        if form not in FORMS:
            continue
        out.append(dict(form=form, filing_date=fd, accepted=block["acceptanceDateTime"][i], items=block.get("items", [""] * n)[i], accession=block["accessionNumber"][i], primary=block.get("primaryDocument", [""] * n)[i]))
    return out


def main(panel, start, tag):
    z = np.load(panel, allow_pickle=True)
    syms = [str(s) for s in z["syms"]]
    out_dir = os.path.join(D, tag)
    os.makedirs(out_dir, exist_ok=True)
    pacer = Pacer(RPS)
    mp_path = os.path.join(D, "company_tickers.json")
    if not os.path.exists(mp_path):
        json.dump(get("https://www.sec.gov/files/company_tickers.json", pacer), open(mp_path, "w"))
    mp = json.load(open(mp_path))
    cik = {v["ticker"].upper(): int(v["cik_str"]) for v in mp.values()}
    missing, n_rows, t0 = [], 0, time.time()
    for k, s in enumerate(syms):
        out = os.path.join(out_dir, s + ".jsonl")
        if os.path.exists(out):
            continue
        c = cik.get(s.upper().replace(".", "-"))
        if c is None:
            missing.append(s)
            continue
        sub = get(f"https://data.sec.gov/submissions/CIK{c:010d}.json", pacer)
        if sub is None:
            missing.append(s)
            continue
        rows = rows_from(sub["filings"]["recent"], start)
        rec = sub["filings"]["recent"]
        if rec.get("filingDate") and min(rec["filingDate"]) > start:
            for f in sub["filings"].get("files", []):
                if f.get("filingTo", "") >= start:
                    more = get("https://data.sec.gov/submissions/" + f["name"], pacer)
                    if more:
                        rows += rows_from(more, start)
        rows.sort(key=lambda r: r["accepted"])
        with open(out + ".part", "w", encoding="utf-8") as f:
            f.write(json.dumps(dict(ticker=s, cik=c, name=sub.get("name"), sic=sub.get("sic"), exchanges=sub.get("exchanges"))) + "\n")
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(out + ".part", out)
        n_rows += len(rows)
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(syms)} filings {n_rows} missing {len(missing)} {time.time() - t0:.0f}s", flush=True)
    json.dump(missing, open(os.path.join(out_dir, "_missing.json"), "w"))
    print(f"done {len(syms)} tickers, filings {n_rows}, missing cik {len(missing)}: {missing[:30]}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
