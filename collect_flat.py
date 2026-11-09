import os
import sys
import time
from datetime import date, datetime

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError

import cred

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "massive")
DS = {"minute": "us_stocks_sip/minute_aggs_v1", "day": "us_stocks_sip/day_aggs_v1"}
XFER = TransferConfig(max_concurrency=4)


def log(*a):
    msg = datetime.now().isoformat(timespec="seconds") + " " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(os.path.join(ROOT, "collect.log"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def client():
    ak, sk = cred.load("MassiveS3")
    return boto3.client("s3", endpoint_url="https://files.massive.com", aws_access_key_id=ak, aws_secret_access_key=sk, config=Config(signature_version="s3v4", retries={"max_attempts": 8, "mode": "adaptive"}))


def listing(s3, ds, y, m):
    out, tok = [], None
    while True:
        kw = dict(Bucket="flatfiles", Prefix=f"{DS[ds]}/{y}/{m:02d}/")
        if tok:
            kw["ContinuationToken"] = tok
        r = s3.list_objects_v2(**kw)
        out += [(o["Key"], o["Size"]) for o in r.get("Contents", [])]
        if not r.get("IsTruncated"):
            return out
        tok = r["NextContinuationToken"]


def main(ds, start, end=None):
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end) if end else date.today()
    out = os.path.join(ROOT, "flat", ds)
    os.makedirs(out, exist_ok=True)
    s3 = client()
    months, (y, m) = [], (d1.year, d1.month)
    while (y, m) >= (d0.year, d0.month):
        months.append((y, m))
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    n, nb, skipped, t0 = 0, 0, 0, time.time()
    log(f"{ds} {d0}..{d1}, {len(months)} months, newest first")
    for y, m in months:
        for key, size in sorted(listing(s3, ds, y, m), reverse=True):
            day = key.split("/")[-1][:10]
            if not (str(d0) <= day <= str(d1)):
                continue
            path = os.path.join(out, key.split("/")[-1])
            if os.path.exists(path) and os.path.getsize(path) == size:
                skipped += 1
                continue
            tmp = path + ".part"
            for attempt in range(5):
                try:
                    s3.download_file("flatfiles", key, tmp, Config=XFER)
                    break
                except ClientError as e:
                    code = e.response.get("Error", {}).get("Code")
                    if code in ("403", "AccessDenied"):
                        log(f"{key}: {code}, plan window reached, stopping")
                        log(f"{ds} done {n} files {nb / 1e9:.2f} GB, {skipped} already present, {time.time() - t0:.0f}s")
                        return
                    log(f"{key} attempt {attempt} {code}: {str(e)[:120]}")
                    time.sleep(10 * (attempt + 1))
                except Exception as e:
                    log(f"{key} attempt {attempt} {type(e).__name__}: {str(e)[:120]}")
                    time.sleep(10 * (attempt + 1))
            else:
                continue
            os.replace(tmp, path)
            n += 1
            nb += size
            if n % 20 == 0:
                log(f"{ds} {day}: {n} files {nb / 1e9:.2f} GB {nb / 1e6 / (time.time() - t0):.1f} MB/s")
    log(f"{ds} done {n} files {nb / 1e9:.2f} GB, {skipped} already present, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
