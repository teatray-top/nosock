import hashlib
import json
import os
import sys

import shutil

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_STOCKS = os.path.join(HERE, "..", "data", "toss", "1m")


STEP = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "12h": "12h", "1d": "1D"}
COMPRESSED_BASE = "2020-01-01"


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_1m(path):
    df = pd.read_csv(path)
    df["end"] = pd.to_datetime(df["ts"]).dt.tz_localize("Asia/Seoul")
    df["date"] = (df["end"] - pd.Timedelta(minutes=1)).dt.tz_convert("UTC")
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    return df[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def resample(df, tf):
    if tf == "1m":
        return df.copy()
    rule = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "12h": "12h", "1d": "1D"}[tf]
    g = df.set_index("date")
    out = g.resample(rule, origin="epoch", label="left", closed="left").agg(dict(open="first", high="max", low="min", close="last", volume="sum")).dropna(subset=["open"])
    out = out.reset_index().rename(columns={"index": "date"})
    return out


def main(mapping_path, tfs, out_dir):
    mapping = json.load(open(mapping_path, encoding="utf-8"))
    os.makedirs(out_dir, exist_ok=True)
    manifest = dict(mapping_file=os.path.abspath(mapping_path), mapping_sha256=sha(mapping_path), price_scale=mapping.get("price_scale", 1.0), time_axis=mapping.get("time_axis", "real"), compressed_base=COMPRESSED_BASE, timeframes=tfs, convention="source 1m rows are labelled by bar end (KST); converted to bar start in UTC; resampled with left-closed, left-labelled buckets; the 15:30 KST bucket holds the closing-auction print; no forward fill, no synthetic bars", files=[])
    snap_dir = mapping.get("snapshot_dir") or os.path.join(os.path.dirname(os.path.abspath(out_dir)), "source_snapshot")
    os.makedirs(snap_dir, exist_ok=True)
    win = mapping.get("window_kst")
    manifest["window_kst"] = win
    for item in mapping["pairs"]:
        live = os.path.join(SRC_STOCKS, item["source"]) if not os.path.isabs(item["source"]) else item["source"]
        src = os.path.join(snap_dir, os.path.basename(live))
        if not os.path.exists(src):
            shutil.copyfile(live, src)
        df = load_1m(src)
        if win:
            kst_day = df["date"].dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
            df = df[(kst_day >= win[0]) & (kst_day <= win[1])].reset_index(drop=True)
        scale = float(mapping.get("price_scale", 1.0))
        for c in ("open", "high", "low", "close"):
            df[c] = df[c] / scale
        for tf in tfs:
            out = resample(df, tf)
            name = item["pair"].replace("/", "_") + f"-{tf}.feather"
            path = os.path.join(out_dir, name)
            if mapping.get("time_axis") == "compressed":
                step = pd.Timedelta(STEP[tf])
                real = out["date"].copy()
                out["date"] = pd.Timestamp(COMPRESSED_BASE, tz="UTC") + step * pd.Series(range(len(out)))
                pd.DataFrame(dict(date=out["date"], real_date=real)).to_csv(os.path.join(out_dir, name.replace(".feather", ".timemap.csv")), index=False)
            out.to_feather(path)
            manifest["files"].append(dict(pair=item["pair"], symbol=item["symbol"], source=os.path.abspath(src), source_sha256=sha(src), timeframe=tf, file=name, rows=len(out), first=str(out["date"].iloc[0]), last=str(out["date"].iloc[-1])))
            print(item["symbol"], item["pair"], tf, len(out), out["date"].iloc[0], out["date"].iloc[-1])
    with open(os.path.join(out_dir, "kr_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","), sys.argv[3])
