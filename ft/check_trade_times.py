import glob
import json
import os
import sys
import zipfile

import pandas as pd


def main(result_dir, datadir, strategy):
    z = sorted(glob.glob(os.path.join(result_dir, "*.zip")))[-1]
    with zipfile.ZipFile(z) as f:
        name = [n for n in f.namelist() if n.endswith(".json") and not n.endswith("_config.json")][0]
        st = json.loads(f.read(name))["strategy"][strategy]
    tf = st["timeframe"]
    bars = {}
    bad = []
    for t in st["trades"]:
        pair = t["pair"]
        if pair not in bars:
            df = pd.read_feather(os.path.join(datadir, pair.replace("/", "_") + f"-{tf}.feather"))
            bars[pair] = set(pd.to_datetime(df["date"], utc=True))
        for k in ("open_date", "close_date"):
            ts = pd.Timestamp(t[k])
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            if ts not in bars[pair]:
                bad.append(dict(pair=pair, field=k, ts=str(ts), exit_reason=t.get("exit_reason")))
    print(json.dumps(dict(result=z, timeframe=tf, trades=len(st["trades"]), fills_outside_input=len(bad), examples=bad[:5]), ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
