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
    out = dict(result=z, stop_loss=dict(n=0, gap_open_fill=0, at_stop=0, other=0, examples=[]), roi=dict(n=0, gap_open_fill=0, at_roi_price=0, other=0, examples=[]), entries=dict(n=0, at_bar_open=0, other=0))
    for t in st["trades"]:
        pair = t["pair"]
        if pair not in bars:
            df = pd.read_feather(os.path.join(datadir, pair.replace("/", "_") + f"-{tf}.feather"))
            df["date"] = pd.to_datetime(df["date"], utc=True)
            bars[pair] = df.set_index("date")
        b = bars[pair]
        od = pd.Timestamp(t["open_date"]).tz_convert("UTC") if pd.Timestamp(t["open_date"]).tzinfo else pd.Timestamp(t["open_date"]).tz_localize("UTC")
        cd = pd.Timestamp(t["close_date"]).tz_convert("UTC") if pd.Timestamp(t["close_date"]).tzinfo else pd.Timestamp(t["close_date"]).tz_localize("UTC")
        out["entries"]["n"] += 1
        if abs(t["open_rate"] - b.loc[od, "open"]) < 1e-9:
            out["entries"]["at_bar_open"] += 1
        else:
            out["entries"]["other"] += 1
        bar = b.loc[cd]
        if t["exit_reason"] == "stop_loss":
            s = out["stop_loss"]
            s["n"] += 1
            stop = t["stop_loss_abs"]
            if bar["open"] < stop - 1e-12 and abs(t["close_rate"] - bar["open"]) < 1e-9:
                s["gap_open_fill"] += 1
                if len(s["examples"]) < 3:
                    s["examples"].append(dict(pair=pair, close_date=str(cd), bar_open=float(bar["open"]), stop=stop, close_rate=t["close_rate"], profit_ratio=t["profit_ratio"]))
            elif abs(t["close_rate"] - stop) < 1e-6 * max(1.0, stop):
                s["at_stop"] += 1
            else:
                s["other"] += 1
        elif t["exit_reason"] == "roi":
            r = out["roi"]
            r["n"] += 1
            if abs(t["close_rate"] - bar["open"]) < 1e-9 and bar["open"] > t["open_rate"]:
                r["gap_open_fill"] += 1
            elif bar["low"] - 1e-9 <= t["close_rate"] <= bar["high"] + 1e-9:
                r["at_roi_price"] += 1
            else:
                r["other"] += 1
                if len(r["examples"]) < 3:
                    r["examples"].append(dict(pair=pair, close_date=str(cd), close_rate=t["close_rate"], bar=dict(open=float(bar["open"]), high=float(bar["high"]), low=float(bar["low"]))))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
