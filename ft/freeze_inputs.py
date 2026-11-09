import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GROUPS = {
    "kr_data": ["user_data/data_kr_rt/binance/*"],
    "kr_source_snapshot": ["user_data/data_kr_c/source_snapshot/*.csv"],
    "binance_data": ["user_data/data/binance/*.feather"],
    "strategies": ["user_data/strategies/*.py"],
    "code_and_config": ["kr_to_ft.py", "ft_nofill.py", "check_trade_times.py", "check_exit_prices.py", "sweep_summary.py", "freeze_inputs.py", "map_kr_top10_rt.json", "config_sweep_binance.json", "config_sweep_kr.json", "sweep_strategies_v1.txt"],
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect():
    out = {}
    for g, pats in GROUPS.items():
        files = sorted({os.path.relpath(p, HERE).replace("\\", "/") for pat in pats for p in glob.glob(os.path.join(HERE, pat)) if os.path.isfile(p)})
        out[g] = {f: sha(os.path.join(HERE, f)) for f in files}
    return out


def main(cmd, path):
    if cmd == "freeze":
        if os.path.exists(path):
            raise SystemExit("frozen manifest exists")
        m = collect()
        with open(path, "x", encoding="utf-8") as f:
            json.dump(dict(version="frozen-inputs-v1", groups=m, counts={g: len(v) for g, v in m.items()}), f, indent=1)
        print("frozen", {g: len(v) for g, v in m.items()}, sha(path)[:8])
    elif cmd == "verify":
        want = json.load(open(path, encoding="utf-8"))["groups"]
        have = collect()
        bad = []
        for g in want:
            for f, h in want[g].items():
                if have.get(g, {}).get(f) != h:
                    bad.append(f)
            for f in have.get(g, {}):
                if f not in want[g] and g != "code_and_config":
                    bad.append("unexpected " + f)
        if bad:
            print("FROZEN INPUT MISMATCH", bad[:20])
            sys.exit(3)
        print("frozen inputs verified", {g: len(v) for g, v in want.items()})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
