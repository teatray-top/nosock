import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def h(f):
    return hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest()


def main(manifest_path, label):
    m = json.load(open(os.path.join(HERE, manifest_path), encoding="utf-8"))
    bad = [f for f, want in m["files"].items() if h(f) != want]
    lf = hashlib.sha256(open(os.path.join(HERE, "user_data/strategies/NostalgiaForInfinityX8.py"), "rb").read().replace(b"\r\n", b"\n")).hexdigest()
    if lf != m["strategy_lf_sha256"]:
        bad.append("strategy LF hash")
    rec = dict(label=label, manifest=manifest_path, manifest_sha256=h(manifest_path), files_checked=len(m["files"]), mismatches=bad)
    print(json.dumps(rec))
    if bad:
        sys.exit(3)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
