import json
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ft"))
import toss
import msma_v1 as M1

OUT = os.path.join(HERE, "data", "toss_checks")


def main():
    t = toss.Toss()
    if not t.tok or t.exp < time.time() + 120:
        print("cached token missing or expiring; not issuing")
        return 1

    def refuse():
        raise RuntimeError("token issue refused")

    t._issue = refuse
    codes = list(M1.CLASS)
    rec = dict(fetched_at=datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"), codes=codes)
    try:
        rec["stocks"] = t.stocks(codes)
    except Exception as e:
        rec["stocks_error"] = type(e).__name__ + " " + str(getattr(e, "code", ""))
    rec["warnings"] = {}
    for c in codes:
        try:
            rec["warnings"][c] = t.warnings(c)
        except Exception as e:
            rec["warnings"][c] = dict(error=type(e).__name__ + " " + str(getattr(e, "code", "")))
        time.sleep(0.2)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"etf_tradability_{rec['fetched_at'][:19].replace(':', '').replace('-', '')}.json")
    with open(p, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print("saved", p)
    print(json.dumps(rec, ensure_ascii=False)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
