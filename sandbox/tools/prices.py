import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8")
rows = list(csv.DictReader(open(os.path.join(HERE, "prices.csv"), encoding="utf-8")))
code = sys.argv[1] if len(sys.argv) > 1 else None
n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
if not code or code not in rows[0]:
    print("종목코드:", ", ".join(k for k in rows[0] if k != "date"))
    sys.exit(0)
for r in rows[-n:]:
    print(r["date"], r[code])
