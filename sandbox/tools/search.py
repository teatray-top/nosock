import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8")
rows = [json.loads(x) for x in open(os.path.join(HERE, "news.jsonl"), encoding="utf-8")]
args = sys.argv[1:]
if args[:1] == ["--day"] and len(args) > 1:
    hits = [r for r in rows if r["date"] == args[1]]
else:
    keys = [a.lower() for a in args]
    hits = [r for r in rows if not keys or any(k in (r["title"] + " " + r["body"]).lower() for k in keys)]
    hits = sorted(hits, key=lambda r: (-sum(k in (r["title"] + " " + r["body"]).lower() for k in keys), -r["day"]))[:20]
    hits = sorted(hits, key=lambda r: -r["day"])
for r in hits:
    print(f"[{r['date']}] ({r['outlet']}) {r['title']}\n  {r['body']}")
print(f"-- {len(hits)}건")
