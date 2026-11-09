import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
import numpy as np

from sandbox import run as SR
from sandbox import world as W


def load_runs(out):
    best = {}
    for d in sorted(os.listdir(out)):
        parts = d.split("_")
        if len(parts) < 3 or not parts[0].startswith("w") or not os.path.exists(os.path.join(out, d, "report.json")):
            continue
        seed, agent = int(parts[0][1:]), parts[1]
        best[(seed, agent)] = os.path.join(out, d)
    return best


def rescore(path, seed, days):
    w = W.make_world(seed, days)
    cs = SR.codes()
    w0 = np.array([1 / len(SR.BASE) if c in SR.BASE else 0.0 for c in cs])
    decisions, logs = {}, {}
    for line in open(os.path.join(path, "decisions.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        js = r.get("agent")
        ok = js and not js.get("error")
        tw, _ = SR.tilt(js, w0, w, r["t"]) if ok else (w0.copy(), None)
        decisions[r["t"]] = tw
        logs[r["t"]] = js if ok else None
    sim = SR.simulate(w, decisions)
    return dict(tilt=sim["tilt"]["total"], base=sim["base"]["total"], tilt_dd=sim["tilt"]["maxdd"], base_dd=sim["base"]["maxdd"], brier=SR.brier(w, logs), decisions=len(decisions))


def main():
    out = os.environ.get("NOSOCK_SANDBOX_OUT", SR.OUT)
    runs = load_runs(out)
    rows = {}
    for (seed, agent), path in sorted(runs.items()):
        rep = json.load(open(os.path.join(path, "report.json"), encoding="utf-8"))
        rows.setdefault(agent, {})[seed] = rescore(path, seed, rep.get("days", 60))
    seeds = sorted(set(s for a in rows.values() for s in a))
    summary = dict(agents={}, seeds=seeds, note=f"P&L re-scored from logged agent outputs with the current optimizer (cap {SR.CAP}, vol risk budget, inverse harmonized, costs); brier: lower is better")
    for agent, d in rows.items():
        ex = [v["tilt"] - v["base"] for v in d.values()]
        ba = [v["brier"]["agent"] for v in d.values() if v["brier"]["agent"] is not None]
        bm = [v["brier"]["market"] for v in d.values() if v["brier"]["agent"] is not None]
        summary["agents"][agent] = dict(worlds=len(d), mean_excess_pct=100 * float(np.mean(ex)), wins=int(sum(x > 0 for x in ex)), mean_brier=float(np.mean(ba)) if ba else None, mean_brier_market_same=float(np.mean(bm)) if bm else None, per_seed={s: dict(excess_pct=100 * (v["tilt"] - v["base"]), tilt_dd=v["tilt_dd"], base_dd=v["base_dd"], brier=v["brier"]["agent"]) for s, v in d.items()})
    json.dump(summary, open(os.path.join(out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    for agent, s in summary["agents"].items():
        print(f"{agent:7s} worlds {s['worlds']} mean excess vs base {s['mean_excess_pct']:+.2f}%p, wins {s['wins']}/{s['worlds']}, brier {s['mean_brier']} (market {s['mean_brier_market_same']})")


if __name__ == "__main__":
    main()
