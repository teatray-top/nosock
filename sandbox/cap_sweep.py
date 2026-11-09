import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
import numpy as np

from sandbox import evaluate_prob as EP
from sandbox import run as SR
from sandbox import world as W

CAPS = (0.0, 0.025, 0.05, 0.10, 0.175, 0.30)
METHODS = ("chatgpt_as_is", "chatgpt_prob_edge", "market_prob_edge")


def weights_for(method, w, t, js, w0, lib, alpha_hat):
    if method == "chatgpt_as_is":
        return SR.tilt(js, w0, w, t)[0] if js and not js.get("error") else w0.copy()
    ep = (js or {}).get("event_probs") or {}

    def pf(e, q):
        if method.startswith("market"):
            return q
        v = ep.get(e["id"])
        return W.norm([float(v.get(b["key"], 0)) for b in e["branches"]]) if isinstance(v, dict) else q
    return SR.tilt(EP.prob_agent(w, t, pf, lib, alpha_hat, alpha_override=1.0), w0, w, t)[0]


def sweep(out_dir):
    runs = {}
    for d in sorted(os.listdir(out_dir)):
        parts = d.split("_")
        if len(parts) >= 3 and parts[0].startswith("w") and parts[1] == "codex" and os.path.exists(os.path.join(out_dir, d, "report.json")):
            runs[int(parts[0][1:])] = os.path.join(out_dir, d, "decisions.jsonl")
    res = {m: {str(c): {} for c in CAPS} for m in METHODS}
    for seed, path in sorted(runs.items()):
        w = W.make_world(seed, 60)
        lib, alpha_hat = EP.event_library(w)
        cs = SR.codes()
        w0 = np.array([1 / len(SR.BASE) if c in SR.BASE else 0.0 for c in cs])
        logs = [json.loads(x) for x in open(path, encoding="utf-8")]
        for cap in CAPS:
            SR.CAP = cap
            for m in METHODS:
                dec = {r["t"]: weights_for(m, w, r["t"], r.get("agent") or {}, w0, lib, alpha_hat) for r in logs}
                sim = SR.simulate(w, dec)
                res[m][str(cap)][seed] = dict(excess=100 * (sim["tilt"]["total"] - sim["base"]["total"]), dd_tilt=100 * sim["tilt"]["maxdd"], dd_base=100 * sim["base"]["maxdd"])
    table = {}
    for m in METHODS:
        for cap in CAPS:
            v = list(res[m][str(cap)].values())
            if not v:
                continue
            ex = [x["excess"] for x in v]
            table.setdefault(m, {})[str(cap)] = dict(worlds=len(v), mean_excess_pct=float(np.mean(ex)), worst_pct=float(np.min(ex)), best_pct=float(np.max(ex)), wins=int(sum(x > 0 for x in ex)), mean_dd_change_pct=float(np.mean([x["dd_tilt"] - x["dd_base"] for x in v])))
    out = dict(worlds=sorted(runs), caps=list(CAPS), methods=list(METHODS), table=table, per_world=res, note="excess = tilt total - base total (%p); dd change > 0 means shallower drawdown than base")
    json.dump(out, open(os.path.join(out_dir, "cap_sweep.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    lines = [f"worlds {sorted(runs)}"]
    for m in METHODS:
        lines.append(m)
        for cap in CAPS:
            r = table.get(m, {}).get(str(cap))
            if r:
                lines.append(f"  cap {100 * cap:5.1f}%: mean {r['mean_excess_pct']:+.2f}%p  worst {r['worst_pct']:+.2f}  best {r['best_pct']:+.2f}  wins {r['wins']}/{r['worlds']}  dd change {r['mean_dd_change_pct']:+.2f}%p")
    txt = "\n".join(lines)
    open(os.path.join(out_dir, "cap_sweep.txt"), "w", encoding="utf-8").write(txt + "\n")
    return txt


if __name__ == "__main__":
    print(sweep(os.environ.get("NOSOCK_SANDBOX_OUT", SR.OUT)))
