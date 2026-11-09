import itertools
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

LIB_NOISE = float(os.environ.get("NOSOCK_LIB_NOISE", 0.4))
ALPHA_NOISE = 0.15


def event_library(w):
    rng = np.random.default_rng(10_000 + w["seed"])
    lib = {}
    for e in w["events"]:
        lib[e["id"]] = {}
        for b in e["branches"]:
            J = W.factor_impact(b["jump"])
            lib[e["id"]][b["key"]] = J * np.exp(rng.normal(0, LIB_NOISE, J.shape)) + rng.normal(0, 0.003, J.shape) * (J != 0)
    alpha_hat = float(np.clip(SR.ALPHA + rng.normal(0, ALPHA_NOISE), 0.2, 0.95))
    return lib, alpha_hat


def prob_agent(w, t, probs_for, lib, alpha_hat, use_true_impacts=False, alpha_override=None):
    evs = SR.open_events(w, t)[:3]
    if not evs:
        return None
    per = []
    for e in evs:
        q = np.array(w["consensus"][e["id"]].get(t, [1 / len(e["branches"])] * len(e["branches"])))
        if use_true_impacts:
            J = np.array([W.factor_impact(b["jump"]) for b in e["branches"]])
            a = SR.ALPHA
        else:
            J = np.array([lib[e["id"]][b["key"]] for b in e["branches"]])
            a = alpha_hat
        if alpha_override is not None:
            a = alpha_override
        priced = a * (q @ J)
        p = probs_for(e, q)
        per.append([(e["id"], b["key"], float(pk), J[k] - priced) for k, (b, pk) in enumerate(zip(e["branches"], p))])
    scen = []
    for combo in itertools.product(*per):
        r = SR.inst_return(sum(x[3] for x in combo))
        scen.append(dict(name="", probability=float(np.prod([x[2] for x in combo])), returns_pct={c: 100 * v for c, v in r.items()}))
    return dict(scenarios=scen)


def evaluate(seed, decisions_path, days=60):
    w = W.make_world(seed, days)
    lib, alpha_hat = event_library(w)
    cs = SR.codes()
    w0 = np.array([1 / len(SR.BASE) if c in SR.BASE else 0.0 for c in cs])
    logs = [json.loads(x) for x in open(decisions_path, encoding="utf-8")]
    out = {}
    variants = ("codex_as_is", "codex_probs+library", "market_probs+library", "true_probs+library", "codex_probs+library_fullypriced", "market_probs+library_fullypriced", "true_probs+library_fullypriced", "true_probs+true_impacts")
    for label in variants:
        dec = {}
        for r in logs:
            t, js = r["t"], r.get("agent") or {}
            if label == "codex_as_is":
                tw, _ = SR.tilt(js, w0, w, t) if js and not js.get("error") else (w0.copy(), None)
            else:
                ep = js.get("event_probs") or {}

                def pf(e, q, ep=ep, label=label):
                    if label.startswith("market"):
                        return q
                    if label.startswith("true"):
                        return np.array([b["p"] for b in e["branches"]])
                    v = ep.get(e["id"])
                    return W.norm([float(v.get(b["key"], 0)) for b in e["branches"]]) if isinstance(v, dict) else q
                tw, _ = SR.tilt(prob_agent(w, t, pf, lib, alpha_hat, use_true_impacts=label.endswith("true_impacts"), alpha_override=1.0 if label.endswith("fullypriced") else None), w0, w, t)
            dec[t] = tw
        sim = SR.simulate(w, dec)
        out[label] = 100 * (sim["tilt"]["total"] - sim["base"]["total"])
    return out, alpha_hat


def main():
    out_dir = os.environ.get("NOSOCK_SANDBOX_OUT", SR.OUT)
    runs = {}
    for d in sorted(os.listdir(out_dir)):
        parts = d.split("_")
        if len(parts) >= 3 and parts[0].startswith("w") and parts[1] == "codex" and os.path.exists(os.path.join(out_dir, d, "report.json")):
            runs[int(parts[0][1:])] = os.path.join(out_dir, d, "decisions.jsonl")
    allr = {}
    for seed, path in sorted(runs.items()):
        r, a = evaluate(seed, path)
        allr[seed] = r
        print(f"world {seed} (alpha_hat {a:.2f}): " + " | ".join(f"{k} {v:+.2f}%p" for k, v in r.items()))
    if allr:
        keys = list(next(iter(allr.values())))
        print("mean: " + " | ".join(f"{k} {np.mean([r[k] for r in allr.values()]):+.2f}%p ({sum(r[k] > 0 for r in allr.values())}/{len(allr)} wins)" for k in keys))
        json.dump(dict(lib_noise=LIB_NOISE, worlds=allr), open(os.path.join(out_dir, "evaluate_prob.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
