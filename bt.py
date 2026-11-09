import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

import checks
import panel
import reader as R
import sim
from sim import Data, run_days, summary

iv = os.environ.get("BT_IV", "us5m")
h = int(os.environ.get("BT_H", 36))
FOLD = int(os.environ.get("BT_FOLD", 14))
NF = int(os.environ.get("BT_NFOLD", 3))
VAL = int(os.environ.get("BT_VAL", max(5, FOLD // 2)))
SEEDS = int(os.environ.get("BT_SEEDS", 20))
HEAD = int(os.environ.get("BT_HEAD", 0))
THRS = [float(x) for x in os.environ.get("BT_THRS", "0.4,0.45,0.5,0.55,0.6,0.65,0.7").split(",")]
R.STOP = sim.STOP = float(os.environ.get("BT_STOP", sim.STOP))
TAG = os.environ.get("BT_TAG", f"bt_{iv}_h{h}_s{R.STOP}_head{HEAD}" + (f"_tp{sim.TP}" if sim.TP > 0 else ""))

TRAIN_MIN = int(os.environ.get("BT_TRAIN_MIN", 5))
TRAIN_MAX = int(os.environ.get("BT_TRAIN_MAX", 0))
X = Data(R.DEV, iv)
PV = int(os.path.getmtime(os.environ.get("PANEL_PATH") or os.path.join(panel.D, f"panel{iv}_v{panel.VER}.npz")))
ret, stop, hor = R.labels(X, h)
days = list(range(1, X.D))
if checks.check(X, ret, h, R.STOP):
    sys.exit("checks failed, backtest aborted")


GATE = None
if os.environ.get("BT_GATE"):
    _g = np.load(os.environ["BT_GATE"], allow_pickle=True)
    assert [str(d) for d in _g["days"]] == [str(d) for d in X.days] and [str(s) for s in _g["syms"]] == [str(s) for s in X.syms], "gate panel mismatch"
    GATE = torch.tensor(_g["gate"], device=X.dev)
    print(f"GATE {os.environ['BT_GATE']} pass_frac {float(GATE.float().mean()):.3f}", flush=True)


def policy(P, thr, gen=None):
    held = {}
    N = X.N

    def act(env, per, g, mk):
        d, b, pos = int(env.d[0]), int(env.b[0]), int(env.pos[0])
        if pos < N:
            c = float(X.C[d, b, pos])
            tp = X.us and X.tp > 0 and c >= float(env.tpp[0])
            if b - held["b"] >= h or c <= float(env.entry[0]) * (1 - X.stop) or tp:
                return torch.tensor([N], device=X.dev)
            return env.pos.clone()
        sc = P[d][b].masked_fill(~mk[0, :-1], 0)
        if GATE is not None:
            sc = sc.masked_fill(~GATE[d], 0)
        j = int(sc.argmax())
        if float(sc[j]) > thr:
            if gen is not None:
                ix = mk[0, :-1].nonzero().flatten()
                j = int(ix[int(torch.randint(len(ix), (1,), generator=gen))])
            held["b"] = b
            return torch.tensor([j], device=X.dev)
        return torch.tensor([N], device=X.dev)

    return act


def seed_of(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


import hashlib
_meta = dict(exit_bar=int(X.exit_bar), entry_last=int(X.entry_last), us_exit_env=os.environ.get("DOBBY_US_EXIT"), panel=os.environ.get("PANEL_PATH"), panel_mtime=PV, script_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16], sim_sha256=hashlib.sha256(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim.py"), "rb").read()).hexdigest()[:16], train_min=TRAIN_MIN, train_max=TRAIN_MAX, thrs=list(THRS), slip=sim.SLIP, fees=dict(fee_buy=float(X.eff_rates(X.deposit)[0]), fee_sell=float(X.eff_rates(X.deposit)[1])))
def _sha(path):
    hh = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            hh.update(b)
    return hh.hexdigest()[:16]
if os.environ.get("BT_GATE"):
    _gp = os.environ["BT_GATE"]
    _gj = _gp[:-4] + ".json"
    _meta.update(gate=_gp, gate_sha256=_sha(_gp), gate_stats=json.load(open(_gj)) if os.path.exists(_gj) else None, gate_daily_input="data/panel_daily_v2.npz", gate_daily_input_sha256=_sha(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "panel_daily_v2.npz")), gate_info_false_cells=int((~_g["info"]).sum()), gate_info_false_frac=float((~_g["info"]).mean()))
print("META " + json.dumps(_meta), flush=True)
out = dict(iv=iv, h=h, stop=R.STOP, tp=sim.TP, head=R.HEADS[HEAD], fold=FOLD, val=VAL, seeds=SEEDS, deposit=X.deposit, meta=_meta, folds=[])
allr, allm = [], []
for k in range(NF):
    end = len(days) - k * FOLD
    te = days[end - FOLD:end]
    va = days[end - FOLD - VAL:end - FOLD]
    tr = days[:end - FOLD - VAL]
    if TRAIN_MAX:
        tr = tr[-TRAIN_MAX:]
    if len(tr) < TRAIN_MIN or end - FOLD < 1:
        print(f"fold {k}: train too short ({len(tr)}), stop", flush=True)
        break
    t0 = time.time()
    m = R.fit_or_load(X, tr, ret, stop, hor, f"runs/reader_{iv}_h{h}_s{R.STOP}" + (f"_tp{sim.TP}" if sim.TP > 0 else "") + f"_c{int(round(sum(X.eff_rates(X.deposit)) * 1e5))}_e{end}_{PV}.pt")
    P = {}
    for ds in (va, te):
        p = R.predict(m, X, ds, ret, stop, hor)[..., HEAD]
        P.update({d: p[i] for i, d in enumerate(ds)})
    vt = {}
    for thr in THRS:
        s = summary(run_days(X, va, policy(P, thr), flat=True))
        vt[thr] = (s["total_pnl"], s["trades_per_day"])
    thr = max(THRS, key=lambda t: vt[t][0])
    rows = run_days(X, te, policy(P, thr), flat=True)
    rp = np.array([r["pnl"] for r in rows])
    rand = []
    for s in range(SEEDS):
        g = torch.Generator().manual_seed(seed_of(iv, h, end, thr, s))
        rand.append([r["pnl"] for r in run_days(X, te, policy(P, thr, g), flat=True)])
    rand = np.array(rand)
    rt = rand.sum(1)
    allr.append(rp)
    allm.append(rand.mean(0))
    sm = summary(rows)
    fold = dict(k=k, test=[str(X.days[te[0]]), str(X.days[te[-1]])], train_days=len(tr), val_days=len(va), thr=thr, val_table={str(a): b for a, b in vt.items()}, reader=sm, rand_total_mean=float(rt.mean()), rand_total_sd=float(rt.std()), reader_pct_among_rand=float((rt < rp.sum()).mean()), diff_mean_day=float((rp - rand.mean(0)).mean()), rows=rows, rand_daily_mean=rand.mean(0).tolist(), secs=time.time() - t0)
    out["folds"].append(fold)
    print(f"fold {k} {fold['test'][0]}..{fold['test'][1]} train {len(tr)} thr {thr}: reader {sm['total_pnl']:+.3f} (ex-top3 {sm['ex_top3']:+.3f}, median {sm['median_day']:+.3f}, trades/day {sm['trades_per_day']:.1f}, zero {sm['zero_days']}) rand {rt.mean():+.3f}±{rt.std():.3f} pct {fold['reader_pct_among_rand']:.2f} {fold['secs']:.0f}s", flush=True)

if allr:
    d = np.concatenate(allr) - np.concatenate(allm)
    rng = np.random.default_rng(0)
    bs = np.array([rng.choice(d, len(d)).mean() for _ in range(10000)])
    rsum = float(np.concatenate(allr).sum())
    out["agg"] = dict(days=int(len(d)), reader_total=rsum, rand_mean_total=float(np.concatenate(allm).sum()), diff_mean_day=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], p_le0=float((bs <= 0).mean()), folds_beating_rand=int(sum(f["reader_pct_among_rand"] >= 0.95 for f in out["folds"])), reader_total_pct_of_deposit_per_day=rsum / X.deposit / len(d) * 100)
    a = out["agg"]
    print(f"AGG {iv} h{h} head {R.HEADS[HEAD]}: {a['days']} days reader {a['reader_total']:+.3f} ({a['reader_total_pct_of_deposit_per_day']:+.2f}%/day) rand {a['rand_mean_total']:+.3f} diff/day {a['diff_mean_day']:+.4f} CI95 [{a['ci95'][0]:+.4f}, {a['ci95'][1]:+.4f}] P(<=0) {a['p_le0']:.3f} folds>=p95 {a['folds_beating_rand']}/{len(out['folds'])}", flush=True)
os.makedirs("runs", exist_ok=True)
json.dump(out, open(f"runs/{TAG}.json", "w"), indent=1)
