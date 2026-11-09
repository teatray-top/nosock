import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
import math
import sys
import warnings
from datetime import date

sys.dont_write_bytecode = True
warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import lightgbm as lgb
import learn_inverse as LI
import harvest_v8 as H8
import msma_v1 as M1

ETF_DIR = M1.ETF_DIR
DS = (0.0, 0.01, 0.02, 0.03, 0.05)
TPS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20)
SLS = (0.02, 0.04, 0.07, 0.10, 0.15, 0.0)
HZS = (1, 5, 20)
GRID = np.array([(d, tp, sl, hz) for d in DS for tp in TPS for sl in SLS for hz in HZS] + [(0.0, 0.0, 0.0, hz) for hz in HZS])
SAMPLES = 60
KAPPA = 0.5
VAULT_SHARE = 0.5
DEPOSIT = 10_000.0
TEST_YEARS = list(range(2016, 2027))
FEATS = ["r1", "r5", "r20", "r60", "v20", "v60", "vratio", "dd250", "eq", "dow", "next_gap"] + [f"cls_{c}" for c in M1.CLASSES]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load():
    man = json.load(open(os.path.join(ETF_DIR, "manifest.json"), encoding="utf-8"))
    recs = {}
    for code in M1.CLASS:
        info = man["files"][f"{code}_adj"]
        p = os.path.join(ETF_DIR, info["file"])
        if sha(p) != info["sha256"]:
            raise SystemExit(f"etf hash differs {code}")
        recs[code] = json.load(open(p, encoding="utf-8"))
    return recs


def instrument(rec, code):
    days = [d for d in rec["days"] if d <= M1.END]
    n = len(days)
    O, H, L, C = (np.asarray(rec[k][:n], float) for k in ("O", "H", "L", "C"))
    r = np.r_[np.nan, np.diff(np.log(C))]
    rs = lambda w: np.array([np.std(r[max(1, i - w + 1):i + 1]) if i >= w else np.nan for i in range(n)])
    v20, v60 = rs(20), rs(60)
    S, _, _ = M1.signals(np.array(days), C[:, None])
    eq = np.nanmean(np.tanh(np.stack([S[:, :, M1.SIG.index(s)] for s in M1.EQ_SIGNALS], axis=2)), axis=2)[:, 0]
    hi = np.array([C[max(0, i - 249):i + 1].max() for i in range(n)])
    lc = np.log(C)
    lag = lambda k: np.r_[[np.nan] * k, lc[k:] - lc[:-k]]
    gaps = np.array([(date.fromisoformat(days[i + 1]) - date.fromisoformat(days[i])).days if i + 1 < n else 1 for i in range(n)], float)
    dow = np.array([date.fromisoformat(x).weekday() for x in days], float)
    cls = np.zeros((n, len(M1.CLASSES)))
    cls[:, M1.CLASSES.index(M1.CLASS[code])] = 1.0
    F = np.column_stack([lag(1), lag(5), lag(20), lag(60), v20, v60, v20 / v60, C / hi - 1, eq, dow, gaps, cls])
    return dict(code=code, days=days, O=O, H=H, L=L, C=C, F=F, v20=v20)


def outcome(ins, t, act):
    d, tp, sl, hz = act
    O, H, L, C, days = ins["O"], ins["H"], ins["L"], ins["C"], ins["days"]
    n = len(C)
    e = t + 1
    if e >= n:
        return np.nan, t
    if d == 0:
        pin, k = O[e], e
    else:
        lvl = C[t] * (1 - d)
        if L[e] > lvl:
            return 0.0, e
        pin, k = (min(lvl, O[e]) if O[e] < lvl else lvl), e
    cost = lambda p: LI.FEE + H8.tick(days[min(k, n - 1)], p) / p
    cin = cost(pin)
    stop = min(k + int(hz) - 1, n - 1)
    up = pin * (1 + tp + 2 * cin) if tp > 0 else np.inf
    dn = pin * (1 - sl) if sl > 0 else -np.inf
    for j in range(k, stop + 1):
        first = j > k or d == 0
        if L[j] <= dn and (j > k or d == 0 or True):
            px = min(dn, O[j]) if (j > k and O[j] < dn) else dn
            return math.log(px * (1 - cost(px)) / (pin * (1 + cin))), j
        if H[j] >= up and first:
            px = max(up, O[j]) if (j > k and O[j] > up) else up
            return math.log(px * (1 - cost(px)) / (pin * (1 + cin))), j
    px = C[stop]
    return math.log(px * (1 - cost(px)) / (pin * (1 + cin))), stop


def reward(R, kappa):
    P = np.expm1(R)
    pos = np.maximum(P, 0.0)
    return VAULT_SHARE * pos + kappa * (P - VAULT_SHARE * pos)


def rows(Fv, acts, v20):
    A = np.asarray(acts, float)
    sc = v20 * math.sqrt(20) if v20 > 0 else np.nan
    return np.hstack([np.repeat(Fv[None, :], len(A), axis=0), A, A[:, :3] / sc])


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, pth in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), pth)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    recs = load()
    inst = {c: instrument(recs[c], c) for c in M1.CLASS}
    rng = np.random.default_rng(31)
    hold_idx = np.nonzero(GRID[:, 1] == 0)[0]
    samp = {}
    for c, ins in inst.items():
        n = len(ins["C"])
        rows_ = []
        for t in range(260, n - 1):
            if np.isnan(ins["F"][t]).any():
                continue
            idx = np.r_[hold_idx, rng.choice(len(GRID) - len(hold_idx), size=SAMPLES - len(hold_idx), replace=False)]
            out = [outcome(ins, t, GRID[a]) for a in idx]
            rows_.append((t, idx, np.array([o[0] for o in out]), np.array([o[1] for o in out])))
        samp[c] = rows_
        print("outcomes", c, len(rows_), flush=True)
    books = {c: dict(acct=0.0, vault=0.0, dep=0.0, busy=-1, trades=0, roles={}) for c in inst}
    hold_books = {c: dict(units=0.0, dep=0.0) for c in inst}
    all_days = sorted(set(d for ins in inst.values() for d in ins["days"] if d >= f"{TEST_YEARS[0]}-01-01"))
    for y in TEST_YEARS:
        cut = f"{y}-01-01"
        X, Y = [], []
        for c, ins in inst.items():
            for t, idx, R, ex in samp[c]:
                if ins["days"][t] < cut and ex.max() < len(ins["days"]) and ins["days"][int(ex.max())] < cut and not np.isnan(R).any():
                    X.append(rows(ins["F"][t], GRID[idx], ins["v20"][t]))
                    Y.append(reward(R, KAPPA))
        m = lgb.train(LI.PARAMS, lgb.Dataset(np.vstack(X), label=np.concatenate(Y)), num_boost_round=LI.ROUNDS)
        ydays = [d for d in all_days if d[:4] == str(y)]
        for dday in ydays:
            avail = [c for c, ins in inst.items() if dday in ins["_pos"]] if "_pos" in next(iter(inst.values())) else None
            if avail is None:
                for ins in inst.values():
                    ins["_pos"] = {d: i for i, d in enumerate(ins["days"])}
                avail = [c for c, ins in inst.items() if dday in ins["_pos"]]
            avail = [c for c in avail if inst[c]["_pos"][dday] >= 260 and not np.isnan(inst[c]["F"][inst[c]["_pos"][dday]]).any()]
            if not avail:
                continue
            share = DEPOSIT / len(avail)
            for c in avail:
                ins, bk = inst[c], books[c]
                t = ins["_pos"][dday]
                bk["acct"] += share
                bk["dep"] += share
                hb = hold_books[c]
                if t + 1 < len(ins["O"]):
                    hb["units"] += share / (ins["O"][t + 1] * (1 + LI.FEE))
                hb["dep"] += share
                if t > bk["busy"] and t + 1 < len(ins["C"]):
                    p = m.predict(rows(ins["F"][t], GRID, ins["v20"][t]))
                    a = int(np.argmax(p))
                    if p[a] > 0:
                        R, ex = outcome(ins, t, GRID[a])
                        if not np.isnan(R):
                            P = math.expm1(R)
                            prof = bk["acct"] * P
                            if prof > 0:
                                bk["vault"] += VAULT_SHARE * prof
                                bk["acct"] += prof - VAULT_SHARE * prof
                            else:
                                bk["acct"] += prof
                            bk["busy"] = ex
                            bk["trades"] += 1
                            hz = int(GRID[a][3])
                            bk["roles"][hz] = bk["roles"].get(hz, 0) + 1
        print("year", y, "vault total", round(sum(b["vault"] for b in books.values()) / 1e4, 1), "만", flush=True)
    out = {}
    for c in inst:
        bk, hb = books[c], hold_books[c]
        ins = inst[c]
        nroles = sum(bk["roles"].values())
        out[c] = dict(name=json.load(open(os.path.join(ETF_DIR, "manifest.json"), encoding="utf-8"))["names"][c], deposits_won=bk["dep"], vault_won=bk["vault"], acct_won=bk["acct"], trades=bk["trades"], role_share={str(k): v / nroles for k, v in sorted(bk["roles"].items())} if nroles else {}, hold_value_won=hb["units"] * ins["C"][-1])
        x = out[c]
        main_role = max(bk["roles"], key=bk["roles"].get) if nroles else None
        print(f"{c} {x['name'][:16]:16s} deposits {x['deposits_won'] / 1e4:6.1f}만 vault {x['vault_won'] / 1e4:6.1f}만 acct {x['acct_won'] / 1e4:6.1f}만 | hold {x['hold_value_won'] / 1e4:6.1f}만 | trades {x['trades']:4d} main cycle {main_role} roles {x['role_share']}", flush=True)
    tot = dict(deposits_won=sum(v["deposits_won"] for v in out.values()), vault_won=sum(v["vault_won"] for v in out.values()), acct_won=sum(v["acct_won"] for v in out.values()), hold_value_won=sum(v["hold_value_won"] for v in out.values()))
    print("TOTAL", {k: round(v / 1e4, 1) for k, v in tot.items()}, "만", flush=True)
    rec = dict(version="ms-role-v2", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], per_instrument=out, total=tot)
    with open(out_path, "x", encoding="utf-8") as fo:
        json.dump(rec, fo, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
