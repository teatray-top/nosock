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
import rl_band_v1 as R1
import rl_band_v4_base as B

CAPITAL = 100_000_000.0
VAULT_SHARE = 0.5
SAMPLES = 150
SEEDS = 20
GRID = B.GRID
HOLD_IDX = np.nonzero(GRID[:, 1] == 0)[0]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def reward(R, kappa):
    P = np.expm1(R)
    pos = np.maximum(P, 0.0)
    return VAULT_SHARE * pos + kappa * (P - VAULT_SHARE * pos)


class Book:
    def __init__(self):
        self.acct, self.vault, self.busy, self.trades, self.wins = CAPITAL, 0.0, -1, 0, 0
        self.vault_by_year = {}

    def apply(self, R, exit_day):
        P = math.expm1(R)
        profit = self.acct * P
        if profit > 0:
            v = VAULT_SHARE * profit
            self.vault += v
            self.acct += profit - v
            self.vault_by_year[exit_day[:4]] = self.vault_by_year.get(exit_day[:4], 0.0) + v
            self.wins += 1
        else:
            self.acct += profit
        self.trades += 1

    def summary(self):
        return dict(vault_won=self.vault, acct_won=self.acct, total_pct=100 * ((self.acct + self.vault) / CAPITAL - 1), vault_pct=100 * self.vault / CAPITAL, trades=self.trades, win=self.wins / self.trades if self.trades else None, vault_by_year_won=self.vault_by_year)


def prep():
    d = R1.load()
    b = R1.build(d)
    b["first_min"] = np.r_[True, b["dk"][1:] != b["dk"][:-1]]
    mod, ce = d["mod"].to_numpy(), d["ce"].to_numpy()
    nd = int(b["dk"][-1]) + 1
    eod = np.array(b["last"], dtype=np.int64)
    for k in range(len(mod)):
        if mod[k] <= ce[k] - 1:
            eod[b["dk"][k]] = k
    b["eod"] = eod
    ud = b["udays"]
    b["gapd"] = np.array([(date.fromisoformat(ud[k + 1]) - date.fromisoformat(ud[k])).days if k + 1 < nd else 1 for k in range(nd)])
    return d, b


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, pth in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), pth)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    d, b = prep()
    F = b["F"][b["dec"]]
    v20 = F[:, R1.FEATS.index("v_20d")]
    dday = b["day"][b["dec"]]
    rng = np.random.default_rng(4242)
    nq = len(b["dec"])
    S_idx = np.zeros((nq, SAMPLES), dtype=np.int64)
    S_r = np.zeros((nq, SAMPLES))
    S_x = np.zeros((nq, SAMPLES), dtype=np.int64)
    zero_d = np.setdiff1d(np.nonzero(GRID[:, 0] == 0)[0], HOLD_IDX)
    rest = np.setdiff1d(np.arange(len(GRID)), HOLD_IDX)
    for q in range(nq):
        idx = np.r_[HOLD_IDX, rng.choice(zero_d, size=30, replace=False), rng.choice(rest, size=SAMPLES - 30 - len(HOLD_IDX), replace=False)]
        r, x = B.outcome_set(b, q, GRID[idx])
        S_idx[q], S_r[q], S_x[q] = idx, r, x
    print("outcomes done", nq, flush=True)
    books = {f"RL_kappa{k}": Book() for k in B.KAPPAS}
    books["fixed_best_kappa1"] = Book()
    books["always_hold60_realize"] = Book()
    rnd = [Book() for _ in range(SEEDS)]
    rngs = [np.random.default_rng(800 + s) for s in range(SEEDS)]
    chosen = {k: {} for k in B.KAPPAS}
    first_test = None
    hold60 = int(np.nonzero((GRID[:, 1] == 0) & (GRID[:, 4] == 60))[0][0])
    folds = []
    for a_, b_ in R1.quarters():
        start_min = int(np.searchsorted(b["day"], a_))
        tr = np.nonzero(dday < a_)[0]
        tr = np.array([q for q in tr if S_x[q].max() < start_min], dtype=int)
        te = np.nonzero((dday >= a_) & (dday < b_))[0]
        if len(te) == 0 or len(tr) < 1000:
            continue
        if first_test is None:
            first_test = te[0]
        Xtr = np.vstack([B.rows(F[q], GRID[S_idx[q]], v20[q]) for q in tr])
        models = {}
        for k in B.KAPPAS:
            models[k] = lgb.train(LI.PARAMS, lgb.Dataset(Xtr, label=reward(S_r[tr], k).ravel()), num_boost_round=LI.ROUNDS)
        sums = np.zeros(len(GRID))
        cnts = np.zeros(len(GRID))
        np.add.at(sums, S_idx[tr].ravel(), reward(S_r[tr], 1.0).ravel())
        np.add.at(cnts, S_idx[tr].ravel(), 1)
        fixed = int(np.argmax(np.where(cnts >= 30, sums / np.maximum(cnts, 1), -np.inf)))
        folds.append(dict(test=[a_, b_], train_points=int(len(tr)), fixed_best=GRID[fixed].tolist()))
        print("fold", a_, "train", len(tr), "fixed best", GRID[fixed].tolist(), flush=True)
        for q in te:
            i = b["dec"][q]
            for k in B.KAPPAS:
                bk = books[f"RL_kappa{k}"]
                if i > bk.busy:
                    p = models[k].predict(B.rows(F[q], GRID, v20[q]))
                    a = int(np.argmax(p))
                    if p[a] > 0:
                        r, x = B.outcome_set(b, q, GRID[a:a + 1])
                        bk.apply(r[0], b["day"][x[0]])
                        bk.busy = x[0]
                        key = tuple(round(float(z), 4) for z in GRID[a])
                        chosen[k][key] = chosen[k].get(key, 0) + 1
            for nm, a in (("fixed_best_kappa1", fixed), ("always_hold60_realize", hold60)):
                bk = books[nm]
                if i > bk.busy:
                    r, x = B.outcome_set(b, q, GRID[a:a + 1])
                    bk.apply(r[0], b["day"][x[0]])
                    bk.busy = x[0]
            for s in range(SEEDS):
                if i > rnd[s].busy:
                    a = int(rngs[s].integers(len(GRID)))
                    r, x = B.outcome_set(b, q, GRID[a:a + 1])
                    rnd[s].apply(r[0], b["day"][x[0]])
                    rnd[s].busy = x[0]
        print("   after", b_, {nm: (round(bk.vault / 1e6, 1), round(100 * ((bk.acct + bk.vault) / CAPITAL - 1), 1)) for nm, bk in books.items()}, "random vault/total", round(float(np.mean([x.vault for x in rnd])) / 1e6, 1), round(100 * (float(np.mean([x.acct + x.vault for x in rnd])) / CAPITAL - 1), 1), flush=True)
    i0 = b["dec"][first_test] + 1
    C, O = b["C"], b["O"]
    hold_pct = 100 * (C[-1] / O[i0] - 1)
    q_ = O[i0]
    basis, units, cash_v = CAPITAL, CAPITAL / O[i0], 0.0
    r1_years = {}
    for q in range(first_test, nq):
        i = b["dec"][q]
        val = units * C[i]
        if val >= basis * 1.10:
            take = 0.5 * (val - basis)
            px = O[i + 1]
            su = take / C[i]
            got = su * px * (1 - LI.FEE - LI.TICK / px)
            units -= su
            cash_v += got
            yr = b["day"][i + 1][:4]
            r1_years[yr] = r1_years.get(yr, 0.0) + got
            basis = units * px
    r1 = dict(vault_won=cash_v, acct_won=units * C[-1], total_pct=100 * ((units * C[-1] + cash_v) / CAPITAL - 1), vault_pct=100 * cash_v / CAPITAL, vault_by_year_won=r1_years)
    out = {nm: bk.summary() for nm, bk in books.items()}
    out["random_mean"] = dict(vault_pct=100 * float(np.mean([x.vault for x in rnd])) / CAPITAL, total_pct=100 * (float(np.mean([x.acct + x.vault for x in rnd])) / CAPITAL - 1))
    out["buy_and_hold"] = dict(vault_pct=0.0, total_pct=hold_pct)
    out["R1_hold_skim10"] = r1
    for k in B.KAPPAS:
        ch = chosen[k]
        out[f"RL_kappa{k}"]["chosen_top"] = sorted([(list(kk), v) for kk, v in ch.items()], key=lambda z: -z[1])[:12]
        n = sum(ch.values())
        out[f"RL_kappa{k}"]["share_hold_actions"] = sum(v for kk, v in ch.items() if kk[1] == 0) / n if n else None
        out[f"RL_kappa{k}"]["carry_share"] = {c: sum(v for kk, v in ch.items() if kk[3] == c) / n if n else None for c in (0.0, 1.0, 2.0)}
    print("TEST", json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("chosen_top",)} for k, v in out.items()}, default=float), flush=True)
    for k in B.KAPPAS:
        print(f"RL kappa {k} top choices (dip, tp, sl, carry, horizon):", out[f"RL_kappa{k}"]["chosen_top"][:8], flush=True)
    rec = dict(version="rl-band-v4", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], folds=folds, results=out)
    with open(out_path, "x", encoding="utf-8") as fo:
        json.dump(rec, fo, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
