import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
import math
import sys
import warnings

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
import rl_band_v4 as V4

CYCLES = (1, 2, 5, 20, 60)
KAPPA = 0.5
DEPOSIT = 10_000.0
VAULT_SHARE = 0.5
SAMPLES = 150


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def grid_for(n):
    g = [(d, tp, sl, cr, float(n)) for d in B.DS for tp in B.TPS for sl in B.SLS for cr in B.CARRY]
    g.append((0.0, 0.0, 0.0, 2.0, float(n)))
    return np.array(g)


class DBook:
    def __init__(self):
        self.acct, self.vault, self.deposits, self.busy = 0.0, 0.0, 0.0, -1
        self.events = []
        self.trades = 0

    def deposit(self):
        self.acct += DEPOSIT
        self.deposits += DEPOSIT

    def apply(self, R, exit_day):
        P = math.expm1(R)
        profit = self.acct * P
        self.trades += 1
        if profit > 0:
            v = VAULT_SHARE * profit
            self.vault += v
            self.acct += profit - v
            self.events.append((exit_day, v))
        else:
            self.acct += profit

    def summary(self, months):
        days = sorted(set(e[0] for e in self.events))
        gaps = []
        from datetime import date
        for a, b in zip(days[:-1], days[1:]):
            gaps.append((date.fromisoformat(b) - date.fromisoformat(a)).days)
        last12 = sum(v for d, v in self.events if d >= "2025-11-06")
        return dict(deposits_won=self.deposits, vault_won=self.vault, acct_won=self.acct, total_won=self.acct + self.vault, gain_over_deposits_pct=100 * ((self.acct + self.vault) / self.deposits - 1) if self.deposits else None, trades=self.trades, cash_out_events=len(self.events), cash_out_days=len(days), median_days_between_cash_outs=float(np.median(gaps)) if gaps else None, vault_last_12_months_won=last12, vault_per_month_avg_won=self.vault / months)


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, pth in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), pth)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    d, b = V4.prep()
    F = b["F"][b["dec"]]
    v20 = F[:, R1.FEATS.index("v_20d")]
    dday = b["day"][b["dec"]]
    nq = len(b["dec"])
    quarters = R1.quarters()
    first_test_day = quarters[0][0]
    months = (len(set(x for x in dday if x >= first_test_day)) / 21.0)
    res = {}
    for n in CYCLES:
        G = grid_for(n)
        hold_i = len(G) - 1
        zero_d = np.nonzero((G[:, 0] == 0) & (G[:, 1] > 0))[0]
        rng = np.random.default_rng(5000 + n)
        S_idx = np.zeros((nq, SAMPLES), dtype=np.int64)
        S_r = np.zeros((nq, SAMPLES))
        S_x = np.zeros((nq, SAMPLES), dtype=np.int64)
        for q in range(nq):
            idx = np.r_[[hold_i], rng.choice(zero_d, size=30, replace=False), rng.choice(len(G) - 1, size=SAMPLES - 31, replace=False)]
            r, x = B.outcome_set(b, q, G[idx])
            S_idx[q], S_r[q], S_x[q] = idx, r, x
        rl, hold_cycle = DBook(), DBook()
        last_dep_day = None
        for a_, b_ in quarters:
            start_min = int(np.searchsorted(b["day"], a_))
            tr = np.nonzero(dday < a_)[0]
            tr = np.array([q for q in tr if S_x[q].max() < start_min], dtype=int)
            te = np.nonzero((dday >= a_) & (dday < b_))[0]
            if len(te) == 0 or len(tr) < 1000:
                continue
            Xtr = np.vstack([B.rows(F[q], G[S_idx[q]], v20[q]) for q in tr])
            m = lgb.train(LI.PARAMS, lgb.Dataset(Xtr, label=V4.reward(S_r[tr], KAPPA).ravel()), num_boost_round=LI.ROUNDS)
            for q in te:
                i = b["dec"][q]
                if dday[q] != last_dep_day:
                    rl.deposit()
                    hold_cycle.deposit()
                    last_dep_day = dday[q]
                if i > rl.busy and rl.acct > 0:
                    p = m.predict(B.rows(F[q], G, v20[q]))
                    a = int(np.argmax(p))
                    if p[a] > 0:
                        r, x = B.outcome_set(b, q, G[a:a + 1])
                        rl.apply(r[0], b["day"][x[0]])
                        rl.busy = x[0]
                if i > hold_cycle.busy and hold_cycle.acct > 0:
                    r, x = B.outcome_set(b, q, G[hold_i:hold_i + 1])
                    hold_cycle.apply(r[0], b["day"][x[0]])
                    hold_cycle.busy = x[0]
        res[str(n)] = dict(RL=rl.summary(months), hold_and_realize_every_cycle=hold_cycle.summary(months))
        x = res[str(n)]["RL"]
        y = res[str(n)]["hold_and_realize_every_cycle"]
        print(f"cycle <= {n:2d} sessions | RL: deposits {x['deposits_won'] / 1e4:.0f}만 vault {x['vault_won'] / 1e4:.1f}만 acct {x['acct_won'] / 1e4:.1f}만 total {x['total_won'] / 1e4:.1f}만 (+{x['gain_over_deposits_pct']:.1f}% over deposits) trades {x['trades']} cash-outs {x['cash_out_events']} median gap {x['median_days_between_cash_outs']} days, last 12m vault {x['vault_last_12_months_won'] / 1e4:.1f}만 | hold-and-realize-every-{n}: vault {y['vault_won'] / 1e4:.1f}만 total {y['total_won'] / 1e4:.1f}만", flush=True)
    C, O = b["C"], b["O"]
    units, dep, last = 0.0, 0.0, None
    for q in range(nq):
        if dday[q] < first_test_day or dday[q] == last:
            continue
        last = dday[q]
        i = b["dec"][q]
        px = O[i + 1]
        units += DEPOSIT * (1 - LI.FEE - LI.TICK / px) / px
        dep += DEPOSIT
    res["deposit_and_hold"] = dict(deposits_won=dep, vault_won=0.0, total_won=units * C[-1], gain_over_deposits_pct=100 * (units * C[-1] / dep - 1))
    print("deposit-and-hold: deposits", round(dep / 1e4), "만 total", round(units * C[-1] / 1e4, 1), "만 vault 0", flush=True)
    rec = dict(version="rl-band-v5", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], months=months, results=res)
    with open(out_path, "x", encoding="utf-8") as fo:
        json.dump(rec, fo, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
