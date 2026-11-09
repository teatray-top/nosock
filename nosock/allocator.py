import bisect
import math
import os
import sys
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np

H = 20
ASSET_F = ("m20", "m126", "m12_1", "vol60", "volr", "dd252", "gap200")
ASSET_LABEL = {"m20": "20일 수익", "m126": "6개월 수익", "m12_1": "12-1개월 수익", "vol60": "60일 변동성", "volr": "변동성 확대(20/60일)", "dd252": "1년 고점 대비", "gap200": "200일선 괴리"}
STATE_F = ("s_vol20", "s_trend200", "s_dd252", "s_breadth")
STATE_LABEL = {"s_vol20": "시장 20일 변동성", "s_trend200": "시장 200일 추세", "s_dd252": "시장 1년 고점 대비", "s_breadth": "200일선 위 종목 비율"}
LAMBDA = 0.02
MIN_W = 0.01


def asset_arrays(ins):
    C = np.asarray(ins["C"], float)
    n = len(C)
    lc = np.log(C)
    r = np.r_[np.nan, np.diff(lc)]
    nan = lambda: np.full(n, np.nan)
    out = {k: nan() for k in ASSET_F}
    cs = np.cumsum(np.r_[0.0, lc])
    for t in range(252, n):
        out["m20"][t] = lc[t] - lc[t - 20]
        out["m126"][t] = lc[t] - lc[t - 126]
        out["m12_1"][t] = lc[t - 21] - lc[t - 252]
        s60 = np.std(r[t - 59:t + 1])
        s20 = np.std(r[t - 19:t + 1])
        out["vol60"][t] = s60 * math.sqrt(252)
        out["volr"][t] = s20 / s60 if s60 > 0 else 1.0
        out["dd252"][t] = math.exp(lc[t] - lc[t - 251:t + 1].max()) - 1
        out["gap200"][t] = lc[t] - (cs[t + 1] - cs[t - 199]) / 200
    return out


class Features:
    def __init__(self, m, state):
        self.m = m
        self.state = state
        self.A = {c: asset_arrays(m.inst[c]) for c in m.codes}

    def at(self, D, extra=()):
        m = self.m
        codes, X = [], []
        for c in m.eligible(D):
            _, i = m.close_at(c, D)
            if i is None:
                continue
            v = [self.A[c][k][i] for k in ASSET_F]
            if any(np.isnan(v)):
                continue
            codes.append(c)
            X.append(v)
        if len(codes) < 3:
            return None
        X = np.array(X, float)
        mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
        Z = np.clip((X - mu) / sd, -4, 4)
        Z[:, ASSET_F.index("vol60")] = np.clip((X[:, ASSET_F.index("vol60")] - 0.2) / 0.15, -4, 4)
        st = self.state
        i = bisect.bisect_right(m.days, D) - 1
        if i < 260:
            return None
        er = st.eq_ret
        I = st.eqi
        breadth = float(np.mean(Z[:, ASSET_F.index("gap200")] * sd[ASSET_F.index("gap200")] + mu[ASSET_F.index("gap200")] > 0))
        s = np.array([er[i - 19:i + 1].std() * math.sqrt(252), I[i] - I[i - 199:i + 1].mean(), math.exp(I[i] - I[i - 251:i + 1].max()) - 1, breadth])
        lim = (date.fromisoformat(D) - timedelta(days=7)).isoformat()
        ex_codes, ex_X = [], []
        for c in extra:
            if c in codes or c not in m.inst:
                continue
            _, j = m.close_at(c, D)
            if j is None or m.inst[c]["days"][j] < lim:
                continue
            if c not in self.A:
                self.A[c] = asset_arrays(m.inst[c])
            v = [self.A[c][k][j] for k in ASSET_F]
            if any(np.isnan(v)):
                continue
            ex_codes.append(c)
            ex_X.append(v)
        if ex_codes:
            EX = np.array(ex_X, float)
            EZ = np.clip((EX - mu) / sd, -4, 4)
            EZ[:, ASSET_F.index("vol60")] = np.clip((EX[:, ASSET_F.index("vol60")] - 0.2) / 0.15, -4, 4)
            codes = codes + ex_codes
            Z = np.vstack([Z, EZ])
            X = np.vstack([X, EX])
        return dict(codes=codes, Z=Z, raw=X, s=s, extra=ex_codes)


def gross(m, c, t, t_end):
    o, i, d0 = m.next_open(c, t)
    cl, _ = m.close_at(c, t_end)
    if o is None or cl is None or d0 is None or d0 > t_end:
        return 1.0, 0.0
    return cl / o, m.inst[c]["cost"](i, o)


def dataset(m, feats, start, end, step=5):
    ks = [d for d in m.days if start <= d <= end and d in m.kr_days]
    rows = []
    for a in range(0, len(ks) - H, step):
        t, t_end = ks[a], ks[a + H]
        f = feats.at(t)
        if f is None:
            continue
        R, Cc = zip(*[gross(m, c, t, t_end) for c in f["codes"]])
        rows.append(dict(t=t, t_end=t_end, codes=f["codes"], Z=f["Z"], s=f["s"], R=np.array(R), cost=np.array(Cc)))
    return rows


class Policy:
    def __init__(self, beta, gamma, c0, smu, ssd):
        self.beta, self.gamma, self.c0, self.smu, self.ssd = beta, gamma, c0, smu, ssd

    def weights(self, f):
        sz = np.clip((f["s"] - self.smu) / self.ssd, -4, 4)
        sc = f["Z"] @ self.beta
        cash = float(sz @ self.gamma + self.c0)
        z = np.r_[sc, cash]
        z = z - z.max()
        a = np.exp(z)
        a = a / a.sum()
        w = {c: float(x) for c, x in zip(f["codes"], a[:-1])}
        w = {c: x for c, x in w.items() if x >= MIN_W}
        return w, float(a[-1]), sc, sz

    def explain(self, f):
        w, cash, sc, sz = self.weights(f)
        per = {}
        for j, c in enumerate(f["codes"]):
            per[c] = dict(score=float(sc[j]), contrib={k: float(self.beta[q] * f["Z"][j, q]) for q, k in enumerate(ASSET_F)}, raw={k: float(f["raw"][j, q]) for q, k in enumerate(ASSET_F)}, weight=w.get(c, 0.0))
        return dict(weights=w, cash_model=cash, assets=per, state={k: float(f["s"][q]) for q, k in enumerate(STATE_F)}, state_z={k: float(sz[q]) for q, k in enumerate(STATE_F)}, beta={k: float(self.beta[q]) for q, k in enumerate(ASSET_F)}, gamma={k: float(self.gamma[q]) for q, k in enumerate(STATE_F)}, c0=float(self.c0))


def fit(rows, lam=LAMBDA):
    import torch
    S = np.array([r["s"] for r in rows])
    smu, ssd = S.mean(axis=0), S.std(axis=0) + 1e-9
    nmax = max(len(r["codes"]) for r in rows)
    T, F = len(rows), len(ASSET_F)
    Z = np.zeros((T, nmax, F))
    R = np.ones((T, nmax))
    K = np.zeros((T, nmax))
    M = np.zeros((T, nmax), bool)
    for t, r in enumerate(rows):
        n = len(r["codes"])
        Z[t, :n], R[t, :n], K[t, :n], M[t, :n] = r["Z"], r["R"], r["cost"], True
    Sz = np.clip((S - smu) / ssd, -4, 4)
    z_t, r_t, k_t, m_t, s_t = (torch.tensor(x) for x in (Z, R, K, M, Sz))
    beta = torch.zeros(F, dtype=torch.float64, requires_grad=True)
    gamma = torch.zeros(len(STATE_F), dtype=torch.float64, requires_grad=True)
    c0 = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([beta, gamma, c0], max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        sc = (z_t * beta).sum(-1).masked_fill(~m_t, -1e9)
        cash = (s_t * gamma).sum(-1, keepdim=True) + c0
        a = torch.softmax(torch.cat([sc, cash], dim=1), dim=1)
        port = (a[:, :-1] * r_t * (1 - k_t)).sum(1) + a[:, -1]
        loss = -torch.log(port).mean() + lam * ((beta ** 2).sum() + (gamma ** 2).sum())
        loss.backward()
        return loss

    opt.step(closure)
    return Policy(beta.detach().numpy().copy(), gamma.detach().numpy().copy(), float(c0.detach()[0]), smu, ssd)


class Allocator:
    def __init__(self, m, feats, policies):
        self.m = m
        self.feats = feats
        self.policies = policies

    def decide(self, D, k, extra=()):
        pol = self.policies.get(int(D[:4])) if isinstance(self.policies, dict) else self.policies
        f = self.feats.at(D, extra)
        if pol is None or f is None:
            el = self.m.eligible(D)
            return ({c: 1.0 / len(el) for c in el} if el else {}), dict(rule="allocator", note="학습 전 균등", candidates=[])
        ex = pol.explain(f)
        w = ex["weights"]
        cands = sorted(([dict(code=c, score=v["score"], weight=v["weight"], selected=v["weight"] > 0, raw=v["raw"], contrib=v["contrib"], outside=c in f["extra"]) for c, v in ex["assets"].items()]), key=lambda x: -x["score"])
        for rank, x in enumerate(cands, 1):
            x["rank"] = rank
        return w, dict(rule="allocator", k=len(w), candidates=cands, cash_weight=max(0.0, 1.0 - sum(w.values())), cash_model=ex["cash_model"], state=ex["state"], state_z=ex["state_z"], beta=ex["beta"], gamma=ex["gamma"])


LAMS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)


def growth(pol, rows):
    g = []
    for r in rows:
        w, _, _, _ = pol.weights(dict(codes=r["codes"], Z=r["Z"], s=r["s"]))
        port = sum(w.get(c, 0.0) * R * (1 - k) for c, R, k in zip(r["codes"], r["R"], r["cost"])) + (1 - sum(w.values()))
        g.append(math.log(port))
    return float(np.mean(g)) if g else -1e9


def fit_auto(rows, asof, val_years=3):
    cut = f"{int(asof[:4]) - val_years}{asof[4:10]}"
    fit_rows = [r for r in rows if r["t_end"] < cut]
    val = [r for r in rows if r["t"] >= cut]
    scores = {}
    if len(fit_rows) >= 100 and len(val) >= 30:
        scores = {lam: growth(fit(fit_rows, lam), val) for lam in LAMS}
        lam = max(scores, key=scores.get)
    else:
        lam = LAMBDA
    return fit(rows, lam), lam, scores
