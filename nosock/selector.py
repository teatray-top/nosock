import bisect
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np

H = 20
EXPERTS = ("mom", "eq", "def")
EXPERT_LABEL = {"mom": "모멘텀 집중", "eq": "29종 균등", "def": "방어 (단기채·현금성)"}
DEFENSIVE = ("SHY", "153130", "114260")
FEATURES = ("vol20", "vol_ratio", "trend200", "dd252", "disp126", "mom_top3", "rel_last")
FEATURE_LABEL = {"vol20": "최근 20일 변동성", "vol_ratio": "변동성 확대 (20일/120일)", "trend200": "200일 추세", "dd252": "1년 고점 대비 낙폭", "disp126": "종목 간 6개월 수익 차이", "mom_top3": "상위 3종목 모멘텀 점수", "rel_last": "직전 20일 모멘텀−균등 성과"}
LAMBDA = 0.05


class Experts:
    def __init__(self, m, mom, k_of):
        self.m = m
        self.mom = mom
        self.k_of = k_of

    def weights(self, D):
        m = self.m
        w_mom, why = self.mom.decide(D, self.k_of(D))
        el = m.eligible(D)
        w_eq = {c: 1.0 / len(el) for c in el} if el else {}
        d = [c for c in DEFENSIVE if c in el]
        w_def = {c: 1.0 / len(d) for c in d} if d else {}
        return dict(mom=w_mom, eq=w_eq, **{"def": w_def}), why


class State:
    def __init__(self, m, mom):
        self.m = m
        self.mom = mom
        days = m.days
        n = len(days)
        self.pos = {d: i for i, d in enumerate(days)}
        codes = m.codes
        lc = np.full((n, len(codes)), np.nan)
        for j, c in enumerate(codes):
            ins = m.inst[c]
            k = 0
            cd, C = ins["days"], ins["C"]
            last = np.nan
            for i, d in enumerate(days):
                while k < len(cd) and cd[k] <= d:
                    last = math.log(C[k])
                    k += 1
                lc[i, j] = last
        self.lc = lc
        r = np.diff(lc, axis=0, prepend=np.nan)
        r[np.isnan(r)] = 0.0
        self.r = r
        el = np.zeros((n, len(codes)), bool)
        for i, d in enumerate(days):
            s = set(m.eligible(d))
            el[i] = [c in s for c in codes]
        self.el = el
        cnt = el.sum(axis=1)
        self.eq_ret = np.where(cnt > 0, (r * el).sum(axis=1) / np.maximum(cnt, 1), 0.0)
        self.eqi = np.cumsum(self.eq_ret)

    def features(self, D):
        i = bisect.bisect_right(self.m.days, D) - 1
        if i < 260:
            return None
        er = self.eq_ret
        v20 = er[i - 19:i + 1].std() * math.sqrt(252)
        v120 = er[i - 119:i + 1].std() * math.sqrt(252)
        I = self.eqi
        trend = I[i] - I[i - 199:i + 1].mean()
        dd = math.exp(I[i] - I[i - 251:i + 1].max()) - 1
        e = self.el[i]
        ret126 = self.lc[i, e] - self.lc[i - 126, e]
        ret126 = ret126[~np.isnan(ret126)]
        disp = float(ret126.std()) if len(ret126) > 2 else 0.0
        rows = []
        for c in self.m.eligible(D):
            _, k = self.m.close_at(c, D)
            v = self.mom.score[c][k] if k is not None else np.nan
            if not np.isnan(v):
                rows.append(v)
        top3 = float(np.mean(sorted(rows, reverse=True)[:3])) if rows else 0.0
        return dict(vol20=v20, vol_ratio=v20 / v120 if v120 > 0 else 1.0, trend200=trend, dd252=dd, disp126=disp, mom_top3=top3)


def kr_sessions(m, start, end):
    return [d for d in m.days if start <= d <= end and d in m.kr_days]


def label(m, w, t, t_end):
    tot, cost, cw = 0.0, 0.0, 1.0 - sum(w.values())
    for c, x in w.items():
        if x <= 0:
            continue
        o, i, d0 = m.next_open(c, t)
        cl, _ = m.close_at(c, t_end)
        if o is None or cl is None or d0 is None or d0 > t_end:
            cw += x
            continue
        tot += x * cl / o
        cost += x * m.inst[c]["cost"](i, o)
    return math.log(max(1e-9, tot + max(0.0, cw))) - cost


def dataset(m, experts, state, start, end, step=5):
    ks = kr_sessions(m, start, end)
    rows = []
    for a in range(0, len(ks) - H, step):
        t, t_end = ks[a], ks[a + H]
        f = state.features(t)
        if f is None:
            continue
        W, _ = experts.weights(t)
        y = [label(m, W[e], t, t_end) for e in EXPERTS]
        rows.append(dict(t=t, t_end=t_end, f=f, y=y))
    for j, r in enumerate(rows):
        prev = [q for q in rows[:j] if q["t_end"] <= r["t"]]
        r["f"]["rel_last"] = (prev[-1]["y"][0] - prev[-1]["y"][1]) if prev else 0.0
    return rows


def rel_last_at(rows, D):
    prev = [q for q in rows if q["t_end"] <= D]
    return (prev[-1]["y"][0] - prev[-1]["y"][1]) if prev else 0.0


class Policy:
    def __init__(self, mu, sd, B, b, conditional):
        self.mu, self.sd, self.B, self.b, self.conditional = mu, sd, B, b, conditional

    def alpha(self, f):
        x = np.clip((np.array([f[k] for k in FEATURES]) - self.mu) / self.sd, -4, 4)
        z = (self.B @ x if self.conditional else 0.0) + self.b
        z = z - z.max()
        a = np.exp(z)
        return a / a.sum(), x

    def explain(self, f):
        a, x = self.alpha(f)
        contrib = {e: {k: float(self.B[i, j] * x[j]) for j, k in enumerate(FEATURES)} for i, e in enumerate(EXPERTS)} if self.conditional else {}
        return dict(alpha={e: float(a[i]) for i, e in enumerate(EXPERTS)}, features={k: float(f[k]) for k in FEATURES}, z={k: float(x[j]) for j, k in enumerate(FEATURES)}, contrib=contrib)


def fit(rows, conditional=True, lam=LAMBDA, seed=0):
    import torch
    X = np.array([[r["f"][k] for k in FEATURES] for r in rows], float)
    Y = np.array([r["y"] for r in rows], float)
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Xs = np.clip((X - mu) / sd, -4, 4)
    torch.manual_seed(seed)
    xt, yt = torch.tensor(Xs, dtype=torch.float64), torch.tensor(Y, dtype=torch.float64)
    B = torch.zeros((len(EXPERTS), len(FEATURES)), dtype=torch.float64, requires_grad=True)
    b = torch.zeros(len(EXPERTS), dtype=torch.float64, requires_grad=True)
    params = [B, b] if conditional else [b]
    opt = torch.optim.LBFGS(params, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = (xt @ B.T) + b if conditional else b.expand(xt.shape[0], -1)
        a = torch.softmax(z, dim=1)
        g = torch.log((a * torch.exp(yt)).sum(dim=1)).mean()
        loss = -g + lam * (B ** 2).sum() + 1e-4 * (b ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return Policy(mu, sd, B.detach().numpy().copy(), b.detach().numpy().copy(), conditional)


class Selector:
    def __init__(self, experts, state, policies, rows):
        self.experts = experts
        self.state = state
        self.policies = policies
        self.rows = rows
        self.score = experts.mom.score

    def policy_for(self, D):
        return self.policies.get(int(D[:4]))

    def decide(self, D, k):
        W, why = self.experts.weights(D)
        pol = self.policy_for(D)
        f = self.state.features(D)
        if pol is None or f is None:
            return W["eq"], dict(rule="selector", k=k, note="모델 학습 전: 균등", candidates=why.get("candidates", []))
        f["rel_last"] = rel_last_at(self.rows, D)
        ex = pol.explain(f)
        a = ex["alpha"]
        w = {}
        for e in EXPERTS:
            for c, x in W[e].items():
                w[c] = w.get(c, 0.0) + a[e] * x
        w = {c: x for c, x in w.items() if x > 1e-6}
        return w, dict(rule="selector", k=k, alpha=a, features=ex["features"], z=ex["z"], contrib=ex["contrib"], candidates=why.get("candidates", []), cash_weight=max(0.0, 1.0 - sum(w.values())))
