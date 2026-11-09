import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ADJ = os.path.join(HERE, "data", "kr_etf", "kodex_daily_adj.json")
OUT = os.path.join(HERE, "runs", "market_model_2026-11-06_v2.json")
C0 = 50000.0
SYM = "069500"
LAST = "2026-11-05"
N_PATHS = 200
SEED = 20261106
MA_N = 200
DEPOSIT, FEE, TICK = 10000.0, 0.00015, 5.0


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def load_returns():
    a = json.load(open(ADJ, encoding="utf-8"))[SYM]
    keep = [i for i, d in enumerate(a["days"]) if d <= LAST]
    C = np.array([a["C"][i] for i in keep], float)
    return np.diff(np.log(C)), C, [a["days"][i] for i in keep]


def facts(r):
    r = np.asarray(r, float)
    m, s = r.mean(), r.std(ddof=1)
    z = (r - m) / s
    def ac(x, k):
        return float(np.corrcoef(x[:-k], x[k:])[0, 1])
    logp = np.concatenate([[0.0], np.cumsum(r)])
    peak = np.maximum.accumulate(logp)
    dd = np.exp(logp - peak) - 1
    under = dd < 0
    longest, cur = 0, 0
    for u in under:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    r250 = np.exp(logp[250:] - logp[:-250]) - 1
    r60 = np.exp(logp[60:] - logp[:-60]) - 1
    return dict(mean_d=float(m), std_d=float(s), skew=float((z ** 3).mean()), ex_kurt=float((z ** 4).mean() - 3), ac1_r=ac(r, 1), ac_abs_1_10=float(np.mean([ac(np.abs(r), k) for k in range(1, 11)])), tail_3s=float((np.abs(z) > 3).mean()), max_dd=float(dd.min()), longest_dd_days=int(longest), r250_mean=float(r250.mean()), r250_p05=float(np.quantile(r250, 0.05)), r60_p05=float(np.quantile(r60, 0.05)))


FACT_KEYS = ["mean_d", "std_d", "skew", "ex_kurt", "ac1_r", "ac_abs_1_10", "tail_3s", "max_dd", "longest_dd_days", "r250_mean", "r250_p05", "r60_p05"]
FACT_LABEL = dict(mean_d="일 평균수익", std_d="일 표준편차", skew="왜도", ex_kurt="초과첨도(꼬리)", ac1_r="수익 자기상관(1일)", ac_abs_1_10="|수익| 자기상관(1~10일, 군집)", tail_3s="3σ 초과 비율", max_dd="최대 낙폭", longest_dd_days="최장 낙폭 기간(일)", r250_mean="1년 수익 평균", r250_p05="1년 수익 하위5%", r60_p05="60일 수익 하위5%")


def fit_t_nu(ex_kurt):
    return float(4 + 6 / ex_kurt) if ex_kurt > 0.5 else 30.0


def student_t(rng, nu, size):
    g = rng.standard_normal(size)
    chi = rng.chisquare(nu, size)
    return g / np.sqrt(chi / nu) * np.sqrt((nu - 2) / nu)


def fit_garch(r, nu):
    x = r - r.mean()
    v0 = x.var()
    best = None
    for alpha in np.arange(0.02, 0.17, 0.01):
        for beta in np.arange(0.80, 0.98, 0.01):
            if alpha + beta >= 0.9985:
                continue
            omega = v0 * (1 - alpha - beta)
            h = np.empty_like(x)
            h[0] = v0
            for t in range(1, len(x)):
                h[t] = omega + alpha * x[t - 1] ** 2 + beta * h[t - 1]
            zt = x / np.sqrt(h)
            ll = -0.5 * np.sum(np.log(h)) - 0.5 * (nu + 1) * np.sum(np.log(1 + zt ** 2 / (nu - 2)))
            if best is None or ll > best[0]:
                best = (ll, alpha, beta, omega)
    return dict(alpha=float(best[1]), beta=float(best[2]), omega=float(best[3]), loglik=float(best[0]))


def fit_regimes(r, C):
    ma = np.full(len(C), np.nan)
    for i in range(MA_N - 1, len(C)):
        ma[i] = C[i - MA_N + 1:i + 1].mean()
    state = (C > ma)[:-1]
    valid = np.isfinite(ma[:-1])
    s, rr = state[valid], r[valid]
    bull, bear = rr[s], rr[~s]
    trans = {}
    for a, b, k in ((True, False, "bull_to_bear"), (False, True, "bear_to_bull")):
        n_a = int(np.sum(s[:-1] == a))
        trans[k] = float(np.sum((s[:-1] == a) & (s[1:] == b)) / n_a) if n_a else 0.0
    return dict(mu_bull=float(bull.mean()), mu_bear=float(bear.mean()), sd_bull=float(bull.std(ddof=1)), sd_bear=float(bear.std(ddof=1)), p_bull=float(s.mean()), **trans, n_bull=int(s.sum()), n_bear=int((~s).sum()))


def fit_herding(r):
    n = len(r)
    mom = np.array([r[i - 20:i].mean() for i in range(250, n)])
    logp = np.cumsum(r)
    dev = np.array([logp[i - 1] - logp[i - 250:i].mean() for i in range(250, n)])
    y = r[250:]
    X = np.column_stack([np.ones(len(y)), mom, dev])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return dict(phi_mom20=float(beta[1]), kappa_dev250=float(beta[2]), resid_sd=float(resid.std(ddof=1)), r2=float(1 - resid.var() / y.var()))


def knn_skill(r, dim=5, k=20, train_frac=0.7):
    X = np.column_stack([r[i:len(r) - dim + i] for i in range(dim)])
    y = np.sign(r[dim:])
    n = len(y)
    cut = int(n * train_frac)
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
    sd = Xtr.std(0) + 1e-12
    Xtr, Xte = Xtr / sd, Xte / sd
    hits = 0
    for i in range(len(Xte)):
        d = ((Xtr - Xte[i]) ** 2).sum(1)
        nn = np.argpartition(d, k)[:k]
        pred = np.sign(ytr[nn].sum()) or 1.0
        hits += pred == yte[i]
    return float(hits / len(Xte)), int(len(Xte))


def simulate(model, params, n, rng):
    mu, sd = params["mu"], params["sd"]
    nu = params.get("nu", 30.0)
    eps = student_t(rng, nu, n) if model != "M0" else rng.standard_normal(n)
    r = np.empty(n)
    if model in ("M0", "M1"):
        return mu + sd * eps
    g = params["garch"]
    h = sd ** 2
    state = rng.random() < params["reg"]["p_bull"]
    s_ch = params.get("chaos_s0", 0.3)
    sw = rng.random(n)
    logp = np.empty(n + 1)
    logp[0] = 0.0
    mom_sum, lp_sum250 = 0.0, 0.0
    phi, kappa = params["herd"]["phi_mom20"], params["herd"]["kappa_dev250"]
    reg = params["reg"]
    for t in range(n):
        if model in ("M3", "M4", "M5"):
            if sw[t] < (reg["bull_to_bear"] if state else reg["bear_to_bull"]):
                state = not state
            mu_t = reg["mu_bull"] if state else reg["mu_bear"]
            scale = (reg["sd_bull"] if state else reg["sd_bear"]) / sd
        else:
            mu_t, scale = mu, 1.0
        if model in ("M4", "M5") and t >= 250:
            mu_t = mu_t + phi * (mom_sum / 20.0) + kappa * (logp[t] - lp_sum250 / 250.0)
        if model == "M5":
            s_ch = 4.0 * s_ch * (1.0 - s_ch)
            mu_t = mu_t + params["chaos_gamma"] * (s_ch - 0.5)
        r[t] = mu_t + np.sqrt(h) * scale * eps[t]
        shock = (r[t] - mu_t) / scale
        h = g["omega"] + g["alpha"] * shock ** 2 + g["beta"] * h
        logp[t + 1] = logp[t] + r[t]
        mom_sum += r[t] - (r[t - 20] if t >= 20 else 0.0)
        lp_sum250 += logp[t + 1] - (logp[t + 1 - 250] if t + 1 >= 250 else 0.0)
    return r


def rules_on_path(r, rng_unused=None):
    C = C0 * np.exp(np.cumsum(np.concatenate([[0.0], r])))
    C = np.round(C / TICK) * TICK
    O = C
    n = len(C)
    ma = np.full(n, np.nan)
    cs = np.cumsum(C)
    ma[MA_N - 1:] = (cs[MA_N - 1:] - np.concatenate([[0.0], cs[:-MA_N]])) / MA_N
    out = {}
    for rung in ("R1_buyhold", "R3_ma200_in_out"):
        cash, qty, dep = 0.0, 0, 0.0
        vals = []
        for i in range(n - 1):
            dep += DEPOSIT
            cash += DEPOSIT
            buy, sell = True, False
            if rung == "R3_ma200_in_out" and np.isfinite(ma[i]):
                buy, sell = C[i] > ma[i], C[i] < ma[i]
            px = O[i + 1]
            if sell and qty > 0:
                cash += qty * (px - TICK) * (1 - FEE)
                qty = 0
            if buy:
                q = int(cash // ((px + TICK) * (1 + FEE)))
                if q >= 1:
                    cash -= q * (px + TICK) * (1 + FEE)
                    qty += q
            vals.append(cash + qty * C[i + 1] - dep)
        adj = np.array(vals)
        peak = np.maximum.accumulate(np.concatenate([[0.0], adj]))[1:]
        out[rung] = dict(pnl_pct=float(adj[-1] / dep), dd_pct=float((adj - peak).min() / dep))
    return out


def band(vals, q=(0.05, 0.5, 0.95)):
    v = np.asarray(vals, float)
    return [float(np.quantile(v, x)) for x in q]


def main():
    r, C, days = load_returns()
    real = facts(r)
    n = len(r)
    rng = np.random.default_rng(SEED)
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    nu = fit_t_nu(real["ex_kurt"])
    garch = fit_garch(r, nu)
    reg = fit_regimes(r, C)
    herd = fit_herding(r)
    skill, n_test = knn_skill(r)
    sh = []
    for i in range(20):
        rs = rng.permutation(r)
        sh.append(knn_skill(rs)[0])
    chaos_gamma = float(np.sqrt(0.10) * sd / np.sqrt(1 / 8))
    params = dict(mu=mu, sd=sd, nu=nu, garch=garch, reg=reg, herd=herd, chaos_gamma=chaos_gamma, chaos_s0=0.3)
    models = {"M0": "방향+정규 잡음", "M1": "+두꺼운 꼬리(t분포)", "M2": "+변동성 군집(GARCH)", "M3": "+상승/하락 국면 전환", "M4": "+군중 쏠림·되돌림", "M5": "+결정론적 카오스 성분"}
    table, rules = {}, {}
    for m in models:
        fs, rr = [], []
        for p in range(N_PATHS):
            path = simulate(m, params, n, rng)
            fs.append(facts(path))
            if p < 60:
                rr.append(rules_on_path(path))
        table[m] = {k: dict(band=band([f[k] for f in fs]), real=real[k], inside=bool(band([f[k] for f in fs])[0] <= real[k] <= band([f[k] for f in fs])[2])) for k in FACT_KEYS}
        rules[m] = {rg: dict(pnl_pct=band([x[rg]["pnl_pct"] for x in rr]), dd_pct=band([x[rg]["dd_pct"] for x in rr])) for rg in ("R1_buyhold", "R3_ma200_in_out")}
    real_rules = rules_on_path(r)
    lines = [f"실제 KODEX200(조정) {days[0]}~{days[-1]} {n}일 | 모델별 {N_PATHS}경로 동일 길이, 띠 = 5%~95% 분위, ○ = 실제값이 띠 안"]
    lines.append("| 성질 | 실제 | " + " | ".join(f"{m} {models[m]}" for m in models) + " |")
    for k in FACT_KEYS:
        def fmt(v):
            if k == "longest_dd_days":
                return f"{v:.0f}"
            if k in ("mean_d",):
                return f"{v*100:.3f}%"
            if k in ("std_d", "tail_3s", "max_dd", "r250_mean", "r250_p05", "r60_p05"):
                return f"{v*100:.1f}%"
            return f"{v:.2f}"
        row = f"| {FACT_LABEL[k]} | {fmt(real[k])} |"
        for m in models:
            b = table[m][k]["band"]
            row += f" {'○' if table[m][k]['inside'] else '×'} {fmt(b[1])} [{fmt(b[0])},{fmt(b[2])}] |"
        lines.append(row)
    score = {m: sum(table[m][k]["inside"] for k in FACT_KEYS) for m in models}
    lines.append("띠 안 성질 수(12개 중, 같은 자료에 맞춘 주변 지표 진단이며 검증 통과율이 아님): " + ", ".join(f"{m} {score[m]}" for m in models))
    lines.append(f"비선형 예측 가능성(k-NN, 5일 임베딩, 뒤 30% 시험 {n_test}일): 방향 적중 {skill*100:.1f}% vs 뒤섞은 자료 {np.mean(sh)*100:.1f}% [{np.min(sh)*100:.1f}, {np.max(sh)*100:.1f}] → 이 설정에서는 예측 우위를 찾지 못함(결정론 구조의 부재나 카오스 크기 제한을 뜻하지 않음; 실제자료 카오스 여부 미확인)")
    lines.append("모수(전기간 표본 적합 = 표본 내 진단, API 0): μ %.4f%%/일 σ %.2f%% ν %.1f(초과첨도로 적합) | GARCH α %.2f β %.2f(격자 최대우도) | 국면(MA200 관측 대리) 상승 μ %.3f%% σ %.2f%% (일수 %d) 하락 μ %.3f%% σ %.2f%% (일수 %d) 전환확률 상→하 %.3f 하→상 %.3f | OLS: 20일 모멘텀 계수 φ %.2f, 250일 MA 괴리 계수 κ %.4f (+는 괴리 증폭, −는 복원; 기준가치 괴리 x와 다른 변수), R² %.4f" % (mu * 100, sd * 100, nu, garch["alpha"], garch["beta"], reg["mu_bull"] * 100, reg["sd_bull"] * 100, reg["n_bull"], reg["mu_bear"] * 100, reg["sd_bear"] * 100, reg["n_bear"], reg["bull_to_bear"], reg["bear_to_bull"], herd["phi_mom20"], herd["kappa_dev250"], herd["r2"]))
    lines.append("가상 적립 규칙 비교(조정가격을 %.0f원부터 합성, 다음 종가를 시가로 사용, 소수 수수료, 낙폭 = 최종 누적입금 대비 손익 낙폭; 원시 시가·정수 수수료 계좌 ladder_kr과 다름): 실제 수익률 경로 R1 보유 %+.0f%% 낙폭 %.0f%% / R3 200일선 매도 %+.0f%% 낙폭 %.0f%%" % (C0, real_rules["R1_buyhold"]["pnl_pct"] * 100, real_rules["R1_buyhold"]["dd_pct"] * 100, real_rules["R3_ma200_in_out"]["pnl_pct"] * 100, real_rules["R3_ma200_in_out"]["dd_pct"] * 100))
    for m in models:
        a, b = rules[m]["R1_buyhold"], rules[m]["R3_ma200_in_out"]
        lines.append(f"  {m}: 보유 중앙 {a['pnl_pct'][1]*100:+.0f}% [{a['pnl_pct'][0]*100:+.0f},{a['pnl_pct'][2]*100:+.0f}] 낙폭 중앙 {a['dd_pct'][1]*100:.0f}% | 매도규칙 중앙 {b['pnl_pct'][1]*100:+.0f}% [{b['pnl_pct'][0]*100:+.0f},{b['pnl_pct'][2]*100:+.0f}] 낙폭 중앙 {b['dd_pct'][1]*100:.0f}%")
    out = dict(meta=dict(script_sha256=sha(os.path.abspath(__file__)), adj_sha256=sha(ADJ), sym=SYM, span=[days[0], days[-1]], n_returns=n, n_paths=N_PATHS, seed=SEED, models=models), params=params, real_facts=real, table=table, rules=rules, real_rules=real_rules, knn=dict(skill=skill, n_test=n_test, shuffled=sh), lines=lines)
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print("\n".join(lines))
    print("saved", OUT, "sha", sha(OUT))


def selftest():
    rng = np.random.default_rng(0)
    r = rng.standard_normal(3000) * 0.01 + 0.0003
    f = facts(r)
    assert abs(f["std_d"] - 0.01) < 0.001 and abs(f["ex_kurt"]) < 0.5 and f["max_dd"] < 0 and f["longest_dd_days"] > 0
    assert fit_t_nu(3.0) == 6.0 and fit_t_nu(0.0) == 30.0
    t = student_t(rng, 5.0, 200000)
    assert abs(t.std() - 1.0) < 0.02
    g = fit_garch(r, 30.0)
    assert 0 < g["alpha"] < 0.2 and 0.7 < g["beta"] < 1 and g["omega"] > 0
    C = 1000 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    reg = fit_regimes(r, C)
    assert 0 < reg["p_bull"] < 1 and 0 <= reg["bull_to_bear"] <= 1
    h = fit_herding(r)
    assert abs(h["phi_mom20"]) < 1 and h["r2"] < 0.05
    sk, nt = knn_skill(r[:1500])
    assert 0.3 < sk < 0.7 and nt > 0
    params = dict(mu=0.0003, sd=0.01, nu=5.0, garch=g, reg=reg, herd=h, chaos_gamma=0.001, chaos_s0=0.3)
    for m in ("M0", "M1", "M2", "M3", "M4", "M5"):
        p = simulate(m, params, 600, rng)
        assert p.shape == (600,) and np.isfinite(p).all(), m
    p1 = simulate("M5", params, 600, np.random.default_rng(1))
    p2 = simulate("M5", params, 600, np.random.default_rng(1))
    assert np.array_equal(p1, p2)
    rules = rules_on_path(r[:1200])
    assert set(rules) == {"R1_buyhold", "R3_ma200_in_out"} and all(-1 < v["dd_pct"] <= 0 for v in rules.values())
    flat = np.zeros(800)
    rf = rules_on_path(flat)
    assert abs(rf["R1_buyhold"]["pnl_pct"]) < 0.01
    print("selftest ok", dict(sha=sha(os.path.abspath(__file__))))


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    else:
        main()
