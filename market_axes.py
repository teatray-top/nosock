import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
import sys
import threading
import time
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work\market_axes_F0F1_20261106")
OUT = os.path.join(OUT_DIR, "market_axes_F0F1.json")
SPEC = dict(variants={"F0": dict(a=0.0, b=0.0), "F1": dict(a=0.1, b=0.0)}, mu=0.0002, sigma=0.015, x0=0.0, seed=20261106, paired_innovations=True, paths_per_variant=20, burn_in=1000, evaluation_sessions=2500, horizon=250, max_threads=1, max_wall_seconds=120, purpose="mechanism demonstration with example constants; not a fitted market, not a trading test")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def closed_forms(a, sigma, mu, h):
    rho = 1.0 - a
    if a <= 0:
        return dict(stationary=False, has_finite_stationary_variance=False, var_x=None, var_r=sigma ** 2, acf1_r=0.0, var_h=h * sigma ** 2, mean_r=mu, mean_h=h * mu, var_h_ratio_vs_random_walk=1.0, note="a=0: x is a Gaussian random walk (non-stationary; variance at each finite t is finite but grows linearly); returns are iid")
    var_x = sigma ** 2 / (1 - rho ** 2)
    var_r = 2 * sigma ** 2 / (1 + rho)
    acf1 = -a / 2.0
    var_h = 2 * var_x * (1 - rho ** h)
    return dict(stationary=True, has_finite_stationary_variance=True, var_x=var_x, sd_x=float(np.sqrt(var_x)), var_r=var_r, acf1_r=acf1, var_h=var_h, mean_r=mu, mean_h=h * mu, var_h_ratio_vs_random_walk=var_h / (h * sigma ** 2), local_stability_condition="b < 1 + a(1-c)/c (b=0 here, satisfied)", note="x_{t+1}=(1-a)x_t+sigma*eps: stationary Gaussian AR(1) with finite variance (not an absolute bound on paths); r_t = mu + x_{t+1}-x_t")


def simulate(a, mu, sigma, x0, eps):
    n_paths, n = eps.shape
    x = np.empty((n_paths, n + 1))
    x[:, 0] = x0
    rho = 1.0 - a
    for t in range(n):
        x[:, t + 1] = rho * x[:, t] + sigma * eps[:, t]
    v = mu * np.arange(n + 1)
    p = v[None, :] + x
    r = np.diff(p, axis=1)
    return x, p, r


def stats(r, x, burn, h):
    r_eval = r[:, burn:]
    x_eval = x[:, burn + 1:]
    p_eval = np.cumsum(np.concatenate([np.zeros((r.shape[0], 1)), r_eval], axis=1), axis=1)
    rh = p_eval[:, h:] - p_eval[:, :-h]
    ac1 = [float(np.corrcoef(row[:-1], row[1:])[0, 1]) for row in r_eval]
    return dict(mean_r=float(r_eval.mean()), std_r=float(r_eval.std(ddof=1)), var_r=float(r_eval.var(ddof=1)), acf1_r_mean=float(np.mean(ac1)), acf1_r_band=[float(np.quantile(ac1, 0.05)), float(np.quantile(ac1, 0.95))], var_h=float(rh.var(ddof=1)), mean_h=float(rh.mean()), max_abs_x=float(np.abs(x_eval).max()), sd_x_eval=float(x_eval.std(ddof=1)), n_paths=int(r_eval.shape[0]), n_eval=int(r_eval.shape[1]))


def run(spec=SPEC, out=OUT):
    t0 = time.monotonic()
    rng = np.random.default_rng(spec["seed"])
    n = spec["burn_in"] + spec["evaluation_sessions"]
    eps = rng.standard_normal((spec["paths_per_variant"], n))
    results = {}
    for name, v in spec["variants"].items():
        if v["b"] != 0.0:
            raise SystemExit("b != 0 not in scope")
        x, p, r = simulate(v["a"], spec["mu"], spec["sigma"], spec["x0"], eps)
        cf = closed_forms(v["a"], spec["sigma"], spec["mu"], spec["horizon"])
        st = stats(r, x, spec["burn_in"], spec["horizon"])
        results[name] = dict(params=v, closed_form=cf, simulated=st, var_h_ratio_vs_random_walk_sim=st["var_h"] / (spec["horizon"] * spec["sigma"] ** 2))
        if time.monotonic() - t0 > spec["max_wall_seconds"]:
            raise SystemExit("wall cap exceeded")
    meta = dict(script_sha256=sha(os.path.abspath(__file__)), spec=spec, paired_innovations_check=bool(spec["paired_innovations"]), wall_s=round(time.monotonic() - t0, 3), generated=datetime.now().isoformat(timespec="seconds"), threads=dict(env={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}, python_threads_active=threading.active_count()), fits=0, api_calls=0, orders=0)
    lines = []
    for name, res in results.items():
        cf, st = res["closed_form"], res["simulated"]
        xdesc = f"정상·유한분산(식 sd {cf['sd_x']:.4f}, 모의 sd {st['sd_x_eval']:.4f}, 관측 max|x| {st['max_abs_x']:.4f}; 절대 유계 아님)" if cf["stationary"] else f"랜덤워크(비정상, 분산 선형 증가; 관측 max|x| {st['max_abs_x']:.4f})"
        lines.append(f"{name} a={res['params']['a']}: 일수익 평균 모의 {st['mean_r']*100:.4f}% (식 {cf['mean_r']*100:.4f}%) | 표준편차 모의 {st['std_r']*100:.3f}% (식 {np.sqrt(cf['var_r'])*100:.3f}%) | AC1 모의 {st['acf1_r_mean']:+.4f} [{st['acf1_r_band'][0]:+.4f},{st['acf1_r_band'][1]:+.4f}] (식 {cf['acf1_r']:+.4f}) | 250일 로그수익 분산 모의 {st['var_h']:.5f} (식 {cf['var_h']:.5f}), 랜덤워크 대비 {res['var_h_ratio_vs_random_walk_sim']*100:.2f}% (식 {cf['var_h_ratio_vs_random_walk']*100:.2f}%) | x: {xdesc}")
    payload = dict(meta=meta, results=results, lines=lines)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if os.path.exists(out):
            raise SystemExit(f"{out} exists; refusing to overwrite")
        with open(out, "xb") as f:
            f.write(json.dumps(payload, ensure_ascii=False, indent=1).encode("utf-8"))
    return payload


def selftest():
    cf = closed_forms(0.1, 0.015, 0.0002, 250)
    assert abs(cf["acf1_r"] + 0.05) < 1e-12 and abs(cf["var_h_ratio_vs_random_walk"] - 0.0421) < 0.0005, cf
    cf0 = closed_forms(0.0, 0.015, 0.0002, 250)
    assert cf0["var_h_ratio_vs_random_walk"] == 1.0 and not cf0["stationary"] and not cf0["has_finite_stationary_variance"] and cf["has_finite_stationary_variance"]
    assert all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"))
    rng = np.random.default_rng(1)
    eps = rng.standard_normal((4, 200000))
    x, p, r = simulate(0.1, 0.0002, 0.015, 0.0, eps)
    st = stats(r, x, 1000, 250)
    assert abs(st["acf1_r_mean"] + 0.05) < 0.005 and abs(st["var_h"] / (250 * 0.015 ** 2) - 0.0421) < 0.004 and abs(st["std_r"] - np.sqrt(cf["var_r"])) < 0.0002, st
    x0_, p0_, r0_ = simulate(0.0, 0.0002, 0.015, 0.0, eps)
    assert np.allclose(x0_[:, 1:], np.cumsum(0.015 * eps, axis=1)) and abs(stats(r0_, x0_, 1000, 250)["acf1_r_mean"]) < 0.005
    spec = dict(SPEC, paths_per_variant=3, burn_in=50, evaluation_sessions=300)
    out = run(spec, out=None)
    assert set(out["results"]) == {"F0", "F1"} and out["results"]["F1"]["simulated"]["n_paths"] == 3 and out["results"]["F1"]["simulated"]["n_eval"] == 300
    rng_a = np.random.default_rng(spec["seed"])
    e1 = rng_a.standard_normal((3, 350))
    xa, _, _ = simulate(0.0, spec["mu"], spec["sigma"], 0.0, e1)
    xb, _, _ = simulate(0.1, spec["mu"], spec["sigma"], 0.0, e1)
    assert np.allclose(xa[:, 1], xb[:, 1]) and not np.allclose(xa[:, 50], xb[:, 50])
    print("selftest ok", dict(sha=sha(os.path.abspath(__file__)), threads=os.environ.get("OMP_NUM_THREADS")))


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    else:
        payload = run()
        print("\n".join(payload["lines"]))
        print("saved", OUT, "sha", sha(OUT), "wall", payload["meta"]["wall_s"], "s")
