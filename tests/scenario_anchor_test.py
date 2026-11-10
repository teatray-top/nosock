import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import scenario_desk as SD
import scorecard as S

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  {detail}"))


days = [f"d{k:05d}" for k in range(1200)]
rng = np.random.default_rng(7)
C = 100 * np.exp(np.cumsum(0.0005 + 0.01 * rng.standard_normal(1200)))
inst = {"069500": dict(days=days, C=list(C), fxv=[1.0] * 1200, market="KR"),
        "SPY": dict(days=days, C=list(C * 1400), fxv=[1400.0] * 1200, market="US"),
        "NEW": dict(days=days[-100:], C=list(C[-100:]), fxv=[1.0] * 100, market="KR")}
br = SD.base_rates(inst, ["069500", "SPY", "NEW"], days[-1])
r = C[20:] / C[:-20] - 1
check("base rate = mean of past 20-day returns", abs(br["069500"]["mean"] - round(100 * r[-(SD.BASE_DAYS):].mean(), 2)) < 1e-9 and abs(br["069500"]["up"] - round(float((r > 0).mean()), 2)) < 1e-9, br["069500"])
check("US base rate uses dollar prices (KRW / fx)", br["SPY"]["mean"] == br["069500"]["mean"], br["SPY"])
check("short history -> no base rate", "NEW" not in br)

codes = ["069500", "114800", "XLE"]
p = np.array([0.25, 0.25, 0.25, 0.25])
R = np.array([[-0.20, 0.0, 0.04], [-0.10, 0.0, 0.02], [-0.05, 0.0, 0.0], [0.15, 0.0, -0.04]])
R = SD.pair_inverse(R, codes)
base = {"069500": dict(mean=1.0, sd=4.0), "XLE": dict(mean=0.5, sd=6.0)}
Ra, shift = SD.anchor(p, R, codes, base)
m = p @ Ra
check("bearish desk mean (-5%) pulled up to base - sd/4 (0%)", abs(m[0] - 0.0) < 1e-12 and abs(shift["069500"] - 5.0) < 1e-9, (m, shift))
check("scenario spread kept (best - worst unchanged)", abs((Ra[:, 0].max() - Ra[:, 0].min()) - 0.35) < 1e-12)
check("inverse rebuilt from the anchored index", np.allclose(Ra[:, 1], -Ra[:, 0] - SD.INV_DRAG_20D))
check("mean already inside the band -> untouched", "XLE" not in shift and np.allclose(Ra[:, 2], R[:, 2]))
check("no base rate -> untouched", SD.anchor(p, R, codes, {})[1] == {})

row = dict(at=S.parse_ts("2026-11-11T08:41:00+09:00"), trigger="morning", tilt={}, probs=list(p), scen=[dict(returns_pct={"069500": -20})] * 4, exp={"069500": 0.0, "XLE": 1.25, "114800": -0.75}, base={"XLE": dict(mean=0.5), "114800": dict(mean=-1.2)})
dc = {c["code"]: c for c in S.desk_calls([row])}
check("scorecard grades the anchored forecast, not the raw scenario mean", "069500" not in dc and dc["XLE"]["magnitude"] == 1.25 and dc["XLE"]["direction"] == "up" and dc["114800"]["direction"] == "down", dc)
check("each desk call carries its base rate", dc["XLE"]["base"] == 0.5 and dc["114800"]["base"] == -1.2, dc)

print(f"{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
