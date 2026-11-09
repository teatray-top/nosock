import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
panel5 = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "panelusm5m_w1_v4.npz")
out = sys.argv[2] if len(sys.argv) > 2 else panel5[:-4] + "_gate_r5m10_ma20.npz"
RET5_MIN = float(os.environ.get("GATE_RET5_MIN", -0.10))
MA_N = int(os.environ.get("GATE_MA_N", 20))

z5 = np.load(panel5, allow_pickle=True)
days5 = [str(d) for d in z5["days"]]
syms5 = [str(s) for s in z5["syms"]]
zd = np.load(os.path.join(HERE, "data", "panel_daily_v2.npz"), allow_pickle=True)
daysd = [str(d) for d in zd["days"]]
symsd = {str(s): i for i, s in enumerate(zd["syms"])}
C = zd["C"]
didx = {d: i for i, d in enumerate(daysd)}

D, N = len(days5), len(syms5)
gate = np.ones((D, N), dtype=bool)
info = np.zeros((D, N), dtype=bool)
ret5 = np.full((D, N), np.nan)
ma_rel = np.full((D, N), np.nan)
missing_sym = [s for s in syms5 if s not in symsd]
for i, d in enumerate(days5):
    k = didx.get(d)
    if k is None:
        prev = [j for j, dd in enumerate(daysd) if dd < d]
        if not prev:
            continue
        k = prev[-1] + 1
    p = k - 1
    if p < MA_N:
        continue
    for j, s in enumerate(syms5):
        jj = symsd.get(s)
        if jj is None:
            continue
        c = C[p - MA_N + 1:p + 1, jj]
        if not np.isfinite(c).all() or c[-1] <= 0:
            continue
        c5 = C[p - 5, jj]
        r5 = c[-1] / c5 - 1 if np.isfinite(c5) and c5 > 0 else np.nan
        ma = c.mean()
        info[i, j] = True
        ret5[i, j] = r5
        ma_rel[i, j] = c[-1] / ma - 1
        gate[i, j] = (np.isnan(r5) or r5 >= RET5_MIN) and c[-1] >= ma
np.savez_compressed(out, gate=gate, info=info, ret5=ret5, ma_rel=ma_rel, days=np.array(days5), syms=np.array(syms5))
stats = dict(panel=os.path.basename(panel5), out=os.path.basename(out), ret5_min=RET5_MIN, ma_n=MA_N, days=D, syms=N, syms_missing_in_daily=len(missing_sym), missing_examples=missing_sym[:10], info_frac=float(info.mean()), gate_pass_frac_of_info=float(gate[info].mean()), gate_pass_frac_all=float(gate.mean()), blocked_by_ret5=float(((ret5 < RET5_MIN) & info).sum() / max(1, info.sum())), blocked_by_ma=float(((ma_rel < 0) & info).sum() / max(1, info.sum())))
print(json.dumps(stats))
json.dump(stats, open(out[:-4] + ".json", "w"), indent=1)
